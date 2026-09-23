"""通过双渠道 Anthropic Messages API 执行 Stage5 单轮裁决任务。"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import httpx
from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError, ValidationError

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
    ContractViolation,
    StructuredOutputViolation,
    _parse_single_json_result,
    canonical_json,
    validate_structured_output,
    validate_formula_audit_scope,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    ClaudeRunResult,
    ClaudeRunner,
    ClaudeRunnerCircuitBreaker,
    JsonDict,
    TaskDefinition,
    _atomic_write_json,
    _sanitize_for_audit,
    _sha256_file,
    _sha256_text,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SimplificationContractError,
    SymbolicEvidenceError,
)


API_TRANSPORT_VERSION = "anthropic_messages_nonstream_dual_channel.v1"
STRICT_EVALUATOR_SYSTEM_PROMPT = """You are a strict mathematical benchmark evaluator.
All domain assumptions, declared variables/functions, instantiated constants, and output-schema rules supplied by the user are hard constraints.
Never add assumptions from physical interpretation, typical parameter signs, or observed probe ranges.
Judge preservation in the declared number system, not in a broader complex domain.
If an expression has no valid value in the declared real-valued domain, or relies on undeclared symbolic entities, follow the task's unable/undetermined rule rather than treating textual identity as sufficient.
Return only the requested JSON object, with no additional properties or prose."""


@dataclass(frozen=True)
class AnthropicApiChannel:
    name: str
    base_url: str
    auth_token: str = field(repr=False)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/messages"


@dataclass(frozen=True)
class AnthropicApiResponse:
    status_code: int
    body: object
    text: str
    headers: Mapping[str, str]


ApiTransport = Callable[
    [AnthropicApiChannel, Mapping[str, object], float],
    AnthropicApiResponse,
]


class HttpxAnthropicTransport:
    """为每个渠道保留独立连接池；Authorization 不进入审计对象。"""

    def __init__(self, channels: Sequence[AnthropicApiChannel], *, max_connections: int) -> None:
        self._clients = {
            channel.name: httpx.Client(
                headers={
                    "Authorization": f"Bearer {channel.auth_token}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections,
                ),
            )
            for channel in channels
        }

    def __call__(
        self,
        channel: AnthropicApiChannel,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> AnthropicApiResponse:
        response = self._clients[channel.name].post(
            channel.endpoint,
            json=dict(payload),
            timeout=timeout_seconds,
        )
        try:
            body: object = response.json()
        except ValueError:
            body = None
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"request-id", "anthropic-request-id", "retry-after"}
        }
        return AnthropicApiResponse(
            status_code=response.status_code,
            body=body,
            text=response.text,
            headers=headers,
        )


def _estimated_cost_cny(usage: Mapping[str, object]) -> float | None:
    def token_count(name: str) -> int:
        value = usage.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name)
        return max(0, int(value))

    try:
        input_tokens = token_count("input_tokens")
        output_tokens = token_count("output_tokens")
        cache_read_tokens = token_count("cache_read_input_tokens")
    except ValueError:
        return None
    return (
        input_tokens * 3.0
        + output_tokens * 15.0
        + cache_read_tokens * 0.3
    ) / 1_000_000.0


def _response_error_message(response: AnthropicApiResponse) -> str:
    if isinstance(response.body, Mapping):
        error = response.body.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(response.body.get("message"), str):
            return str(response.body["message"])
    return response.text[:2000] or f"HTTP {response.status_code}"


def _is_retryable_upstream_deployment_failure(
    response: AnthropicApiResponse,
    error_message: str,
) -> bool:
    if response.status_code != 400:
        return False
    normalized = error_message.lower()
    return any(
        marker in normalized
        for marker in (
            "allmodelsfailed",
            "deployment request could not be completed",
            "555420",
        )
    )


def _response_body_for_audit(body: object) -> object:
    sanitized = _sanitize_for_audit(body)
    if not isinstance(sanitized, Mapping):
        return sanitized
    content = sanitized.get("content")
    if not isinstance(content, list):
        return dict(sanitized)
    compact_content: list[object] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") == "text":
            compact_content.append(block)
            continue
        compact_content.append(
            {
                "type": "redacted_thinking",
                "redacted": True,
                "original_type": block.get("type"),
                "block_sha256": _sha256_text(canonical_json(block)),
            }
        )
    result = dict(sanitized)
    result["content"] = compact_content
    return result


class AnthropicApiRunner(ClaudeRunner):
    """复用 Stage5 状态机，以非流式 HTTP 传输替换 Claude Code CLI。"""

    def __init__(
        self,
        store: TaskStateStore,
        *,
        attempts_dir: str | Path,
        frozen_dir: str | Path,
        channels: Sequence[AnthropicApiChannel],
        transport: ApiTransport | None = None,
        timeout_seconds: float = 300.0,
        lease_seconds: float | None = None,
        max_tokens: int = 4096,
        per_channel_concurrency: int = 32,
        semantic_validation_concurrency: int = 4,
        backoff_schedule_seconds: Sequence[float] = (1.0, 2.0),
        allow_single_channel: bool = False,
        system_prompt: str = STRICT_EVALUATOR_SYSTEM_PROMPT,
    ) -> None:
        if len(channels) < 2 and not allow_single_channel:
            raise ValueError("双渠道 API runner 至少需要两个渠道")
        if len({channel.name for channel in channels}) != len(channels):
            raise ValueError("API 渠道名称不得重复")
        if any(not channel.name or not channel.base_url or not channel.auth_token for channel in channels):
            raise ValueError("API 渠道 name/base_url/auth_token 均为必填")
        if max_tokens <= 0 or per_channel_concurrency <= 0 or semantic_validation_concurrency <= 0:
            raise ValueError("max_tokens 与并发参数必须为正数")
        super().__init__(
            store,
            attempts_dir=attempts_dir,
            frozen_dir=frozen_dir,
            timeout_seconds=timeout_seconds,
            lease_seconds=lease_seconds,
            backoff_schedule_seconds=backoff_schedule_seconds,
        )
        self.channels = tuple(channels)
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError('API system_prompt 必须为非空字符串')
        self.system_prompt = system_prompt
        self.max_tokens = int(max_tokens)
        self.transport: ApiTransport = transport or HttpxAnthropicTransport(
            self.channels,
            max_connections=per_channel_concurrency,
        )
        self._channel_semaphores = {
            channel.name: threading.BoundedSemaphore(per_channel_concurrency)
            for channel in self.channels
        }
        self._semantic_semaphore = threading.BoundedSemaphore(semantic_validation_concurrency)

    def _build_and_validate_command(self, schema: Mapping[str, object]) -> list[str]:
        try:
            Draft7Validator.check_schema(schema)
        except SchemaError as exc:
            raise ClaudeRunnerCircuitBreaker(
                f"task definition drift: 输出 schema 不是合法 Draft-07: {exc.message}",
                evaluation_key="unknown",
                attempt_id=None,
            ) from exc
        return [
            "anthropic-messages-api",
            "--model",
            CONTRACT_CANONICAL_MODEL,
            "--effort",
            CONTRACT_EFFORT,
            "--stream",
            "false",
        ]

    def _select_channel(self, evaluation_key: str, *, attempt_number: int = 1) -> AnthropicApiChannel:
        try:
            shard_value = int(evaluation_key, 16)
        except ValueError:
            shard_value = int(hashlib.sha256(evaluation_key.encode("utf-8")).hexdigest(), 16)
        return self.channels[(shard_value + attempt_number - 1) % len(self.channels)]

    def _api_payload(self, prompt: str) -> JsonDict:
        return {
            "model": CONTRACT_CANONICAL_MODEL,
            "stream": False,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": CONTRACT_EFFORT},
        }

    @staticmethod
    def _extract_structured_output(
        response: AnthropicApiResponse,
        *,
        task_kind: str,
        schema: Mapping[str, object],
    ) -> tuple[JsonDict, JsonDict, str, JsonDict]:
        if not isinstance(response.body, Mapping):
            raise StructuredOutputViolation("API 响应体不是 JSON object")
        body = response.body
        if body.get("type") != "message" or body.get("role") != "assistant":
            raise ContractViolation("API 响应不是 assistant message")
        actual_model = body.get("model")
        if actual_model != CONTRACT_CANONICAL_MODEL:
            raise ContractViolation(f"API 实际模型不符: {actual_model!r}")
        if body.get("stop_reason") != "end_turn":
            raise StructuredOutputViolation(
                f"API stop_reason 不符: {body.get('stop_reason')!r}"
            )
        content = body.get("content")
        if not isinstance(content, list):
            raise ContractViolation("API content 缺失或无效")
        text_blocks: list[str] = []
        content_types: list[str] = []
        for block in content:
            if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
                raise ContractViolation("API content block 无效")
            block_type = str(block["type"])
            content_types.append(block_type)
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ContractViolation("API text block 缺少文本")
                text_blocks.append(text)
            elif block_type not in {"thinking", "redacted_thinking"}:
                raise ContractViolation(f"API 返回禁止的 content block: {block_type!r}")
        if not text_blocks or not "".join(text_blocks).strip():
            raise StructuredOutputViolation("API 未返回非空 text block")
        output_text = "\n".join(text_blocks)

        def validate_parsed(parsed_candidate: Mapping[str, object]) -> Mapping[str, object]:
            try:
                Draft7Validator(schema).validate(parsed_candidate)
            except ValidationError as exc:
                raise StructuredOutputViolation(
                    f"API result 未通过 Draft-07 schema: {exc.message}"
                ) from exc
            try:
                return validate_structured_output(task_kind, parsed_candidate)
            except ContractViolation as exc:
                raise StructuredOutputViolation(str(exc)) from exc

        def validate_candidate(candidate: str) -> Mapping[str, object]:
            return validate_parsed(_parse_single_json_result(candidate))

        recovery: str | None = None
        try:
            structured = validate_candidate(output_text)
        except StructuredOutputViolation as strict_error:
            valid_fenced_outputs: list[Mapping[str, object]] = []
            for match in re.finditer(
                r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```",
                output_text,
            ):
                try:
                    valid_fenced_outputs.append(validate_candidate(match.group(1)))
                except StructuredOutputViolation:
                    continue
            if len(valid_fenced_outputs) == 1:
                structured = valid_fenced_outputs[0]
                recovery = "single_valid_json_fence"
            else:
                valid_embedded_outputs: dict[str, Mapping[str, object]] = {}
                decoder = json.JSONDecoder()
                for match in re.finditer(r"(?m)^[ \t]*\{", output_text):
                    try:
                        parsed_object, _ = decoder.raw_decode(output_text[match.start() :].lstrip())
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed_object, Mapping):
                        continue
                    try:
                        validated = validate_parsed(parsed_object)
                    except StructuredOutputViolation:
                        continue
                    valid_embedded_outputs[canonical_json(validated)] = validated
                if len(valid_embedded_outputs) != 1:
                    raise strict_error
                structured = next(iter(valid_embedded_outputs.values()))
                recovery = "single_valid_embedded_json_object"
        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            raise ContractViolation("API 响应缺少 usage")
        response_metadata: JsonDict = {
            "message_id": body.get("id"),
            "model": actual_model,
            "stop_reason": body.get("stop_reason"),
            "stop_sequence": body.get("stop_sequence"),
            "content_types": content_types,
        }
        if recovery is not None:
            response_metadata["structured_output_recovery"] = recovery
        return (
            dict(structured),
            dict(usage),
            output_text,
            response_metadata,
        )

    def _run_attempt(
        self,
        *,
        definition: TaskDefinition,
        task_kind: str,
        prompt: str,
        prompt_sha256: str,
        schema_sha256: str,
        command: list[str],
        attempt_id: str,
        attempt_number: int,
        lease_expires_at: float,
    ) -> ClaudeRunResult:
        started_at = self.now_fn()
        started_monotonic = self.monotonic_fn()
        channel = self._select_channel(
            definition.task_spec.evaluation_key,
            attempt_number=attempt_number,
        )
        api_payload = self._api_payload(prompt)
        response: AnthropicApiResponse | None = None
        structured_output: JsonDict | None = None
        usage: JsonDict | None = None
        output_text = ""
        response_metadata: JsonDict = {}
        error_class: str | None = None
        error_message: str | None = None
        retryable = False
        circuit_break = False
        semantic_evidence: JsonDict | None = None

        try:
            with self._channel_semaphores[channel.name]:
                response = self.transport(channel, api_payload, self.timeout_seconds)
            if response.status_code != 200:
                error_message = _response_error_message(response)
                if response.status_code in {401, 403}:
                    error_class = "api_auth_error"
                    circuit_break = True
                elif (
                    response.status_code in {408, 409, 429}
                    or response.status_code >= 500
                    or _is_retryable_upstream_deployment_failure(response, error_message)
                ):
                    error_class = "api_http_transient"
                    retryable = True
                else:
                    error_class = "api_http_error"
            else:
                try:
                    structured_output, usage, output_text, response_metadata = self._extract_structured_output(
                        response,
                        task_kind=task_kind,
                        schema=definition.schema,
                    )
                    if task_kind == "formula_audit":
                        validate_formula_audit_scope(
                            definition.request,
                            structured_output,
                        )
                except StructuredOutputViolation as exc:
                    error_class = "structured_output_invalid"
                    error_message = str(exc)
                    retryable = True
                except ContractViolation as exc:
                    error_class = "api_contract_drift"
                    error_message = str(exc)
                    circuit_break = True
                if error_class is None and task_kind == "simplify":
                    try:
                        with self._semantic_semaphore:
                            semantic_evidence = self._run_semantic_validator(
                                definition,
                                structured_output or {},
                            )
                    except (SimplificationContractError, SymbolicEvidenceError) as exc:
                        error_class = "validation_failed"
                        error_message = str(exc)
                        retryable = True
                        evidence = getattr(exc, "evidence", None)
                        if isinstance(evidence, Mapping):
                            semantic_evidence = dict(evidence)
                    except TimeoutError as exc:
                        error_class = "semantic_validator_timeout"
                        error_message = str(exc)
                        retryable = True
                    except Exception as exc:  # pragma: no cover - 真实符号进程兜底
                        error_class = "semantic_validator_error"
                        error_message = str(exc)
                        retryable = True
        except (httpx.TimeoutException, TimeoutError) as exc:
            error_class = "api_timeout"
            error_message = str(exc)
            retryable = True
        except (httpx.TransportError, OSError) as exc:
            error_class = "api_transport_error"
            error_message = str(exc)
            retryable = True
        except Exception as exc:  # pragma: no cover - transport 注入兜底
            error_class = "api_transport_error"
            error_message = str(exc)
            retryable = True

        finished_at = self.now_fn()
        response_body = response.body if response is not None else None
        audit_response_body = _response_body_for_audit(response_body)
        if usage is None and isinstance(response_body, Mapping) and isinstance(response_body.get("usage"), Mapping):
            usage = dict(response_body["usage"])
        validation: JsonDict = {
            "ok": error_class is None,
            "error_class": error_class,
            "error_message": error_message,
            "structured_output": structured_output,
        }
        if semantic_evidence is not None:
            validation["semantic_evidence"] = semantic_evidence
        metadata: JsonDict = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "evaluation_key": definition.task_spec.evaluation_key,
            "logical_id": definition.task_spec.logical_id,
            "task_type": definition.task_spec.task_type,
            "task_kind": task_kind,
            "requested_model": CONTRACT_CANONICAL_MODEL,
            "requested_effort": CONTRACT_EFFORT,
            "api_channel": channel.name,
            "api_base_url": channel.base_url,
            "api_endpoint": channel.endpoint,
            "transport_version": API_TRANSPORT_VERSION,
            "stream": False,
            "prompt_path": str(definition.prompt_path),
            "prompt_sha256": prompt_sha256,
            "rendered_prompt_sha256": _sha256_text(prompt),
            "system_prompt_sha256": _sha256_text(self.system_prompt),
            "semantic_validator_sha256": _sha256_file(Path(__file__).with_name('symbolic_evidence.py')),
            "schema_path": str(definition.schema_path),
            "schema_sha256": schema_sha256,
            "request_sha256": _sha256_text(canonical_json(definition.request)),
            "api_request_sha256": _sha256_text(canonical_json(api_payload)),
            "api_response_sha256": _sha256_text(
                canonical_json(response_body) if response_body is not None else ""
            ),
            "persisted_api_response_sha256": _sha256_text(
                canonical_json(audit_response_body) if audit_response_body is not None else ""
            ),
            "raw_response_sha256": _sha256_text(
                response.text if response is not None else ""
            ),
            "http_status": response.status_code if response is not None else None,
            "response_headers": dict(response.headers) if response is not None else {},
            "response_model": response_body.get("model") if isinstance(response_body, Mapping) else None,
            "response_metadata": response_metadata,
            "lease_expires_at": lease_expires_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_latency_seconds": max(0.0, self.monotonic_fn() - started_monotonic),
            "timeout_seconds": self.timeout_seconds,
            "usage": usage,
            "estimated_cost_cny": _estimated_cost_cny(usage or {}),
            "tariff_cny_per_million_tokens": {
                "input": 3.0,
                "output": 15.0,
                "cache_read": 0.3,
            },
            "error_class": error_class,
            "retryable": retryable if error_class is not None else False,
        }
        attempt_payload: JsonDict = {
            "attempt_id": attempt_id,
            "evaluation_key": definition.task_spec.evaluation_key,
            "request": _sanitize_for_audit(dict(definition.request)),
            "prompt": _sanitize_for_audit(prompt),
            "api_request": _sanitize_for_audit(api_payload),
            "api_response": audit_response_body,
            "output_text": _sanitize_for_audit(output_text),
            "validation": _sanitize_for_audit(validation),
            "metadata": _sanitize_for_audit(metadata),
        }
        attempt_path = self.attempts_dir / f"{attempt_id}.json"
        _atomic_write_json(attempt_path, attempt_payload)

        if error_class is None:
            return self._freeze_success(
                definition=definition,
                attempt_id=attempt_id,
                prompt=prompt,
                command=command,
                stdout=(
                    canonical_json(audit_response_body)
                    if isinstance(audit_response_body, Mapping)
                    else ""
                ),
                stderr="",
                envelope=dict(audit_response_body) if isinstance(audit_response_body, Mapping) else {},
                structured_output=structured_output or {},
                validation=validation,
                metadata=metadata,
            )

        next_state = self.store.finish_failure(
            attempt_id,
            error_class=error_class,
            retryable=retryable,
            now=finished_at,
        )
        if circuit_break:
            raise ClaudeRunnerCircuitBreaker(
                f"Anthropic API 全局熔断: {error_message}",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=attempt_id,
            )
        return ClaudeRunResult(
            evaluation_key=definition.task_spec.evaluation_key,
            state=next_state,
            attempt_id=attempt_id,
            result_path=None,
            result_sha256=None,
            structured_output=None,
            from_cache=False,
            error_class=error_class,
            usage=usage,
            total_cost_usd=None,
            claude_version=(
                str(response_body.get("model"))
                if isinstance(response_body, Mapping) and isinstance(response_body.get("model"), str)
                else None
            ),
            total_cost_cny=(
                float(metadata["estimated_cost_cny"])
                if isinstance(metadata.get("estimated_cost_cny"), (int, float))
                else None
            ),
        )
