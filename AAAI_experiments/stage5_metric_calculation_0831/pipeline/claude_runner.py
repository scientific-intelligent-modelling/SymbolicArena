"""Claude Code 逻辑任务执行器。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
    ContractViolation,
    StructuredOutputViolation,
    build_claude_command,
    canonical_json,
    render_prompt,
    validate_claude_envelope,
    validate_claude_execution_metadata,
    validate_claude_model_metadata,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    StateContractError,
    TaskSpec,
    TaskStateStore,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.semantic_validation_worker import (
    WORKER_MODULE as SEMANTIC_VALIDATION_WORKER_MODULE,
    build_semantic_validation_payload,
    validate_simplify_semantics_payload,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SimplificationContractError,
    SymbolicEvidenceError,
)


JsonDict = dict[str, object]

_SECRET_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:auth(?:entication)?[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"api[_-]?key|authorization|secret|password|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"([A-Z0-9_]*(?:TOKEN|API_KEY|SECRET|PASSWORD)[A-Z0-9_]*)=([^\s\"']+)", re.IGNORECASE),
)
_CIRCUIT_BREAK_PATTERNS = (
    "模型",
    "工具",
    "权限",
    "subagent",
    "session",
    "settings",
    "setting-sources",
    "stop_reason",
    "num_turns",
    "usage",
    "server_tool_use",
    "command contract",
    "task definition",
    "frozen result",
    "外层输出",
    "未正常完成",
)
_CLI_UNKNOWN_OPTION_PATTERN = re.compile(
    r"""
    \b(?:unknown|unrecognized|unexpected|invalid)\s+
    (?:option|argument|flag)\b
    [^-\n\r]*
    [:=]?
    [^-\n\r]*
    (?P<flag>--[a-z0-9][a-z0-9-]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_CLI_CONTRACT_FLAGS = frozenset(
    {
        "--print",
        "--safe-mode",
        "--setting-sources",
        "--model",
        "--effort",
        "--tools",
        "--max-turns",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--output-format",
        "--json-schema",
    }
)
SEMANTIC_VALIDATOR_VERSION = "symbolic_evidence.v2"
SEMANTIC_VALIDATOR_TRANSPORT_VERSION = "semantic_validation_worker.v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(path)


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.pattern.startswith("("):
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _sanitize_for_audit(value: object, *, parent_key: str | None = None) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY_PATTERN.search(key_text):
                sanitized[key_text] = "[REDACTED]"
            else:
                sanitized[key_text] = _sanitize_for_audit(item, parent_key=key_text)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_audit(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_audit(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        if parent_key and _SECRET_KEY_PATTERN.search(parent_key):
            return "[REDACTED]"
        return _redact_string(value)
    return value


def _parse_total_cost_usd(envelope: Mapping[str, object]) -> float | None:
    for candidate in (envelope.get("total_cost_usd"), envelope.get("cost_usd")):
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            return float(candidate)
    usage = envelope.get("usage")
    if isinstance(usage, Mapping):
        candidate = usage.get("total_cost_usd")
        if isinstance(candidate, bool):
            return None
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def _is_zero_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    return False


def _is_strict_empty_response(
    envelope: Mapping[str, object],
    usage: Mapping[str, object],
    server_tool_use: Mapping[str, object],
    model_usage: Mapping[str, object],
) -> bool:
    required_server_tool_keys = {"web_search_requests", "web_fetch_requests"}
    if not required_server_tool_keys.issubset(server_tool_use):
        return False
    if envelope.get("num_turns") != 1:
        return False
    if envelope.get("stop_reason") not in {None, ""}:
        return False
    result_text = envelope.get("result")
    if result_text != "":
        return False
    if model_usage:
        return False
    total_cost_usd = _parse_total_cost_usd(envelope)
    if total_cost_usd != 0.0:
        return False
    if any(not _is_zero_number(value) for value in server_tool_use.values()):
        return False

    zero_usage_keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    if any(not _is_zero_number(usage.get(key)) for key in zero_usage_keys):
        return False

    output_tokens_details = usage.get("output_tokens_details")
    if not isinstance(output_tokens_details, Mapping):
        return False
    if "thinking_tokens" not in output_tokens_details:
        return False
    if any(not _is_zero_number(value) for value in output_tokens_details.values()):
        return False

    cache_creation = usage.get("cache_creation")
    if not isinstance(cache_creation, Mapping):
        return False
    required_cache_creation_keys = {
        "ephemeral_1h_input_tokens",
        "ephemeral_5m_input_tokens",
    }
    if not required_cache_creation_keys.issubset(cache_creation):
        return False
    if any(not _is_zero_number(value) for value in cache_creation.values()):
        return False

    return True


def _normalize_text_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _default_semantic_validator_command_builder() -> list[str]:
    return [sys.executable, "-m", SEMANTIC_VALIDATION_WORKER_MODULE]


def _infer_task_kind(task_type: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if task_type.endswith("simplify"):
        return "simplify"
    if task_type.endswith("equivalence"):
        return "equivalence"
    if task_type.endswith("structure"):
        return "structure"
    raise ValueError(f"无法从 task_type 推断 task_kind: {task_type!r}")


def _is_retryable_cli_exit(stdout: str, stderr: str) -> tuple[str, bool]:
    text = f"{stdout}\n{stderr}".lower()
    if "prompt_too_long" in text or "prompt is too long" in text:
        # 请求体不会在原地随机变短；继续重试只会无效消耗任务级尝试预算。
        return ("prompt_too_long", False)
    if any(code in text for code in (" 408", " 429", " 500", " 502", " 503", " 504")):
        return ("http_transient", True)
    if "timed out" in text or "timeout" in text:
        return ("timeout", True)
    return ("cli_exit", True)


def _looks_like_cli_contract_drift(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if "--json-schema" in lowered and "not a valid json schema" in lowered:
        return True
    for match in _CLI_UNKNOWN_OPTION_PATTERN.finditer(text):
        if match.group("flag").lower() in _CLI_CONTRACT_FLAGS:
            return True
    return False


def _is_circuit_break_violation(exc: Exception) -> bool:
    if not isinstance(exc, ContractViolation):
        return False
    message = str(exc)
    return any(pattern in message for pattern in _CIRCUIT_BREAK_PATTERNS)


def _validate_simplify_semantics(
    definition: "TaskDefinition",
    structured_output: Mapping[str, object],
) -> JsonDict:
    return validate_simplify_semantics_payload(
        evaluation_key=definition.task_spec.evaluation_key,
        request=definition.request,
        structured_output=structured_output,
    )


class _SemanticValidatorTimeoutError(TimeoutError):
    def __init__(self, message: str, *, semantic_evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.semantic_evidence = dict(semantic_evidence)


class _SemanticValidatorSymbolicEvidenceFailure(SymbolicEvidenceError):
    def __init__(self, message: str, *, semantic_evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.semantic_evidence = dict(semantic_evidence)


def _enforce_runtime_envelope_contract(envelope: Mapping[str, object]) -> None:
    validate_claude_execution_metadata(envelope)
    usage = envelope.get("usage")
    assert isinstance(usage, Mapping)
    server_tool_use = usage.get("server_tool_use")
    assert isinstance(server_tool_use, Mapping)
    model_usage = envelope.get("modelUsage")
    if not isinstance(model_usage, Mapping):
        raise ContractViolation("modelUsage 元数据缺失或无效")
    if _is_strict_empty_response(envelope, usage, server_tool_use, model_usage):
        raise EmptyClaudeResponse("Claude 返回全零空响应")
    validate_claude_model_metadata(envelope)

    stop_reason = envelope.get("stop_reason")
    num_turns = envelope.get("num_turns")
    valid_stop_pairs = {(1, "end_turn")}
    if (num_turns, stop_reason) not in valid_stop_pairs:
        raise TurnContractViolation(
            f"num_turns/stop_reason 契约不符: {num_turns!r}/{stop_reason!r}"
        )


def _load_json_file(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return payload


def _canonicalize_schema(schema: Mapping[str, object]) -> JsonDict:
    payload = json.loads(canonical_json(schema))
    if not isinstance(payload, dict):
        raise ValueError("schema 必须是 JSON object")
    return payload


def _default_version_getter(executable: str) -> str | None:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return None
    text = completed.stdout.strip() or completed.stderr.strip()
    if not text:
        return None
    return text.splitlines()[0]


@dataclass(frozen=True)
class TaskDefinition:
    task_spec: TaskSpec
    request: Mapping[str, object]
    prompt_path: Path
    prompt_sha256: str
    schema_path: Path
    schema_sha256: str
    prompt_template: str
    schema: Mapping[str, object]
    task_kind: str | None = None


@dataclass(frozen=True)
class ClaudeRunResult:
    evaluation_key: str
    state: str
    attempt_id: str | None
    result_path: str | None
    result_sha256: str | None
    structured_output: JsonDict | None
    from_cache: bool
    error_class: str | None
    usage: JsonDict | None
    total_cost_usd: float | None
    claude_version: str | None


class ClaudeRunnerCircuitBreaker(RuntimeError):
    """模型/工具/定义/命令漂移导致的全局熔断。"""

    def __init__(self, message: str, *, evaluation_key: str, attempt_id: str | None) -> None:
        super().__init__(message)
        self.evaluation_key = evaluation_key
        self.attempt_id = attempt_id


class TurnContractViolation(ContractViolation):
    """turn 数与 stop_reason 未命中约定的单轮调用契约。"""


class EmptyClaudeResponse(ContractViolation):
    """Claude 返回了可审计但不可接受的全零空响应。"""


class ClaudeRunner:
    """包装单任务的 reserve -> invoke -> audit -> freeze/fail 生命周期。"""

    def __init__(
        self,
        store: TaskStateStore,
        *,
        attempts_dir: str | Path,
        frozen_dir: str | Path,
        timeout_seconds: float = 1800.0,
        lease_seconds: float | None = None,
        command_builder: Callable[[Mapping[str, object]], list[str]] = build_claude_command,
        subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        version_getter: Callable[[str], str | None] = _default_version_getter,
        now_fn: Callable[[], float] = time.time,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        allow_non_claude_executable: bool = False,
        backoff_schedule_seconds: Sequence[float] = (1.0, 2.0),
        backoff_jitter_ratio: float = 0.2,
        uniform_fn: Callable[[float, float], float] = random.uniform,
        semantic_validator_timeout_seconds: float = 90.0,
        semantic_validator_command_builder: Callable[[], list[str]] = _default_semantic_validator_command_builder,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")
        if not 0.0 <= backoff_jitter_ratio <= 1.0:
            raise ValueError("backoff_jitter_ratio 必须位于 [0, 1]")
        if semantic_validator_timeout_seconds <= 0:
            raise ValueError("semantic_validator_timeout_seconds 必须为正数")
        self.store = store
        self.attempts_dir = Path(attempts_dir)
        self.frozen_dir = Path(frozen_dir)
        self.scratch_dir = self.attempts_dir.parent / "scratch"
        self.timeout_seconds = float(timeout_seconds)
        self.lease_seconds = float(lease_seconds) if lease_seconds is not None else float(timeout_seconds) + 60.0
        self.command_builder = command_builder
        self.subprocess_run = subprocess_run
        self.version_getter = version_getter
        self.now_fn = now_fn
        self.monotonic_fn = monotonic_fn
        self.sleep_fn = sleep_fn
        self.allow_non_claude_executable = bool(allow_non_claude_executable)
        self.backoff_schedule_seconds = tuple(float(item) for item in backoff_schedule_seconds)
        self.backoff_jitter_ratio = float(backoff_jitter_ratio)
        self.uniform_fn = uniform_fn
        self.semantic_validator_timeout_seconds = float(semantic_validator_timeout_seconds)
        self.semantic_validator_command_builder = semantic_validator_command_builder

    def execute(self, definition: TaskDefinition) -> ClaudeRunResult:
        self.store.register_task(definition.task_spec)
        cached = self._load_existing_frozen(definition.task_spec.evaluation_key)
        if cached is not None:
            return cached

        task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
        prompt_sha256, schema_sha256 = self._verify_task_definition(definition)
        prompt = render_prompt(
            definition.prompt_template,
            definition.request,
            definition.schema,
        )
        command = self._build_and_validate_command(definition.schema)
        cumulative_cost_usd = 0.0
        has_cost = False

        while True:
            cached = self._load_existing_frozen(definition.task_spec.evaluation_key)
            if cached is not None:
                return cached

            lease = self.store.reserve_attempt(
                definition.task_spec.evaluation_key,
                now=self.now_fn(),
                lease_seconds=self.lease_seconds,
            )
            result = self._run_attempt(
                definition=definition,
                task_kind=task_kind,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                command=command,
                attempt_id=lease.attempt_id,
                attempt_number=lease.attempt_number,
                lease_expires_at=lease.lease_expires_at,
            )
            if result.total_cost_usd is not None:
                cumulative_cost_usd += float(result.total_cost_usd)
                has_cost = True
            if result.state == "retry_wait":
                delay = self._backoff_delay(lease.attempt_number)
                if delay > 0:
                    self.sleep_fn(delay)
                continue
            if has_cost:
                return replace(result, total_cost_usd=cumulative_cost_usd)
            return result

    def _verify_task_definition(self, definition: TaskDefinition) -> tuple[str, str]:
        prompt_path = Path(definition.prompt_path)
        schema_path = Path(definition.schema_path)
        if not prompt_path.is_file():
            raise ClaudeRunnerCircuitBreaker(
                f"task definition drift: prompt 文件不存在 {prompt_path}",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        if not schema_path.is_file():
            raise ClaudeRunnerCircuitBreaker(
                f"task definition drift: schema 文件不存在 {schema_path}",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        actual_prompt_sha = _sha256_file(prompt_path)
        actual_schema_sha = _sha256_file(schema_path)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        schema_payload = _load_json_file(schema_path)
        if actual_prompt_sha != definition.prompt_sha256:
            raise ClaudeRunnerCircuitBreaker(
                "task definition drift: prompt sha256 不匹配",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        if actual_schema_sha != definition.schema_sha256:
            raise ClaudeRunnerCircuitBreaker(
                "task definition drift: schema sha256 不匹配",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        if prompt_text != definition.prompt_template:
            raise ClaudeRunnerCircuitBreaker(
                "task definition drift: prompt 内容与冻结模板不一致",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        if _canonicalize_schema(schema_payload) != _canonicalize_schema(definition.schema):
            raise ClaudeRunnerCircuitBreaker(
                "task definition drift: schema 内容与冻结版本不一致",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=None,
            )
        return actual_prompt_sha, actual_schema_sha

    def _build_and_validate_command(self, schema: Mapping[str, object]) -> list[str]:
        expected = build_claude_command(schema)
        actual = list(self.command_builder(schema))
        if not actual:
            raise ClaudeRunnerCircuitBreaker(
                "command contract drift: command 为空",
                evaluation_key="unknown",
                attempt_id=None,
            )
        if self.allow_non_claude_executable:
            if actual[1:] != expected[1:]:
                raise ClaudeRunnerCircuitBreaker(
                    "command contract drift: 非 claude 测试命令必须保留完全一致的参数",
                    evaluation_key="unknown",
                    attempt_id=None,
                )
        elif actual != expected:
            raise ClaudeRunnerCircuitBreaker(
                "command contract drift: Claude 命令参数与 build_claude_command 不一致",
                evaluation_key="unknown",
                attempt_id=None,
            )
        if not self.allow_non_claude_executable and actual[0] != "claude":
            raise ClaudeRunnerCircuitBreaker(
                "command contract drift: 生产执行器只允许 claude 可执行文件",
                evaluation_key="unknown",
                attempt_id=None,
            )
        return actual

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
        start_at = self.now_fn()
        started_monotonic = self.monotonic_fn()
        claude_version_error: str | None = None
        try:
            claude_version = self.version_getter(command[0])
        except Exception as exc:  # pragma: no cover - 仅兜底
            claude_version = None
            claude_version_error = str(exc)

        stdout = ""
        stderr = ""
        envelope: JsonDict | None = None
        validation: JsonDict = {"ok": False, "structured_output": None}
        error_class: str | None = None
        retryable = True
        circuit_break = False
        returncode: int | None = None
        timed_out = False
        total_cost_usd: float | None = None
        usage: JsonDict | None = None
        structured_output: JsonDict | None = None
        scratch_path = self.scratch_dir / attempt_id
        semantic_algorithm_path = Path(__file__).with_name("symbolic_evidence.py")
        semantic_worker_path = Path(__file__).with_name("semantic_validation_worker.py")

        try:
            scratch_path.mkdir(parents=True, exist_ok=False)
            request_text = canonical_json(definition.request)
            rendered_prompt_sha256 = _sha256_text(prompt)
            completed = self.subprocess_run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                cwd=str(scratch_path),
                timeout=self.timeout_seconds,
            )
            returncode = int(completed.returncode)
            stdout = _normalize_text_output(completed.stdout)
            stderr = _normalize_text_output(completed.stderr)
            parsed_stdout: object | None = None
            stdout_json_error: json.JSONDecodeError | None = None
            if stdout.strip():
                try:
                    parsed_stdout = json.loads(stdout)
                except json.JSONDecodeError as exc:
                    stdout_json_error = exc
                else:
                    if isinstance(parsed_stdout, dict):
                        envelope = parsed_stdout
                        usage_raw = envelope.get("usage")
                        usage = dict(usage_raw) if isinstance(usage_raw, Mapping) else None
                        total_cost_usd = _parse_total_cost_usd(envelope)

            if returncode != 0:
                if _looks_like_cli_contract_drift(stdout, stderr):
                    error_class, retryable = ("contract_drift", False)
                    circuit_break = True
                else:
                    error_class, retryable = _is_retryable_cli_exit(stdout, stderr)
                validation = {
                    "ok": False,
                    "error_class": error_class,
                    "error_message": stderr.strip() or stdout.strip() or f"rc={returncode}",
                    "structured_output": None,
                }
            elif not stdout.strip():
                error_class, retryable = ("empty_stdout", True)
                validation = {
                    "ok": False,
                    "error_class": error_class,
                    "error_message": "Claude stdout 为空",
                    "structured_output": None,
                }
            else:
                if stdout_json_error is not None:
                    error_class, retryable = ("outer_json_invalid", True)
                    validation = {
                        "ok": False,
                        "error_class": error_class,
                        "error_message": str(stdout_json_error),
                        "structured_output": None,
                    }
                elif not isinstance(parsed_stdout, dict):
                    error_class, retryable = ("outer_json_invalid", True)
                    validation = {
                        "ok": False,
                        "error_class": error_class,
                        "error_message": "Claude 外层输出不是 JSON object",
                        "structured_output": None,
                    }
                else:
                    assert envelope is not None
                    try:
                        _enforce_runtime_envelope_contract(envelope)
                        structured_output = validate_claude_envelope(
                            envelope,
                            task_kind=task_kind,
                            schema=definition.schema,
                        )
                    except EmptyClaudeResponse as exc:
                        error_class = "empty_response"
                        retryable = True
                        circuit_break = False
                        validation = {
                            "ok": False,
                            "error_class": error_class,
                            "error_message": str(exc),
                            "structured_output": None,
                        }
                    except TurnContractViolation as exc:
                        error_class = "turn_contract_violation"
                        retryable = True
                        circuit_break = False
                        validation = {
                            "ok": False,
                            "error_class": error_class,
                            "error_message": str(exc),
                            "structured_output": None,
                        }
                    except StructuredOutputViolation as exc:
                        error_class = "validation_failed"
                        retryable = True
                        circuit_break = False
                        validation = {
                            "ok": False,
                            "error_class": error_class,
                            "error_message": str(exc),
                            "structured_output": None,
                        }
                    except ContractViolation as exc:
                        error_class = "contract_drift" if _is_circuit_break_violation(exc) else "validation_failed"
                        retryable = not _is_circuit_break_violation(exc)
                        circuit_break = _is_circuit_break_violation(exc)
                        validation = {
                            "ok": False,
                            "error_class": error_class,
                            "error_message": str(exc),
                            "structured_output": None,
                        }
                    else:
                        semantic_evidence: JsonDict | None = None
                        if task_kind == "simplify":
                            try:
                                semantic_evidence = self._run_semantic_validator(
                                    definition,
                                    structured_output,
                                )
                            except SimplificationContractError as exc:
                                error_class = "validation_failed"
                                retryable = True
                                semantic_evidence = dict(exc.evidence)
                                validation = {
                                    "ok": False,
                                    "error_class": error_class,
                                    "error_message": str(exc),
                                    "structured_output": structured_output,
                                    "semantic_evidence": semantic_evidence,
                                }
                            except _SemanticValidatorSymbolicEvidenceFailure as exc:
                                error_class = "validation_failed"
                                retryable = True
                                semantic_evidence = dict(exc.semantic_evidence)
                                validation = {
                                    "ok": False,
                                    "error_class": error_class,
                                    "error_message": str(exc),
                                    "structured_output": structured_output,
                                    "semantic_evidence": semantic_evidence,
                                }
                            except _SemanticValidatorTimeoutError as exc:
                                error_class = "semantic_validator_timeout"
                                retryable = True
                                semantic_evidence = dict(exc.semantic_evidence)
                                validation = {
                                    "ok": False,
                                    "error_class": error_class,
                                    "error_message": str(exc),
                                    "structured_output": structured_output,
                                    "semantic_evidence": semantic_evidence,
                                }
                            except Exception as exc:  # pragma: no cover - 符号库异常兜底
                                error_class = "semantic_validator_error"
                                retryable = True
                                semantic_evidence = {
                                    "decision": "validator_error",
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                                validation = {
                                    "ok": False,
                                    "error_class": error_class,
                                    "error_message": str(exc),
                                    "structured_output": structured_output,
                                    "semantic_evidence": semantic_evidence,
                                }
                        if error_class is None:
                            validation = {
                                "ok": True,
                                "error_class": None,
                                "error_message": None,
                                "structured_output": structured_output,
                            }
                            if semantic_evidence is not None:
                                validation["semantic_evidence"] = semantic_evidence
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            request_text = canonical_json(definition.request)
            rendered_prompt_sha256 = _sha256_text(prompt)
            stdout = _normalize_text_output(exc.stdout)
            stderr = _normalize_text_output(exc.stderr)
            error_class, retryable = ("timeout", True)
            validation = {
                "ok": False,
                "error_class": error_class,
                "error_message": str(exc),
                "structured_output": None,
            }
        except OSError as exc:
            request_text = canonical_json(definition.request)
            rendered_prompt_sha256 = _sha256_text(prompt)
            error_class, retryable = ("local_io_error", True)
            validation = {
                "ok": False,
                "error_class": error_class,
                "error_message": str(exc),
                "structured_output": None,
            }
        else:
            request_text = canonical_json(definition.request)
            rendered_prompt_sha256 = _sha256_text(prompt)

        finished_at = self.now_fn()
        wall_latency_seconds = max(0.0, self.monotonic_fn() - started_monotonic)
        metadata: JsonDict = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "evaluation_key": definition.task_spec.evaluation_key,
            "logical_id": definition.task_spec.logical_id,
            "task_type": definition.task_spec.task_type,
            "task_kind": task_kind,
            "requested_model": CONTRACT_MODEL,
            "requested_effort": CONTRACT_EFFORT,
            "transport_version": CONTRACT_TRANSPORT_VERSION,
            "prompt_path": str(definition.prompt_path),
            "prompt_sha256": prompt_sha256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "schema_path": str(definition.schema_path),
            "schema_sha256": schema_sha256,
            "semantic_validator_version": SEMANTIC_VALIDATOR_VERSION,
            "semantic_validator_sha256": _sha256_file(semantic_algorithm_path),
            "semantic_validator_transport_version": SEMANTIC_VALIDATOR_TRANSPORT_VERSION,
            "semantic_validator_worker_sha256": _sha256_file(semantic_worker_path),
            "semantic_validator_timeout_seconds": self.semantic_validator_timeout_seconds,
            "request_sha256": _sha256_text(request_text),
            "stdout_sha256": _sha256_text(stdout),
            "stderr_sha256": _sha256_text(stderr),
            "lease_expires_at": lease_expires_at,
            "started_at": start_at,
            "finished_at": finished_at,
            "wall_latency_seconds": wall_latency_seconds,
            "timeout_seconds": self.timeout_seconds,
            "scratch_dir": str(scratch_path),
            "returncode": returncode,
            "timed_out": timed_out,
            "claude_version": claude_version,
            "claude_version_error": claude_version_error,
            "usage": usage,
            "total_cost_usd": total_cost_usd,
            "error_class": error_class,
            "retryable": retryable if error_class is not None else False,
        }
        attempt_payload: JsonDict = {
            "attempt_id": attempt_id,
            "evaluation_key": definition.task_spec.evaluation_key,
            "request": _sanitize_for_audit(dict(definition.request)),
            "prompt": _sanitize_for_audit(prompt),
            "command": _sanitize_for_audit(list(command)),
            "stdout": _sanitize_for_audit(stdout),
            "stderr": _sanitize_for_audit(stderr),
            "envelope": _sanitize_for_audit(envelope),
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
                stdout=stdout,
                stderr=stderr,
                envelope=envelope or {},
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
                f"Claude 全局熔断: {validation.get('error_message')}",
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
            total_cost_usd=total_cost_usd,
            claude_version=claude_version,
        )

    def _terminate_semantic_validator_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.terminate()
            except Exception:
                return
        try:
            process.wait(timeout=0.2)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.kill()
            except Exception:
                return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

    def _run_semantic_validator(
        self,
        definition: TaskDefinition,
        structured_output: Mapping[str, object],
    ) -> JsonDict:
        payload = build_semantic_validation_payload(
            evaluation_key=definition.task_spec.evaluation_key,
            request=definition.request,
            structured_output=structured_output,
        )
        command = list(self.semantic_validator_command_builder())
        if not command:
            raise RuntimeError("semantic validator command 为空")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(_REPO_ROOT),
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"启动 semantic validator 失败: {exc}") from exc
        try:
            stdout, stderr = process.communicate(
                canonical_json(payload),
                timeout=self.semantic_validator_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_semantic_validator_process(process)
            try:
                stdout, stderr = process.communicate(timeout=5.0)
            except Exception:
                stdout = _normalize_text_output(exc.stdout)
                stderr = _normalize_text_output(exc.stderr)
            else:
                stdout = _normalize_text_output(stdout)
                stderr = _normalize_text_output(stderr)
            semantic_evidence = {
                "decision": "validator_timeout",
                "error_type": "TimeoutExpired",
                "error_message": (
                    f"semantic validator 超时，超过 {self.semantic_validator_timeout_seconds:g} 秒"
                ),
            }
            raise _SemanticValidatorTimeoutError(
                str(semantic_evidence["error_message"]),
                semantic_evidence=semantic_evidence,
            ) from exc
        stdout = _normalize_text_output(stdout)
        stderr = _normalize_text_output(stderr)
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"rc={process.returncode}"
            raise RuntimeError(f"semantic validator 进程异常退出: {detail}")
        if not stdout.strip():
            raise RuntimeError("semantic validator stdout 为空")
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"semantic validator 输出不是合法 JSON: {exc}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("semantic validator 输出不是 JSON object")
        semantic_evidence = response.get("semantic_evidence")
        if not isinstance(semantic_evidence, Mapping):
            raise RuntimeError("semantic validator 输出缺少 semantic_evidence")
        semantic_evidence_payload = dict(semantic_evidence)
        status = response.get("status")
        if status == "ok":
            return semantic_evidence_payload
        error_message = str(response.get("error_message") or "semantic validator 返回错误")
        if status == "simplification_contract_error":
            raise SimplificationContractError(error_message, evidence=semantic_evidence_payload)
        if status == "symbolic_evidence_error":
            raise _SemanticValidatorSymbolicEvidenceFailure(
                error_message,
                semantic_evidence=semantic_evidence_payload,
            )
        raise RuntimeError(f"semantic validator 返回未知状态: {status!r}")

    def _freeze_success(
        self,
        *,
        definition: TaskDefinition,
        attempt_id: str,
        prompt: str,
        command: Sequence[str],
        stdout: str,
        stderr: str,
        envelope: Mapping[str, object],
        structured_output: Mapping[str, object],
        validation: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> ClaudeRunResult:
        frozen_path = self.frozen_dir / f"{definition.task_spec.evaluation_key}.json"
        frozen_payload: JsonDict = {
            "attempt_id": attempt_id,
            "evaluation_key": definition.task_spec.evaluation_key,
            "logical_id": definition.task_spec.logical_id,
            "task_type": definition.task_spec.task_type,
            "task_kind": _infer_task_kind(definition.task_spec.task_type, definition.task_kind),
            "request": _sanitize_for_audit(dict(definition.request)),
            "prompt": _sanitize_for_audit(prompt),
            "command": _sanitize_for_audit(list(command)),
            "stdout": _sanitize_for_audit(stdout),
            "stderr": _sanitize_for_audit(stderr),
            "envelope": _sanitize_for_audit(dict(envelope)),
            "structured_output": _sanitize_for_audit(dict(structured_output)),
            "validation": _sanitize_for_audit(dict(validation)),
            "metadata": _sanitize_for_audit(dict(metadata)),
        }
        _atomic_write_json(frozen_path, frozen_payload)
        frozen_sha = _sha256_file(frozen_path)
        try:
            self.store.freeze_result(
                attempt_id,
                result_path=str(frozen_path),
                result_sha256=frozen_sha,
                now=self.now_fn(),
            )
        except StateContractError:
            cached = self._load_existing_frozen(definition.task_spec.evaluation_key)
            if cached is not None:
                return cached
            raise
        return ClaudeRunResult(
            evaluation_key=definition.task_spec.evaluation_key,
            state="frozen",
            attempt_id=attempt_id,
            result_path=str(frozen_path),
            result_sha256=frozen_sha,
            structured_output=dict(structured_output),
            from_cache=False,
            error_class=None,
            usage=dict(metadata["usage"]) if isinstance(metadata.get("usage"), Mapping) else None,
            total_cost_usd=float(metadata["total_cost_usd"]) if isinstance(metadata.get("total_cost_usd"), (int, float)) else None,
            claude_version=str(metadata["claude_version"]) if isinstance(metadata.get("claude_version"), str) else None,
        )

    def _load_existing_frozen(self, evaluation_key: str) -> ClaudeRunResult | None:
        frozen = self.store.frozen_result(evaluation_key)
        if frozen is None:
            return None
        frozen_path = Path(frozen["result_path"])
        if not frozen_path.is_file():
            raise ClaudeRunnerCircuitBreaker(
                f"frozen result drift: 结果文件不存在 {frozen_path}",
                evaluation_key=evaluation_key,
                attempt_id=frozen.get("attempt_id"),
            )
        actual_sha = _sha256_file(frozen_path)
        if actual_sha != frozen["result_sha256"]:
            raise ClaudeRunnerCircuitBreaker(
                "frozen result drift: 文件 SHA256 与状态库不一致",
                evaluation_key=evaluation_key,
                attempt_id=frozen.get("attempt_id"),
            )
        try:
            payload = _load_json_file(frozen_path)
        except Exception as exc:
            raise ClaudeRunnerCircuitBreaker(
                f"frozen result drift: 结果文件不可解析: {exc}",
                evaluation_key=evaluation_key,
                attempt_id=frozen.get("attempt_id"),
            ) from exc
        structured_output = payload.get("structured_output")
        usage = None
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("usage"), Mapping):
            usage = dict(metadata["usage"])
        return ClaudeRunResult(
            evaluation_key=evaluation_key,
            state="frozen",
            attempt_id=frozen.get("attempt_id"),
            result_path=frozen["result_path"],
            result_sha256=frozen["result_sha256"],
            structured_output=dict(structured_output) if isinstance(structured_output, Mapping) else None,
            from_cache=True,
            error_class=None,
            usage=usage,
            total_cost_usd=float(metadata["total_cost_usd"]) if isinstance(metadata, Mapping) and isinstance(metadata.get("total_cost_usd"), (int, float)) else None,
            claude_version=str(metadata["claude_version"]) if isinstance(metadata, Mapping) and isinstance(metadata.get("claude_version"), str) else None,
        )

    def _backoff_delay(self, attempt_number: int) -> float:
        index = max(0, attempt_number - 1)
        if index >= len(self.backoff_schedule_seconds):
            base_delay = (
                0.0
                if not self.backoff_schedule_seconds
                else self.backoff_schedule_seconds[-1]
            )
        else:
            base_delay = self.backoff_schedule_seconds[index]
        if base_delay <= 0.0 or self.backoff_jitter_ratio == 0.0:
            return max(0.0, base_delay)
        jitter = self.uniform_fn(
            -self.backoff_jitter_ratio,
            self.backoff_jitter_ratio,
        )
        return max(0.0, base_delay * (1.0 + jitter))
