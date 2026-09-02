from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (  # noqa: E402
    STRICT_EVALUATOR_SYSTEM_PROMPT,
    AnthropicApiChannel,
    AnthropicApiResponse,
    AnthropicApiRunner,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (  # noqa: E402
    ClaudeRunnerCircuitBreaker,
    TaskDefinition,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_api_frozen_simplifications import (  # noqa: E402
    audit_api_frozen_simplifications,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_anthropic_api_plan import (  # noqa: E402
    load_api_channels,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _definition(tmp_path: Path, evaluation_key: str) -> TaskDefinition:
    prompt_template = "Judge the expressions.\n{{REQUEST_JSON}}\n"
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string"},
            "evidence_basis": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string"},
        },
        "required": [
            "decision",
            "evidence_basis",
            "assumptions",
            "confidence",
            "brief_reason",
        ],
    }
    prompt_path = tmp_path / f"{evaluation_key[:4]}.txt"
    schema_path = tmp_path / f"{evaluation_key[:4]}.json"
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(json.dumps(schema, sort_keys=True) + "\n", encoding="utf-8")
    spec = TaskSpec(
        evaluation_key=evaluation_key,
        logical_id=f"equivalence::{evaluation_key[:8]}",
        task_type="equivalence",
        condition="clean",
        priority=30,
        input_hash=f"input::{evaluation_key}",
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )
    return TaskDefinition(
        task_spec=spec,
        request={"lhs": "x0 + x1", "rhs": "x1 + x0", "evidence_hash": "fixture"},
        prompt_path=prompt_path,
        prompt_sha256=_sha256_text(prompt_template),
        schema_path=schema_path,
        schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        prompt_template=prompt_template,
        schema=schema,
        task_kind="equivalence",
    )


def _response(*, model: str = "claude-opus-5", text: str | None = None) -> AnthropicApiResponse:
    output = {
        "decision": "equivalent",
        "evidence_basis": "symbolic_proof",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "The expressions are identical.",
    }
    return AnthropicApiResponse(
        status_code=200,
        body={
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "thinking",
                    "thinking": "private-reasoning-must-not-be-persisted",
                    "signature": "fixture-signature",
                },
                {"type": "text", "text": text or json.dumps(output)},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
        text="fixture-response",
        headers={"request-id": "req_fixture"},
    )


def _simplify_definition(tmp_path: Path) -> TaskDefinition:
    prompt_template = "Simplify.\n{{REQUEST_JSON}}\n"
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "outcome": {"type": "string"},
            "simplified_expression": {"type": ["string", "null"]},
            "equivalence_assessment": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string"},
        },
        "required": [
            "outcome",
            "simplified_expression",
            "equivalence_assessment",
            "assumptions",
            "confidence",
            "brief_reason",
        ],
    }
    prompt_path = tmp_path / "simplify.txt"
    schema_path = tmp_path / "simplify.json"
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(json.dumps(schema, sort_keys=True) + "\n", encoding="utf-8")
    request = {
        "expression": "x0 + 0",
        "original_expression": "x0 + 0",
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": "fixture",
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": _sha256_text(prompt_template),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }
    logical_id = "pred_simplify::demo::g0001::s520::clean"
    task_key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=normalized_input["prompt_sha256"],
        schema_sha256=normalized_input["schema_sha256"],
        normalized_input=normalized_input,
        evidence_hash="fixture",
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash=_sha256_text(canonical_json(normalized_input)),
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    return TaskDefinition(
        task_spec=spec,
        request=request,
        prompt_path=prompt_path,
        prompt_sha256=_sha256_text(prompt_template),
        schema_path=schema_path,
        schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        prompt_template=prompt_template,
        schema=schema,
        task_kind="simplify",
    )


def _simplify_response() -> AnthropicApiResponse:
    output = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "Removed the additive zero.",
    }
    response = _response(text=json.dumps(output))
    return response


class FakeTransport:
    def __init__(self, responses: list[AnthropicApiResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        channel: AnthropicApiChannel,
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> AnthropicApiResponse:
        self.calls.append(
            {
                "channel": channel.name,
                "token": channel.auth_token,
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.responses:
            raise AssertionError("缺少 fake API 响应")
        return self.responses.pop(0)


def _runner(
    tmp_path: Path,
    store: TaskStateStore,
    transport: FakeTransport,
) -> AnthropicApiRunner:
    return AnthropicApiRunner(
        store,
        attempts_dir=tmp_path / "attempts",
        frozen_dir=tmp_path / "frozen",
        channels=(
            AnthropicApiChannel("routify", "https://routify.invalid/protocol/anthropic", "sk-routify"),
            AnthropicApiChannel("yapi", "https://yapi.invalid", "sk-yapi"),
        ),
        transport=transport,
        timeout_seconds=5,
        per_channel_concurrency=1,
        semantic_validation_concurrency=1,
        backoff_schedule_seconds=(0, 0),
    )


def test_non_stream_xhigh_payload_stable_sharding_and_secret_redaction(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_response(), _response()])
    runner = _runner(tmp_path, store, transport)

    result_even = runner.execute(_definition(tmp_path, "0" * 64))
    result_odd = runner.execute(_definition(tmp_path, "1" * 64))

    assert result_even.state == result_odd.state == "frozen"
    assert result_even.total_cost_cny == pytest.approx(0.0006)
    assert result_odd.total_cost_cny == pytest.approx(0.0006)
    assert [call["channel"] for call in transport.calls] == ["routify", "yapi"]
    for call in transport.calls:
        payload = call["payload"]
        assert payload["model"] == "claude-opus-5"
        assert payload["stream"] is False
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "xhigh"}
        assert payload["system"] == STRICT_EVALUATOR_SYSTEM_PROMPT

    audit_paths = [
        *(tmp_path / "attempts").glob("*.json"),
        *(tmp_path / "frozen").glob("*.json"),
    ]
    audit_text = "\n".join(path.read_text(encoding="utf-8") for path in audit_paths)
    assert "sk-routify" not in audit_text
    assert "sk-yapi" not in audit_text
    assert "private-reasoning-must-not-be-persisted" not in audit_text
    assert '"api_channel": "routify"' in audit_text
    assert '"api_channel": "yapi"' in audit_text


def test_invalid_json_retries_on_other_channel_then_freezes(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_response(text="not-json"), _response()])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "2" * 64))

    assert result.state == "frozen"
    assert result.total_cost_cny == pytest.approx(0.0012)
    assert [call["channel"] for call in transport.calls] == ["routify", "yapi"]
    assert store.attempts_reserved() == 2
    first = json.loads((tmp_path / "attempts" / f"{'2' * 64}.a01.json").read_text(encoding="utf-8"))
    assert first["metadata"]["error_class"] == "structured_output_invalid"
    assert first["metadata"]["retryable"] is True


def test_upstream_deployment_400_retries_on_other_channel(tmp_path: Path) -> None:
    deployment_failure = AnthropicApiResponse(
        status_code=400,
        body={
            "error": {
                "message": "AllModelsFailed: 555420: The deployment request could not be completed."
            }
        },
        text="deployment failure",
        headers={},
    )
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([deployment_failure, _response()])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "3" * 64))

    assert result.state == "frozen"
    channels = [call["channel"] for call in transport.calls]
    assert len(channels) == 2
    assert set(channels) == {"routify", "yapi"}
    first = json.loads((tmp_path / "attempts" / f"{'3' * 64}.a01.json").read_text(encoding="utf-8"))
    assert first["metadata"]["error_class"] == "api_http_transient"
    assert first["metadata"]["retryable"] is True


def test_ordinary_bad_request_remains_non_retryable(tmp_path: Path) -> None:
    bad_request = AnthropicApiResponse(
        status_code=400,
        body={"error": {"message": "invalid request schema"}},
        text="bad request",
        headers={},
    )
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([bad_request])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "4" * 64))

    assert result.state == "exhausted"
    assert [call["channel"] for call in transport.calls] == ["routify"]


def test_corrected_final_json_fence_is_recovered_without_retry(tmp_path: Path) -> None:
    valid = {
        "decision": "not_equivalent",
        "evidence_basis": "mixed",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "A deterministic counterexample separates the expressions.",
    }
    invalid = {**valid, "_note": None}
    text = (
        f"```json\n{json.dumps(invalid)}\n```\n\n"
        "Correction: the schema forbids extra properties.\n\n"
        f"```json\n{json.dumps(valid)}\n```"
    )
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_response(text=text)])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "9" * 64))

    assert result.state == "frozen"
    assert store.attempts_reserved() == 1
    attempt = json.loads(
        (tmp_path / "attempts" / f"{'9' * 64}.a01.json").read_text(encoding="utf-8")
    )
    assert attempt["validation"]["structured_output"] == valid
    assert attempt["metadata"]["response_metadata"]["structured_output_recovery"] == (
        "single_valid_json_fence"
    )


def test_corrected_bare_json_object_is_recovered_without_retry(tmp_path: Path) -> None:
    valid = {
        "decision": "not_equivalent",
        "evidence_basis": "mixed",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "A deterministic counterexample separates the expressions.",
    }
    invalid = {**valid, "unexpected": None}
    text = (
        f"{json.dumps(invalid)}\n\n"
        "Correction: the schema forbids the extra property.\n\n"
        f"{json.dumps(valid)}"
    )
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_response(text=text)])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "a" * 64))

    assert result.state == "frozen"
    assert store.attempts_reserved() == 1
    attempt = json.loads(
        (tmp_path / "attempts" / f"{'a' * 64}.a01.json").read_text(encoding="utf-8")
    )
    assert attempt["validation"]["structured_output"] == valid
    assert attempt["metadata"]["response_metadata"]["structured_output_recovery"] == (
        "single_valid_embedded_json_object"
    )


def test_transient_response_without_usage_rotates_channel_and_retries(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transient = AnthropicApiResponse(
        status_code=429,
        body={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        text="rate limited",
        headers={"retry-after": "1"},
    )
    transport = FakeTransport([transient, _response()])
    runner = _runner(tmp_path, store, transport)

    result = runner.execute(_definition(tmp_path, "8" * 64))

    assert result.state == "frozen"
    assert result.total_cost_cny == pytest.approx(0.0006)
    assert [call["channel"] for call in transport.calls] == ["routify", "yapi"]


def test_response_model_mismatch_triggers_circuit_breaker(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_response(model="claude-sonnet-4-6")])
    runner = _runner(tmp_path, store, transport)

    with pytest.raises(ClaudeRunnerCircuitBreaker, match="实际模型不符"):
        runner.execute(_definition(tmp_path, "4" * 64))

    assert store.task_state("4" * 64) == "exhausted"
    assert store.attempts_reserved() == 1


def test_load_api_channels_reads_named_settings_without_using_configured_model(tmp_path: Path) -> None:
    routify = tmp_path / "routify.json"
    yapi = tmp_path / "yapi.json"
    routify.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://routify.invalid/protocol/anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "sk-route",
                    "ANTHROPIC_MODEL": "some-other-model",
                }
            }
        ),
        encoding="utf-8",
    )
    yapi.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://yapi.invalid",
                    "ANTHROPIC_AUTH_TOKEN": "sk-yapi",
                }
            }
        ),
        encoding="utf-8",
    )

    channels = load_api_channels((f"routify={routify}", f"yapi={yapi}"))

    assert [(item.name, item.base_url) for item in channels] == [
        ("routify", "https://routify.invalid/protocol/anthropic"),
        ("yapi", "https://yapi.invalid"),
    ]
    assert [item.auth_token for item in channels] == ["sk-route", "sk-yapi"]


def test_single_api_channel_requires_explicit_degraded_mode(tmp_path: Path) -> None:
    routify = tmp_path / "routify.json"
    routify.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://routify.invalid/protocol/anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "sk-route",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="至少需要两个"):
        load_api_channels((f"routify={routify}",))

    channels = load_api_channels(
        (f"routify={routify}",),
        allow_single_channel=True,
    )
    assert [channel.name for channel in channels] == ["routify"]

    store = TaskStateStore(tmp_path / "single_state.sqlite3")
    runner = AnthropicApiRunner(
        store,
        attempts_dir=tmp_path / "single_attempts",
        frozen_dir=tmp_path / "single_frozen",
        channels=channels,
        transport=FakeTransport([_response()]),
        allow_single_channel=True,
        per_channel_concurrency=1,
        semantic_validation_concurrency=1,
        backoff_schedule_seconds=(0, 0),
    )
    result = runner.execute(_definition(tmp_path, "5" * 64))
    assert result.state == "frozen"


def test_api_simplification_frozen_result_passes_independent_audit(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    transport = FakeTransport([_simplify_response()])
    runner = _runner(tmp_path, store, transport)
    definition = _simplify_definition(tmp_path)

    result = runner.execute(definition)
    assert result.state == "frozen"
    plan_row = {
        **json.loads(definition.task_spec.canonical_json()),
        "task_kind": "simplify",
        "prompt_sha256": definition.prompt_sha256,
        "schema_sha256": definition.schema_sha256,
        "prompt_path": str(definition.prompt_path),
        "schema_path": str(definition.schema_path),
        "prompt_template": definition.prompt_template,
        "schema_content": definition.schema,
        "normalized_input": {
            "request": definition.request,
            "prompt_sha256": definition.prompt_sha256,
            "schema_sha256": definition.schema_sha256,
        },
        "request": definition.request,
        "task_spec": json.loads(definition.task_spec.canonical_json()),
        "rendered_prompt": render_prompt(
            definition.prompt_template,
            definition.request,
            definition.schema,
        ),
    }
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(plan_row) + "\n", encoding="utf-8")

    report = audit_api_frozen_simplifications(
        plan_jsonl=plan_path,
        state_db=tmp_path / "state.sqlite3",
        attempts_dir=tmp_path / "attempts",
        frozen_dir=tmp_path / "frozen",
        output_jsonl=tmp_path / "audit.jsonl",
        report_json=tmp_path / "audit-report.json",
        expected_api_count=1,
        semantic_timeout_seconds=10,
        workers=1,
    )

    assert report["status"] == "ok"
    assert report["passed_count"] == 1
    assert sum(report["channel_counts"].values()) == 1
    assert set(report["channel_counts"]) <= {"routify", "yapi"}
