from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_MODEL,
    canonical_json,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    ClaudeRunner,
    ClaudeRunnerCircuitBreaker,
    TaskDefinition,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec, TaskStateStore


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_definition_files(tmp_path: Path) -> tuple[Path, Path, str, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_template = "Solve this carefully.\n{{REQUEST_JSON}}\n"
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string"},
            "evidence_basis": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string"},
        },
        "required": ["decision", "evidence_basis", "assumptions", "confidence", "brief_reason"],
    }
    prompt_path = tmp_path / "equivalence.v1.txt"
    schema_path = tmp_path / "equivalence.v1.json"
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prompt_path, schema_path, prompt_template, schema


def _task_spec(key: str) -> TaskSpec:
    return TaskSpec(
        evaluation_key=key,
        logical_id=f"equivalence::{key}",
        task_type="equivalence",
        condition="clean",
        priority=1,
        input_hash=f"input::{key}",
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )


def _task_definition(tmp_path: Path, key: str, *, request: dict[str, object] | None = None) -> TaskDefinition:
    prompt_path, schema_path, prompt_template, schema = _write_definition_files(tmp_path)
    payload = {
        "lhs": "x0 + x1",
        "rhs": "x1 + x0",
        "anthropic_api_key": "sk-top-secret-123",
    }
    if request is not None:
        payload.update(request)
    return TaskDefinition(
        task_spec=_task_spec(key),
        request=payload,
        prompt_path=prompt_path,
        prompt_sha256=_sha256_text(prompt_template),
        schema_path=schema_path,
        schema_sha256=_sha256_text(schema_path.read_text(encoding="utf-8")),
        prompt_template=prompt_template,
        schema=schema,
        task_kind="equivalence",
    )


def _valid_envelope() -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 2,
        "stop_reason": "tool_use",
        "permission_denials": [],
        "structured_output": {
            "decision": "equivalent",
            "evidence_basis": "symbolic_proof",
            "assumptions": [],
            "confidence": 1.0,
            "brief_reason": "Expressions are identical.",
        },
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
            "total_cost_usd": 0.03125,
        },
        "subagent_stats": {"spawned": 0},
        "modelUsage": {
            CONTRACT_MODEL: {
                "canonicalModel": "claude-opus-5",
                "provider": "firstParty",
            }
        },
    }


class FakeSubprocessRun:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append({"command": list(command), **kwargs})
        if not self.outcomes:
            raise AssertionError("没有预设的 fake subprocess 结果")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _runner(tmp_path: Path, store: TaskStateStore, fake_run: FakeSubprocessRun) -> ClaudeRunner:
    return ClaudeRunner(
        store,
        attempts_dir=tmp_path / "llm" / "attempts",
        frozen_dir=tmp_path / "llm" / "frozen",
        timeout_seconds=5.0,
        lease_seconds=10.0,
        subprocess_run=fake_run,
        version_getter=lambda executable: f"{executable} 1.2.3",
        sleep_fn=lambda seconds: None,
        allow_non_claude_executable=False,
        backoff_schedule_seconds=(0.0, 0.0),
    )


def test_success_freezes_once_and_restart_is_idempotent(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-success")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            )
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    first = runner.execute(definition)
    second = runner.execute(definition)

    assert first.state == "frozen"
    assert first.from_cache is False
    assert first.structured_output == _valid_envelope()["structured_output"]
    assert first.total_cost_usd == pytest.approx(0.03125)
    assert first.claude_version == "claude 1.2.3"
    assert second.state == "frozen"
    assert second.from_cache is True
    assert len(fake_run.calls) == 1
    assert store.attempts_reserved() == 1

    attempt_audit = json.loads((tmp_path / "llm" / "attempts" / f"{first.attempt_id}.json").read_text(encoding="utf-8"))
    assert set(attempt_audit) == {
        "attempt_id",
        "evaluation_key",
        "request",
        "prompt",
        "command",
        "stdout",
        "stderr",
        "envelope",
        "validation",
        "metadata",
    }
    assert attempt_audit["metadata"]["prompt_sha256"] == definition.prompt_sha256
    assert attempt_audit["metadata"]["schema_sha256"] == definition.schema_sha256
    assert attempt_audit["metadata"]["wall_latency_seconds"] >= 0.0
    assert attempt_audit["metadata"]["usage"]["server_tool_use"] == {
        "web_fetch_requests": 0,
        "web_search_requests": 0,
    }
    audit_text = json.dumps(attempt_audit, ensure_ascii=False)
    assert "sk-top-secret-123" not in audit_text
    assert "[REDACTED]" in audit_text
    assert "env" not in attempt_audit["metadata"]
    assert attempt_audit["metadata"]["request_sha256"] == _sha256_text(canonical_json(definition.request))
    assert attempt_audit["metadata"]["rendered_prompt_sha256"] == _sha256_text(
        render_prompt(definition.prompt_template, definition.request)
    )
    assert attempt_audit["metadata"]["stdout_sha256"] == _sha256_text(
        json.dumps(_valid_envelope(), ensure_ascii=False)
    )
    assert attempt_audit["metadata"]["stderr_sha256"] == _sha256_text("")


def test_one_turn_end_turn_envelope_is_also_a_valid_single_call(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-one-turn")
    envelope = _valid_envelope()
    envelope["num_turns"] = 1
    envelope["stop_reason"] = "end_turn"
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(envelope, ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert result.structured_output == envelope["structured_output"]
    assert store.attempts_reserved() == 1


def test_cached_frozen_sha_drift_triggers_circuit_breaker_without_invocation(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-frozen-drift")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    first_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            )
        ]
    )
    runner = _runner(tmp_path, store, first_run)
    result = runner.execute(definition)
    assert result.state == "frozen"

    frozen_path = Path(result.result_path or "")
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["structured_output"]["brief_reason"] = "tampered"
    frozen_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    second_run = FakeSubprocessRun([])
    restarted_runner = _runner(tmp_path, store, second_run)
    with pytest.raises(ClaudeRunnerCircuitBreaker, match="frozen result"):
        restarted_runner.execute(definition)
    assert len(second_run.calls) == 0


@pytest.mark.parametrize(
    ("failure", "expected_error_class"),
    [
        (
            subprocess.TimeoutExpired(cmd=["claude"], timeout=5, output="", stderr="deadline reached"),
            "timeout",
        ),
        (
            subprocess.CompletedProcess(args=["claude"], returncode=2, stdout="", stderr="HTTP 429 please retry"),
            "http_transient",
        ),
        (
            subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="{bad json", stderr=""),
            "outer_json_invalid",
        ),
    ],
)
def test_retryable_failures_are_audited_then_retried(
    tmp_path: Path,
    failure: object,
    expected_error_class: str,
) -> None:
    definition = _task_definition(tmp_path, f"ek-retry-{expected_error_class}")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            failure,
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            ),
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    result = runner.execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 2
    first_attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(encoding="utf-8")
    )
    second_attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a02.json").read_text(encoding="utf-8")
    )
    assert first_attempt["validation"]["error_class"] == expected_error_class
    assert first_attempt["metadata"]["retryable"] is True
    assert second_attempt["validation"]["ok"] is True
    assert len(fake_run.calls) == 2


def test_timeout_expired_bytes_are_normalized_and_hashed_without_leaking_secrets(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-timeout-bytes")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    timeout_exc = subprocess.TimeoutExpired(
        cmd=["claude"],
        timeout=5,
        output=b"partial sk-timeout-secret",
        stderr=b"Bearer top-secret-token",
    )
    fake_run = FakeSubprocessRun(
        [
            timeout_exc,
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            ),
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    result = runner.execute(definition)

    assert result.state == "frozen"
    first_attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(encoding="utf-8")
    )
    assert first_attempt["stdout"] == "partial [REDACTED]"
    assert first_attempt["stderr"] == "[REDACTED]"
    assert first_attempt["metadata"]["stdout_sha256"] == _sha256_text("partial sk-timeout-secret")
    assert first_attempt["metadata"]["stderr_sha256"] == _sha256_text("Bearer top-secret-token")
    assert "sk-timeout-secret" not in json.dumps(first_attempt, ensure_ascii=False)
    assert "top-secret-token" not in json.dumps(first_attempt, ensure_ascii=False)


def test_three_attempts_exhaust_task(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-exhausted")
    store = TaskStateStore(
        tmp_path / "control" / "state.sqlite3",
        attempt_cap=10,
        max_attempts_per_task=3,
    )
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="{", stderr=""),
            subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="{", stderr=""),
            subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="{", stderr=""),
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    result = runner.execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "outer_json_invalid"
    assert store.attempts_reserved() == 3
    assert len(fake_run.calls) == 3
    with pytest.raises(Exception):
        runner.execute(definition)


def test_global_budget_is_reserved_before_invocation(tmp_path: Path) -> None:
    first = _task_definition(tmp_path / "task1", "ek-budget-a")
    second = _task_definition(tmp_path / "task2", "ek-budget-b")
    store = TaskStateStore(
        tmp_path / "control" / "state.sqlite3",
        attempt_cap=1,
        max_attempts_per_task=3,
    )
    fake_run = FakeSubprocessRun(
        [subprocess.CompletedProcess(args=["claude"], returncode=2, stdout="", stderr="temporary failure")]
    )
    runner = _runner(tmp_path, store, fake_run)

    first_result = runner.execute(first)
    assert first_result.state == "exhausted"
    assert len(fake_run.calls) == 1

    with pytest.raises(Exception):
        runner.execute(second)
    assert len(fake_run.calls) == 1


@pytest.mark.parametrize(
    ("mutator", "label"),
    [
        (lambda envelope: envelope["usage"]["server_tool_use"].__setitem__("web_search_requests", 1), "server-tool"),
        (lambda envelope: envelope.__setitem__("stop_reason", "completed"), "stop-reason"),
        (lambda envelope: envelope.__setitem__("num_turns", 1), "num-turns"),
        (lambda envelope: envelope.__setitem__("modelUsage", {"claude-sonnet": {"canonicalModel": "claude-sonnet"}}), "model"),
        (lambda envelope: envelope.__setitem__("usage", {}), "usage-metadata"),
    ],
)
def test_contract_drift_variants_raise_global_circuit_breaker_and_finish_failure(
    tmp_path: Path,
    mutator: object,
    label: str,
) -> None:
    definition = _task_definition(tmp_path, "ek-circuit")
    drifted = _valid_envelope()
    mutator(drifted)
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)  # type: ignore[misc]
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(drifted, ensure_ascii=False),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            ),
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    with pytest.raises(ClaudeRunnerCircuitBreaker, match="Claude 全局熔断"):
        runner.execute(definition)

    assert store.attempts_reserved() == 1
    assert store.task_state(definition.task_spec.evaluation_key) == "retry_wait"
    attempt_audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(encoding="utf-8")
    )
    assert attempt_audit["validation"]["error_class"] == "contract_drift"
    assert len(fake_run.calls) == 1, label
