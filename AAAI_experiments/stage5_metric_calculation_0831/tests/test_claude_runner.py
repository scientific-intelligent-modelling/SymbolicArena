from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
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
    _is_retryable_cli_exit,
    _looks_like_cli_contract_drift,
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
    structured_output = {
        "decision": "equivalent",
        "evidence_basis": "symbolic_proof",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "Expressions are identical.",
    }
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 1,
        "stop_reason": "end_turn",
        "permission_denials": [],
        "result": json.dumps(structured_output, ensure_ascii=False),
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


def _empty_response_envelope() -> dict[str, object]:
    envelope = _valid_envelope()
    envelope.update(
        {
            "result": "",
            "stop_reason": None,
            "total_cost_usd": 0,
            "modelUsage": {},
        }
    )
    envelope["usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "total_cost_usd": 0,
    }
    return envelope


def _simplify_task_definition(tmp_path: Path, key: str) -> TaskDefinition:
    prompt_template = "Simplify this expression.\n{{REQUEST_JSON}}\n"
    schema = {
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
    prompt_path = tmp_path / "simplify.v1.txt"
    schema_path = tmp_path / "simplify.v1.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=f"gt_simplify::{key}",
        task_type="gt_simplify",
        condition="clean",
        priority=1,
        input_hash=f"input::{key}",
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    return TaskDefinition(
        task_spec=spec,
        request={
            "expression": "x0 + x1",
            "variables": ["x0", "x1"],
            "allowed_functions": [],
            "evidence_hash": "evidence",
        },
        prompt_path=prompt_path,
        prompt_sha256=_sha256_text(prompt_template),
        schema_path=schema_path,
        schema_sha256=_sha256_text(schema_path.read_text(encoding="utf-8")),
        prompt_template=prompt_template,
        schema=schema,
        task_kind="simplify",
    )


def _simplify_envelope(
    expression: str | None,
    *,
    outcome: str = "simplified",
) -> dict[str, object]:
    envelope = _valid_envelope()
    envelope["result"] = json.dumps({
        "outcome": outcome,
        "simplified_expression": expression,
        "equivalence_assessment": "undetermined" if outcome == "unable" else "preserved",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "Simplification result.",
    }, ensure_ascii=False)
    return envelope


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


def _runner(
    tmp_path: Path,
    store: TaskStateStore,
    fake_run: FakeSubprocessRun,
    **overrides: object,
) -> ClaudeRunner:
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
        **overrides,
    )


def _write_validator_script(tmp_path: Path, name: str, body: str) -> Path:
    script_path = tmp_path / name
    script_path.write_text(body, encoding="utf-8")
    return script_path


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
    assert first.structured_output == json.loads(str(_valid_envelope()["result"]))
    assert first.total_cost_usd == pytest.approx(0.03125)
    assert first.claude_version == "claude 1.2.3"
    assert second.state == "frozen"
    assert second.from_cache is True
    assert len(fake_run.calls) == 1
    assert store.attempts_reserved() == 1
    scratch_path = tmp_path / "llm" / "scratch" / str(first.attempt_id)
    assert fake_run.calls[0]["cwd"] == str(scratch_path)
    assert scratch_path.is_dir()

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
    assert attempt_audit["metadata"]["scratch_dir"] == str(scratch_path)
    algorithm_path = (
        REPO_ROOT
        / "AAAI_experiments"
        / "stage5_metric_calculation_0831"
        / "pipeline"
        / "symbolic_evidence.py"
    )
    worker_path = (
        REPO_ROOT
        / "AAAI_experiments"
        / "stage5_metric_calculation_0831"
        / "pipeline"
        / "semantic_validation_worker.py"
    )
    assert attempt_audit["metadata"]["semantic_validator_version"] == "symbolic_evidence.v2"
    assert (
        attempt_audit["metadata"]["semantic_validator_transport_version"]
        == "semantic_validation_worker.v1"
    )
    assert attempt_audit["metadata"]["semantic_validator_sha256"] == _sha256_text(
        algorithm_path.read_text(encoding="utf-8")
    )
    assert attempt_audit["metadata"]["semantic_validator_worker_sha256"] == _sha256_text(
        worker_path.read_text(encoding="utf-8")
    )
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
        render_prompt(
            definition.prompt_template,
            definition.request,
            definition.schema,
        )
    )
    assert attempt_audit["metadata"]["stdout_sha256"] == _sha256_text(
        json.dumps(_valid_envelope(), ensure_ascii=False)
    )
    assert attempt_audit["metadata"]["stderr_sha256"] == _sha256_text("")


def test_plain_json_end_turn_envelope_is_a_valid_single_call(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-one-turn")
    envelope = _valid_envelope()
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
    assert result.structured_output == json.loads(str(envelope["result"]))
    assert store.attempts_reserved() == 1


def test_simplify_semantic_validation_is_audited_before_freeze(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-valid")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x1 + x0"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{result.attempt_id}.json").read_text(encoding="utf-8")
    )
    evidence = audit["validation"]["semantic_evidence"]
    assert evidence["decision"] == "equivalent"
    assert evidence["probe_seed"] >= 0


def test_simplify_uses_and_audits_frozen_dataset_probes(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-dataset-probes")
    points = [
        {"split": "id_test", "row_index": 3, "values": {"x0": 2.0, "x1": 5.0}}
    ]
    probe: dict[str, object] = {
        "schema_version": "dataset_probes_v1",
        "core50_index": 1,
        "dataset_name": "DemoSet",
        "basename": "DemoSet",
        "dataset_dir": "/tmp/DemoSet",
        "target_name": "y",
        "variables": ["x0", "x1"],
        "point_count": 1,
        "points": points,
        "source_sha256": {
            "metadata_yaml": "a" * 64,
            "id_test_csv": "b" * 64,
            "ood_test_csv": "c" * 64,
        },
    }
    sample_payload = {
        "schema_version": probe["schema_version"],
        "dataset_name": probe["dataset_name"],
        "variables": probe["variables"],
        "points": points,
    }
    probe["sample_sha256"] = _sha256_text(canonical_json(sample_payload))
    probe["evidence_sha256"] = _sha256_text(canonical_json(probe))
    request = dict(definition.request)
    request.update(
        {
            "dataset_id": "DemoSet",
            "probe_points": points,
            "probe_source": "dataset_probes_v1",
            "probe_sample_sha256": probe["sample_sha256"],
            "dataset_probe_evidence": probe,
        }
    )
    definition = replace(definition, request=request)
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x1 + x0"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{result.attempt_id}.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = audit["validation"]["semantic_evidence"]
    assert evidence["probe_source"] == "dataset_probes_v1"
    assert evidence["probe_sample_sha256"] == probe["sample_sha256"]
    assert evidence["probe_count"] == 1
    assert evidence["normalized_probe_points_sha256"] != probe["sample_sha256"]


def test_simplify_counterexample_retries_then_freezes_first_valid_result(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-retry")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x0 - x1"), ensure_ascii=False),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x0 + x1", outcome="unchanged"), ensure_ascii=False),
                stderr="",
            ),
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 2
    assert result.total_cost_usd == pytest.approx(0.0625)
    first = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["validation"]["error_class"] == "validation_failed"
    assert first["validation"]["semantic_evidence"]["counterexample"]


def test_simplify_symbolic_evidence_error_is_preserved_in_attempt_audit(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-symbolic-error")
    definition = replace(
        definition,
        request={
            "expression": "x0 + x1",
            "variables": ["x0", "x1"],
            "evidence_hash": "evidence",
        },
    )
    store = TaskStateStore(
        tmp_path / "control" / "state.sqlite3",
        attempt_cap=5,
        max_attempts_per_task=1,
    )
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x1 + x0"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "validation_failed"
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    semantic_evidence = attempt["validation"]["semantic_evidence"]
    assert semantic_evidence == {
        "decision": "contract_error",
        "error_type": "SymbolicEvidenceError",
        "error_message": "simplify request.allowed_functions 缺失或无效",
    }


def test_simplify_semantic_validator_timeout_is_retryable_and_kills_worker(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-validator-timeout")
    pid_path = tmp_path / "validator-timeout.pid"
    script_path = _write_validator_script(
        tmp_path,
        "sleeping_validator.py",
        "\n".join(
            [
                "import json",
                "import os",
                "import pathlib",
                "import sys",
                "import time",
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')",
                "json.load(sys.stdin)",
                "time.sleep(30)",
            ]
        )
        + "\n",
    )
    store = TaskStateStore(
        tmp_path / "control" / "state.sqlite3",
        attempt_cap=5,
        max_attempts_per_task=1,
    )
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x1 + x0"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(
        tmp_path,
        store,
        fake_run,
        semantic_validator_timeout_seconds=0.2,
        semantic_validator_command_builder=lambda: [sys.executable, str(script_path), str(pid_path)],
    ).execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "semantic_validator_timeout"
    assert pid_path.is_file()
    pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.time() + 3.0
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        if time.time() >= deadline:
            raise AssertionError(f"语义验证超时后子进程仍存活: pid={pid}")
        time.sleep(0.05)
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["validation"]["error_class"] == "semantic_validator_timeout"
    assert attempt["metadata"]["retryable"] is True
    assert attempt["validation"]["semantic_evidence"]["decision"] == "validator_timeout"


def test_simplify_semantic_validator_subprocess_crash_maps_to_retryable_error(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-validator-crash")
    script_path = _write_validator_script(
        tmp_path,
        "crashing_validator.py",
        "\n".join(
            [
                "import json",
                "import sys",
                "json.load(sys.stdin)",
                "sys.stderr.write('validator crashed\\n')",
                "raise RuntimeError('boom')",
            ]
        )
        + "\n",
    )
    store = TaskStateStore(
        tmp_path / "control" / "state.sqlite3",
        attempt_cap=5,
        max_attempts_per_task=1,
    )
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope("x1 + x0"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(
        tmp_path,
        store,
        fake_run,
        semantic_validator_timeout_seconds=1.0,
        semantic_validator_command_builder=lambda: [sys.executable, str(script_path)],
    ).execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "semantic_validator_error"
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["validation"]["error_class"] == "semantic_validator_error"
    assert attempt["metadata"]["retryable"] is True
    assert attempt["validation"]["semantic_evidence"]["decision"] == "validator_error"
    assert "validator crashed" in attempt["validation"]["semantic_evidence"]["error_message"]


def test_simplify_unable_is_valid_terminal_without_formula_validation(tmp_path: Path) -> None:
    definition = _simplify_task_definition(tmp_path, "ek-simplify-unable")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_simplify_envelope(None, outcome="unable"), ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 1
    frozen = json.loads(Path(result.result_path or "").read_text(encoding="utf-8"))
    assert frozen["validation"]["semantic_evidence"] == {
        "decision": "not_applicable",
        "reason": "outcome_unable",
    }


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


def test_retry_backoff_applies_bounded_jitter(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "state.sqlite3")
    runner = ClaudeRunner(
        store,
        attempts_dir=tmp_path / "attempts",
        frozen_dir=tmp_path / "frozen",
        backoff_schedule_seconds=(10.0, 20.0),
        backoff_jitter_ratio=0.25,
        uniform_fn=lambda lower, upper: upper,
    )
    assert runner._backoff_delay(1) == 12.5
    assert runner._backoff_delay(2) == 25.0
    assert runner._backoff_delay(3) == 25.0


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


def test_usage_token_counts_are_preserved_while_credentials_are_redacted(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-usage-audit")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    envelope = _valid_envelope()
    envelope["usage"]["anthropic_auth_token"] = "Bearer credential-test-value"  # type: ignore[index]
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
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["metadata"]["usage"]["input_tokens"] == 11
    assert attempt["metadata"]["usage"]["output_tokens"] == 7
    assert attempt["metadata"]["usage"]["anthropic_auth_token"] == "[REDACTED]"
    assert "credential-test-value" not in json.dumps(attempt, ensure_ascii=False)


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


def test_cli_rejected_schema_triggers_immediate_circuit_breaker(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-schema-rejected")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout="",
                stderr=(
                    "Error: --json-schema is not a valid JSON Schema: no schema "
                    'with key or ref "https://json-schema.org/draft/2020-12/schema"'
                ),
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            ),
        ]
    )

    with pytest.raises(ClaudeRunnerCircuitBreaker, match="Claude 全局熔断"):
        _runner(tmp_path, store, fake_run).execute(definition)

    assert store.attempts_reserved() == 1
    assert store.task_state(definition.task_spec.evaluation_key) == "exhausted"
    attempt = json.loads(
        (
            tmp_path
            / "llm"
            / "attempts"
            / f"{definition.task_spec.evaluation_key}.a01.json"
        ).read_text(encoding="utf-8")
    )
    assert attempt["validation"]["error_class"] == "contract_drift"
    assert attempt["metadata"]["retryable"] is False
    assert len(fake_run.calls) == 1


def test_structured_output_schema_violation_retries_without_global_breaker(
    tmp_path: Path,
) -> None:
    definition = _task_definition(tmp_path, "ek-result-schema-invalid")
    schema = json.loads(json.dumps(definition.schema, ensure_ascii=False))
    schema["properties"]["brief_reason"]["maxLength"] = 1000
    definition.schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    definition = replace(
        definition,
        schema=schema,
        schema_sha256=_sha256_text(definition.schema_path.read_text(encoding="utf-8")),
    )
    invalid = _valid_envelope()
    invalid_result = json.loads(str(invalid["result"]))
    invalid_result["brief_reason"] = "模型输出解释" * 201
    invalid["result"] = json.dumps(invalid_result, ensure_ascii=False)
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(invalid, ensure_ascii=False),
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

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 2
    first_attempt = json.loads(
        (
            tmp_path
            / "llm"
            / "attempts"
            / f"{definition.task_spec.evaluation_key}.a01.json"
        ).read_text(encoding="utf-8")
    )
    assert first_attempt["validation"]["error_class"] == "validation_failed"
    assert first_attempt["metadata"]["retryable"] is True
    assert len(fake_run.calls) == 2


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("Error: unknown option '--effort'", True),
        ('Error: unrecognized option "--no-session-persistence"', True),
        ("Error: unknown option: --json-schema", True),
        ("Error: invalid argument = --setting-sources", True),
        ("Error: unexpected flag '--max-turns' found", True),
        (
            'APIError: {"type":"invalid_request_error","message":"Unknown option for response_format: json_schema"}',
            False,
        ),
        ("Error: request failed with status 400", False),
    ],
)
def test_looks_like_cli_contract_drift_distinguishes_cli_flags_from_api_errors(
    stderr: str,
    expected: bool,
) -> None:
    assert _looks_like_cli_contract_drift("", stderr) is expected


@pytest.mark.parametrize(
    "payload",
    [
        '{"terminal_reason":"prompt_too_long"}',
        "Prompt is too long; reduce request size.",
    ],
)
def test_prompt_too_long_is_task_terminal_and_not_retried(payload: str) -> None:
    assert _is_retryable_cli_exit(payload, "") == ("prompt_too_long", False)


def test_nonzero_prompt_too_long_envelope_stops_after_one_attempt(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-prompt-too-long")
    envelope = _valid_envelope()
    envelope.update(
        {
            "is_error": True,
            "terminal_reason": "prompt_too_long",
            "stop_reason": "stop_sequence",
            "result": "Prompt is too long",
            "total_cost_usd": 0,
        }
    )
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout=json.dumps(envelope, ensure_ascii=False),
                stderr="",
            )
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "prompt_too_long"
    assert store.attempts_reserved() == 1
    assert len(fake_run.calls) == 1
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["validation"]["error_class"] == "prompt_too_long"
    assert attempt["metadata"]["retryable"] is False
    assert attempt["metadata"]["total_cost_usd"] == 0


def test_nonzero_exit_with_valid_envelope_keeps_usage_cost_audit_and_retries(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-nonzero-envelope")
    first_envelope = _valid_envelope()
    first_envelope["usage"]["total_cost_usd"] = 0.125  # type: ignore[index]
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout=json.dumps(first_envelope, ensure_ascii=False),
                stderr="Error: max_turns exceeded before completion",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_valid_envelope(), ensure_ascii=False),
                stderr="",
            ),
        ]
    )

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert result.total_cost_usd == pytest.approx(0.15625)
    first_attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_attempt["validation"]["ok"] is False
    assert first_attempt["validation"]["error_class"] == "cli_exit"
    assert first_attempt["metadata"]["retryable"] is True
    assert first_attempt["metadata"]["returncode"] == 1
    assert first_attempt["metadata"]["total_cost_usd"] == pytest.approx(0.125)
    assert first_attempt["metadata"]["usage"]["total_cost_usd"] == pytest.approx(0.125)
    assert first_attempt["envelope"]["usage"]["total_cost_usd"] == pytest.approx(0.125)
    assert len(fake_run.calls) == 2


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
    assert store.task_state(definition.task_spec.evaluation_key) == "exhausted"
    attempt_audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(encoding="utf-8")
    )
    assert attempt_audit["validation"]["error_class"] == "contract_drift"
    assert attempt_audit["metadata"]["retryable"] is False
    assert len(fake_run.calls) == 1, label


@pytest.mark.parametrize(
    ("mutator", "label"),
    [
        (lambda envelope: envelope.__setitem__("num_turns", 2), "num-turns"),
        (lambda envelope: envelope.__setitem__("stop_reason", "completed"), "stop-reason"),
    ],
)
def test_turn_contract_violation_is_rejected_then_retried_without_global_breaker(
    tmp_path: Path,
    mutator: object,
    label: str,
) -> None:
    definition = _task_definition(tmp_path, f"ek-turn-contract-{label}")
    drifted = _valid_envelope()
    mutator(drifted)  # type: ignore[misc]
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
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

    result = runner.execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 2
    assert store.task_state(definition.task_spec.evaluation_key) == "frozen"
    attempt_audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt_audit["validation"]["error_class"] == "turn_contract_violation"
    assert attempt_audit["metadata"]["retryable"] is True
    assert len(fake_run.calls) == 2


def test_repeated_turn_contract_violation_stops_at_task_attempt_cap(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-turn-contract-exhausted")
    drifted = _valid_envelope()
    drifted["num_turns"] = 2
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(drifted, ensure_ascii=False),
                stderr="",
            )
            for _ in range(3)
        ]
    )
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "exhausted"
    assert result.error_class == "turn_contract_violation"
    assert store.attempts_reserved() == 3
    assert store.task_state(definition.task_spec.evaluation_key) == "exhausted"
    assert len(fake_run.calls) == 3


@pytest.mark.parametrize(
    ("mutator", "label"),
    [
        (lambda envelope: envelope["usage"]["server_tool_use"].__setitem__("web_search_requests", 1), "server-tool"),
        (lambda envelope: envelope.__setitem__("modelUsage", {"claude-sonnet": {"canonicalModel": "claude-sonnet"}}), "model"),
        (
            lambda envelope: envelope["modelUsage"][CONTRACT_MODEL].__setitem__(
                "canonicalModel", "claude-sonnet"
            ),
            "canonical-model",
        ),
        (lambda envelope: envelope.__setitem__("permission_denials", ["tool"]), "permission"),
        (lambda envelope: envelope["subagent_stats"].__setitem__("spawned", 1), "subagent"),
        (lambda envelope: envelope.__setitem__("terminal_reason", "error"), "terminal-status"),
    ],
)
def test_non_turn_contract_drift_takes_precedence_over_turn_contract_violation(
    tmp_path: Path,
    mutator: object,
    label: str,
) -> None:
    definition = _task_definition(tmp_path, f"ek-drift-precedence-{label}")
    drifted = _valid_envelope()
    drifted["num_turns"] = 2
    mutator(drifted)  # type: ignore[misc]
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(drifted, ensure_ascii=False),
                stderr="",
            )
        ]
    )
    runner = _runner(tmp_path, store, fake_run)

    with pytest.raises(ClaudeRunnerCircuitBreaker, match="Claude 全局熔断"):
        runner.execute(definition)

    assert store.attempts_reserved() == 1
    assert store.task_state(definition.task_spec.evaluation_key) == "exhausted"
    attempt_audit = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt_audit["validation"]["error_class"] == "contract_drift"
    assert attempt_audit["metadata"]["retryable"] is False
    assert len(fake_run.calls) == 1


def test_strict_empty_response_is_retryable_without_circuit_break(tmp_path: Path) -> None:
    definition = _task_definition(tmp_path, "ek-empty-response")
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(_empty_response_envelope(), ensure_ascii=False),
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

    result = _runner(tmp_path, store, fake_run).execute(definition)

    assert result.state == "frozen"
    assert store.attempts_reserved() == 2
    first_attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_attempt["validation"]["error_class"] == "empty_response"
    assert first_attempt["metadata"]["retryable"] is True
    assert len(fake_run.calls) == 2


@pytest.mark.parametrize(
    ("mutator", "label"),
    [
        (lambda envelope: envelope["usage"].__setitem__("input_tokens", 1), "nonzero-usage"),
        (
            lambda envelope: envelope["usage"]["server_tool_use"].__setitem__("web_search_requests", 1),
            "server-tool",
        ),
        (
            lambda envelope: envelope.__setitem__(
                "modelUsage",
                {"claude-sonnet": {"canonicalModel": "claude-sonnet"}},
            ),
            "wrong-model-key",
        ),
        (lambda envelope: envelope.__setitem__("result", "{\"decision\":\"equivalent\"}"), "nonempty-result"),
        (
            lambda envelope: envelope["usage"]["cache_creation"].pop(
                "ephemeral_5m_input_tokens"
            ),
            "incomplete-zero-usage",
        ),
    ],
)
def test_empty_response_guard_is_strict_and_other_drift_keeps_original_behavior(
    tmp_path: Path,
    mutator: object,
    label: str,
) -> None:
    definition = _task_definition(tmp_path, f"ek-empty-response-negative-{label}")
    drifted = _empty_response_envelope()
    mutator(drifted)  # type: ignore[misc]
    store = TaskStateStore(tmp_path / "control" / "state.sqlite3", attempt_cap=5)
    fake_run = FakeSubprocessRun(
        [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(drifted, ensure_ascii=False),
                stderr="",
            )
        ]
    )

    with pytest.raises(ClaudeRunnerCircuitBreaker, match="Claude 全局熔断"):
        _runner(tmp_path, store, fake_run).execute(definition)

    assert store.attempts_reserved() == 1
    assert store.task_state(definition.task_spec.evaluation_key) == "exhausted"
    attempt = json.loads(
        (tmp_path / "llm" / "attempts" / f"{definition.task_spec.evaluation_key}.a01.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["validation"]["error_class"] == "contract_drift"
    assert attempt["metadata"]["retryable"] is False
    assert len(fake_run.calls) == 1
