from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (  # noqa: E402
    ClaudeRunResult,
    ClaudeRunnerCircuitBreaker,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plan_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_predecessor_attempt_manifest(tmp_path: Path, *, attempt_count: int) -> Path:
    manifest_path = tmp_path / f"predecessor-{attempt_count}.json"
    source_state_db = tmp_path / "predecessor-state.sqlite3"
    source_attempts_dir = tmp_path / "predecessor-attempts"
    source_attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_ids = [f"legacy::{index:05d}" for index in range(attempt_count)]
    with sqlite3.connect(source_state_db) as connection:
        connection.execute("CREATE TABLE attempts(attempt_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO attempts(attempt_id) VALUES (?)",
            [(attempt_id,) for attempt_id in attempt_ids],
        )
    attempt_hashes: dict[str, str] = {}
    for attempt_id in attempt_ids:
        audit_path = source_attempts_dir / f"{attempt_id}.json"
        _write_json(audit_path, {"attempt_id": attempt_id})
        attempt_hashes[attempt_id] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    payload = {
        "attempt_count": attempt_count,
        "attempt_ids": attempt_ids,
        "attempt_file_sha256": attempt_hashes,
        "schema_version": "predecessor_attempts.v1",
        "source_attempts_dir": str(source_attempts_dir),
        "source_state_db": str(source_state_db),
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _task_spec(
    *,
    evaluation_key_value: str,
    logical_id: str,
    task_type: str = "equivalence",
    priority: int = 1,
) -> TaskSpec:
    return TaskSpec(
        evaluation_key=evaluation_key_value,
        logical_id=logical_id,
        task_type=task_type,
        condition="clean",
        priority=priority,
        input_hash="",
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )


def _build_plan_row(tmp_path: Path, logical_id: str, *, priority: int = 1) -> dict[str, Any]:
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = contract_dir / "equivalence.v1.txt"
    schema_path = contract_dir / "equivalence.v1.json"
    prompt_template = "Judge equivalence.\n{{REQUEST_JSON}}\n"
    schema_content = {
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
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(
        json.dumps(schema_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "lhs": "x0 + x1",
        "rhs": "x1 + x0",
        "evidence_hash": f"evidence::{logical_id}",
    }
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    task_key = evaluation_key(
        task_type="equivalence",
        logical_id=logical_id,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="equivalence",
        condition="clean",
        priority=priority,
        input_hash=input_hash,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )
    return {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "equivalence",
        "condition": "clean",
        "priority": priority,
        "input_hash": input_hash,
        "prompt_version": "equivalence.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "equivalence.v1",
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema_content,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request, schema_content),
    }


class FakeRunner:
    def __init__(
        self,
        *,
        store: TaskStateStore,
        attempts_dir: Path,
        frozen_dir: Path,
        behaviors: dict[str, str],
        calls: list[str],
        block_started: threading.Event | None = None,
        release_block: threading.Event | None = None,
    ) -> None:
        self.store = store
        self.attempts_dir = attempts_dir
        self.frozen_dir = frozen_dir
        self.behaviors = behaviors
        self.calls = calls
        self.lock = threading.Lock()
        self.block_started = block_started
        self.release_block = release_block

    def execute(self, definition: Any) -> ClaudeRunResult:
        logical_id = definition.task_spec.logical_id
        with self.lock:
            self.calls.append(logical_id)
        behavior = self.behaviors.get(logical_id, "success")
        lease = self.store.reserve_attempt(definition.task_spec.evaluation_key, lease_seconds=30.0)
        if behavior == "block_then_success":
            assert self.block_started is not None
            assert self.release_block is not None
            self.block_started.set()
            self.release_block.wait(timeout=5)
            if not self.release_block.is_set():
                raise AssertionError("测试阻塞任务没有被释放")
            return self._freeze_success(definition, lease.attempt_id, total_cost_usd=0.2)
        if behavior == "success":
            return self._freeze_success(definition, lease.attempt_id, total_cost_usd=0.1)
        if behavior == "retry_wait":
            next_state = self.store.finish_failure(
                lease.attempt_id,
                error_class="outer_json_invalid",
                retryable=True,
            )
            return ClaudeRunResult(
                evaluation_key=definition.task_spec.evaluation_key,
                state=next_state,
                attempt_id=lease.attempt_id,
                result_path=None,
                result_sha256=None,
                structured_output=None,
                from_cache=False,
                error_class="outer_json_invalid",
                usage=None,
                total_cost_usd=None,
                claude_version="fake-clause",
            )
        if behavior == "circuit_breaker":
            if self.release_block is not None:
                self.release_block.set()
            self.store.finish_failure(
                lease.attempt_id,
                error_class="contract_drift",
                retryable=True,
            )
            raise ClaudeRunnerCircuitBreaker(
                "fake breaker",
                evaluation_key=definition.task_spec.evaluation_key,
                attempt_id=lease.attempt_id,
            )
        raise AssertionError(f"未知 fake 行为: {behavior}")

    def _freeze_success(self, definition: Any, attempt_id: str, *, total_cost_usd: float) -> ClaudeRunResult:
        payload = {
            "attempt_id": attempt_id,
            "evaluation_key": definition.task_spec.evaluation_key,
            "logical_id": definition.task_spec.logical_id,
            "task_type": definition.task_spec.task_type,
            "structured_output": {
                "decision": "equivalent",
                "evidence_basis": "symbolic_proof",
                "assumptions": [],
                "confidence": 1.0,
                "brief_reason": "fake runner",
            },
            "metadata": {
                "total_cost_usd": total_cost_usd,
                "claude_version": "fake-claude 1.0",
            },
        }
        frozen_path = self.frozen_dir / f"{definition.task_spec.evaluation_key}.json"
        _write_json(frozen_path, payload)
        frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        self.store.freeze_result(
            attempt_id,
            result_path=str(frozen_path),
            result_sha256=frozen_sha,
        )
        return ClaudeRunResult(
            evaluation_key=definition.task_spec.evaluation_key,
            state="frozen",
            attempt_id=attempt_id,
            result_path=str(frozen_path),
            result_sha256=frozen_sha,
            structured_output=dict(payload["structured_output"]),
            from_cache=False,
            error_class=None,
            usage=None,
            total_cost_usd=total_cost_usd,
            claude_version="fake-claude 1.0",
        )


def _runner_factory(
    *,
    behaviors: dict[str, str],
    calls: list[str],
    block_started: threading.Event | None = None,
    release_block: threading.Event | None = None,
):
    def factory(store: TaskStateStore, *, attempts_dir: Path, frozen_dir: Path) -> FakeRunner:
        return FakeRunner(
            store=store,
            attempts_dir=attempts_dir,
            frozen_dir=frozen_dir,
            behaviors=behaviors,
            calls=calls,
            block_started=block_started,
            release_block=release_block,
        )

    return factory


def test_registers_all_tasks_but_limit_only_caps_this_run(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    rows = [
        _build_plan_row(tmp_path, "equivalence::task-1", priority=1),
        _build_plan_row(tmp_path, "equivalence::task-2", priority=2),
        _build_plan_row(tmp_path, "equivalence::task-3", priority=3),
    ]
    plan_path = tmp_path / "plan.jsonl"
    expected_plan_sha = _write_plan_jsonl(plan_path, rows)
    state_db = tmp_path / "control" / "state.sqlite3"
    report_json = tmp_path / "reports" / "progress.json"
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
            "--limit",
            "1",
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert exit_code == 0
    assert calls == ["equivalence::task-1"]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["plan_sha256"] == expected_plan_sha
    assert report["registered_task_count"] == 3
    assert report["selected_task_count"] == 3
    assert report["submitted_task_count"] == 1
    assert report["result_counts"]["success"] == 1
    assert report["result_counts"]["cache"] == 0
    assert report["state_distribution"] == {"frozen": 1, "pending": 2}
    store = TaskStateStore(state_db)
    assert store.task_state(rows[0]["evaluation_key"]) == "frozen"
    assert store.task_state(rows[1]["evaluation_key"]) == "pending"
    assert store.task_state(rows[2]["evaluation_key"]) == "pending"


def test_cached_resume_skips_runner_and_counts_cache(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    rows = [
        _build_plan_row(tmp_path, "equivalence::cached", priority=1),
        _build_plan_row(tmp_path, "equivalence::pending", priority=2),
    ]
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, rows)
    state_db = tmp_path / "control" / "state.sqlite3"
    frozen_dir = tmp_path / "llm" / "frozen"
    report_json = tmp_path / "reports" / "progress.json"
    store = TaskStateStore(state_db)
    cached_spec = TaskSpec(
        evaluation_key=rows[0]["evaluation_key"],
        logical_id=rows[0]["logical_id"],
        task_type=rows[0]["task_type"],
        condition=rows[0]["condition"],
        priority=rows[0]["priority"],
        input_hash=rows[0]["input_hash"],
        prompt_version=rows[0]["prompt_version"],
        schema_version=rows[0]["schema_version"],
        dependencies=(),
    )
    store.register_task(cached_spec)
    lease = store.reserve_attempt(cached_spec.evaluation_key, lease_seconds=30.0)
    frozen_path = frozen_dir / f"{cached_spec.evaluation_key}.json"
    _write_json(
        frozen_path,
        {
            "attempt_id": lease.attempt_id,
            "evaluation_key": cached_spec.evaluation_key,
            "logical_id": cached_spec.logical_id,
            "task_type": cached_spec.task_type,
            "structured_output": {"decision": "equivalent"},
            "metadata": {"total_cost_usd": 0.0, "claude_version": "cached"},
        },
    )
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path),
        result_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    )
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(frozen_dir),
            "--report-json",
            str(report_json),
        ],
        runner_factory=_runner_factory(
            behaviors={"equivalence::pending": "success"},
            calls=calls,
        ),
    )

    assert exit_code == 0
    assert calls == ["equivalence::pending"]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["result_counts"]["cache"] == 1
    assert report["result_counts"]["success"] == 1
    assert report["total_cost_usd"] == pytest.approx(0.1)
    assert report["state_distribution"] == {"frozen": 2}
    assert report["attempts_reserved_total"] == 2


def test_circuit_breaker_stops_new_submissions_but_allows_inflight_to_finish(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    rows = [
        _build_plan_row(tmp_path, "equivalence::slow", priority=1),
        _build_plan_row(tmp_path, "equivalence::boom", priority=1),
        _build_plan_row(tmp_path, "equivalence::never", priority=1),
    ]
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, rows)
    state_db = tmp_path / "control" / "state.sqlite3"
    report_json = tmp_path / "reports" / "progress.json"
    calls: list[str] = []
    block_started = threading.Event()
    release_block = threading.Event()

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
            "--workers",
            "2",
        ],
        runner_factory=_runner_factory(
            behaviors={
                "equivalence::slow": "block_then_success",
                "equivalence::boom": "circuit_breaker",
                "equivalence::never": "success",
            },
            calls=calls,
            block_started=block_started,
            release_block=release_block,
        ),
    )

    assert block_started.is_set()
    assert exit_code == 1
    assert calls.count("equivalence::never") == 0
    assert set(calls) == {"equivalence::slow", "equivalence::boom"}
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["stopped_by_circuit_breaker"] is True
    assert report["result_counts"]["circuit_breaker"] == 1
    assert report["result_counts"]["success"] == 1
    assert report["submitted_task_count"] == 2


def test_workers_must_be_positive(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::task-1")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])

    with pytest.raises(SystemExit):
        main(
            [
                "--plan-jsonl",
                str(plan_path),
                "--state-db",
                str(tmp_path / "control" / "state.sqlite3"),
                "--attempts-dir",
                str(tmp_path / "llm" / "attempts"),
                "--frozen-dir",
                str(tmp_path / "llm" / "frozen"),
                "--report-json",
                str(tmp_path / "reports" / "progress.json"),
                "--workers",
                "0",
            ],
            runner_factory=_runner_factory(behaviors={}, calls=[]),
        )


def test_plan_definition_drift_fails_before_runner_invocation(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::drift")
    row["rendered_prompt"] = "tampered"
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    report_json = tmp_path / "reports" / "progress.json"
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(tmp_path / "control" / "state.sqlite3"),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert exit_code == 2
    assert calls == []
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "plan_contract_error"
    assert "rendered_prompt" in report["error"]


@pytest.mark.parametrize("duplicate_field", ["logical_id", "evaluation_key"])
def test_plan_rejects_duplicate_identity_fields(tmp_path: Path, duplicate_field: str) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    first = _build_plan_row(tmp_path, "equivalence::first")
    second = _build_plan_row(tmp_path, "equivalence::second")
    if duplicate_field == "logical_id":
        second["logical_id"] = first["logical_id"]
        second["task_spec"]["logical_id"] = first["logical_id"]
        second_key = evaluation_key(
            task_type=second["task_type"],
            logical_id=second["logical_id"],
            prompt_version=second["prompt_version"],
            schema_version=second["schema_version"],
            prompt_sha256=second["prompt_sha256"],
            schema_sha256=second["schema_sha256"],
            normalized_input=second["normalized_input"],
            evidence_hash=second["request"]["evidence_hash"],
        )
        second["evaluation_key"] = second_key
        second["task_spec"]["evaluation_key"] = second_key
    else:
        second = json.loads(canonical_json(first))
    plan_path = tmp_path / "duplicate.jsonl"
    _write_plan_jsonl(plan_path, [first, second])
    report_json = tmp_path / "reports" / "duplicate.json"

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(tmp_path / "control" / "state.sqlite3"),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=[]),
    )

    assert exit_code == 2
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "plan_contract_error"
    assert duplicate_field in report["error"]


def test_startup_recovers_expired_lease_before_selecting_runnable_tasks(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::expired")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "control" / "state.sqlite3"
    report_json = tmp_path / "reports" / "progress.json"
    store = TaskStateStore(state_db)
    spec = TaskSpec(**json.loads(canonical_json(row["task_spec"])))
    store.register_task(spec, now=0.0)
    expired = store.reserve_attempt(spec.evaluation_key, now=1.0, lease_seconds=1.0)
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert exit_code == 0
    assert calls == ["equivalence::expired"]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["recovered_expired_attempt_ids"] == [expired.attempt_id]
    assert report["recovered_expired_attempt_count"] == 1


def test_cached_result_sha_drift_fails_before_runner_invocation(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::cached-drift")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "control" / "state.sqlite3"
    frozen_dir = tmp_path / "llm" / "frozen"
    report_json = tmp_path / "reports" / "progress.json"
    store = TaskStateStore(state_db)
    spec = TaskSpec(**json.loads(canonical_json(row["task_spec"])))
    store.register_task(spec)
    lease = store.reserve_attempt(spec.evaluation_key, lease_seconds=30.0)
    frozen_path = frozen_dir / f"{spec.evaluation_key}.json"
    _write_json(
        frozen_path,
        {
            "attempt_id": lease.attempt_id,
            "evaluation_key": spec.evaluation_key,
            "logical_id": spec.logical_id,
            "task_type": spec.task_type,
            "structured_output": {"decision": "equivalent"},
        },
    )
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path),
        result_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    )
    frozen_path.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(frozen_dir),
            "--report-json",
            str(report_json),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert exit_code == 2
    assert calls == []
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "plan_contract_error"
    assert "SHA256" in report["error"]


def test_state_contract_error_is_reported_before_runner_invocation(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::legacy-offset")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    report_json = tmp_path / "reports" / "progress.json"
    calls: list[str] = []

    exit_code = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(tmp_path / "control" / "state.sqlite3"),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(tmp_path / "llm" / "frozen"),
            "--report-json",
            str(report_json),
            "--physical-attempt-offset",
            "11",
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert exit_code == 2
    assert calls == []
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["status"] == "state_contract_error"
    assert report["model_invoked"] is False
    assert "predecessor_attempt_manifest" in report["error"]


def test_predecessor_manifest_resume_keeps_same_db_budget_without_recount(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import main

    row = _build_plan_row(tmp_path, "equivalence::migrated")
    plan_path = tmp_path / "plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    manifest_path = _write_predecessor_attempt_manifest(tmp_path, attempt_count=11)
    state_db = tmp_path / "control" / "state.sqlite3"
    frozen_dir = tmp_path / "llm" / "frozen"
    report_json = tmp_path / "reports" / "progress.json"
    calls: list[str] = []

    first_exit = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(frozen_dir),
            "--report-json",
            str(report_json),
            "--predecessor-attempt-manifest",
            str(manifest_path),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert first_exit == 0
    first_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert first_report["attempts_reserved_total"] == 12
    assert first_report["physical_attempt_offset"] == 11
    assert first_report["predecessor_attempt_count"] == 11

    second_exit = main(
        [
            "--plan-jsonl",
            str(plan_path),
            "--state-db",
            str(state_db),
            "--attempts-dir",
            str(tmp_path / "llm" / "attempts"),
            "--frozen-dir",
            str(frozen_dir),
            "--report-json",
            str(report_json),
            "--predecessor-attempt-manifest",
            str(manifest_path),
        ],
        runner_factory=_runner_factory(behaviors={}, calls=calls),
    )

    assert second_exit == 0
    assert calls == ["equivalence::migrated"]
    second_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert second_report["attempts_reserved_total"] == 12
    assert second_report["result_counts"]["cache"] == 1
