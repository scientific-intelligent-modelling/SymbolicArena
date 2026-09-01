from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.materialize_recovered_attempts import (
    RecoveredAttemptAuditError,
    materialize_recovered_attempts,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prepare_expired_attempt(tmp_path: Path) -> tuple[Path, Path, str, str]:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "dataset_id": "demo",
        "expression": "x0 + 0",
        "original_expression": "x0 + 0",
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": "demo-evidence",
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    logical_id = "pred_simplify::demo::g0001::s520::clean"
    task_key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash="demo-evidence",
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash=input_hash,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    plan_row = {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "pred_simplify",
        "task_kind": "simplify",
        "condition": "clean",
        "priority": 20,
        "input_hash": input_hash,
        "prompt_version": "simplify.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "simplify.v1",
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request, schema),
    }
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(plan_row) + "\n", encoding="utf-8")

    state_db = tmp_path / "control/state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    lease = store.reserve_attempt(task_key, now=10.0, lease_seconds=5.0)
    assert store.recover_expired_leases(now=20.0) == [lease.attempt_id]
    return plan_path, state_db, task_key, lease.attempt_id


def test_materializes_truthful_lease_expired_artifact_idempotently(tmp_path: Path) -> None:
    plan_path, state_db, task_key, attempt_id = _prepare_expired_attempt(tmp_path)
    attempts_dir = tmp_path / "attempts"
    report_path = tmp_path / "reports/recovered.json"

    first = materialize_recovered_attempts(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempts_dir=attempts_dir,
        report_json=report_path,
    )
    assert first["eligible_count"] == 1
    assert first["created_count"] == 1
    assert first["reused_count"] == 0
    assert first["model_invoked"] is False
    assert first["attempt_count_delta"] == 0

    attempt_path = attempts_dir / f"{attempt_id}.json"
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert payload["evaluation_key"] == task_key
    assert payload["prompt"] is None
    assert payload["stdout"] is None
    assert payload["structured_output"] is None
    assert payload["validation"]["ok"] is False
    assert payload["validation"]["error_class"] == "lease_expired"
    assert payload["metadata"]["model_output_available"] is False
    assert payload["metadata"]["model_invocation_status"] == (
        "unknown_after_process_termination"
    )
    assert payload["metadata"]["recovery_event_details"] == {
        "next_state": "retry_wait"
    }

    second = materialize_recovered_attempts(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempts_dir=attempts_dir,
        report_json=report_path,
    )
    assert second["created_count"] == 0
    assert second["reused_count"] == 1


def test_rejects_conflicting_existing_recovery_artifact(tmp_path: Path) -> None:
    plan_path, state_db, _, attempt_id = _prepare_expired_attempt(tmp_path)
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()
    (attempts_dir / f"{attempt_id}.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        RecoveredAttemptAuditError,
        match="既有恢复 attempt 与当前状态证据冲突",
    ):
        materialize_recovered_attempts(
            plan_jsonl=plan_path,
            state_db=state_db,
            attempts_dir=attempts_dir,
            report_json=tmp_path / "report.json",
        )
