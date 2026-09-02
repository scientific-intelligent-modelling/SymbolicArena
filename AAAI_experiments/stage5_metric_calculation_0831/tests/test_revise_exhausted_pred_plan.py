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
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.revise_exhausted_pred_plan import (
    revise_exhausted_pred_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.repair_audited_pred_simplifications import (
    repair_audited_pred_simplifications,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plan_row(logical_id: str, *, condition: str = "clean") -> dict[str, object]:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256_bytes(prompt_path.read_bytes())
    schema_sha256 = _sha256_bytes(schema_path.read_bytes())
    request = {
        "dataset_id": logical_id,
        "expression": "x0 + 0",
        "original_expression": "x0 + 0",
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": hashlib.sha256(logical_id.encode()).hexdigest(),
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = hashlib.sha256(canonical_json(normalized_input).encode()).hexdigest()
    task_key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition=condition,
        priority=20,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "pred_simplify",
        "task_kind": "simplify",
        "condition": condition,
        "priority": 20,
        "input_hash": input_hash,
        "prompt_version": prompt_path.stem,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_path.stem,
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


@pytest.mark.parametrize(
    ("predecessor_suffix", "successor_suffix"),
    [(None, "v2"), ("v2", "v3")],
)
def test_revises_only_exhausted_tasks_with_new_prompt_fingerprint(
    tmp_path: Path,
    predecessor_suffix: str | None,
    successor_suffix: str,
) -> None:
    frozen_row = _plan_row("pred_simplify::demo::g0001::s520::clean")
    exhausted_base_id = "pred_simplify::demo::g0001::s521::clean"
    exhausted_row = _plan_row(
        exhausted_base_id
        if predecessor_suffix is None
        else f"{exhausted_base_id}::{predecessor_suffix}"
    )
    predecessor_plan = tmp_path / "predecessor.jsonl"
    predecessor_plan.write_text(
        canonical_json(frozen_row) + "\n" + canonical_json(exhausted_row) + "\n",
        encoding="utf-8",
    )

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    for row in (frozen_row, exhausted_row):
        store.register_task(TaskSpec(**row["task_spec"]))
    frozen_lease = store.reserve_attempt(str(frozen_row["evaluation_key"]), now=1.0)
    store.freeze_result(
        frozen_lease.attempt_id,
        result_path="/tmp/frozen.json",
        result_sha256="a" * 64,
        now=2.0,
    )
    for number in range(3):
        lease = store.reserve_attempt(str(exhausted_row["evaluation_key"]), now=3.0 + number)
        store.finish_failure(
            lease.attempt_id,
            error_class="validation_failed",
            retryable=True,
            now=3.5 + number,
        )

    recovery_prompt = tmp_path / "simplify.recovery.v1.txt"
    recovery_prompt.write_text(
        "Return one conservative JSON result.\n{{REQUEST_JSON}}\n",
        encoding="utf-8",
    )
    output_plan = tmp_path / "successor.jsonl"
    report_path = tmp_path / "report.json"
    report = revise_exhausted_pred_plan(
        predecessor_plan_jsonl=predecessor_plan,
        state_db=state_db,
        recovery_prompt_path=recovery_prompt,
        output_jsonl=output_plan,
        report_json=report_path,
        logical_id_suffix=successor_suffix,
        expected_task_count=2,
        expected_exhausted_count=1,
    )

    loaded = load_plan_jsonl(output_plan)
    by_logical_id = {entry.logical_id: entry for entry in loaded.entries}
    assert frozen_row["logical_id"] in by_logical_id
    successor_id = f"{exhausted_base_id}::{successor_suffix}"
    assert successor_id in by_logical_id
    successor = by_logical_id[successor_id]
    assert successor.evaluation_key != exhausted_row["evaluation_key"]
    assert successor.definition.request == exhausted_row["request"]
    assert successor.definition.task_spec.prompt_version == "simplify.recovery.v1"
    assert report["state_db_mutated"] is False
    assert report["model_invoked"] is False
    assert report["preserved_frozen_count"] == 1
    assert report["successor_task_count"] == 1
    assert report["output_task_count"] == 2
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_revises_noise_exhausted_tasks_without_condition_drift(tmp_path: Path) -> None:
    logical_id = "pred_simplify::demo::g0001::s520::noise001"
    exhausted_row = _plan_row(logical_id, condition="noise001")
    predecessor_plan = tmp_path / "predecessor.jsonl"
    predecessor_plan.write_text(canonical_json(exhausted_row) + "\n", encoding="utf-8")

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(TaskSpec(**exhausted_row["task_spec"]))
    for number in range(3):
        lease = store.reserve_attempt(str(exhausted_row["evaluation_key"]), now=1.0 + number)
        store.finish_failure(
            lease.attempt_id,
            error_class="validation_failed",
            retryable=True,
            now=1.5 + number,
        )

    recovery_prompt = tmp_path / "simplify.recovery.v1.txt"
    recovery_prompt.write_text("Return JSON.\n{{REQUEST_JSON}}\n", encoding="utf-8")
    output_plan = tmp_path / "successor.jsonl"
    report = revise_exhausted_pred_plan(
        predecessor_plan_jsonl=predecessor_plan,
        state_db=state_db,
        recovery_prompt_path=recovery_prompt,
        output_jsonl=output_plan,
        report_json=tmp_path / "report.json",
        condition="noise001",
        logical_id_suffix="v2",
        expected_task_count=1,
        expected_exhausted_count=1,
    )

    successor = load_plan_jsonl(output_plan).entries[0]
    assert successor.logical_id == f"{logical_id}::v2"
    assert successor.definition.task_spec.condition == "noise001"
    assert report["condition"] == "noise001"


def test_replaces_only_frozen_audit_failures_and_registers_successors(tmp_path: Path) -> None:
    passed_row = _plan_row("pred_simplify::demo::g0001::s520::clean")
    failed_row = _plan_row("pred_simplify::demo::g0001::s521::clean")
    predecessor_plan = tmp_path / "predecessor.jsonl"
    predecessor_plan.write_text(
        canonical_json(passed_row) + "\n" + canonical_json(failed_row) + "\n",
        encoding="utf-8",
    )
    predecessor_sha = hashlib.sha256(predecessor_plan.read_bytes()).hexdigest()

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    for index, row in enumerate((passed_row, failed_row), start=1):
        store.register_task(TaskSpec(**row["task_spec"]))
        lease = store.reserve_attempt(str(row["evaluation_key"]), now=float(index))
        frozen_path = tmp_path / f"frozen-{index}.json"
        frozen_path.write_text("{}\n", encoding="utf-8")
        store.freeze_result(
            lease.attempt_id,
            result_path=str(frozen_path),
            result_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            now=float(index) + 0.5,
        )

    audit_path = tmp_path / "audit.jsonl"
    audit_rows = [
        {"logical_id": passed_row["logical_id"], "status": "passed"},
        {
            "logical_id": failed_row["logical_id"],
            "status": "failed",
            "failure_class": "semantic_rejected",
        },
    ]
    audit_path.write_text(
        "".join(canonical_json(row) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    audit_report = tmp_path / "audit-report.json"
    audit_report.write_text(
        json.dumps(
            {
                "plan_sha256": predecessor_sha,
                "output_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "failed_count": 1,
            }
        ),
        encoding="utf-8",
    )

    output_plan = tmp_path / "successor.jsonl"
    report = repair_audited_pred_simplifications(
        predecessor_plan_jsonl=predecessor_plan,
        audit_jsonl=audit_path,
        audit_report_json=audit_report,
        state_db=state_db,
        output_plan_jsonl=output_plan,
        backup_state_db=tmp_path / "state.before_repair.sqlite3",
        report_json=tmp_path / "repair-report.json",
        logical_id_suffix="v2",
        expected_plan_count=2,
        expected_failed_count=1,
        now=10.0,
    )

    loaded = load_plan_jsonl(output_plan)
    logical_ids = {entry.logical_id for entry in loaded.entries}
    assert passed_row["logical_id"] in logical_ids
    successor_id = f"{failed_row['logical_id']}::v2"
    assert successor_id in logical_ids
    assert store.task_state(str(failed_row["evaluation_key"])) == "superseded"
    successor = next(entry for entry in loaded.entries if entry.logical_id == successor_id)
    assert store.task_state(successor.evaluation_key) == "pending"
    assert report["failed_count"] == 1
    assert report["backup"]["integrity_check"] == "ok"
