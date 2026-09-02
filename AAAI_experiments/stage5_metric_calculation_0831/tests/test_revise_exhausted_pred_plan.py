from __future__ import annotations

import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.revise_exhausted_pred_plan import (
    revise_exhausted_pred_plan,
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


def _plan_row(logical_id: str) -> dict[str, object]:
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
        condition="clean",
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
        "condition": "clean",
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


def test_revises_only_exhausted_tasks_with_new_prompt_fingerprint(tmp_path: Path) -> None:
    frozen_row = _plan_row("pred_simplify::demo::g0001::s520::clean")
    exhausted_row = _plan_row("pred_simplify::demo::g0001::s521::clean")
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
        logical_id_suffix="v2",
        expected_task_count=2,
        expected_exhausted_count=1,
    )

    loaded = load_plan_jsonl(output_plan)
    by_logical_id = {entry.logical_id: entry for entry in loaded.entries}
    assert frozen_row["logical_id"] in by_logical_id
    successor_id = f"{exhausted_row['logical_id']}::v2"
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
