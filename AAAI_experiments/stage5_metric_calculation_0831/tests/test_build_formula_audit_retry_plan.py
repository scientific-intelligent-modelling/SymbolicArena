from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_formula_audit_retry_plan import (  # noqa: E402
    FormulaAuditRetryPlanError,
    build_formula_audit_retry_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (  # noqa: E402
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (  # noqa: E402
    TaskSpec,
    TaskStateStore,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _plan_row(*, logical_id: str, prompt_path: Path, schema_path: Path) -> dict[str, Any]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256(prompt_path.read_bytes())
    schema_sha256 = _sha256(schema_path.read_bytes())
    request_without_hash = {
        "audit_scope": "prediction_formula",
        "audit_binding_sha256": _sha256(logical_id.encode()),
        "variables": ["x0"],
        "allowed_functions": [],
        "domain_assumptions": {"variable_domain": "real"},
        "original_expression": "x0 + 0",
        "candidate_simplified_expression": "x0",
        "reference_simplified_expression": "x0",
        "review_round": 2,
    }
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256(canonical_json(request_without_hash).encode())
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256(canonical_json(normalized_input).encode())
    key = evaluation_key(
        task_type="formula_audit",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="formula_audit",
        condition="clean",
        priority=50,
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_path.stem,
        dependencies=(),
    )
    return {
        "evaluation_key": key,
        "logical_id": logical_id,
        "task_type": "formula_audit",
        "task_kind": "formula_audit",
        "condition": "clean",
        "priority": 50,
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
    }


def _spec(row: dict[str, Any]) -> TaskSpec:
    payload = row["task_spec"]
    return TaskSpec(
        evaluation_key=payload["evaluation_key"],
        logical_id=payload["logical_id"],
        task_type=payload["task_type"],
        condition=payload["condition"],
        priority=payload["priority"],
        input_hash=payload["input_hash"],
        prompt_version=payload["prompt_version"],
        schema_version=payload["schema_version"],
        dependencies=tuple(payload["dependencies"]),
    )


def _terminal_state(store: TaskStateStore, row: dict[str, Any], *, frozen: bool, now: float) -> None:
    lease = store.reserve_attempt(row["evaluation_key"], now=now)
    if not frozen:
        store.finish_failure(
            lease.attempt_id,
            error_class="structured_output_invalid",
            retryable=False,
            now=now + 0.01,
        )
        return
    result = Path(store.path).parent / f"{row['evaluation_key']}.json"
    result.write_text("{}\n", encoding="utf-8")
    store.freeze_result(
        lease.attempt_id,
        result_path=str(result),
        result_sha256=_sha256(result.read_bytes()),
        now=now + 0.01,
    )


def test_retry_plan_selects_only_exhausted_and_keeps_blind_semantic_request(
    tmp_path: Path,
) -> None:
    config = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    prompt = config / "prompts/formula_audit.v1.txt"
    schema = config / "schemas/formula_audit.v1.json"
    rows = [
        _plan_row(
            logical_id=f"formula_audit::prediction::alg::g000{index}::s520::clean::v2",
            prompt_path=prompt,
            schema_path=schema,
        )
        for index in range(1, 4)
    ]
    plan_path = tmp_path / "round2.jsonl"
    _write_jsonl(plan_path, rows)
    state_path = tmp_path / "round2.sqlite"
    store = TaskStateStore(
        state_path, attempt_cap=10, logical_task_cap=10, max_attempts_per_task=1
    )
    store.register_tasks([_spec(row) for row in rows], now=1.0)
    _terminal_state(store, rows[0], frozen=True, now=2.0)
    _terminal_state(store, rows[1], frozen=False, now=3.0)
    _terminal_state(store, rows[2], frozen=False, now=4.0)

    report = build_formula_audit_retry_plan(
        round2_plan_jsonl=plan_path,
        round2_state_db=state_path,
        output_jsonl=tmp_path / "retry.jsonl",
        report_json=tmp_path / "retry_report.json",
    )

    retry_rows = [
        json.loads(line)
        for line in Path(report["outputs"]["retry_plan_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["counts"]["round2_task_count"] == 3
    assert report["counts"]["retry_task_count"] == 2
    assert report["execution_contract"]["max_attempts_per_task"] == 1
    assert len(load_plan_jsonl(report["outputs"]["retry_plan_jsonl"]).entries) == 2
    predecessor = {row["evaluation_key"]: row for row in rows}
    for retry in retry_rows:
        original = predecessor[retry["retry_predecessor_evaluation_key"]]
        assert retry["logical_id"] == f"{original['logical_id']}::retry1"
        assert retry["request"] == original["request"]
        assert retry["request"]["review_round"] == 2
        assert retry["request"]["evidence_hash"] == original["request"]["evidence_hash"]
        assert retry["prompt_version"] == "formula_audit.v3"
        assert retry["prompt_path"].endswith("formula_audit.v3.txt")


def test_retry_plan_rejects_state_task_set_drift(tmp_path: Path) -> None:
    config = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/config"
    row = _plan_row(
        logical_id="formula_audit::prediction::alg::g0001::s520::clean::v2",
        prompt_path=config / "prompts/formula_audit.v1.txt",
        schema_path=config / "schemas/formula_audit.v1.json",
    )
    plan_path = tmp_path / "round2.jsonl"
    _write_jsonl(plan_path, [row])
    state_path = tmp_path / "round2.sqlite"
    store = TaskStateStore(
        state_path, attempt_cap=10, logical_task_cap=10, max_attempts_per_task=1
    )
    store.register_task(_spec(row), now=1.0)
    extra = TaskSpec(
        evaluation_key="e" * 64,
        logical_id="formula_audit::prediction::extra::g0001::s520::clean::v2",
        task_type="formula_audit",
        condition="clean",
        priority=50,
        input_hash="extra",
        prompt_version="formula_audit.v1",
        schema_version="formula_audit.v1",
        dependencies=(),
    )
    store.register_task(extra, now=1.1)

    with pytest.raises(FormulaAuditRetryPlanError, match="任务集合"):
        build_formula_audit_retry_plan(
            round2_plan_jsonl=plan_path,
            round2_state_db=state_path,
            output_jsonl=tmp_path / "retry.jsonl",
            report_json=tmp_path / "retry_report.json",
        )
