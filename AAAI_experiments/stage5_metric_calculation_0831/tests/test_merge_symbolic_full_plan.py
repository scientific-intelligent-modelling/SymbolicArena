from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.merge_symbolic_full_plan import (
    MergeSymbolicFullPlanError,
    merge_symbolic_full_plan,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _callable_row(
    tmp_path: Path, logical_id: str, *, priority: int
) -> dict[str, Any]:
    prompt_path = tmp_path / f"{logical_id}.prompt.txt"
    schema_path = tmp_path / f"{logical_id}.schema.json"
    prompt_template = "Judge this request: {{REQUEST_JSON}}"
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(canonical_json(schema), encoding="utf-8")
    prompt_sha256 = _sha256_text(prompt_template)
    schema_sha256 = _sha256_text(canonical_json(schema))
    request = {"expression": logical_id, "evidence_hash": "a" * 64}
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    spec = TaskSpec(
        evaluation_key=key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition="noise005",
        priority=priority,
        input_hash=_sha256_text(canonical_json(normalized_input)),
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    return {
        "logical_id": logical_id,
        "evaluation_key": key,
        "task_type": "pred_simplify",
        "condition": "noise005",
        "priority": priority,
        "input_hash": spec.input_hash,
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


def _non_applicable_row(
    tmp_path: Path,
    logical_id: str,
    *,
    evaluation_key_value: str | None = None,
    priority: int = 20,
) -> dict[str, Any]:
    row = _callable_row(tmp_path, logical_id, priority=priority)
    if evaluation_key_value is not None:
        row["evaluation_key"] = evaluation_key_value
        row["task_spec"]["evaluation_key"] = evaluation_key_value
    row["status"] = "planned_non_applicable"
    return row


def test_merge_symbolic_full_plan_writes_sorted_canonical_output_and_report(
    tmp_path: Path,
) -> None:
    callable_path = tmp_path / "callable.jsonl"
    non_applicable_path = tmp_path / "non_applicable.jsonl"
    output_path = tmp_path / "full.jsonl"
    report_path = tmp_path / "report.json"
    callable_rows = [
        _callable_row(tmp_path, "pred_simplify::z", priority=30),
        _callable_row(tmp_path, "pred_simplify::a", priority=10),
    ]
    non_applicable_rows = [
        _non_applicable_row(tmp_path, "pred_simplify::m", priority=20)
    ]
    _write_jsonl(callable_path, callable_rows)
    _write_jsonl(non_applicable_path, non_applicable_rows)

    report = merge_symbolic_full_plan(
        callable_plan_jsonl=callable_path,
        non_applicable_index_jsonl=non_applicable_path,
        output_jsonl=output_path,
        report_json=report_path,
        expected_total_count=3,
        expected_callable_count=2,
        expected_non_applicable_count=1,
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_rows = [json.loads(line) for line in output_lines]
    assert [row["logical_id"] for row in output_rows] == [
        "pred_simplify::a",
        "pred_simplify::m",
        "pred_simplify::z",
    ]
    assert output_lines == [canonical_json(row) for row in output_rows]
    assert len(load_plan_jsonl(output_path).entries) == 3
    assert report["status"] == "ok"
    assert report["model_invoked"] is False
    assert report["counts"] == {
        "callable_count": 2,
        "non_applicable_count": 1,
        "total_count": 3,
    }
    assert report["inputs"]["callable_plan_jsonl"] == str(callable_path.resolve())
    assert report["inputs"]["non_applicable_index_jsonl"] == str(
        non_applicable_path.resolve()
    )
    assert report["outputs"]["full_plan_jsonl"] == str(output_path.resolve())
    assert report["outputs"]["full_plan_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("overlap_field", ["evaluation_key", "logical_id"])
def test_merge_symbolic_full_plan_rejects_cross_side_identity_overlap(
    tmp_path: Path,
    overlap_field: str,
) -> None:
    callable_path = tmp_path / "callable.jsonl"
    non_applicable_path = tmp_path / "non_applicable.jsonl"
    callable_row = _callable_row(tmp_path, "pred_simplify::call", priority=10)
    no_call = _non_applicable_row(tmp_path, "pred_simplify::skip", priority=20)
    if overlap_field == "evaluation_key":
        no_call["evaluation_key"] = callable_row["evaluation_key"]
        no_call["task_spec"]["evaluation_key"] = callable_row["evaluation_key"]
    else:
        no_call["logical_id"] = callable_row["logical_id"]
        no_call["task_spec"]["logical_id"] = callable_row["logical_id"]
    _write_jsonl(callable_path, [callable_row])
    _write_jsonl(non_applicable_path, [no_call])

    with pytest.raises(MergeSymbolicFullPlanError, match="重叠"):
        merge_symbolic_full_plan(
            callable_plan_jsonl=callable_path,
            non_applicable_index_jsonl=non_applicable_path,
            output_jsonl=tmp_path / "full.jsonl",
            report_json=tmp_path / "report.json",
        )


def test_merge_symbolic_full_plan_rejects_expected_count_drift(tmp_path: Path) -> None:
    callable_path = tmp_path / "callable.jsonl"
    non_applicable_path = tmp_path / "non_applicable.jsonl"
    _write_jsonl(
        callable_path,
        [_callable_row(tmp_path, "pred_simplify::call", priority=10)],
    )
    _write_jsonl(
        non_applicable_path,
        [_non_applicable_row(tmp_path, "pred_simplify::skip")],
    )

    with pytest.raises(MergeSymbolicFullPlanError, match="callable.*期望 2.*实际 1"):
        merge_symbolic_full_plan(
            callable_plan_jsonl=callable_path,
            non_applicable_index_jsonl=non_applicable_path,
            output_jsonl=tmp_path / "full.jsonl",
            report_json=tmp_path / "report.json",
            expected_callable_count=2,
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda row: row.update(status="pending"), "status"),
        (lambda row: row.pop("task_spec"), "task_spec"),
        (lambda row: row.update(priority="20"), "priority"),
        (
            lambda row: row["task_spec"].update(logical_id="pred_simplify::drift"),
            "task_spec.logical_id",
        ),
    ],
)
def test_merge_symbolic_full_plan_rejects_invalid_non_applicable_row(
    tmp_path: Path,
    mutation: Any,
    error_match: str,
) -> None:
    callable_path = tmp_path / "callable.jsonl"
    non_applicable_path = tmp_path / "non_applicable.jsonl"
    _write_jsonl(
        callable_path,
        [_callable_row(tmp_path, "pred_simplify::call", priority=10)],
    )
    row = _non_applicable_row(tmp_path, "pred_simplify::skip")
    mutation(row)
    _write_jsonl(non_applicable_path, [row])

    with pytest.raises(MergeSymbolicFullPlanError, match=error_match):
        merge_symbolic_full_plan(
            callable_plan_jsonl=callable_path,
            non_applicable_index_jsonl=non_applicable_path,
            output_jsonl=tmp_path / "full.jsonl",
            report_json=tmp_path / "report.json",
        )
