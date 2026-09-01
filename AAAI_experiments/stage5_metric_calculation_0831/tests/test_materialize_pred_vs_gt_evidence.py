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

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (  # noqa: E402
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.materialize_pred_vs_gt_evidence import (  # noqa: E402
    MaterializePredVsGtEvidenceError,
    main,
    materialize_pred_vs_gt_evidence,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec  # noqa: E402


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    return _sha256_text(canonical_json(payload))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def _plan_assets(tmp_path: Path) -> tuple[Path, str, Path, str, dict[str, Any], str]:
    prompt_path = tmp_path / "contract" / "equivalence.v1.txt"
    schema_path = tmp_path / "contract" / "equivalence.v1.json"
    prompt_template = "Judge equivalence.\n{{REQUEST_JSON}}\n"
    schema_content = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string"},
            "evidence_basis": {"type": "string"},
        },
        "required": ["decision", "evidence_basis"],
    }
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(json.dumps(schema_content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (
        prompt_path,
        _sha256_file(prompt_path),
        schema_path,
        _sha256_file(schema_path),
        schema_content,
        prompt_template,
    )


def _pair_evidence_payload() -> dict[str, Any]:
    payload = {
        "schema_version": "symbolic_pair_evidence.v2",
        "phase": "equivalence",
        "pair_seed": 520,
        "pair_evidence": {
            "tree": {"tree_similarity": 1.0},
            "variable": {"f1": 1.0},
            "operator": {"f1": 1.0},
        },
        "lhs_binding": {
            "role": "lhs",
            "frozen_logical_id": "gt_simplify::BPG3",
            "plan_symbolic_artifact_sha256": _sha256_text("gt-artifact"),
            "frozen_simplified_expression": "P + t",
        },
        "rhs_binding": {
            "role": "rhs",
            "frozen_logical_id": "pred_simplify::qlattice::g0005::s520::clean",
            "plan_symbolic_artifact_sha256": _sha256_text("pred-artifact"),
            "frozen_simplified_expression": "P + t",
        },
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def _plan_row(
    tmp_path: Path,
    *,
    logical_id: str,
    request: dict[str, Any],
    priority: int = 30,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    (
        prompt_path,
        prompt_sha256,
        schema_path,
        schema_sha256,
        schema_content,
        prompt_template,
    ) = _plan_assets(tmp_path)
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_json(normalized_input)
    evaluation_key_value = evaluation_key(
        task_type="equivalence",
        logical_id=logical_id,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=request["evidence_hash"],
    )
    task_spec = TaskSpec(
        evaluation_key=evaluation_key_value,
        logical_id=logical_id,
        task_type="equivalence",
        condition="clean",
        priority=priority,
        input_hash=input_hash,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )
    row: dict[str, Any] = {
        "evaluation_key": evaluation_key_value,
        "logical_id": logical_id,
        "task_type": "equivalence",
        "task_kind": "equivalence",
        "condition": "clean",
        "priority": priority,
        "input_hash": input_hash,
        "prompt_version": "equivalence.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "equivalence.v1",
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "prompt_template": prompt_template,
        "schema_content": schema_content,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(task_spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request, schema_content),
    }
    if extra:
        row.update(extra)
    return row


def test_materialize_pred_vs_gt_evidence_accepts_full_plan_and_skips_non_applicable(
    tmp_path: Path,
) -> None:
    evidence = _pair_evidence_payload()
    callable_row = _plan_row(
        tmp_path,
        logical_id="equivalence::qlattice::g0005::s520::clean",
        request={
            "algorithm": "QLattice",
            "algorithm_slug": "qlattice",
            "dataset_id": "BPG3",
            "dataset_index": "g0005",
            "seed": 520,
            "ground_truth_logical_id": "gt_simplify::BPG3",
            "prediction_logical_id": "pred_simplify::qlattice::g0005::s520::clean",
            "deterministic_evidence": evidence,
            "evidence_hash": evidence["evidence_sha256"],
        },
    )
    no_call_row = _plan_row(
        tmp_path,
        logical_id="equivalence::qlattice::g0005::s521::clean",
        request={
            "algorithm": "QLattice",
            "algorithm_slug": "qlattice",
            "dataset_id": "BPG3",
            "dataset_index": "g0005",
            "seed": 521,
            "ground_truth_logical_id": "gt_simplify::BPG3",
            "prediction_logical_id": "pred_simplify::qlattice::g0005::s521::clean",
            "evidence_hash": _sha256_text("non-applicable"),
        },
        extra={
            "phase": "equivalence",
            "reason": "upstream_pred_unavailable",
            "status": "planned_non_applicable",
        },
    )
    plan_jsonl = tmp_path / "equivalence_full_plan.jsonl"
    output_jsonl = tmp_path / "clean_pred_vs_gt_evidence.jsonl"
    report_json = tmp_path / "report.json"
    _write_jsonl(plan_jsonl, [callable_row, no_call_row])

    report = materialize_pred_vs_gt_evidence(
        equivalence_plan_jsonl=plan_jsonl,
        output_jsonl=output_jsonl,
        report_json=report_json,
        expected_row_count=1,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["logical_key"] == "QLattice::BPG3::s520::clean"
    assert rows[0]["gt_logical_id"] == "gt_simplify::BPG3"
    assert rows[0]["pred_logical_id"] == "pred_simplify::qlattice::g0005::s520::clean"
    assert rows[0]["evidence_hash"] == evidence["evidence_sha256"]
    assert rows[0]["tree"]["tree_similarity"] == pytest.approx(1.0)
    assert rows[0]["variable"]["f1"] == pytest.approx(1.0)
    assert rows[0]["operator"]["f1"] == pytest.approx(1.0)
    assert report["counts"]["callable_row_count"] == 1
    assert report["counts"]["skipped_non_applicable_row_count"] == 1
    assert report["outputs"]["evidence_jsonl_row_count"] == 1


def test_materialize_pred_vs_gt_evidence_preserves_gt_v2_logical_id(tmp_path: Path) -> None:
    evidence = _pair_evidence_payload()
    evidence["lhs_binding"]["frozen_logical_id"] = "gt_simplify::BPG3::v2"
    evidence["evidence_sha256"] = _sha256_json(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    row = _plan_row(
        tmp_path,
        logical_id="equivalence::qlattice::g0005::s520::clean",
        request={
            "algorithm": "QLattice",
            "algorithm_slug": "qlattice",
            "dataset_id": "BPG3",
            "dataset_index": "g0005",
            "seed": 520,
            "ground_truth_logical_id": "gt_simplify::BPG3::v2",
            "prediction_logical_id": "pred_simplify::qlattice::g0005::s520::clean",
            "deterministic_evidence": evidence,
            "evidence_hash": evidence["evidence_sha256"],
        },
    )
    plan_jsonl = tmp_path / "equivalence_callable.jsonl"
    output_jsonl = tmp_path / "clean_pred_vs_gt_evidence.jsonl"
    report_json = tmp_path / "report.json"
    _write_jsonl(plan_jsonl, [row])

    materialize_pred_vs_gt_evidence(
        equivalence_plan_jsonl=plan_jsonl,
        output_jsonl=output_jsonl,
        report_json=report_json,
        expected_row_count=1,
    )

    materialized = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert materialized["gt_logical_id"] == "gt_simplify::BPG3::v2"
    assert materialized["logical_key"] == "QLattice::BPG3::s520::clean"


def test_cli_rejects_request_hash_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    evidence = _pair_evidence_payload()
    bad_row = _plan_row(
        tmp_path,
        logical_id="equivalence::qlattice::g0005::s520::clean",
        request={
            "algorithm": "QLattice",
            "algorithm_slug": "qlattice",
            "dataset_id": "BPG3",
            "dataset_index": "g0005",
            "seed": 520,
            "ground_truth_logical_id": "gt_simplify::BPG3",
            "prediction_logical_id": "pred_simplify::qlattice::g0005::s520::clean",
            "deterministic_evidence": evidence,
            "evidence_hash": _sha256_text("drifted"),
        },
    )
    plan_jsonl = tmp_path / "equivalence_callable.jsonl"
    _write_jsonl(plan_jsonl, [bad_row])

    exit_code = main(
        [
            "--equivalence-plan-jsonl",
            str(plan_jsonl),
            "--output-jsonl",
            str(tmp_path / "out.jsonl"),
            "--report-json",
            str(tmp_path / "report.json"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "request.evidence_hash" in captured.err


def test_rejects_non_integer_request_seed(tmp_path: Path) -> None:
    evidence = _pair_evidence_payload()
    row = _plan_row(
        tmp_path,
        logical_id="equivalence::qlattice::g0005::s520::clean",
        request={
            "algorithm": "QLattice",
            "algorithm_slug": "qlattice",
            "dataset_id": "BPG3",
            "dataset_index": "g0005",
            "seed": "520",
            "ground_truth_logical_id": "gt_simplify::BPG3",
            "prediction_logical_id": "pred_simplify::qlattice::g0005::s520::clean",
            "deterministic_evidence": evidence,
            "evidence_hash": evidence["evidence_sha256"],
        },
    )
    plan_jsonl = tmp_path / "equivalence_callable.jsonl"
    _write_jsonl(plan_jsonl, [row])

    with pytest.raises(MaterializePredVsGtEvidenceError, match="request.seed 必须是整数"):
        materialize_pred_vs_gt_evidence(
            equivalence_plan_jsonl=plan_jsonl,
            output_jsonl=tmp_path / "out.jsonl",
            report_json=tmp_path / "report.json",
        )
