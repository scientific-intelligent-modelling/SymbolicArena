from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_clean_metrics import (
    AggregateCleanMetricsError,
    aggregate_clean_metrics,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    efficiency_from_qualities,
    minimality_score,
    phi_nmse,
    symbolic_fidelity_score,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)


def _fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_json(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _trajectory(start: float, stop: float) -> list[float]:
    step = (stop - start) / 179.0
    return [start + step * index for index in range(180)]


def _eff_row(
    *,
    logical_key: str,
    algorithm: str,
    dataset_id: str,
    seed: int,
    task_id: str,
    host: str,
    trajectory: list[float],
    bundle_sha: str,
    bundle_report_sha: str,
    freeze_binding_sha: str,
    repair_manifest_sha: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "logical_key": logical_key,
        "algorithm": algorithm,
        "dataset_id": dataset_id,
        "seed": seed,
        "task_id": task_id,
        "host": host,
        "noise_tag": "clean",
        "m_eff": f"{efficiency_from_qualities(trajectory, horizon=180):.17g}",
        "best_quality": f"{max(trajectory):.17g}",
        "audited_repair_points": "0",
        "future_backfill_ignored_points": "0",
        "checkpoint_normalization_points": "0",
        "bundle_sha256": bundle_sha,
        "bundle_report_sha256": bundle_report_sha,
        "freeze_binding_report_sha256": freeze_binding_sha,
        "repair_manifest_sha256": repair_manifest_sha,
    }
    for index, quality in enumerate(trajectory, start=1):
        row[f"q_{index:04d}"] = f"{quality:.17g}"
    return row


def _evidence_row(
    *,
    pred_logical_id: str,
    pred_expression: str,
) -> dict[str, object]:
    gt_expression = "P + t"
    gt_artifact = build_symbolic_artifact(gt_expression)
    pred_artifact = build_symbolic_artifact(pred_expression)
    return {
        "gt_logical_id": "gt_simplify::BPG3",
        "pred_logical_id": pred_logical_id,
        "evidence_hash": _fake_sha(f"evidence::{pred_logical_id}"),
        "ground_truth": {
            "simplified_expression": gt_expression,
            "artifact_sha256": gt_artifact["artifact_sha256"],
        },
        "prediction": {
            "simplified_expression": pred_expression,
            "artifact_sha256": pred_artifact["artifact_sha256"],
        },
        "tree": {"tree_similarity": tree_similarity(gt_artifact, pred_artifact)},
        "variable": {"f1": variable_f1(gt_artifact, pred_artifact)},
        "operator": {"f1": operator_f1(gt_artifact, pred_artifact)},
    }


def _non_applicable_evidence_payload(
    *,
    logical_id: str,
    task_type: str,
    reason: str,
    request_context: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "symbolic_non_applicable.v1",
        "logical_id": logical_id,
        "task_type": task_type,
        "condition": "clean",
        "reason": reason,
        "dependencies": [],
        "request_context": request_context,
    }


def _plan_row(
    *,
    tmp_path: Path,
    task_type: str,
    logical_id: str,
    request: dict[str, object],
    priority: int,
    task_kind: str | None = None,
) -> dict[str, object]:
    prompt_path = tmp_path / "plan_assets" / f"{task_type}.prompt.txt"
    schema_path = tmp_path / "plan_assets" / f"{task_type}.schema.json"
    prompt_template = f"{task_type}\n{{{{REQUEST_JSON}}}}\n{{{{SCHEMA_JSON}}}}\n"
    schema_content = {
        "type": "object",
        "additionalProperties": True,
    }
    if not prompt_path.exists():
        _write_text(prompt_path, prompt_template)
    if not schema_path.exists():
        _write_json(schema_path, schema_content)
    prompt_sha256 = _sha256_file(prompt_path)
    schema_sha256 = _sha256_file(schema_path)
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    prompt_version = f"{task_type}.v1"
    schema_version = f"{task_type}.v1"
    evaluation_key_value = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    task_spec = TaskSpec(
        evaluation_key=evaluation_key_value,
        logical_id=logical_id,
        task_type=task_type,
        condition="clean",
        priority=priority,
        input_hash=input_hash,
        prompt_version=prompt_version,
        schema_version=schema_version,
        dependencies=(),
    )
    row: dict[str, object] = {
        "evaluation_key": evaluation_key_value,
        "logical_id": logical_id,
        "task_type": task_type,
        "condition": "clean",
        "priority": priority,
        "input_hash": input_hash,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "prompt_template": prompt_template,
        "request": request,
        "normalized_input": normalized_input,
        "schema_content": schema_content,
        "task_spec": json.loads(task_spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request, schema_content),
    }
    if task_kind is not None:
        row["task_kind"] = task_kind
    return row


def _write_frozen_group(
    *,
    tmp_path: Path,
    name: str,
    task_type: str,
    plan_rows: list[dict[str, object]],
    row_specs: list[dict[str, object]],
) -> dict[str, Path]:
    plan_path = tmp_path / "inputs" / f"{name}_plan.jsonl"
    index_path = tmp_path / "inputs" / f"{name}_index.jsonl"
    summary_path = tmp_path / "inputs" / f"{name}_summary.json"
    _write_jsonl(plan_path, plan_rows)
    plan_sha256 = _sha256_file(plan_path)
    plan_by_logical_id = {str(row["logical_id"]): row for row in plan_rows}
    index_rows: list[dict[str, object]] = []
    state_counts = {"frozen": 0, "non_applicable": 0}
    if task_type == "pred_simplify":
        state_counts["exhausted"] = 0
    for spec in row_specs:
        logical_id = str(spec["logical_id"])
        plan_row = plan_by_logical_id[logical_id]
        evaluation_key_value = str(plan_row["evaluation_key"])
        if spec["state"] == "frozen":
            result_path = tmp_path / "frozen_results" / name / f"{evaluation_key_value}.json"
            result_payload = {
                "evaluation_key": evaluation_key_value,
                "logical_id": logical_id,
                "task_type": task_type,
                "plan_sha256": plan_sha256,
                "structured_output": spec["structured_output"],
            }
            _write_json(result_path, result_payload)
            index_rows.append(
                {
                    "plan_sha256": plan_sha256,
                    "evaluation_key": evaluation_key_value,
                    "logical_id": logical_id,
                    "task_type": task_type,
                    "condition": "clean",
                    "priority": int(plan_row["priority"]),
                    "state": "frozen",
                    "attempt_id": f"attempt::{evaluation_key_value}",
                    "result_path": str(result_path.resolve()),
                    "result_sha256": _sha256_file(result_path),
                    "structured_output": spec["structured_output"],
                    "non_applicable": None,
                    "exhausted": None,
                }
            )
        elif spec["state"] == "non_applicable":
            reason = str(spec["reason"])
            evidence_path = tmp_path / "non_applicable" / name / f"{evaluation_key_value}.json"
            request_context = {
                key: value
                for key, value in dict(plan_row["request"]).items()
                if key != "evidence_hash"
            }
            evidence_payload = _non_applicable_evidence_payload(
                logical_id=logical_id,
                task_type=task_type,
                reason=reason,
                request_context=request_context,
            )
            _write_json(evidence_path, evidence_payload)
            evidence_sha256 = _sha256_file(evidence_path)
            assert evidence_sha256 == plan_row["request"]["evidence_hash"]
            index_rows.append(
                {
                    "plan_sha256": plan_sha256,
                    "evaluation_key": evaluation_key_value,
                    "logical_id": logical_id,
                    "task_type": task_type,
                    "condition": "clean",
                    "priority": int(plan_row["priority"]),
                    "state": "non_applicable",
                    "attempt_id": None,
                    "result_path": None,
                    "result_sha256": None,
                    "structured_output": None,
                    "non_applicable": {
                        "reason": reason,
                        "evidence_path": str(evidence_path.resolve()),
                        "evidence_sha256": evidence_sha256,
                    },
                    "exhausted": None,
                }
            )
        elif spec["state"] == "exhausted":
            assert task_type == "pred_simplify"
            attempts: list[dict[str, object]] = []
            for attempt_number in range(1, 4):
                attempt_id = f"{evaluation_key_value}.a{attempt_number:02d}"
                attempt_path = (
                    tmp_path / "attempts" / name / f"{attempt_id}.json"
                ).resolve()
                error_class = "timeout"
                attempt_payload = {
                    "attempt_id": attempt_id,
                    "evaluation_key": evaluation_key_value,
                    "metadata": {
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        "evaluation_key": evaluation_key_value,
                        "logical_id": logical_id,
                        "task_type": task_type,
                        "error_class": error_class,
                        "retryable": True,
                    },
                    "validation": {
                        "ok": False,
                        "error_class": error_class,
                        "error_message": f"fixture timeout {attempt_number}",
                    },
                }
                _write_json(attempt_path, attempt_payload)
                attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        "status": "failed",
                        "error_class": error_class,
                        "retryable": True,
                        "attempt_path": str(attempt_path),
                        "attempt_sha256": _sha256_file(attempt_path),
                    }
                )
            index_rows.append(
                {
                    "plan_sha256": plan_sha256,
                    "evaluation_key": evaluation_key_value,
                    "logical_id": logical_id,
                    "task_type": task_type,
                    "condition": "clean",
                    "priority": int(plan_row["priority"]),
                    "state": "exhausted",
                    "attempt_id": None,
                    "result_path": None,
                    "result_sha256": None,
                    "structured_output": None,
                    "non_applicable": None,
                    "exhausted": {
                        "attempt_count": 3,
                        "last_error_class": "timeout",
                        "attempts": attempts,
                    },
                }
            )
        else:
            raise AssertionError(f"未知 frozen index 测试状态: {spec['state']!r}")
        state_counts[str(spec["state"])] += 1
    _write_jsonl(index_path, index_rows)
    _write_json(
        summary_path,
        {
            "output_jsonl": str(index_path.resolve()),
            "output_sha256": _sha256_file(index_path),
            "plan_jsonl": str(plan_path.resolve()),
            "plan_sha256": plan_sha256,
            "row_count": len(index_rows),
            "state_counts": state_counts,
            "status": "ok",
        },
    )
    return {
        "plan_jsonl": plan_path,
        "summary_json": summary_path,
        "index_jsonl": index_path,
    }


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    numeric_csv = tmp_path / "inputs/clean_numeric_run_metrics.csv"
    eff_csv = tmp_path / "inputs/clean_eff_run_metrics.csv"
    clean_numeric_preparation_report_json = tmp_path / "inputs/clean_numeric_preparation.json"
    eff_preparation_report_json = tmp_path / "inputs/eff_preparation.json"
    evidence_jsonl = tmp_path / "inputs/clean_pred_vs_gt_evidence.jsonl"

    numeric_rows = [
        {
            "logical_key": "QLattice::BPG3::s520::clean",
            "algorithm": "QLattice",
            "dataset_id": "BPG3",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "qlattice_s520_clean_g0005",
            "host": "iaaccn25",
            "valid_output": "true",
            "id_quality": f"{phi_nmse(1e-6):.17g}",
            "ood_quality": f"{phi_nmse(1e-4):.17g}",
        },
        {
            "logical_key": "QLattice::BPG3::s521::clean",
            "algorithm": "QLattice",
            "dataset_id": "BPG3",
            "seed": "521",
            "noise_tag": "clean",
            "task_id": "qlattice_s521_clean_g0005",
            "host": "iaaccn24",
            "valid_output": "true",
            "id_quality": f"{phi_nmse(1e-5):.17g}",
            "ood_quality": f"{phi_nmse(1e-3):.17g}",
        },
        {
            "logical_key": "QLattice::BPG3::s522::clean",
            "algorithm": "QLattice",
            "dataset_id": "BPG3",
            "seed": "522",
            "noise_tag": "clean",
            "task_id": "qlattice_s522_clean_g0005",
            "host": "iaaccn23",
            "valid_output": "false",
            "id_quality": "0",
            "ood_quality": "0",
        },
    ]
    _write_csv(numeric_csv, numeric_rows)

    freeze_binding_json = tmp_path / "inputs/freeze_binding.json"
    freeze_bundle = tmp_path / "inputs/trajectory_freeze/clean_freeze_iaaccn25.jsonl.gz"
    source_runs_csv = tmp_path / "inputs/source_runs.csv"
    _write_json(freeze_binding_json, {"status": "frozen"})
    _write_text(freeze_bundle, "bundle\n")
    _write_text(source_runs_csv, "task_id\nqlattice_s520_clean_g0005\n")
    _write_json(
        clean_numeric_preparation_report_json,
        {
            "status": "ok",
            "contract_ok": True,
            "counts": {"runs": 3, "algorithms": 1, "valid_outputs": 2, "invalid_outputs": 1},
            "inputs": {
                "freeze_binding_json": str(freeze_binding_json.resolve()),
                "freeze_binding_sha256": _sha256_file(freeze_binding_json),
                "freeze_bundles": [
                    {
                        "path": str(freeze_bundle.resolve()),
                        "sha256": _sha256_file(freeze_bundle),
                    }
                ],
                "source_runs_csv": str(source_runs_csv.resolve()),
                "source_runs_sha256": _sha256_file(source_runs_csv),
            },
            "outputs": {
                "run_csv": str(numeric_csv.resolve()),
                "run_csv_sha256": _sha256_file(numeric_csv),
                "run_csv_row_count": 3,
            },
        },
    )

    freeze_binding_report = tmp_path / "inputs/eff_freeze_binding.json"
    repair_manifest = tmp_path / "inputs/trajectory_repairs.v1.json"
    freeze_record = tmp_path / "inputs/eff_freeze/clean_freeze_iaaccn25.jsonl.gz"
    freeze_report = tmp_path / "inputs/eff_freeze/clean_freeze_iaaccn25.report.json"
    _write_json(freeze_binding_report, {"status": "ok"})
    _write_json(repair_manifest, {"manifest_sha256": _fake_sha("repair-manifest")})
    _write_text(freeze_record, "record\n")
    _write_json(freeze_report, {"host": "iaaccn25"})
    freeze_binding_sha = _sha256_file(freeze_binding_report)
    repair_manifest_sha = _sha256_file(repair_manifest)
    bundle_sha = _sha256_file(freeze_record)
    bundle_report_sha = _sha256_file(freeze_report)

    trajectories = {
        520: _trajectory(0.2, 0.9),
        521: _trajectory(0.1, 0.8),
        522: [0.0] * 180,
    }
    _write_csv(
        eff_csv,
        [
            _eff_row(
                logical_key=f"QLattice::BPG3::s{seed}::clean",
                algorithm="QLattice",
                dataset_id="BPG3",
                seed=seed,
                task_id=f"qlattice_s{seed}_clean_g0005",
                host=f"iaaccn{25 - (seed - 520)}",
                trajectory=trajectories[seed],
                bundle_sha=bundle_sha,
                bundle_report_sha=bundle_report_sha,
                freeze_binding_sha=freeze_binding_sha,
                repair_manifest_sha=repair_manifest_sha,
            )
            for seed in (520, 521, 522)
        ],
    )
    _write_json(
        eff_preparation_report_json,
        {
            "status": "ok",
            "contract_ok": True,
            "condition": "clean",
            "horizon": 180,
            "inputs": {
                "freeze_binding_report": {
                    "path": str(freeze_binding_report.resolve()),
                    "sha256": freeze_binding_sha,
                },
                "freeze_records": [
                    {
                        "path": str(freeze_record.resolve()),
                        "sha256": bundle_sha,
                    }
                ],
                "freeze_reports": [
                    {
                        "path": str(freeze_report.resolve()),
                        "sha256": bundle_report_sha,
                    }
                ],
                "repair_manifest": {
                    "path": str(repair_manifest.resolve()),
                    "sha256": repair_manifest_sha,
                },
            },
            "outputs": {
                "eff_csv": str(eff_csv.resolve()),
                "eff_csv_sha256": _sha256_file(eff_csv),
                "eff_csv_row_count": 3,
            },
            "summary": {
                "processed_run_count": 3,
                "success_count": 3,
                "unresolved_run_count": 0,
                "full_contract_checked": True,
            },
            "unresolved": [],
        },
    )

    pred_s522_request_context = {
        "algorithm": "QLattice",
        "algorithm_slug": "qlattice",
        "dataset_id": "BPG3",
        "dataset_index": "g0005",
        "noise_tag": "clean",
        "seed": 522,
    }
    pred_s522_evidence_payload = _non_applicable_evidence_payload(
        logical_id="pred_simplify::qlattice::g0005::s522::clean",
        task_type="pred_simplify",
        reason="missing_final_expression",
        request_context=pred_s522_request_context,
    )
    eq_s522_request_context = {
        "algorithm_slug": "qlattice",
        "dataset_id": "BPG3",
        "dataset_index": "g0005",
        "noise_tag": "clean",
        "seed": 522,
    }
    eq_s522_evidence_payload = _non_applicable_evidence_payload(
        logical_id="equivalence::qlattice::g0005::s522::clean",
        task_type="equivalence",
        reason="upstream_pred_unavailable",
        request_context=eq_s522_request_context,
    )
    structure_520_522_request_context = {
        "algorithm_slug": "qlattice",
        "dataset_id": "BPG3",
        "dataset_index": "g0005",
        "left_seed": 520,
        "right_seed": 522,
        "noise_tag": "clean",
    }
    structure_520_522_evidence_payload = _non_applicable_evidence_payload(
        logical_id="stab_structure::qlattice::g0005::s520-s522",
        task_type="stab_structure",
        reason="invalid_seed_or_expression",
        request_context=structure_520_522_request_context,
    )
    structure_521_522_request_context = {
        "algorithm_slug": "qlattice",
        "dataset_id": "BPG3",
        "dataset_index": "g0005",
        "left_seed": 521,
        "right_seed": 522,
        "noise_tag": "clean",
    }
    structure_521_522_evidence_payload = _non_applicable_evidence_payload(
        logical_id="stab_structure::qlattice::g0005::s521-s522",
        task_type="stab_structure",
        reason="invalid_seed_or_expression",
        request_context=structure_521_522_request_context,
    )

    gt_plan_rows = [
        _plan_row(
            tmp_path=tmp_path,
            task_type="gt_simplify",
            logical_id="gt_simplify::BPG3",
            request={
                "dataset_id": "BPG3",
                "dataset_index": "g0005",
                "noise_tag": "clean",
                "evidence_hash": _fake_sha("gt_simplify::BPG3"),
            },
            priority=10,
        )
    ]
    pred_plan_rows = [
        _plan_row(
            tmp_path=tmp_path,
            task_type="pred_simplify",
            logical_id=f"pred_simplify::qlattice::g0005::s{seed}::clean",
            request={
                "algorithm": "QLattice",
                "algorithm_slug": "qlattice",
                "dataset_id": "BPG3",
                "dataset_index": "g0005",
                "noise_tag": "clean",
                "seed": seed,
                "evidence_hash": (
                    _sha256_text(_render_json(pred_s522_evidence_payload))
                    if seed == 522
                    else _fake_sha(f"pred_simplify::{seed}")
                ),
            },
            priority=20,
        )
        for seed in (520, 521, 522)
    ]
    equivalence_plan_rows = [
        _plan_row(
            tmp_path=tmp_path,
            task_type="equivalence",
            logical_id=f"equivalence::qlattice::g0005::s{seed}::clean",
            request=(
                {
                    **eq_s522_request_context,
                    "evidence_hash": _sha256_text(_render_json(eq_s522_evidence_payload)),
                }
                if seed == 522
                else {
                    "algorithm_slug": "qlattice",
                    "dataset_id": "BPG3",
                    "dataset_index": "g0005",
                    "noise_tag": "clean",
                    "seed": seed,
                    "evidence_hash": _fake_sha(f"equivalence::{seed}"),
                }
            ),
            priority=30,
        )
        for seed in (520, 521, 522)
    ]
    structure_plan_rows = [
        _plan_row(
            tmp_path=tmp_path,
            task_type="stab_structure",
            logical_id=f"stab_structure::qlattice::g0005::s{left}-s{right}",
            request=(
                {
                    **structure_520_522_request_context,
                    "evidence_hash": _sha256_text(_render_json(structure_520_522_evidence_payload)),
                }
                if (left, right) == (520, 522)
                else {
                    **structure_521_522_request_context,
                    "evidence_hash": _sha256_text(_render_json(structure_521_522_evidence_payload)),
                }
                if (left, right) == (521, 522)
                else {
                    "algorithm_slug": "qlattice",
                    "dataset_id": "BPG3",
                    "dataset_index": "g0005",
                    "left_seed": left,
                    "right_seed": right,
                    "noise_tag": "clean",
                    "evidence_hash": _fake_sha(f"stab_structure::{left}-{right}"),
                }
            ),
            priority=40,
        )
        for left, right in ((520, 521), (520, 522), (521, 522))
    ]

    gt_group = _write_frozen_group(
        tmp_path=tmp_path,
        name="gt_frozen",
        task_type="gt_simplify",
        plan_rows=gt_plan_rows,
        row_specs=[
            {
                "logical_id": "gt_simplify::BPG3",
                "state": "frozen",
                "structured_output": {
                    "outcome": "unchanged",
                    "simplified_expression": "P + t",
                    "equivalence_assessment": "preserved",
                },
            }
        ],
    )
    pred_group = _write_frozen_group(
        tmp_path=tmp_path,
        name="pred_frozen",
        task_type="pred_simplify",
        plan_rows=pred_plan_rows,
        row_specs=[
            {
                "logical_id": "pred_simplify::qlattice::g0005::s520::clean",
                "state": "frozen",
                "structured_output": {
                    "outcome": "unchanged",
                    "simplified_expression": "P + t",
                    "equivalence_assessment": "preserved",
                },
            },
            {
                "logical_id": "pred_simplify::qlattice::g0005::s521::clean",
                "state": "frozen",
                "structured_output": {
                    "outcome": "simplified",
                    "simplified_expression": "t",
                    "equivalence_assessment": "preserved",
                },
            },
            {
                "logical_id": "pred_simplify::qlattice::g0005::s522::clean",
                "state": "exhausted",
            },
        ],
    )
    equivalence_group = _write_frozen_group(
        tmp_path=tmp_path,
        name="equivalence_frozen",
        task_type="equivalence",
        plan_rows=equivalence_plan_rows,
        row_specs=[
            {
                "logical_id": "equivalence::qlattice::g0005::s520::clean",
                "state": "frozen",
                "structured_output": {
                    "decision": "equivalent",
                    "evidence_basis": "symbolic_proof",
                },
            },
            {
                "logical_id": "equivalence::qlattice::g0005::s521::clean",
                "state": "frozen",
                "structured_output": {
                    "decision": "undetermined",
                    "evidence_basis": "structural_analysis",
                },
            },
            {
                "logical_id": "equivalence::qlattice::g0005::s522::clean",
                "state": "non_applicable",
                "reason": "upstream_pred_unavailable",
            },
        ],
    )
    structure_group = _write_frozen_group(
        tmp_path=tmp_path,
        name="structure_frozen",
        task_type="stab_structure",
        plan_rows=structure_plan_rows,
        row_specs=[
            {
                "logical_id": "stab_structure::qlattice::g0005::s520-s521",
                "state": "frozen",
                "structured_output": {"decision": "same_canonical_structure"},
            },
            {
                "logical_id": "stab_structure::qlattice::g0005::s520-s522",
                "state": "non_applicable",
                "reason": "invalid_seed_or_expression",
            },
            {
                "logical_id": "stab_structure::qlattice::g0005::s521-s522",
                "state": "non_applicable",
                "reason": "invalid_seed_or_expression",
            },
        ],
    )

    _write_jsonl(
        evidence_jsonl,
        [
            _evidence_row(
                pred_logical_id="pred_simplify::qlattice::g0005::s520::clean",
                pred_expression="P + t",
            ),
            _evidence_row(
                pred_logical_id="pred_simplify::qlattice::g0005::s521::clean",
                pred_expression="t",
            ),
        ],
    )

    return {
        "numeric_csv": numeric_csv,
        "clean_numeric_preparation_report_json": clean_numeric_preparation_report_json,
        "eff_csv": eff_csv,
        "eff_preparation_report_json": eff_preparation_report_json,
        "gt_plan_jsonl": gt_group["plan_jsonl"],
        "gt_summary_json": gt_group["summary_json"],
        "gt_index_jsonl": gt_group["index_jsonl"],
        "pred_plan_jsonl": pred_group["plan_jsonl"],
        "pred_summary_json": pred_group["summary_json"],
        "pred_index_jsonl": pred_group["index_jsonl"],
        "equivalence_plan_jsonl": equivalence_group["plan_jsonl"],
        "equivalence_summary_json": equivalence_group["summary_json"],
        "equivalence_index_jsonl": equivalence_group["index_jsonl"],
        "structure_plan_jsonl": structure_group["plan_jsonl"],
        "structure_summary_json": structure_group["summary_json"],
        "structure_index_jsonl": structure_group["index_jsonl"],
        "evidence_jsonl": evidence_jsonl,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite_csv_rows(path: Path, mutate: Any) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    mutate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(paths: dict[str, Path], tmp_path: Path) -> dict[str, Any]:
    return aggregate_clean_metrics(
        numeric_csv=paths["numeric_csv"],
        clean_numeric_preparation_report_json=paths["clean_numeric_preparation_report_json"],
        eff_csv=paths["eff_csv"],
        eff_preparation_report_json=paths["eff_preparation_report_json"],
        gt_plan_jsonl=paths["gt_plan_jsonl"],
        gt_summary_json=paths["gt_summary_json"],
        gt_index_jsonl=paths["gt_index_jsonl"],
        pred_plan_jsonl=paths["pred_plan_jsonl"],
        pred_summary_json=paths["pred_summary_json"],
        pred_index_jsonl=paths["pred_index_jsonl"],
        equivalence_plan_jsonl=paths["equivalence_plan_jsonl"],
        equivalence_summary_json=paths["equivalence_summary_json"],
        equivalence_index_jsonl=paths["equivalence_index_jsonl"],
        structure_plan_jsonl=paths["structure_plan_jsonl"],
        structure_summary_json=paths["structure_summary_json"],
        structure_index_jsonl=paths["structure_index_jsonl"],
        evidence_jsonl=paths["evidence_jsonl"],
        clean_run_csv=tmp_path / "outputs/clean_run_metrics.csv",
        task_stability_csv=tmp_path / "outputs/task_stability.csv",
        algorithm_csv=tmp_path / "outputs/algorithm_six_axis.csv",
        report_json=tmp_path / "outputs/aggregate_clean_metrics.json",
        expected_runs=3,
        expected_algorithms=1,
        expected_datasets=1,
    )


def _rewrite_pred_exhausted_as_non_applicable(paths: dict[str, Path], tmp_path: Path) -> None:
    rows = _read_jsonl(paths["pred_index_jsonl"])
    target = next(
        row for row in rows if row["logical_id"] == "pred_simplify::qlattice::g0005::s522::clean"
    )
    evidence_path = tmp_path / "non_applicable" / "pred_frozen" / f"{target['evaluation_key']}.json"
    request_context = {
        "algorithm": "QLattice",
        "algorithm_slug": "qlattice",
        "dataset_id": "BPG3",
        "dataset_index": "g0005",
        "noise_tag": "clean",
        "seed": 522,
    }
    evidence_payload = _non_applicable_evidence_payload(
        logical_id="pred_simplify::qlattice::g0005::s522::clean",
        task_type="pred_simplify",
        reason="missing_final_expression",
        request_context=request_context,
    )
    _write_json(evidence_path, evidence_payload)
    target["state"] = "non_applicable"
    target["attempt_id"] = None
    target["result_path"] = None
    target["result_sha256"] = None
    target["structured_output"] = None
    target["non_applicable"] = {
        "reason": "missing_final_expression",
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256_file(evidence_path),
    }
    target["exhausted"] = None
    _write_jsonl(paths["pred_index_jsonl"], rows)
    pred_summary = json.loads(paths["pred_summary_json"].read_text(encoding="utf-8"))
    pred_summary["output_sha256"] = _sha256_file(paths["pred_index_jsonl"])
    pred_summary["state_counts"] = {
        "frozen": 2,
        "non_applicable": 1,
        "exhausted": 0,
    }
    _write_json(paths["pred_summary_json"], pred_summary)


def test_aggregate_clean_metrics_valid_fixture_passes(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)

    report = _aggregate(paths, tmp_path)

    clean_run_csv = tmp_path / "outputs/clean_run_metrics.csv"
    with clean_run_csv.open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    assert [row["logical_key"] for row in run_rows] == [
        "QLattice::BPG3::s520::clean",
        "QLattice::BPG3::s521::clean",
        "QLattice::BPG3::s522::clean",
    ]
    assert run_rows[0]["equivalence_decision"] == "equivalent"
    assert float(run_rows[0]["m_eff"]) == pytest.approx(
        efficiency_from_qualities(_trajectory(0.2, 0.9), horizon=180)
    )

    gt_artifact = build_symbolic_artifact("P + t")
    pred_artifact = build_symbolic_artifact("t")
    expected_partial_sym = symbolic_fidelity_score(
        equivalent=False,
        tree_similarity=tree_similarity(gt_artifact, pred_artifact),
        variable_f1=variable_f1(gt_artifact, pred_artifact),
        operator_f1=operator_f1(gt_artifact, pred_artifact),
    )
    assert run_rows[1]["equivalence_decision"] == "undetermined"
    assert float(run_rows[1]["m_sym"]) == pytest.approx(expected_partial_sym)
    assert float(run_rows[1]["m_min"]) == pytest.approx(
        minimality_score(
            int(run_rows[1]["reference_complexity"]),
            int(run_rows[1]["predicted_complexity"]),
        )
    )
    assert run_rows[2]["pred_state"] == "non_applicable"
    assert float(run_rows[2]["m_sym"]) == pytest.approx(0.0)
    assert float(run_rows[2]["m_min"]) == pytest.approx(0.0)

    report_payload = json.loads((tmp_path / "outputs/aggregate_clean_metrics.json").read_text("utf-8"))
    assert report["summary_sha256"] == report_payload["summary_sha256"]
    assert report_payload["inputs"]["clean_numeric_preparation_report_json"]["outputs"]["run_csv"]["row_count"] == 3
    assert report_payload["inputs"]["eff_preparation_report_json"]["outputs"]["eff_csv"]["row_count"] == 3
    assert report_payload["inputs"]["gt_frozen_index"]["summary_json"]["state_counts"] == {
        "frozen": 1,
        "non_applicable": 0,
    }
    assert report_payload["inputs"]["pred_frozen_index"]["summary_json"]["state_counts"] == {
        "frozen": 2,
        "non_applicable": 1,
        "exhausted": 0,
    }
    assert report_payload["summary"]["judge_exhausted_count"] == 0


def test_aggregate_clean_metrics_hard_fails_when_pred_index_contains_exhausted(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)

    with pytest.raises(AggregateCleanMetricsError, match="pred_frozen_index 仍包含 exhausted 终态"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_hard_fails_when_numeric_csv_binding_is_tampered(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    _rewrite_csv_rows(paths["numeric_csv"], lambda rows: rows.__setitem__(0, {**rows[0], "id_quality": "0.9"}))

    with pytest.raises(AggregateCleanMetricsError, match="run_csv.sha256 与目标文件不一致"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_hard_fails_when_eff_q_trajectory_is_tampered(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    _rewrite_csv_rows(paths["eff_csv"], lambda rows: rows[0].__setitem__("q_0005", "0.123456789"))
    report = json.loads(paths["eff_preparation_report_json"].read_text(encoding="utf-8"))
    report["outputs"]["eff_csv_sha256"] = _sha256_file(paths["eff_csv"])
    _write_json(paths["eff_preparation_report_json"], report)

    with pytest.raises(AggregateCleanMetricsError, match="m_eff 与 180 个 q 点重算结果不一致"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_hard_fails_when_equivalence_decision_is_tampered(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    rows = _read_jsonl(paths["equivalence_index_jsonl"])
    rows[0]["structured_output"]["decision"] = "not_equivalent"
    _write_jsonl(paths["equivalence_index_jsonl"], rows)
    summary = json.loads(paths["equivalence_summary_json"].read_text(encoding="utf-8"))
    summary["output_sha256"] = _sha256_file(paths["equivalence_index_jsonl"])
    _write_json(paths["equivalence_summary_json"], summary)

    with pytest.raises(AggregateCleanMetricsError, match="result.structured_output 与 index 不一致"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_hard_fails_on_noncanonical_logical_id(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    rows = _read_jsonl(paths["evidence_jsonl"])
    rows[0]["pred_logical_id"] = "pred_simplify::qlattice::BPG3::s520::clean"
    _write_jsonl(paths["evidence_jsonl"], rows)

    with pytest.raises(AggregateCleanMetricsError, match="无法解析 pred_simplify logical_id"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_hard_fails_on_extra_evidence(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    rows = _read_jsonl(paths["evidence_jsonl"])
    rows.append(
        _evidence_row(
            pred_logical_id="pred_simplify::qlattice::g0005::s522::clean",
            pred_expression="t",
        )
    )
    _write_jsonl(paths["evidence_jsonl"], rows)

    with pytest.raises(AggregateCleanMetricsError, match="evidence_jsonl logical_key 集合不闭合"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_revalidates_exhausted_attempt_artifacts(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    rows = _read_jsonl(paths["pred_index_jsonl"])
    exhausted_row = next(row for row in rows if row["state"] == "exhausted")
    attempt = exhausted_row["exhausted"]["attempts"][1]
    attempt_path = Path(attempt["attempt_path"])
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["validation"]["error_class"] = "tampered_error"
    _write_json(attempt_path, payload)
    attempt["attempt_sha256"] = _sha256_file(attempt_path)
    _write_jsonl(paths["pred_index_jsonl"], rows)
    summary = json.loads(paths["pred_summary_json"].read_text(encoding="utf-8"))
    summary["output_sha256"] = _sha256_file(paths["pred_index_jsonl"])
    _write_json(paths["pred_summary_json"], summary)

    with pytest.raises(AggregateCleanMetricsError, match="validation.error_class"):
        _aggregate(paths, tmp_path)


def test_aggregate_clean_metrics_missing_required_inputs_raises_type_error(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)

    with pytest.raises(TypeError):
        aggregate_clean_metrics(
            numeric_csv=paths["numeric_csv"],
            eff_csv=paths["eff_csv"],
            gt_index_jsonl=paths["gt_index_jsonl"],
            pred_index_jsonl=paths["pred_index_jsonl"],
            equivalence_index_jsonl=paths["equivalence_index_jsonl"],
            structure_index_jsonl=paths["structure_index_jsonl"],
            evidence_jsonl=paths["evidence_jsonl"],
            clean_run_csv=tmp_path / "outputs/clean_run_metrics.csv",
            task_stability_csv=tmp_path / "outputs/task_stability.csv",
            algorithm_csv=tmp_path / "outputs/algorithm_six_axis.csv",
            report_json=tmp_path / "outputs/aggregate_clean_metrics.json",
            expected_runs=3,
            expected_algorithms=1,
            expected_datasets=1,
        )


def test_cli_accepts_frozen_summary_and_plan_flags(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    _rewrite_pred_exhausted_as_non_applicable(paths, tmp_path)
    clean_run_csv = tmp_path / "cli/clean_run_metrics.csv"
    task_stability_csv = tmp_path / "cli/task_stability.csv"
    algorithm_csv = tmp_path / "cli/algorithm_six_axis.csv"
    report_json = tmp_path / "cli/aggregate_clean_metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_clean_metrics",
            "--numeric-csv",
            str(paths["numeric_csv"]),
            "--clean-numeric-preparation-report-json",
            str(paths["clean_numeric_preparation_report_json"]),
            "--eff-csv",
            str(paths["eff_csv"]),
            "--eff-preparation-report-json",
            str(paths["eff_preparation_report_json"]),
            "--gt-plan-jsonl",
            str(paths["gt_plan_jsonl"]),
            "--gt-summary-json",
            str(paths["gt_summary_json"]),
            "--gt-index-jsonl",
            str(paths["gt_index_jsonl"]),
            "--pred-plan-jsonl",
            str(paths["pred_plan_jsonl"]),
            "--pred-summary-json",
            str(paths["pred_summary_json"]),
            "--pred-index-jsonl",
            str(paths["pred_index_jsonl"]),
            "--equivalence-plan-jsonl",
            str(paths["equivalence_plan_jsonl"]),
            "--equivalence-summary-json",
            str(paths["equivalence_summary_json"]),
            "--equivalence-index-jsonl",
            str(paths["equivalence_index_jsonl"]),
            "--structure-plan-jsonl",
            str(paths["structure_plan_jsonl"]),
            "--structure-summary-json",
            str(paths["structure_summary_json"]),
            "--structure-index-jsonl",
            str(paths["structure_index_jsonl"]),
            "--evidence-jsonl",
            str(paths["evidence_jsonl"]),
            "--clean-run-csv",
            str(clean_run_csv),
            "--task-stability-csv",
            str(task_stability_csv),
            "--algorithm-csv",
            str(algorithm_csv),
            "--report-json",
            str(report_json),
            "--expected-runs",
            "3",
            "--expected-algorithms",
            "1",
            "--expected-datasets",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["summary"]["run_row_count"] == 3
    assert report_json.exists()
    assert clean_run_csv.exists()
    assert task_stability_csv.exists()
    assert algorithm_csv.exists()


def test_cli_requires_new_report_and_plan_flags(tmp_path: Path) -> None:
    paths = _build_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_clean_metrics",
            "--numeric-csv",
            str(paths["numeric_csv"]),
            "--eff-csv",
            str(paths["eff_csv"]),
            "--gt-index-jsonl",
            str(paths["gt_index_jsonl"]),
            "--pred-index-jsonl",
            str(paths["pred_index_jsonl"]),
            "--equivalence-index-jsonl",
            str(paths["equivalence_index_jsonl"]),
            "--structure-index-jsonl",
            str(paths["structure_index_jsonl"]),
            "--evidence-jsonl",
            str(paths["evidence_jsonl"]),
            "--clean-run-csv",
            str(tmp_path / "cli/clean_run_metrics.csv"),
            "--task-stability-csv",
            str(tmp_path / "cli/task_stability.csv"),
            "--algorithm-csv",
            str(tmp_path / "cli/algorithm_six_axis.csv"),
            "--report-json",
            str(tmp_path / "cli/aggregate_clean_metrics.json"),
            "--expected-runs",
            "3",
            "--expected-algorithms",
            "1",
            "--expected-datasets",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "--clean-numeric-preparation-report-json" in completed.stderr
    assert "--gt-plan-jsonl" in completed.stderr
    assert "--pred-summary-json" in completed.stderr
