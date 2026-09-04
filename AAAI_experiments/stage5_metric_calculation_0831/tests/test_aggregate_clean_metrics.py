"""Stage5 clean 六轴聚合器的定向回归测试。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import aggregate_clean_metrics as aggregate_module
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_clean_metrics import (
    AggregateCleanMetricsError,
    _load_frozen_index_rows,
    _parse_run_logical_id,
    _parse_structure_logical_id,
    _validate_simplify_structured_output,
    aggregate_clean_metrics,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    RunQuality,
    efficiency_from_qualities,
    minimality_score,
    numerical_consistency,
    phi_nmse,
    stability_score,
    symbolic_fidelity_score,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.frozen_result_index import (
    LLM_SIMPLIFIED_EXPRESSION,
    ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE,
    build_frozen_result_index,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskStateStore
from AAAI_experiments.stage5_metric_calculation_0831.tests.test_frozen_result_index import (
    _build_plan_row,
    _freeze_task,
    _exhaust_task,
    _write_plan_jsonl,
)


def _fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


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
        "bundle_sha256": _fake_sha(f"bundle::{task_id}"),
        "bundle_report_sha256": _fake_sha(f"bundle-report::{task_id}"),
        "freeze_binding_report_sha256": freeze_binding_sha,
        "repair_manifest_sha256": repair_manifest_sha,
    }
    for index, value in enumerate(trajectory, start=1):
        row[f"q_{index:04d}"] = f"{value:.17g}"
    return row


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "numeric_csv": tmp_path / "inputs/clean_numeric_run_metrics.csv",
        "eff_csv": tmp_path / "inputs/clean_eff_run_metrics.csv",
        "simplify_plan_jsonl": tmp_path / "inputs/clean_pred_simplify_tasks.jsonl",
        "eff_preparation_report_json": tmp_path / "inputs/eff_preparation.json",
        "gt_index_jsonl": tmp_path / "inputs/clean_gt_simplify_frozen_index.jsonl",
        "pred_index_jsonl": tmp_path / "inputs/clean_pred_simplify_frozen_index.jsonl",
        "equivalence_index_jsonl": tmp_path / "inputs/clean_equivalence_frozen_index.jsonl",
        "structure_index_jsonl": tmp_path / "inputs/clean_structure_frozen_index.jsonl",
        "evidence_jsonl": tmp_path / "inputs/clean_pred_vs_gt_evidence.jsonl",
    }


def _pred_plan_row(seed: int) -> dict[str, object]:
    logical_id = f"pred_simplify::algoa::g0001::s{seed}::clean"
    return {
        "logical_id": logical_id,
        "task_type": "pred_simplify",
        "evaluation_key": _fake_sha(f"plan::{logical_id}"),
        "priority": 20,
        "request": {
            "algorithm": "AlgoA",
            "algorithm_slug": "algoa",
            "dataset_id": "demo_ds",
            "dataset_index": "g0001",
            "noise_tag": "clean",
            "seed": seed,
            "expression": "x0 + x1",
            "original_expression": "x0 + x1 + 0",
        },
    }


def _evidence_row(
    *,
    logical_key: str,
    pred_logical_id: str,
    pred_expression: str,
    gt_logical_id: str = "gt_simplify::demo_ds",
    gt_expression: str = "x0 + x1",
) -> dict[str, object]:
    gt_artifact = build_symbolic_artifact(gt_expression)
    pred_artifact = build_symbolic_artifact(pred_expression)
    return {
        "logical_key": logical_key,
        "gt_logical_id": gt_logical_id,
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


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _fixture_paths(tmp_path)

    numeric_rows = [
        {
            "logical_key": "AlgoA::demo_ds::s520::clean",
            "algorithm": "AlgoA",
            "dataset_id": "demo_ds",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "algoa_s520_clean_g0001",
            "host": "host1",
            "result_sha256": _fake_sha("numeric-520"),
            "valid_output": "true",
            "evaluation_status": "valid",
            "invalid_reason": "",
            "replay_error": "",
            "formula_source": "canonical_artifact",
            "id_nmse": "1e-06",
            "ood_nmse": "1e-04",
            "id_quality": f"{phi_nmse(1e-06):.17g}",
            "ood_quality": f"{phi_nmse(1e-04):.17g}",
        },
        {
            "logical_key": "AlgoA::demo_ds::s521::clean",
            "algorithm": "AlgoA",
            "dataset_id": "demo_ds",
            "seed": "521",
            "noise_tag": "clean",
            "task_id": "algoa_s521_clean_g0001",
            "host": "host1",
            "result_sha256": _fake_sha("numeric-521"),
            "valid_output": "true",
            "evaluation_status": "valid",
            "invalid_reason": "",
            "replay_error": "",
            "formula_source": "canonical_artifact",
            "id_nmse": "1e-05",
            "ood_nmse": "1e-02",
            "id_quality": f"{phi_nmse(1e-05):.17g}",
            "ood_quality": f"{phi_nmse(1e-02):.17g}",
        },
        {
            "logical_key": "AlgoA::demo_ds::s522::clean",
            "algorithm": "AlgoA",
            "dataset_id": "demo_ds",
            "seed": "522",
            "noise_tag": "clean",
            "task_id": "algoa_s522_clean_g0001",
            "host": "host1",
            "result_sha256": _fake_sha("numeric-522"),
            "valid_output": "false",
            "evaluation_status": "invalid_output",
            "invalid_reason": "canonical prediction 含 NaN/Inf",
            "replay_error": "",
            "formula_source": "canonical_artifact",
            "id_nmse": "1",
            "ood_nmse": "1",
            "id_quality": "0",
            "ood_quality": "0",
        },
    ]
    _write_csv(paths["numeric_csv"], numeric_rows)

    freeze_binding_sha = _fake_sha("freeze-binding")
    repair_manifest_sha = _fake_sha("repair-manifest")
    trajectories = {
        520: _trajectory(0.2, 0.9),
        521: _trajectory(0.1, 0.8),
        522: [0.0] * 180,
    }
    eff_rows = [
        _eff_row(
            logical_key=f"AlgoA::demo_ds::s{seed}::clean",
            algorithm="AlgoA",
            dataset_id="demo_ds",
            seed=seed,
            task_id=f"algoa_s{seed}_clean_g0001",
            host="host1",
            trajectory=trajectories[seed],
            freeze_binding_sha=freeze_binding_sha,
            repair_manifest_sha=repair_manifest_sha,
        )
        for seed in (520, 521, 522)
    ]
    _write_csv(paths["eff_csv"], eff_rows)

    _write_jsonl(paths["simplify_plan_jsonl"], [_pred_plan_row(seed) for seed in (520, 521, 522)])

    paths["eff_preparation_report_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["eff_preparation_report_json"].write_text(
        json.dumps(
            {
                "condition": "clean",
                "horizon": 180,
                "inputs": {
                    "freeze_binding_report": {
                        "path": str(tmp_path / "inputs/freeze_binding.json"),
                        "sha256": freeze_binding_sha,
                    },
                    "repair_manifest": {
                        "path": str(tmp_path / "inputs/trajectory_repairs.v1.json"),
                        "sha256": repair_manifest_sha,
                    },
                },
                "summary": {
                    "processed_run_count": 3,
                    "success_count": 3,
                    "unresolved_run_count": 0,
                    "full_contract_checked": True,
                },
                "unresolved": [],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gt_index_rows = [
        {
            "logical_id": "gt_simplify::demo_ds",
            "task_type": "gt_simplify",
            "task_kind": "simplify",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("gt"),
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        }
    ]
    _write_jsonl(paths["gt_index_jsonl"], gt_index_rows)

    pred_index_rows = [
        {
            "logical_id": "pred_simplify::algoa::g0001::s520::clean",
            "task_type": "pred_simplify",
            "task_kind": "simplify",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("pred-520"),
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        },
        {
            "logical_id": "pred_simplify::algoa::g0001::s521::clean",
            "task_type": "pred_simplify",
            "task_kind": "simplify",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("pred-521"),
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1 + x2",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1 + x2",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        },
        {
            "logical_id": "pred_simplify::algoa::g0001::s522::clean",
            "task_type": "pred_simplify",
            "task_kind": "simplify",
            "condition": "clean",
            "state": "non_applicable",
            "result_sha256": None,
            "structured_output": None,
            "effective_expression": None,
            "expression_resolution": None,
            "non_applicable": {
                "reason": "missing_final_expression",
                "evidence_path": "audit/non_applicable_pred.json",
                "evidence_sha256": _fake_sha("pred-na"),
            },
        },
    ]
    _write_jsonl(paths["pred_index_jsonl"], pred_index_rows)

    equivalence_rows = [
        {
            "logical_id": "equivalence::algoa::g0001::s520::clean",
            "task_type": "equivalence",
            "task_kind": "equivalence",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("eq-520"),
            "structured_output": {
                "decision": "equivalent",
                "evidence_basis": "symbolic_proof",
            },
            "non_applicable": None,
        },
        {
            "logical_id": "equivalence::algoa::g0001::s521::clean",
            "task_type": "equivalence",
            "task_kind": "equivalence",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("eq-521"),
            "structured_output": {
                "decision": "undetermined",
                "evidence_basis": "structural_analysis",
            },
            "non_applicable": None,
        },
        {
            "logical_id": "equivalence::algoa::g0001::s522::clean",
            "task_type": "equivalence",
            "task_kind": "equivalence",
            "condition": "clean",
            "state": "non_applicable",
            "result_sha256": None,
            "structured_output": None,
            "non_applicable": {
                "reason": "upstream_pred_unavailable",
                "evidence_path": "audit/non_applicable_equivalence.json",
                "evidence_sha256": _fake_sha("eq-na"),
            },
        },
    ]
    _write_jsonl(paths["equivalence_index_jsonl"], equivalence_rows)

    structure_rows = [
        {
            "logical_id": "stab_structure::algoa::g0001::s520-s521",
            "task_type": "stab_structure",
            "task_kind": "structure",
            "condition": "clean",
            "state": "frozen",
            "result_sha256": _fake_sha("structure-520-521"),
            "structured_output": {"decision": "same_canonical_structure"},
            "non_applicable": None,
        },
        {
            "logical_id": "stab_structure::algoa::g0001::s520-s522",
            "task_type": "stab_structure",
            "task_kind": "structure",
            "condition": "clean",
            "state": "non_applicable",
            "result_sha256": None,
            "structured_output": None,
            "non_applicable": {
                "reason": "invalid_seed_or_expression",
                "evidence_path": "audit/non_applicable_structure_520_522.json",
                "evidence_sha256": _fake_sha("structure-na-520-522"),
            },
        },
        {
            "logical_id": "stab_structure::algoa::g0001::s521-s522",
            "task_type": "stab_structure",
            "task_kind": "structure",
            "condition": "clean",
            "state": "non_applicable",
            "result_sha256": None,
            "structured_output": None,
            "non_applicable": {
                "reason": "invalid_seed_or_expression",
                "evidence_path": "audit/non_applicable_structure_521_522.json",
                "evidence_sha256": _fake_sha("structure-na-521-522"),
            },
        },
    ]
    _write_jsonl(paths["structure_index_jsonl"], structure_rows)

    evidence_rows = [
        _evidence_row(
            logical_key="AlgoA::demo_ds::s520::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s520::clean",
            pred_expression="x0 + x1",
        ),
        _evidence_row(
            logical_key="AlgoA::demo_ds::s521::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s521::clean",
            pred_expression="x0 + x1 + x2",
        ),
    ]
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    return paths


def _patch_fixture_contract(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    rows_by_type = {
        "gt_simplify": [
            json.loads(line)
            for line in paths["gt_index_jsonl"].read_text(encoding="utf-8").splitlines()
        ],
        "pred_simplify": [
            json.loads(line)
            for line in paths["pred_index_jsonl"].read_text(encoding="utf-8").splitlines()
        ],
        "equivalence": [
            json.loads(line)
            for line in paths["equivalence_index_jsonl"].read_text(encoding="utf-8").splitlines()
        ],
        "stab_structure": [
            json.loads(line)
            for line in paths["structure_index_jsonl"].read_text(encoding="utf-8").splitlines()
        ],
    }
    state_counts: dict[str, dict[str, int]] = {}
    for task_type, rows in rows_by_type.items():
        counts = {"frozen": 0, "non_applicable": 0}
        for row in rows:
            counts[str(row["state"])] += 1
        state_counts[task_type] = counts

    def fake_load_frozen_index_rows(*, expected_task_type: str, **_: object):
        rows = rows_by_type[expected_task_type]
        plan_requests_by_logical_id: dict[str, dict[str, object]] = {}
        for row in rows_by_type["gt_simplify"]:
            logical_id = str(row["logical_id"])
            plan_requests_by_logical_id[logical_id] = {
                "dataset_id": "demo_ds",
                "expression": "x0 + x1",
                "original_expression": "x0 + x1 + 0",
            }
        for row in rows_by_type["pred_simplify"]:
            logical_id = str(row["logical_id"])
            seed = int(logical_id.split("::s", maxsplit=1)[1].split("::", maxsplit=1)[0])
            plan_requests_by_logical_id[logical_id] = {
                "dataset_id": "demo_ds",
                "dataset_index": "g0001",
                "seed": seed,
                "expression": "x0 + x1",
                "original_expression": "x0 + x1 + 0",
            }
        info = {
            "index_jsonl": {
                "path": str(tmp_path / f"{expected_task_type}.index.jsonl"),
                "sha256": _fake_sha(f"{expected_task_type}-index"),
                "row_count": len(rows),
            },
            "summary_json": {
                "path": str(tmp_path / f"{expected_task_type}.summary.json"),
                "sha256": _fake_sha(f"{expected_task_type}-summary"),
                "status": "ok",
                "state_counts": state_counts[expected_task_type],
            },
            "plan_jsonl": {
                "path": str(tmp_path / f"{expected_task_type}.plan.jsonl"),
                "sha256": _fake_sha(f"{expected_task_type}-plan"),
                "row_count": len(rows),
            },
            "plan_requests_by_logical_id": plan_requests_by_logical_id,
        }
        return rows, info

    monkeypatch.setattr(
        aggregate_module,
        "_validate_clean_numeric_preparation_report",
        lambda *args, **kwargs: {
            "path": str(tmp_path / "numeric_preparation.json"),
            "sha256": _fake_sha("numeric-preparation"),
            "row_count": 3,
        },
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_eff_preparation_report",
        lambda *args, **kwargs: {
            "path": str(paths["eff_preparation_report_json"]),
            "sha256": _fake_sha("eff-preparation"),
            "row_count": 3,
        },
    )
    monkeypatch.setattr(aggregate_module, "_load_frozen_index_rows", fake_load_frozen_index_rows)


def _aggregate_kwargs(paths: dict[str, Path], tmp_path: Path) -> dict[str, object]:
    return {
        "numeric_csv": paths["numeric_csv"],
        "clean_numeric_preparation_report_json": tmp_path / "inputs/numeric_preparation.json",
        "eff_csv": paths["eff_csv"],
        "eff_preparation_report_json": paths["eff_preparation_report_json"],
        "gt_plan_jsonl": tmp_path / "inputs/gt.plan.jsonl",
        "gt_summary_json": tmp_path / "inputs/gt.summary.json",
        "gt_index_jsonl": paths["gt_index_jsonl"],
        "pred_plan_jsonl": paths["simplify_plan_jsonl"],
        "pred_summary_json": tmp_path / "inputs/pred.summary.json",
        "pred_index_jsonl": paths["pred_index_jsonl"],
        "equivalence_plan_jsonl": tmp_path / "inputs/equivalence.plan.jsonl",
        "equivalence_summary_json": tmp_path / "inputs/equivalence.summary.json",
        "equivalence_index_jsonl": paths["equivalence_index_jsonl"],
        "structure_plan_jsonl": tmp_path / "inputs/structure.plan.jsonl",
        "structure_summary_json": tmp_path / "inputs/structure.summary.json",
        "structure_index_jsonl": paths["structure_index_jsonl"],
        "evidence_jsonl": paths["evidence_jsonl"],
        "clean_run_csv": tmp_path / "outputs/clean_run_metrics.csv",
        "task_stability_csv": tmp_path / "outputs/task_stability.csv",
        "algorithm_csv": tmp_path / "outputs/algorithm_six_axis.csv",
        "report_json": tmp_path / "outputs/aggregate_clean_metrics.json",
        "expected_runs": 3,
        "expected_algorithms": 1,
        "expected_datasets": 1,
    }


def test_eff_readiness_gate_rejects_unclosed_or_unbound_repairs() -> None:
    rows = {
        "AlgoA::demo_ds::s520::clean": {
            "audited_repair_points": "14",
            "future_backfill_ignored_points": "2",
            "checkpoint_normalization_points": "1",
        }
    }
    summary = {
        "formal_eff_ready": False,
        "original_missing_points": 15,
        "missing_points_after_repairs": 1,
        "audited_repair_points": 14,
        "future_backfill_ignored_points": 2,
        "checkpoint_normalization_points": 1,
    }

    with pytest.raises(AggregateCleanMetricsError, match="formal_eff_ready"):
        aggregate_module._validate_eff_readiness_summary(summary, eff_rows=rows)

    summary["formal_eff_ready"] = True
    summary["missing_points_after_repairs"] = 0
    summary["audited_repair_points"] = 15
    with pytest.raises(AggregateCleanMetricsError, match="audited_repair_points"):
        aggregate_module._validate_eff_readiness_summary(summary, eff_rows=rows)


def test_eff_readiness_gate_accepts_exact_repair_closure() -> None:
    rows = {
        "AlgoA::demo_ds::s520::clean": {
            "audited_repair_points": "15",
            "future_backfill_ignored_points": "2",
            "checkpoint_normalization_points": "1",
        },
        "AlgoA::demo_ds::s521::clean": {
            "audited_repair_points": "0",
            "future_backfill_ignored_points": "0",
            "checkpoint_normalization_points": "0",
        },
    }
    summary = {
        "formal_eff_ready": True,
        "original_missing_points": 15,
        "missing_points_after_repairs": 0,
        "audited_repair_points": 15,
        "future_backfill_ignored_points": 2,
        "checkpoint_normalization_points": 1,
    }

    assert aggregate_module._validate_eff_readiness_summary(summary, eff_rows=rows) == {
        "formal_eff_ready": True,
        "raw_missing_points": 15,
        "repaired_missing_points": 15,
        "unresolved_missing_points": 0,
        "future_backfill_ignored_points": 2,
        "checkpoint_normalization_points": 1,
    }


def test_canonical_replay_gate_accepts_closed_report_and_rows() -> None:
    rows = {
        "AlgoA::demo_ds::s520::clean": {
            "evaluation_path": "canonical_replay.v1",
            "canonical_replay_attempted_points": "180",
            "canonical_replay_succeeded_points": "180",
            "canonical_replay_failed_points": "0",
            "canonical_replay_invalid_output_points": "2",
        }
    }
    summary = {
        "evaluation_path": "canonical_replay.v1",
        "canonical_replay_attempted_points": 180,
        "canonical_replay_succeeded_points": 180,
        "canonical_replay_failed_points": 0,
        "canonical_replay_invalid_output_points": 2,
    }

    assert aggregate_module._validate_canonical_replay_contract(
        summary,
        rows=rows,
        report_context="eff_preparation_report.summary",
        row_point_fields=True,
    ) == {
        "evaluation_path": "canonical_replay.v1",
        "attempted_points": 180,
        "succeeded_points": 180,
        "failed_points": 0,
        "invalid_output_points": 2,
    }


@pytest.mark.parametrize(
    ("summary_update", "row_update", "message"),
    [
        ({"evaluation_path": "frozen_metrics.v1"}, {}, "evaluation_path"),
        ({"canonical_replay_failed_points": 1}, {}, "failed_points"),
        ({"canonical_replay_succeeded_points": 179}, {}, "attempted_points"),
        ({}, {"evaluation_path": "frozen_metrics.v1"}, "CSV.*evaluation_path"),
        (
            {},
            {"canonical_replay_succeeded_points": "179"},
            "CSV.*canonical_replay_succeeded_points",
        ),
    ],
)
def test_canonical_replay_gate_rejects_open_or_mixed_contract(
    summary_update: dict[str, object],
    row_update: dict[str, object],
    message: str,
) -> None:
    row = {
        "evaluation_path": "canonical_replay.v1",
        "canonical_replay_attempted_points": "180",
        "canonical_replay_succeeded_points": "180",
        "canonical_replay_failed_points": "0",
        "canonical_replay_invalid_output_points": "2",
    }
    row.update(row_update)
    summary = {
        "evaluation_path": "canonical_replay.v1",
        "canonical_replay_attempted_points": 180,
        "canonical_replay_succeeded_points": 180,
        "canonical_replay_failed_points": 0,
        "canonical_replay_invalid_output_points": 2,
    }
    summary.update(summary_update)

    with pytest.raises(AggregateCleanMetricsError, match=message):
        aggregate_module._validate_canonical_replay_contract(
            summary,
            rows={"AlgoA::demo_ds::s520::clean": row},
            report_context="eff_preparation_report.summary",
            row_point_fields=True,
        )


def test_clean_numeric_replay_gate_rejects_report_or_row_errors() -> None:
    payload = {
        "evaluation_path": "canonical_replay.v1",
        "counts": {
            "valid_outputs": 1,
            "canonical_invalid_outputs": 0,
            "replay_unavailable": 0,
            "replay_errors": 0,
        },
    }
    rows = {
        "AlgoA::demo_ds::s520::clean": {
            "evaluation_path": "canonical_replay.v1",
            "evaluation_status": "valid",
            "valid_output": "true",
            "invalid_reason": "",
            "id_quality": "0.8",
            "ood_quality": "0.7",
            "replay_error": "",
        }
    }
    assert aggregate_module._validate_clean_numeric_replay_contract(payload, rows=rows) == {
        "evaluation_path": "canonical_replay.v1",
        "valid_outputs": 1,
        "replay_errors": 0,
        "invalid_outputs": 0,
        "replay_unavailable": 0,
    }

    broken_payload = json.loads(json.dumps(payload))
    broken_payload["counts"]["replay_errors"] = 1
    with pytest.raises(AggregateCleanMetricsError, match="replay_errors"):
        aggregate_module._validate_clean_numeric_replay_contract(broken_payload, rows=rows)

    broken_rows = json.loads(json.dumps(rows))
    broken_rows["AlgoA::demo_ds::s520::clean"]["replay_error"] = "cannot parse"
    with pytest.raises(AggregateCleanMetricsError, match="CSV replay_error"):
        aggregate_module._validate_clean_numeric_replay_contract(payload, rows=broken_rows)

    unavailable_rows = json.loads(json.dumps(rows))
    unavailable_rows["AlgoA::demo_ds::s520::clean"].update(
        {
            "evaluation_status": "replay_unavailable",
            "valid_output": "",
            "id_quality": "",
            "ood_quality": "",
            "replay_error": "missing evidence",
        }
    )
    unavailable_payload = json.loads(json.dumps(payload))
    unavailable_payload["counts"].update(
        {"valid_outputs": 0, "replay_unavailable": 1, "replay_errors": 1}
    )
    with pytest.raises(AggregateCleanMetricsError, match="replay_unavailable"):
        aggregate_module._validate_clean_numeric_replay_contract(
            unavailable_payload, rows=unavailable_rows
        )


def test_formal_contract_requires_formula_audit_corrections_manifest() -> None:
    with pytest.raises(AggregateCleanMetricsError, match="audit corrections manifest"):
        aggregate_module._validate_audit_corrections_requirement(
            None,
            expected_runs=2250,
            expected_algorithms=15,
            expected_datasets=50,
        )

    assert (
        aggregate_module._validate_audit_corrections_requirement(
            None,
            expected_runs=3,
            expected_algorithms=1,
            expected_datasets=1,
        )
        is None
    )


def test_effective_view_accepts_hash_bound_audit_metadata() -> None:
    logical_id = "pred_simplify::algoa::g0001::s520::clean"
    base = {
        "logical_id": logical_id,
        "effective_expression": "x0 + 0",
        "expression_resolution": "llm_simplified_expression",
    }
    overlay_revision = _fake_sha("overlay")
    view = {
        **base,
        "effective_expression": "x0",
        "expression_resolution": "original_identity_fallback_after_audit",
        "base_file_sha256": _fake_sha("base-file"),
        "base_row_sha256": _fake_sha(
            json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "overlay_action": "replace_expression",
        "overlay_revision_sha256": overlay_revision,
        "source_audit_logical_id": "formula_audit::prediction::algoa::g0001::s520::clean",
        "source_final_record_sha256": _fake_sha("audit-row"),
        "suggested_simplified_expression": "x0",
    }
    view["output_row_sha256"] = _fake_sha(
        json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )

    assert aggregate_module._validate_effective_views(
        rows=[view],
        base_rows={logical_id: base},
        plan_rows={logical_id: {"request": {"original_expression": "x0"}}},
        base_file_sha256=_fake_sha("base-file"),
        overlay_revision_sha256=overlay_revision,
        context="test.pred_effective",
    ) == {logical_id: "x0"}


def test_aggregate_clean_metrics_builds_run_task_and_algorithm_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)
    kwargs = _aggregate_kwargs(paths, tmp_path)
    clean_run_csv = Path(kwargs["clean_run_csv"])
    task_stability_csv = Path(kwargs["task_stability_csv"])
    algorithm_csv = Path(kwargs["algorithm_csv"])
    report_json = Path(kwargs["report_json"])

    payload = aggregate_clean_metrics(**kwargs)

    assert payload["summary"]["run_row_count"] == 3
    assert payload["summary"]["task_row_count"] == 1
    assert payload["summary"]["algorithm_row_count"] == 1
    assert clean_run_csv.exists()
    assert task_stability_csv.exists()
    assert algorithm_csv.exists()
    assert report_json.exists()

    with clean_run_csv.open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    assert [row["seed"] for row in run_rows] == ["520", "521", "522"]

    run_520 = run_rows[0]
    assert run_520["equivalence_decision"] == "equivalent"
    assert float(run_520["m_sym"]) == pytest.approx(1.0)
    assert float(run_520["m_min"]) == pytest.approx(1.0)

    gt_artifact = build_symbolic_artifact("x0 + x1")
    pred_artifact = build_symbolic_artifact("x0 + x1 + x2")
    run_521 = run_rows[1]
    expected_partial_sym = symbolic_fidelity_score(
        equivalent=False,
        tree_similarity=tree_similarity(gt_artifact, pred_artifact),
        variable_f1=variable_f1(gt_artifact, pred_artifact),
        operator_f1=operator_f1(gt_artifact, pred_artifact),
    )
    assert run_521["equivalence_decision"] == "undetermined"
    assert float(run_521["m_sym"]) == pytest.approx(expected_partial_sym)
    assert float(run_521["m_min"]) == pytest.approx(
        minimality_score(
            int(run_521["reference_complexity"]),
            int(run_521["predicted_complexity"]),
        )
    )

    run_522 = run_rows[2]
    assert run_522["pred_state"] == "non_applicable"
    assert float(run_522["m_sym"]) == pytest.approx(0.0)
    assert float(run_522["m_min"]) == pytest.approx(0.0)

    with task_stability_csv.open("r", encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(task_rows) == 1
    task_row = task_rows[0]
    expected_n = numerical_consistency(
        [
            RunQuality(id_quality=phi_nmse(1e-06), ood_quality=phi_nmse(1e-04), valid=True),
            RunQuality(id_quality=phi_nmse(1e-05), ood_quality=phi_nmse(1e-02), valid=True),
            RunQuality(id_quality=0.0, ood_quality=0.0, valid=False),
        ]
    )
    expected_stab = stability_score(
        [
            RunQuality(id_quality=phi_nmse(1e-06), ood_quality=phi_nmse(1e-04), valid=True),
            RunQuality(id_quality=phi_nmse(1e-05), ood_quality=phi_nmse(1e-02), valid=True),
            RunQuality(id_quality=0.0, ood_quality=0.0, valid=False),
        ],
        structural_pair_results=[True, False, False],
    )
    assert float(task_row["numerical_consistency"]) == pytest.approx(expected_n)
    assert float(task_row["validity"]) == pytest.approx(2.0 / 3.0)
    assert float(task_row["structural_consistency"]) == pytest.approx(1.0 / 3.0)
    assert float(task_row["m_stab"]) == pytest.approx(expected_stab.score)

    with algorithm_csv.open("r", encoding="utf-8", newline="") as handle:
        algorithm_rows = list(csv.DictReader(handle))
    assert len(algorithm_rows) == 1
    algorithm_row = algorithm_rows[0]
    expected_eff = 100.0 * sum(
        efficiency_from_qualities(trajectory, horizon=180)
        for trajectory in (_trajectory(0.2, 0.9), _trajectory(0.1, 0.8), [0.0] * 180)
    ) / 3.0
    assert float(algorithm_row["eff_score"]) == pytest.approx(expected_eff)
    assert 0.0 <= float(algorithm_row["id_score"]) <= 100.0
    assert 0.0 <= float(algorithm_row["ood_score"]) <= 100.0
    assert 0.0 <= float(algorithm_row["sym_score"]) <= 100.0
    assert 0.0 <= float(algorithm_row["min_score"]) <= 100.0
    assert 0.0 <= float(algorithm_row["stab_score"]) <= 100.0

    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["summary_sha256"] == payload["summary_sha256"]
    assert report_payload["outputs"]["clean_run_metrics_csv"]["sha256"]
    for key in (
        "gt_frozen_index",
        "pred_frozen_index",
        "equivalence_frozen_index",
        "structure_frozen_index",
    ):
        assert "plan_requests_by_logical_id" not in report_payload["inputs"][key]


def test_aggregate_clean_metrics_accepts_gt_v2_logical_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    gt_rows = [json.loads(line) for line in paths["gt_index_jsonl"].read_text(encoding="utf-8").splitlines()]
    gt_rows[0]["logical_id"] = "gt_simplify::demo_ds::v2"
    _write_jsonl(paths["gt_index_jsonl"], gt_rows)

    evidence_rows = [
        _evidence_row(
            logical_key="AlgoA::demo_ds::s520::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s520::clean",
            pred_expression="x0 + x1",
            gt_logical_id="gt_simplify::demo_ds::v2",
        ),
        _evidence_row(
            logical_key="AlgoA::demo_ds::s521::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s521::clean",
            pred_expression="x0 + x1 + x2",
            gt_logical_id="gt_simplify::demo_ds::v2",
        ),
    ]
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    kwargs = _aggregate_kwargs(paths, tmp_path)
    aggregate_clean_metrics(**kwargs)

    with Path(kwargs["clean_run_csv"]).open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    assert {row["gt_logical_id"] for row in run_rows} == {"gt_simplify::demo_ds::v2"}


def test_aggregate_clean_metrics_accepts_pred_v2_logical_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    pred_logical_id = "pred_simplify::algoa::g0001::s520::clean::v2"

    plan_rows = [
        json.loads(line)
        for line in paths["simplify_plan_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    plan_rows[0]["logical_id"] = pred_logical_id
    _write_jsonl(paths["simplify_plan_jsonl"], plan_rows)

    pred_rows = [
        json.loads(line)
        for line in paths["pred_index_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    pred_rows[0]["logical_id"] = pred_logical_id
    _write_jsonl(paths["pred_index_jsonl"], pred_rows)

    evidence_rows = [
        json.loads(line)
        for line in paths["evidence_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    evidence_rows[0]["pred_logical_id"] = pred_logical_id
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    kwargs = _aggregate_kwargs(paths, tmp_path)
    aggregate_clean_metrics(**kwargs)

    with Path(kwargs["clean_run_csv"]).open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    run_520 = next(row for row in run_rows if row["seed"] == "520")
    assert run_520["pred_logical_id"] == pred_logical_id


def test_symbolic_logical_id_parsers_accept_successor_versions() -> None:
    assert _parse_run_logical_id(
        "equivalence::algoa::g0001::s520::clean::v2",
        expected_prefix="equivalence",
    ) == ("algoa", "g0001", 520)
    assert _parse_structure_logical_id(
        "stab_structure::algoa::g0001::s520-s521::v2"
    ) == ("algoa", "g0001", (520, 521))


def test_validate_simplify_normalizes_edge_whitespace_consistently() -> None:
    expression, state = _validate_simplify_structured_output(
        {
            "structured_output": {
                "outcome": "simplified",
                "simplified_expression": "x0 + x1 ",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1 ",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
        },
        logical_id="pred_simplify::algoa::g0001::s520::clean",
        plan_request={"original_expression": "x0 + x1"},
        allow_missing=True,
    )

    assert expression == "x0 + x1"
    assert state == "frozen"


def test_aggregate_clean_metrics_accepts_gt_unable_with_original_identity_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    gt_rows = [json.loads(line) for line in paths["gt_index_jsonl"].read_text(encoding="utf-8").splitlines()]
    gt_rows[0] = {
        **gt_rows[0],
        "logical_id": "gt_simplify::demo_ds::v2",
        "structured_output": {
            "outcome": "unable",
            "simplified_expression": None,
            "equivalence_assessment": "undetermined",
        },
        "effective_expression": "x0 + x1 + 0",
        "expression_resolution": ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE,
    }
    _write_jsonl(paths["gt_index_jsonl"], gt_rows)

    evidence_rows = [
        _evidence_row(
            logical_key="AlgoA::demo_ds::s520::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s520::clean",
            pred_expression="x0 + x1",
            gt_logical_id="gt_simplify::demo_ds::v2",
            gt_expression="x0 + x1 + 0",
        ),
        _evidence_row(
            logical_key="AlgoA::demo_ds::s521::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s521::clean",
            pred_expression="x0 + x1 + x2",
            gt_logical_id="gt_simplify::demo_ds::v2",
            gt_expression="x0 + x1 + 0",
        ),
    ]
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)

    def _fake_load_frozen_index_rows(*, expected_task_type: str, **_: object):
        rows_by_type = {
            "gt_simplify": [
                json.loads(line) for line in paths["gt_index_jsonl"].read_text(encoding="utf-8").splitlines()
            ],
            "pred_simplify": [
                json.loads(line) for line in paths["pred_index_jsonl"].read_text(encoding="utf-8").splitlines()
            ],
            "equivalence": [
                json.loads(line)
                for line in paths["equivalence_index_jsonl"].read_text(encoding="utf-8").splitlines()
            ],
            "stab_structure": [
                json.loads(line) for line in paths["structure_index_jsonl"].read_text(encoding="utf-8").splitlines()
            ],
        }
        rows = rows_by_type[expected_task_type]
        counts = {"frozen": 0, "non_applicable": 0}
        for row in rows:
            counts[str(row["state"])] += 1
        return rows, {
            "index_jsonl": {
                "path": str(tmp_path / f"{expected_task_type}.index.jsonl"),
                "sha256": _fake_sha(f"{expected_task_type}-index"),
                "row_count": len(rows),
            },
            "summary_json": {
                "path": str(tmp_path / f"{expected_task_type}.summary.json"),
                "sha256": _fake_sha(f"{expected_task_type}-summary"),
                "status": "ok",
                "state_counts": counts,
            },
            "plan_jsonl": {
                "path": str(tmp_path / f"{expected_task_type}.plan.jsonl"),
                "sha256": _fake_sha(f"{expected_task_type}-plan"),
                "row_count": len(rows),
            },
            "plan_requests_by_logical_id": {
                "gt_simplify::demo_ds::v2": {
                    "dataset_id": "demo_ds",
                    "expression": "x0 + x1",
                    "original_expression": "x0 + x1 + 0",
                }
            },
        }

    monkeypatch.setattr(
        aggregate_module,
        "_validate_clean_numeric_preparation_report",
        lambda *args, **kwargs: {
            "path": str(tmp_path / "numeric_preparation.json"),
            "sha256": _fake_sha("numeric-preparation"),
            "row_count": 3,
        },
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_eff_preparation_report",
        lambda *args, **kwargs: {
            "path": str(paths["eff_preparation_report_json"]),
            "sha256": _fake_sha("eff-preparation"),
            "row_count": 3,
        },
    )
    monkeypatch.setattr(aggregate_module, "_load_frozen_index_rows", _fake_load_frozen_index_rows)

    kwargs = _aggregate_kwargs(paths, tmp_path)
    aggregate_clean_metrics(**kwargs)

    with Path(kwargs["clean_run_csv"]).open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    assert {row["gt_logical_id"] for row in run_rows} == {"gt_simplify::demo_ds::v2"}
    assert {int(row["reference_complexity"]) for row in run_rows} == {3}


def test_aggregate_clean_metrics_rejects_tampered_gt_unable_identity_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    gt_rows = [json.loads(line) for line in paths["gt_index_jsonl"].read_text(encoding="utf-8").splitlines()]
    gt_rows[0] = {
        **gt_rows[0],
        "structured_output": {
            "outcome": "unable",
            "simplified_expression": None,
            "equivalence_assessment": "undetermined",
        },
        "effective_expression": "x0 + x1",
        "expression_resolution": ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE,
    }
    _write_jsonl(paths["gt_index_jsonl"], gt_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    with pytest.raises(AggregateCleanMetricsError, match="effective_expression"):
        aggregate_clean_metrics(**_aggregate_kwargs(paths, tmp_path))


def test_aggregate_clean_metrics_treats_unable_as_original_identity_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    pred_rows = [json.loads(line) for line in paths["pred_index_jsonl"].read_text(encoding="utf-8").splitlines()]
    pred_rows[2] = {
        **pred_rows[2],
        "state": "frozen",
        "result_sha256": _fake_sha("pred-522"),
        "structured_output": {
            "outcome": "unable",
            "simplified_expression": None,
            "equivalence_assessment": "undetermined",
        },
        "effective_expression": "x0 + x1 + 0",
        "expression_resolution": ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE,
        "non_applicable": None,
    }
    _write_jsonl(paths["pred_index_jsonl"], pred_rows)

    equivalence_rows = [
        json.loads(line) for line in paths["equivalence_index_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    equivalence_rows[2] = {
        **equivalence_rows[2],
        "state": "frozen",
        "result_sha256": _fake_sha("eq-522"),
        "structured_output": {
            "decision": "equivalent",
            "evidence_basis": "mixed",
        },
        "non_applicable": None,
    }
    _write_jsonl(paths["equivalence_index_jsonl"], equivalence_rows)

    evidence_rows = [
        _evidence_row(
            logical_key="AlgoA::demo_ds::s520::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s520::clean",
            pred_expression="x0 + x1",
        ),
        _evidence_row(
            logical_key="AlgoA::demo_ds::s521::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s521::clean",
            pred_expression="x0 + x1 + x2",
        ),
        _evidence_row(
            logical_key="AlgoA::demo_ds::s522::clean",
            pred_logical_id="pred_simplify::algoa::g0001::s522::clean",
            pred_expression="x0 + x1 + 0",
        ),
    ]
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    kwargs = _aggregate_kwargs(paths, tmp_path)
    aggregate_clean_metrics(**kwargs)

    with Path(kwargs["clean_run_csv"]).open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    run_522 = next(row for row in run_rows if row["seed"] == "522")
    assert run_522["pred_state"] == "frozen"
    assert run_522["equivalence_state"] == "frozen"
    assert run_522["equivalence_decision"] == "equivalent"
    assert int(run_522["predicted_complexity"]) > 0
    assert float(run_522["m_sym"]) == pytest.approx(1.0)
    assert float(run_522["m_min"]) > 0.0


def test_aggregate_clean_metrics_hard_fails_when_valid_run_lacks_deterministic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)
    _write_jsonl(
        paths["evidence_jsonl"],
        [
            _evidence_row(
                logical_key="AlgoA::demo_ds::s520::clean",
                pred_logical_id="pred_simplify::algoa::g0001::s520::clean",
                pred_expression="x0 + x1",
            )
        ],
    )

    with pytest.raises(AggregateCleanMetricsError, match="evidence_jsonl.*不闭合"):
        aggregate_clean_metrics(**_aggregate_kwargs(paths, tmp_path))


def test_aggregate_accepts_recanonicalized_artifact_when_expression_and_metrics_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    evidence_rows = [
        json.loads(line)
        for line in paths["evidence_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    evidence_rows[0]["prediction"]["artifact_sha256"] = _fake_sha("older-sympy-artifact")
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    report = aggregate_clean_metrics(**_aggregate_kwargs(paths, tmp_path))

    assert report["summary"]["artifact_recanonicalization_count"] == 1


def test_aggregate_recomputes_metric_when_bound_artifact_was_recanonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    evidence_rows = [
        json.loads(line)
        for line in paths["evidence_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    evidence_rows[0]["prediction"]["artifact_sha256"] = _fake_sha("older-sympy-artifact")
    evidence_rows[0]["tree"]["tree_similarity"] = 0.25
    _write_jsonl(paths["evidence_jsonl"], evidence_rows)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)

    report = aggregate_clean_metrics(**_aggregate_kwargs(paths, tmp_path))

    assert report["summary"]["artifact_recanonicalization_count"] == 1
    assert report["summary"]["metric_recanonicalization_count"] == 1
    assert report["diagnostics"]["symbolic_recanonicalization"]["artifact_bindings"][0][
        "logical_id"
    ] == "pred_simplify::algoa::g0001::s520::clean"
    assert report["diagnostics"]["runtime"]["sympy_version"]


def test_cli_writes_atomic_outputs_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    _patch_fixture_contract(monkeypatch, paths, tmp_path)
    kwargs = _aggregate_kwargs(paths, tmp_path)
    kwargs["clean_run_csv"] = tmp_path / "cli/clean_run_metrics.csv"
    kwargs["task_stability_csv"] = tmp_path / "cli/task_stability.csv"
    kwargs["algorithm_csv"] = tmp_path / "cli/algorithm_six_axis.csv"
    kwargs["report_json"] = tmp_path / "cli/aggregate_clean_metrics.json"

    exit_code = aggregate_module.main(
        [
            "--numeric-csv",
            str(kwargs["numeric_csv"]),
            "--clean-numeric-preparation-report-json",
            str(kwargs["clean_numeric_preparation_report_json"]),
            "--eff-csv",
            str(kwargs["eff_csv"]),
            "--eff-preparation-report-json",
            str(kwargs["eff_preparation_report_json"]),
            "--gt-plan-jsonl",
            str(kwargs["gt_plan_jsonl"]),
            "--gt-summary-json",
            str(kwargs["gt_summary_json"]),
            "--gt-index-jsonl",
            str(kwargs["gt_index_jsonl"]),
            "--pred-plan-jsonl",
            str(kwargs["pred_plan_jsonl"]),
            "--pred-summary-json",
            str(kwargs["pred_summary_json"]),
            "--pred-index-jsonl",
            str(kwargs["pred_index_jsonl"]),
            "--equivalence-plan-jsonl",
            str(kwargs["equivalence_plan_jsonl"]),
            "--equivalence-summary-json",
            str(kwargs["equivalence_summary_json"]),
            "--equivalence-index-jsonl",
            str(kwargs["equivalence_index_jsonl"]),
            "--structure-plan-jsonl",
            str(kwargs["structure_plan_jsonl"]),
            "--structure-summary-json",
            str(kwargs["structure_summary_json"]),
            "--structure-index-jsonl",
            str(kwargs["structure_index_jsonl"]),
            "--evidence-jsonl",
            str(kwargs["evidence_jsonl"]),
            "--clean-run-csv",
            str(kwargs["clean_run_csv"]),
            "--task-stability-csv",
            str(kwargs["task_stability_csv"]),
            "--algorithm-csv",
            str(kwargs["algorithm_csv"]),
            "--report-json",
            str(kwargs["report_json"]),
            "--expected-runs",
            "3",
            "--expected-algorithms",
            "1",
            "--expected-datasets",
            "1",
        ]
    )
    assert exit_code == 0
    report_payload = json.loads(Path(kwargs["report_json"]).read_text(encoding="utf-8"))
    assert report_payload["summary"]["run_row_count"] == 3
    assert report_payload["summary"]["task_row_count"] == 1
    assert report_payload["summary"]["algorithm_row_count"] == 1
    assert Path(kwargs["clean_run_csv"]).exists()
    assert Path(kwargs["task_stability_csv"]).exists()
    assert Path(kwargs["algorithm_csv"]).exists()


def test_load_frozen_index_rows_accepts_pred_exhausted_and_revalidates_attempts(
    tmp_path: Path,
) -> None:
    row = _build_plan_row(
        tmp_path,
        "pred_simplify::fixture::g0001::s520::clean",
        task_type="pred_simplify",
    )
    plan_path = tmp_path / "pred_plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    attempts_dir = tmp_path / "attempts"
    store = TaskStateStore(state_db, attempt_cap=10)
    _exhaust_task(store, row, attempts_dir, error_class="validation_failed")

    output_jsonl = tmp_path / "pred_frozen_index.jsonl"
    summary_json = tmp_path / "pred_frozen_index.summary.json"
    build_frozen_result_index(
        plan_jsonl=plan_path,
        state_db=state_db,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        allow_exhausted=True,
        attempts_dir=attempts_dir,
    )

    rows, info = _load_frozen_index_rows(
        index_path=output_jsonl,
        summary_path=summary_json,
        plan_path=plan_path,
        expected_task_type="pred_simplify",
        label="pred_frozen_index",
        allow_exhausted=True,
    )

    assert len(rows) == 1
    exhausted_row = rows[0]
    assert exhausted_row["state"] == "exhausted"
    assert exhausted_row["exhausted"]["attempt_count"] == 3
    assert [item["attempt_number"] for item in exhausted_row["exhausted"]["attempts"]] == [1, 2, 3]
    assert info["summary_json"]["state_counts"]["exhausted"] == 1


def test_load_frozen_index_rows_rejects_tampered_unable_identity_fallback(
    tmp_path: Path,
) -> None:
    row = _build_plan_row(
        tmp_path,
        "pred_simplify::fixture::g0001::s520::clean",
        task_type="pred_simplify",
        request={
            "dataset_id": "demo_ds",
            "dataset_index": "g0001",
            "algorithm": "AlgoA",
            "algorithm_slug": "fixture",
            "seed": 520,
            "noise_tag": "clean",
            "task_id": "fixture_s520_clean_g0001",
            "expression": "x0 + x1",
            "original_expression": "x0 + x1 + 0",
            "evidence_hash": _fake_sha("unable-plan"),
        },
    )
    plan_path = tmp_path / "pred_plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db, attempt_cap=10)
    result_path = tmp_path / "frozen" / "unable.json"
    _freeze_task(
        store,
        row,
        result_path,
        structured_output={
            "outcome": "unable",
            "simplified_expression": None,
            "equivalence_assessment": "undetermined",
            "assumptions": [],
            "confidence": 0.2,
            "brief_reason": "fixture unable",
        },
    )

    output_jsonl = tmp_path / "pred_frozen_index.jsonl"
    summary_json = tmp_path / "pred_frozen_index.summary.json"
    build_frozen_result_index(
        plan_jsonl=plan_path,
        state_db=state_db,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
    )

    output_row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    output_row["effective_expression"] = "x0 + x1"
    _write_jsonl(output_jsonl, [output_row])
    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    summary_payload["output_sha256"] = hashlib.sha256(output_jsonl.read_bytes()).hexdigest()
    summary_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AggregateCleanMetricsError, match="effective_expression"):
        _load_frozen_index_rows(
            index_path=output_jsonl,
            summary_path=summary_json,
            plan_path=plan_path,
            expected_task_type="pred_simplify",
            label="pred_frozen_index",
        )


def test_load_frozen_index_rows_rejects_tampered_exhausted_attempt_artifact(
    tmp_path: Path,
) -> None:
    row = _build_plan_row(
        tmp_path,
        "pred_simplify::fixture::g0002::s520::clean",
        task_type="pred_simplify",
    )
    plan_path = tmp_path / "pred_plan.jsonl"
    _write_plan_jsonl(plan_path, [row])
    state_db = tmp_path / "state.sqlite3"
    attempts_dir = tmp_path / "attempts"
    store = TaskStateStore(state_db, attempt_cap=10)
    _exhaust_task(store, row, attempts_dir, error_class="validation_failed")

    output_jsonl = tmp_path / "pred_frozen_index.jsonl"
    summary_json = tmp_path / "pred_frozen_index.summary.json"
    build_frozen_result_index(
        plan_jsonl=plan_path,
        state_db=state_db,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        allow_exhausted=True,
        attempts_dir=attempts_dir,
    )
    exhausted_row = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
    attempt_path = Path(exhausted_row["exhausted"]["attempts"][0]["attempt_path"])
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["metadata"]["logical_id"] = "pred_simplify::drift::g9999::s520::clean"
    attempt_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AggregateCleanMetricsError, match="attempt_sha256|attempt_json"):
        _load_frozen_index_rows(
            index_path=output_jsonl,
            summary_path=summary_json,
            plan_path=plan_path,
            expected_task_type="pred_simplify",
            label="pred_frozen_index",
            allow_exhausted=True,
        )


def test_aggregate_clean_metrics_rejects_exhausted_clean_judge_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    clean_run_csv = tmp_path / "outputs/clean_run_metrics.csv"
    task_stability_csv = tmp_path / "outputs/task_stability.csv"
    algorithm_csv = tmp_path / "outputs/algorithm_six_axis.csv"
    report_json = tmp_path / "outputs/aggregate_clean_metrics.json"

    dummy_info = {
        "index_jsonl": {"path": str(tmp_path / "dummy.jsonl"), "sha256": _fake_sha("idx"), "row_count": 1},
        "summary_json": {"path": str(tmp_path / "dummy.summary.json"), "sha256": _fake_sha("sum"), "status": "ok", "state_counts": {"frozen": 1, "non_applicable": 0}},
        "plan_jsonl": {"path": str(tmp_path / "dummy.plan.jsonl"), "sha256": _fake_sha("plan"), "row_count": 1},
        "plan_requests_by_logical_id": {
            "gt_simplify::demo_ds": {
                "dataset_id": "demo_ds",
                "expression": "x0 + x1",
                "original_expression": "x0 + x1 + 0",
            }
        },
    }

    gt_rows = [
        {
            "logical_id": "gt_simplify::demo_ds",
            "task_type": "gt_simplify",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        }
    ]
    pred_rows = [
        {
            "logical_id": "pred_simplify::algoa::g0001::s520::clean",
            "task_type": "pred_simplify",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        },
        {
            "logical_id": "pred_simplify::algoa::g0001::s521::clean",
            "task_type": "pred_simplify",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x0 + x1 + x2",
                "equivalence_assessment": "preserved",
            },
            "effective_expression": "x0 + x1 + x2",
            "expression_resolution": LLM_SIMPLIFIED_EXPRESSION,
            "non_applicable": None,
        },
        {
            "logical_id": "pred_simplify::algoa::g0001::s522::clean",
            "task_type": "pred_simplify",
            "condition": "clean",
            "state": "exhausted",
            "structured_output": None,
            "effective_expression": None,
            "expression_resolution": None,
            "non_applicable": None,
            "exhausted": {
                "attempt_count": 3,
                "last_error_class": "timeout",
                "attempts": [
                    {
                        "attempt_id": f"attempt-{index}",
                        "attempt_number": index,
                        "status": "failed",
                        "error_class": "timeout",
                        "retryable": True,
                        "attempt_path": str(tmp_path / f"attempt-{index}.json"),
                        "attempt_sha256": _fake_sha(f"attempt-{index}"),
                    }
                    for index in (1, 2, 3)
                ],
            },
        },
    ]
    eq_rows = [
        {
            "logical_id": "equivalence::algoa::g0001::s520::clean",
            "task_type": "equivalence",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {
                "decision": "equivalent",
                "evidence_basis": "symbolic_proof",
            },
            "non_applicable": None,
        },
        {
            "logical_id": "equivalence::algoa::g0001::s521::clean",
            "task_type": "equivalence",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {
                "decision": "undetermined",
                "evidence_basis": "structural_analysis",
            },
            "non_applicable": None,
        },
        {
            "logical_id": "equivalence::algoa::g0001::s522::clean",
            "task_type": "equivalence",
            "condition": "clean",
            "state": "non_applicable",
            "structured_output": None,
            "non_applicable": {
                "reason": "upstream_pred_unavailable",
                "evidence_path": "audit/non_applicable_equivalence.json",
                "evidence_sha256": _fake_sha("eq-na"),
            },
        },
    ]
    structure_rows = [
        {
            "logical_id": "stab_structure::algoa::g0001::s520-s521",
            "task_type": "stab_structure",
            "condition": "clean",
            "state": "frozen",
            "structured_output": {"decision": "same_canonical_structure"},
            "non_applicable": None,
        },
        {
            "logical_id": "stab_structure::algoa::g0001::s520-s522",
            "task_type": "stab_structure",
            "condition": "clean",
            "state": "non_applicable",
            "structured_output": None,
            "non_applicable": {
                "reason": "invalid_seed_or_expression",
                "evidence_path": "audit/non_applicable_structure_520_522.json",
                "evidence_sha256": _fake_sha("structure-na-520-522"),
            },
        },
        {
            "logical_id": "stab_structure::algoa::g0001::s521-s522",
            "task_type": "stab_structure",
            "condition": "clean",
            "state": "non_applicable",
            "structured_output": None,
            "non_applicable": {
                "reason": "invalid_seed_or_expression",
                "evidence_path": "audit/non_applicable_structure_521_522.json",
                "evidence_sha256": _fake_sha("structure-na-521-522"),
            },
        },
    ]

    exhausted_evaluation_key = _fake_sha("pred-exhausted-evaluation-key")
    exhausted_attempts: list[dict[str, object]] = []
    for index in (1, 2, 3):
        attempt_id = f"{exhausted_evaluation_key}.a0{index}"
        attempt_path = tmp_path / "attempts" / f"{attempt_id}.json"
        attempt_payload = {
            "attempt_id": attempt_id,
            "evaluation_key": exhausted_evaluation_key,
            "metadata": {
                "attempt_id": attempt_id,
                "attempt_number": index,
                "evaluation_key": exhausted_evaluation_key,
                "logical_id": "pred_simplify::algoa::g0001::s522::clean",
                "task_type": "pred_simplify",
                "error_class": "timeout",
                "retryable": True,
            },
            "validation": {
                "ok": False,
                "error_class": "timeout",
                "error_message": f"fixture timeout {index}",
            },
        }
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path.write_text(
            json.dumps(attempt_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exhausted_attempts.append(
            {
                "attempt_id": attempt_id,
                "attempt_number": index,
                "status": "failed",
                "error_class": "timeout",
                "retryable": True,
                "attempt_path": str(attempt_path),
                "attempt_sha256": hashlib.sha256(
                    attempt_path.read_bytes()
                ).hexdigest(),
            }
        )
    pred_rows[2]["evaluation_key"] = exhausted_evaluation_key
    pred_rows[2]["exhausted"]["attempts"] = exhausted_attempts

    def _fake_load_frozen_index_rows(*, expected_task_type: str, label: str, **_: object):
        if expected_task_type == "gt_simplify":
            return gt_rows, dummy_info
        if expected_task_type == "pred_simplify":
            pred_info = json.loads(json.dumps(dummy_info))
            pred_info["summary_json"]["state_counts"] = {"frozen": 2, "non_applicable": 0, "exhausted": 1}
            pred_info["index_jsonl"]["row_count"] = 3
            pred_info["plan_jsonl"]["row_count"] = 3
            return pred_rows, pred_info
        if expected_task_type == "equivalence":
            info = json.loads(json.dumps(dummy_info))
            info["index_jsonl"]["row_count"] = 3
            info["plan_jsonl"]["row_count"] = 3
            info["summary_json"]["state_counts"] = {"frozen": 2, "non_applicable": 1}
            return eq_rows, info
        if expected_task_type == "stab_structure":
            info = json.loads(json.dumps(dummy_info))
            info["index_jsonl"]["row_count"] = 3
            info["plan_jsonl"]["row_count"] = 3
            info["summary_json"]["state_counts"] = {"frozen": 1, "non_applicable": 2}
            return structure_rows, info
        raise AssertionError(label)

    monkeypatch.setattr(
        aggregate_module,
        "_validate_clean_numeric_preparation_report",
        lambda *args, **kwargs: {"path": str(tmp_path / "numeric_prep.json"), "sha256": _fake_sha("numeric-prep"), "row_count": 3},
    )
    monkeypatch.setattr(
        aggregate_module,
        "_validate_eff_preparation_report",
        lambda *args, **kwargs: {"path": str(tmp_path / "eff_prep.json"), "sha256": _fake_sha("eff-prep"), "row_count": 3},
    )
    monkeypatch.setattr(aggregate_module, "_load_frozen_index_rows", _fake_load_frozen_index_rows)

    with pytest.raises(AggregateCleanMetricsError, match="exhausted=1"):
        aggregate_module.aggregate_clean_metrics(
            numeric_csv=paths["numeric_csv"],
            clean_numeric_preparation_report_json=tmp_path / "inputs/numeric_prep.json",
            eff_csv=paths["eff_csv"],
            eff_preparation_report_json=paths["eff_preparation_report_json"],
            gt_plan_jsonl=tmp_path / "inputs/gt.plan.jsonl",
            gt_summary_json=tmp_path / "inputs/gt.summary.json",
            gt_index_jsonl=tmp_path / "inputs/gt.index.jsonl",
            pred_plan_jsonl=paths["simplify_plan_jsonl"],
            pred_summary_json=tmp_path / "inputs/pred.summary.json",
            pred_index_jsonl=tmp_path / "inputs/pred.index.jsonl",
            equivalence_plan_jsonl=tmp_path / "inputs/eq.plan.jsonl",
            equivalence_summary_json=tmp_path / "inputs/eq.summary.json",
            equivalence_index_jsonl=tmp_path / "inputs/eq.index.jsonl",
            structure_plan_jsonl=tmp_path / "inputs/structure.plan.jsonl",
            structure_summary_json=tmp_path / "inputs/structure.summary.json",
            structure_index_jsonl=tmp_path / "inputs/structure.index.jsonl",
            evidence_jsonl=paths["evidence_jsonl"],
            clean_run_csv=clean_run_csv,
            task_stability_csv=task_stability_csv,
            algorithm_csv=algorithm_csv,
            report_json=report_json,
            expected_runs=3,
            expected_algorithms=1,
            expected_datasets=1,
        )
