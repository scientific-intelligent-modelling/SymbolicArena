from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_noise_six_axis import (
    AggregateNoiseSixAxisError,
    aggregate_noise_six_axis,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_context = (
        gzip.open(path, mode="wt", encoding="utf-8")
        if gzip_output
        else path.open(mode="w", encoding="utf-8")
    )
    with handle_context as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bundle(tmp_path: Path) -> dict[str, Path]:
    condition = "noise001"
    raw_path = tmp_path / "raw.jsonl.gz"
    raw_rows = []
    numeric_rows = []
    pred_plan = []
    pred_index = []
    eq_plan = []
    eq_index = []
    result_shas: dict[int, str] = {}
    for seed in (520, 521, 522):
        raw_text = json.dumps({"equation": f"x + {seed - 520}"}, sort_keys=True)
        result_sha = hashlib.sha256(raw_text.encode()).hexdigest()
        result_shas[seed] = result_sha
        raw_rows.append(
            {
                "source": {
                    "algorithm": "Algo",
                    "dataset_id": "D1",
                    "seed": seed,
                    "noise_tag": condition,
                    "task_id": f"algo_s{seed}_{condition}_g0001",
                },
                "result": {"raw_text": raw_text, "sha256": result_sha},
            }
        )
        logical_key = f"Algo::D1::s{seed}::{condition}"
        task_id = f"algo_s{seed}_{condition}_g0001"
        numeric_rows.append(
            {
                "logical_key": logical_key,
                "algorithm": "Algo",
                "dataset_id": "D1",
                "seed": seed,
                "noise_tag": condition,
                "task_id": task_id,
                "host": "host1",
                "result_sha256": result_sha,
                "evaluation_status": "valid",
                "valid_output": "true",
                "evaluation_path": "canonical_replay.v1",
                "id_quality": "0.8",
                "ood_quality": "0.6",
                "replay_error": "",
            }
        )
        pred_eval = hashlib.sha256(f"pred-{seed}".encode()).hexdigest()
        pred_id = f"pred_simplify::algo::g0001::s{seed}::{condition}"
        pred_plan.append(
            {
                "logical_id": pred_id,
                "task_type": "pred_simplify",
                "condition": condition,
                "evaluation_key": pred_eval,
                "dependencies": [],
                "request": {
                    "algorithm": "algo",
                    "algorithm_slug": "algo",
                    "dataset_id": "D1",
                    "dataset_index": "g0001",
                    "seed": seed,
                    "noise_tag": condition,
                    "task_id": task_id,
                    "result_raw_sha256": result_sha,
                    "original_expression": f"x + {seed - 520}",
                    "expression": f"x + {seed - 520}",
                },
            }
        )
        pred_index.append(
            {
                "logical_id": pred_id,
                "task_type": "pred_simplify",
                "condition": condition,
                "evaluation_key": pred_eval,
                "state": "frozen",
                "effective_expression": f"x + {seed - 520}",
                "structured_output": {
                    "outcome": "unchanged",
                    "simplified_expression": f"x + {seed - 520}",
                    "equivalence_assessment": "preserved",
                },
            }
        )
        eq_eval = hashlib.sha256(f"eq-{seed}".encode()).hexdigest()
        eq_id = f"equivalence::algo::g0001::s{seed}::{condition}"
        eq_plan.append(
            {
                "logical_id": eq_id,
                "task_type": "equivalence",
                "condition": condition,
                "evaluation_key": eq_eval,
                "dependencies": ["gt-eval", pred_eval],
                "request": {
                    "algorithm_slug": "algo",
                    "dataset_id": "D1",
                    "dataset_index": "g0001",
                    "seed": seed,
                    "noise_tag": condition,
                    "ground_truth_frozen_evaluation_key": "gt-eval",
                    "prediction_frozen_evaluation_key": pred_eval,
                    "prediction_result_sha256": result_sha,
                    "prediction_logical_id": pred_id,
                    "ground_truth_logical_id": "gt_simplify::D1",
                },
            }
        )
        eq_index.append(
            {
                "logical_id": eq_id,
                "task_type": "equivalence",
                "condition": condition,
                "evaluation_key": eq_eval,
                "state": "frozen",
                "structured_output": {
                    "decision": "equivalent" if seed == 520 else "not_equivalent",
                    "evidence_basis": "structural_analysis",
                },
            }
        )

    gt_plan = [
        {
            "logical_id": "gt_simplify::D1",
            "task_type": "gt_simplify",
            "condition": "clean",
            "evaluation_key": "gt-eval",
            "dependencies": [],
            "request": {
                "dataset_id": "D1",
                "original_expression": "x",
                "expression": "x",
            },
        }
    ]
    gt_index = [
        {
            "logical_id": "gt_simplify::D1",
            "task_type": "gt_simplify",
            "condition": "clean",
            "evaluation_key": "gt-eval",
            "state": "frozen",
            "effective_expression": "x",
            "structured_output": {
                "outcome": "unchanged",
                "simplified_expression": "x",
                "equivalence_assessment": "preserved",
            },
        }
    ]
    structure_plan = []
    structure_index = []
    for left, right, decision in (
        (520, 521, "same_canonical_structure"),
        (520, 522, "different_structure"),
        (521, 522, "mathematically_equivalent"),
    ):
        logical_id = f"stab_structure::algo::g0001::s{left}-s{right}::{condition}"
        evaluation_key = hashlib.sha256(f"structure-{left}-{right}".encode()).hexdigest()
        dependencies = [
            next(row["evaluation_key"] for row in pred_plan if row["request"]["seed"] == left),
            next(row["evaluation_key"] for row in pred_plan if row["request"]["seed"] == right),
        ]
        structure_plan.append(
            {
                "logical_id": logical_id,
                "task_type": "stab_structure",
                "condition": condition,
                "evaluation_key": evaluation_key,
                "dependencies": dependencies,
                "request": {
                    "algorithm_slug": "algo",
                    "dataset_id": "D1",
                    "dataset_index": "g0001",
                    "noise_tag": condition,
                    "seed_left": left,
                    "seed_right": right,
                    "prediction_a_result_sha256": result_shas[left],
                    "prediction_b_result_sha256": result_shas[right],
                },
            }
        )
        structure_index.append(
            {
                "logical_id": logical_id,
                "task_type": "stab_structure",
                "condition": condition,
                "evaluation_key": evaluation_key,
                "state": "frozen",
                "structured_output": {"decision": decision},
            }
        )

    paths = {
        "raw_results_jsonl": raw_path,
        "numeric_csv": tmp_path / "numeric.csv",
        "pred_plan_jsonl": tmp_path / "pred_plan.jsonl",
        "pred_index_jsonl": tmp_path / "pred_index.jsonl",
        "gt_plan_jsonl": tmp_path / "gt_plan.jsonl",
        "gt_index_jsonl": tmp_path / "gt_index.jsonl",
        "equivalence_plan_jsonl": tmp_path / "eq_plan.jsonl",
        "equivalence_index_jsonl": tmp_path / "eq_index.jsonl",
        "structure_plan_jsonl": tmp_path / "structure_plan.jsonl",
        "structure_index_jsonl": tmp_path / "structure_index.jsonl",
        "eff_180min_csv": tmp_path / "eff.csv",
        "six_axis_csv": tmp_path / "out/six_axis.csv",
        "run_formulas_csv": tmp_path / "out/run_formulas.csv",
        "task_stability_csv": tmp_path / "out/task_stability.csv",
        "output_eff_180min_csv": tmp_path / "out/eff.csv",
        "report_json": tmp_path / "out/report.json",
    }
    _write_jsonl(raw_path, raw_rows, gzip_output=True)
    _write_csv(paths["numeric_csv"], numeric_rows)
    for key, rows in (
        ("pred_plan_jsonl", pred_plan),
        ("pred_index_jsonl", pred_index),
        ("gt_plan_jsonl", gt_plan),
        ("gt_index_jsonl", gt_index),
        ("equivalence_plan_jsonl", eq_plan),
        ("equivalence_index_jsonl", eq_index),
        ("structure_plan_jsonl", structure_plan),
        ("structure_index_jsonl", structure_index),
    ):
        _write_jsonl(paths[key], rows)
    eff_rows = []
    for minute in range(1, 181):
        eff_rows.append(
            {
                "condition": condition,
                "algorithm": "Algo",
                "minute": minute,
                "expected_run_count": 3,
                "available_run_count": 3,
                "coverage_rate": "1",
                "mean_id_quality": "0.5",
                "mean_ood_quality": "0.4",
                "mean_quality": "0.45",
                "mean_relative_progress": "0.75",
                "cumulative_eff_score": "75",
                "base_run_count": 3,
                "targeted_overlay_run_count": 0,
                "trajectory_basis": "observed_numeric_best_so_far_native_snapshot.v1",
                "formal_ready": "false",
            }
        )
    _write_csv(paths["eff_180min_csv"], eff_rows)
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_aggregate_noise_full_six_axis_and_marks_eff_supplementary(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)

    report = aggregate_noise_six_axis(
        condition="noise001",
        expected_algorithms=1,
        expected_datasets=1,
        **paths,
    )

    six_axis = _read_csv(paths["six_axis_csv"])
    runs = _read_csv(paths["run_formulas_csv"])
    tasks = _read_csv(paths["task_stability_csv"])
    eff = _read_csv(paths["output_eff_180min_csv"])
    assert len(six_axis) == 1
    assert len(runs) == 3
    assert len(tasks) == 1
    assert len(eff) == 180
    assert float(six_axis[0]["ID"]) == pytest.approx(80.0)
    assert float(six_axis[0]["OOD"]) == pytest.approx(60.0)
    assert float(six_axis[0]["EFF"]) == pytest.approx(75.0)
    assert 0.0 <= float(six_axis[0]["SYM"]) <= 100.0
    assert runs[0]["original_prediction_expression"] == "x + 0"
    assert runs[0]["effective_ground_truth_expression"] == "x"
    assert runs[0]["m_eff_basis"] == "algorithm_mean_from_supplementary_trajectory.v1"
    assert tasks[0]["pair_520_521"] == "same_canonical_structure"
    assert eff[-1]["trajectory_basis"] == "observed_numeric_best_so_far_native_snapshot.v1"
    assert eff[-1]["formal_ready"] == "false"
    assert report["formal_ready"] is False
    assert report["summary"]["canonical_replay_unavailable_count"] == 0


def test_rejects_replay_unavailable_instead_of_scoring_zero(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    rows = _read_csv(paths["numeric_csv"])
    rows[0]["evaluation_status"] = "replay_unavailable"
    rows[0]["replay_error"] = "missing artifact"
    _write_csv(paths["numeric_csv"], rows)

    with pytest.raises(AggregateNoiseSixAxisError, match="canonical replay"):
        aggregate_noise_six_axis(
            condition="noise001",
            expected_algorithms=1,
            expected_datasets=1,
            **paths,
        )
