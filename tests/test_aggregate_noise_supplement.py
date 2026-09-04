from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_noise_supplement import (
    AggregateNoiseSupplementError,
    DEFAULT_EXPECTED_ALGORITHMS,
    DEFAULT_EXPECTED_DATASETS,
    DEFAULT_EXPECTED_RUNS_PER_CONDITION,
    aggregate_noise_supplement,
)


SEEDS = (520, 521, 522)


def test_default_contract_is_exactly_6750_runs_and_30_supplement_rows() -> None:
    assert DEFAULT_EXPECTED_RUNS_PER_CONDITION == 2250
    assert DEFAULT_EXPECTED_ALGORITHMS == 15
    assert DEFAULT_EXPECTED_DATASETS == 50
    assert DEFAULT_EXPECTED_RUNS_PER_CONDITION * 3 == 6750
    assert DEFAULT_EXPECTED_ALGORITHMS * 2 == 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _clean_row(algorithm: str, dataset: str, seed: int, quality: float) -> dict[str, object]:
    return {
        "logical_key": f"{algorithm}::{dataset}::s{seed}::clean",
        "algorithm": algorithm,
        "dataset_id": dataset,
        "seed": seed,
        "noise_tag": "clean",
        "task_id": f"{algorithm.lower()}_s{seed}_clean_{dataset}",
        "host": "host1",
        "valid_output": "true",
        "pred_state": "frozen",
        "equivalence_state": "frozen",
        "id_quality": quality,
        "ood_quality": quality / 2,
        "m_eff": 0.25,
        "equivalence_decision": "not_equivalent",
        "equivalent": "false",
        "tree_similarity": 0.5,
        "variable_f1": 0.5,
        "operator_f1": 0.5,
        "m_sym": 0.4,
        "reference_complexity": 10,
        "predicted_complexity": 12,
        "m_min": 0.8,
        "gt_logical_id": f"gt_simplify::{dataset}::v2",
        "pred_logical_id": f"pred_simplify::{algorithm.lower()}::{dataset}::s{seed}::clean",
        "equivalence_logical_id": f"equivalence::{algorithm.lower()}::{dataset}::s{seed}::clean",
    }


def _numeric_row(
    algorithm: str,
    dataset: str,
    seed: int,
    condition: str,
    quality: float,
) -> dict[str, object]:
    return {
        "logical_key": f"{algorithm}::{dataset}::s{seed}::{condition}",
        "algorithm": algorithm,
        "dataset_id": dataset,
        "seed": seed,
        "noise_tag": condition,
        "task_id": f"{algorithm.lower()}_s{seed}_{condition}_{dataset}",
        "host": "host2",
        "result_sha256": "a" * 64,
        "evaluation_status": "valid",
        "valid_output": "true",
        "invalid_reason": "",
        "replay_error": "",
        "formula_source": "canonical_artifact",
        "evaluation_path": "canonical_replay.v1",
        "id_nmse": 0.1,
        "ood_nmse": 0.2,
        "id_quality": quality,
        "ood_quality": quality / 2,
    }


def _write_equivalence_bundle(
    root: Path,
    *,
    condition: str,
    algorithms: tuple[str, ...],
    datasets: tuple[str, ...],
    non_applicable_key: tuple[str, str, int] | None = None,
    successor_key: tuple[str, str, int] | None = None,
) -> tuple[Path, Path, Path]:
    plan_path = root / f"{condition}_equivalence_full_plan.jsonl"
    index_path = root / f"{condition}_equivalence_frozen_index.jsonl"
    summary_path = root / f"{condition}_equivalence_summary.json"
    plan_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        for dataset_index, dataset in enumerate(datasets, start=1):
            for seed in SEEDS:
                run_key = (algorithm, dataset, seed)
                logical_id = (
                    f"equivalence::{algorithm.lower()}::g{dataset_index:04d}::s{seed}::{condition}"
                )
                if run_key == successor_key:
                    logical_id += "::v2"
                plan_rows.append(
                    {
                        "evaluation_key": hashlib.sha256(logical_id.encode()).hexdigest(),
                        "logical_id": logical_id,
                        "task_type": "equivalence",
                        "condition": condition,
                        "request": {
                            "algorithm": algorithm,
                            "dataset_id": dataset,
                            "dataset_index": f"g{dataset_index:04d}",
                            "seed": seed,
                            "noise_tag": condition,
                            "prediction_task_id": f"{algorithm.lower()}_s{seed}_{condition}_{dataset}",
                            "prediction_valid_output": True,
                            "prediction_logical_id": (
                                f"pred_simplify::{algorithm.lower()}::g{dataset_index:04d}::s{seed}::{condition}"
                            ),
                        },
                    }
                )
    _write_jsonl(plan_path, plan_rows)
    plan_sha = _sha256(plan_path)

    index_rows: list[dict[str, object]] = []
    for plan_row in plan_rows:
        request = plan_row["request"]
        assert isinstance(request, dict)
        key = (str(request["algorithm"]), str(request["dataset_id"]), int(request["seed"]))
        non_applicable = key == non_applicable_key
        index_rows.append(
            {
                "plan_sha256": plan_sha,
                "evaluation_key": plan_row["evaluation_key"],
                "logical_id": plan_row["logical_id"],
                "task_type": "equivalence",
                "condition": condition,
                "state": "non_applicable" if non_applicable else "frozen",
                "structured_output": None
                if non_applicable
                else {
                    "decision": "equivalent" if int(request["seed"]) == 520 else "not_equivalent",
                    "evidence_basis": "mixed",
                    "confidence": 0.9,
                },
                "non_applicable": {"reason": "upstream_gt_unavailable"}
                if non_applicable
                else None,
            }
        )
    _write_jsonl(index_path, index_rows)
    state_counts = {
        "frozen": sum(row["state"] == "frozen" for row in index_rows),
        "non_applicable": sum(row["state"] == "non_applicable" for row in index_rows),
    }
    summary = {
        "status": "ok",
        "plan_jsonl": str(plan_path),
        "plan_sha256": plan_sha,
        "output_jsonl": str(index_path),
        "output_sha256": _sha256(index_path),
        "row_count": len(index_rows),
        "state_counts": state_counts,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return plan_path, index_path, summary_path


def _fixture(tmp_path: Path) -> dict[str, object]:
    algorithms = ("AlgoA", "AlgoB")
    datasets = ("D1", "D2")
    clean_path = tmp_path / "clean.csv"
    noise001_path = tmp_path / "noise001.csv"
    noise005_path = tmp_path / "noise005.csv"
    _write_csv(
        clean_path,
        [_clean_row(a, d, s, 0.8 if a == "AlgoA" else 0.4) for a in algorithms for d in datasets for s in SEEDS],
    )
    _write_csv(
        noise001_path,
        [_numeric_row(a, d, s, "noise001", 0.6 if a == "AlgoA" else 0.2) for a in algorithms for d in datasets for s in SEEDS],
    )
    _write_csv(
        noise005_path,
        [_numeric_row(a, d, s, "noise005", 0.4 if a == "AlgoA" else 0.1) for a in algorithms for d in datasets for s in SEEDS],
    )
    eq001 = _write_equivalence_bundle(
        tmp_path, condition="noise001", algorithms=algorithms, datasets=datasets
    )
    eq005 = _write_equivalence_bundle(
        tmp_path,
        condition="noise005",
        algorithms=algorithms,
        datasets=datasets,
        non_applicable_key=("AlgoB", "D2", 522),
    )
    return {
        "clean_run_csv": clean_path,
        "noise001_numeric_csv": noise001_path,
        "noise005_numeric_csv": noise005_path,
        "noise001_equivalence_plan_jsonl": eq001[0],
        "noise001_equivalence_index_jsonl": eq001[1],
        "noise001_equivalence_summary_json": eq001[2],
        "noise005_equivalence_plan_jsonl": eq005[0],
        "noise005_equivalence_index_jsonl": eq005[1],
        "noise005_equivalence_summary_json": eq005[2],
        "all_conditions_csv": tmp_path / "out" / "all.csv",
        "noise_supplement_csv": tmp_path / "out" / "supplement.csv",
        "report_json": tmp_path / "out" / "report.json",
        "expected_runs_per_condition": 12,
        "expected_algorithms": 2,
        "expected_datasets": 2,
    }


def test_aggregate_noise_supplement_happy_path_and_non_applicable(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    report = aggregate_noise_supplement(**arguments)

    all_rows = list(csv.DictReader(Path(arguments["all_conditions_csv"]).open()))
    supplement = list(csv.DictReader(Path(arguments["noise_supplement_csv"]).open()))
    assert len(all_rows) == 36
    assert len(supplement) == 4
    assert {row["noise_tag"] for row in all_rows} == {"clean", "noise001", "noise005"}
    assert all(row["m_eff"] == "" for row in all_rows if row["noise_tag"] != "clean")
    assert all(row["m_sym"] == "" for row in all_rows if row["noise_tag"] != "clean")

    algo_a_001 = next(row for row in supplement if row["algorithm"] == "AlgoA" and row["noise_tag"] == "noise001")
    assert float(algo_a_001["mean_id_quality"]) == pytest.approx(0.6)
    assert float(algo_a_001["id_quality_retention"]) == pytest.approx(0.75)
    assert float(algo_a_001["id_quality_drop"]) == pytest.approx(0.25)
    assert float(algo_a_001["equivalence_coverage"]) == pytest.approx(1.0)
    assert float(algo_a_001["equivalence_rate"]) == pytest.approx(1 / 3)

    algo_b_005 = next(row for row in supplement if row["algorithm"] == "AlgoB" and row["noise_tag"] == "noise005")
    assert algo_b_005["equivalence_non_applicable_count"] == "1"
    assert algo_b_005["equivalence_applicable_count"] == "5"
    assert float(algo_b_005["equivalence_coverage"]) == pytest.approx(1.0)
    assert report["summary"]["all_conditions_row_count"] == 36
    assert report["summary"]["noise_supplement_row_count"] == 4
    for section in ("inputs", "outputs"):
        assert all(len(item["sha256"]) == 64 for item in report[section].values())


def test_aggregate_noise_supplement_accepts_successor_logical_id(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    algorithms = ("AlgoA", "AlgoB")
    datasets = ("D1", "D2")
    successor_bundle = _write_equivalence_bundle(
        tmp_path / "successor",
        condition="noise001",
        algorithms=algorithms,
        datasets=datasets,
        successor_key=("AlgoA", "D1", 520),
    )
    arguments["noise001_equivalence_plan_jsonl"] = successor_bundle[0]
    arguments["noise001_equivalence_index_jsonl"] = successor_bundle[1]
    arguments["noise001_equivalence_summary_json"] = successor_bundle[2]

    report = aggregate_noise_supplement(**arguments)

    assert report["summary"]["all_conditions_row_count"] == 36


def test_aggregate_noise_supplement_rejects_duplicate_numeric_key(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    noise_path = Path(arguments["noise001_numeric_csv"])
    rows = list(csv.DictReader(noise_path.open()))
    rows[-1] = dict(rows[0])
    _write_csv(noise_path, rows)

    with pytest.raises(AggregateNoiseSupplementError, match="重复"):
        aggregate_noise_supplement(**arguments)


def test_aggregate_noise_supplement_rejects_replay_unavailable_instead_of_scoring_zero(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    noise_path = Path(arguments["noise001_numeric_csv"])
    rows = list(csv.DictReader(noise_path.open()))
    rows[0].update(
        {
            "evaluation_status": "replay_unavailable",
            "valid_output": "",
            "id_quality": "",
            "ood_quality": "",
        }
    )
    _write_csv(noise_path, rows)

    with pytest.raises(AggregateNoiseSupplementError, match="replay_unavailable"):
        aggregate_noise_supplement(**arguments)


def test_aggregate_noise_supplement_rejects_legacy_blank_status(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    noise_path = Path(arguments["noise001_numeric_csv"])
    rows = list(csv.DictReader(noise_path.open()))
    rows[0]["evaluation_status"] = ""
    _write_csv(noise_path, rows)

    with pytest.raises(AggregateNoiseSupplementError, match="evaluation_status"):
        aggregate_noise_supplement(**arguments)


def test_aggregate_noise_supplement_rejects_noncanonical_evaluation_path(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    noise_path = Path(arguments["noise001_numeric_csv"])
    rows = list(csv.DictReader(noise_path.open()))
    rows[0]["evaluation_path"] = "frozen_metrics.v1"
    _write_csv(noise_path, rows)

    with pytest.raises(AggregateNoiseSupplementError, match="evaluation_path"):
        aggregate_noise_supplement(**arguments)


def test_aggregate_noise_supplement_rejects_invalid_equivalence_enum(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    index_path = Path(arguments["noise001_equivalence_index_jsonl"])
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["structured_output"]["decision"] = "maybe"
    _write_jsonl(index_path, rows)
    summary_path = Path(arguments["noise001_equivalence_summary_json"])
    summary = json.loads(summary_path.read_text())
    summary["output_sha256"] = _sha256(index_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AggregateNoiseSupplementError, match="decision"):
        aggregate_noise_supplement(**arguments)


def test_cli_prints_summary(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    option_names = {
        "clean_run_csv": "--clean-run-csv",
        "noise001_numeric_csv": "--noise001-numeric-csv",
        "noise005_numeric_csv": "--noise005-numeric-csv",
        "noise001_equivalence_plan_jsonl": "--noise001-equivalence-plan-jsonl",
        "noise001_equivalence_index_jsonl": "--noise001-equivalence-index-jsonl",
        "noise001_equivalence_summary_json": "--noise001-equivalence-summary-json",
        "noise005_equivalence_plan_jsonl": "--noise005-equivalence-plan-jsonl",
        "noise005_equivalence_index_jsonl": "--noise005-equivalence-index-jsonl",
        "noise005_equivalence_summary_json": "--noise005-equivalence-summary-json",
        "all_conditions_csv": "--all-conditions-csv",
        "noise_supplement_csv": "--noise-supplement-csv",
        "report_json": "--report-json",
        "expected_runs_per_condition": "--expected-runs-per-condition",
        "expected_algorithms": "--expected-algorithms",
        "expected_datasets": "--expected-datasets",
    }
    command = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_noise_supplement",
    ]
    for key, option in option_names.items():
        command.extend((option, str(arguments[key])))
    command.append("--print-summary")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["all_conditions_row_count"] == 36
