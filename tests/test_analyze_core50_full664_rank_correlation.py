from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "analyze_core50_full664_rank_correlation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_core50_full664_rank_correlation",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_inputs(tmp_path: Path):
    module = _load_module()
    datasets = [f"sim-datasets-data/family/d{index}" for index in range(1, 4)]
    core = datasets[:2]

    full7_rows: list[dict[str, object]] = []
    for algorithm_index, algorithm in enumerate(module.THREE_SEED_ALGORITHMS):
        for dataset_index, dataset_dir in enumerate(datasets):
            for seed_offset, seed in enumerate((520, 521, 522)):
                score = algorithm_index + dataset_index + seed_offset
                full7_rows.append(
                    {
                        "algorithm": algorithm,
                        "dataset_rel": dataset_dir,
                        "seed": seed,
                        "id_log_nmse_penalized": score - 0.5,
                        "ood_log_nmse_penalized": score,
                        "source_group": "fixture",
                    }
                )

    probe2_rows: list[dict[str, object]] = []
    for algorithm_index, algorithm in enumerate(module.PROBE2_ALGORITHMS):
        for dataset_index, dataset_dir in enumerate(datasets):
            id_nmse: object = 10 ** (8 + algorithm_index + dataset_index)
            ood_nmse: object = 10 ** (9 + algorithm_index + dataset_index)
            if algorithm == "pysr" and dataset_index == 2:
                id_nmse = ""
                ood_nmse = ""
            probe2_rows.append(
                {
                    "method": algorithm,
                    "dataset_dir": dataset_dir,
                    "seed": 1314,
                    "id_nmse": id_nmse,
                    "ood_nmse": ood_nmse,
                }
            )

    core_rows = [{"dataset_dir": dataset_dir} for dataset_dir in core]
    full7_path = tmp_path / "full7.csv"
    probe2_path = tmp_path / "probe2.csv"
    core_path = tmp_path / "core.csv"
    _write_csv(full7_path, full7_rows)
    _write_csv(probe2_path, probe2_rows)
    _write_csv(core_path, core_rows)
    return module, full7_path, probe2_path, core_path


def test_seed_median_and_split_level_missing_penalty(tmp_path: Path) -> None:
    module, full7_path, probe2_path, core_path = _make_inputs(tmp_path)
    runs = [
        *module.load_full7_runs(full7_path),
        *module.load_probe2_runs(probe2_path),
    ]
    core_dirs = {
        module._normalize_dataset_dir(row["dataset_dir"])
        for row in _read_csv(core_path)
    }
    dataset_scores = module.build_dataset_algorithm_scores(runs, core_dirs)

    dso_d1 = next(
        row
        for row in dataset_scores
        if row["algorithm"] == "dso" and row["dataset_dir"].endswith("/d1")
    )
    assert dso_d1["ood_log_nmse_seed_median_penalized"] == pytest.approx(1.0)
    assert dso_d1["id_log_nmse_seed_median_penalized"] == pytest.approx(0.5)
    assert dso_d1["aggregate_id_ood_log_score"] == pytest.approx(0.75)
    assert dso_d1["seed_count"] == 3

    pysr_d3 = next(
        row
        for row in dataset_scores
        if row["algorithm"] == "pysr" and row["dataset_dir"].endswith("/d3")
    )
    assert pysr_d3["id_log_nmse_seed_median_penalized"] == 12.0
    assert pysr_d3["ood_log_nmse_seed_median_penalized"] == 12.0
    assert pysr_d3["seed_count"] == 1


def test_end_to_end_writes_auditable_outputs(tmp_path: Path) -> None:
    module, full7_path, probe2_path, core_path = _make_inputs(tmp_path)
    output_dir = tmp_path / "output"
    summary = module.analyze(
        full7_runs_path=full7_path,
        probe2_runs_path=probe2_path,
        core50_manifest_path=core_path,
        output_dir=output_dir,
        expected_dataset_count=3,
        expected_core_count=2,
        repo_root=tmp_path,
    )

    assert summary["coverage"]["run_rows"] == 69
    assert summary["coverage"]["dataset_count"] == 3
    assert summary["coverage"]["core_dataset_count"] == 2
    assert len(summary["algorithm_scores"]) == 9
    assert len(summary["correlations"]) == 10
    assert summary["sources"]["full7_runs"]["path"] == "full7.csv"
    assert all(not Path(value).is_absolute() for value in summary["outputs"].values())

    dataset_rows = _read_csv(output_dir / "dataset_algorithm_scores.csv")
    algorithm_rows = _read_csv(output_dir / "algorithm_scores_and_ranks.csv")
    correlation_rows = _read_csv(output_dir / "correlation_metrics.csv")
    assert len(dataset_rows) == 27
    assert len(algorithm_rows) == 9
    assert len(correlation_rows) == 10
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "correlation_summary.json").is_file()


def test_mismatched_algorithm_dataset_grid_is_rejected(
    tmp_path: Path,
) -> None:
    module, full7_path, probe2_path, core_path = _make_inputs(tmp_path)
    rows = _read_csv(probe2_path)
    _write_csv(probe2_path, rows[:-1])

    with pytest.raises(ValueError, match="数据集键空间不一致|run 数量错误"):
        module.analyze(
            full7_runs_path=full7_path,
            probe2_runs_path=probe2_path,
            core50_manifest_path=core_path,
            output_dir=tmp_path / "output",
            expected_dataset_count=3,
            expected_core_count=2,
            repo_root=tmp_path,
        )
