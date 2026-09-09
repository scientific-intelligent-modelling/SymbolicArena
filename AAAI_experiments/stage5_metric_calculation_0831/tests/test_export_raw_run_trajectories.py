from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest


def _quality_columns(value: float) -> dict[str, str]:
    return {f"q_{minute:04d}": str(value) for minute in range(1, 181)}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clean_row(seed: int, value: float) -> dict[str, object]:
    return {
        "logical_key": f"Demo::TaskA::s{seed}::clean",
        "algorithm": "Demo",
        "dataset_id": "TaskA",
        "seed": seed,
        "task_id": f"demo_s{seed}_clean_g0001",
        "host": "iaaccn22",
        "noise_tag": "clean",
        "m_eff": "1",
        "best_quality": str(value),
        **_quality_columns(value),
    }


def _noise_record(condition: str, seed: int, value: float, *, source_path: str) -> dict:
    snapshots = []
    for minute in range(1, 181):
        snapshots.append(
            {
                "minute": minute,
                "id_nmse": value,
                "ood_nmse": value * 2,
            }
        )
    return {
        "source": {
            "algorithm": "Demo",
            "dataset_id": "TaskA",
            "seed": seed,
            "noise_tag": condition,
            "task_id": f"demo_s{seed}_{condition}_g0001",
            "host": "iaaccn22",
            "batch": "fixture",
            "path": source_path,
        },
        "snapshots": snapshots,
    }


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exports_reproducible_run_level_clean_and_noise_tables(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
        export_raw_run_trajectories as module,
    )

    current = tmp_path / "current.csv"
    legacy = tmp_path / "legacy.csv"
    base = tmp_path / "noise_base.jsonl.gz"
    overlay = tmp_path / "noise_overlay.jsonl.gz"
    output = tmp_path / "export"

    _write_csv(legacy, [_clean_row(520, 0.2), _clean_row(521, 0.3)])
    _write_csv(current, [_clean_row(520, 0.4)])
    _write_jsonl_gz(
        base,
        [
            _noise_record(condition, seed, 1.0, source_path=f"/base/{condition}/{seed}")
            for condition in ("noise001", "noise005")
            for seed in (520, 521)
        ],
    )
    _write_jsonl_gz(
        overlay,
        [_noise_record("noise005", 521, 0.01, source_path="/overlay/noise005/521")],
    )

    reference_paths: dict[str, Path] = {}
    clean_reference = []
    for minute in range(1, 181):
        clean_reference.append(
            {
                "condition": "clean",
                "algorithm": "Demo",
                "minute": minute,
                "mean_quality": 0.35,
                "mean_relative_progress": 1.0,
                "cumulative_eff_score": 100.0,
                "current_path_run_count": 1,
                "fallback_run_count": 1,
            }
        )
    reference_paths["clean"] = tmp_path / "clean_eff_180min.csv"
    _write_csv(reference_paths["clean"], clean_reference)
    for condition, nmse_values, base_count, overlay_count in (
        ("noise001", (1.0, 1.0), 2, 0),
        ("noise005", (1.0, 0.01), 1, 1),
    ):
        id_values = [module.phi_nmse(value) for value in nmse_values]
        ood_values = [module.phi_nmse(value * 2) for value in nmse_values]
        quality_values = [
            (id_quality + ood_quality) / 2
            for id_quality, ood_quality in zip(id_values, ood_values)
        ]
        rows = []
        for minute in range(1, 181):
            rows.append(
                {
                    "condition": condition,
                    "algorithm": "Demo",
                    "minute": minute,
                    "mean_id_quality": sum(id_values) / 2,
                    "mean_ood_quality": sum(ood_values) / 2,
                    "mean_quality": sum(quality_values) / 2,
                    "mean_relative_progress": 1.0,
                    "cumulative_eff_score": 100.0,
                    "base_run_count": base_count,
                    "targeted_overlay_run_count": overlay_count,
                }
            )
        reference_paths[condition] = tmp_path / f"{condition}_eff_180min.csv"
        _write_csv(reference_paths[condition], rows)

    contract = module.GridContract(
        expected_algorithm_count=1,
        expected_dataset_count=1,
        expected_seeds=(520, 521),
        expected_runs_per_condition=2,
        expected_runs_per_algorithm=2,
    )
    manifest = module.export_raw_run_trajectories(
        current_clean_path=current,
        legacy_clean_path=legacy,
        noise_base_paths=[base],
        noise_overlay_paths=[overlay],
        output_dir=output,
        contract=contract,
        reference_eff_paths=reference_paths,
    )

    clean_rows = _read_csv_gz(output / "clean_run_trajectories_180min.csv.gz")
    assert len(clean_rows) == 2
    assert {row["source_tier"] for row in clean_rows} == {
        "current_canonical",
        "legacy_fallback",
    }
    assert clean_rows[0]["q_0180"]
    assert float(next(row for row in clean_rows if row["seed"] == "520")["q_star"]) == 0.4

    noise_rows = _read_csv_gz(output / "noise005_run_trajectories_180min.csv.gz")
    overlaid = next(row for row in noise_rows if row["seed"] == "521")
    assert overlaid["source_tier"] == "targeted_overlay"
    assert overlaid["id_q_0001"]
    assert overlaid["ood_q_0180"]
    assert float(overlaid["q_0001"]) > float(
        next(row for row in noise_rows if row["seed"] == "520")["q_0001"]
    )
    reverse_rows = module.reverse_aggregate_rows(noise_rows, condition="noise005")
    first_minute = next(row for row in reverse_rows if row["minute"] == 1)
    assert first_minute["mean_quality"] == pytest.approx(
        sum(float(row["q_0001"]) for row in noise_rows) / 2
    )
    assert first_minute["targeted_overlay_run_count"] == 1

    assert manifest["status"] == "ok"
    assert manifest["forbidden_source_check"]["passed"] is True
    assert manifest["conditions"]["clean"]["record_count"] == 2
    assert manifest["conditions"]["noise005"]["source_tier_counts"] == {
        "noise_base": 1,
        "targeted_overlay": 1,
    }
    assert manifest["reverse_aggregation_validation"]["status"] == "passed"
    assert manifest["outputs"]["noise005"]["sha256"]


def test_rejects_forbidden_source_text_without_manifest(tmp_path: Path) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
        export_raw_run_trajectories as module,
    )

    current = tmp_path / "current.csv"
    legacy = tmp_path / "legacy.csv"
    base = tmp_path / "noise_base.jsonl.gz"
    overlay = tmp_path / "noise_overlay.jsonl.gz"
    output = tmp_path / "export"
    _write_csv(legacy, [_clean_row(520, 0.2)])
    _write_csv(current, [_clean_row(520, 0.2)])
    _write_jsonl_gz(
        base,
        [
            _noise_record(
                condition,
                520,
                1.0,
                source_path="/experiments/all_15alg_fullcpu_v1/result.json",
            )
            for condition in ("noise001", "noise005")
        ],
    )
    _write_jsonl_gz(overlay, [])
    contract = module.GridContract(1, 1, (520,), 1, 1)

    with pytest.raises(module.RawTrajectoryExportError, match="all_15alg_fullcpu_v1"):
        module.export_raw_run_trajectories(
            current_clean_path=current,
            legacy_clean_path=legacy,
            noise_base_paths=[base],
            noise_overlay_paths=[overlay],
            output_dir=output,
            contract=contract,
        )
    assert not (output / "manifest.json").exists()
