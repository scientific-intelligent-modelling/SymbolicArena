from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_clean_numeric import (
    NumericPreparationError,
    prepare_clean_numeric,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "source_runs.csv"
    fieldnames = [
        "batch",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "id_nmse",
        "ood_nmse",
        "path",
        "logical_key",
    ]
    rows = []
    frozen = []
    for seed, nmse, equation in ((520, "1e-12", "x0"), (521, "100", "")):
        task_id = f"demo_s{seed}_clean_g0001"
        remote_path = f"/remote/{task_id}/result.json"
        logical_key = f"demo::dataset::s{seed}::clean"
        rows.append(
            {
                "batch": "batch",
                "algorithm": "demo",
                "dataset_id": "dataset",
                "seed": seed,
                "noise_tag": "clean",
                "task_id": task_id,
                "host": "iaaccn22",
                "status": "ok",
                "id_nmse": nmse,
                "ood_nmse": nmse,
                "path": remote_path,
                "logical_key": logical_key,
            }
        )
        result_payload = {
            "status": "ok",
            "equation": equation,
            "id_test": {"nmse": float(nmse)},
            "ood_test": {"nmse": float(nmse)},
        }
        raw_text = json.dumps(result_payload, ensure_ascii=False, sort_keys=True)
        frozen.append(
            {
                "source": {
                    "batch": "batch",
                    "algorithm": "demo",
                    "dataset_id": "dataset",
                    "seed": seed,
                    "noise_tag": "clean",
                    "task_id": task_id,
                    "host": "iaaccn22",
                    "path": remote_path,
                },
                "result": {
                    "status": "ok",
                    "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "raw_text": raw_text,
                },
                "snapshots": [],
            }
        )

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bundle = tmp_path / "clean_freeze_iaaccn22.jsonl.gz"
    with gzip.open(bundle, "wt", encoding="utf-8") as handle:
        for row in frozen:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    binding = tmp_path / "freeze_binding.json"
    binding.write_text(
        json.dumps(
            {
                "contract_ok": True,
                "freeze_counts": {"tasks": 2},
                "input_files": {
                    "freeze_records": [
                        {
                            "host": "iaaccn22",
                            "path": str(bundle),
                            "sha256": _sha256(bundle),
                            "size_bytes": bundle.stat().st_size,
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest, bundle, binding


def test_prepare_clean_numeric_maps_each_run_before_aggregation(tmp_path: Path) -> None:
    manifest, bundle, binding = _write_fixture(tmp_path)
    run_csv = tmp_path / "results/run.csv"
    algorithm_csv = tmp_path / "results/algorithm.csv"
    report_json = tmp_path / "reports/report.json"

    report = prepare_clean_numeric(
        source_runs_csv=manifest,
        freeze_paths=[bundle],
        freeze_binding_json=binding,
        run_csv=run_csv,
        algorithm_csv=algorithm_csv,
        report_json=report_json,
        expected_runs=2,
        expected_algorithms=1,
        expected_runs_per_algorithm=2,
    )

    with run_csv.open(encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    with algorithm_csv.open(encoding="utf-8", newline="") as handle:
        algorithm_rows = list(csv.DictReader(handle))

    assert report["contract_ok"] is True
    assert report["counts"]["valid_outputs"] == 1
    assert report["counts"]["invalid_outputs"] == 1
    assert [float(row["id_quality"]) for row in run_rows] == [1.0, 0.0]
    assert float(algorithm_rows[0]["ID"]) == pytest.approx(50.0)
    assert float(algorithm_rows[0]["OOD"]) == pytest.approx(50.0)
    assert report_json.exists()


def test_prepare_clean_numeric_rejects_bundle_hash_drift(tmp_path: Path) -> None:
    manifest, bundle, binding = _write_fixture(tmp_path)
    with bundle.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(NumericPreparationError, match="SHA-256"):
        prepare_clean_numeric(
            source_runs_csv=manifest,
            freeze_paths=[bundle],
            freeze_binding_json=binding,
            run_csv=tmp_path / "run.csv",
            algorithm_csv=tmp_path / "algorithm.csv",
            report_json=tmp_path / "report.json",
            expected_runs=2,
            expected_algorithms=1,
            expected_runs_per_algorithm=2,
        )
