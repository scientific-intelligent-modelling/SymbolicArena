from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
    prepare_clean_numeric as module,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.performance_replay import (
    PerformanceReplayError,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_clean_numeric import (
    NumericPreparationError,
    prepare_clean_numeric,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset_dir = tmp_path / "sim-datasets-data/ssr50/datasets/demo/dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        "dataset:\n  name: dataset\n  target:\n    name: y\n  features:\n    - name: x\n",
        encoding="utf-8",
    )
    for split in ("train", "valid", "id_test", "ood_test"):
        (dataset_dir / f"{split}.csv").write_text("x,y\n1,1\n2,2\n3,3\n", encoding="utf-8")
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
    for seed, nmse, equation in (
        (520, "1e-12", "x0"),
        (521, "100", "1 / (x0 - x0)"),
    ):
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
            "dataset_dir": str(dataset_dir),
            "expected_dataset_rel": "sim-datasets-data/ssr50/datasets/demo/dataset",
            "feature_names": ["x"],
            "canonical_artifact": {
                "tool_name": "demo",
                "normalized_expression": equation,
                "instantiated_expression": equation,
            }
            if equation
            else None,
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
    recovery = tmp_path / "formula_recovery.v1.json"
    recovery.write_text(
        json.dumps(
            {
                "schema_version": "formula_recovery.v1",
                "condition": "clean",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest, bundle, binding, recovery


def test_prepare_clean_numeric_maps_each_run_before_aggregation(tmp_path: Path) -> None:
    manifest, bundle, binding, recovery = _write_fixture(tmp_path)
    run_csv = tmp_path / "results/run.csv"
    algorithm_csv = tmp_path / "results/algorithm.csv"
    report_json = tmp_path / "reports/report.json"

    report = prepare_clean_numeric(
        source_runs_csv=manifest,
        freeze_paths=[bundle],
        freeze_binding_json=binding,
        formula_recovery_json=recovery,
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
    assert report["status"] == "ok"
    assert report["counts"]["valid_outputs"] == 1
    assert report["counts"]["canonical_invalid_outputs"] == 1
    assert report["counts"]["replay_unavailable"] == 0
    assert report["evaluation_path"] == "canonical_replay.v1"
    assert report["inputs"]["formula_recovery_manifest"] == {
        "path": str(recovery.resolve()),
        "sha256": _sha256(recovery),
        "condition": "clean",
        "entry_count": 0,
    }
    assert report["outputs"]["run_csv_row_count"] == 2
    assert report["outputs"]["algorithm_csv_row_count"] == 1
    assert [float(row["id_quality"]) for row in run_rows] == [1.0, 0.0]
    assert float(run_rows[0]["id_nmse"]) == pytest.approx(0.0)
    assert float(run_rows[0]["native_id_nmse"]) == pytest.approx(1.0e-12)
    assert run_rows[0]["evaluation_path"] == "canonical_replay.v1"
    assert float(algorithm_rows[0]["ID"]) == pytest.approx(50.0)
    assert float(algorithm_rows[0]["OOD"]) == pytest.approx(50.0)
    assert report_json.exists()


def test_prepare_clean_numeric_rejects_bundle_hash_drift(tmp_path: Path) -> None:
    manifest, bundle, binding, recovery = _write_fixture(tmp_path)
    with bundle.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(NumericPreparationError, match="SHA-256"):
        prepare_clean_numeric(
            source_runs_csv=manifest,
            freeze_paths=[bundle],
            freeze_binding_json=binding,
            formula_recovery_json=recovery,
            run_csv=tmp_path / "run.csv",
            algorithm_csv=tmp_path / "algorithm.csv",
            report_json=tmp_path / "report.json",
            expected_runs=2,
            expected_algorithms=1,
            expected_runs_per_algorithm=2,
        )


def test_prepare_clean_numeric_counts_invalid_replay_without_replay_error(
    tmp_path: Path,
) -> None:
    manifest, bundle, binding, recovery = _write_fixture(tmp_path)
    run_csv = tmp_path / "run.csv"
    report = prepare_clean_numeric(
        source_runs_csv=manifest,
        freeze_paths=[bundle],
        freeze_binding_json=binding,
        formula_recovery_json=recovery,
        run_csv=run_csv,
        algorithm_csv=tmp_path / "algorithm.csv",
        report_json=tmp_path / "report.json",
        expected_runs=2,
        expected_algorithms=1,
        expected_runs_per_algorithm=2,
        repo_root=tmp_path,
    )

    rows = list(csv.DictReader(run_csv.open(encoding="utf-8", newline="")))
    replayed = next(row for row in rows if row["seed"] == "521")
    assert replayed["evaluation_status"] == "invalid_output"
    assert replayed["valid_output"] == "false"
    assert replayed["invalid_reason"]
    assert replayed["replay_error"] == ""
    assert float(replayed["id_quality"]) == 0.0
    assert report["counts"]["canonical_invalid_outputs"] == 1
    assert report["counts"]["replay_unavailable"] == 0
    assert report["counts"]["replay_errors"] == 0


def test_prepare_clean_numeric_keeps_replay_unavailable_out_of_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, bundle, binding, recovery = _write_fixture(tmp_path)
    original_replay = module.replay_payload_performance

    def replay_or_unavailable(payload, **kwargs):
        if payload["equation"] == "x0":
            raise PerformanceReplayError("frozen evidence unavailable")
        return original_replay(payload, **kwargs)

    monkeypatch.setattr(module, "replay_payload_performance", replay_or_unavailable)
    run_csv = tmp_path / "run.csv"
    algorithm_csv = tmp_path / "algorithm.csv"
    report = prepare_clean_numeric(
        source_runs_csv=manifest,
        freeze_paths=[bundle],
        freeze_binding_json=binding,
        formula_recovery_json=recovery,
        run_csv=run_csv,
        algorithm_csv=algorithm_csv,
        report_json=tmp_path / "report.json",
        expected_runs=2,
        expected_algorithms=1,
        expected_runs_per_algorithm=2,
        repo_root=tmp_path,
    )

    rows = list(csv.DictReader(run_csv.open(encoding="utf-8", newline="")))
    unavailable = next(row for row in rows if row["seed"] == "520")
    assert unavailable["evaluation_status"] == "replay_unavailable"
    assert unavailable["valid_output"] == ""
    assert unavailable["invalid_reason"] == ""
    assert unavailable["replay_error"] == "frozen evidence unavailable"
    for field in (
        "id_nmse",
        "ood_nmse",
        "id_quality",
        "ood_quality",
        "id_quality_delta_from_native",
        "ood_quality_delta_from_native",
    ):
        assert unavailable[field] == ""
    assert report["status"] == "error"
    assert report["contract_ok"] is False
    assert report["counts"]["valid_outputs"] == 0
    assert report["counts"]["canonical_invalid_outputs"] == 1
    assert report["counts"]["replay_unavailable"] == 1
    assert report["counts"]["replay_errors"] == 1
    assert report["outputs"]["algorithm_csv_row_count"] == 0
    assert list(csv.DictReader(algorithm_csv.open(encoding="utf-8", newline=""))) == []


def test_clean_numeric_cli_returns_nonzero_when_contract_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "prepare_clean_numeric",
        lambda **_kwargs: {"counts": {"replay_unavailable": 1}, "contract_ok": False},
    )

    assert module.main([]) == 2
