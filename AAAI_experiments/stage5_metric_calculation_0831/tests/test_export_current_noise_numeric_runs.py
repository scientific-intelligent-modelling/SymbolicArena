from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _base_rows() -> list[dict[str, object]]:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_noise_numeric import (
        NUMERIC_FIELDS,
    )

    rows: list[dict[str, object]] = []
    for algorithm in ("DemoA", "DemoB"):
        for seed in (520, 521, 522):
            values: dict[str, object] = {field: "" for field in NUMERIC_FIELDS}
            values.update(
                {
                    "logical_key": f"{algorithm}::DemoGT::s{seed}::noise001",
                    "algorithm": algorithm,
                    "dataset_id": "DemoGT",
                    "seed": seed,
                    "noise_tag": "noise001",
                    "task_id": f"old_{algorithm}_{seed}",
                    "host": "iaaccn22",
                    "result_sha256": "a" * 64,
                    "evaluation_status": "valid",
                    "valid_output": "true",
                    "formula_source": "canonical_artifact",
                    "evaluation_path": "canonical_replay.v1",
                    "canonical_artifact_sha256": "b" * 64,
                    "artifact_rebuilt": "false",
                    "native_id_nmse": "1",
                    "native_ood_nmse": "2",
                    "id_nmse": "1",
                    "ood_nmse": "2",
                    "id_quality": "0.5",
                    "ood_quality": "0.4",
                    "id_nmse_delta_from_native": "0",
                    "ood_nmse_delta_from_native": "0",
                    "id_quality_delta_from_native": "0",
                    "ood_quality_delta_from_native": "0",
                }
            )
            rows.append(values)
    return rows


def _write_overlay(path: Path, *, source_path: str = "/remote/result.json") -> str:
    payload = {
        "status": "ok",
        "equation": "x0 + 2",
        "canonical_artifact": {"instantiated_expression": "x0 + 2"},
        "id_test": {"nmse": 0.25},
        "ood_test": {"nmse": 4.0},
        "train_label_noise": {
            "enabled": True,
            "requested": True,
            "sigma": 0.01,
        },
    }
    raw_text = json.dumps(payload, sort_keys=True)
    result_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    record = {
        "source": {
            "algorithm": "DemoA",
            "dataset_id": "DemoGT",
            "seed": 520,
            "noise_tag": "noise001",
            "task_id": "new_demo_a_520",
            "host": "iaaccn29",
            "batch": "targeted_rerun",
            "path": source_path,
        },
        "result": {"raw_text": raw_text, "sha256": result_sha},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return result_sha


def test_export_current_noise_numeric_runs_overlays_binding_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
        export_current_noise_numeric_runs as module,
    )

    base_csv = tmp_path / "noise001_base.csv"
    _write_csv(base_csv, _base_rows())
    overlay_path = tmp_path / "targeted.jsonl.gz"
    result_sha = _write_overlay(overlay_path)

    def fake_replay(payload, **kwargs):
        assert kwargs["task_id"] == "new_demo_a_520"
        assert kwargs["result_sha256"] == result_sha
        return {
            "evaluation_path": "canonical_replay.v1",
            "canonical_artifact": {"instantiated_expression": "x0 + 2"},
            "canonical_artifact_sha256": "c" * 64,
            "artifact_rebuilt": True,
            "id_test": {"nmse": 0.125},
            "ood_test": {"nmse": 8.0},
            "id_quality": 0.9,
            "ood_quality": 0.3,
            "valid_output": True,
            "invalid_reason": None,
            "native_id_nmse": 0.25,
            "native_ood_nmse": 4.0,
            "error": None,
        }

    monkeypatch.setattr(module, "replay_payload_performance", fake_replay)
    output_csv = tmp_path / "noise001_current.csv"

    report = module.export_current_noise_numeric_runs(
        base_numeric_csv=base_csv,
        overlay_jsonl_paths=[overlay_path],
        condition="noise001",
        output_csv=output_csv,
        repo_root=tmp_path,
        expected_runs=6,
        expected_algorithms=2,
        expected_datasets=1,
        expected_overlay_runs=1,
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8", newline="")))
    assert len(rows) == 6
    current = next(row for row in rows if row["logical_key"] == "DemoA::DemoGT::s520::noise001")
    assert current["task_id"] == "new_demo_a_520"
    assert current["host"] == "iaaccn29"
    assert current["result_sha256"] == result_sha
    assert current["id_nmse"] == "0.125"
    assert current["ood_nmse"] == "8"
    assert current["id_quality"] == "0.90000000000000002"
    assert current["evaluation_path"] == "canonical_replay.v1"
    assert report["row_count"] == 6
    assert report["overlay_run_count"] == 1
    assert report["replay_unavailable_count"] == 0


def test_export_current_noise_numeric_runs_rejects_forbidden_overlay_source(
    tmp_path: Path,
) -> None:
    from AAAI_experiments.stage5_metric_calculation_0831.pipeline.export_current_noise_numeric_runs import (
        CurrentNoiseNumericExportError,
        export_current_noise_numeric_runs,
    )

    base_csv = tmp_path / "noise001_base.csv"
    _write_csv(base_csv, _base_rows())
    overlay_path = tmp_path / "targeted.jsonl.gz"
    _write_overlay(
        overlay_path,
        source_path="/remote/all_15alg_fullcpu_v1/result.json",
    )

    with pytest.raises(CurrentNoiseNumericExportError, match="禁止来源"):
        export_current_noise_numeric_runs(
            base_numeric_csv=base_csv,
            overlay_jsonl_paths=[overlay_path],
            condition="noise001",
            output_csv=tmp_path / "out.csv",
            repo_root=tmp_path,
            expected_runs=6,
            expected_algorithms=2,
            expected_datasets=1,
            expected_overlay_runs=1,
        )
