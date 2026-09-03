from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.trajectory_coverage import (
    TrajectoryCoverageContractError,
    build_trajectory_coverage_summary,
    main,
    validate_trajectory_coverage_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE5_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _write_source_runs_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "batch",
        "noise_order",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "seconds",
        "id_nmse",
        "ood_nmse",
        "id_r2",
        "ood_r2",
        "id_acc",
        "ood_acc",
        "path",
        "logical_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_inventory(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_row(*, algorithm: str, dataset_id: str, seed: int, host: str, task_id: str) -> dict[str, object]:
    logical_key = f"{algorithm}::{dataset_id}::s{seed}::clean"
    return {
        "batch": "demo_batch",
        "noise_order": 0,
        "algorithm": algorithm,
        "dataset_id": dataset_id,
        "seed": seed,
        "noise_tag": "clean",
        "task_id": task_id,
        "host": host,
        "status": "ok",
        "seconds": "10.0",
        "id_nmse": "0.1",
        "ood_nmse": "0.2",
        "id_r2": "0.0",
        "ood_r2": "0.0",
        "id_acc": "1.0",
        "ood_acc": "1.0",
        "path": f"/remote/{task_id}/result.json",
        "logical_key": logical_key,
    }


def _snapshot(
    minute: int,
    *,
    record_type: str = "periodic_best",
    payload_status: str | None = "ok",
    has_expression: bool = True,
    id_nmse: float | None = 0.1,
    ood_nmse: float | None = 0.2,
    backfilled_from_minute: int | None = None,
) -> dict[str, object]:
    return {
        "minute": minute,
        "status": "ok",
        "record_type": record_type,
        "payload_status": payload_status,
        "checkpoint_index": minute,
        "elapsed_minutes": minute,
        "has_expression": has_expression,
        "expression": "x0" if has_expression else "",
        "id_nmse": id_nmse,
        "ood_nmse": ood_nmse,
        "backfilled_from_minute": backfilled_from_minute,
    }


def _inventory_record(source_row: dict[str, object], snapshots: list[dict[str, object]]) -> dict[str, object]:
    return {
        "source": {
            "batch": source_row["batch"],
            "algorithm": source_row["algorithm"],
            "host": source_row["host"],
            "seed": source_row["seed"],
            "noise_tag": source_row["noise_tag"],
            "task_id": source_row["task_id"],
            "dataset_id": source_row["dataset_id"],
            "path": source_row["path"],
        },
        "result": {
            "status": "ok",
            "exists": True,
            "payload_summary": {
                "payload_status": "ok",
                "has_expression": True,
                "expression": "x0",
                "id_nmse": 0.1,
                "ood_nmse": 0.2,
                "record_type": None,
                "checkpoint_index": None,
                "elapsed_minutes": None,
                "backfilled_from_minute": None,
            },
        },
        "snapshots": snapshots,
        "summary": {
            "expected_snapshots": len(snapshots),
            "available_snapshots": sum(1 for item in snapshots if item["status"] == "ok"),
            "missing_snapshots": sum(1 for item in snapshots if item["status"] == "missing"),
            "conflicting_snapshots": 0,
            "parse_errors": 0,
        },
    }


def test_real_clean_inventory_summary_matches_expected_counts_and_missing_points() -> None:
    inventory_dir = STAGE5_ROOT / "source_snapshot/trajectory_inventory"
    summary = build_trajectory_coverage_summary(
        source_runs_csv=STAGE5_ROOT / "manifests/source_runs.csv",
        inventory_paths=inventory_dir.glob("clean_inventory_*.jsonl.gz"),
        report_paths=inventory_dir.glob("clean_inventory_*.report.json"),
        noise_tag="clean",
        horizon=180,
    )

    assert summary["source_manifest"]["task_count"] == 2250
    assert summary["inventory"]["unique_task_keys"] == 2250
    assert summary["inventory"]["expected_points"] == 405000
    assert summary["inventory"]["available_points"] == 404985
    assert summary["inventory"]["point_classification_counts"] == {
        "direct_valid": 399953,
        "evaluator_error": 1388,
        "explicit_no_output": 3610,
        "future_backfill_ignored": 34,
        "missing_file": 15,
    }
    missing_points = summary["inventory"]["missing_point_details"]
    assert len(missing_points) == 15
    assert [point["minute"] for point in missing_points] == list(range(165, 180))
    assert {point["logical_key"] for point in missing_points} == {"fepysr::Nguyen-12::s520::clean"}
    assert summary["inventory"]["future_backfill_points"]
    assert len(summary["inventory"]["future_backfill_points"]) == 34
    assert summary["final_result_audit"]["missing_task_keys"] == []
    assert summary["final_result_audit"]["unexpected_task_keys"] == []
    assert summary["final_result_audit"]["inventory_identity_mismatches"] == []
    assert summary["final_result_audit"]["dataset_mismatches"] == []
    assert summary["final_result_audit"]["seed_mismatches"] == []
    assert summary["final_result_audit"]["id_nmse_mismatches"] == []
    assert summary["final_result_audit"]["ood_nmse_mismatches"] == []
    assert summary["readiness"]["identity_contract_ok"] is True
    assert summary["readiness"]["raw_missing_points"] == 15
    assert summary["readiness"]["repaired_missing_points"] == 0
    assert summary["readiness"]["unresolved_missing_points"] == 15
    assert summary["readiness"]["repair_manifest_valid"] is False
    assert summary["readiness"]["formal_eff_ready"] is False
    assert summary["readiness"]["blocking_missing_points"] == 15
    validate_trajectory_coverage_summary(summary)
    with pytest.raises(TrajectoryCoverageContractError, match="future backfill"):
        validate_trajectory_coverage_summary(summary, reject_future_backfill=True)


def test_real_clean_inventory_repair_manifest_closes_raw_missing_points() -> None:
    inventory_dir = STAGE5_ROOT / "source_snapshot/trajectory_inventory"
    summary = build_trajectory_coverage_summary(
        source_runs_csv=STAGE5_ROOT / "manifests/source_runs.csv",
        inventory_paths=inventory_dir.glob("clean_inventory_*.jsonl.gz"),
        report_paths=inventory_dir.glob("clean_inventory_*.report.json"),
        noise_tag="clean",
        horizon=180,
        repair_manifest_path=STAGE5_ROOT / "manifests/trajectory_repairs.v1.json",
        repo_root=REPO_ROOT,
    )

    # raw inventory 保持原始证据；修复仅关闭有效 EFF 门禁，不静默改写来源计数。
    assert summary["inventory"]["missing_points"] == 15
    assert summary["readiness"]["raw_missing_points"] == 15
    assert summary["readiness"]["repaired_missing_points"] == 15
    assert summary["readiness"]["unresolved_missing_points"] == 0
    assert summary["readiness"]["repair_manifest_valid"] is True
    assert summary["readiness"]["formal_eff_ready"] is True
    validate_trajectory_coverage_summary(summary, require_formal_eff_ready=True)


def test_repair_manifest_must_exactly_close_raw_missing_points(tmp_path: Path) -> None:
    manifest_path = STAGE5_ROOT / "manifests/trajectory_repairs.v1.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["repairs"][0]["missing_minutes"][-1] = 180
    tampered_path = tmp_path / "trajectory_repairs.v1.json"
    tampered_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inventory_dir = STAGE5_ROOT / "source_snapshot/trajectory_inventory"
    with pytest.raises(TrajectoryCoverageContractError, match="exact closure"):
        build_trajectory_coverage_summary(
            source_runs_csv=STAGE5_ROOT / "manifests/source_runs.csv",
            inventory_paths=inventory_dir.glob("clean_inventory_*.jsonl.gz"),
            report_paths=inventory_dir.glob("clean_inventory_*.report.json"),
            noise_tag="clean",
            horizon=180,
            repair_manifest_path=tampered_path,
            repo_root=REPO_ROOT,
        )


def test_cli_writes_summary_and_records_ignored_future_backfill(tmp_path: Path) -> None:
    source_csv = tmp_path / "manifests/source_runs.csv"
    inventory_dir = tmp_path / "inventory"
    output_json = tmp_path / "reports/trajectory_coverage.json"

    row = _source_row(
        algorithm="demo",
        dataset_id="dataset",
        seed=520,
        host="iaaccn22",
        task_id="demo_s520_clean_g0001",
    )
    _write_source_runs_csv(source_csv, [row])
    record = _inventory_record(
        row,
        [
            _snapshot(1, record_type="periodic_backfill", backfilled_from_minute=2),
            _snapshot(2, record_type="final_best"),
        ],
    )
    _write_inventory(inventory_dir / "clean_inventory_iaaccn22.jsonl.gz", [record])
    _write_report(
        inventory_dir / "clean_inventory_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 2,
            "available_snapshots": 2,
            "missing_snapshots": 0,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": False,
        },
    )

    rc = main(
        [
            "--source-runs-csv",
            str(source_csv),
            "--inventory-dir",
            str(inventory_dir),
            "--noise-tag",
            "clean",
            "--horizon",
            "2",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["inventory"]["available_points"] == 2
    assert len(payload["inventory"]["future_backfill_points"]) == 1
    assert payload["inventory"]["point_classification_counts"]["future_backfill_ignored"] == 1
    assert payload["readiness"]["raw_missing_points"] == 0
    assert payload["readiness"]["repaired_missing_points"] == 0
    assert payload["readiness"]["unresolved_missing_points"] == 0
    assert payload["readiness"]["repair_manifest_valid"] is True
    assert payload["readiness"]["formal_eff_ready"] is True


def test_strict_validation_rejects_future_backfill_summary(tmp_path: Path) -> None:
    source_csv = tmp_path / "manifests/source_runs.csv"
    inventory_dir = tmp_path / "inventory"

    row = _source_row(
        algorithm="demo",
        dataset_id="dataset",
        seed=520,
        host="iaaccn22",
        task_id="demo_s520_clean_g0001",
    )
    _write_source_runs_csv(source_csv, [row])
    record = _inventory_record(
        row,
        [
            _snapshot(1, record_type="periodic_best"),
            _snapshot(2, record_type="periodic_backfill", backfilled_from_minute=3),
            {
                "minute": 3,
                "status": "missing",
                "record_type": None,
                "payload_status": None,
                "has_expression": False,
                "id_nmse": None,
                "ood_nmse": None,
                "backfilled_from_minute": None,
            },
        ],
    )
    _write_inventory(inventory_dir / "clean_inventory_iaaccn22.jsonl.gz", [record])
    _write_report(
        inventory_dir / "clean_inventory_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 3,
            "available_snapshots": 2,
            "missing_snapshots": 1,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": False,
        },
    )

    summary = build_trajectory_coverage_summary(
        source_runs_csv=source_csv,
        inventory_paths=inventory_dir.glob("clean_inventory_*.jsonl.gz"),
        report_paths=inventory_dir.glob("clean_inventory_*.report.json"),
        noise_tag="clean",
        horizon=3,
    )
    assert summary["inventory"]["point_classification_counts"] == {
        "direct_valid": 1,
        "evaluator_error": 0,
        "explicit_no_output": 0,
        "future_backfill_ignored": 1,
        "missing_file": 1,
    }
    with pytest.raises(TrajectoryCoverageContractError, match="future backfill"):
        validate_trajectory_coverage_summary(summary, reject_future_backfill=True)
