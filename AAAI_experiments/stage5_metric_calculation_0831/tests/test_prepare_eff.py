"""prepare_eff 真实仓库手工校验命令：

dry:
  python -m AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff \
    --limit-runs 32 \
    --output-jsonl /tmp/eff_prepare_dry.jsonl \
    --output-csv /tmp/eff_prepare_dry.csv \
    --output-report /tmp/eff_prepare_dry.json \
    --print-summary

full:
  python -m AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff \
    --output-jsonl /tmp/eff_prepare_full.jsonl \
    --output-csv /tmp/eff_prepare_full.csv \
    --output-report /tmp/eff_prepare_full.json \
    --print-summary
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    efficiency_from_qualities,
    phi_nmse,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff import (
    build_eff_preparation,
)


STAGE5_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_payload(
    minute: int,
    *,
    record_type: str,
    expression: str | None,
    id_nmse: float | None,
    ood_nmse: float | None,
    backfilled_from_minute: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": "fixture",
        "record_type": record_type,
        "checkpoint_index": minute,
        "elapsed_seconds": minute * 60,
        "elapsed_minutes": minute,
        "equation": expression,
        "status": "ok" if expression is not None and id_nmse is not None and ood_nmse is not None else "running",
        "id_test": {"nmse": id_nmse} if id_nmse is not None else None,
        "ood_test": {"nmse": ood_nmse} if ood_nmse is not None else None,
    }
    if backfilled_from_minute is not None:
        payload["backfilled_from_minute"] = backfilled_from_minute
    if expression is not None:
        payload["canonical_artifact"] = {"instantiated_expression": expression}
    return payload


def _frozen_snapshot_from_payload(
    minute: int,
    payload: dict[str, object],
    *,
    outer_path: str,
) -> dict[str, object]:
    raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return {
        "status": "ok",
        "minute": minute,
        "checkpoint_index": minute,
        "record_type": payload["record_type"],
        "payload_status": payload.get("status"),
        "backfilled_from_minute": payload.get("backfilled_from_minute"),
        "has_expression": payload.get("equation") is not None,
        "expression": payload.get("equation"),
        "id_nmse": payload["id_test"]["nmse"] if payload.get("id_test") else None,
        "ood_nmse": payload["ood_test"]["nmse"] if payload.get("ood_test") else None,
        "outer_path": outer_path,
        "outer_status": "ok",
        "outer_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "selected_path": outer_path,
        "selected_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_text": raw_text,
    }


def _heartbeat_snapshot(minute: int, *, outer_path: str) -> dict[str, object]:
    payload = _snapshot_payload(
        minute,
        record_type="periodic_heartbeat",
        expression=None,
        id_nmse=None,
        ood_nmse=None,
    )
    return _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)


def _missing_snapshot(minute: int, *, outer_path: str) -> dict[str, object]:
    return {
        "status": "missing",
        "minute": minute,
        "outer_path": outer_path,
        "outer_status": "missing",
        "outer_sha256": None,
        "selected_path": None,
        "selected_sha256": None,
        "raw_text": None,
        "conflict": False,
        "duplicate_semantically_equal": False,
        "inner_path": None,
        "inner_sha256": None,
        "inner_status": None,
    }


def _write_bundle(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    stage5_root = tmp_path / "AAAI_experiments/stage5_metric_calculation_0831"
    freeze_dir = stage5_root / "source_snapshot/trajectory_freeze"
    reports_dir = stage5_root / "reports"
    manifests_dir = stage5_root / "manifests"
    evidence_dir = stage5_root / "source_snapshot/recovery_evidence"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    current_best_path = evidence_dir / "fepysr_s520_clean_g0021.current_best.json"
    current_best_payload = {"equation": "repair_eq"}
    current_best_path.write_text(
        json.dumps(current_best_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    wrapper_path = tmp_path / "scientific_intelligent_modelling/algorithms/fepysr_wrapper/wrapper.py"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text("# fixture wrapper\n", encoding="utf-8")

    host = "fixturehost"
    bundle_path = freeze_dir / f"clean_freeze_{host}.jsonl.gz"
    bundle_report_path = freeze_dir / f"clean_freeze_{host}.report.json"

    repair_record = {
        "source": {
            "algorithm": "fepysr",
            "dataset_id": "Nguyen-12",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "fepysr_s520_clean_g0021",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/fepysr/minute_{minute:04d}.json"
        if minute == 164:
            repair_payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="repair_eq",
                id_nmse=1.0e-3,
                ood_nmse=2.0e-3,
            )
            repair_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, repair_payload, outer_path=outer_path)
            )
        elif 165 <= minute <= 179:
            repair_record["snapshots"].append(_missing_snapshot(minute, outer_path=outer_path))
        elif minute == 180:
            final_payload = _snapshot_payload(
                minute,
                record_type="recovered_final",
                expression="future_eq",
                id_nmse=1.0e-8,
                ood_nmse=1.0e-8,
            )
            repair_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, final_payload, outer_path=outer_path)
            )
        else:
            repair_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    future_record = {
        "source": {
            "algorithm": "tpsr",
            "dataset_id": "CRK11",
            "seed": "520",
            "noise_tag": "clean",
            "task_id": "tpsr_s520_clean_g0001",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/tpsr/minute_{minute:04d}.json"
        if minute == 53:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="pre_future_eq",
                id_nmse=1.0e-2,
                ood_nmse=1.0e-2,
            )
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        elif minute in (54, 55):
            payload = _snapshot_payload(
                minute,
                record_type="periodic_backfill",
                expression="leaked_eq",
                id_nmse=1.0e-9,
                ood_nmse=1.0e-9,
                backfilled_from_minute=56,
            )
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        elif minute == 56:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="future_anchor_eq",
                id_nmse=1.0e-4,
                ood_nmse=1.0e-4,
            )
            payload["checkpoint_index"] = 54
            future_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        else:
            future_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    plain_record = {
        "source": {
            "algorithm": "dso",
            "dataset_id": "plain",
            "seed": "521",
            "noise_tag": "clean",
            "task_id": "dso_s521_clean_g0002",
            "host": host,
        },
        "snapshots": [],
    }
    for minute in range(1, 181):
        outer_path = f"/bundle/dso/minute_{minute:04d}.json"
        if minute == 1:
            payload = _snapshot_payload(
                minute,
                record_type="periodic_best",
                expression="plain_eq",
                id_nmse=1.0,
                ood_nmse=1.0,
            )
            plain_record["snapshots"].append(
                _frozen_snapshot_from_payload(minute, payload, outer_path=outer_path)
            )
        else:
            plain_record["snapshots"].append(_heartbeat_snapshot(minute, outer_path=outer_path))

    _write_bundle(bundle_path, [repair_record, future_record, plain_record])
    bundle_report_payload = {
        "tasks": 3,
        "expected_snapshots": 540,
        "available_snapshots": 525,
        "missing_snapshots": 15,
        "conflicting_snapshots": 0,
        "parse_errors": 0,
        "result_missing_or_invalid": 0,
        "freeze_raw": True,
        "verify_inner": False,
    }
    bundle_report_path.write_text(
        json.dumps(bundle_report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repair_manifest_path = manifests_dir / "trajectory_repairs.v1.json"
    minute_164_snapshot = repair_record["snapshots"][163]
    minute_180_snapshot = repair_record["snapshots"][179]
    repair_manifest_payload = {
        "schema_version": "trajectory_repairs.v1",
        "condition": "clean",
        "horizon": 180,
        "repairs": [
            {
                "logical_key": "fepysr::Nguyen-12::s520::clean",
                "task_id": "fepysr_s520_clean_g0021",
                "host": host,
                "rule": "carry_forward_last_observed_best",
                "missing_minutes": list(range(165, 180)),
                "source_minute": 164,
                "source_snapshot_sha256": minute_164_snapshot["selected_sha256"],
                "source_record_type": "periodic_best",
                "source_elapsed_seconds": 164 * 60,
                "excluded_future_minute": 180,
                "excluded_future_snapshot_sha256": minute_180_snapshot["selected_sha256"],
                "supporting_evidence": {
                    "current_best_path": str(current_best_path.relative_to(tmp_path)),
                    "current_best_sha256": _sha256_file(current_best_path),
                    "wrapper_path": str(wrapper_path.relative_to(tmp_path)),
                    "wrapper_sha256": _sha256_file(wrapper_path),
                },
            }
        ],
    }
    repair_manifest_path.write_text(
        json.dumps(repair_manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    binding_report_path = reports_dir / "freeze_binding.json"
    binding_payload = {
        "noise_tag": "clean",
        "horizon": 180,
        "inventory_counts": {
            "hosts": 1,
            "tasks": 3,
            "expected_points": 540,
            "existing_points": 525,
            "missing_points": 15,
        },
        "freeze_counts": {
            "hosts": 1,
            "tasks": 3,
            "expected_points": 540,
            "existing_points": 525,
            "missing_points": 15,
        },
        "missing_point_details": [
            {
                "logical_key": "fepysr::Nguyen-12::s520::clean",
                "minute": minute,
                "host": host,
                "task_id": "fepysr_s520_clean_g0021",
            }
            for minute in range(165, 180)
        ],
        "input_files": {
            "freeze_records": [
                {
                    "host": host,
                    "path": str(bundle_path.resolve()),
                    "sha256": _sha256_file(bundle_path),
                    "size_bytes": bundle_path.stat().st_size,
                }
            ],
            "freeze_reports": [
                {
                    "host": host,
                    "path": str(bundle_report_path.resolve()),
                    "sha256": _sha256_file(bundle_report_path),
                    "size_bytes": bundle_report_path.stat().st_size,
                }
            ],
            "inventory_records": [],
            "inventory_reports": [],
        },
        "per_host": {
            host: {
                "freeze": {
                    "counts": {
                        "tasks": 3,
                        "expected_points": 540,
                        "existing_points": 525,
                        "missing_points": 15,
                        "result_ok": 3,
                        "result_missing_or_invalid": 0,
                    },
                    "records_file": {
                        "host": host,
                        "path": str(bundle_path.resolve()),
                        "sha256": _sha256_file(bundle_path),
                        "size_bytes": bundle_path.stat().st_size,
                    },
                    "report_file": {
                        "host": host,
                        "path": str(bundle_report_path.resolve()),
                        "sha256": _sha256_file(bundle_report_path),
                        "size_bytes": bundle_report_path.stat().st_size,
                    },
                    "report": bundle_report_payload,
                }
            }
        },
        "binding": {
            "drift_count": 0,
            "drift_details": [],
            "truncated_drift_details": 0,
        },
        "contract_ok": True,
    }
    binding_report_path.write_text(
        json.dumps(binding_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "stage5_root": stage5_root,
        "binding_report": binding_report_path,
        "repair_manifest": repair_manifest_path,
    }


def test_build_eff_preparation_applies_audited_repairs_and_future_backfill_counts(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    rows, report = build_eff_preparation(
        freeze_binding_report=fixture["binding_report"],
        repair_manifest=fixture["repair_manifest"],
        repo_root=tmp_path,
        expected_hosts=1,
        expected_tasks=3,
        expected_points=540,
        expected_existing_points=525,
        expected_missing_points=15,
        expected_audited_repair_points=15,
        expected_future_backfill_ignored_points=2,
        expected_checkpoint_normalization_points=1,
    )

    assert len(rows) == 3
    assert report["summary"]["success_count"] == 3
    assert report["summary"]["unresolved_run_count"] == 0
    assert report["summary"]["audited_repair_points"] == 15
    assert report["summary"]["future_backfill_ignored_points"] == 2
    assert report["summary"]["checkpoint_normalization_points"] == 1
    assert report["summary"]["missing_points_after_repairs"] == 0

    by_key = {row["logical_key"]: row for row in rows}
    repaired = by_key["fepysr::Nguyen-12::s520::clean"]
    assert repaired["audited_repair_points"] == 15
    assert repaired["future_backfill_ignored_points"] == 0
    assert repaired["trajectory_sources"][164] == "audited_repair:164"
    assert repaired["trajectory_sources"][178] == "audited_repair:164"
    assert repaired["trajectory_sources"][179] == "final:180"
    assert repaired["quality_trajectory"][163] == pytest.approx((phi_nmse(1.0e-3) + phi_nmse(2.0e-3)) / 2.0)
    assert repaired["m_eff"] == pytest.approx(
        efficiency_from_qualities(repaired["quality_trajectory"], horizon=180)
    )

    future = by_key["tpsr::CRK11::s520::clean"]
    assert future["future_backfill_ignored_points"] == 2
    assert future["checkpoint_normalizations"] == [
        {
            "minute": 56,
            "original_checkpoint_index": 54,
            "normalized_checkpoint_index": 56,
            "future_backfill_minutes": [54, 55],
        }
    ]
    assert future["trajectory_sources"][53] == "future_backfill_ignored:56;carry_forward:53"
    assert future["trajectory_sources"][54] == "future_backfill_ignored:56;carry_forward:53"
    assert future["quality_trajectory"][55] == pytest.approx((phi_nmse(1.0e-4) + phi_nmse(1.0e-4)) / 2.0)

    plain = by_key["dso::plain::s521::clean"]
    assert plain["audited_repair_points"] == 0
    assert all(0.0 <= value <= 1.0 for value in plain["quality_trajectory"])


def test_cli_writes_jsonl_csv_and_report(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    output_jsonl = tmp_path / "eff_preparation.jsonl"
    output_csv = tmp_path / "eff_preparation.csv"
    output_report = tmp_path / "eff_preparation.json"

    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--freeze-binding-report",
        str(fixture["binding_report"]),
        "--repair-manifest",
        str(fixture["repair_manifest"]),
        "--repo-root",
        str(tmp_path),
        "--expected-hosts",
        "1",
        "--expected-tasks",
        "3",
        "--expected-points",
        "540",
        "--expected-existing-points",
        "525",
        "--expected-missing-points",
        "15",
        "--expected-audited-repair-points",
        "15",
        "--expected-future-backfill-ignored-points",
        "2",
        "--expected-checkpoint-normalization-points",
        "1",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(completed.stdout)
    assert report["summary"]["success_count"] == 3
    assert output_jsonl.exists()
    assert output_csv.exists()
    assert output_report.exists()

    with output_jsonl.open("r", encoding="utf-8") as handle:
        jsonl_rows = [json.loads(line) for line in handle if line.strip()]
    assert len(jsonl_rows) == 3
    assert all(len(row["quality_trajectory"]) == 180 for row in jsonl_rows)

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3
    assert "q_0180" in csv_rows[0]
    assert float(csv_rows[0]["m_eff"]) >= 0.0


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE5_EFF_DRY") != "1",
    reason="仅手工触发真实仓库 dry CLI 校验",
)
def test_real_repo_cli_dry_run(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "real_dry.jsonl"
    output_csv = tmp_path / "real_dry.csv"
    output_report = tmp_path / "real_dry.json"
    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--limit-runs",
        "32",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["summary"]["success_count"] == 32
    assert payload["summary"]["full_contract_checked"] is False


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE5_EFF_FULL") != "1",
    reason="仅手工触发真实仓库 full CLI 校验",
)
def test_real_repo_cli_full_run(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "real_full.jsonl"
    output_csv = tmp_path / "real_full.csv"
    output_report = tmp_path / "real_full.json"
    cmd = [
        sys.executable,
        "-m",
        "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_eff",
        "--output-jsonl",
        str(output_jsonl),
        "--output-csv",
        str(output_csv),
        "--output-report",
        str(output_report),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)
    payload = json.loads(output_report.read_text(encoding="utf-8"))
    assert payload["summary"]["success_count"] == 2250
    assert payload["summary"]["audited_repair_points"] == 15
    assert payload["summary"]["future_backfill_ignored_points"] == 34
