from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.freeze_binding import (
    build_freeze_binding_summary,
    main,
    validate_freeze_binding_summary,
)


STAGE5_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source(*, host: str, task_id: str) -> dict[str, object]:
    source = {
        "batch": "demo_batch",
        "algorithm": "demo",
        "host": host,
        "seed": 520,
        "noise_tag": "clean",
        "task_id": task_id,
        "dataset_id": "dataset",
        "path": f"/remote/{task_id}/result.json",
    }
    source["source_row_sha256"] = __import__("hashlib").sha256(
        _canonical_json(source).encode("utf-8")
    ).hexdigest()
    return source


def _result(raw_text: str) -> dict[str, object]:
    sha256 = __import__("hashlib").sha256(raw_text.encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "exists": True,
        "path": "/remote/result.json",
        "sha256": sha256,
        "raw_text": raw_text,
    }


def _snapshot(minute: int, *, status: str, raw_text: str | None = None) -> dict[str, object]:
    payload = {
        "minute": minute,
        "status": status,
        "selected_sha256": None,
    }
    if status == "ok":
        assert raw_text is not None
        payload["selected_sha256"] = __import__("hashlib").sha256(
            raw_text.encode("utf-8")
        ).hexdigest()
        payload["raw_text"] = raw_text
    return payload


def _inventory_snapshot(minute: int, *, status: str, selected_sha256: str | None) -> dict[str, object]:
    return {
        "minute": minute,
        "status": status,
        "selected_sha256": selected_sha256,
    }


def _write_jsonl_gz(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_record(host: str, *, include_raw: bool) -> tuple[dict[str, object], dict[str, object]]:
    result_raw = json.dumps({"status": "ok", "equation": "x0"}, ensure_ascii=False)
    snap1_raw = json.dumps({"minute": 1, "equation": "x0"}, ensure_ascii=False)
    result_sha = __import__("hashlib").sha256(result_raw.encode("utf-8")).hexdigest()
    snap1_sha = __import__("hashlib").sha256(snap1_raw.encode("utf-8")).hexdigest()
    source = _source(host=host, task_id="demo_s520_clean_g0001")
    inventory = {
        "source": source,
        "result": {
            "status": "ok",
            "exists": True,
            "path": source["path"],
            "sha256": result_sha,
        },
        "snapshots": [
            _inventory_snapshot(1, status="ok", selected_sha256=snap1_sha),
            _inventory_snapshot(2, status="missing", selected_sha256=None),
        ],
    }
    freeze = {
        "source": source,
        "result": _result(result_raw),
        "snapshots": [
            _snapshot(1, status="ok", raw_text=snap1_raw),
            _snapshot(2, status="missing"),
        ],
    }
    if not include_raw:
        freeze["snapshots"][1] = {
            "minute": 2,
            "status": "ok",
            "selected_sha256": snap1_sha,
            "raw_text": snap1_raw,
        }
    return inventory, freeze


def test_cli_writes_binding_report_for_small_sample(tmp_path: Path) -> None:
    inventory_dir = tmp_path / "inventory"
    freeze_dir = tmp_path / "freeze"
    output_json = tmp_path / "reports/binding.json"
    inventory_record, freeze_record = _sample_record("iaaccn22", include_raw=True)

    _write_jsonl_gz(inventory_dir / "clean_inventory_iaaccn22.jsonl.gz", [inventory_record])
    _write_jsonl_gz(freeze_dir / "clean_freeze_iaaccn22.jsonl.gz", [freeze_record])
    _write_report(
        inventory_dir / "clean_inventory_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 2,
            "available_snapshots": 1,
            "missing_snapshots": 1,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": False,
            "verify_inner": True,
        },
    )
    _write_report(
        freeze_dir / "clean_freeze_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 2,
            "available_snapshots": 1,
            "missing_snapshots": 1,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": True,
            "verify_inner": False,
        },
    )

    rc = main(
        [
            "--inventory-dir",
            str(inventory_dir),
            "--freeze-dir",
            str(freeze_dir),
            "--horizon",
            "2",
            "--expected-hosts",
            "1",
            "--expected-tasks",
            "1",
            "--expected-points",
            "2",
            "--expected-existing-points",
            "1",
            "--expected-missing-points",
            "1",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["contract_ok"] is True
    assert payload["binding"]["drift_count"] == 0
    assert payload["inventory_counts"] == {
        "hosts": 1,
        "tasks": 1,
        "expected_points": 2,
        "existing_points": 1,
        "missing_points": 1,
    }
    assert payload["missing_point_details"] == [
        {
            "host": "iaaccn22",
            "logical_key": "demo::dataset::s520::clean",
            "minute": 2,
            "task_id": "demo_s520_clean_g0001",
        }
    ]


def test_cli_fails_nonzero_when_missing_point_is_silently_backfilled(tmp_path: Path) -> None:
    inventory_dir = tmp_path / "inventory"
    freeze_dir = tmp_path / "freeze"
    output_json = tmp_path / "reports/binding.json"
    inventory_record, freeze_record = _sample_record("iaaccn22", include_raw=False)

    _write_jsonl_gz(inventory_dir / "clean_inventory_iaaccn22.jsonl.gz", [inventory_record])
    _write_jsonl_gz(freeze_dir / "clean_freeze_iaaccn22.jsonl.gz", [freeze_record])
    _write_report(
        inventory_dir / "clean_inventory_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 2,
            "available_snapshots": 1,
            "missing_snapshots": 1,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": False,
            "verify_inner": True,
        },
    )
    _write_report(
        freeze_dir / "clean_freeze_iaaccn22.report.json",
        {
            "tasks": 1,
            "expected_snapshots": 2,
            "available_snapshots": 2,
            "missing_snapshots": 0,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
            "result_missing_or_invalid": 0,
            "freeze_raw": True,
            "verify_inner": False,
        },
    )

    rc = main(
        [
            "--inventory-dir",
            str(inventory_dir),
            "--freeze-dir",
            str(freeze_dir),
            "--horizon",
            "2",
            "--expected-hosts",
            "1",
            "--expected-tasks",
            "1",
            "--expected-points",
            "2",
            "--expected-existing-points",
            "1",
            "--expected-missing-points",
            "1",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["contract_ok"] is False
    assert payload["binding"]["drift_count"] >= 1
    assert any(
        item["type"] == "fatal_error" or item["type"] == "snapshot_status_mismatch"
        for item in payload["binding"]["drift_details"]
    )


def test_real_clean_freeze_binding_matches_expected_counts() -> None:
    summary = build_freeze_binding_summary(
        inventory_dir=STAGE5_ROOT / "source_snapshot/trajectory_inventory",
        freeze_dir=STAGE5_ROOT / "source_snapshot/trajectory_freeze",
        noise_tag="clean",
        horizon=180,
    )

    assert summary["inventory_counts"] == {
        "hosts": 8,
        "tasks": 2250,
        "expected_points": 405000,
        "existing_points": 404985,
        "missing_points": 15,
    }
    assert summary["freeze_counts"] == summary["inventory_counts"]
    assert summary["binding"]["drift_count"] == 0
    assert summary["contract_ok"] is True
    assert len(summary["missing_point_details"]) == 15
    assert [item["minute"] for item in summary["missing_point_details"]] == list(range(165, 180))
    assert {item["logical_key"] for item in summary["missing_point_details"]} == {
        "fepysr::Nguyen-12::s520::clean"
    }
    validate_freeze_binding_summary(summary)
