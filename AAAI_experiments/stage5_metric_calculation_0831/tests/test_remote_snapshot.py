from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.remote_snapshot import (
    scan_task,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def snapshot(minute: int, equation: str = "x0") -> dict[str, object]:
    return {
        "record_type": "periodic_best",
        "checkpoint_index": minute,
        "elapsed_minutes": minute,
        "equation": equation,
        "id_test": {"nmse": 0.1},
        "ood_test": {"nmse": 0.2},
    }


def make_task(tmp_path: Path, *, conflict: bool = False) -> dict[str, object]:
    outer = tmp_path / "run"
    inner = outer / "experiments" / "one"
    result = {
        "status": "ok",
        "equation": "x0",
        "experiment_dir": str(inner),
        "id_test": {"nmse": 0.1},
        "ood_test": {"nmse": 0.2},
    }
    write_json(outer / "result.json", result)
    for minute in range(1, 4):
        payload = snapshot(minute)
        write_json(outer / "progress" / f"minute_{minute:04d}.json", payload)
        if conflict and minute == 2:
            payload = snapshot(minute, equation="x1")
        write_json(inner / "progress" / f"minute_{minute:04d}.json", payload)
    return {
        "algorithm": "demo",
        "dataset_id": "dataset",
        "seed": 520,
        "noise_tag": "clean",
        "task_id": "demo_s520_clean_g0001",
        "host": "local",
        "path": str(outer / "result.json"),
    }


def test_inventory_scans_result_and_semantic_duplicate_snapshots(tmp_path: Path) -> None:
    record = scan_task(make_task(tmp_path), horizon=3, freeze_raw=False)
    assert record["result"]["exists"] is True
    assert record["result"]["sha256"]
    assert "raw_text" not in record["result"]
    assert record["summary"] == {
        "expected_snapshots": 3,
        "available_snapshots": 3,
        "missing_snapshots": 0,
        "conflicting_snapshots": 0,
        "parse_errors": 0,
    }
    assert all(item["duplicate_semantically_equal"] for item in record["snapshots"])


def test_freeze_mode_retains_exact_raw_text(tmp_path: Path) -> None:
    record = scan_task(make_task(tmp_path), horizon=3, freeze_raw=True)
    assert json.loads(record["result"]["raw_text"])["equation"] == "x0"
    assert json.loads(record["snapshots"][0]["raw_text"])["checkpoint_index"] == 1


def test_conflicting_outer_and_inner_snapshot_is_not_silently_selected(tmp_path: Path) -> None:
    record = scan_task(make_task(tmp_path, conflict=True), horizon=3, freeze_raw=False)
    assert record["summary"]["conflicting_snapshots"] == 1
    minute_two = record["snapshots"][1]
    assert minute_two["conflict"] is True
    assert minute_two["selected_path"] is None


def test_outer_only_freeze_skips_already_audited_inner_copy(tmp_path: Path) -> None:
    record = scan_task(
        make_task(tmp_path, conflict=True),
        horizon=3,
        freeze_raw=True,
        verify_inner=False,
    )
    assert record["summary"]["conflicting_snapshots"] == 0
    assert all(item["inner_path"] is None for item in record["snapshots"])
    assert json.loads(record["snapshots"][1]["raw_text"])["equation"] == "x0"


def test_missing_snapshot_is_explicit(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    result_path = Path(task["path"])
    (result_path.parent / "progress/minute_0002.json").unlink()
    inner = Path(json.loads(result_path.read_text())["experiment_dir"])
    (inner / "progress/minute_0002.json").unlink()
    record = scan_task(task, horizon=3, freeze_raw=False)
    assert record["summary"]["missing_snapshots"] == 1
    assert record["snapshots"][1]["status"] == "missing"


def test_controller_returns_nonzero_when_any_host_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_timeout = fake_bin / "timeout"
    fake_timeout.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_timeout.chmod(0o755)
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/scan_clean_via_iaaccn22.sh"
    )
    completed = subprocess.run(
        ["/bin/bash", str(script), str(tmp_path / "remote"), "inventory"],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "FAILED_HOSTS" in completed.stdout
