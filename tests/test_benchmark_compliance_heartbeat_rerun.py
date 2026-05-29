from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark_control_compliance_manifest_import import load_for_test


def test_heartbeat_marks_codex_needed_when_failures_exist(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n"
        "alg__seed520__d1,alg,d1,520,failed,12.0,true,true,true,true,early_stop,short\n",
        encoding="utf-8",
    )

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="repair")

    assert payload["needs_codex"] is True
    assert payload["failed"] == 1
    saved = json.loads((batch_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert saved["phase"] == "repair"


def test_heartbeat_rejects_invalid_phase(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")

    with pytest.raises(ValueError, match="invalid heartbeat phase: invalid"):
        heartbeat.write_heartbeat(batch_dir=tmp_path / "batch", phase="invalid")


def test_rerun_queue_contains_only_failed_tasks(tmp_path: Path) -> None:
    rerun = load_for_test("rerun")
    batch_dir = tmp_path / "batch"
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n"
        "alg__seed520__d1,alg,d1,520,failed,12.0,true,true,true,true,early_stop,short\n",
        encoding="utf-8",
    )

    output = rerun.write_rerun_queue(batch_dir=batch_dir, round_id=1)

    assert output == batch_dir / "repair" / "round_001" / "rerun_tasks.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "task_id": "alg__seed520__d1",
            "algorithm": "alg",
            "dataset_id": "d1",
            "seed": "520",
            "failure_class": "early_stop",
        }
    ]
