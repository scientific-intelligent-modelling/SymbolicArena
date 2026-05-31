from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark_control_compliance_manifest_import import load_for_test


def _write_manifest(batch_dir: Path, dataset_ids: tuple[str, ...]) -> None:
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "task_id,algorithm,dataset_id,dataset_dir,seed,timeout_in_seconds,progress_snapshot_interval_seconds"
    ]
    for dataset_id in dataset_ids:
        rows.append(
            f"alg__seed520__{dataset_id},alg,{dataset_id},/data/{dataset_id},520,3600,60"
        )
    (manifest_dir / "tasks.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_task_audit(batch_dir: Path, dataset_ids: tuple[str, ...]) -> None:
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason"
    ]
    for dataset_id in dataset_ids:
        rows.append(
            f"alg__seed520__{dataset_id},alg,{dataset_id},520,ok,3501.000,true,true,true,true,,"
        )
    (audit_dir / "task_audit.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_success_result(batch_dir: Path, dataset_id: str) -> None:
    task_dir = batch_dir / "runs" / "alg" / "seed520" / dataset_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "runtime_seconds": 3501.0,
                "valid": {"nmse": 0.1},
                "id_test": {"nmse": 0.1},
                "ood_test": {"nmse": 0.1},
                "canonical_artifact": {"expression": "x0"},
            }
        ),
        encoding="utf-8",
    )
    progress_dir = task_dir / "progress"
    progress_dir.mkdir()
    (progress_dir / "minute_0001.json").write_text("{}", encoding="utf-8")
    (progress_dir / "minute_0055.json").write_text("{}", encoding="utf-8")


def _run_audit_cli(batch_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "benchmark-control/compliance/launchers/audit_batch.py",
            "--batch-dir",
            str(batch_dir),
            *extra_args,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


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


def test_heartbeat_marks_missing_audit_as_needing_codex(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1", "d2"))

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="repair")

    assert payload["total_tasks"] == 2
    assert payload["finished"] == 0
    assert payload["failed"] == 0
    assert payload["needs_codex"] is True
    assert payload["codex_reason"] == "audit/failure_cases.csv missing"
    assert payload["latest_audit"] == ""


def test_heartbeat_uses_load_queue_state_before_audit_exists(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1", "d2", "d3", "d4", "d5"))
    state_dir = batch_dir / "queues" / "load_queue_full" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "batch.state.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "pending_task": {"state": "pending"},
                    "running_task": {"state": "running"},
                    "done_task": {"state": "done"},
                    "failed_task": {"state": "failed"},
                    "dispatching_task": {"state": "dispatching"},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="running")

    assert payload["pending"] == 2
    assert payload["running"] == 1
    assert payload["finished"] == 1
    assert payload["failed"] == 1
    assert payload["needs_codex"] is True


def test_heartbeat_resolves_latest_symlink_to_real_batch_id(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "compliance_15alg_ssr50_seed520_1h_20260529-235959"
    latest = tmp_path / "latest"
    _write_manifest(batch_dir, ("d1",))
    latest.symlink_to(batch_dir, target_is_directory=True)

    payload = heartbeat.write_heartbeat(batch_dir=latest, phase="preparing")

    assert payload["batch_id"] == batch_dir.name
    saved = json.loads((batch_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert saved["batch_id"] == batch_dir.name


def test_heartbeat_treats_header_only_failures_as_clean_audit(tmp_path: Path) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1", "d2", "d3"))
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    _write_task_audit(batch_dir, ("d1", "d2", "d3"))
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n",
        encoding="utf-8",
    )

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="done")

    assert payload["total_tasks"] == 3
    assert payload["finished"] == 3
    assert payload["failed"] == 0
    assert payload["needs_codex"] is False
    assert payload["codex_reason"] == ""
    assert payload["latest_audit"] == "audit/failure_cases.csv"


def test_heartbeat_marks_header_only_failures_without_task_audit_as_needing_codex(
    tmp_path: Path,
) -> None:
    heartbeat = load_for_test("heartbeat")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1", "d2"))
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n",
        encoding="utf-8",
    )

    payload = heartbeat.write_heartbeat(batch_dir=batch_dir, phase="repair")

    assert payload["total_tasks"] == 2
    assert payload["finished"] == 0
    assert payload["failed"] == 0
    assert payload["needs_codex"] is True
    assert payload["codex_reason"] == "audit/task_audit.csv missing or incomplete"
    assert payload["latest_audit"] == "audit/failure_cases.csv"


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


def test_rerun_queue_raises_when_failure_cases_missing(tmp_path: Path) -> None:
    rerun = load_for_test("rerun")

    with pytest.raises(FileNotFoundError, match="audit/failure_cases.csv"):
        rerun.write_rerun_queue(batch_dir=tmp_path / "batch", round_id=1)


def test_rerun_queue_allows_header_only_failure_cases(tmp_path: Path) -> None:
    rerun = load_for_test("rerun")
    batch_dir = tmp_path / "batch"
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "failure_cases.csv").write_text(
        "task_id,algorithm,dataset_id,seed,status,runtime_seconds,has_result,has_progress,metrics_valid,artifact_valid,failure_class,reason\n",
        encoding="utf-8",
    )

    output = rerun.write_rerun_queue(batch_dir=batch_dir, round_id=1)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []


def test_audit_cli_writes_heartbeat_and_rerun_queue_for_failed_batch(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1",))

    completed = _run_audit_cli(
        batch_dir,
        "--write-heartbeat",
        "--write-rerun",
        "--round-id",
        "1",
    )

    assert completed.returncode == 0, completed.stderr
    heartbeat_path = batch_dir / "heartbeat.json"
    rerun_path = batch_dir / "repair" / "round_001" / "rerun_tasks.csv"
    assert heartbeat_path.exists()
    assert rerun_path.exists()
    heartbeat_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat_payload["phase"] == "repair"
    assert heartbeat_payload["latest_rerun_queue"] == "repair/round_001/rerun_tasks.csv"
    with rerun_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == ["alg__seed520__d1"]


def test_audit_cli_skips_rerun_queue_when_batch_is_clean(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir, ("d1",))
    _write_success_result(batch_dir, "d1")

    completed = _run_audit_cli(
        batch_dir,
        "--write-heartbeat",
        "--write-rerun",
        "--round-id",
        "2",
    )

    assert completed.returncode == 0, completed.stderr
    heartbeat_path = batch_dir / "heartbeat.json"
    rerun_path = batch_dir / "repair" / "round_002" / "rerun_tasks.csv"
    assert heartbeat_path.exists()
    assert not rerun_path.exists()
    heartbeat_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat_payload["phase"] == "done"
    assert heartbeat_payload["failed"] == 0
