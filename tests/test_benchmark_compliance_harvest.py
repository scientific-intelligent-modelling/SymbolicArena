from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

from benchmark_control_compliance_manifest_import import load_for_test


def _write_manifest(batch_dir: Path) -> None:
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tasks.csv").write_text(
        "\n".join(
            [
                "task_id,algorithm,dataset_id,dataset_dir,seed,timeout_in_seconds,progress_snapshot_interval_seconds",
                "QLattice__seed520__Keijzer-11,QLattice,Keijzer-11,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-11,520,3600,60",
                "iMCTS__seed520__Keijzer-2,iMCTS,Keijzer-2,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-2,520,3600,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queue_dir = batch_dir / "queues"
    queue_dir.mkdir()
    (queue_dir / "ssr50_source.csv").write_text(
        "\n".join(
            [
                "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel",
                "1,Keijzer-11,Keijzer-11,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-11,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-11",
                "2,Keijzer-2,Keijzer-2,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-2,sim-datasets-data/ssr50/datasets/keijzer/Keijzer-2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_remote_result(
    experiment_root: Path,
    *,
    tool_key: str,
    tool_arg: str,
    seed: int,
    noise_tag: str | None = None,
    global_index: int,
    dataset_name: str,
    payload: dict[str, object],
    report_payload: dict[str, object] | None = None,
) -> None:
    task_id = f"{tool_key}_s{seed}_g{global_index:04d}"
    if noise_tag:
        task_id = f"{tool_key}_s{seed}_{noise_tag}_g{global_index:04d}"
    result_dir = (
        experiment_root
        / tool_key
        / f"seed{seed}"
        / "tasks"
        / task_id
        / "iaaccn22"
        / tool_arg
        / f"g{global_index:04d}_{dataset_name}"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    if report_payload is not None:
        log_dir = (
            experiment_root
            / tool_key
            / f"seed{seed}"
            / "tasks"
            / task_id
            / "iaaccn22"
            / "__launcher__"
            / "logs"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{global_index}_{dataset_name}.report.json").write_text(
            json.dumps(report_payload),
            encoding="utf-8",
        )
    progress_dir = result_dir / "progress"
    progress_dir.mkdir()
    (progress_dir / "minute_0001.json").write_text("{}", encoding="utf-8")


def test_harvest_maps_scheduler_outputs_into_audit_runs_layout(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    audit = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir)
    experiment_root = tmp_path / "experiments" / "batch"
    valid_payload = {
        "status": "ok",
        "runtime_seconds": 3501.0,
        "valid": {"nmse": 0.1},
        "id_test": {"nmse": 0.1},
        "ood_test": {"nmse": 0.1},
        "canonical_artifact": {"expression": "x0"},
    }
    _write_remote_result(
        experiment_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        global_index=1,
        dataset_name="Keijzer-11",
        payload=valid_payload,
    )
    _write_remote_result(
        experiment_root,
        tool_key="imcts",
        tool_arg="iMCTS",
        seed=520,
        global_index=2,
        dataset_name="Keijzer-2",
        payload={**valid_payload, "runtime_seconds": 3502.0},
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[experiment_root])

    assert summary == {"total_tasks": 2, "harvested": 2, "missing": 0}
    qlattice_result = batch_dir / "runs" / "QLattice" / "seed520" / "Keijzer-11" / "result.json"
    imcts_result = batch_dir / "runs" / "iMCTS" / "seed520" / "Keijzer-2" / "result.json"
    assert json.loads(qlattice_result.read_text(encoding="utf-8"))["runtime_seconds"] == 3501.0
    assert json.loads(imcts_result.read_text(encoding="utf-8"))["runtime_seconds"] == 3502.0
    assert (qlattice_result.parent / "progress" / "minute_0001.json").exists()

    with (batch_dir / "harvest" / "harvested_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == [
        "QLattice__seed520__Keijzer-11",
        "iMCTS__seed520__Keijzer-2",
    ]

    audit_summary = audit.audit_batch(batch_dir=batch_dir)
    assert audit_summary == {"total_tasks": 2, "failed": 0, "passed": 2}


def test_harvest_maps_noise_aware_outputs_into_noise_runs_layout(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    batch_dir = tmp_path / "batch"
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tasks.csv").write_text(
        "\n".join(
            [
                "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds",
                "pysr__seed520__noise001__d0,pysr,d0,sim-datasets-data/ssr50/d0,520,noise001,0.01,86400,82800,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queue_dir = batch_dir / "queues"
    queue_dir.mkdir()
    (queue_dir / "ssr50_source.csv").write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        "1,d0,d0,sim-datasets-data/ssr50/d0,sim-datasets-data/ssr50/d0\n",
        encoding="utf-8",
    )
    experiment_root = tmp_path / "experiments" / "batch"
    _write_remote_result(
        experiment_root,
        tool_key="pysr",
        tool_arg="pysr",
        seed=520,
        noise_tag="noise001",
        global_index=1,
        dataset_name="d0",
        payload={"status": "ok", "runtime_seconds": 83000.0},
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[experiment_root])

    assert summary == {"total_tasks": 1, "harvested": 1, "missing": 0}
    assert (batch_dir / "runs" / "pysr" / "seed520" / "noise001" / "d0" / "result.json").exists()


def test_harvest_counts_existing_recovered_result_without_remote_candidate(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    batch_dir = tmp_path / "batch"
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tasks.csv").write_text(
        "\n".join(
            [
                "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds",
                "dso__seed521__clean__g0032,dso,g0032,sim-datasets-data/ssr50/g0032,521,clean,0.0,10800,10500,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queue_dir = batch_dir / "queues"
    queue_dir.mkdir()
    (queue_dir / "ssr50_source.csv").write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        "32,g0032,g0032,sim-datasets-data/ssr50/g0032,sim-datasets-data/ssr50/g0032\n",
        encoding="utf-8",
    )
    run_dir = batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032"
    (run_dir / "progress").mkdir(parents=True)
    (run_dir / "progress" / "minute_0180.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "runtime_seconds": 10809.0,
                "valid": {"nmse": 0.1},
                "id_test": {"nmse": 0.2},
                "ood_test": {"nmse": 0.3},
                "canonical_artifact": {"expression": "x0 + 1"},
                "recovered_from_24h": True,
            }
        ),
        encoding="utf-8",
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[tmp_path / "empty"])

    assert summary == {"total_tasks": 1, "harvested": 1, "missing": 0}
    rows = list(csv.DictReader((batch_dir / "harvest" / "harvested_tasks.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["source_result"].endswith("runs/dso/seed521/clean/g0032/result.json")


def test_harvest_preserves_launcher_report_runtime_for_budget_audit(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    audit = load_for_test("audit")
    batch_dir = tmp_path / "batch"
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "tasks.csv").write_text(
        "\n".join(
            [
                "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds",
                "QLattice__seed520__clean__d0,QLattice,d0,sim-datasets-data/ssr50/d0,520,clean,0,86400,82800,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queue_dir = batch_dir / "queues"
    queue_dir.mkdir()
    (queue_dir / "ssr50_source.csv").write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        "1,d0,d0,sim-datasets-data/ssr50/d0,sim-datasets-data/ssr50/d0\n",
        encoding="utf-8",
    )
    experiment_root = tmp_path / "remote-experiments" / "iaaccn22"
    _write_remote_result(
        experiment_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        noise_tag="clean",
        global_index=1,
        dataset_name="d0",
        payload={
            "status": "ok",
            "valid": {"nmse": 0.1},
            "id_test": {"nmse": 0.1},
            "ood_test": {"nmse": 0.1},
            "canonical_artifact": {"expression": "x0"},
        },
        report_payload={
            "status": "ok",
            "seconds": 86404.725,
            "budget_exhausted": True,
            "recovered_from_timeout": True,
            "termination_reason": "budget_exhausted_with_output",
            "timeout_type": "budget_exhausted_with_output",
        },
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[experiment_root])

    assert summary == {"total_tasks": 1, "harvested": 1, "missing": 0}
    result_path = batch_dir / "runs" / "QLattice" / "seed520" / "clean" / "d0" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["runtime_seconds"] == 86404.725
    assert result["budget_exhausted"] is True
    assert (result_path.parent / "launcher_report.json").exists()
    assert audit.audit_batch(batch_dir=batch_dir) == {"total_tasks": 1, "failed": 0, "passed": 1}


def test_harvest_prefers_candidate_from_state_assigned_host(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir)
    state_dir = batch_dir / "queues" / "load_queue_full" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "formal24h.state.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "qlattice_s520_g0001": {
                        "state": "running",
                        "assigned_host": "iaaccn23",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assigned_root = tmp_path / "remote-experiments" / "iaaccn23"
    duplicate_root = tmp_path / "remote-experiments" / "iaaccn25"
    _write_remote_result(
        assigned_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        global_index=1,
        dataset_name="Keijzer-11",
        payload={"status": "ok", "runtime_seconds": 3501.0},
    )
    _write_remote_result(
        duplicate_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        global_index=1,
        dataset_name="Keijzer-11",
        payload={"status": "ok", "runtime_seconds": 1.0},
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[assigned_root, duplicate_root])

    assert summary["harvested"] == 1
    result = batch_dir / "runs" / "QLattice" / "seed520" / "Keijzer-11" / "result.json"
    assert json.loads(result.read_text(encoding="utf-8"))["runtime_seconds"] == 3501.0
    source = json.loads((result.parent / "harvest_source.json").read_text(encoding="utf-8"))
    assert "iaaccn23" in source["source_result"]


def test_harvest_marks_missing_when_assigned_host_has_no_candidate(tmp_path: Path) -> None:
    harvest = load_for_test("harvest")
    batch_dir = tmp_path / "batch"
    _write_manifest(batch_dir)
    state_dir = batch_dir / "queues" / "load_queue_full" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "formal24h.state.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "qlattice_s520_g0001": {
                        "state": "running",
                        "assigned_host": "iaaccn23",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    duplicate_root = tmp_path / "remote-experiments" / "iaaccn25"
    _write_remote_result(
        duplicate_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        global_index=1,
        dataset_name="Keijzer-11",
        payload={"status": "ok", "runtime_seconds": 1.0},
    )

    summary = harvest.harvest_batch(batch_dir=batch_dir, experiment_roots=[duplicate_root])

    assert summary == {"total_tasks": 2, "harvested": 0, "missing": 2}
    result = batch_dir / "runs" / "QLattice" / "seed520" / "Keijzer-11" / "result.json"
    assert not result.exists()
    with (batch_dir / "harvest" / "harvested_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "missing"
    assert rows[0]["reason"] == "result not found for qlattice_s520_g0001"


def test_harvest_launcher_accepts_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "benchmark-control"
        / "compliance"
        / "launchers"
        / "harvest_batch.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_compliance_harvest_batch", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    fake_root = tmp_path / "repo"
    batch_dir = fake_root / "benchmark-runs" / "compliance" / "batch"
    _write_manifest(batch_dir)
    experiment_root = fake_root / "experiments" / "batch"
    _write_remote_result(
        experiment_root,
        tool_key="qlattice",
        tool_arg="QLattice",
        seed=520,
        global_index=1,
        dataset_name="Keijzer-11",
        payload={"status": "ok", "runtime_seconds": 3501.0},
    )
    monkeypatch.setattr(launcher, "ROOT", fake_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harvest_batch.py",
            "--batch-dir",
            "benchmark-runs/compliance/batch",
            "--experiment-root",
            "experiments/batch",
        ],
    )

    assert launcher.main() == 0
    assert (
        batch_dir / "runs" / "QLattice" / "seed520" / "Keijzer-11" / "result.json"
    ).exists()
