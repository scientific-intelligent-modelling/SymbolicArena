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
    global_index: int,
    dataset_name: str,
    payload: dict[str, object],
) -> None:
    result_dir = (
        experiment_root
        / tool_key
        / f"seed{seed}"
        / "tasks"
        / f"{tool_key}_s{seed}_g{global_index:04d}"
        / "iaaccn22"
        / tool_arg
        / f"g{global_index:04d}_{dataset_name}"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
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
