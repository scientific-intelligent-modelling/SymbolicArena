from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "benchmark-control" / "compliance" / "lib"


def load_compliance_module(name: str):
    path = LIB / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"formal3h_test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    previous_module = sys.modules.get(spec.name)
    try:
        sys.path.insert(0, str(LIB))
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module


def write_formal3h_one_task_batch(batch_dir: Path, *, min_runtime_seconds: int = 10500) -> None:
    (batch_dir / "manifest").mkdir(parents=True)
    (batch_dir / "queues").mkdir()
    (batch_dir / "manifest" / "tasks.csv").write_text(
        "task_id,algorithm,dataset_id,dataset_dir,seed,noise_tag,noise_sigma,timeout_in_seconds,min_runtime_seconds,progress_snapshot_interval_seconds\n"
        f"dso__seed521__clean__g0032,dso,g0032,sim-datasets-data/ssr50/g0032,521,clean,0.0,10800,{min_runtime_seconds},60\n",
        encoding="utf-8",
    )
    (batch_dir / "queues" / "ssr50_source.csv").write_text(
        "global_index,dataset_id,dataset_name,dataset_dir,dataset_rel\n"
        "32,g0032,g0032,sim-datasets-data/ssr50/g0032,sim-datasets-data/ssr50/g0032\n",
        encoding="utf-8",
    )


def test_formal3h_spec_matches_goal_budget() -> None:
    models = load_compliance_module("models")

    assert models.FORMAL3H_SPEC.name == "formal3h_13alg_3seed_3noise"
    assert models.FORMAL3H_SPEC.algorithms == (
        "gplearn",
        "pyoperon",
        "pysr",
        "dso",
        "tpsr",
        "e2esr",
        "fepysr",
        "jaxsr",
        "QLattice",
        "iMCTS",
        "udsr",
        "ragsr",
        "symbolfit",
    )
    assert models.FORMAL3H_SPEC.seeds == (520, 521, 522)
    assert [level.tag for level in models.FORMAL3H_SPEC.noise_levels] == ["clean", "noise001", "noise005"]
    assert models.FORMAL3H_SPEC.budget.timeout_in_seconds == 10800
    assert models.FORMAL3H_SPEC.budget.min_runtime_seconds == 10500
    assert models.FORMAL3H_SPEC.budget.progress_snapshot_interval_seconds == 60
    assert models.get_experiment_spec("formal3h_13alg_3seed_3noise") is models.FORMAL3H_SPEC


def test_prepare_formal3h_generates_manifest_and_params(tmp_path: Path) -> None:
    batch_dir = tmp_path / "benchmark-runs" / "formal3h" / "batch"
    result = subprocess.run(
        [
            "python",
            "benchmark-control/compliance/launchers/prepare_batch.py",
            "--profile",
            "formal3h_13alg_3seed_3noise",
            "--batch-dir",
            str(batch_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "5850" in result.stdout
    tasks = (batch_dir / "manifest" / "tasks.csv").read_text(encoding="utf-8")
    assert "10800,10500,60" in tasks
    assert (batch_dir / "params" / "gplearn__clean.json").exists()
    clean = json.loads((batch_dir / "params" / "gplearn__clean.json").read_text(encoding="utf-8"))
    smoke = json.loads((batch_dir / "params_smoke" / "gplearn__clean.json").read_text(encoding="utf-8"))
    assert clean["timeout_in_seconds"] == 10800
    assert smoke["timeout_in_seconds"] == 600


def test_recover_formal3h_snapshot_into_runs_layout(tmp_path: Path) -> None:
    recovery = load_compliance_module("snapshot_recovery")
    batch_dir = tmp_path / "formal3h"
    source_root = tmp_path / "experiments" / "formal24h_batch"
    run_dir = (
        source_root
        / "dso"
        / "seed521"
        / "tasks"
        / "dso_s521_clean_g0032"
        / "iaaccn22"
        / "dso"
        / "g0032_case"
    )
    progress_dir = run_dir / "progress"
    progress_dir.mkdir(parents=True)
    snapshot = {
        "status": "ok",
        "elapsed_seconds": 10809.949,
        "equation": "x0 + 1",
        "canonical_artifact": {"sympy": "x0 + 1"},
        "valid": {"nmse": 0.1},
        "id_test": {"nmse": 0.2},
        "ood_test": {"nmse": 0.3},
    }
    (progress_dir / "minute_0180.json").write_text(json.dumps(snapshot), encoding="utf-8")
    write_formal3h_one_task_batch(batch_dir)

    summary = recovery.recover_snapshots(
        batch_dir=batch_dir,
        source_roots=[source_root],
        source_batch="formal24h_batch",
        snapshot_name="minute_0180.json",
    )

    assert summary["recovered"] == 1
    target = batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032"
    assert json.loads((target / "result.json").read_text(encoding="utf-8"))["recovered_from_24h"] is True
    assert (target / "progress" / "minute_0180.json").exists()
    assert (target / "recovery_source.json").exists()
    rows = list(csv.DictReader((batch_dir / "recovery" / "recovered_tasks.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["task_id"] == "dso__seed521__clean__g0032"


def test_snapshot_index_maps_scheduler_task_ids(tmp_path: Path) -> None:
    recovery = load_compliance_module("snapshot_recovery")
    source_root = tmp_path / "experiments" / "formal24h_batch"
    snapshot_path = (
        source_root
        / "dso"
        / "seed521"
        / "tasks"
        / "dso_s521_clean_g0032"
        / "iaaccn22"
        / "dso"
        / "g0032_case"
        / "progress"
        / "minute_0180.json"
    )
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text("{}", encoding="utf-8")

    index = recovery.index_snapshot_candidates(
        source_roots=[source_root],
        snapshot_name="minute_0180.json",
    )

    assert index == {"dso_s521_clean_g0032": [snapshot_path]}


def test_recover_formal3h_rejects_invalid_snapshot(tmp_path: Path) -> None:
    recovery = load_compliance_module("snapshot_recovery")
    batch_dir = tmp_path / "formal3h"
    source_root = tmp_path / "experiments" / "formal24h_batch"
    progress_dir = (
        source_root
        / "dso"
        / "seed521"
        / "tasks"
        / "dso_s521_clean_g0032"
        / "iaaccn22"
        / "dso"
        / "g0032_case"
        / "progress"
    )
    progress_dir.mkdir(parents=True)
    (progress_dir / "minute_0180.json").write_text(
        json.dumps({"status": "error", "elapsed_seconds": 10809.0}),
        encoding="utf-8",
    )
    write_formal3h_one_task_batch(batch_dir)

    summary = recovery.recover_snapshots(
        batch_dir=batch_dir,
        source_roots=[source_root],
        source_batch="formal24h_batch",
        snapshot_name="minute_0180.json",
    )

    assert summary["recovered"] == 0
    assert summary["invalid"] == 1
    assert not (batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032" / "result.json").exists()
    rows = list(
        csv.DictReader((batch_dir / "recovery" / "invalid_snapshot_tasks.csv").open(newline="", encoding="utf-8"))
    )
    assert rows[0]["reason"] == "snapshot status is error"


def test_audit_accepts_elapsed_seconds_for_recovered_snapshot(tmp_path: Path) -> None:
    audit = load_compliance_module("audit")
    batch_dir = tmp_path / "formal3h"
    write_formal3h_one_task_batch(batch_dir, min_runtime_seconds=10500)
    run_dir = batch_dir / "runs" / "dso" / "seed521" / "clean" / "g0032"
    (run_dir / "progress").mkdir(parents=True)
    (run_dir / "progress" / "minute_0180.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "elapsed_seconds": 10809.949,
                "equation": "x0 + 1",
                "canonical_artifact": {"sympy": "x0 + 1"},
                "valid": {"nmse": 0.1},
                "id_test": {"nmse": 0.2},
                "ood_test": {"nmse": 0.3},
                "recovered_from_24h": True,
            }
        ),
        encoding="utf-8",
    )

    summary = audit.audit_batch(batch_dir=batch_dir)

    assert summary == {"total_tasks": 1, "failed": 0, "passed": 1}


def test_readiness_passes_for_formal3h_batch(tmp_path: Path) -> None:
    batch_dir = tmp_path / "formal3h"
    subprocess.run(
        [
            "python",
            "benchmark-control/compliance/launchers/prepare_batch.py",
            "--profile",
            "formal3h_13alg_3seed_3noise",
            "--batch-dir",
            str(batch_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "python",
            "benchmark-control/compliance/launchers/write_full3h_queue_commands.py",
            "--batch-dir",
            str(batch_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    readiness = load_compliance_module("readiness")

    result = readiness.check_readiness(batch_dir=batch_dir, profile="formal3h_13alg_3seed_3noise")

    assert result["ready"] is True
    assert result["total_tasks"] == 5850
    assert result["total_algorithms"] == 13
    assert result["total_noise_levels"] == 3
