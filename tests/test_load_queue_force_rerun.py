from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_force_rerun_does_not_skip_existing_done_status() -> None:
    launcher = _load_module("launch_e1_benchmark", "check/launch_e1_benchmark.py")
    row = {"dataset_dir": "sim-datasets-data/ssr50/Keijzer-2"}
    completed = {
        "sim-datasets-data/ssr50/Keijzer-2|seed=522": {
            "status": "ok",
        }
    }

    assert launcher._should_skip(row, 522, completed, retry_failed=False, force_rerun=False) is True
    assert launcher._should_skip(row, 522, completed, retry_failed=False, force_rerun=True) is False


def test_load_queue_support_script_can_force_rerun_existing_results(tmp_path: Path) -> None:
    queue = _load_module("run_e1_candidate200_12alg_load_queue", "check/run_e1_candidate200_12alg_load_queue.py")

    script_path = queue._write_remote_support_script(tmp_path / "queue")
    script = script_path.read_text(encoding="utf-8")

    assert "RERUN_MODE" in script
    assert "--force-rerun" in script


def test_load_queue_support_script_limits_julia_and_pysr_threads(tmp_path: Path) -> None:
    queue = _load_module(
        "run_e1_candidate200_12alg_load_queue_thread_limits",
        "check/run_e1_candidate200_12alg_load_queue.py",
    )

    script_path = queue._write_remote_support_script(tmp_path / "queue")
    script = script_path.read_text(encoding="utf-8")

    assert "export JULIA_NUM_THREADS=1" in script
    assert "export JULIA_MAX_NUM_THREADS=1" in script
    assert "export JULIA_NUM_GC_THREADS=1" in script
    assert "export PYTHON_JULIACALL_THREADS=1" in script
    assert "export PYTHON_JULIACALL_PROCS=1" in script
    assert "export PYSR_PROCS=1" in script


def test_load_queue_support_script_limits_jax_cpu_threads(tmp_path: Path) -> None:
    queue = _load_module(
        "run_e1_candidate200_12alg_load_queue_jax_limits",
        "check/run_e1_candidate200_12alg_load_queue.py",
    )

    script_path = queue._write_remote_support_script(tmp_path / "queue")
    script = script_path.read_text(encoding="utf-8")

    assert "export JAX_NUM_THREADS=1" in script
    assert "--xla_cpu_multi_thread_eigen=false" in script
    assert "intra_op_parallelism_threads=1" in script
