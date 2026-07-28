from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(value: float, *, record_type: str = "periodic_best") -> dict:
    return {
        "record_type": record_type,
        "equation": "x0",
        "id_test": {"nmse": value},
        "ood_test": {"nmse": value},
    }


def test_remote_extractor_rejects_future_backfill(tmp_path: Path) -> None:
    module = load_module(
        "stage4_remote_extractor_test",
        "check/extract_neurips_stage4_anytime_remote.py",
    )
    experiment = tmp_path / "experiment"
    progress = experiment / "progress"
    progress.mkdir(parents=True)
    (progress / "minute_0001.json").write_text(
        json.dumps(candidate(1.0)), encoding="utf-8"
    )
    future = candidate(0.01, record_type="periodic_backfill")
    future["backfilled_from_minute"] = 5
    (progress / "minute_0002.json").write_text(json.dumps(future), encoding="utf-8")
    task = {
        "algorithm": "demo",
        "gid": "g0001",
        "dataset": "demo",
        "seed": 0,
        "host": "local",
        "experiment_dir": str(experiment),
        "final_seconds": 3600,
        "final_payload": candidate(0.001),
    }

    rows = module.extract_task(task)

    assert rows[0]["id_test_nmse"] == 1.0
    assert rows[1]["id_test_nmse"] == 1.0
    assert rows[1]["source"] == "snapshot:1"
    assert rows[-1]["source"] == "final_carry_forward"


def test_dynamic_symf_module_loads_with_dataclasses() -> None:
    module = load_module(
        "stage4_anytime_analysis_test",
        "check/analyze_neurips_stage4_anytime.py",
    )

    symf = module.load_symf_module()

    assert callable(symf.run_formal_metrics)


def test_phi_failure_and_best_quality_bounds() -> None:
    module = load_module(
        "stage4_anytime_analysis_math_test",
        "check/analyze_neurips_stage4_anytime.py",
    )

    assert module.phi(1e2) == 0.0
    assert module.phi(1e-12) == 1.0
    assert 0.0 <= module.retention(1e-4, 1.0) <= 1.0
    assert module.set_f1({"x0"}, {"x1"}) == 0.0
    assert module.structure_signature(sp.Symbol("x0") + 2) == "add(C,X)"


def test_top3_checkpoint_contract() -> None:
    module = load_module(
        "stage4_top3_checkpoints_test",
        "check/analyze_neurips_stage4_top3_checkpoints.py",
    )

    assert module.MINUTES == (10, 20, 30, 40, 50, 60)
    assert module.ALGORITHMS == ("udsr", "imcts", "pysr")
    assert module.METRICS == ("ID-Q", "OOD-G", "SYM-F", "EFF", "STAB")
