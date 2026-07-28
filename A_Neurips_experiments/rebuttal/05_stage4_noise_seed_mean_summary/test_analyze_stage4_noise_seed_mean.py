from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("analyze_stage4_noise_seed_mean.py")
SPEC = importlib.util.spec_from_file_location("stage4_noise_seed_mean", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phi_and_failure_penalty(tmp_path: Path) -> None:
    assert MODULE.phi_from_nmse(1e-12) == 1.0
    assert MODULE.phi_from_nmse(1e2) == 0.0
    clean = pd.read_csv(MODULE.DEFAULT_CLEAN, nrows=1)
    noise = pd.read_csv(MODULE.DEFAULT_NOISE, nrows=1)
    clean.loc[0, ["valid_output", "metric_complete"]] = False
    clean.loc[0, ["id_test_nmse", "ood_test_nmse"]] = np.nan
    noise.loc[0, ["valid_output", "metric_complete"]] = False
    clean_path = tmp_path / "clean.csv"
    noise_path = tmp_path / "noise.csv"
    clean.to_csv(clean_path, index=False)
    noise.to_csv(noise_path, index=False)
    runs = MODULE.prepare_runs(clean_path, noise_path)
    assert (runs["id_nmse_used"] == 100.0).all()
    assert (runs["ood_nmse_used"] == 100.0).all()
    assert (runs["joint_quality"] == 0.0).all()


def test_stage4_grid_and_outputs() -> None:
    runs = MODULE.prepare_runs(MODULE.DEFAULT_CLEAN, MODULE.DEFAULT_NOISE)
    audit = MODULE.audit_grid(runs)
    assert audit["run_count"] == 12000
    assert audit["condition_counts"] == {
        "clean": 3000,
        "noise_0.01": 3000,
        "noise_0.05": 3000,
        "noise_0.10": 3000,
    }
    assert audit["cells_without_exactly_five_seeds"] == 0
    cell, condition, long, wide = MODULE.compute_condition_tables(runs)
    assert len(cell) == 2400
    assert len(condition) == 48
    assert len(long) == 240
    assert len(wide) == 60
    assert set(wide["metric"]) == set(MODULE.METRIC_ORDER)


def test_seed_mean_robu_agrees_with_formal_ordering() -> None:
    runs = MODULE.prepare_runs(MODULE.DEFAULT_CLEAN, MODULE.DEFAULT_NOISE)
    _, _, _, wide = MODULE.compute_condition_tables(runs)
    robu = wide[["algorithm", "robu_seed_mean", "robu_seed_mean_rank"]].drop_duplicates()
    comparison, summary = MODULE.compare_formal(robu, MODULE.DEFAULT_FORMAL)
    assert len(comparison) == 12
    assert summary["spearman_rank_correlation"] > 0.95
    assert summary["mean_absolute_score_difference"] < 1.5
    assert summary["max_absolute_score_difference"] < 3.1
