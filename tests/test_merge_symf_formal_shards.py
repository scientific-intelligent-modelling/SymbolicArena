from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "merge_symf_formal_shards.py"


def _load_module():
    module_name = "merge_symf_formal_shards_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for algorithm in ("fepysr", "jaxsr"):
        for gid in ("g0001", "g0002"):
            for seed in (520, 521, 522):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "gid": gid,
                        "dataset": "shared_display_name",
                        "seed": seed,
                        "valid_for_symbolic": True,
                        "pred_parse_ok": True,
                        "cas_equiv": False,
                        "numeric_equiv": False,
                        "numeric_equiv_reason": "not_equivalent",
                        "equiv_final": False,
                        "tree_similarity": 0.5,
                        "var_f1": 1.0,
                        "op_f1": 1.0,
                        "sym_f_formal": 0.35,
                    }
                )
    params = pd.DataFrame(
        [
            {"gid": "g0001", "dataset": "shared_display_name"},
            {"gid": "g0002", "dataset": "shared_display_name"},
        ]
    )
    return pd.DataFrame(rows), params


def test_validate_metrics_grid_uses_stable_keys() -> None:
    module = _load_module()
    metrics, params = _grid()

    summary = module.validate_metrics_grid(
        metrics,
        params,
        expected_algorithm_keys=("fepysr", "jaxsr"),
        expected_runs=12,
        expected_datasets=2,
        expected_seeds=(520, 521, 522),
        expected_runs_per_algorithm=6,
    )

    assert summary == {
        "runs": 12,
        "datasets": 2,
        "algorithms": 2,
        "algorithm_keys": ["fepysr", "jaxsr"],
        "seeds": [520, 521, 522],
        "runs_per_algorithm": {"fepysr": 6, "jaxsr": 6},
    }


def test_validate_metrics_grid_rejects_duplicate_or_incomplete_keys() -> None:
    module = _load_module()
    metrics, params = _grid()
    duplicate = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="重复"):
        module.validate_metrics_grid(
            duplicate,
            params,
            expected_algorithm_keys=("fepysr", "jaxsr"),
            expected_runs=13,
            expected_datasets=2,
            expected_seeds=(520, 521, 522),
            expected_runs_per_algorithm=6,
        )

    incomplete = metrics[
        ~(
            (metrics["algorithm"] == "jaxsr")
            & (metrics["gid"] == "g0002")
            & (metrics["seed"] == 522)
        )
    ].copy()
    with pytest.raises(ValueError, match="运行网格"):
        module.validate_metrics_grid(
            incomplete,
            params,
            expected_algorithm_keys=("fepysr", "jaxsr"),
            expected_runs=11,
            expected_datasets=2,
            expected_seeds=(520, 521, 522),
            expected_runs_per_algorithm=None,
        )


def test_cli_merges_validated_shards_and_writes_formal_outputs(
    tmp_path: Path,
) -> None:
    metrics, params = _grid()
    params_csv = tmp_path / "params.csv"
    params.to_csv(params_csv, index=False)
    shard_paths = []
    for algorithm in ("fepysr", "jaxsr"):
        shard_path = tmp_path / f"{algorithm}.csv"
        metrics[metrics["algorithm"] == algorithm].to_csv(
            shard_path,
            index=False,
        )
        shard_paths.append(shard_path)

    outdir = tmp_path / "merged"
    command = [
        sys.executable,
        str(SCRIPT),
        "--params-csv",
        str(params_csv),
    ]
    for shard_path in shard_paths:
        command.extend(["--shard-csv", str(shard_path)])
    command.extend(
        [
            "--expected-algorithm-keys",
            "fepysr,jaxsr",
            "--expected-runs",
            "12",
            "--expected-datasets",
            "2",
            "--expected-seeds",
            "520,521,522",
            "--expected-runs-per-algorithm",
            "6",
            "--outdir",
            str(outdir),
        ]
    )
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (outdir / "shard_merge_summary.json").read_text(encoding="utf-8")
    )
    assert summary["runs"] == 12
    assert summary["algorithm_keys"] == ["fepysr", "jaxsr"]
    formal_summary = json.loads(
        (
            outdir / "symbolic_metrics_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert formal_summary["runs"] == 12
    assert formal_summary["datasets"] == 2
    assert formal_summary["algorithms"] == 2
