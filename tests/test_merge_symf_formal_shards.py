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


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("seed", 520.5, "整数"),
        ("sym_f_formal", 999.0, "范围"),
        ("sym_f_formal", 0.9, "评分公式"),
        ("tree_similarity", float("inf"), "有限"),
        ("dataset", "wrong_dataset", "映射"),
        ("cas_equiv", True, "equiv_final"),
        ("valid_for_symbolic", False, "sym_f_formal"),
    ],
)
def test_validate_metrics_grid_rejects_semantic_corruption(
    column: str,
    value,
    message: str,
) -> None:
    module = _load_module()
    metrics, params = _grid()
    corrupted = metrics.copy()
    if column == "seed":
        corrupted["seed"] = corrupted["seed"].astype(float)
    if column == "valid_for_symbolic":
        corrupted.loc[0, "pred_parse_ok"] = False
    corrupted.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        module.validate_metrics_grid(
            corrupted,
            params,
            expected_algorithm_keys=("fepysr", "jaxsr"),
            expected_runs=12,
            expected_datasets=2,
            expected_seeds=(520, 521, 522),
            expected_runs_per_algorithm=6,
        )


def test_shard_provenance_binds_current_inputs(tmp_path: Path) -> None:
    module = _load_module()
    params_csv = tmp_path / "params.csv"
    run_level_csv = tmp_path / "run_level.csv"
    generator_script = tmp_path / "generator.py"
    params_csv.write_text("gid,dataset\ng0001,d1\n", encoding="utf-8")
    run_level_csv.write_text(
        "algorithm,gid,seed\nfepysr,g0001,520\n",
        encoding="utf-8",
    )
    generator_script.write_text("SCHEMA = 1\n", encoding="utf-8")
    shard_dir = tmp_path / "fepysr"
    shard_dir.mkdir()
    shard_csv = shard_dir / "symbolic_metrics_formal.csv"
    shard_csv.write_text("placeholder\n", encoding="utf-8")

    expected = module.build_source_fingerprint(
        params_csv=params_csv,
        run_level_csv=run_level_csv,
        generator_script=generator_script,
        expression_source="run_level.expression_canonical",
    )
    provenance = {
        "schema_version": 1,
        "source_fingerprint": expected,
        "algorithm_keys": ["fepysr"],
        "runs": 1,
        "metrics_sha256": module.generator.file_sha256(shard_csv),
    }
    (shard_dir / "symbolic_metrics_formal_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    validated = module.validate_shard_provenance(
        [shard_csv],
        expected_source_fingerprint=expected,
    )
    assert validated[0]["algorithm_keys"] == ["fepysr"]

    params_csv.write_text(
        "gid,dataset\ng0001,changed\n",
        encoding="utf-8",
    )
    changed = module.build_source_fingerprint(
        params_csv=params_csv,
        run_level_csv=run_level_csv,
        generator_script=generator_script,
        expression_source="run_level.expression_canonical",
    )
    with pytest.raises(ValueError, match="provenance"):
        module.validate_shard_provenance(
            [shard_csv],
            expected_source_fingerprint=changed,
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
    assert summary["params_csv"] == "params.csv"
    assert summary["shard_csvs"] == [
        "fepysr/fepysr.csv",
        "jaxsr/jaxsr.csv",
    ]
    assert not Path(summary["params_csv"]).is_absolute()
    formal_summary = json.loads(
        (
            outdir / "symbolic_metrics_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert formal_summary["runs"] == 12
    assert formal_summary["datasets"] == 2
    assert formal_summary["algorithms"] == 2

    verified = subprocess.run(
        [
            *command[:-2],
            "--validate-only",
            "--verify-output-dir",
            str(outdir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr

    (
        outdir / "symbolic_metrics_formal_algorithm_summary.csv"
    ).write_text("tampered\n", encoding="utf-8")
    rejected = subprocess.run(
        [
            *command[:-2],
            "--validate-only",
            "--verify-output-dir",
            str(outdir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "输出哈希" in rejected.stderr
