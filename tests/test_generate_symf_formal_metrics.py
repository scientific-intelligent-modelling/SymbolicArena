from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "generate_symf_formal_metrics.py"


def _load_module():
    module_name = "generate_symf_formal_metrics_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_formula(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    (dataset_dir / "formula.py").write_text(
        "def y(x):\n    return x + 1\n",
        encoding="utf-8",
    )


def _params(dataset_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "core50_index": 7,
                "gid": "g0007",
                "dataset": "catalog_display_name",
                "dataset_dir": str(dataset_dir),
                "target_name": "y",
                "feature_count": 1,
                "metadata_feature_names": json.dumps(["x"]),
                "formula_target_function": "y",
                "feature_to_x_map": json.dumps({"x": "x0"}),
                "gt_expression_x": "x0 + 1",
                "probe_ranges": json.dumps([[-4, 4]]),
                "probe_samples": 1024,
                "probe_random_seed": 123,
                "cas_equivalence_enabled": True,
            }
        ]
    )


def test_formal_metrics_join_ground_truth_by_stable_gid(tmp_path: Path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_formula(dataset_dir)
    runs = pd.DataFrame(
        [
            {
                "algorithm": "fepysr",
                "gid": "g0007",
                "dataset": "different_display_name",
                "seed": 520,
                "status": "ok",
                "valid_output": True,
                "metric_complete": True,
                "result_path": "",
                "expression_canonical": "x0 + 1",
            }
        ]
    )

    metrics = module.run_formal_metrics(_params(dataset_dir), runs)

    assert len(metrics) == 1
    assert metrics.loc[0, "failure_reason"] == "missing_result_json"
    assert bool(metrics.loc[0, "equiv_final"]) is True
    assert metrics.loc[0, "sym_f_formal"] == pytest.approx(1.0)


def test_formal_metrics_can_freeze_run_level_expression(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_formula(dataset_dir)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "canonical_artifact": {
                    "normalized_expression": "x0 + 999"
                }
            }
        ),
        encoding="utf-8",
    )
    runs = pd.DataFrame(
        [
            {
                "algorithm": "fepysr",
                "gid": "g0007",
                "dataset": "catalog_display_name",
                "seed": 520,
                "status": "ok",
                "valid_output": True,
                "metric_complete": True,
                "result_path": str(result_path),
                "expression_canonical": "x0 + 1",
            }
        ]
    )

    metrics = module.run_formal_metrics(
        _params(dataset_dir),
        runs,
        prefer_run_level_expression=True,
    )

    assert metrics.loc[0, "pred_expression_raw"] == "x0 + 1"
    assert metrics.loc[0, "expr_source"] == (
        "run_level.expression_canonical"
    )
    assert bool(metrics.loc[0, "equiv_final"]) is True


def test_ground_truth_probe_uses_formula_source_frozen_in_params(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_formula(dataset_dir)
    frozen_source = (dataset_dir / "formula.py").read_bytes()
    params = _params(dataset_dir)
    params["formula_source_sha256"] = hashlib.sha256(
        frozen_source
    ).hexdigest()
    params["formula_source_b64"] = base64.b64encode(
        frozen_source
    ).decode("ascii")
    (dataset_dir / "formula.py").write_text(
        "def y(x):\n    return x + 999\n",
        encoding="utf-8",
    )
    runs = pd.DataFrame(
        [
            {
                "algorithm": "fepysr",
                "gid": "g0007",
                "dataset": "catalog_display_name",
                "seed": 520,
                "status": "ok",
                "valid_output": True,
                "metric_complete": True,
                "result_path": "",
                "expression_canonical": "x0 + 1",
            }
        ]
    )

    metrics = module.run_formal_metrics(
        params,
        runs,
        prefer_run_level_expression=True,
    )

    assert bool(metrics.loc[0, "numeric_equiv"]) is True
    assert bool(metrics.loc[0, "equiv_final"]) is True


def test_formal_metrics_can_require_frozen_formula_source(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_formula(dataset_dir)
    runs = pd.DataFrame(
        [
            {
                "algorithm": "fepysr",
                "gid": "g0007",
                "dataset": "catalog_display_name",
                "seed": 520,
                "status": "ok",
                "valid_output": True,
                "metric_complete": True,
                "result_path": "",
                "expression_canonical": "x0 + 1",
            }
        ]
    )

    with pytest.raises(ValueError, match="formula_source_b64"):
        module.run_formal_metrics(
            _params(dataset_dir),
            runs,
            prefer_run_level_expression=True,
            require_frozen_formula_source=True,
        )


def test_source_fingerprint_detects_input_change(tmp_path: Path) -> None:
    module = _load_module()
    params_csv = tmp_path / "params.csv"
    run_level_csv = tmp_path / "runs.csv"
    params_csv.write_text("gid\ng0001\n", encoding="utf-8")
    run_level_csv.write_text(
        "algorithm,gid,seed\nfepysr,g0001,520\n",
        encoding="utf-8",
    )
    initial = module.build_source_fingerprint(
        params_csv=params_csv,
        run_level_csv=run_level_csv,
        generator_script=SCRIPT,
        expression_source="run_level.expression_canonical",
    )

    run_level_csv.write_text(
        "algorithm,gid,seed\nfepysr,g0001,521\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="运行期间发生变化"):
        module.assert_source_fingerprint_unchanged(
            initial,
            params_csv=params_csv,
            run_level_csv=run_level_csv,
            generator_script=SCRIPT,
            expression_source="run_level.expression_canonical",
        )


def test_load_run_level_parses_booleans_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "run_level.csv"
    rows = pd.DataFrame(
        [
            {
                "algorithm": "FePySR",
                "gid": "g0007",
                "dataset": "d7",
                "seed": 520,
                "status": "ok",
                "valid_output": "False",
                "metric_complete": "1",
                "result_path": "",
                "expression_canonical": "x0",
            }
        ]
    )
    rows.to_csv(path, index=False)

    loaded = module.load_expected_runs_from_run_level(
        path,
        algorithms={"fepysr"},
        expected_runs=1,
    )
    assert loaded.loc[0, "algorithm"] == "fepysr"
    assert bool(loaded.loc[0, "valid_output"]) is False
    assert bool(loaded.loc[0, "metric_complete"]) is True

    pd.concat([rows, rows], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="重复"):
        module.load_expected_runs_from_run_level(path)


def test_output_summary_counts_datasets_by_stable_gid(tmp_path: Path) -> None:
    module = _load_module()
    metrics = pd.DataFrame(
        [
            {
                "algorithm": "fepysr",
                "gid": gid,
                "dataset": "shared_display_name",
                "seed": 520,
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
            for gid in ("g0001", "g0002")
        ]
    )

    module.write_outputs(
        tmp_path,
        metrics,
        pd.DataFrame([{"gid": "g0001"}, {"gid": "g0002"}]),
    )

    summary = json.loads(
        (tmp_path / "symbolic_metrics_formal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["datasets"] == 2
    assert summary["algorithm_summary"][0]["datasets"] == 2


def test_resolve_dataset_path_prefers_home_dataset_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    home_root = tmp_path / "home"
    relative = Path("sim-datasets-data/synthetic/unit/shared_name")
    repo_candidate = repo_root / relative
    home_candidate = home_root / relative
    repo_candidate.mkdir(parents=True)
    home_candidate.mkdir(parents=True)

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setenv("HOME", str(home_root))

    assert module.resolve_dataset_path(relative) == home_candidate
    assert module.resolve_dataset_path("assets/example") == (
        repo_root / "assets/example"
    )


def test_ground_truth_probe_calls_formula_in_declared_argument_order(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "formula.py").write_text(
        "def y(G, m1, m2):\n    return G + 2 * m1 + 3 * m2\n",
        encoding="utf-8",
    )
    params = pd.DataFrame(
        [
            {
                "core50_index": 7,
                "gid": "g0007",
                "dataset": "d7",
                "dataset_dir": str(dataset_dir),
                "target_name": "y",
                "feature_count": 3,
                "metadata_feature_names": json.dumps(["m1", "m2", "G"]),
                "formula_target_function": "y",
                "formula_arg_feature_indices": json.dumps([2, 0, 1]),
                "probe_ranges": json.dumps([[-4, 4], [-4, 4], [-4, 4]]),
                "probe_samples": 1024,
                "probe_random_seed": 123,
            }
        ]
    )

    probe = module.load_gt_probe_cache(params)["g0007"]
    samples = probe["samples"]
    expected = samples[:, 2] + 2 * samples[:, 0] + 3 * samples[:, 1]
    assert np.allclose(probe["y_gt"], expected)
