from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "prepare_symf_formal_judge_params.py"


def _load_module():
    module_name = "prepare_symf_formal_judge_params_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.yaml").write_text(
        """
dataset:
  features:
    - name: x
      train_range: [-2, 2]
      ood_range: [-4, 4]
  target:
    name: y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "formula.py").write_text(
        "def y(x):\n    return x + 1\n",
        encoding="utf-8",
    )
    for split in ("train.csv", "valid.csv", "id_test.csv", "ood_test.csv"):
        (dataset_dir / split).write_text("x,y\n0,1\n1,2\n", encoding="utf-8")


def _write_catalog(path: Path, dataset_dir: Path) -> None:
    rows = [
        {
            "dataset_id": "g0007",
            "global_index": 7,
            "dataset_name": "shared_name",
            "family": "synthetic",
            "subgroup": "unit",
            "dataset_dir": str(dataset_dir),
            "dataset_rel": "sim-datasets-data/synthetic/unit/shared_name",
            "formula_py": str(dataset_dir / "formula.py"),
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_parameters_accepts_full664_manifest_schema(tmp_path: Path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    catalog = tmp_path / "datasets.csv"
    _write_catalog(catalog, dataset_dir)
    outdir = tmp_path / "params"

    summary = module.prepare_parameters(
        catalog_csv=catalog,
        outdir=outdir,
        expected_datasets=1,
        probe_samples=128,
        probe_random_seed=123,
    )

    params = pd.read_csv(outdir / "symf_formal_judge_parameters.csv")
    assert summary["datasets"] == 1
    assert summary["auto_confirmed"] == 1
    assert summary["catalog_csv"] == str(catalog.resolve())
    assert params.loc[0, "gid"] == "g0007"
    assert params.loc[0, "core50_index"] == 7
    assert params.loc[0, "dataset"] == "shared_name"
    assert params.loc[0, "target_name"] == "y"
    assert params.loc[0, "feature_count"] == 1
    assert params.loc[0, "gt_expression_sympy"] == "x0 + 1"
    assert params.loc[0, "probe_samples"] == 128
    assert params.loc[0, "probe_random_seed"] == 123
    formula_source = (dataset_dir / "formula.py").read_bytes()
    assert params.loc[0, "formula_source_sha256"] == hashlib.sha256(
        formula_source
    ).hexdigest()
    assert base64.b64decode(
        params.loc[0, "formula_source_b64"]
    ) == formula_source


def test_resolve_repo_path_prefers_home_dataset_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    repo_root = tmp_path / "repo"
    home_root = tmp_path / "home"
    relative = Path(
        "sim-datasets-data/synthetic/unit/shared_name/metadata.yaml"
    )
    repo_candidate = repo_root / relative
    home_candidate = home_root / relative
    repo_candidate.parent.mkdir(parents=True)
    home_candidate.parent.mkdir(parents=True)
    repo_candidate.write_text("repo-partial\n", encoding="utf-8")
    home_candidate.write_text("home-complete\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.setenv("HOME", str(home_root))

    assert module._resolve_repo_path(relative) == home_candidate
    assert module._resolve_repo_path("assets/example.csv") == (
        repo_root / "assets/example.csv"
    )


def test_prepare_parameters_rejects_duplicate_stable_gid(tmp_path: Path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    catalog = tmp_path / "datasets.csv"
    _write_catalog(catalog, dataset_dir)
    frame = pd.read_csv(catalog)
    pd.concat([frame, frame], ignore_index=True).to_csv(catalog, index=False)

    try:
        module.prepare_parameters(
            catalog_csv=catalog,
            outdir=tmp_path / "params",
            expected_datasets=2,
        )
    except ValueError as exc:
        assert "gid" in str(exc)
    else:
        raise AssertionError("duplicate gid should be rejected")


def test_formula_extraction_does_not_call_unbounded_simplify(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)

    def _forbidden_simplify(*args, **kwargs):
        raise AssertionError("formal 参数准备阶段不得执行无超时 simplify")

    monkeypatch.setattr(module.sp, "simplify", _forbidden_simplify)
    formula = module._extract_formula(dataset_dir / "formula.py", "y")
    assert formula.ok is True
    assert formula.sympy_sstr == "x0 + 1"


def test_unresolved_issue_counts_are_json_serializable(tmp_path: Path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    (dataset_dir / "formula.py").unlink()
    catalog = tmp_path / "datasets.csv"
    _write_catalog(catalog, dataset_dir)
    outdir = tmp_path / "params"

    summary = module.prepare_parameters(
        catalog_csv=catalog,
        outdir=outdir,
        expected_datasets=1,
    )

    assert summary["needs_review"] == 1
    persisted = json.loads(
        (outdir / "symf_formal_judge_config.json").read_text(encoding="utf-8")
    )
    assert persisted["unresolved_issue_counts"]["missing_formula_py"] == 1


def test_formula_extraction_inlines_local_derived_expressions(
    tmp_path: Path,
) -> None:
    module = _load_module()
    formula_path = tmp_path / "formula.py"
    formula_path.write_text(
        """
def y(x):
    numerator = x + 1
    denominator = x + 2
    ratio = numerator / denominator
    return ratio
""".strip()
        + "\n",
        encoding="utf-8",
    )

    formula = module._extract_formula(formula_path, "y")

    assert formula.ok is True
    assert formula.free_symbols == ["x0"]
    assert "numerator" not in str(formula.expression_x)
    assert "denominator" not in str(formula.expression_x)


def test_formula_arguments_follow_dataset_feature_order(tmp_path: Path) -> None:
    module = _load_module()
    formula_path = tmp_path / "formula.py"
    formula_path.write_text(
        "def y(G, m1, m2):\n    return G * m1 * m2\n",
        encoding="utf-8",
    )

    formula = module._extract_formula(
        formula_path,
        "y",
        variable_order=["m1", "m2", "G"],
    )
    mapping = module._formula_arg_feature_indices(
        formula.arg_names,
        ["m1", "m2", "G"],
    )

    assert formula.ok is True
    assert formula.sympy_sstr == "x0*x1*x2"
    assert mapping == ([2, 0, 1], [])
