from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "analyze_symf_weight_sensitivity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_symf_weight_sensitivity",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(path: Path) -> None:
    rows = []
    configurations = {
        "alpha": [
            (True, True, 1.0, 1.0, 1.0),
            (False, True, 0.4, 1.0, 0.5),
            (False, False, 0.9, 1.0, 1.0),
            (False, True, 0.2, 0.8, 0.4),
        ],
        "beta": [
            (False, True, 0.6, 0.6, 0.8),
            (False, True, 0.5, 0.4, 0.6),
            (False, True, 0.3, 0.4, 0.4),
            (False, True, 0.2, 0.2, 0.4),
        ],
    }
    for algorithm, values in configurations.items():
        for index, (exact, eligible, tree, var, op) in enumerate(
            values
        ):
            sof1 = 0.5 * (var + op)
            score = (
                1.0
                if exact
                else 0.3 * tree + 0.2 * sof1
                if eligible
                else 0.0
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "gid": f"g{index // 2 + 1:04d}",
                    "dataset": f"d{index // 2 + 1}",
                    "seed": index % 2,
                    "valid_for_symbolic": eligible,
                    "pred_parse_ok": eligible,
                    "gt_parse_ok": True,
                    "equiv_final": exact,
                    "tree_similarity": tree,
                    "var_f1": var,
                    "op_f1": op,
                    "sof1": sof1,
                    "sym_f_formal": score,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_runs_recomputes_frozen_formula_and_gates_invalid(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "symf.csv"
    _write_fixture(path)

    rows = module.load_runs(path, strict_core50=False)

    assert len(rows) == 8
    invalid = next(
        row
        for row in rows
        if row["algorithm"] == "alpha"
        and row["component_eligible"] is False
    )
    assert invalid["tree_similarity"] == 0.9
    scores = module.score_runs(
        rows,
        tree_weight=0.3,
        var_weight=0.1,
        op_weight=0.1,
    )
    assert scores["alpha"] == pytest.approx(
        100 * (1.0 + 0.3 * 0.4 + 0.1 * 1.0 + 0.1 * 0.5 + 0.0
               + 0.3 * 0.2 + 0.1 * 0.8 + 0.1 * 0.4) / 4
    )


def test_load_runs_rejects_stale_symf_values(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "symf.csv"
    _write_fixture(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["sym_f_formal"] = "0.25"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="冻结公式不一致"):
        module.load_runs(path, strict_core50=False)


def test_load_runs_rejects_exact_but_invalid_row(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "symf.csv"
    _write_fixture(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["valid_for_symbolic"] = "False"
    rows[0]["pred_parse_ok"] = "False"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="不满足符号分数组件"):
        module.load_runs(path, strict_core50=False)


def test_weight_grids_have_expected_sizes_and_preserve_caps(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "symf.csv"
    _write_fixture(path)
    rows = module.load_runs(path, strict_core50=False)
    baseline = module.score_runs(
        rows,
        tree_weight=0.3,
        var_weight=0.1,
        op_weight=0.1,
    )

    local, broad, cap, share = module.build_weight_grids(
        rows,
        baseline,
    )

    assert len(local) == 25
    assert len(broad) == 441
    assert len(cap) == 101
    assert len(share) == 101
    assert min(row["nonexact_cap"] for row in broad) == pytest.approx(0.25)
    assert max(row["nonexact_cap"] for row in broad) == pytest.approx(0.75)
    assert min(row["tree_share"] for row in broad) == pytest.approx(0.0)
    assert max(row["tree_share"] for row in broad) == pytest.approx(1.0)
    assert len(
        {
            (row["nonexact_cap"], row["tree_share"])
            for row in broad
        }
    ) == 441


def test_end_to_end_writes_auditable_outputs(tmp_path: Path) -> None:
    module = _load_module()
    input_path = tmp_path / "input" / "symf.csv"
    output_dir = tmp_path / "output"
    _write_fixture(input_path)

    original_root = module.REPO_ROOT
    module.REPO_ROOT = tmp_path
    try:
        summary = module.analyze(
            input_path=input_path,
            output_dir=output_dir,
            strict_core50=False,
            make_plots=False,
        )
    finally:
        module.REPO_ROOT = original_root

    assert summary["schema_version"] == 1
    assert summary["input"]["runs"] == 8
    assert summary["local_weight_grid"]["configurations"] == 25
    assert summary["broad_stress_grid"]["configurations"] == 441
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "sensitivity_summary.json").is_file()
    assert (output_dir / "component_correlations.csv").is_file()
    saved = json.loads(
        (output_dir / "sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["input"]["path"] == "input/symf.csv"
