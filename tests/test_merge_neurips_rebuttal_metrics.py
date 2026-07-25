from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check" / "merge_neurips_rebuttal_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "merge_neurips_rebuttal_metrics",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_merge_preserves_performance_rank_and_adds_symbolic_metrics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    performance = tmp_path / "performance.csv"
    symbolic = tmp_path / "symbolic.csv"
    output = tmp_path / "merged.csv"
    _write_csv(
        performance,
        [
            {
                "Rank": 1,
                "Algorithm key": "fepysr",
                "Algorithm": "FePySR",
                "Runs expected": 3,
                "Mean OOD log NMSE": -2.5,
            },
            {
                "Rank": 2,
                "Algorithm key": "jaxsr",
                "Algorithm": "JAXSR",
                "Runs expected": 3,
                "Mean OOD log NMSE": -1.0,
            },
        ],
    )
    _write_csv(
        symbolic,
        [
            {
                "algorithm": "jaxsr",
                "datasets": 1,
                "SYM_F_formal": 20.0,
                "exact_equiv_rate": 0.1,
                "cas_equiv_rate": 0.05,
                "numeric_equiv_rate": 0.08,
                "pred_parse_rate": 0.9,
                "mean_tree_similarity": 0.2,
            },
            {
                "algorithm": "fepysr",
                "datasets": 1,
                "SYM_F_formal": 30.0,
                "exact_equiv_rate": 0.25,
                "cas_equiv_rate": 0.2,
                "numeric_equiv_rate": 0.1,
                "pred_parse_rate": 1.0,
                "mean_tree_similarity": 0.4,
            },
        ],
    )

    summary = module.merge_metrics(
        performance_csv=performance,
        symbolic_csv=symbolic,
        output_csv=output,
        expected_algorithms=2,
    )

    rows = _read_csv(output)
    assert summary["algorithms"] == 2
    assert [row["Algorithm key"] for row in rows] == ["fepysr", "jaxsr"]
    assert rows[0]["SYM-F"] == "30.0"
    assert rows[0]["Exact equiv %"] == "25.0"
    assert rows[0]["TreeSim"] == "0.4"
    assert rows[1]["Rank"] == "2"
    assert summary["performance_sha256"] == module._file_sha256(
        performance
    )
    assert summary["symbolic_sha256"] == module._file_sha256(symbolic)
    assert summary["output_sha256"] == module._file_sha256(output)
    assert summary["performance_csv"] == performance.name
    assert summary["symbolic_csv"] == symbolic.name
    assert summary["output_csv"] == output.name


def test_merge_rejects_algorithm_set_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    performance = tmp_path / "performance.csv"
    symbolic = tmp_path / "symbolic.csv"
    _write_csv(
        performance,
        [{"Rank": 1, "Algorithm key": "fepysr", "Algorithm": "FePySR"}],
    )
    _write_csv(
        symbolic,
        [
            {
                "algorithm": "jaxsr",
                "datasets": 1,
                "SYM_F_formal": 20,
                "exact_equiv_rate": 0,
                "cas_equiv_rate": 0,
                "numeric_equiv_rate": 0,
                "pred_parse_rate": 1,
                "mean_tree_similarity": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="算法集合"):
        module.merge_metrics(
            performance_csv=performance,
            symbolic_csv=symbolic,
            output_csv=tmp_path / "merged.csv",
            expected_algorithms=1,
        )
