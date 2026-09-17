"""Regression cases for invalid scalar-expression replay outputs."""

from __future__ import annotations

import pytest

from check.apply_strict_scalar_terminal_policy import correct_trajectory


def _run(algorithm: str, expressions: list[str], qualities: list[float]) -> dict:
    return {
        "algorithm": algorithm, "condition": "noise005", "dataset_id": "BPG3",
        "seed": 520, "logical_key": f"{algorithm}::BPG3::s520::noise005",
        "expression": expressions, "id_quality": qualities[:], "ood_quality": qualities[:],
        "quality": qualities[:], "valid_output": [True] * len(expressions),
        "q_star": max(qualities),
        "m_eff": sum(qualities) / len(qualities) / max(qualities) if max(qualities) else 0.0,
        "source_sha256": ["a" * 64] * len(expressions),
    }


def test_indexed_feature_is_zeroed_without_replacing_native_expression() -> None:
    original = _run("llmsr", ["x0+1", "x0[0]+1", "x0+2"], [0.2, 0.8, 0.5])
    corrected, changes = correct_trajectory(original)
    assert original["id_quality"] == [0.2, 0.8, 0.5]
    assert corrected["expression"] == original["expression"]
    assert corrected["id_quality"] == [0.2, 0.0, 0.5]
    assert corrected["ood_quality"] == [0.2, 0.0, 0.5]
    assert corrected["valid_output"] == [True, False, True]
    assert corrected["q_star"] == 0.5
    assert corrected["m_eff"] == pytest.approx((0.4 + 0 + 1) / 3)
    assert [change["minute"] for change in changes] == [2]


def test_nan_is_invalid_but_gplearn_protected_display_is_not_overridden() -> None:
    symbolfit, changes = correct_trajectory(_run("SymbolFit", ["nan"], [0.0]))
    assert symbolfit["valid_output"] == [False]
    assert len(changes) == 1
    gplearn, changes = correct_trajectory(_run("gplearn", ["nan"], [0.5]))
    assert gplearn["valid_output"] == [True]
    assert gplearn["id_quality"] == [0.5]
    assert changes == []
