"""Current gplearn symbolic scores use typed native trees and explicit nonproof labels."""

from __future__ import annotations

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.gplearn_current_metrics import (
    score_symbolic_run,
    structure_is_positive,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.gplearn_native_prefix_evidence import (
    build_prefix_evidence,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    build_symbolic_artifact,
)


def test_undetermined_equivalence_gets_partial_not_full_sym() -> None:
    pred = build_prefix_evidence("add(X0,1)", ["x0"])
    gt = build_symbolic_artifact("x0 + 2", allowed_variables=["x0"])
    scored = score_symbolic_run(pred, gt, "undetermined")
    assert 0 < scored["m_sym"] <= 0.5
    assert scored["equivalence_established"] is False
    assert scored["m_min"] == pytest.approx(min(1, gt["node_count"] / pred["node_count"]))


def test_structure_undetermined_is_not_positive() -> None:
    assert structure_is_positive("undetermined") is False
    assert structure_is_positive("different_structure") is False
    assert structure_is_positive("same_canonical_structure") is True
    assert structure_is_positive("mathematically_equivalent") is True
