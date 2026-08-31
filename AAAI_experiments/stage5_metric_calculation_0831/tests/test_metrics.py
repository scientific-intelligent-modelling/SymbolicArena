from __future__ import annotations

import math

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import (
    MetricContractError,
    RunQuality,
    aggregate_quality,
    efficiency_from_qualities,
    minimality_score,
    numerical_consistency,
    phi_nmse,
    set_f1,
    stability_score,
    symbolic_fidelity_score,
)


def test_phi_nmse_fixed_mapping_and_invalid_policy() -> None:
    assert phi_nmse(1e-12) == pytest.approx(1.0)
    assert phi_nmse(1e2) == pytest.approx(0.0)
    assert phi_nmse(1e-5) == pytest.approx(0.5)
    assert phi_nmse(0.0) == pytest.approx(1.0)
    assert phi_nmse(None) == pytest.approx(0.0)
    assert phi_nmse(float("nan")) == pytest.approx(0.0)
    assert phi_nmse(float("inf")) == pytest.approx(0.0)
    assert phi_nmse(-1.0) == pytest.approx(0.0)


def test_aggregate_quality_maps_each_seed_before_averaging() -> None:
    values = [1e-12, 1e-12, 1e2]
    expected = 100.0 * (1.0 + 1.0 + 0.0) / 3.0
    assert aggregate_quality(values) == pytest.approx(expected)
    assert aggregate_quality([]) == pytest.approx(0.0)


def test_symbolic_fidelity_exact_and_partial_credit() -> None:
    assert symbolic_fidelity_score(equivalent=True, tree_similarity=0, variable_f1=0, operator_f1=0) == 1.0
    assert symbolic_fidelity_score(equivalent=False, tree_similarity=1, variable_f1=1, operator_f1=1) == 0.5
    assert symbolic_fidelity_score(equivalent=False, tree_similarity=0.8, variable_f1=0.5, operator_f1=0.25) == pytest.approx(
        0.5 * (0.8 * 0.5 * 0.25) ** (1.0 / 3.0)
    )
    assert symbolic_fidelity_score(equivalent=False, tree_similarity=1, variable_f1=1, operator_f1=1, valid=False) == 0.0


def test_symbolic_inputs_must_be_bounded() -> None:
    with pytest.raises(MetricContractError):
        symbolic_fidelity_score(equivalent=False, tree_similarity=1.1, variable_f1=1, operator_f1=1)


def test_minimality_score_contract() -> None:
    assert minimality_score(10, 10) == pytest.approx(1.0)
    assert minimality_score(10, 20) == pytest.approx(0.5)
    assert minimality_score(10, 5) == pytest.approx(1.0)
    assert minimality_score(10, 20, valid=False) == pytest.approx(0.0)
    with pytest.raises(MetricContractError):
        minimality_score(0, 10)


def test_efficiency_relative_to_own_best() -> None:
    qualities = [0.25] * 90 + [1.0] * 90
    assert efficiency_from_qualities(qualities, horizon=180) == pytest.approx(0.625)
    assert efficiency_from_qualities([0.0] * 180, horizon=180) == pytest.approx(0.0)


def test_efficiency_rejects_missing_or_invalid_trajectory() -> None:
    with pytest.raises(MetricContractError):
        efficiency_from_qualities([0.5] * 179, horizon=180)
    with pytest.raises(MetricContractError):
        efficiency_from_qualities([0.5] * 179 + [None], horizon=180)
    with pytest.raises(MetricContractError):
        efficiency_from_qualities([0.5] * 179 + [math.nan], horizon=180)


def test_set_f1_empty_and_partial_sets() -> None:
    assert set_f1(set(), set()) == pytest.approx(1.0)
    assert set_f1({"x0"}, set()) == pytest.approx(0.0)
    assert set_f1({"x0", "x1"}, {"x1", "x2"}) == pytest.approx(0.5)


def test_numerical_consistency_uses_all_three_seed_pairs() -> None:
    runs = [
        RunQuality(id_quality=1.0, ood_quality=1.0, valid=True),
        RunQuality(id_quality=0.5, ood_quality=1.0, valid=True),
        RunQuality(id_quality=0.0, ood_quality=0.0, valid=False),
    ]
    disagreements = [0.25, 1.0, 0.75]
    assert numerical_consistency(runs) == pytest.approx(1.0 - sum(disagreements) / 3.0)


def test_stability_geometric_mean_without_performance_correction() -> None:
    runs = [
        RunQuality(1.0, 1.0, True),
        RunQuality(0.5, 1.0, True),
        RunQuality(0.0, 0.0, False),
    ]
    result = stability_score(runs, structural_pair_results=[True, False, False])
    n = numerical_consistency(runs)
    v = 2.0 / 3.0
    c = 1.0 / 3.0
    assert result.numerical_consistency == pytest.approx(n)
    assert result.validity == pytest.approx(v)
    assert result.structural_consistency == pytest.approx(c)
    assert result.score == pytest.approx((n * v * c) ** (1.0 / 3.0))


def test_stability_requires_exactly_three_seeds_and_pairs() -> None:
    runs = [RunQuality(1.0, 1.0, True)] * 2
    with pytest.raises(MetricContractError):
        stability_score(runs, structural_pair_results=[True])

