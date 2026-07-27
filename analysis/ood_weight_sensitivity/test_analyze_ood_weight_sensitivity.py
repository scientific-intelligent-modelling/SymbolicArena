from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_ood_weight_sensitivity as analysis


def test_phi_boundaries_match_formal_mapping() -> None:
    assert analysis.phi_from_nmse(1e-12) == 1.0
    assert analysis.phi_from_nmse(1e2) == 0.0
    assert analysis.phi_from_nmse(float("nan")) == 0.0


def test_failure_semantics_precede_seed_median() -> None:
    rows = []
    for seed, value, complete in [
        (0, 1e-6, True),
        (1, 1e-5, True),
        (2, 1e-4, True),
        (3, 1e-8, False),
        (4, 1e-9, False),
    ]:
        rows.append(
            {
                "algorithm": "a",
                "gid": "g0001",
                "dataset": "d",
                "seed": seed,
                "status": "ok",
                "valid_output": True,
                "metric_complete": complete,
                "id_test_nmse": value,
                "ood_test_nmse": value * 10,
                "analysis_penalty_applied": not complete,
            }
        )
    _, components = analysis.compute_task_components(
        pd.DataFrame(rows), expected_component_count=None
    )
    assert components.iloc[0]["median_id_nmse"] == 1e-4
    assert components.iloc[0]["median_ood_nmse"] == 1e-3


def test_average_rank_ties_and_stable_topk() -> None:
    ranks = analysis.average_descending_ranks([3.0, 2.0, 2.0, 1.0])
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])
    assert analysis.stable_topk(["z", "b", "a"], [2.0, 1.0, 1.0], 3) == [
        "z",
        "a",
        "b",
    ]
    assert analysis.spearman_rank_correlation(ranks, ranks) == 1.0
    assert analysis.kendall_tau_b(ranks, ranks) == 1.0


def test_pairwise_crossing_formula() -> None:
    components = pd.DataFrame(
        [
            {"algorithm": "a", "Q": 1.0, "R": 0.0},
            {"algorithm": "b", "Q": 0.0, "R": 1.0},
        ]
    )
    crossings = analysis.pairwise_crossings(components)
    crossing = crossings.loc[~crossings["is_parallel"]].iloc[0]
    assert math.isclose(crossing["crossing_weight"], 0.5)
    assert crossing["rank_order_below_crossing"] == "b>a"
    assert crossing["rank_order_above_crossing"] == "a>b"


def test_bootstrap_pairwise_stability_uses_directional_probability() -> None:
    probabilities = pd.DataFrame(
        [
            {
                "algorithm_a": "a",
                "algorithm_b": "b",
                "weight": 0.7,
                "p_a_greater_b": 0.96,
                "p_tie": 0.0,
                "p_a_less_b": 0.04,
                "is_baseline_adjacent_pair": True,
                "is_baseline_top5_pair": True,
            },
            {
                "algorithm_a": "a",
                "algorithm_b": "c",
                "weight": 0.7,
                "p_a_greater_b": 0.20,
                "p_tie": 0.0,
                "p_a_less_b": 0.80,
                "is_baseline_adjacent_pair": False,
                "is_baseline_top5_pair": True,
            },
        ]
    )
    summary = analysis.bootstrap_pairwise_stability_summary(probabilities)
    assert summary.iloc[0]["all_pair_stable_count"] == 1
    assert summary.iloc[0]["baseline_adjacent_pair_stable_count"] == 1
    assert summary.iloc[0]["baseline_top5_pair_stable_count"] == 1
    assert math.isclose(
        summary.iloc[0]["minimum_adjacent_directional_probability"],
        0.96,
    )


def test_reconstructed_grid_matches_expected_formal_input() -> None:
    merged, audit = analysis.load_and_merge_inputs(
        analysis.DEFAULT_ORIGINAL_RUNS,
        analysis.DEFAULT_REPLACEMENT_RUNS,
        analysis.DEFAULT_EXPECTED_MERGED_RUNS,
        None,
    )
    assert len(merged) == 3000
    assert not merged.duplicated(analysis.KEY_COLUMNS).any()
    assert audit["replacement_key_match"]
    assert audit["metrics_match_expected_with_float_tolerance"]
    assert audit["max_relative_metric_difference_vs_expected"] <= 1e-14
    assert set(
        merged.loc[
            merged["source_version"].eq(
                "drsr_llmsr_modelsplit_rerun_20260503"
            ),
            "algorithm",
        ]
    ) == {"drsr", "llmsr"}


def test_w070_reproduces_submitted_figure3() -> None:
    merged, _ = analysis.load_and_merge_inputs(
        analysis.DEFAULT_ORIGINAL_RUNS,
        analysis.DEFAULT_REPLACEMENT_RUNS,
        analysis.DEFAULT_EXPECTED_MERGED_RUNS,
        None,
    )
    merged["analysis_penalty_applied"] = (
        ~merged["valid_output"] | ~merged["metric_complete"]
    )
    _, components = analysis.compute_task_components(merged)
    _, errors = analysis.verify_formal_reproduction(
        components,
        analysis.DEFAULT_FORMAL_COMPONENTS,
        analysis.DEFAULT_FORMAL_SCORES,
    )
    assert errors["max_abs_error_algorithm_OOD_G_w070"] < 1e-6
