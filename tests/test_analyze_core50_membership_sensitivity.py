from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "check/analyze_core50_membership_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_core50_membership_sensitivity", MODULE_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_problem() -> MODULE.SelectionProblem:
    rows = []
    difficulties = ("easy", "medium", "hard", "extreme")
    for index in range(64):
        family = "a" if index < 32 else "b"
        rows.append(
            {
                "dataset_id": f"g{index:03d}",
                "dataset_name": f"d{index}",
                "dataset_rel": f"data/d{index}",
                "family": family,
                "subgroup": f"{family}{index % 8}",
                "basename": f"d{index}",
                "semantic_duplicate_group": f"sem{index}",
                "operator_group": f"op{index % 4}",
                "feature_count_bin": f"f{index % 3}",
                "sample_count_bin": f"s{index % 3}",
                "complexity_bin": f"c{index % 3}",
                "dummy_variable_status": "not_applicable",
                "difficulty_bin": difficulties[index % 4],
                "failure_mode": f"mode{index % 4}",
                "winner_probe": MODULE.PROBE4[index % 4],
                "valid_output_pattern": f"p{index % 3}",
                "info_score": index / 63,
            }
        )
    return MODULE.SelectionProblem(pd.DataFrame(rows))


def test_weight_normalization() -> None:
    spec = MODULE.normalize_weights(0.9, 0.35, 0.1)
    assert np.isclose(
        spec.coverage_weight
        + spec.information_weight
        + spec.balance_weight,
        1.0,
    )
    assert spec.coverage_weight > spec.information_weight


def test_prepare_dataset_table_distinguishes_dummy_and_non_dummy() -> None:
    dataset_level = pd.DataFrame(
        [
            {
                "dataset_id": "g001",
                "dataset_rel": "a",
                "family": "srsd",
                "subgroup": "s1",
                "basename": "a",
                "semantic_duplicate_group": "a",
                "difficulty_bin": "easy",
                "info_score": 0.1,
                "metadata_class": "easy non-dummy",
                "operator_group": "polynomial",
                "feature_count_bin": "low",
                "sample_count_bin": "small",
                "complexity_bin": "simple",
                "failure_mode": "all_good",
                "winner_probe": "dso",
            },
            {
                "dataset_id": "g002",
                "dataset_rel": "b",
                "family": "srsd",
                "subgroup": "s2",
                "basename": "b",
                "semantic_duplicate_group": "b",
                "difficulty_bin": "hard",
                "info_score": 0.2,
                "metadata_class": "hard dummy",
                "operator_group": "polynomial",
                "feature_count_bin": "low",
                "sample_count_bin": "small",
                "complexity_bin": "simple",
                "failure_mode": "all_good",
                "winner_probe": "dso",
            },
        ]
    )
    dataset_algorithm = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "method_norm": method,
                "n_valid_total_seeds": 3,
            }
            for dataset_id in ("g001", "g002")
            for method in MODULE.PROBE4
        ]
    )
    prepared = MODULE.prepare_dataset_table(
        dataset_level, dataset_algorithm
    ).set_index("dataset_id")
    assert prepared.loc["g001", "dummy_variable_status"] == "non_dummy"
    assert prepared.loc["g002", "dummy_variable_status"] == "dummy"


def test_nominal_family_bounds_match_formula() -> None:
    problem = make_problem()
    lower, upper, target = problem.family_bounds(MODULE.ConstraintSpec())
    assert target.tolist() == [25.0, 25.0]
    assert lower.tolist() == [24, 24]
    assert upper.tolist() == [27, 27]


def test_objective_swap_delta_matches_full_recomputation() -> None:
    problem = make_problem()
    spec = MODULE.ConstraintSpec()
    objective = MODULE.ObjectiveSpec()
    selected = np.asarray(list(range(25)) + list(range(32, 57)))
    assert problem.is_feasible(selected, spec)
    objective_counts, constraints = problem._selected_counts(selected)
    old = 0
    candidates = np.asarray([57, 58, 59])
    mask = problem._candidate_mask(old, candidates, constraints, spec)
    candidates = candidates[mask]
    deltas = problem._objective_swap_delta(
        old, candidates, objective_counts, objective
    )
    before = problem.objective_terms(selected, objective)["objective"]
    for candidate, delta in zip(candidates, deltas, strict=True):
        changed = set(map(int, selected))
        changed.remove(old)
        changed.add(int(candidate))
        after = problem.objective_terms(changed, objective)["objective"]
        assert np.isclose(after - before, delta, atol=1e-12)


def test_optimizer_is_deterministic_and_feasible() -> None:
    problem = make_problem()
    constraints = MODULE.ConstraintSpec()
    objective = MODULE.ObjectiveSpec()
    start = np.asarray(list(range(25)) + list(range(32, 57)))
    first, _ = problem.optimize(start, objective, constraints)
    second, _ = problem.optimize(start, objective, constraints)
    assert first.tolist() == second.tolist()
    assert problem.is_feasible(first, constraints)
    assert (
        problem.objective_terms(first, objective)["objective"]
        >= problem.objective_terms(start, objective)["objective"]
    )


def test_construct_feasible_respects_constraints() -> None:
    problem = make_problem()
    constraints = MODULE.ConstraintSpec()
    rng = np.random.default_rng(7)
    selected = problem.construct_feasible(constraints, rng, attempts=20)
    second = problem.construct_feasible(constraints, rng, attempts=20)
    assert len(selected) == 50
    assert problem.is_feasible(selected, constraints)
    assert selected.tolist() != second.tolist()


def test_local_grid_contains_nominal_and_is_unique() -> None:
    grid = MODULE.unique_local_weight_grid()
    keys = {
        (
            round(spec.coverage_weight, 12),
            round(spec.information_weight, 12),
            round(spec.balance_weight, 12),
        )
        for _, spec in grid
    }
    assert len(grid) == len(keys)
    assert (0.45, 0.35, 0.2) in keys


def test_simplex_grid_has_66_points() -> None:
    grid = MODULE.broad_simplex_grid(0.1)
    assert len(grid) == 66
    assert all(
        np.isclose(
            spec.coverage_weight
            + spec.information_weight
            + spec.balance_weight,
            1.0,
        )
        for _, spec in grid
    )
