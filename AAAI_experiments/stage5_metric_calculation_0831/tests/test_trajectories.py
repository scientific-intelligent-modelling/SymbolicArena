from __future__ import annotations

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.metrics import phi_nmse
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.trajectories import (
    TrajectoryContractError,
    canonical_expression,
    reconstruct_trajectory,
)


def heartbeat(minute: int) -> dict[str, object]:
    return {
        "record_type": "periodic_heartbeat",
        "checkpoint_index": minute,
        "elapsed_minutes": minute,
        "status": "running",
        "equation": None,
        "id_test": None,
        "ood_test": None,
    }


def candidate(
    minute: int,
    nmse: float,
    *,
    record_type: str = "periodic_best",
    tool: str = "fixture",
    source_loss: float | None = None,
    source_score: float | None = None,
    source_internal_loss: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": tool,
        "record_type": record_type,
        "checkpoint_index": minute,
        "elapsed_minutes": minute,
        "status": "ok",
        "equation": "raw",
        "canonical_artifact": {
            "normalized_expression": "c0*x0",
            "instantiated_expression": "2.0*x0",
        },
        "id_test": {"nmse": nmse},
        "ood_test": {"nmse": nmse},
    }
    if source_loss is not None:
        payload["source_loss"] = source_loss
    if source_score is not None:
        payload["source_score"] = source_score
    if source_internal_loss is not None:
        payload["source_internal_loss"] = source_internal_loss
    return payload


def test_canonical_expression_prefers_instantiated_constants() -> None:
    payload = candidate(1, 1.0)
    assert canonical_expression(payload) == "2.0*x0"
    payload["canonical_artifact"]["instantiated_expression"] = ""
    assert canonical_expression(payload) == "c0*x0"


def test_heartbeat_before_first_candidate_is_zero_then_best_is_carried() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[3] = candidate(3, 1e-5)
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[0].quality == pytest.approx(0.0)
    assert trajectory[1].quality == pytest.approx(0.0)
    assert trajectory[2].quality == pytest.approx(phi_nmse(1e-5))
    assert trajectory[3].quality == pytest.approx(phi_nmse(1e-5))
    assert trajectory[3].source == "carry_forward:3"


def test_later_candidate_replaces_previous_state_even_if_quality_is_worse() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-8)
    snapshots[2] = candidate(2, 1.0)
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[1].quality < trajectory[0].quality
    assert trajectory[1].source == "snapshot:2"


@pytest.mark.parametrize("tool", ["gplearn", "QLattice"])
def test_loss_based_algorithms_keep_internal_historical_best(tool: str) -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-8, tool=tool, source_loss=1.0)
    snapshots[2] = candidate(2, 1.0, tool=tool, source_loss=2.0)
    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm=tool)
    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].source == "internal_best_carry_forward:1"


def test_symbolfit_keeps_pysr_internal_historical_best() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    first = candidate(1, 1e-8, tool="symbolfit")
    first["source_internal_loss"] = 0.1
    second = candidate(2, 1.0, tool="symbolfit")
    second["source_internal_loss"] = 0.5
    snapshots[1] = first
    snapshots[2] = second

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="symbolfit")

    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].source == "internal_best_carry_forward:1"


def test_dso_keeps_highest_internal_reward_across_chunk_reset() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-8, tool="dso", source_score=0.9)
    snapshots[2] = candidate(2, 1.0, tool="dso", source_score=0.1)
    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="dso")
    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].source == "internal_best_carry_forward:1"


@pytest.mark.parametrize("tool", ["iMCTS", "DRSR"])
def test_score_based_algorithms_keep_highest_internal_objective(tool: str) -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-8, tool=tool, source_score=0.9)
    snapshots[2] = candidate(2, 1.0, tool=tool, source_score=0.1)

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm=tool)

    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].source == "internal_best_carry_forward:1"


def test_jaxsr_keeps_lowest_internal_training_loss() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-8, tool="jaxsr", source_loss=0.1)
    snapshots[2] = candidate(2, 1.0, tool="jaxsr", source_loss=0.5)

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="jaxsr")

    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].source == "internal_best_carry_forward:1"


def test_internal_best_selection_never_uses_id_ood_quality() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1.0, tool="gplearn", source_loss=1.0)
    snapshots[2] = candidate(2, 1e-12, tool="gplearn", source_loss=2.0)
    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="gplearn")
    assert trajectory[1].quality == pytest.approx(phi_nmse(1.0))


def test_ranked_invalid_internal_improvement_stays_best_with_zero_quality() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1.0, tool="gplearn", source_loss=1.0)
    snapshots[2] = candidate(2, 1e-12, tool="gplearn", source_loss=0.5)
    snapshots[2]["ood_test"] = None
    snapshots[2]["status"] = "invalid"
    snapshots[3] = candidate(3, 1e-12, tool="gplearn", source_loss=0.8)

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="gplearn")

    assert trajectory[1].quality == pytest.approx(0.0)
    assert trajectory[1].valid_output is False
    assert trajectory[1].source == "snapshot:2"
    assert trajectory[2].quality == pytest.approx(0.0)
    assert trajectory[2].expression == trajectory[1].expression
    assert trajectory[2].source == "internal_best_carry_forward:2"


def test_ranked_invalid_non_improvement_does_not_clear_internal_best() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1.0, tool="dso", source_score=0.9)
    snapshots[2] = candidate(2, 1e-12, tool="dso", source_score=0.5)
    snapshots[2]["id_test"] = None
    snapshots[2]["status"] = "invalid"

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="dso")

    assert trajectory[1].quality == pytest.approx(trajectory[0].quality)
    assert trajectory[1].valid_output is True
    assert trajectory[1].source == "internal_best_carry_forward:1"


@pytest.mark.parametrize(
    ("tool", "objective_field", "objective_value"),
    [
        ("gplearn", "source_loss", 1.0),
        ("QLattice", "source_loss", 1.0),
        ("dso", "source_score", 0.9),
        ("symbolfit", "source_internal_loss", 0.1),
        ("iMCTS", "source_score", 0.9),
        ("jaxsr", "source_loss", 0.1),
        ("drsr", "source_score", 0.9),
    ],
)
def test_unranked_final_cannot_override_proven_internal_best(
    tool: str,
    objective_field: str,
    objective_value: float,
) -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    kwargs = {objective_field: objective_value}
    snapshots[179] = candidate(179, 1.0, tool=tool, **kwargs)
    snapshots[180] = candidate(180, 1e-12, tool=tool, record_type="final_best")

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm=tool)

    assert trajectory[-1].quality == pytest.approx(trajectory[-2].quality)
    assert trajectory[-1].expression == trajectory[-2].expression
    assert trajectory[-1].source == "internal_best_carry_forward:179"


def test_unranked_invalid_final_cannot_clear_proven_internal_best() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[179] = candidate(179, 1.0, tool="jaxsr", source_loss=0.1)
    snapshots[180] = candidate(180, 1e-12, tool="jaxsr", record_type="final_best")
    snapshots[180]["ood_test"] = None

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="jaxsr")

    assert trajectory[-1].quality == pytest.approx(trajectory[-2].quality)
    assert trajectory[-1].source == "internal_best_carry_forward:179"


def test_budget_end_internal_best_is_minute_180_ranked_endpoint() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[179] = candidate(179, 1.0, tool="iMCTS", source_score=0.1)
    snapshots[180] = candidate(
        180,
        1e-12,
        tool="iMCTS",
        record_type="budget_end_internal_best",
        source_score=0.9,
    )

    trajectory = reconstruct_trajectory(snapshots, horizon=180, algorithm="iMCTS")

    assert trajectory[-1].quality == pytest.approx(1.0)
    assert trajectory[-1].source == "budget_end_internal_best:180"


def test_explicit_expression_with_evaluator_error_is_zero_and_clears_carry() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1e-5)
    invalid = candidate(2, 1e-8)
    invalid["ood_test"] = None
    snapshots[2] = invalid
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[1].quality == pytest.approx(0.0)
    assert trajectory[1].source == "snapshot_evaluator_error:2"
    assert trajectory[2].quality == pytest.approx(0.0)
    assert trajectory[2].source == "explicit_no_valid_output"


def test_future_backfill_is_ignored_and_previous_candidate_is_carried() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[1] = candidate(1, 1.0)
    leaked = candidate(2, 1e-8, record_type="periodic_backfill")
    leaked["backfilled_from_minute"] = 5
    snapshots[2] = leaked
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[1].quality == pytest.approx(phi_nmse(1.0))
    assert trajectory[1].source == "future_backfill_ignored:5;carry_forward:1"


def test_future_backfill_before_first_candidate_is_zero() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    leaked = candidate(1, 1e-8, record_type="periodic_backfill")
    leaked["backfilled_from_minute"] = 3
    snapshots[1] = leaked
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[0].quality == pytest.approx(0.0)
    assert trajectory[0].source == "future_backfill_ignored:3;explicit_no_valid_output"


def test_missing_checkpoint_is_infrastructure_error() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    del snapshots[77]
    with pytest.raises(TrajectoryContractError, match="缺失 minute_0077"):
        reconstruct_trajectory(snapshots, horizon=180)


def test_final_best_can_anchor_minute_180() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[180] = candidate(180, 1e-12, record_type="final_best")
    trajectory = reconstruct_trajectory(snapshots, horizon=180)
    assert trajectory[-1].quality == pytest.approx(1.0)
    assert trajectory[-1].source == "final:180"


def test_checkpoint_identity_must_match_filename_minute() -> None:
    snapshots = {minute: heartbeat(minute) for minute in range(1, 181)}
    snapshots[5]["checkpoint_index"] = 6
    with pytest.raises(TrajectoryContractError, match="checkpoint_index"):
        reconstruct_trajectory(snapshots, horizon=180)
