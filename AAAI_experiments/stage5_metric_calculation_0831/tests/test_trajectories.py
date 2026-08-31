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


def candidate(minute: int, nmse: float, *, record_type: str = "periodic_best") -> dict[str, object]:
    return {
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
