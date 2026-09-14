from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_symbolfit_release_metrics import (
    MINUTE_FIELDS,
    SymbolFitReleaseMetricError,
    _write_gzip_csv,
    build_run_trajectory,
)


def _candidate(minute: int, *, equation: str, loss: float) -> dict[str, object]:
    return {
        "tool": "symbolfit",
        "dataset": "demo",
        "seed": 520,
        "condition": "noise001",
        "checkpoint_index": minute,
        "record_type": "budget_end_internal_best" if minute == 180 else "periodic_best",
        "status": "ok",
        "candidate_available": True,
        "equation": equation,
        "canonical_artifact": {
            "artifact_valid": True,
            "tool_name": "symbolfit",
            "raw_equation": equation,
            "normalized_expression": equation,
        },
        "candidate_source": "symbolfit_active_pysr_hall_of_fame",
        "source_internal_loss": loss,
        "source_complexity": 3,
        "candidate_attempt": 1,
        "candidate_first_discovered_minute": 2,
        "id_test": {"nmse": 1.0 if equation == "x0" else 0.01},
        "ood_test": {"nmse": 1.0 if equation == "x0" else 0.01},
    }


def _row() -> dict[str, str]:
    return {
        "logical_key": "SymbolFit::demo::s520::noise001",
        "dataset_id": "demo",
        "dataset_global_index": "1",
        "seed": "520",
        "condition": "noise001",
        "task_id": "symbolfit_s520_noise001_g0001",
        "host": "iaaccn22",
    }


def test_build_run_trajectory_preserves_internal_choice_and_carry_forward() -> None:
    payloads: list[dict[str, object]] = [
        {
            "tool": "symbolfit",
            "dataset": "demo",
            "seed": 520,
            "condition": "noise001",
            "checkpoint_index": 1,
            "record_type": "periodic_heartbeat",
            "status": "running",
            "candidate_available": False,
        }
    ]
    payloads.extend(
        _candidate(minute, equation="x0" if minute < 4 else "x0 + 1", loss=0.2 if minute < 4 else 0.1)
        for minute in range(2, 181)
    )
    minutes, summary = build_run_trajectory(
        _row(),
        payloads,
        source_trajectory_path="source.jsonl.gz",
        source_trajectory_sha256="a" * 64,
    )
    assert len(minutes) == 180
    assert minutes[0]["combined_quality"] == 0.0
    assert minutes[1]["effective_source"] == "symbolfit_active_pysr_hall_of_fame"
    assert minutes[2]["carry_forward"] == "true"
    assert minutes[2]["carry_forward_from_minute"] == 2
    assert minutes[3]["carry_forward"] == "false"
    assert minutes[-1]["record_type"] == "budget_end_internal_best"
    assert all(row["test_metric_used_for_candidate_selection"] == "false" for row in minutes)
    assert summary["heartbeat_snapshot_count"] == 1
    assert summary["candidate_snapshot_count"] == 179
    assert summary["m_eff"] == pytest.approx(
        sum(row["relative_progress"] for row in minutes) / 180
    )


def test_build_run_trajectory_rejects_non_internal_candidate() -> None:
    payloads = [_candidate(minute, equation="x0", loss=0.2) for minute in range(1, 181)]
    payloads[9]["candidate_source"] = "test_selected_candidate"
    with pytest.raises(SymbolFitReleaseMetricError, match="非内部搜索候选"):
        build_run_trajectory(
            _row(),
            payloads,
            source_trajectory_path="source.jsonl.gz",
            source_trajectory_sha256="a" * 64,
        )


def test_gzip_csv_is_deterministic(tmp_path: Path) -> None:
    row = {field: "" for field in MINUTE_FIELDS}
    row["logical_key"] = "key"
    left = tmp_path / "left.csv.gz"
    right = tmp_path / "right.csv.gz"
    _write_gzip_csv(left, [row], MINUTE_FIELDS)
    _write_gzip_csv(right, [row], MINUTE_FIELDS)
    assert hashlib.sha256(left.read_bytes()).hexdigest() == hashlib.sha256(right.read_bytes()).hexdigest()
