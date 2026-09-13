from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_symbolfit_final_20260913.py"
SPEC = importlib.util.spec_from_file_location("collect_symbolfit_final_20260913", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_result_accepts_noise_contract() -> None:
    equation = "x0 + 1"
    artifact = {
        "artifact_valid": True,
        "raw_equation": equation,
        "tool_name": "symbolfit",
    }
    result = {
        "tool": "symbolfit",
        "status": "ok",
        "seed": 520,
        "condition": "noise001",
        "task_global_index": 1,
        "dataset": "Keijzer-11",
        "dataset_identity_check": {"match": True},
        "equation": equation,
        "canonical_artifact": artifact,
        "canonical_artifact_error": None,
        "params": {
            "timeout_in_seconds": 10800,
            "niterations": 1000000,
            "maxsize": 25,
            "max_complexity": 25,
            "procs": 1,
            "parallelism": "serial",
            "deterministic": True,
        },
        "train_label_noise": {
            "enabled": True,
            "sigma": 0.01,
            "protocol": "noise; clean labels are used for evaluation",
        },
        "id_test": {"nmse": 0.1},
        "ood_test": {"nmse": 0.2},
    }
    actual, detail = MODULE.validate_result(
        result,
        "symbolfit_s520_noise001_g0001",
        {"dataset_id": "Keijzer-11"},
    )
    assert actual == equation
    assert detail["id_nmse"] == pytest.approx(0.1)


def test_matching_candidate_requires_unique_binding(tmp_path: Path) -> None:
    transform = {"version": "symbolfit_affine_v1"}
    candidate = {
        "attempt": 2,
        "complexity": 17,
        "internal_loss": 0.125,
        "source": "symbolfit_active_pysr_hall_of_fame",
        "coordinate_transform": transform,
        "first_discovered_attempt": 2,
        "first_discovered_minute": 42,
    }
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    endpoint = {
        "source_internal_loss": 0.125,
        "candidate_attempt": 2,
        "source_complexity": 17,
        "candidate_source": "symbolfit_active_pysr_hall_of_fame",
        "candidate_coordinate_transform": transform,
        "candidate_first_discovered_attempt": 2,
        "candidate_first_discovered_minute": 42,
    }
    matched = MODULE.matching_candidate(history, endpoint)
    assert matched["matching_history_row_count"] == 1
    assert {key: matched[key] for key in candidate} == candidate


def test_snapshot_validation_is_fail_closed(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    progress.mkdir()
    (progress / "minute_0001.json").write_text(
        json.dumps({"checkpoint_index": 1, "candidate_available": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="分钟快照不完整"):
        MODULE.validate_snapshots(progress, "task", write_to=None)


def test_snapshot_validation_excludes_post_budget_minute(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    progress.mkdir()
    for minute in range(1, 182):
        (progress / f"minute_{minute:04d}.json").write_text(
            json.dumps({"checkpoint_index": minute, "candidate_available": False}),
            encoding="utf-8",
        )
    stats = MODULE.validate_snapshots(progress, "task", write_to=None)
    assert stats["snapshot_count"] == 180
    assert stats["excluded_post_budget_snapshot_count"] == 1
    assert stats["excluded_post_budget_minutes"] == "181"
