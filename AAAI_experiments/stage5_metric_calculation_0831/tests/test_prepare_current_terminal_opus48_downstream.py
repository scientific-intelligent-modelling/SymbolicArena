"""Focused fail-closed checks for current-terminal downstream planning."""

from __future__ import annotations

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
    prepare_current_terminal_opus48_downstream as downstream,
    symbolic_task_builder as builder,
)


def test_legacy_frozen_binding_uses_request_evidence() -> None:
    plan = {"evaluation_key": "source-key", "input_hash": "input-hash",
            "request": {"original_expression": "x0 + 1"}}
    current = {"terminal_expression": "x0+1", "terminal_expression_sha256": "formula-sha",
               "terminal_source_sha256": "snapshot-sha"}
    raw = {"status": "frozen", "source_evaluation_key": "source-key",
           "source_input_hash": "input-hash", "terminal_expression": "x0 + 1",
           "semantic_validation": {"status": "promotable"},
           "request": {"terminal_binding_evidence": {
               "terminal_expression_sha256": "formula-sha", "terminal_snapshot_sha256": "snapshot-sha"}}}
    assert downstream._frozen_binding_issues(raw, plan, current) == []
    current["terminal_source_sha256"] = "different-snapshot"
    assert downstream._frozen_binding_issues(raw, plan, current) == ["terminal_snapshot_sha256"]


def test_historical_decision_is_only_a_candidate_on_exact_inputs(monkeypatch) -> None:
    monkeypatch.setattr(downstream, "verify_response_artifact", lambda *_: True)
    old_request = {"algorithm_slug": "pysr", "dataset_id": "BPG3", "variables": ["x0"],
                   "effective_prediction_expression": "x0 + 1",
                   "effective_ground_truth_expression": "x0",
                   "prediction_result_sha256": "a" * 64,
                   "ground_truth_frozen_evaluation_key": "gt-key"}
    old = ({"evaluation_key": "old-key", "request": old_request},
           {"state": "frozen", "evaluation_key": "old-key", "response_path": "artifact.json",
            "response_sha256": "b" * 64, "effective_decision": "not_equivalent"})
    current = {**old_request, "prediction_current_selected_result_sha256": "a" * 64}
    candidate = downstream._candidate(old, current, "equivalence")
    assert candidate is not None
    assert candidate["reuse_status"] == "candidate_only_requires_review"
    assert downstream._candidate(old, {**current, "effective_prediction_expression": "x0 + 2"},
                                 "equivalence") is None
    assert downstream._candidate(old, {**current, "prediction_current_selected_result_sha256": "c" * 64},
                                 "equivalence") is None


def test_model_identity_changes_source_task_key() -> None:
    contract = builder._load_prompt_schema(downstream.REPO_ROOT, task_kind="equivalence")
    common = {"logical_id": "equivalence::test::g0001::s520::clean",
              "task_type": "equivalence", "priority": 30, "evidence_hash": "d" * 64,
              "contract": contract, "dependencies": ("gt", "pred"), "condition": "clean"}
    opus5 = builder._task_from_request(request={"expression": "x"}, **common)
    opus48 = builder._task_from_request(request={"expression": "x", "model_binding": downstream.MODEL}, **common)
    assert opus5.evaluation_key != opus48.evaluation_key
    assert opus48.to_json_record()["request"]["model_binding"] == "claude-opus-4-8"


def test_structure_reuse_rejects_changed_seed_expression(monkeypatch) -> None:
    monkeypatch.setattr(downstream, "verify_response_artifact", lambda *_: True)
    old_request = {"algorithm_slug": "pysr", "dataset_id": "BPG3",
                   "effective_prediction_a_expression": "x0", "effective_prediction_b_expression": "x0 + 1",
                   "prediction_a_result_sha256": "a" * 64, "prediction_b_result_sha256": "b" * 64,
                   "prediction_a_valid_output": True, "prediction_b_valid_output": True}
    old = ({"evaluation_key": "old-key", "request": old_request},
           {"state": "frozen", "evaluation_key": "old-key", "response_path": "artifact.json",
            "response_sha256": "c" * 64})
    current = {**old_request, "prediction_a_current_selected_result_sha256": "a" * 64,
               "prediction_b_current_selected_result_sha256": "b" * 64}
    assert downstream._candidate(old, current, "structure") is not None
    assert downstream._candidate(old, {**current, "effective_prediction_b_expression": "x0 + 2"},
                                 "structure") is None


def test_pair_evidence_runs_in_bounded_child(monkeypatch) -> None:
    monkeypatch.setattr(builder, "_build_full_pair_evidence", lambda **kwargs: {"value": kwargs["value"]})
    assert downstream._pair_evidence_isolated(timeout_seconds=2, memory_limit_bytes=2 * 1024**3,
                                              value=17) == {"value": 17}


def test_pair_evidence_timeout_is_unresolved(monkeypatch) -> None:
    import time

    monkeypatch.setattr(builder, "_build_full_pair_evidence", lambda **_kwargs: time.sleep(1))
    with pytest.raises(downstream.DownstreamPlanError, match="timeout"):
        downstream._pair_evidence_isolated(timeout_seconds=0.01, memory_limit_bytes=2 * 1024**3)
