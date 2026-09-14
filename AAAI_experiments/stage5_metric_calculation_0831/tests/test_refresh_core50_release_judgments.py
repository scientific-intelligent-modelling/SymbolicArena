import hashlib
import json

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.refresh_core50_release_judgments import (
    RefreshError,
    _normalized_index_row,
    can_reuse,
    materialize_effective_clean_judgments,
)


def test_same_result_hash_does_not_allow_changed_formula_reuse():
    request = {"prediction_result_sha256": "same", "prediction_frozen_evaluation_key": "p",
               "ground_truth_frozen_evaluation_key": "g", "effective_prediction_expression": "t",
               "effective_ground_truth_expression": "P"}
    plan = {"evaluation_key": "e", "dependencies": ["g", "p"], "request": request}
    index = {"evaluation_key": "e", "state": "frozen"}
    assert can_reuse(plan, index, request, ["p", "g"], "equivalence")
    assert not can_reuse(plan, index, {**request, "effective_prediction_expression": "P"}, ["p", "g"], "equivalence")
    assert not can_reuse(plan, index, {**request, "ground_truth_frozen_evaluation_key": "g_new"}, ["p", "g_new"], "equivalence")


def test_structure_validity_change_invalidates_old_judgment():
    request = {"prediction_a_result_sha256": "a", "prediction_b_result_sha256": "b",
               "prediction_a_frozen_evaluation_key": "pa", "prediction_b_frozen_evaluation_key": "pb",
               "effective_prediction_a_expression": "x", "effective_prediction_b_expression": "x",
               "prediction_a_valid_output": True, "prediction_b_valid_output": True}
    plan = {"evaluation_key": "s", "dependencies": ["pa", "pb"], "request": request}
    index = {"evaluation_key": "s", "state": "frozen"}
    assert can_reuse(plan, index, request, ["pa", "pb"], "structure")
    assert not can_reuse(plan, index, {**request, "prediction_b_valid_output": False}, ["pa", "pb"], "structure")


def test_composed_index_rebinds_plan_hash_without_changing_frozen_response():
    plan = {"logical_id": "pred_simplify::a::g0001::s520::clean", "evaluation_key": "e",
            "task_type": "pred_simplify", "condition": "clean", "priority": 20,
            "request": {"original_expression": "x"}}
    source = {"logical_id": plan["logical_id"], "evaluation_key": "e", "state": "frozen",
              "attempt_id": "e.a01", "result_path": "frozen/e.json", "frozen_response_sha256": "r",
              "structured_output": {"outcome": "unchanged", "simplified_expression": "x"},
              "effective_expression": "x", "expression_resolution": "llm_simplified_expression",
              "evidence_generation": "preexisting"}
    row = _normalized_index_row(plan=plan, source=source, plan_sha256="p" * 64)
    assert row["plan_sha256"] == "p" * 64
    assert row["result_sha256"] == "r"
    assert row["evaluation_key"] == source["evaluation_key"]
    assert row["evidence_generation"] == "preexisting"


def test_composed_index_rejects_effective_expression_not_returned_by_model():
    plan = {"logical_id": "pred", "evaluation_key": "e", "task_type": "pred_simplify",
            "condition": "clean", "priority": 20, "request": {"original_expression": "x"}}
    source = {"evaluation_key": "e", "state": "frozen", "structured_output": {
        "outcome": "unchanged", "simplified_expression": "x"}, "effective_expression": "y"}
    with pytest.raises(RefreshError, match="Opus"):
        _normalized_index_row(plan=plan, source=source, plan_sha256="p")


def test_audit_decision_is_separate_from_underlying_opus_response(tmp_path):
    eq = tmp_path / "eq.jsonl"
    compact = tmp_path / "compact.jsonl"
    structure = tmp_path / "structure.jsonl"
    overlay = tmp_path / "overlay.jsonl"
    eq.write_text(json.dumps({"logical_id": "eq", "evaluation_key": "e", "structured_output": {"decision": "equivalent"}}) + "\n")
    compact.write_text(json.dumps({"logical_id": "eq", "evaluation_key": "e", "effective_decision": "not_equivalent", "audit_overlay_applied": True}) + "\n")
    structure.write_text(json.dumps({"logical_id": "s", "structured_output": {"decision": "same_canonical_structure"}}) + "\n")
    overlay.write_text(json.dumps({"logical_id": "s", "evaluation_key": "o", "result_path": "o.json", "result_sha256": "h", "structured_output": {"decision": "different_structure"}}) + "\n")
    report = materialize_effective_clean_judgments(eq_index_path=eq, compact_eq_path=compact,
        structure_index_path=structure, structure_overlay_path=overlay, output=tmp_path / "out")
    row = json.loads((tmp_path / "out/clean_equivalence_effective_index.jsonl").read_text())
    assert row["structured_output"]["decision"] == "equivalent"
    assert row["effective_decision"] == "not_equivalent"
    assert report["equivalence_audit_overlays"] == 1
