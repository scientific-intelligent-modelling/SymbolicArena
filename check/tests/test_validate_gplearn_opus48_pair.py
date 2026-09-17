"""Protected pair judgments cannot override frozen counterexamples."""

from check.validate_gplearn_opus48_pair import validate_pair_response


def test_equivalent_is_rejected_after_gt_counterexample() -> None:
    request = {"deterministic_evidence": {
        "schema_version": "gplearn_protected_pair_evidence.v1",
        "prediction_vs_gt": {"status": "numeric_counterexample", "first_counterexample": {"point_index": 0}},
    }}
    assert validate_pair_response("equivalence", request, {"decision": "equivalent"})["status"] == "semantic_rejected"
    assert validate_pair_response("equivalence", request, {"decision": "not_equivalent"})["status"] == "promotable"


def test_structural_positive_is_rejected_after_pair_counterexample() -> None:
    request = {"deterministic_evidence": {
        "schema_version": "gplearn_protected_pair_evidence.v1",
        "typed_structure_consistency": False,
        "native_protected_numeric_comparison": {"status": "numeric_counterexample", "counterexample": {"point_index": 1}},
    }}
    assert validate_pair_response("stab_structure", request,
                                  {"decision": "mathematically_equivalent"})["status"] == "semantic_rejected"
    assert validate_pair_response("stab_structure", request,
                                  {"decision": "different_structure"})["status"] == "promotable"


def test_different_structure_rejected_when_fingerprints_match() -> None:
    request = {"deterministic_evidence": {
        "schema_version": "gplearn_protected_pair_evidence.v1",
        "typed_structure_consistency": True,
        "native_protected_numeric_comparison": {"status": "numeric_support_only"},
    }}
    assert validate_pair_response("stab_structure", request,
                                  {"decision": "different_structure"})["status"] == "semantic_rejected"
