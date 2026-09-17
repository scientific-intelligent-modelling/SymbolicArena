"""Fail closed on Opus pair labels that contradict protected native evidence."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = "gplearn_protected_pair_evidence.v1"


def validate_pair_response(phase: str, request: Mapping[str, Any],
                           output: Mapping[str, Any]) -> dict[str, Any]:
    evidence = request.get("deterministic_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != SCHEMA_VERSION:
        return {"status": "semantic_rejected", "error": "protected_pair_evidence_missing"}
    decision = output.get("decision")
    if phase == "equivalence":
        if decision not in {"equivalent", "not_equivalent", "undetermined"}:
            return {"status": "semantic_rejected", "error": "equivalence_label_invalid"}
        numeric = evidence.get("prediction_vs_gt") or {}
        if numeric.get("status") == "numeric_counterexample" and decision == "equivalent":
            return {"status": "semantic_rejected", "error": "equivalent_contradicts_gt_counterexample",
                    "counterexample": numeric.get("first_counterexample")}
    elif phase == "stab_structure":
        if decision not in {"mathematically_equivalent", "same_canonical_structure",
                            "different_structure", "undetermined"}:
            return {"status": "semantic_rejected", "error": "structure_label_invalid"}
        numeric = evidence.get("native_protected_numeric_comparison") or {}
        same_structure = evidence.get("typed_structure_consistency")
        if numeric.get("status") == "numeric_counterexample" and decision == "mathematically_equivalent":
            return {"status": "semantic_rejected", "error": "equivalent_contradicts_seed_pair_counterexample",
                    "counterexample": numeric.get("counterexample")}
        if same_structure is True and decision == "different_structure":
            return {"status": "semantic_rejected", "error": "different_contradicts_typed_structure_fingerprint"}
        # An ordered prefix fingerprint can differ after a commutative swap;
        # its inequality is not a proof that canonical structures differ.
    else:
        return {"status": "semantic_rejected", "error": f"unsupported_phase:{phase}"}
    return {"status": "promotable", "validation_basis": "protected_evidence_noncontradiction",
            "evidence_sha256": evidence.get("evidence_sha256"),
            "decision": decision}
