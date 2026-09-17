"""Deterministic gplearn terminal symbolic components on protected typed trees."""

from __future__ import annotations

from typing import Any, Mapping

from .gplearn_native_prefix_evidence import typed_tree_similarity
from .metrics import minimality_score, symbolic_fidelity_score
from .symbolic_evidence import operator_f1, variable_f1


class GplearnMetricError(ValueError):
    pass


EQ_LABELS = {"equivalent", "not_equivalent", "undetermined", "not_established"}
POSITIVE_STRUCTURE = {"mathematically_equivalent", "same_canonical_structure"}
STRUCTURE_LABELS = POSITIVE_STRUCTURE | {"different_structure", "undetermined", "not_established"}


def structure_is_positive(label: str) -> bool:
    if label not in STRUCTURE_LABELS:
        raise GplearnMetricError(f"unknown structure label: {label}")
    return label in POSITIVE_STRUCTURE


def score_symbolic_run(
    prediction: Mapping[str, Any], reference: Mapping[str, Any],
    equivalence_label: str,
) -> dict[str, Any]:
    if equivalence_label not in EQ_LABELS:
        raise GplearnMetricError(f"unknown equivalence label: {equivalence_label}")
    tree = typed_tree_similarity(
        dict(prediction), dict(reference["canonical_tree"]),
        max_pred_nodes=20000, max_ref_nodes=100, max_pair_cells=1_000_000,
    )
    if tree["status"] != "computed":
        raise GplearnMetricError(f"typed tree similarity unavailable: {tree}")
    predicted_complexity = int(prediction["node_count"])
    reference_complexity = int(reference["node_count"])
    variables = variable_f1(prediction, reference)
    operators = operator_f1(prediction, reference)
    equivalent = equivalence_label == "equivalent"
    return {
        "equivalence_label": equivalence_label,
        "equivalence_established": equivalent,
        "tree_similarity": tree["tree_similarity"],
        "tree_evidence": tree,
        "variable_f1": variables,
        "operator_f1": operators,
        "c_pred": predicted_complexity,
        "c_ref": reference_complexity,
        "m_sym": symbolic_fidelity_score(
            equivalent=equivalent, tree_similarity=tree["tree_similarity"],
            variable_f1=variables, operator_f1=operators),
        "m_min": minimality_score(reference_complexity, predicted_complexity),
    }
