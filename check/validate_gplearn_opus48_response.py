"""Validate an Opus simplification against gplearn's native protected evaluator."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.gplearn_native_prefix_evidence import (
    PrefixEvidenceError,
    build_prefix_evidence,
    native_equivalence_probe,
)


SEMANTICS_VERSION = "gplearn_native_protected_prefix.v1"


def _reject(reason: str, **details: Any) -> dict[str, Any]:
    return {"status": "semantic_rejected", "error": reason, **details}


def validate_response(request: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("native_semantics_version") != SEMANTICS_VERSION:
        return _reject("native_semantics_version_mismatch")
    artifact = request.get("ast_source_evidence", {}).get("canonical_artifact", {})
    native_prefix = artifact.get("native_prefix") if isinstance(artifact, Mapping) else None
    if not isinstance(native_prefix, str) or not native_prefix.strip():
        return _reject("native_prefix_missing")
    if hashlib.sha256(native_prefix.encode()).hexdigest() != request.get("native_prefix_sha256"):
        return _reject("native_prefix_sha256_mismatch")
    source = request.get("expression")
    if not isinstance(source, str) or not source.strip():
        return _reject("typed_source_missing")
    features = request.get("feature_names")
    if not isinstance(features, list) or not features:
        return _reject("feature_names_missing")
    try:
        native_evidence = build_prefix_evidence(native_prefix, features)
        source_evidence = build_prefix_evidence(source, features)
    except (PrefixEvidenceError, ValueError, TypeError, OverflowError) as exc:
        return _reject(f"plan_source_validation_error:{type(exc).__name__}:{exc}")
    if native_evidence["exact_fingerprint"] != source_evidence["exact_fingerprint"]:
        return _reject("typed_plan_source_differs_from_native_prefix")
    outcome = output.get("outcome")
    candidate = output.get("simplified_expression")
    assessment = output.get("equivalence_assessment")
    if outcome == "unable":
        if candidate is not None or assessment != "undetermined":
            return _reject("unable_output_contract_mismatch")
        return {"status": "promotable", "resolution": "identity_after_model_unable",
                "effective_expression": source, "native_prefix_sha256": request["native_prefix_sha256"],
                "validation_basis": "native_identity_fallback"}
    if outcome not in {"simplified", "unchanged"} or assessment != "preserved":
        return _reject("simplification_output_contract_mismatch")
    if not isinstance(candidate, str) or not candidate.strip():
        return _reject("simplified_expression_missing")
    try:
        if candidate.strip() == source.strip():
            return {"status": "promotable", "resolution": "typed_identity",
                    "effective_expression": source,
                    "native_prefix_sha256": request["native_prefix_sha256"],
                    "source_typed_fingerprint": source_evidence["exact_fingerprint"],
                    "validation_basis": "identical_protected_typed_program"}
        if outcome == "unchanged":
            return _reject("unchanged_output_differs_from_input")
        points = np.asarray(request.get("probe_points"), dtype=float)
        probe = native_equivalence_probe(native_prefix, candidate, features, points,
                                         include_boundary_probes=True)
    except (PrefixEvidenceError, ValueError, TypeError, OverflowError) as exc:
        return _reject(f"native_validation_error:{type(exc).__name__}:{exc}")
    if not probe["passed"]:
        return _reject(f"native_validation_{probe['status']}", probe=probe)
    return {"status": "promotable", "resolution": "llm_simplified_native_probe_supported",
            "effective_expression": candidate.strip(),
            "native_prefix_sha256": request["native_prefix_sha256"],
            "source_typed_fingerprint": source_evidence["exact_fingerprint"],
            "validation_basis": "opus_preservation_claim_plus_native_and_boundary_probes",
            "probe": probe}
