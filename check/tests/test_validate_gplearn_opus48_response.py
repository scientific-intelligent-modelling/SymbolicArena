"""Protected operators must survive model simplification validation."""

from __future__ import annotations

from check.validate_gplearn_opus48_response import validate_response


def _request() -> dict:
    import hashlib

    native = "div(X0,X1)"
    typed = "pdiv(x0,x1)"
    return {
        "expression": typed, "native_prefix_sha256": hashlib.sha256(native.encode()).hexdigest(),
        "feature_names": ["a", "b"],
        "probe_points": [[1.0, 0.0], [1.0, 0.0009], [1.0, 0.0011]],
        "native_semantics_version": "gplearn_native_protected_prefix.v1",
        "ast_source_evidence": {"canonical_artifact": {"native_prefix": native}},
    }


def test_unable_uses_identity_fallback() -> None:
    result = validate_response(_request(), {"outcome": "unable", "simplified_expression": None,
                                           "equivalence_assessment": "undetermined"})
    assert result["status"] == "promotable"
    assert result["effective_expression"] == "pdiv(x0,x1)"
    assert result["resolution"] == "identity_after_model_unable"


def test_protected_candidate_passes_native_boundary_check() -> None:
    result = validate_response(_request(), {"outcome": "unchanged", "simplified_expression": "pdiv(x0,x1)",
                                           "equivalence_assessment": "preserved"})
    assert result["status"] == "promotable"


def test_bare_div_is_rejected() -> None:
    result = validate_response(_request(), {"outcome": "simplified", "simplified_expression": "div(x0,x1)",
                                           "equivalence_assessment": "preserved"})
    assert result["status"] == "semantic_rejected"


def test_plan_source_must_match_native_prefix_even_for_unable() -> None:
    request = _request()
    request["expression"] = "add(x0,x1)"
    result = validate_response(request, {"outcome": "unable", "simplified_expression": None,
                                         "equivalence_assessment": "undetermined"})
    assert result["status"] == "semantic_rejected"
