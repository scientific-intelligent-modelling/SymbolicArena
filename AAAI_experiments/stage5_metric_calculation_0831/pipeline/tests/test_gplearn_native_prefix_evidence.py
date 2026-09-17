"""gplearn 受保护前缀语义的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.gplearn_native_prefix_evidence import (
    PrefixEvidenceError,
    build_inventory_prefix_evidence,
    build_prefix_evidence,
    compare_with_native,
    evaluate_prefix_evidence,
    native_equivalence_probe,
    typed_tree_similarity,
)
from scientific_intelligent_modelling.benchmarks.runner import _predict_gplearn_prefix_expression
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import tree_similarity


def test_protected_ops_keep_runner_threshold_and_are_typed():
    evidence = build_prefix_evidence(
        "add(div(X0, X1), add(log(X1), sqrt(X0)))", ["a", "b"]
    )
    X = np.array([[-4.0, 0.001], [-4.0, -0.002], [9.0, 0.0]])
    predicted = evaluate_prefix_evidence(evidence, X)
    native = _predict_gplearn_prefix_expression(evidence["raw_prefix"], X)
    assert np.allclose(predicted, native)
    assert predicted[0] == pytest.approx(3.0)  # div=1, log=0, sqrt(abs(-4))=2
    assert evidence["nodes"][evidence["root_index"]]["operator"] == "add"
    assert {"protected_div", "protected_log", "protected_sqrt"} <= set(evidence["operator_set"])
    assert evidence["protected_threshold"] == 0.001
    assert "pdiv(" in evidence["typed_expression"]
    assert "plog(" in evidence["typed_expression"]
    assert "psqrt(" in evidence["typed_expression"]


def test_deep_prefix_and_nan_normalized_string_do_not_need_ast_or_sympy():
    depth = 1200
    raw = "sqrt(" * depth + "X0" + ")" * depth
    evidence = build_prefix_evidence(raw, ["x"], normalized_expression="nan")
    assert evidence["node_count"] == depth + 1
    assert evidence["tree_depth"] == depth + 1
    assert evidence["nodes"][evidence["root_index"]]["operator"] == "protected_sqrt"
    assert evaluate_prefix_evidence(evidence, np.array([[-4.0]])).shape == (1,)


def test_unsupported_variable_and_wrong_arity_fail_closed():
    with pytest.raises(PrefixEvidenceError):
        build_prefix_evidence("add(X0, X2)", ["x", "y"])
    with pytest.raises(PrefixEvidenceError):
        build_prefix_evidence("div(X0)", ["x"])
    with pytest.raises(PrefixEvidenceError):
        build_prefix_evidence("add(X0, X0) X0", ["x"])


def test_numeric_validator_includes_protected_boundaries():
    evidence = build_prefix_evidence("div(X0, X1)", ["x", "y"])
    X = np.array([[2.0, 0.5]])
    result = compare_with_native(
        evidence, X,
        native_predict=lambda values: _predict_gplearn_prefix_expression(evidence["raw_prefix"], values),
    )
    assert result["passed"] and result["boundary_probe_count"] > 0
    def ordinary_division(values):
        with np.errstate(all="ignore"):
            return values[:, 0] / values[:, 1]

    bad = compare_with_native(evidence, X, native_predict=ordinary_division)
    assert not bad["passed"] and bad["mismatch_count"] > 0


def test_typed_candidate_probe_returns_support_or_boundary_counterexample():
    X = np.array([[2.0, 0.5]])
    same = native_equivalence_probe("div(X0, X1)", "pdiv(x0,x1)", ["x", "y"], X)
    assert same["passed"] and same["status"] == "numeric_support_only"
    changed = native_equivalence_probe("div(X0, X1)", "mul(x0,pinv(x1))", ["x", "y"], X)
    assert not changed["passed"] and changed["status"] == "numeric_counterexample"
    with pytest.raises(PrefixEvidenceError):
        native_equivalence_probe("div(X0, X1)", "div(x0,x1)", ["x", "y"], X)


def test_frozen_inventory_binds_prefix_sha_and_feature_mapping():
    import hashlib

    raw = "log(X0)"
    row = {
        "algorithm": "gplearn", "dataset_id": "task-a", "condition": "clean", "seed": 520,
        "logical_key": "gplearn::task-a::s520::clean", "native_prefix": raw,
        "native_prefix_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "terminal_snapshot_sha256": "snapshot", "feature_names": ["x"],
        "variable_mapping": {"X0": "x"}, "display_expression": "nan",
        "terminal_expression": "nan",
        "terminal_expression_sha256": hashlib.sha256(b"nan").hexdigest(),
    }
    evidence = build_inventory_prefix_evidence(row)
    assert evidence["typed_expression"] == "plog(x0)"
    assert evidence["normalized_expression_diagnostic"] == "nan"
    row["native_prefix_sha256"] = "old-prefix"
    with pytest.raises(PrefixEvidenceError):
        build_inventory_prefix_evidence(row)


def test_prefix_unparsed_uses_terminal_raw_prefix_not_empty_display():
    import hashlib

    raw = "sqrt(log(div(X0, 0.001)))"
    row = {
        "algorithm": "gplearn", "dataset_id": "task-a", "condition": "clean", "seed": 520,
        "logical_key": "gplearn::task-a::s520::clean", "native_prefix": raw,
        "native_prefix_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "terminal_snapshot_sha256": "snapshot", "feature_names": ["x"],
        "variable_mapping": {"X0": "x"}, "display_expression": "",
        "display_normalization_mode": "gplearn_prefix_unparsed",
        "terminal_expression": raw,
        "terminal_expression_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    evidence = build_inventory_prefix_evidence(row)
    assert evidence["typed_expression"] == "psqrt(plog(pdiv(x0,0.001)))"
    assert evidence["source_binding"]["terminal_expression_sha256"] == row["native_prefix_sha256"]


def test_iterative_tree_distance_matches_same_label_ned_and_guards_deep_tree():
    evidence = build_prefix_evidence("add(X0, 1)", ["x"])
    reference = {"type": "add", "args": [
        {"type": "Symbol", "value": "Symbol('x')"},
        {"type": "One", "value": "Integer(1)"},
    ]}
    matching = typed_tree_similarity(evidence, reference)
    assert matching["status"] == "computed" and matching["tree_similarity"] == 1
    changed = typed_tree_similarity(evidence, {"type": "Symbol", "value": "Symbol('x')"})
    assert changed["status"] == "computed" and 0 < changed["tree_similarity"] < 1
    assert changed["tree_similarity"] == tree_similarity(
        {"canonical_tree": reference},
        {"canonical_tree": {"type": "Symbol", "value": "Symbol('x')"}},
    )
    deep = build_prefix_evidence("sqrt(" * 1200 + "X0" + ")" * 1200, ["x"])
    guarded = typed_tree_similarity(deep, reference, max_pred_nodes=1000)
    assert guarded["status"] == "unavailable_complexity"
    assert guarded["tree_similarity"] is None
