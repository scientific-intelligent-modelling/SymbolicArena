"""Keep feature indexing separate from legitimate fitted-parameter indexing."""

from check.audit_scalar_trajectory_expressions import classify


def test_feature_column_indexing_is_flagged() -> None:
    assert classify("x0[0] + 2*x1") == ("indexed_feature_column",)
    assert classify("params[0]*x0 + params[1]") == ()


def test_nonfinite_expression_is_flagged() -> None:
    assert classify("nan") == ("nonfinite_literal",)
    assert classify("x0 + np.nan") == ("nonfinite_literal",)


def test_parameter_array_annotation_is_not_feature_indexing() -> None:
    source = "def equation(x0: np.ndarray, params: np.ndarray):\n    return params[0]*x0\n"
    assert classify(source) == ()
