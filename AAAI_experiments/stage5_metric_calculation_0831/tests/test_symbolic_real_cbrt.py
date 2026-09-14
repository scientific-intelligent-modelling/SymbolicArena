import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SymbolicEvidenceError,
    build_symbolic_artifact,
)


@pytest.mark.parametrize("spelling", ["real_cbrt", "cbrt", "np.cbrt"])
def test_real_cube_root_preserves_negative_and_zero_inputs(spelling):
    artifact = build_symbolic_artifact(f"{spelling}(x)", allowed_variables={"x"}, allowed_functions={"cbrt"})
    expression = artifact["sympy_expression"]
    variable = next(iter(expression.free_symbols))
    for value, expected in [(-8, -2), (0, 0), (27, 3)]:
        assert float(expression.subs(variable, value).evalf()) == expected


def test_real_cube_root_roundtrip_keeps_unary_tree_and_semantics():
    artifact = build_symbolic_artifact("real_cbrt(x)")
    assert artifact["node_count"] == 2
    restored = build_symbolic_artifact(artifact["canonical_expression"])
    assert restored["artifact_sha256"] == artifact["artifact_sha256"]


def test_real_cube_root_still_requires_function_authorization():
    with pytest.raises(SymbolicEvidenceError, match="未授权函数"):
        build_symbolic_artifact("real_cbrt(x)", allowed_functions={"sqrt"})
