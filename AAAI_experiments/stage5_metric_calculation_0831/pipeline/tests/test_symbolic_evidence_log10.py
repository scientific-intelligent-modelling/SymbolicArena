"""Stage6 ground-truth evidence must preserve NumPy base-ten logarithms."""

import sympy as sp

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    build_symbolic_artifact,
)


def test_numpy_log10_is_a_base_ten_logarithm() -> None:
    artifact = build_symbolic_artifact(
        "np.log10(x0)", allowed_variables={"x0"}, allowed_functions={"log10"}
    )

    x0 = sp.Symbol("x0")
    assert artifact["function_set"] == ("log10",)
    assert sp.simplify(artifact["sympy_expression"] - sp.log(x0, 10)) == 0
    assert float(artifact["sympy_expression"].subs(x0, 10)) == 1.0


def test_numpy_nan_to_num_is_preserved_and_evaluates_on_finite_probe() -> None:
    artifact = build_symbolic_artifact(
        "np.nan_to_num(x0)",
        allowed_variables={"x0"},
        allowed_functions={"nan_to_num"},
    )

    x0 = sp.Symbol("x0")
    expression = artifact["sympy_expression"]
    assert artifact["function_set"] == ("nan_to_num",)
    assert "nan_to_num" in artifact["operator_set"]
    assert expression.subs(x0, 1.25).evalf() == sp.Float("1.25")


def test_arctan2_is_supported_with_numpy_argument_order() -> None:
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    for source in ("np.arctan2(x0, x1)", "arctan2(x0, x1)"):
        artifact = build_symbolic_artifact(
            source,
            allowed_variables={"x0", "x1"},
            allowed_functions={"arctan2"},
        )
        expression = artifact["sympy_expression"]
        assert artifact["function_set"] == ("arctan2",)
        assert sp.simplify(expression - sp.atan2(x0, x1)) == 0
        assert sp.simplify(expression.subs({x0: 1, x1: 0}) - sp.pi / 2) == 0
