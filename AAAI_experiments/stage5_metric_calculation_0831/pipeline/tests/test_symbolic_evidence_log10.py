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
