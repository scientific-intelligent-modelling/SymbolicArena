import unittest

import numpy as np
import pandas as pd
import sympy as sp

from feyn._program import Program

from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import (
    QLatticeRegressor,
)


def _native_linear_model(*, output_bias: float = -1234.5678901234567):
    return Program(["y", "x0"]).to_model(
        params=[
            {
                "scale": 1.2345678901234567,
                "scale_offset": 0.0,
                "w": 9.876543210987654,
                "bias": output_bias,
            },
            {
                "scale": 0.9876543210987654,
                "scale_offset": -12.345678901234567,
                "w": 1.1111111111111112,
                "bias": 0.12345678901234568,
            },
        ]
    )


def _evaluate(equation: str, x: np.ndarray) -> np.ndarray:
    fn = sp.lambdify(sp.Symbol("x0"), sp.sympify(equation), modules="numpy")
    return np.asarray(fn(x), dtype=float)


class QLatticeNativeExportTest(unittest.TestCase):
    def test_native_export_uses_full_precision_despite_display_signif(self) -> None:
        model = _native_linear_model()
        x = np.array([-1.0e6, -1.0, 0.0, 1.0, 1.0e6])
        native = np.asarray(model.predict(pd.DataFrame({"x0": x})), dtype=float)
        four_digit = str(model.sympify(signif=4))

        reg = QLatticeRegressor(signif=4, target_standardize=False)
        reg.model = True
        reg._best_model = model
        reg._models = [model]
        reg._input_vars = ["x0"]
        reg._expr_str = reg._model_equation(model)
        reg._equations = [reg._expr_str]

        self.assertEqual(reg.params["signif"], 4)
        self.assertEqual(reg.get_optimal_equation(), str(model.sympify(signif=17)))
        self.assertEqual(reg.get_total_equations(), [reg.get_optimal_equation()])
        self.assertGreater(np.max(np.abs(native - _evaluate(four_digit, x))), 1.0)

        restored = QLatticeRegressor.deserialize(reg.serialize())
        np.testing.assert_allclose(
            restored.predict(x.reshape(-1, 1)), native, rtol=1.0e-14, atol=1.0e-8
        )

    def test_full_precision_export_preserves_target_scale(self) -> None:
        model = _native_linear_model(output_bias=20.987654321098765)
        reg = QLatticeRegressor(signif=4)
        reg._target_was_standardized = True
        reg._target_offset = 123456.78901234567
        reg._target_scale = 9876.543210987654

        equation = reg._model_equation(model)
        x = np.array([-3.0, 0.0, 4.0])
        expected = reg._restore_target_scale(model.predict(pd.DataFrame({"x0": x})))
        np.testing.assert_allclose(
            _evaluate(equation, x), expected, rtol=1.0e-14, atol=1.0e-8
        )


if __name__ == "__main__":
    unittest.main()
