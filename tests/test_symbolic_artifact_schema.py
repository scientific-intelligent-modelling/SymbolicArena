import unittest

from scientific_intelligent_modelling.algorithms.base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.artifact_schema import (
    CSP_VERSION,
    build_canonical_symbolic_program,
    extract_return_expression_from_python_function,
    infer_raw_equation_kind,
    instantiate_expression,
    validate_canonical_symbolic_program,
)


class _DummyWrapper(BaseWrapper):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X

    def get_optimal_equation(self):
        return "def equation(x0, x1, params):\n    return params[0] + x0 * x1\n"

    def get_total_equations(self):
        return [self.get_optimal_equation()]

    def get_fitted_params(self):
        return [1.5]


class SymbolicArtifactSchemaTest(unittest.TestCase):
    def test_instantiate_expression_parenthesizes_negative_values(self):
        instantiated = instantiate_expression(
            "c0**2 + 1/c1",
            parameter_symbols=["c0", "c1"],
            parameter_values=[-2.0, -4.0],
        )
        self.assertEqual(instantiated, "(-2.0)**2 + 1/(-4.0)")

    def test_infer_raw_equation_kind(self):
        self.assertEqual(infer_raw_equation_kind("x0 + x1"), "plain_expression")
        self.assertEqual(
            infer_raw_equation_kind("def equation(x0, params):\n    return x0\n"),
            "python_function",
        )
        self.assertEqual(infer_raw_equation_kind("add(x0, x1)"), "prefix_expression")

    def test_extract_return_expression_from_python_function(self):
        expr = extract_return_expression_from_python_function(
            "def equation(x0, x1, params):\n    return params[0] + x0 * x1\n"
        )
        self.assertEqual(expr, "params[0] + x0 * x1")

    def test_build_canonical_symbolic_program_for_plain_expression(self):
        artifact = build_canonical_symbolic_program(
            tool_name="pysr",
            raw_equation="x0 + 2*x1",
            expected_n_features=2,
            normalization_mode="direct",
        )
        self.assertEqual(artifact["version"], CSP_VERSION)
        self.assertEqual(artifact["tool_name"], "pysr")
        self.assertEqual(artifact["raw_equation_kind"], "plain_expression")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["instantiated_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["variables"], ["x0", "x1"])
        self.assertTrue(artifact["artifact_valid"])

    def test_build_canonical_symbolic_program_marks_variable_mismatch(self):
        artifact = build_canonical_symbolic_program(
            tool_name="tpsr",
            raw_equation="x0 + x2",
            expected_n_features=2,
            normalization_mode="direct",
        )
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_base_wrapper_default_export(self):
        wrapper = _DummyWrapper()
        artifact = wrapper.export_canonical_symbolic_program()
        validated = validate_canonical_symbolic_program(artifact)
        self.assertEqual(validated["version"], CSP_VERSION)
        self.assertEqual(validated["raw_equation_kind"], "python_function")
        self.assertEqual(validated["return_expression_source"], "params[0] + x0 * x1")
        self.assertEqual(validated["parameter_values"], [1.5])
        self.assertEqual(validated["parameter_symbols"], ["c0"])
        self.assertEqual(validated["instantiated_expression"], "1.5 + x0 * x1")


if __name__ == "__main__":
    unittest.main()
