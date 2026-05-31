import unittest

from scientific_intelligent_modelling.benchmarks.artifact_schema import CSP_VERSION
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_build_canonical_artifact,
    safe_export_canonical_artifact,
)


class _FakeRegressor:
    def export_canonical_symbolic_program(self):
        return {
            "version": CSP_VERSION,
            "tool_name": "fake",
            "raw_equation": "x0 + 1",
            "raw_equation_kind": "plain_expression",
            "python_function_source": None,
            "return_expression_source": "x0 + 1",
            "normalized_expression": "x0 + 1",
            "instantiated_expression": "x0 + 1",
            "variables": ["x0"],
            "parameter_symbols": [],
            "parameter_values": None,
            "expected_n_features": 1,
            "operator_set": ["add"],
            "ast_node_count": 3,
            "tree_depth": 2,
            "normalization_mode": "test",
            "normalization_notes": [],
            "artifact_valid": True,
            "validation_errors": [],
            "fidelity_check": {},
        }


class ResultArtifactsTest(unittest.TestCase):
    def test_safe_export_canonical_artifact(self):
        artifact, error = safe_export_canonical_artifact(_FakeRegressor())
        self.assertIsNone(error)
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact["version"], CSP_VERSION)
        self.assertEqual(artifact["normalized_expression"], "x0 + 1")
        self.assertEqual(artifact["instantiated_expression"], "x0 + 1")

    def test_safe_build_canonical_artifact_for_pysr(self):
        artifact, error = safe_build_canonical_artifact(
            tool_name="pysr",
            equation="x0 + 2*x1",
            expected_n_features=2,
        )
        self.assertIsNone(error)
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["instantiated_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["variables"], ["x0", "x1"])

    def test_safe_build_canonical_artifact_for_symbolfit(self):
        artifact, error = safe_build_canonical_artifact(
            tool_name="symbolfit",
            equation="1.5*x0 + sin(x1)",
            expected_n_features=2,
        )
        self.assertIsNone(error)
        self.assertEqual(artifact["tool_name"], "symbolfit")
        self.assertEqual(artifact["normalized_expression"], "1.5*x0 + sin(x1)")
        self.assertEqual(artifact["variables"], ["x0", "x1"])

    def test_safe_build_canonical_artifact_for_fepysr(self):
        artifact, error = safe_build_canonical_artifact(
            tool_name="fepysr",
            equation="X0 + cos(X1)",
            expected_n_features=2,
        )

        self.assertIsNone(error)
        self.assertEqual(artifact["tool_name"], "fepysr")
        self.assertEqual(artifact["normalized_expression"], "x0 + cos(x1)")
        self.assertEqual(artifact["variables"], ["x0", "x1"])

    def test_safe_build_canonical_artifact_for_llmsr_function(self):
        raw = (
            "def equation(x0, x1, params):\n"
            "    return params[0] * x0 + params[1] * x1 + params[2]\n"
        )
        artifact, error = safe_build_canonical_artifact(
            tool_name="llmsr",
            equation=raw,
            expected_n_features=2,
            parameter_values=[1.0, 2.0, 3.0],
        )
        self.assertIsNone(error)
        self.assertEqual(artifact["normalized_expression"], "c0*x0 + c1*x1 + c2")
        self.assertEqual(artifact["instantiated_expression"], "1.0*x0 + 2.0*x1 + 3.0")
        self.assertEqual(artifact["parameter_values"], [1.0, 2.0, 3.0])

    def test_safe_build_canonical_artifact_for_unknown_tool(self):
        artifact, error = safe_build_canonical_artifact(
            tool_name="unknown_tool",
            equation="x0 + 1",
        )
        self.assertIsNone(artifact)
        self.assertIn("暂不支持的工具名", error)


if __name__ == "__main__":
    unittest.main()
