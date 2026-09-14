import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scientific_intelligent_modelling.benchmarks.normalizers import (
    normalize_drsr_artifact,
    normalize_dso_artifact,
    normalize_e2esr_artifact,
    normalize_gplearn_artifact,
    normalize_imcts_artifact,
    normalize_llmsr_artifact,
    normalize_operon_artifact,
    normalize_pysr_artifact,
    normalize_qlattice_artifact,
    normalize_tpsr_artifact,
)


if "pandas" not in sys.modules:
    sys.modules["pandas"] = types.ModuleType("pandas")
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor
from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper import E2ESRRegressor
from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper import iMCTSRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor
from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import _infer_n_features_from_function_signature
from scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper import OperonRegressor
from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor
from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import QLatticeRegressor
from scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper import TPSRRegressor


class SymbolicNormalizersTest(unittest.TestCase):
    def test_normalize_pysr_artifact(self):
        artifact = normalize_pysr_artifact("x0 + 2*x1")
        self.assertEqual(artifact["tool_name"], "pysr")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["instantiated_expression"], "x0 + 2*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_pysr_square_cube_artifact(self):
        artifact = normalize_pysr_artifact("cube(x0) + square(x1)", expected_n_features=2)
        self.assertEqual(artifact["normalized_expression"], "x0**3 + x1**2")
        self.assertEqual(artifact["operator_set"], ["add", "pow"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_qlattice_artifact(self):
        artifact = normalize_qlattice_artifact("x0 + 2*x1")
        self.assertEqual(artifact["tool_name"], "QLattice")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_qlattice_keeps_zero_based_output_when_x0_is_absent(self):
        """QLattice 的 x1 是原生第二列，不能因 x0 缺失而平移。"""
        artifact = normalize_qlattice_artifact("x1", expected_n_features=2)
        self.assertEqual(artifact["normalized_expression"], "x1")
        self.assertEqual(artifact["variables"], ["x1"])
        self.assertIn("return x1", artifact["python_function_source"])

    def test_normalize_qlattice_keeps_explicit_zero_based_output(self):
        artifact = normalize_qlattice_artifact("x0 + 2*x1", expected_n_features=2)
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["variables"], ["x0", "x1"])

    def test_normalize_gplearn_artifact(self):
        artifact = normalize_gplearn_artifact("add(X0, mul(X1, X1))")
        self.assertEqual(artifact["tool_name"], "gplearn")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_gplearn_keeps_zero_based_feature_indices(self):
        artifact = normalize_gplearn_artifact("log(X3)", expected_n_features=4)
        self.assertEqual(artifact["normalized_expression"], "log(x3)")
        self.assertEqual(artifact["variables"], ["x3"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_gplearn_deep_prefix_fallback(self):
        expr = "X0"
        for _ in range(250):
            expr = f"add({expr}, 1.0)"
        artifact = normalize_gplearn_artifact(expr, expected_n_features=1)
        self.assertEqual(artifact["tool_name"], "gplearn")
        self.assertEqual(artifact["normalization_mode"], "gplearn_prefix_unparsed")
        self.assertEqual(artifact["variables"], ["x0"])
        self.assertFalse(artifact["sympy_parse_ok"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_imcts_artifact(self):
        artifact = normalize_imcts_artifact(
            "lambda x: np.log(x[0] + 1) + np.log(x[0]**2 + 1)",
            expected_n_features=1,
        )
        self.assertEqual(artifact["tool_name"], "iMCTS")
        self.assertEqual(artifact["normalized_expression"], "log(x0 + 1) + log(x0**2 + 1)")
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_imcts_keeps_zero_based_x_without_x0(self):
        artifact = normalize_imcts_artifact("x2 / x3", expected_n_features=4)
        self.assertEqual(artifact["normalized_expression"], "x2/x3")
        self.assertEqual(artifact["variables"], ["x2", "x3"])

    def test_normalize_imcts_rejects_force_expanded_log_without_vector_source(self):
        with self.assertRaisesRegex(ValueError, "expand_log"):
            normalize_imcts_artifact(
                "exp(x3/(x3 + 1))*exp(-2*log(x0)/(x3 + 1))",
                expected_n_features=4,
            )

    def test_normalize_e2esr_artifact(self):
        artifact = normalize_e2esr_artifact("x_0 + x_1**2")
        self.assertEqual(artifact["tool_name"], "e2esr")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_tpsr_artifact(self):
        artifact = normalize_tpsr_artifact("x_0 + x_1**2")
        self.assertEqual(artifact["tool_name"], "tpsr")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_tpsr_artifact_marks_variable_mismatch(self):
        artifact = normalize_tpsr_artifact("x_0 + x_2**2", expected_n_features=2)
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_normalize_operon_artifact(self):
        artifact = normalize_operon_artifact("X1 + X2^2")
        self.assertEqual(artifact["tool_name"], "pyoperon")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_llmsr_artifact(self):
        raw = (
            "def equation(x0, x1, params):\n"
            "    return params[0] + params[1] * x0 + params[2] * x1\n"
        )
        artifact = normalize_llmsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["tool_name"], "llmsr")
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertEqual(artifact["instantiated_expression"], "1.0 + 2.0*x0 + 3.0*x1")
        self.assertEqual(artifact["parameter_symbols"], ["c0", "c1", "c2"])
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_llmsr_maps_signature_order_without_heuristic_shift(self):
        raw = (
            "def equation(alpha, beta, params):\n"
            "    return alpha + 2 * beta\n"
        )
        artifact = normalize_llmsr_artifact(raw, expected_n_features=2)
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["variables"], ["x0", "x1"])

        zero_based = normalize_llmsr_artifact(
            "def equation(x0, x1, params):\n    return x1\n",
            expected_n_features=2,
        )
        self.assertEqual(zero_based["normalized_expression"], "x1")

    def test_normalize_llmsr_resolves_negative_parameter_index(self):
        raw = "def equation(x0, params):\n    return params[-2] * x0 + params[-1]\n"
        artifact = normalize_llmsr_artifact(
            raw,
            parameter_values=[1.5, -2.0, 3.0],
            expected_n_features=1,
        )
        self.assertEqual(artifact["normalized_expression"], "c1*x0 + c2")
        self.assertEqual(artifact["executable_expression"], "(-2.0) * x0 + 3.0")

    def test_normalize_llmsr_rejects_out_of_range_negative_parameter_index(self):
        raw = "def equation(x0, params):\n    return params[-4] * x0\n"
        with self.assertRaisesRegex(ValueError, "参数下标越界"):
            normalize_llmsr_artifact(
                raw,
                parameter_values=[1.0, 2.0, 3.0],
                expected_n_features=1,
            )

    def test_normalize_llmsr_keeps_executable_when_display_sympy_cannot_parse(self):
        raw = (
            "def equation(x0, x1, params):\n"
            "    return params[0] * x0 / np.linalg.norm(x1)\n"
        )
        artifact = normalize_llmsr_artifact(
            raw,
            parameter_values=[2.0],
            expected_n_features=2,
        )
        self.assertFalse(artifact["sympy_parse_ok"])
        self.assertEqual(artifact["variables"], ["x0", "x1"])
        self.assertEqual(artifact["executable_expression"], "2.0 * x0 / np.linalg.norm(x1)")
        self.assertTrue(artifact["artifact_valid"])

    def test_infer_llmsr_n_features_from_signature(self):
        raw = (
            "def equation(x0, x1, params):\n"
            "    return params[0] + params[1] * x0 + params[2] * x1\n"
        )
        self.assertEqual(_infer_n_features_from_function_signature(raw), 2)

    def test_normalize_drsr_artifact(self):
        raw = (
            "def equation(col0, col1, params):\n"
            "    return params[0] + params[1] * col0 + params[2] * col1\n"
        )
        artifact = normalize_drsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["tool_name"], "drsr")
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertEqual(artifact["instantiated_expression"], "1.0 + 2.0*x0 + 3.0*x1")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_drsr_maps_numpy_minmax_to_sympy_elementwise_ops(self):
        raw = (
            "def equation(col0, col1, col2, params):\n"
            "    return params[0] * np.maximum(col0, 1e-6) "
            "+ params[1] * np.minimum(col1, col2)\n"
        )
        artifact = normalize_drsr_artifact(
            raw,
            parameter_values=[2.0, 3.0],
            expected_n_features=3,
        )
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertIn("Max(1.0e-6, x0)", artifact["normalized_expression"])
        self.assertIn("Min(x1, x2)", artifact["normalized_expression"])
        self.assertIn("max", artifact["operator_set"])
        self.assertIn("min", artifact["operator_set"])

    def test_normalize_drsr_legacy_xyv_artifact(self):
        raw = (
            "def equation(col0, col1, params):\n"
            "    return params[0] * x + params[1] * v + params[2]\n"
        )
        artifact = normalize_drsr_artifact(raw, parameter_values=[1.0, 2.0, 3.0])
        self.assertEqual(artifact["normalized_expression"], "c0*x0 + c1*x1 + c2")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_normalize_drsr_metadata_feature_names_from_signature(self):
        raw = (
            "def equation(r, m1, m2, params):\n"
            "    return params[0] + params[1] * r + params[2] * m1 + params[3] * m2\n"
        )
        artifact = normalize_drsr_artifact(
            raw,
            parameter_values=[1.0, 2.0, 3.0, 4.0],
            expected_n_features=3,
        )
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1 + c3*x2")
        self.assertEqual(artifact["variables"], ["x0", "x1", "x2"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_drsr_feynman_signature_names(self):
        raw = (
            "def equation(kappa, t1, t2, a, d, params):\n"
            "    return params[0] + params[1] * kappa + params[2] * t1 + params[3] * t2 + params[4] * a + params[5] * d\n"
        )
        artifact = normalize_drsr_artifact(
            raw,
            parameter_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            expected_n_features=5,
        )
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1 + c3*x2 + c4*x3 + c5*x4")
        self.assertEqual(artifact["variables"], ["x0", "x1", "x2", "x3", "x4"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_drsr_annotated_signature_names(self):
        raw = (
            "def equation(t: np.ndarray, a: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
            "    return params[0] + params[1] * t + params[2] * a\n"
        )
        artifact = normalize_drsr_artifact(
            raw,
            parameter_values=[0.1, 0.2, 0.3],
            expected_n_features=2,
        )
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertEqual(artifact["instantiated_expression"], "0.1 + 0.2*x0 + 0.3*x1")
        self.assertIn("function_arg_map:a->x1,t->x0", artifact["normalization_notes"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_dso_artifact(self):
        artifact = normalize_dso_artifact("x1 + x2**2", expected_n_features=2)
        self.assertEqual(artifact["tool_name"], "dso")
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertTrue(artifact["artifact_valid"])

    def test_wrapper_export_pysr(self):
        model = PySRRegressor()
        model.model = object()
        model.get_optimal_equation = lambda: "x0 + 2*x1"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["tool_name"], "pysr")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_wrapper_export_qlattice(self):
        model = QLatticeRegressor()
        model.model = True
        model.get_optimal_equation = lambda: "x0 + 2*x1"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["tool_name"], "QLattice")
        self.assertTrue(artifact["sympy_parse_ok"])

    def test_wrapper_export_gplearn(self):
        model = GPLearnRegressor()
        model.model = object()
        model.get_optimal_equation = lambda: "add(X0, mul(X1, X1))"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_imcts(self):
        model = iMCTSRegressor()
        model._best_expr_simplified = "np.log(x[0] + 1) + np.log(x[0]**2 + 1)"
        model._n_features = 1
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "log(x0 + 1) + log(x0**2 + 1)")
        self.assertTrue(artifact["artifact_valid"])

    def test_wrapper_export_imcts_prefers_native_vector_expression(self):
        model = iMCTSRegressor()
        model._best_expr_simplified = "exp(x3/(x3 + 1))*exp(-2*log(x0)/(x3 + 1))"
        model._best_expr_vector = "np.exp(x[3] / (x[3] + 1) - 2 * np.log(x[0]) / (x[3] + 1))"
        model._n_features = 4

        self.assertEqual(model.get_optimal_equation(), model._best_expr_vector)
        artifact = model.export_canonical_symbolic_program()

        self.assertEqual(artifact["raw_equation"], model._best_expr_vector)
        self.assertIn("x3", artifact["normalized_expression"])
        self.assertIn("x0", artifact["normalized_expression"])
        self.assertTrue(artifact["artifact_valid"])

    def test_imcts_unranked_final_state_does_not_erase_internal_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = iMCTSRegressor(exp_path=tmp, exp_name="imcts_state")
            model._fit_started_at = 100.0
            with patch(
                "scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper.time.time",
                return_value=110.0,
            ):
                model._write_progress_state(
                    equation="np.exp(x[0])",
                    score=0.9,
                    evaluations=42,
                )
            model._write_progress_state(
                equation="np.exp(x[1])",
                score=None,
                evaluations=84,
            )

            payload = json.loads(
                (Path(tmp) / "imcts_state" / ".imcts_current_best.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["equation"], "np.exp(x[0])")
            self.assertEqual(payload["score"], 0.9)
            self.assertEqual(payload["evaluations"], 42)
            self.assertEqual(payload["first_discovered_minute"], 1)
            self.assertEqual(payload["first_discovered_elapsed_seconds"], 10.0)

    def test_imcts_progress_state_preserves_first_discovery_until_new_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = iMCTSRegressor(exp_path=tmp, exp_name="imcts_discovery")
            model._fit_started_at = 100.0
            state_path = Path(tmp) / "imcts_discovery" / ".imcts_current_best.json"

            with patch(
                "scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper.time.time",
                side_effect=[110.0, 170.0, 230.0],
            ):
                model._write_progress_state(
                    equation="np.exp(x[0])", score=0.5, evaluations=10
                )
                first = json.loads(state_path.read_text(encoding="utf-8"))
                model._write_progress_state(
                    equation="np.exp(x[0])", score=0.5, evaluations=20
                )
                carried = json.loads(state_path.read_text(encoding="utf-8"))
                model._write_progress_state(
                    equation="np.exp(x[1])", score=0.8, evaluations=30
                )
                improved = json.loads(state_path.read_text(encoding="utf-8"))

            self.assertEqual(first["first_discovered_minute"], 1)
            self.assertEqual(carried["first_discovered_minute"], 1)
            self.assertEqual(carried["evaluations"], 10)
            self.assertEqual(improved["first_discovered_minute"], 3)
            self.assertEqual(improved["first_discovered_elapsed_seconds"], 130.0)
            self.assertEqual(improved["score"], 0.8)

    def test_imcts_unranked_candidate_cannot_create_internal_best_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = iMCTSRegressor(exp_path=tmp, exp_name="imcts_unranked")

            model._write_progress_state(
                equation="np.exp(x[0])",
                score=None,
                evaluations=42,
            )

            self.assertFalse(
                (Path(tmp) / "imcts_unranked" / ".imcts_current_best.json").exists()
            )

    def test_imcts_native_reward_controls_incumbent_and_ties_keep_earliest(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = iMCTSRegressor(exp_path=tmp, exp_name="imcts_native")
            model._fit_started_at = 100.0
            state_path = Path(tmp) / "imcts_native" / ".imcts_current_best.json"
            with patch(
                "scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper.time.time",
                side_effect=[110.0, 120.0, 130.0],
            ):
                model._write_progress_state(equation="x[0]", score=0.4, evaluations=10)
                # 即使外部预测质量下降，native reward 变大仍必须更新。
                model._write_progress_state(equation="x[1]", score=0.8, evaluations=20)
                model._write_progress_state(equation="x[2]", score=0.8, evaluations=30)
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["equation"], "x[1]")
            self.assertEqual(payload["expression_vector"], "x[1]")
            self.assertEqual(payload["internal_objective"], "native_reward")
            self.assertEqual(payload["objective_direction"], "max")
            self.assertEqual(payload["evaluations"], 20)
            self.assertEqual(payload["source_timestamp_unix"], 120.0)
            self.assertEqual(len(payload["candidate_sha256"]), 64)

    def test_e2esr_native_model_score_controls_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = E2ESRRegressor.__new__(E2ESRRegressor)
            model._progress_state_path = str(Path(tmp) / ".e2esr_current_best.json")
            model._fit_started_at = 100.0
            with patch(
                "scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper.time.time",
                side_effect=[110.0, 120.0, 130.0],
            ):
                model._write_progress_state(
                    equation="x_0", native_model_score=-2.0, bag_index=0, candidate_rank=0
                )
                model._write_progress_state(
                    equation="x_1", native_model_score=-1.0, bag_index=1, candidate_rank=1
                )
                model._write_progress_state(
                    equation="x_2", native_model_score=-1.0, bag_index=2, candidate_rank=0
                )
            payload = json.loads(Path(model._progress_state_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["equation"], "x_1")
            self.assertEqual(payload["native_model_score"], -1.0)
            self.assertEqual(payload["objective_direction"], "max")
            self.assertEqual(payload["bag_index"], 1)
            self.assertEqual(payload["candidate_rank"], 1)

    def test_e2esr_invalid_native_objective_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = E2ESRRegressor.__new__(E2ESRRegressor)
            model._progress_state_path = str(Path(tmp) / ".e2esr_current_best.json")
            model._fit_started_at = 100.0
            model._write_progress_state(
                equation="x_0", native_model_score=float("nan"), bag_index=0, candidate_rank=0
            )
            self.assertFalse(Path(model._progress_state_path).exists())

    def test_wrapper_export_e2esr(self):
        model = E2ESRRegressor.__new__(E2ESRRegressor)
        model.best_tree = object()
        model.get_optimal_equation = lambda: "x_0 + x_1**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_tpsr(self):
        model = TPSRRegressor()
        model.best_tree = "x_0 + x_1**2"
        model._n_features = 2
        model.get_optimal_equation = lambda: "x_0 + x_1**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_tpsr_marks_variable_mismatch(self):
        model = TPSRRegressor()
        model.best_tree = "x_0 + x_2**2"
        model._n_features = 2
        model.get_optimal_equation = lambda: "x_0 + x_2**2"
        artifact = model.export_canonical_symbolic_program()
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_wrapper_export_operon(self):
        model = OperonRegressor()
        model.best_model_str = "X1 + X2^2"
        model.model = None
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")

    def test_wrapper_export_llmsr(self):
        model = LLMSRRegressor()
        model._load_best_sample = lambda: {
            "function": (
                "def equation(x0, x1, params):\n"
                "    return params[0] + params[1] * x0 + params[2] * x1\n"
            ),
            "params": [1.0, 2.0, 3.0],
        }
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")
        self.assertTrue(artifact["artifact_valid"])

    def test_wrapper_export_llmsr_marks_variable_mismatch(self):
        model = LLMSRRegressor()
        model._n_features = 2
        model._load_best_sample = lambda: {
            "function": (
                "def equation(x0, x1, params):\n"
                "    return params[0] + params[1] * x0 + params[2] * x2\n"
            ),
            "params": [1.0, 2.0, 3.0],
        }
        artifact = model.export_canonical_symbolic_program()
        self.assertFalse(artifact["artifact_valid"])
        self.assertTrue(artifact["validation_errors"])

    def test_wrapper_export_drsr(self):
        model = DRSRRegressor()
        model.get_optimal_equation = lambda: (
            "def equation(col0, col1, params):\n"
            "    return params[0] + params[1] * col0 + params[2] * col1\n"
        )
        model.get_fitted_params = lambda: [1.0, 2.0, 3.0]
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "c0 + c1*x0 + c2*x1")

    def test_wrapper_export_dso(self):
        model = DSORegressor()
        model._dso_expression = "x1 + x2**2"
        model._dso_n_features = 2
        artifact = model.export_canonical_symbolic_program()
        self.assertEqual(artifact["normalized_expression"], "x0 + x1**2")
        self.assertTrue(artifact["artifact_valid"])


if __name__ == "__main__":
    unittest.main()
