import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.benchmarks import runner
from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError


class _FakeRegressor:
    def __init__(self, tool_name, problem_name=None, seed=1314, **kwargs):
        self.tool_name = tool_name
        self.problem_name = problem_name
        self.seed = seed
        self.params = dict(kwargs)
        exp_path = Path(self.params["exp_path"])
        exp_name = self.params["exp_name"]
        self.experiment_dir = str((exp_path / exp_name).resolve())
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, X, y):
        self._n_features = X.shape[1]
        return self

    def predict(self, X):
        return X[:, 0]

    def get_optimal_equation(self):
        return "x0"

    def get_total_equations(self):
        return ["x0", "x0 + 1"]

    def export_canonical_symbolic_program(self):
        return {
            "version": "csp_v1",
            "tool_name": self.tool_name,
            "raw_equation": "x0",
            "raw_equation_kind": "plain_expression",
            "python_function_source": "def equation(x0, params):\n    return x0\n",
            "return_expression_source": "x0",
            "normalized_expression": "x0",
            "instantiated_expression": "x0",
            "variables": ["x0"],
            "parameter_symbols": [],
            "parameter_values": None,
            "expected_n_features": self._n_features,
            "operator_set": [],
            "ast_node_count": 1,
            "tree_depth": 1,
            "normalization_mode": "fake",
            "normalization_notes": [],
            "artifact_valid": True,
            "validation_errors": [],
            "fidelity_check": {},
        }


class _RecordingRegressor(_FakeRegressor):
    last_fit_y = None
    last_params = None

    def __init__(self, tool_name, problem_name=None, seed=1314, **kwargs):
        type(self).last_params = dict(kwargs)
        super().__init__(tool_name, problem_name=problem_name, seed=seed, **kwargs)

    def fit(self, X, y):
        type(self).last_fit_y = np.asarray(y, dtype=float).reshape(-1).copy()
        return super().fit(X, y)


class _TimeoutFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise TimeoutError("fit timeout")


class _TimeoutRecoverableFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise TimeoutError("fit timeout")


class _RecoverableErrorFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise TypeError("Cannot convert complex to int")


class _NoValidOutputFakeRegressor(_FakeRegressor):
    def fit(self, X, y):
        raise NoValidOutputError("fake algorithm produced no model")


class BenchmarkRunnerTest(unittest.TestCase):
    def test_regression_metrics_penalize_nonfinite_predictions_with_finite_nmse(self):
        metrics = regression_metrics(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, np.nan, np.inf]),
            acc_threshold=0.1,
        )

        self.assertIsNotNone(metrics["nmse"])
        self.assertTrue(np.isfinite(metrics["nmse"]))
        self.assertIsNotNone(metrics["rmse"])
        self.assertTrue(np.isfinite(metrics["rmse"]))

    def test_predict_from_canonical_artifact_reports_uninstantiated_params(self):
        artifact = {
            "normalized_expression": "c0 + x0",
            "instantiated_expression": "c0 + x0",
        }
        with self.assertRaisesRegex(ValueError, "非标准变量: c0"):
            runner._predict_from_canonical_artifact(artifact, np.asarray([[1.0], [2.0]]))

    def test_predict_from_canonical_artifact_supports_pysr_square_cube(self):
        artifact = {
            "tool_name": "pysr",
            "instantiated_expression": "cube(x0) + square(x1)",
        }
        pred = runner._predict_from_canonical_artifact(artifact, np.asarray([[2.0, 3.0], [1.0, 4.0]]))
        np.testing.assert_allclose(pred, np.asarray([17.0, 17.0]))

    def test_predict_from_canonical_artifact_uses_real_numpy_cbrt(self):
        artifact = {
            "tool_name": "drsr",
            "instantiated_expression": "cbrt(x0)",
        }
        pred = runner._predict_from_canonical_artifact(
            artifact,
            np.asarray([[-8.0], [-1.0], [8.0]]),
        )
        np.testing.assert_allclose(pred, np.asarray([-2.0, -1.0, 2.0]))

    def test_predict_from_canonical_artifact_prefers_executable_python_tree(self):
        raw = (
            "def equation(x0, params):\n"
            "    return params[0] + params[1] / "
            "(1 + np.exp(-(x0 - params[2]) / params[3]))\n"
        )
        from scientific_intelligent_modelling.benchmarks.normalizers import normalize_llmsr_artifact

        params = [-6.0755525116527105, 0.6781758534940286, 0.5430817954654256, 0.0002585712221404973]
        artifact = normalize_llmsr_artifact(
            raw,
            parameter_values=params,
            expected_n_features=1,
        )
        X = np.asarray([[-10.0], [0.0], [10.0]])

        with np.errstate(all="ignore"):
            expected = params[0] + params[1] / (
                1 + np.exp(-(X[:, 0] - params[2]) / params[3])
            )
        pred = runner._predict_from_canonical_artifact(artifact, X)

        self.assertTrue(np.all(np.isfinite(pred)))
        np.testing.assert_allclose(pred, expected)

    def test_predict_from_executable_expression_preserves_negative_power_base(self):
        artifact = {
            "tool_name": "llmsr",
            "executable_expression": "(-2.0) ** 2 + x0",
            "instantiated_expression": "-2.0**2 + x0",
        }
        pred = runner._predict_from_canonical_artifact(artifact, np.asarray([[1.0], [2.0]]))
        np.testing.assert_allclose(pred, np.asarray([5.0, 6.0]))

    def test_predict_from_executable_expression_rejects_python_capabilities(self):
        artifact = {
            "tool_name": "llmsr",
            "executable_expression": "__import__('os').system('true')",
        }
        with self.assertRaisesRegex(ValueError, "不允许|非标准"):
            runner._predict_from_canonical_artifact(artifact, np.asarray([[1.0]]))

    def test_predict_from_executable_expression_allows_constant_feature_subscript(self):
        artifact = {
            "tool_name": "llmsr",
            "executable_expression": "x0 + x1[0]",
        }
        pred = runner._predict_from_canonical_artifact(
            artifact,
            np.asarray([[1.0, 10.0], [2.0, 20.0]]),
        )
        np.testing.assert_allclose(pred, np.asarray([11.0, 12.0]))

    def test_predict_from_executable_expression_allows_numpy_gradient(self):
        artifact = {
            "tool_name": "llmsr",
            "executable_expression": "2.0 * x0 + np.gradient(x1)",
        }
        X = np.asarray(
            [
                [1.0, 1.0],
                [2.0, 4.0],
                [3.0, 9.0],
                [4.0, 16.0],
            ]
        )

        pred = runner._predict_from_canonical_artifact(artifact, X)

        expected = 2.0 * X[:, 0] + np.gradient(X[:, 1])
        np.testing.assert_allclose(pred, expected)

    def test_drsr_numpy_maximum_replay_is_elementwise(self):
        from scientific_intelligent_modelling.benchmarks.normalizers import normalize_drsr_artifact

        raw = (
            "def equation(col0, col1, col2, params):\n"
            "    return params[0] + params[1] * col0 / "
            "(col1 * np.maximum(col2, 1e-6)) + params[2] * col1 * np.exp(params[3] * col2)\n"
        )
        params = [-8.351879365020947e-05, 1.9999565766692573, 7.676934321751336, -8.465500121281929]
        artifact = normalize_drsr_artifact(raw, parameter_values=params, expected_n_features=3)
        X = np.asarray([[2.0, 4.0, -1.0], [2.0, 4.0, 2.0]])
        expected = (
            params[0]
            + params[1] * X[:, 0] / (X[:, 1] * np.maximum(X[:, 2], 1e-6))
            + params[2] * X[:, 1] * np.exp(params[3] * X[:, 2])
        )
        pred = runner._predict_from_canonical_artifact(artifact, X)
        np.testing.assert_allclose(pred, expected)

    def test_drsr_boolean_indicator_replay_casts_to_zero_one(self):
        from scientific_intelligent_modelling.benchmarks.normalizers import normalize_drsr_artifact

        raw = (
            "def equation(col0, col1, params):\n"
            "    return params[0] + params[7] * (col1 >= 1e-6)\n"
        )
        params = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5]
        artifact = normalize_drsr_artifact(raw, parameter_values=params, expected_n_features=2)
        X = np.asarray([[0.0, -1.0], [0.0, 1e-6], [0.0, 2.0]])
        pred = runner._predict_from_canonical_artifact(artifact, X)
        np.testing.assert_allclose(pred, np.asarray([0.25, -0.25, -0.25]))
        self.assertFalse(artifact["sympy_parse_ok"])
        self.assertTrue(any("sympy_display_parse_failed" in note for note in artifact["normalization_notes"]))

    def test_predict_from_gplearn_artifact_uses_protected_semantics(self):
        artifact = {
            "tool_name": "gplearn",
            "raw_equation": "add(log(log(div(-0.593, X0))), sqrt(X1))",
            "raw_equation_kind": "prefix_expression",
            "normalized_expression": "log(log(-0.593/x0)) + sqrt(x1)",
            "instantiated_expression": "log(log(-0.593/x0)) + sqrt(x1)",
        }
        X = np.asarray([[1.0, -4.0], [0.0, 9.0], [-2.0, 16.0]])

        pred = runner._predict_from_canonical_artifact(artifact, X)

        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertAlmostEqual(pred[1], 3.0, places=10)

    def test_predict_from_canonical_artifact_vectorizes_sympy_min_max(self):
        artifact = {
            "tool_name": "ragsr",
            "instantiated_expression": "Min(0.5, x0**2) + Max(x1, 1.0)",
        }
        X = np.asarray([[0.25, 0.0], [2.0, 3.0]])

        pred = runner._predict_from_canonical_artifact(artifact, X)

        np.testing.assert_allclose(pred, np.asarray([1.0625, 3.5]))

    def test_run_benchmark_task_writes_outer_and_experiment_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: demo dataset
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _FakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="gplearn",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["tool"], "gplearn")
            self.assertEqual(result["dataset"], dataset_dir.name)
            self.assertEqual(result["equation"], "x0")
            self.assertEqual(result["equation_count"], 2)
            self.assertIsNotNone(result["train"])
            self.assertIsNotNone(result["valid"])
            self.assertIsNotNone(result["id_test"])
            self.assertIsNotNone(result["ood_test"])
            self.assertAlmostEqual(result["train"]["nmse"], 0.0, places=10)
            self.assertTrue(result["experiment_dir"])

            experiment_result = Path(result["experiment_dir"]) / "result.json"
            self.assertTrue(experiment_result.exists())
            self.assertEqual(
                json.loads(experiment_result.read_text(encoding="utf-8")),
                result,
            )

    def test_run_benchmark_task_uses_unique_task_label_for_same_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dirs = []
            for parent, values in {"hard": "1,1\n2,2\n", "hard_dummy": "3,3\n4,4\n"}.items():
                dataset_dir = root / parent / "same-name"
                dataset_dir.mkdir(parents=True)
                (dataset_dir / "metadata.yaml").write_text(
                    """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                    encoding="utf-8",
                )
                (dataset_dir / "train.csv").write_text(f"x0,y\n{values}", encoding="utf-8")
                dataset_dirs.append(dataset_dir)

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _FakeRegressor
            try:
                result_path_1 = runner.run_benchmark_task(
                    tool_name="gplearn",
                    dataset_dir=dataset_dirs[0],
                    output_root=root / "bench_results",
                    seed=1314,
                    params_override={
                        "task_label": "g0001_same-name",
                        "task_global_index": 1,
                        "expected_dataset_dir": str(dataset_dirs[0]),
                    },
                )
                result_path_2 = runner.run_benchmark_task(
                    tool_name="gplearn",
                    dataset_dir=dataset_dirs[1],
                    output_root=root / "bench_results",
                    seed=1314,
                    params_override={
                        "task_label": "g0002_same-name",
                        "task_global_index": 2,
                        "expected_dataset_dir": str(dataset_dirs[1]),
                    },
                )
            finally:
                runner.SymbolicRegressor = original_cls

            self.assertNotEqual(result_path_1, result_path_2)
            result_1 = json.loads(result_path_1.read_text(encoding="utf-8"))
            result_2 = json.loads(result_path_2.read_text(encoding="utf-8"))
            self.assertEqual(result_path_1.parent.name, "g0001_same-name")
            self.assertEqual(result_path_2.parent.name, "g0002_same-name")
            self.assertEqual(result_1["task_global_index"], 1)
            self.assertEqual(result_2["task_global_index"], 2)
            self.assertTrue(result_1["dataset_identity_check"]["match"])
            self.assertTrue(result_2["dataset_identity_check"]["match"])
            self.assertNotIn("task_label", result_1["params"])
            self.assertNotIn("task_global_index", result_1["params"])

    def test_run_benchmark_task_ignores_seed_in_params_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _FakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="iMCTS",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                    params_override={"seed": 2025},
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["seed"], 1314)
            self.assertNotIn("seed", result["params"])

    def test_run_benchmark_task_maps_timeout_to_timed_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _TimeoutFakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="pysr",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "timed_out")
            self.assertIn("TimeoutError", result["error"])
            self.assertTrue(result["budget_exhausted"])
            self.assertEqual(result["timeout_type"], "no_valid_output")
            self.assertFalse(result["recovered_from_timeout"])
            self.assertEqual(result["termination_reason"], "no_valid_output")
            self.assertTrue(result["experiment_dir"])

    def test_run_benchmark_task_maps_no_valid_output_to_done_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _NoValidOutputFakeRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="QLattice",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "no_valid_output")
            self.assertIsNone(result["error"])
            self.assertFalse(result["budget_exhausted"])
            self.assertEqual(result["timeout_type"], "no_valid_output")
            self.assertEqual(result["termination_reason"], "no_valid_output")
            self.assertIn("fake algorithm produced no model", result["no_valid_output_reason"])
            self.assertIsNone(result["equation"])
            self.assertIsNone(result["id_test"])

    def test_run_benchmark_task_recovers_timeout_result_from_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            original_extract = runner._extract_periodic_candidate
            runner.SymbolicRegressor = _TimeoutRecoverableFakeRegressor
            runner._extract_periodic_candidate = lambda tool, exp_dir: {"equation": "x0", "loss": 0.1}
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="pysr",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=1314,
                )
            finally:
                runner.SymbolicRegressor = original_cls
                runner._extract_periodic_candidate = original_extract

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["error"])
            self.assertTrue(result["budget_exhausted"])
            self.assertEqual(result["timeout_type"], "budget_exhausted_with_output")
            self.assertTrue(result["recovered_from_timeout"])
            self.assertEqual(result["termination_reason"], "budget_exhausted_with_output")
            self.assertIn("TimeoutError", result["raw_timeout_error"])
            self.assertEqual(result["equation"], "x0")
            self.assertIsNotNone(result["canonical_artifact"])
            self.assertIsNotNone(result["train"])
            self.assertIsNotNone(result["valid"])
            self.assertIsNotNone(result["id_test"])
            self.assertIsNotNone(result["ood_test"])
            self.assertEqual(result["train"]["nmse"], 0.0)
            self.assertEqual(result["equation_count"], 1)

    def test_run_benchmark_task_recovers_snapshot_capable_tool_after_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            artifact, artifact_error = runner.safe_build_canonical_artifact(
                tool_name="symbolfit",
                equation="x0",
                expected_n_features=1,
            )
            self.assertIsNone(artifact_error)
            recovered = {
                "equation": "x0",
                "equation_count": 1,
                "canonical_artifact": artifact,
                "canonical_artifact_error": None,
                "train_metrics": {
                    "rmse": 0.0,
                    "r2": 1.0,
                    "nmse": 0.0,
                    "acc_0_1": 1.0,
                },
                "valid_metrics": {
                    "rmse": 0.0,
                    "r2": 1.0,
                    "nmse": 0.0,
                    "acc_0_1": 1.0,
                },
                "id_metrics": {
                    "rmse": 0.0,
                    "r2": 1.0,
                    "nmse": 0.0,
                    "acc_0_1": 1.0,
                },
                "ood_metrics": {
                    "rmse": 0.0,
                    "r2": 1.0,
                    "nmse": 0.0,
                    "acc_0_1": 1.0,
                },
            }

            original_cls = runner.SymbolicRegressor
            original_recover = runner._recover_timeout_payload_from_candidate
            runner.SymbolicRegressor = _RecoverableErrorFakeRegressor
            runner._recover_timeout_payload_from_candidate = (
                lambda **kwargs: recovered
            )
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="symbolfit",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=520,
                )
            finally:
                runner.SymbolicRegressor = original_cls
                runner._recover_timeout_payload_from_candidate = (
                    original_recover
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["error"])
            self.assertFalse(result["budget_exhausted"])
            self.assertFalse(result["recovered_from_timeout"])
            self.assertTrue(result["recovered_from_error"])
            self.assertIn("TypeError", result["raw_execution_error"])
            self.assertEqual(
                result["termination_reason"],
                "recovered_after_error",
            )
            self.assertEqual(result["equation"], "x0")
            self.assertEqual(result["id_test"]["nmse"], 0.0)
            self.assertEqual(result["ood_test"]["nmse"], 0.0)

    def test_recover_timeout_falls_back_to_latest_finite_progress_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n",
                "valid.csv": "x0,y\n3,3\n",
                "id_test.csv": "x0,y\n4,4\n",
                "ood_test.csv": "x0,y\n5,5\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            exp_dir = root / "udsr_exp"
            progress_dir = exp_dir / "progress"
            progress_dir.mkdir(parents=True)
            artifact, artifact_error = runner.safe_build_canonical_artifact(
                tool_name="udsr",
                equation="x0",
                expected_n_features=1,
            )
            self.assertIsNone(artifact_error)
            snapshot = {
                "tool": "udsr",
                "status": "ok",
                "equation": "x0",
                "equation_count": 1,
                "canonical_artifact": artifact,
                "canonical_artifact_error": None,
                "valid": {"rmse": 0.0, "r2": 1.0, "nmse": 0.0, "acc_0_1": 1.0},
                "id_test": {"rmse": 0.0, "r2": 1.0, "nmse": 0.0, "acc_0_1": 1.0},
                "ood_test": {"rmse": 0.0, "r2": 1.0, "nmse": 0.0, "acc_0_1": 1.0},
            }
            (progress_dir / "minute_0002.json").write_text(
                json.dumps(snapshot),
                encoding="utf-8",
            )

            original_extract = runner._extract_periodic_candidate
            runner._extract_periodic_candidate = lambda tool, exp_dir: {"equation": "x9"}
            try:
                payload = runner._recover_timeout_payload_from_candidate(
                    tool_name="udsr",
                    dataset=dataset,
                    experiment_dir=exp_dir,
                )
            finally:
                runner._extract_periodic_candidate = original_extract

            self.assertIsNotNone(payload)
            self.assertEqual(payload["equation"], "x0")
            self.assertEqual(payload["train_metrics"]["nmse"], 0.0)
            self.assertEqual(payload["valid_metrics"]["nmse"], 0.0)
            self.assertEqual(payload["id_metrics"]["nmse"], 0.0)
            self.assertEqual(payload["ood_metrics"]["nmse"], 0.0)

    def test_extract_drsr_candidate_backfills_params_from_matching_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_dir = root / "drsr_exp"
            samples_dir = exp_dir / "samples"
            best_history_dir = exp_dir / "best_history"
            samples_dir.mkdir(parents=True)
            best_history_dir.mkdir(parents=True)
            function = (
                "def equation(beta, alpha, params):\n"
                "    return params[0] + params[1] * beta + params[2] * alpha\n"
            )
            (samples_dir / "top01_samples_0.json").write_text(
                json.dumps({"function": function, "score": 2.0}),
                encoding="utf-8",
            )
            (best_history_dir / "best_sample_0.json").write_text(
                json.dumps({"function": function, "score": 1.0, "params": [1.0, 2.0, 3.0]}),
                encoding="utf-8",
            )

            candidate = runner._extract_drsr_periodic_candidate(exp_dir)

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["params"], [1.0, 2.0, 3.0])
            self.assertEqual(candidate["params_source"], "matched_drsr_candidate")

    def test_recover_drsr_timeout_payload_uses_backfilled_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: beta
    - name: alpha
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "beta,alpha,y\n1,2,9\n2,3,14\n",
                "valid.csv": "beta,alpha,y\n3,4,19\n",
                "id_test.csv": "beta,alpha,y\n4,5,24\n",
                "ood_test.csv": "beta,alpha,y\n5,6,29\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            exp_dir = root / "drsr_exp"
            samples_dir = exp_dir / "samples"
            best_history_dir = exp_dir / "best_history"
            samples_dir.mkdir(parents=True)
            best_history_dir.mkdir(parents=True)
            function = (
                "def equation(beta, alpha, params):\n"
                "    return params[0] + params[1] * beta + params[2] * alpha\n"
            )
            (samples_dir / "top01_samples_0.json").write_text(
                json.dumps({"function": function, "score": 2.0}),
                encoding="utf-8",
            )
            (best_history_dir / "best_sample_0.json").write_text(
                json.dumps({"function": function, "score": 1.0, "params": [1.0, 2.0, 3.0]}),
                encoding="utf-8",
            )

            dataset = runner.load_canonical_dataset(dataset_dir)
            payload = runner._recover_timeout_payload_from_candidate(
                tool_name="drsr",
                dataset=dataset,
                experiment_dir=exp_dir,
            )

            self.assertIsNotNone(payload)
            self.assertIsNone(payload["canonical_artifact_error"])
            self.assertEqual(payload["canonical_artifact"]["parameter_values"], [1.0, 2.0, 3.0])
            self.assertEqual(payload["canonical_artifact"]["normalized_expression"], "c0 + c1*x0 + c2*x1")
            self.assertEqual(payload["id_metrics"]["nmse"], 0.0)
            self.assertEqual(payload["ood_metrics"]["nmse"], 0.0)

    def test_build_runner_params_can_disable_prompt_semantics_for_llmsr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: hidden semantic description
  target:
    name: y
    description: hidden target
  features:
    - name: x0
      description: hidden feature
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n", encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"inject_prompt_semantics": False},
            )

            self.assertFalse(params["inject_prompt_semantics"])
            self.assertNotIn("metadata_path", params)
            self.assertNotIn("feature_descriptions", params)
            self.assertNotIn("target_description", params)
            self.assertEqual(
                params["background"],
                "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
            )

    def test_build_runner_params_injects_explicit_dataset_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: explicit contract dataset
  target:
    name: output
  features:
    - name: feature_a
    - name: feature_b
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("feature_a,feature_b,output\n1,2,3\n", encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            params = runner.build_runner_params(
                "gplearn",
                dataset,
                root / "bench_results",
                seed=1314,
            )

            self.assertEqual(params["n_features"], 2)
            self.assertEqual(params["feature_names"], ["feature_a", "feature_b"])
            self.assertEqual(params["target_name"], "output")

    def test_train_label_noise_only_affects_fit_labels_and_keeps_clean_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            for name, rows in {
                "train.csv": "x0,y\n1,1\n2,2\n3,3\n4,4\n",
                "valid.csv": "x0,y\n5,5\n",
                "id_test.csv": "x0,y\n6,6\n",
                "ood_test.csv": "x0,y\n7,7\n",
            }.items():
                (dataset_dir / name).write_text(rows, encoding="utf-8")

            original_cls = runner.SymbolicRegressor
            runner.SymbolicRegressor = _RecordingRegressor
            try:
                result_path = runner.run_benchmark_task(
                    tool_name="gplearn",
                    dataset_dir=dataset_dir,
                    output_root=root / "bench_results",
                    seed=42,
                    params_override={
                        "train_label_noise_enabled": True,
                        "train_label_noise_sigma": 0.10,
                    },
                )
            finally:
                runner.SymbolicRegressor = original_cls

            clean_y = np.asarray([1.0, 2.0, 3.0, 4.0])
            self.assertIsNotNone(_RecordingRegressor.last_fit_y)
            self.assertFalse(np.allclose(_RecordingRegressor.last_fit_y, clean_y))
            self.assertNotIn("train_label_noise_enabled", _RecordingRegressor.last_params)
            self.assertNotIn("train_label_noise_sigma", _RecordingRegressor.last_params)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["train_label_noise"]["enabled"])
            self.assertEqual(result["train_label_noise"]["sigma"], 0.10)
            self.assertAlmostEqual(result["train_label_noise"]["scale"], 0.10 * float(np.std(clean_y)))
            self.assertEqual(result["train"]["nmse"], 0.0)
            self.assertEqual(result["valid"]["nmse"], 0.0)
            self.assertEqual(result["id_test"]["nmse"], 0.0)
            self.assertEqual(result["ood_test"]["nmse"], 0.0)

    def test_train_label_noise_is_reproducible_for_same_dataset_seed_and_sigma(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  target:
    name: y
  features:
    - name: x0
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,y\n1,1\n2,2\n3,3\n4,4\n", encoding="utf-8")

            dataset = runner.load_canonical_dataset(dataset_dir)
            params_1 = {"train_label_noise_sigma": 0.05}
            params_2 = {"train_label_noise_sigma": 0.05}
            config_1 = runner._resolve_train_label_noise_config(params_1, dataset=dataset, seed=7)
            config_2 = runner._resolve_train_label_noise_config(params_2, dataset=dataset, seed=7)
            np.testing.assert_allclose(
                runner._train_labels_for_fit(dataset.train, config_1),
                runner._train_labels_for_fit(dataset.train, config_2),
            )
            self.assertEqual(params_1, {})
            self.assertEqual(params_2, {})


class SrsdDistractorTest(unittest.TestCase):
    """测试 SRSD distractor 汇总描述逻辑。"""

    def _make_dataset(
        self,
        root: Path,
        description: str,
        feature_names: list[str],
        target_name: str,
        feature_descs: list[str | None],
        target_desc: str | None = None,
    ):
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        feat_lines = "\n".join(
            f'    - name: {n}\n      description: {d or ""}'
            for n, d in zip(feature_names, feature_descs)
        )
        target_line = f"  target:\n    name: {target_name}\n    description: {target_desc or ''}"
        meta = f"""
dataset:
  description: {description}
{target_line}
  features:
{feat_lines}
""".strip()
        (dataset_dir / "metadata.yaml").write_text(meta, encoding="utf-8")
        header = ",".join(feature_names + [target_name])
        (dataset_dir / "train.csv").write_text(
            f"{header}\n" + "0.5,0.5,1.0\n" * 10, encoding="utf-8"
        )
        return runner.load_canonical_dataset(dataset_dir)

    def test_distractor_summary_with_dummy(self):
        """含 meaningless 变量的数据集应生成汇总描述。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(
                root,
                "Calculate the Energy given Spring constant, Position",
                feature_names=["x0", "x1", "x2", "x3", "x4"],
                target_name="y",
                feature_descs=["meaningless", "k_spring, Spring constant", "x, Position", "meaningless", "meaningless"],
                target_desc="U, Elastic energy",
            )
            params = runner.build_runner_params(
                "llmsr", dataset, root / "bench_results", seed=1314,
            )

            # background 只能追加全局提示，不能泄露哪些变量是 active feature。
            self.assertIn("There are 5 candidate variables", params["background"])
            self.assertIn("unknown order", params["background"])
            self.assertIn('1 variable with semantic role "k_spring, Spring constant"', params["background"])
            self.assertIn('1 variable with semantic role "x, Position"', params["background"])
            self.assertIn("3 distractor/meaningless variables", params["background"])
            self.assertIn("mapping from semantic roles to variable names is intentionally hidden", params["background"])
            self.assertNotIn("Active features", params["background"])
            self.assertNotIn("x1 (k_spring, Spring constant)", params["background"])
            self.assertNotIn("x2 (x, Position)", params["background"])

            # 每个变量仍不暴露具体语义，只暴露无序语义集合。
            self.assertEqual(
                params["feature_descriptions"],
                ["candidate variable; semantic role hidden"] * 5,
            )

    def test_distractor_summary_without_dummy(self):
        """不含 meaningless 变量的数据集保持原始描述。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(
                root,
                "Calculate the Force given Mass, Acceleration",
                feature_names=["m", "a"],
                target_name="F",
                feature_descs=["Mass", "Acceleration"],
                target_desc="Force",
            )
            params = runner.build_runner_params(
                "llmsr", dataset, root / "bench_results", seed=1314,
            )

            # background 保持原样
            self.assertEqual(
                params["background"],
                "Calculate the Force given Mass, Acceleration",
            )
            # feature_descriptions 保持原始
            self.assertEqual(params["feature_descriptions"], ["Mass", "Acceleration"])

    def test_distractor_skipped_for_non_llm_tool(self):
        """非 llmsr/drsr 的工具不应触发 distractor 逻辑。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(
                root,
                "Calculate the Energy",
                feature_names=["x0", "x1"],
                target_name="y",
                feature_descs=["meaningless", "mass"],
                target_desc="Energy",
            )
            params = runner.build_runner_params(
                "gplearn", dataset, root / "bench_results", seed=1314,
            )

            # gplearn 不注入 background/feature_descriptions
            self.assertNotIn("background", params)
            self.assertNotIn("feature_descriptions", params)

    def test_semantic_prompt_variables_are_canonical_for_llmsr_and_drsr(self):
        """语义 prompt 默认使用 x0..xN/y，原始列名仅作为追溯信息保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(
                root,
                "Calculate reaction rate given Time and Concentration",
                feature_names=["t", "A"],
                target_name="dA_dt",
                feature_descs=["Time", "Concentration at time t"],
                target_desc="Rate of change of concentration",
            )

            for tool_name in ("llmsr", "drsr"):
                params = runner.build_runner_params(
                    tool_name,
                    dataset,
                    root / "bench_results",
                    seed=1314,
                )

                self.assertIs(params["canonical_prompt_variables"], True)
                self.assertEqual(params["feature_names"], ["t", "A"])
                self.assertEqual(params["target_name"], "dA_dt")
                self.assertEqual(params["prompt_feature_names"], ["x0", "x1"])
                self.assertEqual(params["prompt_target_name"], "y")
                self.assertEqual(params["original_feature_names"], ["t", "A"])
                self.assertEqual(params["original_target_name"], "dA_dt")
                self.assertEqual(params["feature_descriptions"], ["Time", "Concentration at time t"])
                self.assertEqual(params["target_description"], "Rate of change of concentration")


class AnonymizeTest(unittest.TestCase):
    """测试变量名匿名化逻辑。"""

    def _make_dataset(self, root: Path):
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "metadata.yaml").write_text(
            """
dataset:
  description: Test dataset
  target:
    name: output
    description: the target variable
  features:
    - name: mu
      description: Coefficient of friction
    - name: Nn
      description: Normal force
""".strip(),
            encoding="utf-8",
        )
        (dataset_dir / "train.csv").write_text("mu,Nn,output\n1,2,3\n", encoding="utf-8")
        return runner.load_canonical_dataset(dataset_dir)

    def test_anonymize_enabled_for_llmsr(self):
        """anonymize=True 时变量名应变为 x1..xN，目标名为 y。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(root)
            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"anonymize": True},
            )

            self.assertEqual(params["feature_names"], ["mu", "Nn"])
            self.assertEqual(params["target_name"], "output")
            self.assertEqual(params["prompt_feature_names"], ["x1", "x2"])
            self.assertEqual(params["prompt_target_name"], "y")
            self.assertIs(params["anonymize"], True)

    def test_anonymize_removes_descriptions(self):
        """anonymize=True 时不应注入变量/目标描述。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(root)
            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"anonymize": True},
            )

            self.assertNotIn("feature_descriptions", params)
            self.assertNotIn("target_description", params)

    def test_anonymize_disabled_by_default_when_semantics_disabled(self):
        """关闭语义注入时，默认 anonymize=False，变量名保持原始。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(root)
            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"inject_prompt_semantics": False},
            )

            self.assertEqual(params["feature_names"], ["mu", "Nn"])
            self.assertEqual(params["target_name"], "output")
            self.assertNotIn("anonymize", params)

    def test_anonymize_works_for_drsr(self):
        """anonymize 对 drsr 同样生效。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_dataset(root)
            params = runner.build_runner_params(
                "drsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"anonymize": True},
            )

            self.assertEqual(params["feature_names"], ["mu", "Nn"])
            self.assertEqual(params["target_name"], "output")
            self.assertEqual(params["prompt_feature_names"], ["x1", "x2"])
            self.assertEqual(params["prompt_target_name"], "y")

    def test_anonymize_with_srsd_distractor(self):
        """anonymize + SRSD distractor 时，以 anonymize 优先，不注入描述。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.yaml").write_text(
                """
dataset:
  description: Calculate Energy
  target:
    name: y
    description: Energy
  features:
    - name: x0
      description: meaningless
    - name: x1
      description: k_spring, Spring constant
""".strip(),
                encoding="utf-8",
            )
            (dataset_dir / "train.csv").write_text("x0,x1,y\n1,2,3\n", encoding="utf-8")
            dataset = runner.load_canonical_dataset(dataset_dir)

            params = runner.build_runner_params(
                "llmsr",
                dataset,
                root / "bench_results",
                seed=1314,
                params_override={"anonymize": True},
            )

            self.assertEqual(params["feature_names"], ["x0", "x1"])
            self.assertEqual(params["target_name"], "y")
            self.assertEqual(params["prompt_feature_names"], ["x1", "x2"])
            self.assertEqual(params["prompt_target_name"], "y")
            self.assertNotIn("feature_descriptions", params)
            self.assertNotIn("target_description", params)
