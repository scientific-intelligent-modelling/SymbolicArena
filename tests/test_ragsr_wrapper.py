import sys
import tempfile
import types
import unittest

import numpy as np

from scientific_intelligent_modelling.algorithms.ragsr_wrapper.wrapper import RAGSRRegressor
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_ragsr_artifact
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact


class _FakeEvolutionaryForestRegressor:
    last_params = None

    def __init__(self, **params):
        self.params = params
        _FakeEvolutionaryForestRegressor.last_params = params

    def fit(self, X, y, **fit_kwargs):
        self.X_shape = np.asarray(X).shape
        self.y_shape = np.asarray(y).shape
        self.fit_kwargs = fit_kwargs
        if hasattr(self, "callback"):
            self.callback()
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, 0] + 2.0 * X_arr[:, 1]

    def model(self):
        return "x0 + 2*x1"


class RAGSRWrapperTest(unittest.TestCase):
    def test_params_absorb_runner_contract_and_default_official_categorical_mode(self):
        reg = RAGSRRegressor(
            seed=17,
            n_features=2,
            feature_names=["x0", "x1"],
            target_name="y",
            n_gen="1",
            n_pop="10",
        )

        self.assertEqual(reg.params["random_state"], 17)
        self.assertEqual(reg.params["n_gen"], 1)
        self.assertEqual(reg.params["n_pop"], 10)
        self.assertEqual(reg.params["categorical_encoding"], "Target")
        self.assertNotIn("n_features", reg.params)
        self.assertNotIn("feature_names", reg.params)
        self.assertNotIn("target_name", reg.params)

    def test_timeout_budget_clamps_explicit_small_generation_budget(self):
        reg = RAGSRRegressor(timeout_in_seconds=3600, n_gen=100)

        self.assertEqual(reg.params["n_gen"], 100000)
        self.assertEqual(reg.params["time_limit"], 3595.0)

    def test_fit_predict_and_export_with_fake_backend(self):
        fake_package = types.ModuleType("evolutionary_forest")
        fake_forest = types.ModuleType("evolutionary_forest.forest")
        fake_forest.EvolutionaryForestRegressor = _FakeEvolutionaryForestRegressor

        old_modules = {
            name: sys.modules.get(name)
            for name in ("evolutionary_forest", "evolutionary_forest.forest")
        }
        try:
            sys.modules["evolutionary_forest"] = fake_package
            sys.modules["evolutionary_forest.forest"] = fake_forest

            reg = RAGSRRegressor(
                seed=3,
                n_features=2,
                feature_names=["x0", "x1"],
                target_name="y",
                n_gen=1,
                n_pop=10,
                categorical_features=[False, False],
            )
            X = np.array([[1.0, 2.0], [3.0, 4.0]])
            y = np.array([5.0, 11.0])
            reg.fit(X, y)

            self.assertEqual(_FakeEvolutionaryForestRegressor.last_params["random_state"], 3)
            self.assertNotIn("categorical_features", _FakeEvolutionaryForestRegressor.last_params)
            self.assertEqual(reg.model.fit_kwargs["categorical_features"], [False, False])
            self.assertEqual(reg.get_optimal_equation(), "x0 + 2*x1")
            np.testing.assert_allclose(reg.predict(X), np.array([5.0, 11.0]))

            artifact = reg.export_canonical_symbolic_program()
            self.assertEqual(artifact["tool_name"], "ragsr")
            self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
            self.assertEqual(artifact["expected_n_features"], 2)
            self.assertTrue(artifact["artifact_valid"])

            restored = RAGSRRegressor.deserialize(reg.serialize())
            self.assertEqual(restored.get_optimal_equation(), "x0 + 2*x1")
            np.testing.assert_allclose(restored.predict(X), np.array([5.0, 11.0]))
        finally:
            for name, old_value in old_modules.items():
                if old_value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_value

    def test_fit_writes_ragsr_current_best_snapshot(self):
        fake_package = types.ModuleType("evolutionary_forest")
        fake_forest = types.ModuleType("evolutionary_forest.forest")
        fake_forest.EvolutionaryForestRegressor = _FakeEvolutionaryForestRegressor

        old_modules = {
            name: sys.modules.get(name)
            for name in ("evolutionary_forest", "evolutionary_forest.forest")
        }
        try:
            sys.modules["evolutionary_forest"] = fake_package
            sys.modules["evolutionary_forest.forest"] = fake_forest
            with tempfile.TemporaryDirectory() as tmpdir:
                reg = RAGSRRegressor(
                    seed=3,
                    n_features=2,
                    feature_names=["x0", "x1"],
                    target_name="y",
                    exp_path=tmpdir,
                    exp_name="ragsr_snapshot_test",
                    n_gen=1,
                    n_pop=10,
                )
                X = np.array([[1.0, 2.0], [3.0, 4.0]])
                y = np.array([5.0, 11.0])
                reg.fit(X, y)

                snapshot_path = f"{tmpdir}/ragsr_snapshot_test/.ragsr_current_best.json"
                with open(snapshot_path, encoding="utf-8") as f:
                    content = f.read()

                self.assertIn('"tool": "ragsr"', content)
                self.assertIn('"equation": "x0 + 2*x1"', content)
        finally:
            for name, old_value in old_modules.items():
                if old_value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_value

    def test_fit_rejects_contract_dimension_mismatch_before_backend_import(self):
        reg = RAGSRRegressor(
            n_features=3,
            feature_names=["x0", "x1", "x2"],
            target_name="y",
            n_gen=1,
            n_pop=10,
        )
        X = np.ones((4, 2))
        y = np.ones(4)

        with self.assertRaisesRegex(ValueError, "显式维度契约不一致"):
            reg.fit(X, y)

    def test_deserialized_predict_replays_max_min_with_numpy_broadcasting(self):
        reg = RAGSRRegressor(n_features=1, feature_names=["x0"], target_name="y")
        reg._equation = "Max(1.0, x0) + Min(0.5, x0)"
        restored = RAGSRRegressor.deserialize(reg.serialize())

        pred = restored.predict(np.array([[0.2], [2.0]]))

        np.testing.assert_allclose(pred, np.array([1.2, 2.5]))

    def test_normalize_realistic_evolutionary_forest_expression(self):
        expr = (
            "((0.0*(((sin(3.141592653589793*(-0.09446021299983642)))"
            "--0.29241907596588135)/1.0)+-0.15951381733803377*(((((cos("
            "3.141592653589793*(-x1)))-(log(1+abs((cos(3.141592653589793*x0)))))))"
            "--0.39827555456381547)/0.7351190655448211)+0.4105614779818909)"
            " * (1.5380092765662026 - -0.6142191231161902)) + -0.6142191231161902"
        )
        artifact = normalize_ragsr_artifact(expr, expected_n_features=2)

        self.assertEqual(artifact["tool_name"], "ragsr")
        self.assertEqual(artifact["variables"], ["x0", "x1"])
        self.assertTrue(artifact["sympy_parse_ok"])
        self.assertTrue(artifact["artifact_valid"])

    def test_normalize_ragsr_keeps_zero_based_single_feature_reference(self):
        artifact = normalize_ragsr_artifact("sin(x1)", expected_n_features=2)

        self.assertEqual(artifact["normalized_expression"], "sin(x1)")
        self.assertEqual(artifact["variables"], ["x1"])
        self.assertTrue(artifact["artifact_valid"])

    def test_safe_build_canonical_artifact_for_ragsr(self):
        artifact, error = safe_build_canonical_artifact(
            tool_name="ragsr",
            equation="x0 + 2*x1",
            expected_n_features=2,
        )

        self.assertIsNone(error)
        self.assertEqual(artifact["tool_name"], "ragsr")
        self.assertEqual(artifact["normalized_expression"], "x0 + 2*x1")
        self.assertEqual(artifact["variables"], ["x0", "x1"])


if __name__ == "__main__":
    unittest.main()
