from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gplearn._program import _Program

from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact


WORK_DIR = Path(".agent/work/FIX-002")


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(19)
    X = rng.uniform(-2.0, 2.0, size=(96, 2))
    y = 0.37 + 1.7 * X[:, 0] - 0.4 * X[:, 1] ** 2
    return X, y


def _regressor(**kwargs) -> GPLearnRegressor:
    params = {
        "population_size": 40,
        "generations": 1,
        "tournament_size": 5,
        "stopping_criteria": 0.0,
        "const_range": (-2.0, 2.0),
        "function_set": ("add", "sub", "mul", "div", "sqrt", "log"),
        "init_depth": (2, 4),
        "metric": "mean absolute error",
        "parsimony_coefficient": 0.001,
        "p_crossover": 0.7,
        "p_subtree_mutation": 0.1,
        "p_hoist_mutation": 0.05,
        "p_point_mutation": 0.05,
        "max_samples": 0.65,
        "random_state": 1,
        "n_jobs": 1,
    }
    params.update(kwargs)
    return GPLearnRegressor(**params)


class GPLearnNativeExportTest(unittest.TestCase):
    def test_native_prefix_export_preserves_precision_and_protected_nodes(self) -> None:
        X, y = _training_data()
        regressor = _regressor(generations=1).fit(X, y)
        model = regressor.model
        functions = {function.name: function for function in model._function_set}
        native_program = _Program(
            function_set=model._function_set,
            arities=model._arities,
            init_depth=model.init_depth,
            init_method=model.init_method,
            n_features=2,
            const_range=model.const_range,
            metric=model._metric,
            p_point_replace=model.p_point_replace,
            parsimony_coefficient=model.parsimony_coefficient,
            random_state=np.random.RandomState(0),
            program=[
                functions["add"],
                functions["div"], 0, 0.12345678901234566,
                functions["add"],
                functions["div"], 0, 1,
                functions["log"], functions["sqrt"], 1,
            ],
        )
        model._program = native_program

        equation = regressor.get_optimal_equation()
        artifact = regressor.export_canonical_symbolic_program()
        boundary_X = np.array([[2.0, 0.0], [2.0, -4.0], [-3.0, 0.0005]])

        self.assertNotIn("0.12345678901234566", str(native_program))
        self.assertEqual(
            equation,
            "add(div(X0, 0.12345678901234566), "
            "add(div(X0, X1), log(sqrt(X1))))",
        )
        self.assertEqual(artifact["raw_equation"], equation)
        np.testing.assert_allclose(
            _predict_from_canonical_artifact(artifact, boundary_X),
            native_program.execute(boundary_X),
        )

    def test_fit_restores_metric_best_for_predict_export_and_progress(self) -> None:
        X, y = _training_data()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as temp_dir:
            regressor = _regressor(
                generations=6,
                exp_path=temp_dir,
                exp_name="case",
            ).fit(X, y)
            model = regressor.model
            generation_fitness = np.asarray(model.run_details_["best_fitness"], dtype=float)
            expected_fitness = (
                float(np.max(generation_fitness))
                if model._metric.greater_is_better
                else float(np.min(generation_fitness))
            )
            progress = json.loads(
                (Path(temp_dir) / "case" / ".gplearn_current_best.json").read_text(
                    encoding="utf-8"
                )
            )
            equation = regressor.get_optimal_equation()
            artifact = regressor.export_canonical_symbolic_program()

            self.assertAlmostEqual(model._program.raw_fitness_, expected_fitness)
            self.assertAlmostEqual(progress["loss"], model._program.raw_fitness_)
            self.assertEqual(progress["equation"], equation)
            self.assertEqual(artifact["raw_equation"], equation)
            np.testing.assert_allclose(regressor.predict(X), model._program.execute(X))


if __name__ == "__main__":
    unittest.main()
