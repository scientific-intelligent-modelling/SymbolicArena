import json
import unittest

import numpy as np

from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper import iMCTSRegressor
from scientific_intelligent_modelling.algorithms.udsr_wrapper.wrapper import UDSRRegressor
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact
from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact


class RemainingNativePredictionsTest(unittest.TestCase):
    def test_sparse_variables_match_native_evaluators(self):
        X = np.array([[100.0, 2.0, 3.0], [200.0, 4.0, 5.0]])
        formulas = {'fepysr': 'x1 + 2*x2', 'iMCTS': 'x[1] + 2*x[2]',
                    'dso': 'x2 + 2*x3', 'udsr': 'x2 + 2*x3'}
        for tool, equation in formulas.items():
            with self.subTest(tool=tool):
                artifact, error = safe_build_canonical_artifact(tool_name=tool, equation=equation, expected_n_features=3)
                self.assertIsNone(error)
                if tool == 'fepysr':
                    prediction = FePySRRegressor._build_callable(equation)(X)
                elif tool == 'iMCTS':
                    model = iMCTSRegressor.deserialize(json.dumps({'expr_vector': equation, 'n_features': 3}))
                    prediction = model.predict(X)
                else:
                    model = (UDSRRegressor if tool == 'udsr' else DSORegressor)()
                    model._build_predict_fn_from_equation(equation)
                    prediction = model.predict(X)
                np.testing.assert_allclose(prediction, [8.0, 14.0])
                np.testing.assert_allclose(_predict_from_canonical_artifact(artifact, X), prediction)

    def test_imcts_constant_model_returns_one_value_per_sample(self):
        model = iMCTSRegressor.deserialize(json.dumps({'expr_vector': '2.5', 'n_features': 2}))
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(model.predict(X).shape, (2,))
        np.testing.assert_array_equal(model.predict(X), [2.5, 2.5])


if __name__ == '__main__':
    unittest.main()
