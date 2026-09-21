import unittest

import numpy as np
import sympy as sp

from scientific_intelligent_modelling.benchmarks.normalizers import (
    normalize_operon_artifact,
    normalize_pysr_artifact,
)
from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact


class NativeVariableIndicesTest(unittest.TestCase):
    def test_pysr_keeps_native_indices_without_x0(self):
        X = np.array([[1000.0, 4.0, 3.0], [2000.0, 6.0, 2.0]])
        for formula in ('x1 * square(x2) / 2', 'x_1 * square(x_2) / 2'):
            with self.subTest(formula=formula):
                artifact = normalize_pysr_artifact(formula, expected_n_features=3)
                self.assertEqual(artifact['variables'], ['x1', 'x2'])
                np.testing.assert_allclose(_predict_from_canonical_artifact(artifact, X), [18.0, 12.0])
        self.assertFalse(normalize_pysr_artifact('x3', expected_n_features=3)['artifact_valid'])

    def test_operon_converts_one_based_indices_once(self):
        X = np.array([[1000.0, 4.0, 3.0], [2000.0, 6.0, 2.0]])
        artifact = normalize_operon_artifact('X2 * X3^2 / 2', expected_n_features=3)
        self.assertEqual(artifact['variables'], ['x1', 'x2'])
        np.testing.assert_allclose(_predict_from_canonical_artifact(artifact, X), [18.0, 12.0])
        self.assertFalse(normalize_operon_artifact('X4', expected_n_features=3)['artifact_valid'])

    def test_operon_replay_matches_exported_native_expression(self):
        raw = ('((-938301.437500) + (18550532.000000 * sin(((((((3.716143 * X2) * (5.650434 * X4)) + (-1.268332)) * '
               '((-0.176470) / (sqrt(1 + ((-0.402281) * X3) ^ 2)))) ^ 2) + ((((((3.176241 * X2) * (5.650434 * X4)) ^ 2) * '
               'sin((5.834888 * X4))) + ((((3.176241 * X2) * (5.130416 * X4)) ^ 2) + ((3.176241 * X2) * (5.650434 * X4)))) * '
               '((((5.130416 * X4) * (1.114966 * X5)) + (-1.808234)) / ((-0.176470) / (sqrt(1 + (((3.716143 * X2) * '
               '(5.650434 * X4)) * ((((-0.402281) * X3) + (-1.365669)) / ((-0.176470) / (sqrt(1 + (-0.176470) ^ 2))))) ^ 2)))))))))')
        X = np.array([[5.34016745584456e-27, 7.132301850103342e-9, -3.921061419419192e-24, 48226747.67928406, 1.1321039165375863e-9]])
        native = sp.lambdify(sp.symbols('X1:6'), sp.sympify(raw), modules=['numpy'])
        artifact = normalize_operon_artifact(raw, expected_n_features=5)
        np.testing.assert_allclose(_predict_from_canonical_artifact(artifact, X), native(*X.T), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
