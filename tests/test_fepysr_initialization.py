import tempfile
import unittest
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor


class FePySRInitializationTest(unittest.TestCase):
    def test_mean_baseline_keeps_bootstrap_search_enabled(self):
        regressor = FePySRRegressor(timeout_in_seconds=180, num_experiments=8, fmn_epochs=30)
        regressor._install_mean_constant_baseline(np.array([1.0, 3.0]))
        self.assertIsNotNone(regressor._best_equation)
        params = regressor._prepare_attempt_params(attempt=1, remaining=160, has_fitted_model=False)
        self.assertEqual(params['num_experiments'], 1)
        self.assertEqual(params['fmn_epochs'], 5)
        self.assertEqual(params['timeout_in_seconds'], 40)
        later = regressor._prepare_attempt_params(attempt=2, remaining=160, has_fitted_model=True)
        self.assertEqual(later['num_experiments'], 8)
        self.assertEqual(later['fmn_epochs'], 30)

    def test_missing_native_configuration_fails_before_baseline(self):
        work = Path('.agent/work/FIX-004')
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            package = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'config_regression.yaml'):
                FePySRRegressor._require_native_config(package)
            self.assertFalse((package / '.fepysr_current_best.json').exists())


if __name__ == '__main__':
    unittest.main()
