import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_build_canonical_artifact
from scientific_intelligent_modelling.benchmarks.runner import (
    DatasetSplit,
    LoadedDataset,
    _recover_timeout_payload_from_candidate,
    _recover_timeout_payload_from_progress_snapshots,
)


class NativeBudgetSelectionTest(unittest.TestCase):
    def test_test_set_overflow_does_not_replace_native_incumbent(self):
        root = Path('.agent/work/FIX-002')
        root.mkdir(parents=True, exist_ok=True)
        for tool in ('gplearn', 'QLattice', 'dso', 'udsr', 'iMCTS', 'jaxsr', 'fepysr'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory(dir=root) as directory:
                directory = Path(directory)
                train = DatasetSplit('train', np.array([[0.0], [1.0]]), np.array([1.0, np.e]), 2)
                ood = DatasetSplit('ood_test', np.array([[1000.0], [1001.0]]), np.array([1.0, 2.0]), 2)
                dataset = LoadedDataset(directory, 'selection', {}, 'y', ['x0'], [None], None,
                                        train, train, train, ood)
                variable = 'X0' if tool == 'gplearn' else ('x1' if tool in ('dso', 'udsr') else 'x0')
                equation = f'exp({variable})'
                (directory / f'.{tool.lower()}_current_best.json').write_text(json.dumps({
                    'equation': equation, 'loss': 0.0, 'score': 0.0,
                    'fidelity': {'status': 'verified'},
                }))
                artifact, error = safe_build_canonical_artifact(tool_name=tool, equation='1.0', expected_n_features=1)
                self.assertIsNone(error)
                progress = directory / 'progress'
                progress.mkdir()
                old = {'tool': tool, 'equation': '1.0', 'canonical_artifact': artifact,
                       'valid': {'nmse': 1.0}, 'id_test': {'nmse': 1.0}, 'ood_test': {'nmse': 1.0}}
                (progress / 'minute_0001.json').write_text(json.dumps(old))
                recovered = _recover_timeout_payload_from_candidate(
                    tool_name=tool, dataset=dataset, experiment_dir=directory)
                self.assertIsNotNone(recovered)
                self.assertEqual(recovered['equation'], equation)

                latest = dict(old, equation=equation, ood_test=None)
                latest['canonical_artifact'], error = safe_build_canonical_artifact(
                    tool_name=tool, equation=equation, expected_n_features=1)
                self.assertIsNone(error)
                (progress / 'minute_0002.json').write_text(json.dumps(latest))
                recovered = _recover_timeout_payload_from_progress_snapshots(
                    tool_name=tool, dataset=dataset, experiment_dir=directory)
                self.assertEqual(recovered['equation'], equation)


if __name__ == '__main__':
    unittest.main()
