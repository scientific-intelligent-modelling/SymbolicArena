import csv
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
        root = Path('.agent/work/FIX-003')
        root.mkdir(parents=True, exist_ok=True)
        for tool in ('gplearn', 'QLattice', 'dso', 'udsr', 'iMCTS', 'jaxsr', 'fepysr',
                     'llmsr', 'drsr', 'e2esr', 'tpsr', 'ragsr', 'symbolfit', 'pysr', 'pyoperon'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory(dir=root) as directory:
                directory = Path(directory)
                train = DatasetSplit('train', np.array([[0.0], [1.0]]), np.array([1.0, np.e]), 2)
                # 零目标使 NMSE 不可用，同时覆盖公式在 OOD 上溢出的情形。
                ood = DatasetSplit('ood_test', np.array([[1000.0], [1001.0]]), np.zeros(2), 2)
                dataset = LoadedDataset(directory, 'selection', {}, 'y', ['x0'], [None], None,
                                        train, train, train, ood)
                variable = ('X0' if tool == 'gplearn' else 'X1' if tool == 'pyoperon'
                            else 'x1' if tool in ('dso', 'udsr') else 'x0')
                equation = f'exp({variable})'
                old_equation = '1.0'
                if tool in ('llmsr', 'drsr'):
                    equation = 'def equation(x0, params):\n    return np.exp(x0)'
                    old_equation = 'def equation(x0, params):\n    return 1.0'
                candidate = {
                    'equation': equation, 'loss': 0.0, 'score': 0.0,
                    'fidelity': {'status': 'verified'},
                }
                if tool == 'e2esr':
                    candidate.update(selection_policy='e2esr_training_mse_v1',
                                     internal_objective='native_training_mse',
                                     objective_direction='min', training_mse=0.0)
                if tool in ('llmsr', 'drsr'):
                    samples = directory / 'samples'
                    samples.mkdir()
                    (samples / 'top01_0.json').write_text(json.dumps(
                        dict(candidate, function=equation, mse=0.0)))
                elif tool == 'pysr':
                    with (directory / 'hall_of_fame.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['Equation', 'Loss'])
                        writer.writeheader()
                        writer.writerow({'Equation': equation, 'Loss': 0.0})
                else:
                    (directory / f'.{tool.lower()}_current_best.json').write_text(json.dumps(candidate))
                artifact, error = safe_build_canonical_artifact(tool_name=tool, equation=old_equation, expected_n_features=1)
                self.assertIsNone(error)
                progress = directory / 'progress'
                progress.mkdir()
                old = {'tool': tool, 'equation': old_equation, 'canonical_artifact': artifact,
                       'valid': {'nmse': 1.0}, 'id_test': {'nmse': 1.0}, 'ood_test': {'nmse': 1.0}}
                if tool == 'e2esr':
                    old.update(selection_policy='e2esr_training_mse_v1',
                               internal_objective='native_training_mse',
                               objective_direction='min', training_mse=1.0)
                (progress / 'minute_0001.json').write_text(json.dumps(old))
                recovered = _recover_timeout_payload_from_candidate(
                    tool_name=tool, dataset=dataset, experiment_dir=directory)
                self.assertIsNotNone(recovered)
                self.assertEqual(recovered['equation'], equation)
                self.assertIsNone(recovered['canonical_artifact_error'])
                self.assertTrue(np.isfinite(recovered['train_metrics']['nmse']))
                self.assertTrue(recovered['ood_metrics'] is None
                                or recovered['ood_metrics']['nmse'] is None)

                latest = dict(old, equation=equation, ood_test=None)
                latest['canonical_artifact'], error = safe_build_canonical_artifact(
                    tool_name=tool, equation=equation, expected_n_features=1)
                self.assertIsNone(error)
                if tool == 'e2esr':
                    latest['training_mse'] = 0.0
                (progress / 'minute_0002.json').write_text(json.dumps(latest))
                recovered = _recover_timeout_payload_from_progress_snapshots(
                    tool_name=tool, dataset=dataset, experiment_dir=directory)
                self.assertEqual(recovered['equation'], equation)


if __name__ == '__main__':
    unittest.main()
