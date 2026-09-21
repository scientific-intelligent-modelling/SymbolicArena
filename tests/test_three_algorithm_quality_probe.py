import importlib.util
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / 'diagnostics/three_algorithm_quality_probe.py'
SPEC = importlib.util.spec_from_file_location('quality_probe', PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_limit_rejects_over_three_minutes(tmp_path):
    with pytest.raises(ValueError):
        probe.run_bounded([sys.executable, '-c', 'pass'], 181, tmp_path / 'log')


def test_hard_timeout_reaps_process(tmp_path):
    result = probe.run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'], 0.2, tmp_path / 'log')
    assert result['hard_timeout']
    assert result['returncode'] < 0
    assert result['wall_seconds'] < 3


def test_successful_command_is_not_timeout(tmp_path):
    result = probe.run_bounded([sys.executable, '-c', 'print(123)'], 3, tmp_path / 'log')
    assert result['returncode'] == 0
    assert not result['hard_timeout']
    assert (tmp_path / 'log').read_text().strip() == '123'


def test_nonfinite_predictions_are_unresolved_not_penalized():
    assert probe.metrics([1., 2.], [1., float('nan')])['r2'] is None
    assert probe.metrics([1., 2.], [1e300, -1e300])['mse'] is None


def _valid_report():
    return {'execution': {'returncode': 0, 'hard_timeout': False}, 'worker': {'status': 'ok'},
            'metrics': {'id': {'native': {'finite': True}, 'exported': {'finite': True},
                               'export_matches_native': True}}}


def test_diagnostic_exit_code_reports_worker_error_and_hard_timeout():
    report = _valid_report()
    report['worker']['status'] = 'error'
    assert probe.validation_status(report)['exit_code'] == 1
    report['execution']['hard_timeout'] = True
    assert probe.validation_status(report)['exit_code'] == 124


def test_diagnostic_rejects_export_mismatch_and_nonfinite_metrics():
    report = _valid_report()
    report['metrics']['id']['export_matches_native'] = False
    assert not probe.validation_status(report)['ok']
    report = _valid_report()
    report['metrics']['id']['native']['finite'] = False
    assert probe.validation_status(report)['exit_code'] == 1


def test_diagnostic_success_has_zero_exit_code():
    assert probe.validation_status(_valid_report())['exit_code'] == 0


def test_e2esr_selection_policy_requires_snapshot_to_match_final_model():
    report = _valid_report()
    report['algorithm'] = 'e2esr'
    report['worker']['selection_policy'] = 'e2esr_training_mse_v1'
    assert probe.validation_status(report)['exit_code'] == 1
    report['metrics']['id']['snapshot'] = {'finite': True}
    report['metrics']['id']['snapshot_matches_native'] = True
    assert probe.validation_status(report)['exit_code'] == 0
