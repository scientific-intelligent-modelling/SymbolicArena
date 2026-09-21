import json
import time

import pytest

from scientific_intelligent_modelling.benchmarks import runner
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_e2esr_artifact


def dataset(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'metadata.yaml').write_text('dataset:\n  target: {name: y}\n')
    for split in ('train', 'valid', 'id_test', 'ood_test'):
        (root / (split + '.csv')).write_text('x0,y\n1,3\n2,5\n3,7\n')
    return runner.load_canonical_dataset(root)


def current(loss=0.0):
    return {'equation': '2*x_0+1', 'score': loss, 'loss': loss, 'internal_loss': loss,
            'training_mse': loss, 'native_model_score': -99.0,
            'internal_objective': 'native_training_mse', 'objective_direction': 'min',
            'selection_policy': 'e2esr_training_mse_v1'}


def test_periodic_payload_uses_mse_not_decoder_score(tmp_path):
    data = dataset(tmp_path)
    exp = tmp_path / 'exp'
    exp.mkdir()
    (exp / '.e2esr_current_best.json').write_text(json.dumps(current()))
    payload = runner._build_periodic_snapshot_payload(
        tool_name='e2esr', dataset=data, params={}, seed=520,
        started_at=time.time()-60, experiment_dir=exp, checkpoint_index=1)
    assert payload['internal_objective_value'] == 0.0
    assert payload['internal_objective_direction'] == 'min'
    assert payload['candidate_selection_policy'] == 'e2esr_training_mse_v1'
    assert payload['native_model_score'] == -99.0
    assert payload['id_test']['rmse'] == 0.0


def test_heartbeat_declares_new_selection_contract(tmp_path):
    payload = runner._build_periodic_snapshot_payload(
        tool_name='e2esr', dataset=dataset(tmp_path), params={}, seed=520,
        started_at=time.time(), experiment_dir=tmp_path/'exp', checkpoint_index=1)
    assert payload['internal_objective'] == 'native_training_mse'
    assert payload['internal_objective_direction'] == 'min'
    assert payload['native_objective_unavailable']


@pytest.mark.parametrize('loss', [None, float('nan'), float('inf'), -1.0])
def test_invalid_mse_cannot_be_recovered(tmp_path, loss):
    (tmp_path / '.e2esr_current_best.json').write_text(json.dumps(current(loss)))
    assert runner._extract_e2esr_periodic_candidate(tmp_path) is None


def test_inconsistent_loss_aliases_fail_closed(tmp_path):
    item = current(0.1)
    item['internal_loss'] = 0.2
    (tmp_path / '.e2esr_current_best.json').write_text(json.dumps(item))
    assert runner._extract_e2esr_periodic_candidate(tmp_path) is None


def test_recovery_keeps_candidate_discovery_evidence(tmp_path):
    item = current()
    item.update(first_discovered_elapsed_seconds=12.0, first_discovered_minute=1,
                candidate_sha256='a'*64, source_timestamp_unix=100.0)
    data = dataset(tmp_path)
    (tmp_path / '.e2esr_current_best.json').write_text(json.dumps(item))
    result = runner._recover_timeout_payload_from_candidate(
        tool_name='e2esr', dataset=data, experiment_dir=tmp_path)
    assert result['first_discovered_elapsed_seconds'] == 12.0
    assert result['candidate_sha256'] == 'a'*64


def test_legacy_probability_snapshot_and_minute_backup_are_rejected(tmp_path):
    data = dataset(tmp_path)
    exp = tmp_path / 'exp'
    (exp / 'progress').mkdir(parents=True)
    old = {'equation': '2*x_0+1', 'native_model_score': -.1, 'score': -.1,
           'internal_objective': 'decoder_length_normalized_log_likelihood',
           'objective_direction': 'max'}
    (exp / '.e2esr_current_best.json').write_text(json.dumps(old))
    minute = dict(old, tool='e2esr', canonical_artifact=normalize_e2esr_artifact(old['equation']),
                  valid={'nmse': 0.0}, id_test={'nmse': 0.0}, ood_test={'nmse': 0.0})
    (exp / 'progress/minute_0001.json').write_text(json.dumps(minute))
    assert runner._extract_e2esr_periodic_candidate(exp) is None
    assert runner._recover_timeout_payload_from_candidate(
        tool_name='e2esr', dataset=data, experiment_dir=exp) is None


def test_timeout_recovery_reads_the_training_selected_equation(tmp_path):
    data = dataset(tmp_path)
    exp = tmp_path / 'exp'
    exp.mkdir()
    (exp / '.e2esr_current_best.json').write_text(json.dumps(current()))
    payload = runner._recover_timeout_payload_from_candidate(
        tool_name='e2esr', dataset=data, experiment_dir=exp)
    assert payload['equation'] == '2*x_0+1'
    assert payload['id_metrics']['rmse'] == 0.0
    assert payload['selection_policy'] == 'e2esr_training_mse_v1'


def test_minute_fallback_does_not_choose_using_ood_validity(tmp_path):
    data = dataset(tmp_path)
    exp = tmp_path / 'exp'
    (exp / 'progress').mkdir(parents=True)
    for minute, loss, equation, ood_metrics in [(1, 1.0, 'x_0', {'nmse': 0.0}),
                                               (2, 0.0, '2*x_0+1', None)]:
        item = dict(current(loss), tool='e2esr', equation=equation,
                    canonical_artifact=normalize_e2esr_artifact(equation),
                    valid={'nmse': 0.0}, id_test={'nmse': 0.0}, ood_test=ood_metrics)
        (exp / 'progress' / ('minute_%04d.json' % minute)).write_text(json.dumps(item))
    payload = runner._recover_timeout_payload_from_progress_snapshots(
        tool_name='e2esr', dataset=data, experiment_dir=exp)
    assert payload['equation'] == '2*x_0+1'
    assert payload['ood_metrics'] is None
