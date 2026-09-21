import json
import sys
import types

import numpy as np
import pytest

from scientific_intelligent_modelling.algorithms.e2esr_wrapper import wrapper


def make_writer(tmp_path):
    reg = wrapper.E2ESRRegressor.__new__(wrapper.E2ESRRegressor)
    reg._progress_state_path = str(tmp_path / '.e2esr_current_best.json')
    reg._fit_started_at = 100.0
    return reg


def emit(reg, equation, mse, likelihood=None, **kwargs):
    reg._write_progress_state(equation=equation, training_mse=mse,
                              native_model_score=likelihood, bag_index=0, candidate_rank=0, **kwargs)


def read_best(reg):
    with open(reg._progress_state_path) as handle:
        return json.load(handle)


def write_recovery_snapshot(tmp_path, **updates):
    exp_dir = tmp_path / 'recover'
    exp_dir.mkdir()
    payload = {
        'equation': '2*x_1 + 1',
        'selection_policy': 'e2esr_training_mse_v1',
        'internal_objective': 'native_training_mse',
        'objective_direction': 'min',
        'training_mse': 0.25,
        'internal_loss': 0.25,
        'score': 0.25,
        'n_features': 2,
        'feature_names': ['x_0', 'x_1'],
        'target_name': 'y',
    }
    payload.update(updates)
    (exp_dir / '.e2esr_current_best.json').write_text(json.dumps(payload))
    return exp_dir


def test_training_error_wins_over_decoder_probability_and_ties_keep_first(tmp_path):
    reg = make_writer(tmp_path)
    emit(reg, 'x_0', 0.1, -20.0)
    emit(reg, 'x_1', 8.0, -0.01)
    emit(reg, 'x_2', 0.1, -0.001)
    best = read_best(reg)
    assert best['equation'] == 'x_0'
    assert best['internal_objective'] == 'native_training_mse'
    assert best['objective_direction'] == 'min'
    assert best['internal_loss'] == best['score'] == best['training_mse'] == 0.1
    assert best['native_model_score'] == -20.0
    assert best['selection_policy'] == 'e2esr_training_mse_v1'


def test_bfgs_without_decoder_score_replaces_raw_candidate(tmp_path):
    reg = make_writer(tmp_path)
    emit(reg, 'x_0', 0.3, -0.01)
    emit(reg, '2*x_0', 0.001, refinement_type='BFGS', stage='bfgs_best')
    best = read_best(reg)
    assert best['equation'] == '2*x_0'
    assert best['native_model_score'] is None
    assert best['training_mse'] == 0.001
    assert best['refinement_type'] == 'BFGS'


@pytest.mark.parametrize('loss', [None, float('nan'), float('inf'), -1.0])
def test_invalid_training_loss_never_selects_candidate(tmp_path, loss):
    reg = make_writer(tmp_path)
    emit(reg, 'x_0', loss, -0.1)
    assert not (tmp_path / '.e2esr_current_best.json').exists()


def test_old_likelihood_snapshot_is_not_compared_as_mse(tmp_path):
    reg = make_writer(tmp_path)
    (tmp_path / '.e2esr_current_best.json').write_text(json.dumps({
        'equation': '999', 'score': -0.1, 'native_model_score': -0.1,
        'internal_objective': 'decoder_length_normalized_log_likelihood',
        'objective_direction': 'max'}))
    emit(reg, 'x_0', 0.1)
    assert read_best(reg)['equation'] == 'x_0'


def test_new_fit_does_not_reuse_previous_fit_loss(tmp_path):
    reg = make_writer(tmp_path)
    emit(reg, 'x_0', 0.001)
    reg._fit_started_at = 200.0
    emit(reg, '2*x_0', 0.2)
    assert read_best(reg)['equation'] == '2*x_0'


def test_round_selection_uses_mse_even_for_constant_targets():
    assert wrapper.E2ESRRegressor._tree_score({'_mse': 0.01, 'r2': -100.0}) == -0.01
    assert wrapper.E2ESRRegressor._tree_score({'_mse': float('nan'), 'r2': 1.0}) is None


def test_last_worse_round_does_not_replace_final_or_snapshot(monkeypatch, tmp_path):
    clock = {'now': 100.0}
    monkeypatch.setattr(wrapper.time, 'time', lambda: clock['now'])

    def load_model(reg):
        clock['now'] += 2.0
        reg.model = object()

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', load_model)
    fits = []

    class Tree:
        def __init__(self, value):
            self.value = value

        def infix(self):
            return str(self.value)

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            self.value = 0.5 if not fits else 2.0
            fits.append(self.value)
            clock['now'] += 6.0
            self.params['progress_callback'](
                equation=str(self.value), training_mse=self.value**2,
                native_model_score=None, bag_index=0, candidate_rank=0,
                refinement_type='BFGS', stage='fit_final')

        def retrieve_tree(self, with_infos=False):
            return {'relabed_predicted_tree': Tree(self.value), '_mse': self.value**2,
                    'refinement_type': 'BFGS'}

        def predict(self, X):
            return np.full(len(X), self.value)

    monkeypatch.setitem(sys.modules, 'symbolicregression', types.ModuleType('symbolicregression'))
    module = types.ModuleType('symbolicregression.model')
    module.SymbolicTransformerRegressor = FakeRegressor
    monkeypatch.setitem(sys.modules, 'symbolicregression.model', module)
    reg = wrapper.E2ESRRegressor(timeout_in_seconds=20, timeout_guard_seconds=2,
                                exp_path=str(tmp_path), exp_name='fit')
    reg.fit(np.ones((5, 1)), np.zeros(5))
    assert fits == [0.5, 2.0]
    assert reg._fit_started_at == 100.0
    assert reg._budget_chunks_run == 2
    assert float(reg.get_optimal_equation()) == 0.5
    assert float(read_best(reg)['equation']) == 0.5
    np.testing.assert_allclose(reg.predict(np.ones((2, 1))), 0.5)


def test_valid_timeout_snapshot_skips_model_load_and_replays_native_formula(monkeypatch, tmp_path):
    exp_dir = write_recovery_snapshot(tmp_path)

    def unexpected_load(_reg):
        raise AssertionError('有效恢复不应加载 358M E2ESR 权重')

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', unexpected_load)
    reg = wrapper.E2ESRRegressor(
        existing_exp_dir=str(exp_dir),
        exp_dir=str(exp_dir),
        n_features=2,
        feature_names=['x_0', 'x_1'],
        target_name='y',
    )

    X = np.array([[1.0, 2.0], [-3.0, 0.5]])
    np.testing.assert_allclose(reg.predict(X), [5.0, 2.0])
    assert reg.get_optimal_equation() == '2*x_1 + 1'
    assert reg.get_total_equations() == ['2*x_1 + 1']
    artifact = reg.export_canonical_symbolic_program()
    assert artifact['artifact_valid']
    assert artifact['expected_n_features'] == 2


def test_timeout_recovery_serialize_roundtrip_stays_model_free(monkeypatch, tmp_path):
    exp_dir = write_recovery_snapshot(tmp_path)

    def unexpected_load(_reg):
        raise AssertionError('恢复态序列化往返不应加载模型')

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', unexpected_load)
    reg = wrapper.E2ESRRegressor(existing_exp_dir=str(exp_dir), n_features=2)
    serialized = reg.serialize()
    (exp_dir / '.e2esr_current_best.json').unlink()

    restored = wrapper.E2ESRRegressor.deserialize(serialized)
    np.testing.assert_allclose(restored.predict([[2.0, 3.0]]), [7.0])
    assert restored.get_optimal_equation() == reg.get_optimal_equation()
    assert restored.get_total_equations() == reg.get_total_equations()


def test_subprocess_timeout_recovery_closes_direct_api_loop(monkeypatch, tmp_path):
    from scientific_intelligent_modelling.srkit import subprocess_runner

    exp_dir = write_recovery_snapshot(tmp_path)

    def unexpected_load(_reg):
        raise AssertionError('subprocess 超时恢复不应加载模型')

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', unexpected_load)
    result = subprocess_runner.handle_recover_from_timeout(
        wrapper.E2ESRRegressor,
        {
            'tool_name': 'e2esr',
            'experiment_dir': str(exp_dir),
            'params': {
                'n_features': 2,
                'feature_names': ['x_0', 'x_1'],
                'target_name': 'y',
            },
            'data': {'X': [[1.0, 2.0], [2.0, 3.0]], 'y': [5.0, 7.0]},
        },
    )

    assert result['success']
    assert result['recovered_from_timeout']
    assert result['equation'] == '2*x_1 + 1'
    restored = wrapper.E2ESRRegressor.deserialize(result['serialized_model'])
    np.testing.assert_allclose(restored.predict([[3.0, 4.0]]), [9.0])


@pytest.mark.parametrize(
    'updates',
    [
        {
            'selection_policy': 'e2esr_decoder_probability_v0',
            'internal_objective': 'decoder_length_normalized_log_likelihood',
            'objective_direction': 'max',
        },
        {'training_mse': float('nan'), 'internal_loss': float('nan'), 'score': float('nan')},
        {'n_features': 3},
        {'equation': 'x_2 + 1'},
        {'equation': 'x_0 + rogue'},
    ],
    ids=['old-policy', 'nan-loss', 'metadata-mismatch', 'overflow-variable', 'unknown-symbol'],
)
def test_invalid_timeout_snapshots_are_rejected(monkeypatch, tmp_path, updates):
    exp_dir = write_recovery_snapshot(tmp_path, **updates)
    loads = []

    def record_load(reg):
        loads.append(True)
        reg.model = object()

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', record_load)
    with pytest.raises(ValueError, match='没有符合 training MSE 契约'):
        wrapper.E2ESRRegressor(existing_exp_dir=str(exp_dir), n_features=2)

    assert loads == []


def test_missing_explicit_recovery_snapshot_fails_without_model_load(monkeypatch, tmp_path):
    loads = []

    def record_load(_reg):
        loads.append(True)

    monkeypatch.setattr(wrapper.E2ESRRegressor, '_load_model', record_load)
    with pytest.raises(ValueError, match='没有符合 training MSE 契约'):
        wrapper.E2ESRRegressor(existing_exp_dir=str(tmp_path / 'missing'), n_features=2)

    assert loads == []
