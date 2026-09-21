import sys
import types

import numpy as np
import pytest

from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor


class CandidateTable:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def iterrows(self):
        return iter(enumerate(self.rows))


def test_negative_parameter_substitution_preserves_power_precedence():
    row = {
        'Parameterized equation, unscaled': 'a1**2 + x0/a2',
        'Parameters: (best-fit, +1, -1)': {'a1': (-2.0, 0, 0), 'a2': (-4.0, 0, 0)},
    }
    expression = SymbolFitRegressor._extract_best_equation(row)
    predict = SymbolFitRegressor._build_callable(expression)
    np.testing.assert_allclose(predict([[4.0], [8.0]]), [3.0, 2.0])


def test_nonfinite_rmse_cannot_beat_finite_candidate():
    invalid = {'RMSE': np.nan, 'R2': np.nan, 'Parameterized equation, unscaled': '0'}
    valid = {'RMSE': 0.1, 'R2': 0.9, 'Parameterized equation, unscaled': 'x0'}
    reg = SymbolFitRegressor()
    reg.model = types.SimpleNamespace(func_candidates=CandidateTable([invalid, valid]))
    assert reg._select_best_candidate() is valid


def test_all_invalid_candidate_scores_are_rejected():
    reg = SymbolFitRegressor()
    reg.model = types.SimpleNamespace(func_candidates=CandidateTable([
        {'RMSE': float('nan')}, {'RMSE': float('inf')}, {'RMSE': -1.0},
    ]))
    with pytest.raises(ValueError, match='finite|有效|有限'):
        reg._select_best_candidate()


def test_parameter_substitution_does_not_confuse_a1_and_a10():
    expression = SymbolFitRegressor._extract_best_equation({
        'Parameterized equation, unscaled': 'a1*x0 + a10',
        'Parameters: (best-fit, +1, -1)': {'a1': (-2., 0, 0), 'a10': (10., 0, 0)},
    })
    np.testing.assert_allclose(SymbolFitRegressor._build_callable(expression)([[1.], [2.]]), [8., 6.])


def test_invalid_best_fit_parameter_is_not_exported():
    with pytest.raises(ValueError, match='有限'):
        SymbolFitRegressor._extract_best_equation({
            'Parameterized equation, unscaled': 'a1*x0',
            'Parameters: (best-fit, +1, -1)': {'a1': (np.nan, 0, 0)},
        })


@pytest.mark.parametrize('X,y,mode,expected_rescale,expected_mode,expected_y_scale', [
    ([[2., 0.], [2., 1.]], [1., 2.], 'mean', False, 'mean', 1.),
    ([[-1.], [1.]], [-1., 1.], 'mean', True, None, 1.),
    ([[-1.], [1.]], [-1., 1. + np.finfo(float).eps], 'mean', True, None, 1.),
    ([[0.], [1.]], [-1., 0.], 'max', True, None, 1.),
    ([[0.], [1.]], [0., 0.], 'l2', True, None, 1.),
    ([[0.], [1.]], [1e-30, 2e-30], 'mean', True, 'mean', 1. / 1.5e-30),
])
def test_fit_does_not_pass_singular_scaling_to_upstream(
    monkeypatch, X, y, mode, expected_rescale, expected_mode, expected_y_scale
):
    calls = []

    class FakeSymbolFit:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.func_candidates = CandidateTable([
                {'RMSE': 1.0, 'R2': 0.0, 'Parameterized equation, unscaled': 'x0'},
            ])

        def fit(self):
            return None

    package = types.ModuleType('symbolfit')
    module = types.ModuleType('symbolfit.symbolfit')
    module.SymbolFit = FakeSymbolFit
    monkeypatch.setitem(sys.modules, 'symbolfit', package)
    monkeypatch.setitem(sys.modules, 'symbolfit.symbolfit', module)
    reg = SymbolFitRegressor(fill_timeout_budget=False, input_rescale=True, scale_y_by=mode)
    monkeypatch.setattr(reg, '_build_pysr_config', lambda params: object())
    reg.fit(np.asarray(X), np.asarray(y))
    assert calls[0]['input_rescale'] is expected_rescale
    assert calls[0]['scale_y_by'] == expected_mode
    assert reg._coordinate_transform['input_rescale'] is expected_rescale
    assert reg._coordinate_transform['y_scale'] == pytest.approx(expected_y_scale)
    assert reg.params['input_rescale'] is True
    assert reg.params['scale_y_by'] == mode
