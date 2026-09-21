import importlib.util
from pathlib import Path
import types
import signal

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def load_utils():
    path = (Path(__file__).resolve().parents[1] /
            'scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/utils_wrapper.py')
    spec = importlib.util.spec_from_file_location('e2esr_refinement_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(column_output=True):
    def to_module(tree, dtype):
        def evaluate(x, constants):
            values = x[:, 0] * constants[0]
            return values[:, None] if column_output else values
        return evaluate
    return types.SimpleNamespace(simplifier=types.SimpleNamespace(tree_to_torch_module=to_module),
                                 wrap_equation_floats=lambda tree, constants: np.asarray(constants))


@pytest.mark.parametrize('column_target', [True, False])
def test_bfgs_objective_matches_each_row_not_all_pairs(monkeypatch, column_target):
    module = load_utils()
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = 2.0 * X[:, 0]
    if column_target:
        y = y[:, None]
    objectives = []

    def minimize(fun, x0, **kwargs):
        objectives.append(fun(np.array([2.0])))

    monkeypatch.setattr(module, 'minimize', minimize)
    module.BFGSRefinement().go(environment(), object(), [2.0], X, y)
    assert objectives == [0.0]


def test_bfgs_recovers_correct_slope_for_column_predictions():
    module = load_utils()
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    result = module.BFGSRefinement().go(environment(), object(), [0.5], X, 2.0 * X[:, 0])
    np.testing.assert_allclose(result, [2.0], rtol=1e-5)


def test_timed_objective_does_not_replace_initial_best_with_worse_point():
    module = load_utils()
    objective = module.TimedFun(lambda x: float(x[0]**2))
    initial = np.array([0.1])
    objective.fun(initial)
    initial[0] = 999.0
    objective.fun(np.array([2.0]))
    np.testing.assert_allclose(objective.best_x, [0.1])


def test_scaler_projects_out_of_range_variables_without_looping():
    module = load_utils()
    env = types.SimpleNamespace(word_to_infix=lambda prefix, **kwargs: prefix)
    tree = types.SimpleNamespace(prefix=lambda: 'add,x_0,x_12')

    def expired(signum, frame):
        raise TimeoutError('variable projection did not terminate')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        prefix = module.StandardScaler().rescale_function(env, tree, [2.0], [3.0])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert prefix == ['add', 'add', '3.0', 'mul', '2.0', 'x_0', '0']


def test_scaler_uses_full_variable_index():
    module = load_utils()
    env = types.SimpleNamespace(word_to_infix=lambda prefix, **kwargs: prefix)
    tree = types.SimpleNamespace(prefix=lambda: 'x_10')
    prefix = module.StandardScaler().rescale_function(env, tree, np.arange(1., 12.), np.zeros(11))
    assert prefix == ['add', '0.0', 'mul', '11.0', 'x_10']


def test_constant_feature_scaling_matches_sklearn():
    module = load_utils()
    scaler = module.StandardScaler()
    X = np.array([[2., 5.], [4., 5.], [6., 5.]])
    transformed = scaler.fit_transform(X)
    a, b = scaler.get_params()
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(transformed, X*a+b, atol=1e-15)
    np.testing.assert_allclose(scaler.transform(X), transformed)
