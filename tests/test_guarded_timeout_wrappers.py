import sys
import types

import numpy as np

from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor
import scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper as fepysr_module
import scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper as symbolfit_module


def test_fepysr_uses_guarded_internal_pysr_timeout() -> None:
    reg = FePySRRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_symbolfit_uses_guarded_internal_pysr_timeout() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_explicit_timeout_guard_overrides_default_guard() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=3600, timeout_guard_seconds=120)

    assert reg.params["timeout_in_seconds"] == 3480


def test_fepysr_repeats_successful_fit_until_timeout_budget(monkeypatch) -> None:
    clock = {"now": 0.0}
    fit_calls = []

    class FakeTorch:
        float64 = "float64"

        @staticmethod
        def as_tensor(value, dtype=None):
            return np.asarray(value, dtype=float)

    class FakeFePySR:
        def __init__(self, overrides, custom_pysr_model=None):
            self.overrides = list(overrides)
            self.best_equation_ = "X0"

        def fit(self, X, y):
            fit_calls.append(self.overrides)
            clock["now"] += 3.0

        def predict(self, X):
            return np.asarray(X)[:, 0]

    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "fepysr", types.SimpleNamespace(FePySR=FakeFePySR))
    monkeypatch.setattr(fepysr_module.time, "monotonic", lambda: clock["now"])

    reg = FePySRRegressor(timeout_in_seconds=10)
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert len(fit_calls) >= 2
    assert all(not any("pysr_params.random_state" in item for item in call) for call in fit_calls)
    assert reg.get_optimal_equation() == "X0"


def test_symbolfit_repeats_successful_fit_until_timeout_budget(monkeypatch) -> None:
    clock = {"now": 0.0}
    fit_calls = []

    class FakePySRRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTable:
        def __len__(self):
            return 1

        def iterrows(self):
            yield 0, {
                "RMSE": 1.0,
                "R2": 0.0,
                "Parameterized equation": "X0",
            }

    class FakeSymbolFit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.func_candidates = FakeTable()

        def fit(self):
            fit_calls.append(self.kwargs)
            clock["now"] += 3.0

    symbolfit_pkg = types.ModuleType("symbolfit")
    symbolfit_submodule = types.ModuleType("symbolfit.symbolfit")
    symbolfit_submodule.SymbolFit = FakeSymbolFit
    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=FakePySRRegressor))
    monkeypatch.setitem(sys.modules, "symbolfit", symbolfit_pkg)
    monkeypatch.setitem(sys.modules, "symbolfit.symbolfit", symbolfit_submodule)
    monkeypatch.setattr(symbolfit_module.time, "monotonic", lambda: clock["now"])

    reg = SymbolFitRegressor(timeout_in_seconds=10)
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert len(fit_calls) >= 2
    assert reg.get_optimal_equation() == "X0"
