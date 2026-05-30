from __future__ import annotations

import sys
import types

import numpy as np


def test_jaxsr_repeats_effective_fits_until_timeout_budget(monkeypatch, tmp_path):
    from scientific_intelligent_modelling.algorithms.jaxsr_wrapper import wrapper as jaxsr_wrapper

    clock = {"now": 0.0}
    fitted_random_states: list[int] = []

    class FakeBasisLibrary:
        def __init__(self, *args, **kwargs):
            pass

        def add_constant(self):
            return self

        def add_linear(self):
            return self

        def add_polynomials(self, *args, **kwargs):
            return self

        def add_interactions(self, *args, **kwargs):
            return self

    class FakeSymbolicRegressor:
        def __init__(self, *args, **kwargs):
            self.random_state = kwargs.get("random_state")
            self.expression_ = "x0"

        def fit(self, X, y):
            fitted_random_states.append(int(self.random_state))
            clock["now"] += 4.0
            return self

        def predict(self, X):
            return np.asarray(X, dtype=float).reshape(-1, 1)[:, 0]

        def to_sympy(self):
            return "x0"

        def _state_dict(self):
            return {"random_state": self.random_state}

    fake_jaxsr = types.SimpleNamespace(
        BasisLibrary=FakeBasisLibrary,
        SymbolicRegressor=FakeSymbolicRegressor,
    )
    monkeypatch.setitem(sys.modules, "jaxsr", fake_jaxsr)
    monkeypatch.setattr(jaxsr_wrapper.time, "time", lambda: clock["now"])

    reg = jaxsr_wrapper.JAXSRRegressor(
        seed=10,
        timeout_in_seconds=10,
        timeout_guard_seconds=1,
        exp_path=str(tmp_path),
        exp_name="jaxsr_budget",
    )
    reg.fit(np.asarray([[1.0], [2.0], [3.0]]), np.asarray([1.0, 2.0, 3.0]))

    assert fitted_random_states == [10, 11, 12]
    assert (tmp_path / "jaxsr_budget" / ".jaxsr_current_best.json").exists()
