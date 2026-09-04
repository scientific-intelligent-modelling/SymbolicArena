from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest


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
    progress = json.loads(
        (tmp_path / "jaxsr_budget" / ".jaxsr_current_best.json").read_text(encoding="utf-8")
    )
    assert progress["equation"] == "x0"
    assert progress["loss"] == pytest.approx(0.0)
    assert progress["internal_loss"] == pytest.approx(progress["loss"])
    assert progress["source"] == ".jaxsr_current_best.json"
    assert progress["first_discovered_attempt"] == 1
    assert progress["first_discovered_minute"] == 1
    assert progress["first_discovered_elapsed_seconds"] == pytest.approx(4.0)
    assert progress["fidelity"]["status"] == "verified"
    assert progress["fidelity"]["equation_source"] == "model.to_sympy"
    assert progress["model_state"] == {"random_state": 10}


@pytest.mark.parametrize(
    ("raw_equation", "coefficient", "probe_values"),
    [
        ("0", 5.0e-11, [-10_000.0, -1_000.0, 1_000.0, 10_000.0]),
        ("4.0e-11*x0**3", 5.0e-11, [-10_000.0, -1_000.0, 1_000.0, 10_000.0]),
        ("0", 1.0e-11, [-1.0, -0.5, 0.5, 1.0]),
    ],
)
def test_jaxsr_repairs_lossy_sympy_export_from_fitted_state_and_freezes_fidelity(
    monkeypatch,
    tmp_path,
    raw_equation,
    coefficient,
    probe_values,
):
    """覆盖审计中的零公式和非零但数值失真的两类导出。"""
    from scientific_intelligent_modelling.algorithms.jaxsr_wrapper import wrapper as jaxsr_wrapper

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

    class FakeResult:
        coefficients = np.asarray([coefficient])
        selected_names = ["x0^3"]

    class FakeSymbolicRegressor:
        def __init__(self, *args, **kwargs):
            self._result = FakeResult()

        def fit(self, X, y):
            return self

        def predict(self, X):
            values = np.asarray(X, dtype=float).reshape(-1, 1)[:, 0]
            return coefficient * values**3

        def to_sympy(self):
            return raw_equation

        def _parse_basis_to_sympy(self, name, symbols):
            return symbols["x0"] ** 3

        def _state_dict(self):
            return {
                "result": {
                    "coefficients": [coefficient],
                    "selected_names": ["x0^3"],
                }
            }

        @classmethod
        def _from_dict(cls, state):
            return cls()

    fake_jaxsr = types.SimpleNamespace(
        BasisLibrary=FakeBasisLibrary,
        SymbolicRegressor=FakeSymbolicRegressor,
    )
    monkeypatch.setitem(sys.modules, "jaxsr", fake_jaxsr)

    X = np.asarray(probe_values, dtype=float).reshape(-1, 1)
    y = coefficient * X[:, 0] ** 3
    reg = jaxsr_wrapper.JAXSRRegressor(
        n_features=1,
        feature_names=["x0"],
        target_name="y",
        exp_path=str(tmp_path),
        exp_name="lossy_export",
    ).fit(X, y)

    equation = reg.get_optimal_equation()
    assert equation != "0"
    replay = np.asarray(
        __import__("sympy").lambdify([__import__("sympy").Symbol("x0")], equation, "numpy")(
            X[:, 0]
        ),
        dtype=float,
    ).reshape(-1)
    np.testing.assert_allclose(replay, reg.predict(X), rtol=1e-12, atol=0.0)

    payload = json.loads(reg.serialize())
    assert payload["equation"] == equation
    assert payload["fidelity"]["status"] == "verified"
    assert payload["fidelity"]["equation_source"] == "model_state"
    assert payload["fidelity"]["raw_equation"] == raw_equation
    assert len(payload["fidelity"]["probe_input_sha256"]) == 64
    assert len(payload["fidelity"]["native_prediction_sha256"]) == 64
    assert len(payload["fidelity"]["replay_prediction_sha256"]) == 64
    assert payload["fidelity"]["max_abs_error"] == pytest.approx(0.0)
    assert payload["fidelity"]["probe_definition"]["values"]
    artifact = reg.export_canonical_symbolic_program()
    assert artifact["raw_equation"] == equation
    assert artifact["fidelity_check"] == payload["fidelity"]

    progress = json.loads(
        (tmp_path / "lossy_export" / ".jaxsr_current_best.json").read_text(encoding="utf-8")
    )
    assert progress["equation"] == equation
    assert progress["fidelity"]["status"] == "verified"

    restored = jaxsr_wrapper.JAXSRRegressor.deserialize(reg.serialize())
    assert restored.get_optimal_equation() == equation
    np.testing.assert_allclose(restored.predict(X), reg.predict(X))

    payload["equation"] = "0"
    with pytest.raises(RuntimeError, match="证据哈希不一致"):
        jaxsr_wrapper.JAXSRRegressor.deserialize(json.dumps(payload))


def test_jaxsr_fails_closed_when_native_model_cannot_be_replayed(monkeypatch, tmp_path):
    from scientific_intelligent_modelling.algorithms.jaxsr_wrapper import wrapper as jaxsr_wrapper

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

    class OpaqueSymbolicRegressor:
        expression_ = "0"

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return 2.0 * np.asarray(X, dtype=float).reshape(-1, 1)[:, 0]

        def to_sympy(self):
            return "0"

        def _state_dict(self):
            return {"opaque": True}

    fake_jaxsr = types.SimpleNamespace(
        BasisLibrary=FakeBasisLibrary,
        SymbolicRegressor=OpaqueSymbolicRegressor,
    )
    monkeypatch.setitem(sys.modules, "jaxsr", fake_jaxsr)

    reg = jaxsr_wrapper.JAXSRRegressor(
        exp_path=str(tmp_path),
        exp_name="opaque_export",
    ).fit(np.asarray([[1.0], [2.0], [3.0]]), np.asarray([2.0, 4.0, 6.0]))

    with pytest.raises(RuntimeError, match="fidelity"):
        reg.get_optimal_equation()
    with pytest.raises(RuntimeError, match="fidelity"):
        reg.export_canonical_symbolic_program()

    payload = json.loads(reg.serialize())
    assert payload["equation"] is None
    assert payload["fidelity"]["status"] == "failed"
    assert payload["fidelity"]["raw_equation"] == "0"
    assert payload["state"] == {"opaque": True}

    progress = json.loads(
        (tmp_path / "opaque_export" / ".jaxsr_current_best.json").read_text(encoding="utf-8")
    )
    assert progress["equation"] is None
    assert progress["fidelity"]["status"] == "failed"


def test_jaxsr_timeout_snapshot_invalidates_stale_faithful_candidate(monkeypatch, tmp_path):
    """更优 native 候选若无法忠实导出，timeout 恢复不得捡回旧公式。"""
    from scientific_intelligent_modelling.algorithms.jaxsr_wrapper import wrapper as jaxsr_wrapper

    clock = {"now": 0.0}
    model_count = {"value": 0}

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

    class Candidate:
        def __init__(self, *args, **kwargs):
            self.index = model_count["value"]
            model_count["value"] += 1

        def fit(self, X, y):
            clock["now"] += 4.0
            return self

        def predict(self, X):
            scale = 1.0 if self.index == 0 else 2.0
            return scale * np.asarray(X, dtype=float).reshape(-1, 1)[:, 0]

        def to_sympy(self):
            return "x0" if self.index == 0 else "0"

        def _state_dict(self):
            return {"candidate_index": self.index}

    monkeypatch.setitem(
        sys.modules,
        "jaxsr",
        types.SimpleNamespace(BasisLibrary=FakeBasisLibrary, SymbolicRegressor=Candidate),
    )
    monkeypatch.setattr(jaxsr_wrapper.time, "time", lambda: clock["now"])

    X = np.asarray([[1.0], [2.0], [3.0]])
    reg = jaxsr_wrapper.JAXSRRegressor(
        seed=1,
        timeout_in_seconds=7,
        timeout_guard_seconds=1,
        exp_path=str(tmp_path),
        exp_name="stale_snapshot",
    ).fit(X, 2.0 * X[:, 0])

    progress = json.loads(
        (tmp_path / "stale_snapshot" / ".jaxsr_current_best.json").read_text(encoding="utf-8")
    )
    assert progress["iteration"] == 2
    assert progress["internal_loss"] == pytest.approx(progress["loss"])
    assert progress["source"] == ".jaxsr_current_best.json"
    assert progress["first_discovered_attempt"] == 2
    assert progress["first_discovered_minute"] == 1
    assert progress["first_discovered_elapsed_seconds"] == pytest.approx(8.0)
    assert progress["equation"] is None
    assert progress["fidelity"]["status"] == "failed"
    assert progress["model_state"] == {"candidate_index": 1}
    with pytest.raises(RuntimeError, match="fail-closed"):
        reg.get_optimal_equation()
