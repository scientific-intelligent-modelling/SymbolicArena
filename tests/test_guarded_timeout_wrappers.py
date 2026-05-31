import sys
import types

import numpy as np

from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import QLatticeRegressor
from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor
import scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper as fepysr_module
import scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper as symbolfit_module


def test_fepysr_uses_guarded_internal_pysr_timeout() -> None:
    reg = FePySRRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_symbolfit_uses_guarded_internal_pysr_timeout() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_symbolfit_short_smoke_budget_leaves_refit_and_write_guard() -> None:
    reg = SymbolFitRegressor(timeout_in_seconds=600)

    assert reg.params["timeout_in_seconds"] == 420


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
    assert any("pysr_params.timeout_in_seconds=2" in item for item in fit_calls[0])
    assert all(not any("pysr_params.random_state" in item for item in call) for call in fit_calls)
    assert reg.get_optimal_equation() == "X0"


def test_fepysr_current_best_snapshot_supports_timeout_recovery(tmp_path) -> None:
    exp_dir = tmp_path / "case"
    reg = FePySRRegressor(exp_path=str(tmp_path), exp_name="case", n_features=1)
    reg._best_equation = "X0"
    reg._equations = ["X0"]

    reg._write_current_best_snapshot(attempt=1, score=0.0)

    recovered = FePySRRegressor(existing_exp_dir=str(exp_dir), n_features=1)
    assert recovered.get_optimal_equation() == "X0"
    np.testing.assert_allclose(recovered.predict(np.array([[1.0], [2.0]])), np.array([1.0, 2.0]))


def test_fepysr_runtime_patch_decodes_bytes_equations(monkeypatch) -> None:
    feature_maker = types.ModuleType("fepysr.feature_maker")

    def replace_pysr_variables(pysr_equation, feature_names):
        assert isinstance(pysr_equation, str)
        return f"{pysr_equation}:{','.join(feature_names)}"

    feature_maker.replace_pysr_variables = replace_pysr_variables
    monkeypatch.setitem(sys.modules, "fepysr.feature_maker", feature_maker)

    FePySRRegressor._patch_fepysr_runtime()

    assert feature_maker.replace_pysr_variables(b"x0 + x1", ["x0", "x1"]) == "x0 + x1:x0,x1"


def test_fepysr_runtime_patch_sanitizes_nonfinite_features(monkeypatch) -> None:
    pysr_train_module = types.ModuleType("fepysr.pysr_train")
    fepysr_impl_module = types.ModuleType("fepysr.fepysr")
    observed = {}

    def pysr_train(data_analyzer, cfg, model=None):
        features = np.asarray(data_analyzer.stacked_numpy_features)
        observed["all_finite"] = bool(np.all(np.isfinite(features)))
        observed["max_abs"] = float(np.max(np.abs(features)))
        return "best", 0.0, 0.0, "model"

    pysr_train_module.pysr_train = pysr_train
    fepysr_impl_module.pysr_train = pysr_train
    monkeypatch.setitem(sys.modules, "fepysr.pysr_train", pysr_train_module)
    monkeypatch.setitem(sys.modules, "fepysr.fepysr", fepysr_impl_module)
    data_analyzer = types.SimpleNamespace(
        stacked_numpy_features=np.array([[np.inf, -np.inf, np.nan, 1.0e300, -2.0]])
    )

    FePySRRegressor._patch_fepysr_runtime()
    fepysr_impl_module.pysr_train(data_analyzer, None)

    assert observed == {"all_finite": True, "max_abs": FePySRRegressor._FEATURE_VALUE_LIMIT}


def test_qlattice_standardizes_large_target_and_restores_predictions(monkeypatch) -> None:
    observed = {}

    class FakeModel:
        def sympify(self, signif=4):
            return "x0"

        def predict(self, data):
            return np.asarray(data["x0"], dtype=float)

    class FakeQLattice:
        def auto_run(self, **kwargs):
            observed["target_mean"] = float(np.mean(kwargs["data"]["y"]))
            observed["target_std"] = float(np.std(kwargs["data"]["y"]))
            return [FakeModel()]

    monkeypatch.setitem(sys.modules, "feyn", types.SimpleNamespace(QLattice=FakeQLattice))

    y = np.asarray([1.0e9, 1.02e9, 1.04e9], dtype=float)
    reg = QLatticeRegressor(n_epochs=1, target_standardize="auto")
    reg.fit(np.asarray([[0.0], [1.0], [2.0]]), y)

    assert reg._target_was_standardized is True
    assert abs(observed["target_mean"]) < 1.0e-12
    assert abs(observed["target_std"] - 1.0) < 1.0e-12
    assert "x0" in reg.get_optimal_equation()
    expected = reg._target_offset + reg._target_scale * np.asarray([0.0, 1.0])
    np.testing.assert_allclose(reg.predict(np.asarray([[0.0], [1.0]])), expected)


def test_qlattice_empty_search_falls_back_to_mean_constant(monkeypatch, tmp_path) -> None:
    class FakeQLattice:
        def auto_run(self, **kwargs):
            return []

    monkeypatch.setitem(sys.modules, "feyn", types.SimpleNamespace(QLattice=FakeQLattice))

    reg = QLatticeRegressor(
        n_epochs=2,
        target_standardize=False,
        exp_path=str(tmp_path),
        exp_name="case",
    )
    reg.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([2.0, 4.0, 6.0]))

    assert reg.get_optimal_equation() == "4"
    assert reg.get_total_equations() == ["4"]
    np.testing.assert_allclose(reg.predict(np.asarray([[0.0], [1.0]])), np.asarray([4.0, 4.0]))
    assert (tmp_path / "case" / ".qlattice_current_best.json").exists()


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


def test_symbolfit_current_best_snapshot_supports_timeout_recovery(monkeypatch, tmp_path) -> None:
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
            return None

    symbolfit_pkg = types.ModuleType("symbolfit")
    symbolfit_submodule = types.ModuleType("symbolfit.symbolfit")
    symbolfit_submodule.SymbolFit = FakeSymbolFit
    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=FakePySRRegressor))
    monkeypatch.setitem(sys.modules, "symbolfit", symbolfit_pkg)
    monkeypatch.setitem(sys.modules, "symbolfit.symbolfit", symbolfit_submodule)

    reg = SymbolFitRegressor(exp_path=str(tmp_path), exp_name="case", n_features=1, timeout_in_seconds=10)
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    from scientific_intelligent_modelling.benchmarks import runner

    exp_dir = tmp_path / "case"
    assert (exp_dir / ".symbolfit_current_best.json").exists()
    recovered = runner._extract_periodic_candidate("symbolfit", exp_dir)
    assert recovered["equation"] == "X0"


def test_symbolfit_writes_active_pysr_work_dir_for_progress_snapshots(monkeypatch, tmp_path) -> None:
    observed_work_dirs = []

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
            observed_work_dirs.append(symbolfit_module.Path.cwd())
            return None

    symbolfit_pkg = types.ModuleType("symbolfit")
    symbolfit_submodule = types.ModuleType("symbolfit.symbolfit")
    symbolfit_submodule.SymbolFit = FakeSymbolFit
    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=FakePySRRegressor))
    monkeypatch.setitem(sys.modules, "symbolfit", symbolfit_pkg)
    monkeypatch.setitem(sys.modules, "symbolfit.symbolfit", symbolfit_submodule)

    reg = SymbolFitRegressor(exp_path=str(tmp_path), exp_name="case", n_features=1, timeout_in_seconds=10)
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    exp_dir = tmp_path / "case"
    active = (exp_dir / ".symbolfit_active_run.json").read_text(encoding="utf-8")
    assert "symbolfit_work" in active
    assert observed_work_dirs
    assert exp_dir / "symbolfit_work" in observed_work_dirs[0].parents


def test_symbolfit_short_budget_uses_external_deadline_with_guarded_inner_runs(monkeypatch) -> None:
    clock = {"now": 0.0}
    inner_timeouts = []

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
            inner_timeouts.append(self.kwargs["pysr_config"].kwargs["timeout_in_seconds"])
            clock["now"] += 421.0 if len(inner_timeouts) == 1 else 120.0

    symbolfit_pkg = types.ModuleType("symbolfit")
    symbolfit_submodule = types.ModuleType("symbolfit.symbolfit")
    symbolfit_submodule.SymbolFit = FakeSymbolFit
    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=FakePySRRegressor))
    monkeypatch.setitem(sys.modules, "symbolfit", symbolfit_pkg)
    monkeypatch.setitem(sys.modules, "symbolfit.symbolfit", symbolfit_submodule)
    monkeypatch.setattr(symbolfit_module.time, "monotonic", lambda: clock["now"])

    reg = SymbolFitRegressor(timeout_in_seconds=600)
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert inner_timeouts[0] == 420
    assert len(inner_timeouts) >= 2
