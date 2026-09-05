import builtins
import json
import sys
import types

import numpy as np
import pytest

from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor
from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import QLatticeRegressor
from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor
import scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper as fepysr_module
import scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper as symbolfit_module


def test_fepysr_uses_guarded_internal_pysr_timeout() -> None:
    reg = FePySRRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300


def test_fepysr_fill_deadline_uses_full_explicit_timeout(monkeypatch) -> None:
    monkeypatch.setattr(fepysr_module.time, "monotonic", lambda: 100.0)

    reg = FePySRRegressor(timeout_in_seconds=3600)

    assert reg.params["timeout_in_seconds"] == 3300
    assert reg._budget_deadline() == 3700.0


def test_fepysr_fit_attempt_timeout_accounts_for_nested_experiments() -> None:
    params = {
        "timeout_in_seconds": 10500,
        "num_experiments": 8,
    }

    assert FePySRRegressor._fit_attempt_timeout_seconds(params, 10800) == 150


def test_fepysr_bootstrap_attempt_uses_small_fmn_search() -> None:
    params = {
        "timeout_in_seconds": 150,
        "num_experiments": 8,
        "num_workers": 4,
        "fmn_epochs": 30,
    }

    FePySRRegressor._apply_bootstrap_attempt_params(
        params,
        attempt=1,
        has_best_equation=False,
        remaining=10800,
    )

    assert params["num_experiments"] == 1
    assert params["num_workers"] == 1
    assert params["fmn_epochs"] == 5
    assert params["timeout_in_seconds"] == 150


def test_fepysr_bootstrap_repeats_until_candidate_exists() -> None:
    params = {
        "timeout_in_seconds": 150,
        "num_experiments": 8,
        "num_workers": 4,
        "fmn_epochs": 30,
    }

    FePySRRegressor._apply_bootstrap_attempt_params(
        params,
        attempt=2,
        has_best_equation=False,
        remaining=10800,
    )

    assert params["num_experiments"] == 1
    assert params["num_workers"] == 1
    assert params["fmn_epochs"] == 5


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

    reg = FePySRRegressor(timeout_in_seconds=10, num_experiments=1)
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


def test_fepysr_empty_search_writes_mean_constant_baseline(monkeypatch, tmp_path) -> None:
    clock = {"now": 0.0}

    class FakeTorch:
        float64 = "float64"

        @staticmethod
        def as_tensor(value, dtype=None):
            return np.asarray(value, dtype=float)

    class FakeFePySR:
        best_equation_ = ""

        def __init__(self, overrides, custom_pysr_model=None):
            self.overrides = overrides

        def fit(self, X, y):
            clock["now"] += 10.0

    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "fepysr", types.SimpleNamespace(FePySR=FakeFePySR))
    monkeypatch.setattr(fepysr_module.time, "monotonic", lambda: clock["now"])

    reg = FePySRRegressor(
        timeout_in_seconds=10,
        num_experiments=1,
        exp_path=str(tmp_path),
        exp_name="case",
    )
    reg.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([2.0, 4.0, 6.0]))

    assert reg.get_optimal_equation() == "4"
    assert reg.get_total_equations() == ["4"]
    np.testing.assert_allclose(reg.predict(np.asarray([[0.0], [1.0]])), np.asarray([4.0, 4.0]))
    assert (tmp_path / "case" / ".fepysr_current_best.json").exists()


def test_fepysr_writes_mean_baseline_before_fepysr_import(monkeypatch, tmp_path) -> None:
    class FakeTorch:
        float64 = "float64"

        @staticmethod
        def as_tensor(value, dtype=None):
            return np.asarray(value, dtype=float)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fepysr":
            raise ModuleNotFoundError("blocked fepysr import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    reg = FePySRRegressor(
        timeout_in_seconds=10,
        exp_path=str(tmp_path),
        exp_name="case",
    )

    with pytest.raises(ImportError):
        reg.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([2.0, 4.0, 6.0]))

    recovered = FePySRRegressor(existing_exp_dir=str(tmp_path / "case"), n_features=1)
    assert recovered.get_optimal_equation() == "4"


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


def test_qlattice_incremental_search_latches_global_criterion_best(monkeypatch, tmp_path) -> None:
    class FakeModel:
        def __init__(self, equation: str, bic: float, multiplier: float) -> None:
            self._equation = equation
            self.bic = bic
            self._multiplier = multiplier

        def sympify(self, signif=4):
            return self._equation

        def predict(self, data):
            return self._multiplier * np.asarray(data["x0"], dtype=float)

    epoch_models = iter(
        [
            [FakeModel("x0", 10.0, 1.0)],
            [FakeModel("2*x0", 2.0, 2.0)],
            [FakeModel("3*x0", 5.0, 3.0)],
        ]
    )

    class FakeQLattice:
        def auto_run(self, **kwargs):
            return next(epoch_models)

    monkeypatch.setitem(sys.modules, "feyn", types.SimpleNamespace(QLattice=FakeQLattice))

    reg = QLatticeRegressor(
        n_epochs=3,
        target_standardize=False,
        exp_path=str(tmp_path),
        exp_name="case",
    )
    X = np.asarray([[0.0], [1.0], [2.0]])
    reg.fit(X, 2.0 * X[:, 0])

    assert reg.get_optimal_equation() == "2*x0"
    np.testing.assert_allclose(reg.predict(X), 2.0 * X[:, 0])
    progress = json.loads(
        (tmp_path / "case" / ".qlattice_current_best.json").read_text(encoding="utf-8")
    )
    assert progress["equation"] == "2*x0"
    assert progress["loss"] == 2.0
    assert progress["internal_loss"] == 2.0
    assert progress["epoch"] == 2


def test_qlattice_export_preserves_native_zero_based_feature_semantics(monkeypatch) -> None:
    """导出的 canonical artifact 应与 QLattice 原生 x1 预测使用同一列。"""
    from scientific_intelligent_modelling.benchmarks.runner import _predict_from_canonical_artifact

    class FakeModel:
        bic = 0.0

        def sympify(self, signif=4):
            return "x1"

        def predict(self, data):
            return np.asarray(data["x1"], dtype=float)

    class FakeQLattice:
        def auto_run(self, **kwargs):
            return [FakeModel()]

    monkeypatch.setitem(sys.modules, "feyn", types.SimpleNamespace(QLattice=FakeQLattice))

    X = np.asarray([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    reg = QLatticeRegressor(n_epochs=1, target_standardize=False)
    reg.fit(X, X[:, 1])

    artifact = reg.export_canonical_symbolic_program()
    native_prediction = reg.predict(X)
    canonical_prediction = _predict_from_canonical_artifact(artifact, X)

    assert artifact["normalized_expression"] == "x1"
    np.testing.assert_allclose(native_prediction, X[:, 1])
    np.testing.assert_allclose(canonical_prediction, native_prediction)


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


def test_symbolfit_active_snapshot_records_coordinate_transform(monkeypatch, tmp_path) -> None:
    """active run 元数据应让 runner 能还原 PySR 的缩放坐标。"""
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
                "PySR loss": 0.25,
                "PySR equation": "X0",
                "Parameterized equation, unscaled": "X0",
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

    reg = SymbolFitRegressor(
        exp_path=str(tmp_path),
        exp_name="coordinate_metadata",
        n_features=1,
        timeout_in_seconds=10,
        input_rescale=True,
        scale_y_by="mean",
    )
    reg.fit(np.asarray([[10.0], [20.0], [30.0]]), np.asarray([10.0, 20.0, 30.0]))

    active = json.loads(
        (tmp_path / "coordinate_metadata" / ".symbolfit_active_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert active["input_rescale"] is True
    assert active["x_min"] == [10.0]
    assert active["x_max"] == [30.0]
    assert active["y_scale"] == 0.05
    search_best = json.loads(
        (tmp_path / "coordinate_metadata" / ".symbolfit_search_best.json").read_text(
            encoding="utf-8"
        )
    )
    assert search_best["internal_loss"] == 0.25
    assert search_best["scaled_equation"] == "X0"


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


def test_symbolfit_postfit_history_uses_ceil_minute_and_exact_elapsed(
    monkeypatch, tmp_path
) -> None:
    class FakeTable:
        def iterrows(self):
            yield 0, {"PySR equation": "X0", "PySR loss": 0.25, "Complexity": 1}

    model = types.SimpleNamespace(func_candidates=FakeTable())
    reg = SymbolFitRegressor(
        exp_path=str(tmp_path),
        exp_name="strict_discovery_time",
        timeout_in_seconds=10800,
    )
    reg._fit_started_at = 100.0
    reg._coordinate_transform = {"input_rescale": False, "y_scale": 1.0}
    monkeypatch.setattr(symbolfit_module.time, "monotonic", lambda: 10900.25)

    reg._record_internal_candidates(model, attempt=1, attempt_started_at=100.0)

    history_path = tmp_path / "strict_discovery_time" / ".symbolfit_pysr_candidates.jsonl"
    row = json.loads(history_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["first_discovered_minute"] == 181
    assert row["first_discovered_elapsed_seconds"] == 10800.25


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
