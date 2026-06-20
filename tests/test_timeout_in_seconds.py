import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor
from scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper import OperonRegressor
from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor
from scientific_intelligent_modelling.srkit import subprocess_runner
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def test_strict_wrappers_accept_timeout_meta_param():
    pysr = PySRRegressor(timeout_in_seconds=7, niterations=5, population_size=16)
    assert pysr.params["timeout_in_seconds"] == 7

    operon = OperonRegressor(timeout_in_seconds=9.2, niterations=4, population_size=16, allowed_symbols="add,sub,mul,div")
    assert operon.params["max_time"] == 9
    assert operon.params["generations"] == 4
    assert operon.params["allowed_symbols"] == "add,sub,mul,div,constant,variable"

    gplearn = GPLearnRegressor(timeout_in_seconds=11, generations=3, population_size=32)
    assert "timeout_in_seconds" not in gplearn.params

    dso = DSORegressor(timeout_in_seconds=13)
    assert "timeout_in_seconds" not in dso.params


def test_pyoperon_progress_loop_keeps_max_time_as_int(monkeypatch, tmp_path):
    seen_max_time_types = []

    class FakeSymbolicRegressor:
        def __init__(self, **kwargs):
            self.max_time = kwargs.get("max_time")
            self.generations = kwargs.get("generations")
            self.warm_start = kwargs.get("warm_start", False)
            self.model_ = "X1"
            self.stats_ = {"model_complexity": 1}
            self.pareto_front_ = [{"model": "X1", "minimum_description_length": 0.0, "mean_squared_error": 0.0}]

        def fit(self, X, y):
            seen_max_time_types.append(type(self.max_time))
            assert isinstance(self.max_time, int)

        def predict(self, X):
            return np.asarray(X)[:, 0]

        def get_model_string(self, model):
            return "X1"

    pyoperon_module = types.ModuleType("pyoperon")
    sklearn_module = types.ModuleType("pyoperon.sklearn")
    sklearn_module.SymbolicRegressor = FakeSymbolicRegressor
    monkeypatch.setitem(sys.modules, "pyoperon", pyoperon_module)
    monkeypatch.setitem(sys.modules, "pyoperon.sklearn", sklearn_module)

    reg = OperonRegressor(
        exp_path=str(tmp_path),
        exp_name="case",
        timeout_in_seconds=2,
        niterations=1,
        allowed_symbols="add,mul,constant,variable",
    )
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert seen_max_time_types == [int]


def test_pyoperon_progress_loop_ignores_implicit_generation_cap(monkeypatch, tmp_path):
    fit_calls = []
    fake_clock = {"now": 0.0}

    class FakeSymbolicRegressor:
        def __init__(self, **kwargs):
            self.max_time = kwargs.get("max_time")
            self.generations = kwargs.get("generations")
            self.warm_start = kwargs.get("warm_start", False)
            self.model_ = "X1"
            self.stats_ = {"model_complexity": 1}
            self.pareto_front_ = [{"model": "X1", "minimum_description_length": 0.0, "mean_squared_error": 0.0}]

        def fit(self, X, y):
            fit_calls.append(self.max_time)
            fake_clock["now"] += 1.0

        def predict(self, X):
            return np.asarray(X)[:, 0]

        def get_model_string(self, model):
            return "X1"

    pyoperon_module = types.ModuleType("pyoperon")
    sklearn_module = types.ModuleType("pyoperon.sklearn")
    sklearn_module.SymbolicRegressor = FakeSymbolicRegressor
    monkeypatch.setitem(sys.modules, "pyoperon", pyoperon_module)
    monkeypatch.setitem(sys.modules, "pyoperon.sklearn", sklearn_module)
    monkeypatch.setattr(
        "scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper.time.time",
        lambda: fake_clock["now"],
    )

    reg = OperonRegressor(
        exp_path=str(tmp_path),
        exp_name="case",
        timeout_in_seconds=3,
        allowed_symbols="add,mul,constant,variable",
    )
    reg.params["generations"] = 1
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert len(fit_calls) == 3


def test_gplearn_progress_loop_stops_when_wall_time_budget_is_exhausted(monkeypatch, tmp_path):
    fit_generations = []
    fake_clock = {"now": 0.0}

    class FakeSymbolicRegressor:
        def __init__(self, **kwargs):
            self.generations = kwargs.get("generations")
            self.warm_start = kwargs.get("warm_start", False)
            self._program = "X0"
            self.run_details_ = {
                "generation": [0],
                "best_fitness": [1.0],
                "best_length": [1],
                "generation_time": [0.1],
            }
            self.n_features_in_ = 1

        def fit(self, X, y):
            fit_generations.append(self.generations)
            fake_clock["now"] += 10.0
            self._program = f"X0_gen_{self.generations}"
            self.run_details_["generation"] = [self.generations - 1]
            return self

        def predict(self, X):
            return np.asarray(X)[:, 0]

    gplearn_module = types.ModuleType("gplearn")
    genetic_module = types.ModuleType("gplearn.genetic")
    genetic_module.SymbolicRegressor = FakeSymbolicRegressor
    monkeypatch.setitem(sys.modules, "gplearn", gplearn_module)
    monkeypatch.setitem(sys.modules, "gplearn.genetic", genetic_module)
    monkeypatch.setattr(
        "scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper.time.time",
        lambda: fake_clock["now"],
    )

    reg = GPLearnRegressor(
        exp_path=str(tmp_path),
        exp_name="case",
        timeout_in_seconds=15,
        generations=1000000,
        population_size=4,
    )
    reg.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    assert fit_generations == [1, 2]
    assert (tmp_path / "case" / ".gplearn_current_best.json").exists()


def test_dso_build_fit_config_preserves_logdir_and_disables_gp_meld():
    base_config = {
        "experiment": {"logdir": "/tmp/dso-logdir", "exp_name": "case"},
        "task": {"task_type": "regression"},
        "gp_meld": {"run_gp_meld": True},
    }
    cfg = DSORegressor._build_fit_config(base_config, [[1.0, 2.0]], [3.0])
    assert cfg["experiment"]["logdir"] == "/tmp/dso-logdir"
    assert cfg["task"]["dataset"].startswith("/tmp/e1tmp/dso_data/case__train_")
    assert cfg["task"]["dataset"].endswith(".csv")
    assert cfg["gp_meld"]["run_gp_meld"] is False


def test_subprocess_runner_handle_fit_strips_timeout_meta_param():
    seen = {}

    class DummyRegressor:
        def __init__(self, **kwargs):
            seen["kwargs"] = dict(kwargs)

        def fit(self, X, y):
            seen["shape"] = (tuple(X.shape), tuple(y.shape))

        def serialize(self):
            return "dummy-model"

    result = subprocess_runner.handle_fit(
        DummyRegressor,
        {
            "tool_name": "dummy",
            "data": {"X": [[1.0], [2.0]], "y": [3.0, 4.0]},
            "params": {"timeout_in_seconds": 5, "alpha": 1},
        },
    )

    assert result["serialized_model"] == "dummy-model"
    assert seen["kwargs"] == {"alpha": 1}
    assert seen["shape"] == ((2, 1), (2,))


def test_subprocess_runner_forwards_timeout_to_project_wrappers():
    seen = {}

    class DummyProjectRegressor:
        __module__ = "scientific_intelligent_modelling.algorithms.fake_wrapper.wrapper"

        def __init__(self, **kwargs):
            seen["kwargs"] = dict(kwargs)

        def fit(self, X, y):
            seen["shape"] = (tuple(X.shape), tuple(y.shape))

        def serialize(self):
            return "dummy-model"

    result = subprocess_runner.handle_fit(
        DummyProjectRegressor,
        {
            "tool_name": "dummy",
            "data": {"X": [[1.0], [2.0]], "y": [3.0, 4.0]},
            "params": {"timeout_in_seconds": 5, "alpha": 1},
        },
    )

    assert result["serialized_model"] == "dummy-model"
    assert seen["kwargs"] == {"timeout_in_seconds": 5, "alpha": 1}
    assert seen["shape"] == ((2, 1), (2,))


def test_symbolic_regressor_timeout_updates_manifest_when_recovery_fails(monkeypatch):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )

    def fake_execute(self, command):
        if command["action"] == "fit":
            raise TimeoutError("boom")
        raise RuntimeError("recover failed")

    monkeypatch.setattr(SymbolicRegressor, "_execute_subprocess", fake_execute)

    reg = SymbolicRegressor("gplearn", timeout_in_seconds=3, generations=2, population_size=8)

    with pytest.raises(TimeoutError):
        reg.fit(np.array([[1.0], [2.0]]), np.array([3.0, 4.0]))

    manifest_path = Path(reg.experiment_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "timed_out"
    assert manifest["timeout_in_seconds"] == 3


def test_symbolic_regressor_timeout_with_recovery_marks_success(monkeypatch):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )
    seen_actions = []

    def fake_execute(self, command):
        seen_actions.append(command["action"])
        if command["action"] == "fit":
            raise TimeoutError("boom")
        if command["action"] == "recover_from_timeout":
            assert command["experiment_dir"] == reg.experiment_dir
            return {"serialized_model": "rescued-model"}
        raise AssertionError(f"unexpected action: {command['action']}")

    monkeypatch.setattr(SymbolicRegressor, "_execute_subprocess", fake_execute)

    reg = SymbolicRegressor("gplearn", timeout_in_seconds=3, generations=2, population_size=8)
    fitted = reg.fit(np.array([[1.0], [2.0]]), np.array([3.0, 4.0]))

    assert fitted is reg
    assert reg.serialized_model == "rescued-model"
    assert seen_actions == ["fit", "recover_from_timeout"]

    manifest_path = Path(reg.experiment_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["timeout_in_seconds"] == 3
    assert manifest["recovered_from_timeout"] is True


def test_symbolic_regressor_prefers_explicit_fit_timeout(monkeypatch):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )
    reg = SymbolicRegressor("gplearn", timeout_in_seconds=12)
    timeout = reg._resolve_subprocess_timeout_seconds(
        {"action": "fit", "params": {"timeout_in_seconds": 12}}
    )
    assert timeout == 12


def test_symbolic_regressor_reuses_explicit_exp_path_and_exp_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )

    exp_root = tmp_path / "experiments"
    reg = SymbolicRegressor(
        "gplearn",
        problem_name="oscillator1",
        seed=1316,
        exp_path=str(exp_root),
        exp_name="oscillator1_gplearn_seed1316",
    )

    expected_dir = Path(reg.experiment_dir)
    assert expected_dir.parent == exp_root
    assert expected_dir.name.startswith("oscillator1_gplearn_seed1316_")
    manifest_path = expected_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == expected_dir.name
    assert manifest["config"]["exp_path"] == str(exp_root)
    assert manifest["config"]["exp_name"] == expected_dir.name


def test_symbolic_regressor_respects_explicit_timestamped_exp_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scientific_intelligent_modelling.srkit.regressor.env_manager.get_env_python",
        lambda env_name: sys.executable,
    )

    exp_root = tmp_path / "experiments"
    exp_name = "oscillator1_gplearn_seed1316_20260414-120000"
    reg = SymbolicRegressor(
        "gplearn",
        problem_name="oscillator1",
        seed=1316,
        exp_path=str(exp_root),
        exp_name=exp_name,
    )

    assert Path(reg.experiment_dir) == exp_root / exp_name
    manifest = json.loads((exp_root / exp_name / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == exp_name
    assert manifest["config"]["exp_name"] == exp_name
