from __future__ import annotations

import json
import sys
import types

import numpy as np


def test_dso_chunk_restart_does_not_regress_progress_state(monkeypatch, tmp_path):
    """新的 timeout chunk 不能覆盖此前 reward 更高的在线候选。"""

    from scientific_intelligent_modelling.algorithms.dso_wrapper import wrapper as dso_wrapper

    clock = {"now": 0.0}

    class FakeProgram:
        def __init__(self, equation, reward):
            self.sympy_expr = equation
            self.r = reward
            self.complexity = 1

        def pretty(self):
            return self.sympy_expr

        def execute(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    class FakeTrainer:
        def __init__(self, program):
            self.p_r_best = program
            self.done = False

    class FakeDSO:
        calls = 0

        def __init__(self, config):
            self.config = config
            # 每个 chunk 都是独立搜索；第二个 chunk 故意给出更差 reward。
            reward = 0.9 if FakeDSO.calls == 0 else 0.1
            equation = "x0 + 1" if FakeDSO.calls == 0 else "x0 - 1"
            FakeDSO.calls += 1
            self.trainer = FakeTrainer(FakeProgram(equation, reward))

        def set_config(self, config):
            self.config = config

        def train_one_step(self):
            clock["now"] += 2.0
            return {"program": self.trainer.p_r_best}

        def finish(self):
            return {"program": self.trainer.p_r_best}

    monkeypatch.setitem(sys.modules, "dso", types.SimpleNamespace(DeepSymbolicOptimizer=FakeDSO))
    monkeypatch.setattr(dso_wrapper.time, "time", lambda: clock["now"])

    reg = dso_wrapper.DSORegressor(
        exp_path=str(tmp_path),
        exp_name="dso_chunk_restart",
        seed=11,
        timeout_in_seconds=3,
        timeout_guard_seconds=0.1,
    )
    reg.fit(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0]))

    state_path = tmp_path / "dso_chunk_restart" / ".dso_current_best.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    # wrapper 使用 repr(program.sympy_expr) 落盘，FakeProgram 的表达式是字符串。
    assert state["equation"] == repr("x0 + 1")
    assert state["score"] == 0.9


def test_gplearn_progress_state_keeps_lowest_internal_loss(monkeypatch, tmp_path):
    """跨 generation 快照应保留 gplearn 内部 loss 最低的历史候选。"""

    from scientific_intelligent_modelling.algorithms.gplearn_wrapper import wrapper as gplearn_wrapper

    class FakeSymbolicRegressor:
        # 避免测试触发真实 sklearn 兼容补丁的导入；真实 gplearn 类本身已有该入口。
        _validate_data = object()

        def __init__(self, **kwargs):
            self.generations = kwargs["generations"]
            self.warm_start = kwargs.get("warm_start", False)
            self.n_features_in_ = 1
            self._program = "X0_gen_0"
            self.run_details_ = {
                "generation": [],
                "best_fitness": [],
                "best_length": [],
            }

        def fit(self, X, y):
            generation = int(self.generations) - 1
            self.run_details_["generation"].append(generation)
            # 第二代更差；若按最后一代写入，旧实现会错误覆盖 0.1。
            loss = 0.1 if generation == 0 else 0.5 + generation
            self.run_details_["best_fitness"].append(loss)
            self.run_details_["best_length"].append(1 + generation)
            self._program = f"X0_gen_{generation}"
            return self

        def __str__(self):
            return self._program

        def predict(self, X):
            return np.asarray(X, dtype=float)[:, 0]

    gplearn_module = types.ModuleType("gplearn")
    genetic_module = types.ModuleType("gplearn.genetic")
    genetic_module.SymbolicRegressor = FakeSymbolicRegressor
    monkeypatch.setitem(sys.modules, "gplearn", gplearn_module)
    monkeypatch.setitem(sys.modules, "gplearn.genetic", genetic_module)

    reg = gplearn_wrapper.GPLearnRegressor(
        exp_path=str(tmp_path),
        exp_name="gplearn_generations",
        generations=3,
        population_size=4,
    )
    reg.fit(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0]))

    state_path = tmp_path / "gplearn_generations" / ".gplearn_current_best.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["equation"] == "X0_gen_0"
    assert state["loss"] == 0.1
    assert state["generation"] == 1
