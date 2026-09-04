# tools/gplearn_wrapper/wrapper.py
import json
import os
import sys
from typing import Any, Dict
from copy import deepcopy
import tempfile
import time
import hashlib
import re

import numpy as np
from sympy import sympify, lambdify
from sympy.core.sympify import SympifyError

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_dso_artifact

class DSORegressor(BaseWrapper):
    _PROGRESS_STATE_FILENAME = ".dso_current_best.json"
    _DEFAULT_EXPERIMENT = {
        "logdir": None,
    }
    _DEFAULT_TASK = {
        "task_type": "regression",
        "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
        "metric": "inv_nrmse",
        "metric_params": [1.0],
        "threshold": 1e-12,
        "protected": False,
    }
    _DEFAULT_TRAINING = {
        "batch_size": 1000,
        "n_samples": 2000000,
        "epsilon": 0.05,
        "n_cores_batch": 4,
    }
    _DEFAULT_POLICY_OPTIMIZER = {
        "learning_rate": 0.0005,
        "entropy_weight": 0.03,
        "entropy_gamma": 0.7,
    }
    _DEFAULT_PRIOR = {
        "length": {"min_": 4, "max_": 64, "on": True},
        "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
        "inverse": {"on": True},
        "trig": {"on": True},
        "const": {"on": True},
        "no_inputs": {"on": True},
        "uniform_arity": {"on": True},
        "soft_length": {"loc": 10, "scale": 5, "on": True},
        "domain_range": {"on": False},
    }
    _EXPERIMENT_KEYS = {"logdir", "exp_name", "seed"}
    _TASK_KEYS = {"task_type", "function_set", "metric", "metric_params", "threshold", "protected"}
    _TRAINING_KEYS = {"batch_size", "n_samples", "epsilon", "n_cores_batch"}

    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        raw_kwargs = dict(kwargs or {})
        self._exp_path = raw_kwargs.get("exp_path")
        self._exp_name = raw_kwargs.get("exp_name")
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._timeout_in_seconds = self._as_positive_float(raw_kwargs.get("timeout_in_seconds"))
        self._timeout_guard_seconds = self._as_positive_float(raw_kwargs.get("timeout_guard_seconds")) or 5.0
        self.params = self._build_config(raw_kwargs)
        self.model = None
        self._dso_equation = None
        self._dso_expression = None
        self._dso_pred_fn = None
        self._dso_pred_constant = None
        self._dso_var_count = 0
        self._dso_input_indices = []
        self._dso_n_features = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)
        # timeout 模式会以独立模型实例分 chunk 运行；这里保存跨 chunk 的全局 best，
        # 避免新 chunk 的较差局部最优覆盖周期快照。
        self._progress_best_reward = None
        self._progress_best_payload = None

    @staticmethod
    def _as_positive_float(value):
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

    @classmethod
    def _build_config(cls, raw_kwargs):
        params = dict(raw_kwargs or {})
        experiment = dict(cls._DEFAULT_EXPERIMENT)
        task = dict(cls._DEFAULT_TASK)
        training = dict(cls._DEFAULT_TRAINING)
        policy_optimizer = deepcopy(cls._DEFAULT_POLICY_OPTIMIZER)
        prior = deepcopy(cls._DEFAULT_PRIOR)

        for key in list(params.keys()):
            value = params[key]
            if key == "experiment" and isinstance(value, dict):
                experiment.update(value)
                params.pop(key)
            elif key == "task" and isinstance(value, dict):
                task.update(value)
                params.pop(key)
            elif key == "training" and isinstance(value, dict):
                training.update(value)
                params.pop(key)
            elif key == "policy_optimizer" and isinstance(value, dict):
                policy_optimizer.update(value)
                params.pop(key)
            elif key == "prior" and isinstance(value, dict):
                for prior_key, prior_value in value.items():
                    if isinstance(prior_value, dict) and isinstance(prior.get(prior_key), dict):
                        prior[prior_key].update(prior_value)
                    else:
                        prior[prior_key] = prior_value
                params.pop(key)

        for key in list(params.keys()):
            if key in cls._EXPERIMENT_KEYS:
                experiment[key] = params.pop(key)
            elif key in cls._TASK_KEYS:
                task[key] = params.pop(key)
            elif key in cls._TRAINING_KEYS:
                training[key] = params.pop(key)

        exp_path = params.pop("exp_path", None)
        exp_name = params.pop("exp_name", None)
        problem_name = params.pop("problem_name", None)
        seed = params.pop("seed", None)
        params.pop("timeout_in_seconds", None)
        params.pop("n_features", None)
        params.pop("feature_names", None)
        params.pop("target_name", None)

        if seed is not None and "seed" not in experiment:
            experiment["seed"] = int(seed)
        if exp_name and "exp_name" not in experiment:
            experiment["exp_name"] = str(exp_name)
        if exp_path and experiment.get("logdir") is None:
            # DSO 内部会再用 exp_name 组装 save_path，logdir 这里只传实验根目录，
            # 避免最终路径变成 exp_path/exp_name/exp_name 的双层结构。
            experiment["logdir"] = os.path.abspath(str(exp_path))

        config = dict(params)
        config["experiment"] = experiment
        config["task"] = task
        config["training"] = training
        config["policy_optimizer"] = policy_optimizer
        config["prior"] = prior
        return config

    @staticmethod
    def _program_reward(program):
        reward = getattr(program, "r", None)
        try:
            return float(reward)
        except Exception:
            return None

    def _best_program_from_model(self):
        if self.model is None:
            return None
        trainer = getattr(self.model, "trainer", None)
        program = getattr(trainer, "p_r_best", None) if trainer is not None else None
        if program is not None:
            return program
        return getattr(self.model, "program_", None)

    def _time_budget_exhausted(self, started_at):
        if self._timeout_in_seconds is None:
            return False
        elapsed = time.time() - started_at
        return elapsed >= max(0.0, self._timeout_in_seconds - self._timeout_guard_seconds)
    
    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="DSORegressor.fit",
        )
        # 优先使用子仓库源码，避免环境可复现性差异导致的 editable 安装问题
        repo_root = os.path.dirname(os.path.abspath(__file__))
        local_dso_path = os.path.join(repo_root, "dso", "dso")
        if os.path.isdir(local_dso_path) and local_dso_path not in sys.path:
            sys.path.insert(0, local_dso_path)

        # 仅在需要时导入
        from dso import DeepSymbolicOptimizer
        import warnings
        # 过滤掉特定的FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning, 
                                message="`BaseEstimator._validate_data` is deprecated")
        
        if not hasattr(__import__("dso"), "DeepSymbolicOptimizer"):
            raise ImportError("当前 dso 包未提供 DeepSymbolicOptimizer，请检查 dso 源码或安装版本。")

        def _new_model(chunk_index=0):
            params = deepcopy(self.params)
            if self._timeout_in_seconds is not None:
                training = dict(params.get("training") or {})
                training["early_stopping"] = False
                params["training"] = training
                experiment = dict(params.get("experiment") or {})
                seed = experiment.get("seed")
                if seed is not None:
                    try:
                        experiment["seed"] = int(seed) + int(chunk_index)
                    except Exception:
                        pass
                params["experiment"] = experiment
            model = DeepSymbolicOptimizer(params)
            fit_config = self._build_fit_config(model.config, X, y)
            if self._timeout_in_seconds is not None:
                fit_config.setdefault("training", {})
                fit_config["training"]["early_stopping"] = False
            model.set_config(fit_config)
            return model

        # 创建并训练模型
        self.model = _new_model(0)
        train_result = None
        best_program = None
        best_reward = None
        if self._progress_state_path:
            started_at = time.time()
            chunk_index = 0
            natural_completions = 0
            while True:
                if self._time_budget_exhausted(started_at):
                    break
                step_result = self.model.train_one_step()
                self._update_progress_state_from_model()
                program = self._best_program_from_model()
                reward = self._program_reward(program)
                if program is not None and (best_program is None or (reward is not None and (best_reward is None or reward > best_reward))):
                    best_program = program
                    best_reward = reward
                if step_result is not None:
                    program = step_result.get("program") if isinstance(step_result, dict) else None
                    reward = self._program_reward(program)
                    if program is not None and (best_program is None or (reward is not None and (best_reward is None or reward > best_reward))):
                        best_program = program
                        best_reward = reward
                    if self._timeout_in_seconds is None:
                        train_result = step_result
                        break
                    natural_completions += 1
                    chunk_index += 1
                    self.model = _new_model(chunk_index)
                    continue
                trainer = getattr(self.model, "trainer", None)
                if trainer is not None and getattr(trainer, "done", False):
                    if self._timeout_in_seconds is None:
                        break
                    natural_completions += 1
                    chunk_index += 1
                    self.model = _new_model(chunk_index)
            self._budget_chunks_run = chunk_index + 1
            self._natural_completions = natural_completions
            self._budget_loop_exhausted = self._time_budget_exhausted(started_at)
        else:
            train_result = self.model.train()

        if train_result is None and getattr(self.model, "trainer", None) is not None:
            train_result = self.model.finish()
        if best_program is not None:
            train_result = dict(train_result or {})
            train_result["program"] = best_program
        self.model.program_ = train_result["program"]
        x_arr = np.asarray(X)
        self._dso_n_features = int(x_arr.shape[1]) if x_arr.ndim == 2 else 1
        self._cache_post_fit_state()
        return self

    @staticmethod
    def _short_dataset_path(exp_name: str, prefix: str) -> str:
        tmp_root = os.environ.get("SIM_DSO_DATA_TMP", "/tmp/e1tmp/dso_data")
        os.makedirs(tmp_root, exist_ok=True)
        raw_name = str(exp_name or prefix or "dso_regression")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_") or prefix
        safe_name = safe_name[:80]
        digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:10]
        return os.path.join(tmp_root, f"{safe_name}__train_{digest}.csv")

    @staticmethod
    def _build_fit_config(base_config: Dict[str, Any], X, y) -> Dict[str, Any]:
        config = deepcopy(base_config)
        experiment = config.setdefault("experiment", {})
        logdir = experiment.get("logdir")
        if not logdir:
            logdir = tempfile.mkdtemp(prefix="dso-fit-")
            experiment["logdir"] = logdir
        os.makedirs(logdir, exist_ok=True)
        exp_name = experiment.get("exp_name") or "dso_regression"
        dataset_path = DSORegressor._short_dataset_path(str(exp_name), "dso_regression")
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        stacked = np.concatenate([x_arr, y_arr], axis=1)
        np.savetxt(dataset_path, stacked, delimiter=",")
        config.setdefault("task", {})
        config["task"]["dataset"] = dataset_path
        gp_meld = config.get("gp_meld") or {}
        if gp_meld.get("run_gp_meld"):
            print("WARNING: GP-meld not yet supported for sklearn interface.")
        gp_meld["run_gp_meld"] = False
        config["gp_meld"] = gp_meld
        return config

    @classmethod
    def _resolve_progress_state_path(cls, exp_path, exp_name):
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(os.path.abspath(exp_path.strip()), exp_name.strip(), cls._PROGRESS_STATE_FILENAME)

    def _write_progress_state(self, payload):
        if not self._progress_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            with open(self._progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _refresh_progress_state_path_from_model(self):
        experiment_cfg = getattr(self.model, "config_experiment", None)
        if not isinstance(experiment_cfg, dict):
            return
        save_path = experiment_cfg.get("save_path")
        if not isinstance(save_path, str) or not save_path.strip():
            return
        self._progress_state_path = os.path.join(os.path.abspath(save_path.strip()), self._PROGRESS_STATE_FILENAME)

    def _update_progress_state_from_model(self):
        self._refresh_progress_state_path_from_model()
        trainer = getattr(self.model, "trainer", None)
        program = getattr(trainer, "p_r_best", None) if trainer is not None else None
        if program is None:
            return
        try:
            equation = repr(program.sympy_expr)
        except Exception:
            equation = None
        if not isinstance(equation, str) or not equation.strip():
            return
        complexity = getattr(program, "complexity", None)
        reward = getattr(program, "r", None)
        iteration = getattr(trainer, "iteration", None)
        reward_value = None
        if isinstance(reward, (int, float, np.integer, np.floating)):
            try:
                candidate_reward = float(reward)
                if np.isfinite(candidate_reward):
                    reward_value = candidate_reward
            except (TypeError, ValueError, OverflowError):
                reward_value = None

        # DSO reward 越大越好。独立 timeout chunk 的 trainer 会从头开始，
        # 所以只在新候选严格优于跨 chunk 历史 best 时更新落盘状态。
        if self._progress_best_payload is not None:
            if reward_value is None:
                return
            if self._progress_best_reward is not None and reward_value <= self._progress_best_reward:
                return

        payload = {
            "equation": equation,
            "score": reward_value,
            "complexity": int(complexity) if isinstance(complexity, (int, float, np.integer, np.floating)) else None,
            "iteration": int(iteration) if isinstance(iteration, (int, np.integer)) else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._progress_best_reward = reward_value
        self._progress_best_payload = payload
        self._write_progress_state(payload)

    def _cache_post_fit_state(self):
        if self.model is None or not hasattr(self.model, "program_"):
            self._dso_equation = None
            self._dso_expression = None
            self._dso_pred_fn = None
            self._dso_pred_constant = None
            self._dso_var_count = 0
            return

        self._dso_expression = str(self.model.program_.sympy_expr)
        self._dso_equation = str(self.model.program_.pretty())
        # 仅在运行时可进行反序列化后的预测，不在主进程训练过程里触发额外解析成本
        self._build_predict_fn_from_equation(self._dso_expression)

    def _build_predict_fn_from_equation(self, equation: str):
        try:
            expr = sympify(equation.strip(), locals={"x": None})
        except (SympifyError, TypeError, SyntaxError, ValueError):
            self._dso_var_count = 0
            self._dso_pred_fn = None
            self._dso_pred_constant = None
            return

        raw_indices = []
        for token in expr.free_symbols:
            name = str(token)
            if not name.startswith("x"):
                continue
            suffix = name[1:]
            if suffix.isdigit():
                raw_indices.append(int(suffix))

        raw_indices = sorted(set(raw_indices))
        if not raw_indices:
            self._dso_var_count = 0
            self._dso_pred_fn = None
            self._dso_input_indices = []
            try:
                self._dso_pred_constant = float(expr)
            except Exception:
                self._dso_pred_constant = None
            return

        one_based = 0 not in raw_indices
        input_indices = []
        for idx in raw_indices:
            mapped = idx - 1 if one_based else idx
            if mapped < 0:
                self._dso_var_count = 0
                self._dso_pred_fn = None
                self._dso_pred_constant = None
                self._dso_input_indices = []
                return
            input_indices.append(mapped)

        self._dso_var_count = max(input_indices) + 1
        self._dso_pred_fn = None
        self._dso_pred_constant = None
        self._dso_input_indices = input_indices

        if self._dso_var_count <= 0:
            self._dso_input_indices = []
            return

        symbols = [f"x{idx}" for idx in raw_indices]
        self._dso_pred_fn = lambdify(symbols, expr, modules="numpy")

    def _predict_with_cached_fn(self, X):
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        if self._dso_pred_constant is not None:
            return np.full(x_arr.shape[0], float(self._dso_pred_constant), dtype=float)

        if self._dso_pred_fn is None:
            raise RuntimeError("DSO 反序列化模型不包含可执行方程，无法继续执行 predict")

        n_features = x_arr.shape[1] if x_arr.ndim > 1 else 0
        if n_features < self._dso_var_count:
            raise ValueError("DSO 反序列化状态下的输入特征维度不足")

        args = [x_arr[:, idx] for idx in self._dso_input_indices]
        y = self._dso_pred_fn(*args)
        return np.asarray(y, dtype=float).reshape(-1)
    
    def predict(self, X):
        if self.model is None:
            if self._dso_pred_fn is not None or self._dso_pred_constant is not None:
                return self._predict_with_cached_fn(X)
            raise ValueError("模型尚未训练，请先调用fit方法")
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        if hasattr(self.model, "program_"):
            return self.model.program_.execute(np.asarray(X))
        raise ValueError("DSO 模型状态不完整，无法执行 predict")
    
    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        if self.model is None:
            if self._dso_equation is not None:
                return self._dso_equation
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return str(self.model.program_.pretty())

    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        if self.model is None:
            if self._dso_equation is not None:
                return [self._dso_equation]
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 返回模型的字符串表示，这就是拟合的方程
        return [str(self.model.program_.pretty())]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # 规避 RLock/进程上下文等不可 pickle 对象，保留方程文本供反序列化恢复预测
        if state.get("model") is not None:
            self._cache_post_fit_state()
        state["model"] = None
        state["_dso_input_indices"] = self._dso_input_indices
        state["_dso_pred_fn"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if not hasattr(self, "_dso_pred_constant"):
            self._dso_pred_constant = None
        if self.model is not None:
            self.model = None
        if self._dso_expression:
            self._build_predict_fn_from_expression()
            return
        if self._dso_equation:
            self._build_predict_fn_from_equation(self._dso_equation)

    def _build_predict_fn_from_expression(self):
        if not self._dso_expression:
            return
        self._build_predict_fn_from_equation(self._dso_expression)

    def export_canonical_symbolic_program(self):
        raw_equation = self._dso_expression
        if raw_equation is None and self.model is not None and hasattr(self.model, "program_"):
            try:
                raw_equation = str(self.model.program_.sympy_expr)
            except Exception:
                raw_equation = None
        if raw_equation is None:
            raise ValueError("DSO 当前没有可导出的标准表达式")
        expected_n_features = self._dso_n_features
        if expected_n_features is None and self._dso_var_count:
            expected_n_features = int(self._dso_var_count)
        return normalize_dso_artifact(
            raw_equation,
            expected_n_features=expected_n_features,
        )
    
