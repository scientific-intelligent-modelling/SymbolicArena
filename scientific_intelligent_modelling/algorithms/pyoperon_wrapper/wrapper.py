# algorithms/pyoperon_wrapper/wrapper.py
import base64
import json
import math
import os
import pickle
import time
from copy import deepcopy

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_operon_artifact
from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError


class OperonRegressor(BaseWrapper):
    _DEFAULT_PARAMS = {
        "max_time": 3600,
        "population_size": 500,
        "pool_size": 500,
        "max_length": 50,
        "max_depth": 10,
        "tournament_size": 5,
        "allowed_symbols": "add,sub,mul,div,aq,exp,log,sin,cos,tanh,sqrt,square,constant,variable",
        "offspring_generator": "basic",
        "reinserter": "keep-best",
        "optimizer": "lm",
        "local_search_probability": 1.0,
        "max_evaluations": 500000,
        "n_threads": 4,
    }
    _META_PARAMS = {
        "exp_name",
        "exp_path",
        "problem_name",
        "seed",
        "n_features",
        "feature_names",
        "target_name",
        "timeout_guard_seconds",
        "progress_snapshot_interval_seconds",
        "task_label",
        "task_global_index",
        "expected_dataset_rel",
        "expected_dataset_dir",
    }
    _PROGRESS_STATE_FILENAME = ".pyoperon_current_best.json"
    _ALLOWED_PARAMS = {
        "allowed_symbols",
        "symbolic_mode",
        "crossover_probability",
        "crossover_internal_probability",
        "mutation",
        "mutation_probability",
        "offspring_generator",
        "reinserter",
        "objectives",
        "optimizer",
        "optimizer_likelihood",
        "optimizer_batch_size",
        "optimizer_iterations",
        "local_search_probability",
        "lamarckian_probability",
        "sgd_update_rule",
        "sgd_learning_rate",
        "sgd_beta",
        "sgd_beta2",
        "sgd_epsilon",
        "sgd_debias",
        "max_length",
        "max_depth",
        "initialization_method",
        "initialization_max_length",
        "initialization_max_depth",
        "female_selector",
        "male_selector",
        "population_size",
        "pool_size",
        "generations",
        "max_evaluations",
        "max_selection_pressure",
        "comparison_factor",
        "brood_size",
        "tournament_size",
        "irregularity_bias",
        "epsilon",
        "model_selection_criterion",
        "add_model_scale_term",
        "add_model_intercept_term",
        "uncertainty",
        "n_threads",
        "max_time",
        "random_state",
        "n_jobs",
        "niterations",
        "niteration",
        "population",
    }

    def __init__(self, **kwargs):
        raw_kwargs = dict(kwargs)
        self._exp_path = raw_kwargs.get("exp_path")
        self._exp_name = raw_kwargs.get("exp_name")
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._explicit_generation_budget = any(
            key in raw_kwargs for key in ("generations", "niterations", "niteration")
        )
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self.model = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

    @classmethod
    def _validate_and_normalize_params(cls, raw_params):
        params = {}
        seed = raw_params.get("seed")
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)

        if "random_state" not in raw_params and seed is not None:
            params["random_state"] = int(seed)

        alias_map = {
            "n_jobs": "n_threads",
            "niterations": "generations",
            "niteration": "generations",
            "population": "population_size",
            "timeout_in_seconds": "max_time",
        }

        for old, new in alias_map.items():
            if old in raw_params:
                alias_value = raw_params.pop(old)
                if new not in raw_params:
                    raw_params[new] = alias_value

        raw_params.pop("seed", None)

        for key, value in cls._DEFAULT_PARAMS.items():
            raw_params.setdefault(key, deepcopy(value))

        unknown = sorted(set(raw_params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "PyOperon 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        params.update(raw_params)

        if "max_time" in params and "generations" not in params:
            params["generations"] = 1000000

        if "n_threads" in params:
            try:
                params["n_threads"] = int(params["n_threads"])
            except Exception:
                pass

        if "generations" in params:
            try:
                params["generations"] = int(params["generations"])
            except Exception:
                pass

        if "population_size" in params:
            try:
                params["population_size"] = int(params["population_size"])
            except Exception:
                pass

        if "max_time" in params:
            value = params.get("max_time")
            if value in (None, "", "None"):
                params.pop("max_time", None)
            else:
                try:
                    value = int(float(value))
                except Exception as err:
                    raise TypeError(f"max_time 类型转换失败: {err}") from err
                if value <= 0:
                    params.pop("max_time", None)
                else:
                    params["max_time"] = value

        if "allowed_symbols" in params:
            params["allowed_symbols"] = cls._normalize_allowed_symbols(params["allowed_symbols"])

        return params

    @staticmethod
    def _normalize_allowed_symbols(value):
        if value in (None, "", "None"):
            return None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            if not items:
                return None
            if "constant" not in items:
                items.append("constant")
            if "variable" not in items:
                items.append("variable")
            return ",".join(items)
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if not items:
                return None
            if "constant" not in items:
                items.append("constant")
            if "variable" not in items:
                items.append("variable")
            return ",".join(items)
        raise TypeError("allowed_symbols 需为逗号分隔字符串或 list/tuple/set")

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

    @staticmethod
    def _extract_best_front_entry(model):
        front = getattr(model, "pareto_front_", None) or []
        if not front:
            return None
        try:
            criterion = getattr(model, "model_selection_criterion", "minimum_description_length")
            return min(front, key=lambda item: item.get(criterion, float("inf")))
        except Exception:
            return front[0]

    def _update_progress_state_from_model(self, model, completed_generations: int):
        best_entry = self._extract_best_front_entry(model)
        equation = None
        complexity = None
        loss = None

        if isinstance(best_entry, dict):
            equation = best_entry.get("model")
            complexity = best_entry.get("complexity")
            loss = best_entry.get("mean_squared_error")

        if not equation and getattr(model, "model_", None) is not None:
            try:
                equation = model.get_model_string(model.model_)
            except Exception:
                equation = None

        if complexity is None:
            try:
                complexity = getattr(model, "stats_", {}).get("model_complexity")
            except Exception:
                complexity = None

        if not isinstance(equation, str) or not equation.strip():
            return

        self._write_progress_state(
            {
                "equation": equation,
                "loss": float(loss) if isinstance(loss, (int, float)) else None,
                "complexity": int(complexity) if isinstance(complexity, (int, float)) else None,
                "generation": int(completed_generations),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def fit(self, X, y):
        import pyoperon.sklearn as pyoperon_sklearn

        SymbolicRegressor = pyoperon_sklearn.SymbolicRegressor
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="OperonRegressor.fit",
        )
        X = np.asarray(X)
        self.n_features_ = int(X.shape[1]) if X.ndim == 2 else 1
        params = dict(self.params)
        total_generations = int(params.get("generations", 1) or 1)
        total_generations = max(1, total_generations)
        max_time = params.get("max_time")
        try:
            max_time = float(max_time) if max_time is not None else None
        except Exception:
            max_time = None

        if self._progress_state_path:
            params["warm_start"] = False
            params["generations"] = 1
            model = SymbolicRegressor(**params)
            self.model = model
            started_at = time.time()
            completed = 0
            time_budget_driven = max_time is not None and not self._explicit_generation_budget
            # 未显式给 generations 时，进度模式必须以 wall-time 为主预算，避免快数据集先耗尽隐式代数上限。
            while time_budget_driven or completed < total_generations:
                if completed > 0:
                    model.warm_start = True
                    model.generations = 1
                if max_time is not None:
                    remaining = max_time - (time.time() - started_at)
                    if remaining <= 0:
                        break
                    model.max_time = max(1, int(math.ceil(remaining)))
                self._fit_pyoperon_model(pyoperon_sklearn, model, X, y)
                self._select_finite_model_or_raise(model, X)
                completed += 1
                self._update_progress_state_from_model(model, completed)
                if max_time is not None and time.time() - started_at >= max_time:
                    break
        else:
            model = SymbolicRegressor(**params)
            self.model = model
            self._fit_pyoperon_model(pyoperon_sklearn, model, X, y)
            self._select_finite_model_or_raise(model, X)

        self.best_model_str = self.model.get_model_string(self.model.model_)
        self.pareto_models = self._extract_pareto_models(self.model.pareto_front_)
        return self

    @staticmethod
    def _fit_pyoperon_model(pyoperon_sklearn, model, X, y):
        """运行 PyOperon fit，并避免其 pareto 统计阶段因 NaN 候选直接崩溃。"""
        original_mse = getattr(pyoperon_sklearn, "mean_squared_error", None)
        if original_mse is None:
            model.fit(X, y)
            return

        def finite_guarded_mse(y_true, y_pred, *args, **kwargs):
            pred = np.asarray(y_pred, dtype=float)
            true = np.asarray(y_true, dtype=float)
            if pred.shape != true.shape or pred.size == 0:
                return float("1e300")
            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
                return float("1e300")
            return original_mse(y_true, y_pred, *args, **kwargs)

        pyoperon_sklearn.mean_squared_error = finite_guarded_mse
        try:
            model.fit(X, y)
        except ValueError as exc:
            if "Input contains NaN" in str(exc):
                raise NoValidOutputError("PyOperon 候选预测包含 NaN，未产生可有限评估的候选表达式") from exc
            raise
        finally:
            pyoperon_sklearn.mean_squared_error = original_mse

    @staticmethod
    def _finite_prediction_for_model(model, X):
        try:
            pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
        except Exception:
            return False
        return pred.size == np.asarray(X).shape[0] and bool(np.all(np.isfinite(pred)))

    def _select_finite_model_or_raise(self, model, X):
        """优先保留可有限评估的 pareto 候选，避免 NaN 候选污染最终结果。"""
        if self._finite_prediction_for_model(model, X):
            return

        front = getattr(model, "pareto_front_", None) or []
        criterion = getattr(model, "model_selection_criterion", "minimum_description_length")
        try:
            front = sorted(front, key=lambda item: item.get(criterion, float("inf")))
        except Exception:
            front = list(front)

        for item in front:
            if not isinstance(item, dict) or item.get("tree") is None:
                continue
            original_model = getattr(model, "model_", None)
            try:
                model.model_ = item["tree"]
                if self._finite_prediction_for_model(model, X):
                    return
            except Exception:
                pass
            finally:
                if not self._finite_prediction_for_model(model, X):
                    model.model_ = original_model

        raise NoValidOutputError("PyOperon 未产生可有限评估的候选表达式")

    def predict(self, X):
        if self.model is None:
            if hasattr(self, "best_model_str") and getattr(self, "best_model_str"):
                return self._predict_from_expression(X)
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)

    def get_optimal_equation(self):
        if self.model is None:
            if hasattr(self, "best_model_str") and getattr(self, "best_model_str"):
                return self.best_model_str
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.best_model_str if hasattr(self, "best_model_str") else str(self.model)

    def get_total_equations(self):
        if self.model is None:
            if hasattr(self, "pareto_models") and self.pareto_models is not None:
                return self.pareto_models
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.pareto_models or [self.best_model_str]

    @staticmethod
    def _extract_pareto_models(pareto_front):
        models = []
        if not pareto_front:
            return models
        for item in pareto_front:
            if isinstance(item, dict) and "model" in item:
                models.append(str(item["model"]))
        return models

    def _build_predict_fn(self):
        if not hasattr(self, "best_model_str") or not getattr(self, "best_model_str", None):
            return None
        import sympy as sp

        n_features = getattr(self, "n_features_", 0)
        if n_features <= 0:
            return None

        symbols = [sp.Symbol(f"X{i+1}") for i in range(n_features)]
        try:
            expr = sp.sympify(self.best_model_str)
            fn = sp.lambdify(symbols, expr, modules=["numpy"])
        except Exception:
            try:
                expr = sp.sympify(self.best_model_str.replace(" ", ""))
                fn = sp.lambdify(symbols, expr, modules=["numpy"])
            except Exception:
                return None
        return fn

    def _predict_from_expression(self, X):
        fn = self._build_predict_fn()
        if fn is None:
            raise ValueError("无法从模型表达式构建预测函数，当前状态不支持直接预测")
        X = np.asarray(X)
        cols = [X[:, i] for i in range(X.shape[1])]
        return np.asarray(fn(*cols))

    def serialize(self):
        payload = {
            "params": self.params,
            "best_model_str": getattr(self, "best_model_str", None),
            "pareto_models": getattr(self, "pareto_models", None),
            "n_features_": getattr(self, "n_features_", None),
        }
        return base64.b64encode(pickle.dumps(payload)).decode("utf-8")

    @classmethod
    def deserialize(cls, instance_b64):
        payload = pickle.loads(base64.b64decode(instance_b64))
        inst = cls(**payload.get("params", {}))
        inst.best_model_str = payload.get("best_model_str")
        inst.pareto_models = payload.get("pareto_models")
        inst.n_features_ = payload.get("n_features_")
        inst.model = None
        return inst

    def export_canonical_symbolic_program(self):
        equation = self.get_optimal_equation()
        return normalize_operon_artifact(
            equation,
            expected_n_features=getattr(self, "n_features_", None),
        )


if __name__ == "__main__":
    import numpy as np

    X = np.random.rand(100, 2)
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.1 * np.random.randn(100)

    model = OperonRegressor(niterations=100, population_size=1000)
    model.fit(X, y)

    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    total_equations = model.get_total_equations()
    print(f"所有方程: {total_equations}")


__all__ = ["OperonRegressor"]
