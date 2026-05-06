"""gplearn wrapper（带参数白名单与参数规范化）。"""

from __future__ import annotations

import ast
import json
import numbers
import os
import time
import warnings
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_gplearn_artifact


class GPLearnRegressor(BaseWrapper):
    """gplearn SymbolicRegressor 的参数化适配层。"""
    _PROGRESS_STATE_FILENAME = ".gplearn_current_best.json"
    _DEFAULT_PARAMS = {
        "population_size": 1000,
        "generations": 1000000,
        "tournament_size": 20,
        "stopping_criteria": 0.0,
        "const_range": (-1.0, 1.0),
        "function_set": ("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos"),
        "init_depth": (2, 6),
        "init_method": "half and half",
        "metric": "mean absolute error",
        "parsimony_coefficient": 0.001,
        "p_crossover": 0.9,
        "p_subtree_mutation": 0.01,
        "p_hoist_mutation": 0.01,
        "p_point_mutation": 0.01,
        "p_point_replace": 0.05,
        "max_samples": 1.0,
        "n_jobs": 4,
        "verbose": 0,
        "low_memory": False,
        "warm_start": False,
    }
    _META_PARAMS = {
        "exp_name",
        "exp_path",
        "problem_name",
        "seed",
        "timeout_in_seconds",
        "n_features",
        "feature_names",
        "target_name",
    }

    _ALLOWED_PARAMS = {
        # 主搜索参数
        "population_size": (int, lambda v: int(v), lambda v: v > 0, "population_size > 0"),
        "generations": (int, lambda v: int(v), lambda v: v > 0, "generations > 0"),
        "tournament_size": (int, lambda v: int(v), lambda v: v > 1, "tournament_size > 1"),
        "stopping_criteria": (numbers.Real, float, lambda v: v >= 0, "stopping_criteria >= 0"),
        "parsimony_coefficient": (numbers.Real, float, lambda v: v >= 0, "parsimony_coefficient >= 0"),
        "p_crossover": (numbers.Real, float, lambda v: 0 <= v <= 1, "p_crossover in [0,1]"),
        "p_subtree_mutation": (numbers.Real, float, lambda v: 0 <= v <= 1, "p_subtree_mutation in [0,1]"),
        "p_hoist_mutation": (numbers.Real, float, lambda v: 0 <= v <= 1, "p_hoist_mutation in [0,1]"),
        "p_point_mutation": (numbers.Real, float, lambda v: 0 <= v <= 1, "p_point_mutation in [0,1]"),
        "p_point_replace": (numbers.Real, float, lambda v: 0 <= v <= 1, "p_point_replace in [0,1]"),

        # 结构与表达式空间参数
        "const_range": (object, None, None, "2元数值序列/可解析字符串"),
        "function_set": (object, None, None, "逗号分隔字符串或 list/tuple"),
        "init_depth": (object, None, None, "int 或 2 元组"),
        "init_method": (str, str, lambda v: v in {"half and half", "grow", "full"}, "init_method in {'half and half','grow','full'}"),
        "metric": (str, str, None, "字符串指标名"),
        "max_samples": (numbers.Real, float, lambda v: v > 0, "max_samples > 0"),
        "random_state": (int, int, None, "整数种子"),
        "n_jobs": (int, int, lambda v: v > 0, "n_jobs > 0"),
        "verbose": (int, int, lambda v: v >= 0, "verbose >= 0"),
        "low_memory": (bool, bool, None, "布尔"),
        "warm_start": (bool, bool, None, "布尔"),
    }

    _PROBABILITY_PARAMS = (
        "p_crossover",
        "p_subtree_mutation",
        "p_hoist_mutation",
        "p_point_mutation",
        "p_point_replace",
    )

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        self._exp_path = kwargs.get("exp_path")
        self._exp_name = kwargs.get("exp_name")
        self._contract_n_features = kwargs.get("n_features")
        self._contract_feature_names = kwargs.get("feature_names")
        self._contract_target_name = kwargs.get("target_name")
        # 延迟导入，避免环境问题
        self.params = self._validate_and_normalize_params(kwargs)
        self.model = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

    @classmethod
    def _validate_and_normalize_params(cls, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        raw_params = dict(raw_params)
        seed = raw_params.pop("seed", None)
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)
        if "random_state" not in raw_params and seed is not None:
            raw_params["random_state"] = int(seed)

        for key, value in cls._DEFAULT_PARAMS.items():
            raw_params.setdefault(key, deepcopy(value))

        unknown = sorted(set(raw_params.keys()) - set(cls._ALLOWED_PARAMS))
        if unknown:
            allowed = ", ".join(sorted(cls._ALLOWED_PARAMS))
            raise ValueError(
                f"gplearn 参数不受支持: {', '.join(unknown)}。"
                f"当前允许的参数有: {allowed}。"
            )

        for key, value in raw_params.items():
            type_hint, caster, validator, hint = cls._ALLOWED_PARAMS[key]
            if value is None:
                if key in ("function_set",):
                    # gplearn 的默认 function_set 可用，允许 None
                    params[key] = None
                    continue
                if key in ("init_depth",):
                    continue
                raise ValueError(f"{key} 不允许为 None；建议移除此字段或提供有效值")

            if key == "function_set":
                params[key] = cls._normalize_function_set(value)
                continue

            if key == "init_depth":
                params[key] = cls._normalize_init_depth(value)
                continue

            if key == "const_range":
                params[key] = cls._normalize_const_range(value)
                continue

            if key in ("low_memory", "warm_start"):
                params[key] = bool(value)
                continue

            if type_hint is numbers.Real:
                if not isinstance(value, numbers.Real):
                    raise TypeError(f"{key} 类型应为数值类型；你传的是: {type(value).__name__}")
            elif type_hint is not object and not isinstance(value, type_hint):
                # bool is bool subclass of int，故提前处理
                if type_hint is int and isinstance(value, bool):
                    pass
                else:
                    raise TypeError(f"{key} 类型应为 {type_hint.__name__}；你传的是: {type(value).__name__}")

            if caster is not None:
                try:
                    value = caster(value)
                except Exception as err:
                    raise TypeError(f"{key} 类型转换失败: {err}") from err

            if validator is not None and not validator(value):
                raise ValueError(f"{key} 校验失败：期望 {hint}；当前值: {value}")

            params[key] = value

        cls._validate_mutation_probabilities(params)
        return params

    @staticmethod
    def _ensure_gplearn_sklearn_compat(symbolic_regressor_cls):
        if hasattr(symbolic_regressor_cls, "_validate_data"):
            return

        try:
            from sklearn.utils.validation import check_X_y, check_array
        except Exception as e:
            warnings.warn(f"无法引入 sklearn 兼容校验函数: {e}")
            return

        def _build_validation_kwargs(base_kwargs):
            import inspect

            validation_kwargs = dict(base_kwargs)
            sig = inspect.signature(check_array)
            if "ensure_all_finite" in sig.parameters:
                validation_kwargs.pop("force_all_finite", None)
                validation_kwargs.setdefault("ensure_all_finite", True)
            else:
                validation_kwargs.setdefault("force_all_finite", True)
            return validation_kwargs

        def _validate_data(self, X, y=None, reset=True, validate_separately=False, **kwargs):
            if y is None:
                X_checked = check_array(
                    X,
                    ensure_2d=True,
                    dtype=np.float64,
                    order="C",
                    **_build_validation_kwargs(kwargs),
                )
                self.n_features_in_ = int(X_checked.shape[1])
                return X_checked
            X_checked, y_checked = check_X_y(
                X,
                y,
                ensure_2d=True,
                dtype=np.float64,
                **_build_validation_kwargs(kwargs),
            )
            self.n_features_in_ = int(np.asarray(X_checked).shape[1])
            return X_checked, y_checked

        setattr(symbolic_regressor_cls, "_validate_data", _validate_data)

    @staticmethod
    def _normalize_function_set(value: Any) -> Tuple[str, ...] | None:
        """将 function_set 标准化为 tuple 字符串序列。"""
        if value is None:
            return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("function_set 不能为空")
            # 支持 ["add", "sub", ...] 以及 add,sub,mul
            if text.startswith("[") or text.startswith("("):
                parsed = ast.literal_eval(text)
                if not isinstance(parsed, (list, tuple)):
                    raise TypeError("function_set 字符串解析后需为列表/元组")
                funcs = parsed
            else:
                funcs = [token.strip() for token in text.split(",") if token.strip()]
            return tuple(str(x) for x in funcs)

        if not isinstance(value, (list, tuple, set)):
            raise TypeError("function_set 需为 list/tuple/set 或逗号分隔字符串")
        if not value:
            raise ValueError("function_set 不能为空")
        return tuple(str(x) for x in value)

    @staticmethod
    def _normalize_init_depth(value: Any) -> Tuple[int, int]:
        if isinstance(value, int):
            if value < 1:
                raise ValueError("init_depth 必须 >= 1")
            return (value, value)
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("init_depth 需为单个 int 或长度为 2 的序列")
            low, high = int(value[0]), int(value[1])
            if low < 1 or high < low:
                raise ValueError("init_depth 需满足 1 <= low <= high")
            return (low, high)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("init_depth 字符串不能为空")
            if "," not in value:
                ivalue = int(value)
                return (ivalue, ivalue)
            low, high = (int(item.strip()) for item in value.split(",", 1))
            if low < 1 or high < low:
                raise ValueError("init_depth 需满足 1 <= low <= high")
            return (low, high)
        raise TypeError("init_depth 需为 int 或 2 元组/列表")

    @staticmethod
    def _normalize_const_range(value: Any) -> Tuple[float, float]:
        if isinstance(value, str):
            parsed = ast.literal_eval(value.strip())
        elif isinstance(value, (list, tuple)):
            parsed = value
        else:
            raise TypeError("const_range 需为 2 元组/列表（或其字符串形式）")
        if len(parsed) != 2:
            raise ValueError("const_range 需为 2 元素序列 (low, high)")
        low, high = float(parsed[0]), float(parsed[1])
        if low >= high:
            raise ValueError("const_range 需满足 low < high")
        return (low, high)

    @classmethod
    def _validate_mutation_probabilities(cls, params: Dict[str, Any]) -> None:
        given = []
        for name in cls._PROBABILITY_PARAMS:
            value = params.get(name)
            if value is None:
                continue
            if not isinstance(value, numbers.Real):
                raise TypeError(f"{name} 应为 0~1 的数值")
            if value < 0 or value > 1:
                raise ValueError(f"{name} 应在 [0,1]，当前值 {value}")
            if value > 0:
                given.append(float(value))

        if len(given) >= 2:
            prob_sum = sum(given)
            if prob_sum > 1.0 + 1e-12:
                raise ValueError(
                    "p_* 参数的和不能超过 1.0，当前配置为: "
                    f"{prob_sum:.6f}"
                )

    @classmethod
    def _resolve_progress_state_path(cls, exp_path, exp_name):
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(os.path.abspath(exp_path.strip()), exp_name.strip(), cls._PROGRESS_STATE_FILENAME)

    def _write_progress_state_from_model(self):
        if not self._progress_state_path or self.model is None:
            return
        try:
            run_details = getattr(self.model, "run_details_", {}) or {}
            if not run_details or not run_details.get("generation"):
                return
            last_idx = len(run_details["generation"]) - 1
            payload = {
                "equation": str(self.model),
                "loss": float(run_details["best_fitness"][last_idx]),
                "complexity": int(run_details["best_length"][last_idx]),
                "generation": int(run_details["generation"][last_idx]) + 1,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            with open(self._progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="GPLearnRegressor.fit",
        )
        # 仅在需要时导入
        from gplearn.genetic import SymbolicRegressor as GPLearnSR
        import warnings

        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="`BaseEstimator._validate_data` is deprecated",
        )

        self._ensure_gplearn_sklearn_compat(GPLearnSR)
        params = dict(self.params)
        total_generations = int(params.get("generations", 1) or 1)
        total_generations = max(1, total_generations)

        if self._progress_state_path:
            params["warm_start"] = False
            params["generations"] = 1
            self.model = GPLearnSR(**params)
            self.model.fit(X, y)
            self._write_progress_state_from_model()
            for generation in range(2, total_generations + 1):
                self.model.warm_start = True
                self.model.generations = generation
                self.model.fit(X, y)
                self._write_progress_state_from_model()
        else:
            self.model = GPLearnSR(**params)
            self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)

    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return str(self.model)

    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return [str(self.model)]

    def export_canonical_symbolic_program(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return normalize_gplearn_artifact(
            self.get_optimal_equation(),
            expected_n_features=getattr(self.model, "n_features_in_", None),
        )
    
