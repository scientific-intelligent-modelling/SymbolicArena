from __future__ import annotations

import json
import os
import re
import tempfile
import time
from copy import deepcopy
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact


class SymbolFitRegressor(BaseWrapper):
    """SymbolFit 适配层。

    SymbolFit 以 PySR 搜索结构，再用 LMFIT 重优化常数并估计不确定性。
    本 wrapper 选取 RMSE 最低的 refit candidate 作为工具集统一最优方程。
    """

    _DEFAULT_PARAMS = {
        "niterations": 40,
        "maxsize": 25,
        "model_selection": "accuracy",
        "binary_operators": ["+", "*", "/", "-"],
        "unary_operators": ["sin", "cos", "exp", "log"],
        "max_complexity": 25,
        "input_rescale": True,
        "scale_y_by": "mean",
        "max_stderr": 20,
        "fit_y_unc": False,
        "y_uncertainty": 1.0,
        "procs": 1,
        "parallelism": "serial",
        "deterministic": True,
        "timeout_in_seconds": 600,
    }
    _META_PARAMS = {
        "exp_name",
        "exp_path",
        "problem_name",
        "seed",
        "n_features",
        "feature_names",
        "target_name",
        "task_label",
        "task_global_index",
        "expected_dataset_rel",
        "expected_dataset_dir",
        "progress_snapshot_interval_seconds",
        "timeout_guard_seconds",
        "fill_timeout_budget",
    }
    _MIN_BUDGET_REFIT_SECONDS = 5
    _ALLOWED_PARAMS = set(_DEFAULT_PARAMS) | {
        "random_state",
        "pysr_config",
        "loss_weights",
        "y_up",
        "y_down",
    }

    def __init__(self, **kwargs):
        raw_kwargs = dict(kwargs)
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._explicit_timeout_seconds = self._positive_int(raw_kwargs.get("timeout_in_seconds"))
        self._fill_timeout_budget = bool(raw_kwargs.get("fill_timeout_budget", True))
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self._apply_internal_timeout_guard(raw_kwargs)
        self.model = None
        self._best_candidate = None
        self._best_equation = None
        self._equations: list[str] = []
        self._callable = None

    @classmethod
    def _validate_and_normalize_params(cls, raw_params: dict[str, Any]) -> dict[str, Any]:
        raw_params = dict(raw_params)
        seed = raw_params.get("seed")
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)
        raw_params.pop("seed", None)
        if seed is not None:
            raw_params.setdefault("random_state", int(seed))
        for key, value in cls._DEFAULT_PARAMS.items():
            raw_params.setdefault(key, deepcopy(value))
        unknown = sorted(set(raw_params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "SymbolFit 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )
        for key in ("niterations", "maxsize", "max_complexity", "procs", "timeout_in_seconds"):
            if raw_params.get(key) is not None:
                raw_params[key] = int(raw_params[key])
        for key in ("input_rescale", "fit_y_unc", "deterministic"):
            raw_params[key] = bool(raw_params[key])
        if raw_params.get("max_stderr") is not None:
            raw_params["max_stderr"] = float(raw_params["max_stderr"])
        if raw_params.get("y_uncertainty") is not None:
            raw_params["y_uncertainty"] = float(raw_params["y_uncertainty"])
        return raw_params

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _resolve_timeout_guard(cls, timeout_seconds: int, raw_guard: Any) -> int:
        explicit_guard = cls._positive_int(raw_guard)
        if explicit_guard is not None:
            return explicit_guard
        guard_ratio = 0.3 if timeout_seconds >= 600 else 0.1
        return min(300, max(1, int(timeout_seconds * guard_ratio)))

    def _apply_internal_timeout_guard(self, raw_kwargs: dict[str, Any]) -> None:
        timeout_seconds = self._positive_int(raw_kwargs.get("timeout_in_seconds"))
        if timeout_seconds is None:
            return
        guard_seconds = self._resolve_timeout_guard(
            timeout_seconds,
            raw_kwargs.get("timeout_guard_seconds"),
        )
        self.params["timeout_in_seconds"] = max(1, timeout_seconds - guard_seconds)

    def _build_pysr_config(self, params: dict[str, Any] | None = None):
        params = params or self.params
        if params.get("pysr_config") is not None:
            return params["pysr_config"]
        from pysr import PySRRegressor

        kwargs = {
            "model_selection": params["model_selection"],
            "niterations": params["niterations"],
            "maxsize": params["maxsize"],
            "binary_operators": params["binary_operators"],
            "unary_operators": params["unary_operators"],
            "elementwise_loss": "loss(y, y_pred, weights) = (y - y_pred)^2 * weights",
            "procs": params["procs"],
            "parallelism": params["parallelism"],
            "deterministic": params["deterministic"],
            "timeout_in_seconds": params["timeout_in_seconds"],
        }
        if params.get("random_state") is not None:
            kwargs["random_state"] = int(params["random_state"])
        return PySRRegressor(**kwargs)

    def _budget_deadline(self) -> float | None:
        if not self._fill_timeout_budget or self._explicit_timeout_seconds is None:
            return None
        budget_seconds = self._positive_int(self.params.get("timeout_in_seconds"))
        if budget_seconds is None:
            return None
        return time.monotonic() + budget_seconds

    @classmethod
    def _remaining_budget_seconds(cls, deadline: float | None) -> int | None:
        if deadline is None:
            return None
        return max(0, int(deadline - time.monotonic()))

    @staticmethod
    def _candidate_score(candidate) -> tuple[float, float]:
        rmse = candidate.get("RMSE", np.inf)
        r2 = candidate.get("R2", -np.inf)
        try:
            rmse_value = float(rmse)
        except Exception:
            rmse_value = np.inf
        try:
            r2_value = float(r2)
        except Exception:
            r2_value = -np.inf
        return rmse_value, -r2_value

    def _select_best_candidate(self):
        table = getattr(self.model, "func_candidates", None)
        if table is None or len(table) == 0:
            raise ValueError("SymbolFit 未产生候选方程")
        rows = [row for _, row in table.iterrows()]
        return min(rows, key=self._candidate_score)

    @staticmethod
    def _extract_best_equation(candidate) -> str:
        for key in (
            "Parameterized equation, unscaled",
            "Parameterized equation",
            "PySR equation",
        ):
            try:
                value = candidate[key]
            except Exception:
                continue
            if isinstance(value, str) and value.strip():
                expr = value.strip()
                params = candidate.get("Parameters: (best-fit, +1, -1)", {})
                if isinstance(params, dict):
                    for name, values in sorted(params.items(), key=lambda item: len(str(item[0])), reverse=True):
                        try:
                            best = values[0]
                        except Exception:
                            continue
                        expr = re.sub(rf"\b{re.escape(str(name))}\b", str(best), expr)
                return expr
        raise ValueError("SymbolFit 候选中没有可用表达式字段")

    @staticmethod
    def _build_callable(expr: str):
        import sympy as sp

        text = expr.replace("^", "**")
        text = re.sub(r"\bX(\d+)\b", lambda m: f"x{m.group(1)}", text)
        symbols = [sp.Symbol(f"x{i}") for i in range(64)]
        locals_map = {f"x{i}": symbols[i] for i in range(64)}
        locals_map.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "tanh": sp.tanh,
            }
        )
        parsed = sp.sympify(text, locals=locals_map)
        used = sorted(
            [sym for sym in parsed.free_symbols if str(sym).startswith("x")],
            key=lambda sym: int(str(sym)[1:]) if str(sym)[1:].isdigit() else 10**9,
        )
        if not used:
            const_value = float(parsed)
            return lambda X: np.full((np.asarray(X).shape[0],), const_value, dtype=float)
        fn = sp.lambdify(used, parsed, modules="numpy")

        def _predict(X):
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            args = [X_arr[:, int(str(sym)[1:])] for sym in used]
            return np.asarray(fn(*args), dtype=float).reshape(-1)

        return _predict

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="SymbolFitRegressor.fit",
        )
        try:
            from symbolfit.symbolfit import SymbolFit
        except Exception as err:  # pragma: no cover - exercised in integration env
            raise ImportError(
                "SymbolFitRegressor 需要安装 symbolfit、pysr、lmfit；请先创建/激活 sim_symbolfit 环境。"
            ) from err

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        y_up = self.params.get("y_up")
        y_down = self.params.get("y_down")
        if y_up is None:
            y_up = np.full_like(y_arr, float(self.params["y_uncertainty"]), dtype=float)
        if y_down is None:
            y_down = np.full_like(y_arr, float(self.params["y_uncertainty"]), dtype=float)

        cwd = os.getcwd()
        deadline = self._budget_deadline()
        best_model = None
        best_candidate = None
        best_score: tuple[float, float] | None = None
        attempt = 0
        while True:
            remaining = self._remaining_budget_seconds(deadline)
            if attempt > 0 and remaining is not None and remaining < self._MIN_BUDGET_REFIT_SECONDS:
                break
            attempt += 1
            iteration_params = dict(self.params)
            if remaining is not None:
                iteration_params["timeout_in_seconds"] = max(1, remaining)
            if iteration_params.get("random_state") is not None:
                iteration_params["random_state"] = int(iteration_params["random_state"]) + attempt - 1
            with tempfile.TemporaryDirectory(prefix="symbolfit_") as tmpdir:
                try:
                    os.chdir(tmpdir)
                    model = SymbolFit(
                        x=X_arr,
                        y=y_arr,
                        y_up=y_up,
                        y_down=y_down,
                        pysr_config=self._build_pysr_config(iteration_params),
                        max_complexity=iteration_params["max_complexity"],
                        input_rescale=iteration_params["input_rescale"],
                        scale_y_by=iteration_params["scale_y_by"],
                        max_stderr=iteration_params["max_stderr"],
                        fit_y_unc=iteration_params["fit_y_unc"],
                        random_seed=iteration_params.get("random_state"),
                        loss_weights=iteration_params.get("loss_weights"),
                    )
                    model.fit()
                finally:
                    os.chdir(cwd)
            self.model = model
            candidate = self._select_best_candidate()
            score = self._candidate_score(candidate)
            if best_candidate is None or (best_score is not None and score < best_score):
                best_model = model
                best_candidate = candidate
                best_score = score
            if deadline is None or time.monotonic() >= deadline:
                break
        self.model = best_model
        self._best_candidate = best_candidate
        self._best_equation = self._extract_best_equation(self._best_candidate)
        self._equations = self.get_total_equations()
        self._callable = self._build_callable(self._best_equation)
        return self

    def serialize(self):
        if not isinstance(self._best_equation, str) or not self._best_equation.strip():
            raise ValueError("SymbolFit 未产生可序列化方程")
        params = {k: v for k, v in self.params.items() if k != "pysr_config"}
        return json.dumps(
            {
                "mode": "symbolfit_expression",
                "params": params,
                "contract": {
                    "n_features": self._contract_n_features,
                    "feature_names": self._contract_feature_names,
                    "target_name": self._contract_target_name,
                },
                "best_equation": self._best_equation,
                "equations": self.get_total_equations(),
            },
            ensure_ascii=False,
        )

    @classmethod
    def deserialize(cls, payload):
        try:
            obj = json.loads(payload)
        except Exception:
            return BaseWrapper.deserialize(payload)
        if not isinstance(obj, dict) or obj.get("mode") != "symbolfit_expression":
            return BaseWrapper.deserialize(payload)
        contract = dict(obj.get("contract") or {})
        inst = cls(
            n_features=contract.get("n_features"),
            feature_names=contract.get("feature_names"),
            target_name=contract.get("target_name"),
            **dict(obj.get("params") or {}),
        )
        inst._best_equation = obj.get("best_equation")
        inst._equations = list(obj.get("equations") or ([inst._best_equation] if inst._best_equation else []))
        if inst._best_equation:
            inst._callable = inst._build_callable(inst._best_equation)
        return inst

    def predict(self, X):
        if self._callable is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self._callable(X)

    def get_optimal_equation(self):
        if not isinstance(self._best_equation, str) or not self._best_equation.strip():
            raise ValueError("模型尚未训练或未产生可用方程")
        return self._best_equation

    def get_total_equations(self):
        if self.model is None:
            equations = getattr(self, "_equations", None)
            if equations:
                return list(equations)
            return [self.get_optimal_equation()]
        table = getattr(self.model, "func_candidates", None)
        if table is None or len(table) == 0:
            return [self.get_optimal_equation()]
        equations = []
        for _, row in table.iterrows():
            try:
                equations.append(self._extract_best_equation(row))
            except Exception:
                continue
        return [eq for idx, eq in enumerate(equations) if eq and eq not in equations[:idx]]

    def export_canonical_symbolic_program(self):
        return normalize_external_infix_artifact(
            self.get_optimal_equation(),
            tool_name="symbolfit",
            expected_n_features=self._contract_n_features,
            shift_one_based=False,
        )
