from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
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
    _CURRENT_BEST_FILENAME = ".symbolfit_current_best.json"
    _SEARCH_BEST_FILENAME = ".symbolfit_search_best.json"
    _SEARCH_HISTORY_FILENAME = ".symbolfit_pysr_candidates.jsonl"
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
        self._experiment_dir = self._resolve_experiment_dir(raw_kwargs)
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self._apply_internal_timeout_guard(raw_kwargs)
        self.model = None
        self._best_candidate = None
        self._best_equation = None
        self._equations: list[str] = []
        self._callable = None
        self._fit_started_at: float | None = None
        self._coordinate_transform: dict[str, Any] | None = None
        self._search_best_loss: float | None = None
        self._search_best_payload: dict[str, Any] | None = None

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
        return time.monotonic() + self._explicit_timeout_seconds

    @classmethod
    def _remaining_budget_seconds(cls, deadline: float | None) -> int | None:
        if deadline is None:
            return None
        return max(0, int(deadline - time.monotonic()))

    @staticmethod
    def _resolve_experiment_dir(raw_params: dict[str, Any]) -> Path | None:
        exp_path = raw_params.get("exp_path")
        exp_name = raw_params.get("exp_name")
        if not exp_path or not exp_name:
            return None
        try:
            return Path(str(exp_path)).expanduser().resolve() / str(exp_name)
        except Exception:
            return None

    @classmethod
    def _iteration_timeout_seconds(cls, *, base_timeout: int | None, remaining_seconds: int) -> int:
        guarded_remaining = max(1, remaining_seconds - cls._resolve_timeout_guard(remaining_seconds, None))
        if base_timeout is None:
            return guarded_remaining
        return max(1, min(base_timeout, guarded_remaining))

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

    @staticmethod
    def _search_loss(candidate: Any) -> float | None:
        """读取 PySR 的内部搜索 loss，而不是 LMFIT/ID/OOD 评价分数。"""
        if candidate is None or not hasattr(candidate, "get"):
            return None
        for key in ("PySR loss", "Loss", "loss", "search_loss", "internal_loss"):
            try:
                value = float(candidate.get(key))
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _search_equation(candidate: Any) -> str | None:
        if candidate is None or not hasattr(candidate, "get"):
            return None
        for key in ("PySR equation", "Equation", "equation"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _build_coordinate_transform(
        X: np.ndarray,
        y: np.ndarray,
        *,
        input_rescale: bool,
        scale_y_by: Any,
    ) -> dict[str, Any]:
        """记录 SymbolFit 对 PySR 输入/目标使用的仿射缩放。"""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        x_min = np.min(X_arr, axis=0)
        x_max = np.max(X_arr, axis=0)
        x_range = x_max - x_min
        finite_y = y_arr[np.isfinite(y_arr)]

        # 上游仅在 input_rescale=True 时调用 histogram_scale；关闭输入缩放时
        # y_scale 固定为 1，即使参数仍携带 scale_y_by。
        mode = str(scale_y_by).strip().lower() if scale_y_by is not None else "none"
        if not input_rescale:
            denominator = 1.0
        elif mode == "max" and finite_y.size:
            denominator = abs(float(np.max(finite_y)))
        elif mode == "mean" and finite_y.size:
            denominator = abs(float(np.mean(finite_y)))
        elif mode == "l2" and finite_y.size:
            denominator = float(np.linalg.norm(finite_y))
        else:
            denominator = 1.0
        if not np.isfinite(denominator) or denominator <= 0.0:
            denominator = 1.0
        y_scale = 1.0 / denominator

        return {
            "version": "symbolfit_affine_v1",
            "input_rescale": bool(input_rescale),
            "x_min": [float(value) for value in x_min],
            "x_max": [float(value) for value in x_max],
            "x_range": [float(value) for value in x_range],
            "y_scale": float(y_scale),
            "y_unscale_factor": float(1.0 / y_scale),
            "scale_y_by": scale_y_by,
            "scaled_x_min": [0.0] * X_arr.shape[1],
            "scaled_x_max": [1.0] * X_arr.shape[1],
            "equation_space": "scaled_input_scaled_target"
            if input_rescale
            else "original_input_scaled_target",
        }

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

    def _write_current_best_snapshot(self, *, attempt: int, score: tuple[float, float]) -> None:
        if self._experiment_dir is None or not self._best_equation:
            return
        payload = {
            "tool": "symbolfit",
            "equation": self._best_equation,
            "equations": list(self._equations or [self._best_equation]),
            "attempt": int(attempt),
            "score": float(score[0]) if score and np.isfinite(score[0]) else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            path = self._experiment_dir / self._CURRENT_BEST_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    def _append_search_candidate_history(self, payload: dict[str, Any]) -> None:
        if self._experiment_dir is None:
            return
        try:
            path = self._experiment_dir / self._SEARCH_HISTORY_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            return

    def _write_search_best_snapshot(self, payload: dict[str, Any]) -> None:
        if self._experiment_dir is None:
            return
        try:
            path = self._experiment_dir / self._SEARCH_BEST_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    def _record_internal_candidates(
        self,
        model: Any,
        *,
        attempt: int,
        attempt_started_at: float,
    ) -> None:
        """把 PySR HOF 的内部候选持久化，跨 attempt 按内部 loss 维护 best。"""
        table = getattr(model, "func_candidates", None)
        if table is None:
            return
        try:
            rows = [row for _, row in table.iterrows()]
        except Exception:
            return
        observed_elapsed_seconds = max(
            0.0,
            float(time.monotonic() - (self._fit_started_at or attempt_started_at)),
        )
        observed_minute = max(1, int(math.ceil(observed_elapsed_seconds / 60.0)))
        for row in rows:
            equation = self._search_equation(row)
            loss = self._search_loss(row)
            if not equation or loss is None:
                continue
            candidate_key = hashlib.sha256(equation.encode("utf-8")).hexdigest()
            try:
                complexity = int(row.get("Complexity"))
            except (TypeError, ValueError, OverflowError, AttributeError):
                complexity = None
            payload = {
                "tool": "symbolfit",
                "candidate_key": candidate_key,
                "scaled_equation": equation,
                "internal_loss": loss,
                "loss": loss,
                "complexity": complexity,
                "attempt": int(attempt),
                "first_discovered_attempt": int(attempt),
                "first_discovered_minute": observed_minute,
                "first_discovered_elapsed_seconds": observed_elapsed_seconds,
                "coordinate_transform": self._coordinate_transform,
                "source": "symbolfit_pysr_hof_postfit",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._append_search_candidate_history(payload)
            if self._search_best_loss is None or loss < self._search_best_loss:
                self._search_best_loss = loss
                self._search_best_payload = dict(payload)
                self._write_search_best_snapshot(payload)

    @contextmanager
    def _attempt_work_dir(self, attempt: int):
        if self._experiment_dir is None:
            with tempfile.TemporaryDirectory(prefix="symbolfit_") as tmpdir:
                yield Path(tmpdir)
            return
        work_dir = (
            self._experiment_dir
            / "symbolfit_work"
            / f"attempt_{int(attempt):04d}_{os.getpid()}_{int(time.time())}"
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        yield work_dir

    def _write_active_run_snapshot(
        self,
        *,
        attempt: int,
        work_dir: Path,
        attempt_started_at: float | None = None,
    ) -> None:
        if self._experiment_dir is None:
            return
        payload = {
            "tool": "symbolfit",
            "attempt": int(attempt),
            "work_dir": str(work_dir),
            "attempt_started_at": attempt_started_at,
            "fit_started_at": self._fit_started_at,
            "coordinate_transform": self._coordinate_transform,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self._coordinate_transform:
            payload.update(
                {
                    "input_rescale": self._coordinate_transform.get("input_rescale"),
                    "x_min": self._coordinate_transform.get("x_min"),
                    "x_max": self._coordinate_transform.get("x_max"),
                    "x_range": self._coordinate_transform.get("x_range"),
                    "y_scale": self._coordinate_transform.get("y_scale"),
                    "y_unscale_factor": self._coordinate_transform.get("y_unscale_factor"),
                }
            )
        try:
            path = self._experiment_dir / ".symbolfit_active_run.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

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
        self._fit_started_at = time.monotonic()
        self._coordinate_transform = self._build_coordinate_transform(
            X_arr,
            y_arr,
            input_rescale=bool(self.params.get("input_rescale", True)),
            scale_y_by=self.params.get("scale_y_by"),
        )
        self._search_best_loss = None
        self._search_best_payload = None
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
        base_timeout = self._positive_int(self.params.get("timeout_in_seconds"))
        while True:
            remaining = self._remaining_budget_seconds(deadline)
            if attempt > 0 and remaining is not None and remaining < self._MIN_BUDGET_REFIT_SECONDS:
                break
            attempt += 1
            attempt_started_at = time.monotonic()
            iteration_params = dict(self.params)
            if remaining is not None:
                iteration_params["timeout_in_seconds"] = self._iteration_timeout_seconds(
                    base_timeout=base_timeout,
                    remaining_seconds=remaining,
                )
            if iteration_params.get("random_state") is not None:
                iteration_params["random_state"] = int(iteration_params["random_state"]) + attempt - 1
            with self._attempt_work_dir(attempt) as tmpdir:
                try:
                    self._write_active_run_snapshot(
                        attempt=attempt,
                        work_dir=tmpdir,
                        attempt_started_at=attempt_started_at,
                    )
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
            self._record_internal_candidates(
                model,
                attempt=attempt,
                attempt_started_at=attempt_started_at,
            )
            candidate = self._select_best_candidate()
            score = self._candidate_score(candidate)
            if best_candidate is None or (best_score is not None and score < best_score):
                best_model = model
                best_candidate = candidate
                best_score = score
                self.model = best_model
                self._best_candidate = best_candidate
                self._best_equation = self._extract_best_equation(self._best_candidate)
                self._equations = self.get_total_equations()
                self._callable = self._build_callable(self._best_equation)
                self._write_current_best_snapshot(attempt=attempt, score=best_score)
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
