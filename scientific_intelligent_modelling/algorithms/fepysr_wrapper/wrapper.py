import json
import sys
import time
from copy import deepcopy
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact


class FePySRRegressor(BaseWrapper):
    """FePySR 两阶段符号回归适配层。

    FePySR 本体采用 PyTorch 特征抽取 + PySR 搜索。这里仅做工程集成：
    - 吸收 runner 元参数，避免透传到底层库；
    - 将常用 PySR/FePySR 参数映射成 hydra overrides；
    - 保持与工具集统一 fit/predict/equation/artifact 接口一致。
    """

    _DEFAULT_PARAMS = {
        "num_workers": 4,
        "num_experiments": 8,
        "fmn_epochs": 30,
        "fmn_batch_size": 64,
        "fmn_lr": 0.1,
        "fea_num": 10,
        "pysr_num": 6,
        "niterations": 40,
        "population_size": 40,
        "populations": 4,
        "ncycles_per_iteration": 100,
        "maxsize": 20,
        "maxdepth": 8,
        "timeout_in_seconds": 600,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp", "log"],
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
        "existing_exp_dir",
        "exp_dir",
    }
    _MIN_BUDGET_REFIT_SECONDS = 5
    _BOOTSTRAP_MAX_EXPERIMENTS = 1
    _BOOTSTRAP_MAX_FMN_EPOCHS = 5
    _BOOTSTRAP_MAX_PYSR_TIMEOUT_SECONDS = 300
    _ALLOWED_PARAMS = set(_DEFAULT_PARAMS) | {
        "overrides",
        "custom_pysr_model",
        "device",
        "fmn_only",
        "max_fit_attempt_seconds",
    }
    _FEATURE_VALUE_LIMIT = 1.0e6
    _CURRENT_BEST_FILENAME = ".fepysr_current_best.json"

    def __init__(self, **kwargs):
        raw_kwargs = dict(kwargs)
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._explicit_timeout_seconds = self._positive_int(raw_kwargs.get("timeout_in_seconds"))
        self._fill_timeout_budget = bool(raw_kwargs.get("fill_timeout_budget", True))
        self._experiment_dir = self._resolve_experiment_dir(raw_kwargs)
        self._existing_exp_dir = self._resolve_existing_exp_dir(raw_kwargs)
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self._apply_internal_timeout_guard(raw_kwargs)
        self.model = None
        self._equations: list[str] = []
        self._best_equation: str | None = None
        self._callable = None
        if self._existing_exp_dir is not None:
            self._load_current_best_snapshot(self._existing_exp_dir)

    @classmethod
    def _validate_and_normalize_params(cls, raw_params: dict[str, Any]) -> dict[str, Any]:
        raw_params = dict(raw_params)
        seed = raw_params.get("seed")
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)
        raw_params.pop("seed", None)

        for key, value in cls._DEFAULT_PARAMS.items():
            raw_params.setdefault(key, deepcopy(value))
        if seed is not None:
            raw_params.setdefault("random_state", int(seed))

        # random_state 只用于转成 PySR seed override，不直接暴露给 FePySR 构造器。
        allowed = set(cls._ALLOWED_PARAMS) | {"random_state"}
        unknown = sorted(set(raw_params) - allowed)
        if unknown:
            raise ValueError(
                "FePySR 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(allowed)),
                )
            )

        for key in (
            "num_workers",
            "num_experiments",
            "fmn_epochs",
            "fmn_batch_size",
            "fea_num",
            "pysr_num",
            "niterations",
            "population_size",
            "populations",
            "ncycles_per_iteration",
            "maxsize",
            "maxdepth",
            "timeout_in_seconds",
            "max_fit_attempt_seconds",
        ):
            if key in raw_params and raw_params[key] is not None:
                raw_params[key] = int(raw_params[key])
        if raw_params.get("population_size", 0) < 16:
            raise ValueError("FePySR 的 population_size 需要至少为 16，以满足 PySR tournament_selection_n 约束")
        if "fmn_lr" in raw_params and raw_params["fmn_lr"] is not None:
            raw_params["fmn_lr"] = float(raw_params["fmn_lr"])
        if "fmn_only" in raw_params:
            raw_params["fmn_only"] = bool(raw_params["fmn_only"])
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
        return min(300, max(1, int(timeout_seconds * 0.1)))

    def _apply_internal_timeout_guard(self, raw_kwargs: dict[str, Any]) -> None:
        timeout_seconds = self._positive_int(raw_kwargs.get("timeout_in_seconds"))
        if timeout_seconds is None:
            return
        guard_seconds = self._resolve_timeout_guard(
            timeout_seconds,
            raw_kwargs.get("timeout_guard_seconds"),
        )
        self.params["timeout_in_seconds"] = max(1, timeout_seconds - guard_seconds)

    @staticmethod
    def _format_override_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return repr(value)

    @classmethod
    def _build_overrides(cls, params: dict[str, Any]) -> list[str]:
        overrides = list(params.get("overrides") or [])
        mapping = {
            "num_workers": "Parallel.num_workers",
            "num_experiments": "Parallel.num_experiments",
            "fmn_epochs": "FMN.num_epochs",
            "fmn_batch_size": "FMN.batch_size",
            "fmn_lr": "FMN.lr",
            "fea_num": "data_symbol.fea_num",
            "pysr_num": "data_symbol.pysr_num",
            "niterations": "pysr_params.niterations",
            "population_size": "pysr_params.population_size",
            "populations": "pysr_params.populations",
            "ncycles_per_iteration": "pysr_params.ncycles_per_iteration",
            "maxsize": "pysr_params.maxsize",
            "maxdepth": "pysr_params.maxdepth",
            "timeout_in_seconds": "pysr_params.timeout_in_seconds",
            "binary_operators": "pysr_params.binary_operators",
            "unary_operators": "pysr_params.unary_operators",
            "device": "FMN.device",
            "fmn_only": "FMN.FMN_only",
        }
        for key, path in mapping.items():
            if key in params and params[key] is not None:
                overrides.append(f"{path}={cls._format_override_value(params[key])}")
        return overrides

    def _budget_deadline(self) -> float | None:
        if not self._fill_timeout_budget or self._explicit_timeout_seconds is None:
            return None
        budget_seconds = self._positive_int(self._explicit_timeout_seconds)
        if budget_seconds is None:
            return None
        return time.monotonic() + budget_seconds

    @staticmethod
    def _resolve_experiment_dir(raw_params: dict[str, Any]):
        exp_path = raw_params.get("exp_path")
        exp_name = raw_params.get("exp_name")
        if not exp_path or not exp_name:
            return None
        try:
            from pathlib import Path

            return Path(str(exp_path)).expanduser().resolve() / str(exp_name)
        except Exception:
            return None

    @staticmethod
    def _resolve_existing_exp_dir(raw_params: dict[str, Any]):
        raw = raw_params.get("existing_exp_dir") or raw_params.get("exp_dir")
        if not raw:
            return None
        try:
            from pathlib import Path

            path = Path(str(raw)).expanduser().resolve()
            return path if path.exists() else None
        except Exception:
            return None

    @classmethod
    def _remaining_budget_seconds(cls, deadline: float | None) -> int | None:
        if deadline is None:
            return None
        return max(0, int(deadline - time.monotonic()))

    @classmethod
    def _fit_attempt_timeout_seconds(cls, params: dict[str, Any], remaining: int | None) -> int | None:
        if remaining is None:
            return cls._positive_int(params.get("timeout_in_seconds"))
        explicit = cls._positive_int(params.get("max_fit_attempt_seconds"))
        if explicit is not None:
            attempt_budget = max(1, min(remaining, explicit))
        else:
            attempt_budget = max(1, min(remaining, 1200, max(1, remaining // 4)))
        nested_experiments = cls._positive_int(params.get("num_experiments")) or 1
        if nested_experiments <= 1:
            return attempt_budget
        return max(1, attempt_budget // nested_experiments)

    @classmethod
    def _apply_bootstrap_attempt_params(
        cls,
        params: dict[str, Any],
        *,
        attempt: int,
        has_best_equation: bool,
        remaining: int | None,
    ) -> None:
        if has_best_equation or remaining is None or params.get("fmn_only"):
            return
        num_experiments = cls._positive_int(params.get("num_experiments"))
        if num_experiments is not None:
            params["num_experiments"] = min(num_experiments, cls._BOOTSTRAP_MAX_EXPERIMENTS)
        num_workers = cls._positive_int(params.get("num_workers"))
        if num_workers is not None:
            params["num_workers"] = min(num_workers, cls._BOOTSTRAP_MAX_EXPERIMENTS)
        fmn_epochs = cls._positive_int(params.get("fmn_epochs"))
        if fmn_epochs is not None:
            params["fmn_epochs"] = min(fmn_epochs, cls._BOOTSTRAP_MAX_FMN_EPOCHS)
        timeout_seconds = cls._positive_int(params.get("timeout_in_seconds"))
        if timeout_seconds is not None:
            params["timeout_in_seconds"] = max(
                1,
                min(timeout_seconds, remaining, cls._BOOTSTRAP_MAX_PYSR_TIMEOUT_SECONDS),
            )

    @staticmethod
    def _equation_score(expr: str, X_arr: np.ndarray, y_arr: np.ndarray) -> float:
        try:
            pred = FePySRRegressor._build_callable(expr)(X_arr)
            target = y_arr.reshape(-1)
            if pred.shape[0] != target.shape[0] or not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((pred - target) ** 2))
        except Exception:
            return float("inf")

    @staticmethod
    def _optional_import(name: str):
        module = sys.modules.get(name)
        if module is not None:
            return module
        try:
            return __import__(name, fromlist=["*"])
        except Exception:
            return None

    @classmethod
    def _sanitize_fepysr_features(cls, data_analyzer) -> None:
        features = getattr(data_analyzer, "stacked_numpy_features", None)
        if features is None:
            return
        try:
            arr = np.asarray(features, dtype=float)
        except Exception:
            return
        if arr.size == 0:
            return
        limit = cls._FEATURE_VALUE_LIMIT
        sanitized = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
        sanitized = np.clip(sanitized, -limit, limit)
        if not np.array_equal(arr, sanitized):
            data_analyzer.stacked_numpy_features = sanitized

    @classmethod
    def _patch_fepysr_runtime(cls) -> None:
        feature_maker = cls._optional_import("fepysr.feature_maker")
        if feature_maker is not None and hasattr(feature_maker, "replace_pysr_variables"):
            current_replace = feature_maker.replace_pysr_variables
            original_replace = getattr(current_replace, "_sim_original", current_replace)
            if not getattr(current_replace, "_sim_safe_wrapper", False):

                def safe_replace_pysr_variables(pysr_equation, feature_names):
                    if isinstance(pysr_equation, bytes):
                        pysr_equation = pysr_equation.decode("utf-8", errors="replace")
                    elif not isinstance(pysr_equation, str):
                        pysr_equation = str(pysr_equation)
                    return original_replace(pysr_equation, feature_names)

                safe_replace_pysr_variables._sim_original = original_replace
                safe_replace_pysr_variables._sim_safe_wrapper = True
                feature_maker.replace_pysr_variables = safe_replace_pysr_variables

        pysr_train_module = cls._optional_import("fepysr.pysr_train")
        fepysr_impl_module = cls._optional_import("fepysr.fepysr")
        current_train = None
        if pysr_train_module is not None and hasattr(pysr_train_module, "pysr_train"):
            current_train = pysr_train_module.pysr_train
        elif fepysr_impl_module is not None and hasattr(fepysr_impl_module, "pysr_train"):
            current_train = fepysr_impl_module.pysr_train

        if current_train is None:
            return
        if getattr(current_train, "_sim_safe_wrapper", False):
            safe_pysr_train = current_train
        else:
            original_train = getattr(current_train, "_sim_original", current_train)

            def safe_pysr_train(data_analyzer, *args, **kwargs):
                cls._sanitize_fepysr_features(data_analyzer)
                return original_train(data_analyzer, *args, **kwargs)

            safe_pysr_train._sim_original = original_train
            safe_pysr_train._sim_safe_wrapper = True

        if pysr_train_module is not None:
            pysr_train_module.pysr_train = safe_pysr_train
        if fepysr_impl_module is not None:
            fepysr_impl_module.pysr_train = safe_pysr_train

    @staticmethod
    def _format_constant_equation(value: float) -> str:
        if not np.isfinite(value):
            value = 0.0
        text = format(float(value), ".17g")
        return "0" if text == "-0" else text

    def _install_mean_constant_baseline(self, y_arr: np.ndarray) -> float:
        target = np.asarray(y_arr, dtype=float).reshape(-1)
        finite_target = target[np.isfinite(target)]
        value = float(np.mean(finite_target)) if finite_target.size else 0.0
        equation = self._format_constant_equation(value)
        self._best_equation = equation
        self._equations = [equation]
        self._callable = self._build_callable(equation)
        if not finite_target.size:
            return 0.0
        return float(np.mean((np.full_like(finite_target, value, dtype=float) - finite_target) ** 2))

    def _write_current_best_snapshot(self, *, attempt: int, score: float, source: str | None = None) -> None:
        if self._experiment_dir is None or not self._best_equation:
            return
        payload = {
            "tool": "fepysr",
            "equation": self._best_equation,
            "equations": list(self._equations),
            "attempt": int(attempt),
            "score": float(score) if np.isfinite(score) else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if source:
            payload["source"] = source
        try:
            path = self._experiment_dir / self._CURRENT_BEST_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    def _load_current_best_snapshot(self, experiment_dir) -> None:
        try:
            path = experiment_dir / self._CURRENT_BEST_FILENAME
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        equation = payload.get("equation")
        if not isinstance(equation, str) or not equation.strip():
            return
        self._best_equation = equation
        self._equations = [
            str(item)
            for item in payload.get("equations", [equation])
            if isinstance(item, str) and item.strip()
        ]
        self._callable = self._build_callable(self._best_equation)

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="FePySRRegressor.fit",
        )
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        deadline = self._budget_deadline()
        baseline_score = self._install_mean_constant_baseline(y_arr)
        self._write_current_best_snapshot(
            attempt=0,
            score=baseline_score,
            source="mean_constant_baseline",
        )
        try:
            import pysr  # noqa: F401
            import torch
            from fepysr import FePySR
            self._patch_fepysr_runtime()
        except Exception as err:  # pragma: no cover - exercised in integration env
            raise ImportError(
                "FePySRRegressor 需要安装 fepysr、torch、hydra-core、pysr；"
                "请先创建/激活 sim_fepysr 环境。"
            ) from err

        X_tensor = torch.as_tensor(X_arr, dtype=torch.float64)
        y_tensor = torch.as_tensor(y_arr, dtype=torch.float64)
        best_model = None
        best_equation = self._best_equation
        best_score = baseline_score
        equations: list[str] = list(self._equations)
        attempt = 0
        while True:
            remaining = self._remaining_budget_seconds(deadline)
            if attempt > 0 and remaining is not None and remaining < self._MIN_BUDGET_REFIT_SECONDS:
                break
            attempt += 1
            iteration_params = dict(self.params)
            if remaining is not None:
                attempt_timeout = self._fit_attempt_timeout_seconds(iteration_params, remaining)
                if attempt_timeout is not None:
                    iteration_params["timeout_in_seconds"] = attempt_timeout
                self._apply_bootstrap_attempt_params(
                    iteration_params,
                    attempt=attempt,
                    has_best_equation=best_equation is not None,
                    remaining=remaining,
                )
            if iteration_params.get("random_state") is not None:
                iteration_params["random_state"] = int(iteration_params["random_state"]) + attempt - 1
            iteration_params.pop("max_fit_attempt_seconds", None)
            model = FePySR(
                overrides=self._build_overrides(iteration_params),
                custom_pysr_model=iteration_params.get("custom_pysr_model"),
            )
            model.fit(X_tensor, y_tensor)
            eq = str(getattr(model, "best_equation_", "") or "")
            if eq:
                equations.append(eq)
                score = self._equation_score(eq, X_arr, y_arr)
                if best_equation is None or score < best_score:
                    best_model = model
                    best_equation = eq
                    best_score = score
                    self._best_equation = best_equation
                    self._equations = [
                        item
                        for idx, item in enumerate(equations)
                        if item and item not in equations[:idx]
                    ]
                    self._write_current_best_snapshot(
                        attempt=attempt,
                        score=best_score,
                        source="fepysr_fit",
                    )
            if deadline is None or time.monotonic() >= deadline:
                break
        self.model = best_model
        self._best_equation = best_equation
        if self._best_equation:
            self._callable = self._build_callable(self._best_equation)
        self._equations = [eq for idx, eq in enumerate(equations) if eq and eq not in equations[:idx]]
        return self

    @staticmethod
    def _build_callable(expr: str):
        import sympy as sp

        text = expr.replace("^", "**")
        text = text.replace("torch.", "").replace("np.", "").replace("numpy.", "")
        text = text.replace("abs(", "Abs(")
        text = __import__("re").sub(r"\bX(\d+)\b", lambda m: f"x{m.group(1)}", text)
        symbols = [sp.Symbol(f"x{i}") for i in range(64)]
        locals_map = {f"x{i}": symbols[i] for i in range(64)}
        locals_map.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "Abs": sp.Abs,
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

    def serialize(self):
        if not isinstance(self._best_equation, str) or not self._best_equation.strip():
            raise ValueError("FePySR 未产生可序列化方程")
        params = {k: v for k, v in self.params.items() if k != "custom_pysr_model"}
        return json.dumps(
            {
                "mode": "fepysr_expression",
                "params": params,
                "contract": {
                    "n_features": self._contract_n_features,
                    "feature_names": self._contract_feature_names,
                    "target_name": self._contract_target_name,
                },
                "best_equation": self._best_equation,
                "equations": self._equations,
            },
            ensure_ascii=False,
        )

    @classmethod
    def deserialize(cls, payload):
        try:
            obj = json.loads(payload)
        except Exception:
            return BaseWrapper.deserialize(payload)
        if not isinstance(obj, dict) or obj.get("mode") != "fepysr_expression":
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
        if self.model is None and self._callable is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if self.model is None:
            return self._callable(X)
        pred = self.model.predict(np.asarray(X, dtype=float))
        try:
            pred = pred.detach().cpu().numpy()
        except Exception:
            pred = np.asarray(pred)
        return np.asarray(pred, dtype=float).reshape(-1)

    def get_optimal_equation(self):
        eq = self._best_equation
        if eq is None and self.model is not None:
            eq = getattr(self.model, "best_equation_", None)
        if not isinstance(eq, str) or not eq.strip():
            raise ValueError("FePySR 未产生可用最优方程")
        return eq

    def get_total_equations(self):
        if self.model is None and not self._equations:
            raise ValueError("模型尚未训练，请先调用fit方法")
        equations = list(self._equations)
        solved = getattr(self.model, "solved_pysr_model", None)
        if solved is not None and hasattr(solved, "equations_"):
            try:
                table = solved.equations_
                if hasattr(table, "columns"):
                    for col in ("sympy_format", "equation", "expr", "expression"):
                        if col in table.columns:
                            equations.extend(str(item) for item in table[col].dropna().tolist())
                            break
            except Exception:
                pass
        return [eq for idx, eq in enumerate(equations) if eq and eq not in equations[:idx]]

    def export_canonical_symbolic_program(self):
        return normalize_external_infix_artifact(
            self.get_optimal_equation(),
            tool_name="fepysr",
            expected_n_features=self._contract_n_features,
            shift_one_based=False,
        )
