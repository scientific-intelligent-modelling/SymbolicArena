from __future__ import annotations

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact


class JAXSRRegressor(BaseWrapper):
    """JAXSR sparse basis-library 符号回归适配层。"""

    _PROGRESS_STATE_FILENAME = ".jaxsr_current_best.json"
    _CANDIDATE_SOURCE = _PROGRESS_STATE_FILENAME
    _FIDELITY_PROBE_VERSION = "jaxsr_export_fidelity_v1"
    _FIDELITY_PROBE_ROW_LIMIT = 64
    _FIDELITY_RTOL = 1e-6
    _FIDELITY_ATOL = 1e-14
    _DEFAULT_PARAMS = {
        "max_terms": 5,
        "strategy": "greedy_forward",
        "information_criterion": "bic",
        "cv_folds": 5,
        "regularization": None,
        "max_polynomial_degree": 3,
        "max_interaction_order": 2,
        "include_constant": True,
        "include_linear": True,
        "include_polynomials": True,
        "include_interactions": True,
        "include_transcendental": False,
        "include_ratios": False,
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
        "timeout_in_seconds",
    }
    _ALLOWED_PARAMS = set(_DEFAULT_PARAMS) | {
        "random_state",
        "basis_library",
        "transcendental_functions",
    }

    def __init__(self, **kwargs):
        raw_kwargs = dict(kwargs)
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._timeout_in_seconds = self._as_positive_float(raw_kwargs.get("timeout_in_seconds"))
        self._timeout_guard_seconds = self._as_positive_float(raw_kwargs.get("timeout_guard_seconds")) or 5.0
        self._progress_state_path = self._resolve_progress_state_path(
            raw_kwargs.get("exp_path"),
            raw_kwargs.get("exp_name"),
        )
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self.model = None
        self._export_equation: str | None = None
        self._fidelity_evidence: dict[str, Any] | None = None

    @staticmethod
    def _as_positive_float(value) -> float | None:
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

    @classmethod
    def _resolve_progress_state_path(cls, exp_path, exp_name) -> str | None:
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(
            os.path.abspath(exp_path.strip()),
            exp_name.strip(),
            cls._PROGRESS_STATE_FILENAME,
        )

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
                "JAXSR 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )
        for key in ("max_terms", "cv_folds", "max_polynomial_degree", "max_interaction_order"):
            if raw_params.get(key) is not None:
                raw_params[key] = int(raw_params[key])
        for key in (
            "include_constant",
            "include_linear",
            "include_polynomials",
            "include_interactions",
            "include_transcendental",
            "include_ratios",
        ):
            raw_params[key] = bool(raw_params[key])
        if raw_params.get("regularization") is not None:
            raw_params["regularization"] = float(raw_params["regularization"])
        return raw_params

    def _feature_names_for_fit(self, n_features: int) -> list[str]:
        return [f"x{i}" for i in range(n_features)]

    def _build_basis_library(self, n_features: int):
        from jaxsr import BasisLibrary

        params = self.params
        library = params.get("basis_library")
        if library is not None:
            return library
        library = BasisLibrary(
            n_features=n_features,
            feature_names=self._feature_names_for_fit(n_features),
        )
        if params["include_constant"]:
            library = library.add_constant()
        if params["include_linear"]:
            library = library.add_linear()
        if params["include_polynomials"]:
            library = library.add_polynomials(max_degree=params["max_polynomial_degree"])
        if params["include_interactions"]:
            library = library.add_interactions(max_order=params["max_interaction_order"])
        if params["include_transcendental"]:
            funcs = params.get("transcendental_functions")
            try:
                library = library.add_transcendental(functions=funcs) if funcs else library.add_transcendental()
            except TypeError:
                library = library.add_transcendental()
        if params["include_ratios"]:
            library = library.add_ratios()
        return library

    def _iteration_random_state(self, iteration: int) -> int | None:
        base = self.params.get("random_state")
        if base is None and self._timeout_in_seconds is None:
            return None
        base_value = 0 if base is None else int(base)
        return base_value + int(iteration)

    def _build_model(self, SymbolicRegressor, library, iteration: int):
        return SymbolicRegressor(
            basis_library=library,
            max_terms=self.params["max_terms"],
            strategy=self.params["strategy"],
            information_criterion=self.params["information_criterion"],
            cv_folds=self.params["cv_folds"],
            regularization=self.params["regularization"],
            random_state=self._iteration_random_state(iteration),
        )

    @staticmethod
    def _model_equation(model) -> str:
        if hasattr(model, "to_sympy"):
            return str(model.to_sympy())
        return str(model.expression_)

    @staticmethod
    def _estimate_complexity(equation: str) -> int | None:
        try:
            import sympy as sp

            expr = sp.sympify(equation)
            return int(sum(1 for _ in sp.preorder_traversal(expr)))
        except Exception:
            return None

    @staticmethod
    def _training_mse(model, X_arr: np.ndarray, y_arr: np.ndarray) -> float:
        try:
            pred = np.asarray(model.predict(X_arr), dtype=float).reshape(-1)
            if pred.shape != y_arr.shape or not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((pred - y_arr) ** 2))
        except Exception:
            return float("inf")

    @staticmethod
    def _hash_float_array(values: np.ndarray) -> str:
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest = hashlib.sha256()
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _hash_json(value: Any) -> str | None:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except Exception:
            return None
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _build_fidelity_probe(cls, X_arr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        X_arr = np.asarray(X_arr, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError("fidelity probe 需要非空二维训练输入")

        finite_rows = X_arr[np.all(np.isfinite(X_arr), axis=1)]
        if finite_rows.shape[0] == 0:
            raise ValueError("fidelity probe 找不到全有限训练行")

        sample_count = min(32, finite_rows.shape[0])
        sample_indices = np.linspace(
            0,
            finite_rows.shape[0] - 1,
            num=sample_count,
            dtype=int,
        )
        pieces = [finite_rows[sample_indices]]

        # 固定分位点锚点只依赖训练 X，不查看 y 或任何验证/测试 split。
        quantiles = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        quantile_rows = np.quantile(finite_rows, quantiles, axis=0)
        pieces.append(quantile_rows)
        median = quantile_rows[2]
        feature_anchors = []
        for feature_index in range(finite_rows.shape[1]):
            for quantile_index in (0, 4):
                anchor = median.copy()
                anchor[feature_index] = quantile_rows[quantile_index, feature_index]
                feature_anchors.append(anchor)
        if feature_anchors:
            pieces.append(np.asarray(feature_anchors, dtype=float))

        candidates = np.vstack(pieces)
        unique_rows: list[np.ndarray] = []
        seen: set[bytes] = set()
        for row in candidates:
            key = np.ascontiguousarray(row, dtype="<f8").tobytes()
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(np.asarray(row, dtype=float))
            if len(unique_rows) >= cls._FIDELITY_PROBE_ROW_LIMIT:
                break
        probe = np.vstack(unique_rows)
        definition = {
            "version": cls._FIDELITY_PROBE_VERSION,
            "strategy": "fit_X_stratified_rows_plus_quantile_anchors",
            "source_shape": list(X_arr.shape),
            "finite_source_rows": int(finite_rows.shape[0]),
            "row_limit": cls._FIDELITY_PROBE_ROW_LIMIT,
            "quantiles": quantiles.tolist(),
            "dtype": "float64",
            "values": probe.tolist(),
        }
        return probe, definition

    @staticmethod
    def _replay_equation(equation: str, probe: np.ndarray) -> np.ndarray:
        import sympy as sp

        text = str(equation).strip()
        if "=" in text:
            left, right = text.split("=", 1)
            if left.strip().lower() == "y":
                text = right.strip()
        text = text.replace("^", "**")
        symbols = [sp.Symbol(f"x{index}") for index in range(probe.shape[1])]
        local_symbols = {str(symbol): symbol for symbol in symbols}
        expression = sp.sympify(text, locals=local_symbols)
        unknown_symbols = sorted(str(symbol) for symbol in expression.free_symbols - set(symbols))
        if unknown_symbols:
            raise ValueError(f"公式含未知符号: {unknown_symbols}")
        func = sp.lambdify(symbols, expression, modules="numpy")
        replay = np.asarray(
            func(*[probe[:, index] for index in range(probe.shape[1])]),
            dtype=float,
        )
        if replay.ndim == 0:
            replay = np.full(probe.shape[0], float(replay), dtype=float)
        return replay.reshape(-1)

    @classmethod
    def _compare_equation_to_native(
        cls,
        model,
        equation: str,
        probe: np.ndarray,
        *,
        source: str,
    ) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "source": source,
            "equation": str(equation),
            "equation_sha256": hashlib.sha256(str(equation).encode("utf-8")).hexdigest(),
            "status": "failed",
            "reason": None,
            "native_prediction_sha256": None,
            "replay_prediction_sha256": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "native_abs_scale": None,
            "allowed_max_abs_error": None,
        }
        try:
            native = np.asarray(model.predict(probe), dtype=float).reshape(-1)
            replay = cls._replay_equation(str(equation), probe)
            if native.shape != (probe.shape[0],):
                raise ValueError(
                    f"native prediction 形状异常: {native.shape}, 期望 {(probe.shape[0],)}"
                )
            if replay.shape != native.shape:
                raise ValueError(f"replay prediction 形状异常: {replay.shape}, 期望 {native.shape}")
            if not np.all(np.isfinite(native)):
                raise ValueError("native prediction 含非有限值")
            if not np.all(np.isfinite(replay)):
                raise ValueError("raw equation replay 含非有限值")

            absolute_error = np.abs(native - replay)
            relative_error = absolute_error / np.maximum(np.abs(native), cls._FIDELITY_ATOL)
            native_abs_scale = float(np.max(np.abs(native), initial=0.0))
            allowed_max_abs_error = cls._FIDELITY_ATOL + cls._FIDELITY_RTOL * native_abs_scale
            max_abs_error = float(np.max(absolute_error, initial=0.0))
            matches = bool(max_abs_error <= allowed_max_abs_error)
            attempt.update(
                {
                    "status": "verified" if matches else "mismatch",
                    "reason": None if matches else "native prediction 与 raw equation replay 不一致",
                    "native_prediction_sha256": cls._hash_float_array(native),
                    "replay_prediction_sha256": cls._hash_float_array(replay),
                    "max_abs_error": max_abs_error,
                    "max_rel_error": float(np.max(relative_error, initial=0.0)),
                    "native_abs_scale": native_abs_scale,
                    "allowed_max_abs_error": allowed_max_abs_error,
                }
            )
        except Exception as err:
            attempt["reason"] = f"{err.__class__.__name__}: {err}"
        return attempt

    def _equation_from_model_state(self, model, n_features: int) -> str:
        import sympy as sp

        result = getattr(model, "_result", None)
        coefficients = getattr(result, "coefficients", None)
        selected_names = getattr(result, "selected_names", None)
        if coefficients is None or selected_names is None:
            state = model._state_dict() if hasattr(model, "_state_dict") else None
            result_state = state.get("result") if isinstance(state, dict) else None
            if isinstance(result_state, dict):
                coefficients = result_state.get("coefficients")
                selected_names = result_state.get("selected_names")
        if coefficients is None or selected_names is None:
            raise ValueError("模型状态缺少 coefficients/selected_names")

        coefficient_values = np.asarray(coefficients, dtype=float).reshape(-1)
        names = list(selected_names)
        if len(coefficient_values) != len(names):
            raise ValueError("模型状态中的 coefficients 与 selected_names 长度不一致")
        if not np.all(np.isfinite(coefficient_values)):
            raise ValueError("模型状态中的 coefficients 含非有限值")

        feature_names = [f"x{index}" for index in range(n_features)]
        basis_library = getattr(model, "basis_library", None)
        native_feature_names = getattr(basis_library, "feature_names", None)
        if isinstance(native_feature_names, (list, tuple)) and len(native_feature_names) == n_features:
            feature_names = [str(name) for name in native_feature_names]
        symbols = {name: sp.Symbol(name) for name in feature_names}
        parser = getattr(model, "_parse_basis_to_sympy", None)
        terms = []
        for coefficient, name in zip(coefficient_values, names, strict=True):
            if coefficient == 0.0:
                continue
            if callable(parser):
                basis_expression = parser(str(name), symbols)
            else:
                basis_expression = sp.sympify(str(name).replace("^", "**"), locals=symbols)
            precise_coefficient = sp.Float(repr(float(coefficient)), 17)
            terms.append(precise_coefficient * basis_expression)
        if not terms:
            return "0"
        return str(sp.Add(*terms))

    def _prepare_model_export(
        self,
        model,
        X_arr: np.ndarray,
    ) -> tuple[str | None, dict[str, Any]]:
        probe, probe_definition = self._build_fidelity_probe(X_arr)
        raw_equation: str | None = None
        try:
            raw_equation = self._model_equation(model)
            raw_attempt = self._compare_equation_to_native(
                model,
                raw_equation,
                probe,
                source="model.to_sympy",
            )
        except Exception as err:
            raw_attempt = {
                "source": "model.to_sympy",
                "equation": None,
                "equation_sha256": None,
                "status": "failed",
                "reason": f"{err.__class__.__name__}: {err}",
                "native_prediction_sha256": None,
                "replay_prediction_sha256": None,
                "max_abs_error": None,
                "max_rel_error": None,
                "native_abs_scale": None,
                "allowed_max_abs_error": None,
            }
        attempts = [raw_attempt]
        selected_equation: str | None = None
        selected_attempt = attempts[0]
        equation_source: str | None = None

        if selected_attempt["status"] == "verified":
            selected_equation = str(raw_equation)
            equation_source = "model.to_sympy"
        else:
            try:
                state_equation = self._equation_from_model_state(model, probe.shape[1])
                state_attempt = self._compare_equation_to_native(
                    model,
                    state_equation,
                    probe,
                    source="model_state",
                )
                attempts.append(state_attempt)
                selected_attempt = state_attempt
                if state_attempt["status"] == "verified":
                    selected_equation = state_equation
                    equation_source = "model_state"
            except Exception as err:
                attempts.append(
                    {
                        "source": "model_state",
                        "equation": None,
                        "equation_sha256": None,
                        "status": "failed",
                        "reason": f"{err.__class__.__name__}: {err}",
                        "native_prediction_sha256": attempts[0].get(
                            "native_prediction_sha256"
                        ),
                        "replay_prediction_sha256": None,
                        "max_abs_error": None,
                        "max_rel_error": None,
                        "native_abs_scale": attempts[0].get("native_abs_scale"),
                        "allowed_max_abs_error": attempts[0].get(
                            "allowed_max_abs_error"
                        ),
                    }
                )
                selected_attempt = attempts[-1]

        state = model._state_dict() if hasattr(model, "_state_dict") else None
        evidence = {
            "version": self._FIDELITY_PROBE_VERSION,
            "status": "verified" if selected_equation is not None else "failed",
            "equation_source": equation_source,
            "equation": selected_equation,
            "equation_sha256": (
                hashlib.sha256(selected_equation.encode("utf-8")).hexdigest()
                if selected_equation is not None
                else None
            ),
            "raw_equation": raw_equation,
            "rtol": self._FIDELITY_RTOL,
            "atol": self._FIDELITY_ATOL,
            "probe_definition": probe_definition,
            "probe_input_sha256": self._hash_float_array(probe),
            "native_prediction_sha256": selected_attempt.get("native_prediction_sha256"),
            "replay_prediction_sha256": selected_attempt.get("replay_prediction_sha256"),
            "max_abs_error": selected_attempt.get("max_abs_error"),
            "max_rel_error": selected_attempt.get("max_rel_error"),
            "native_abs_scale": selected_attempt.get("native_abs_scale"),
            "allowed_max_abs_error": selected_attempt.get("allowed_max_abs_error"),
            "model_state_sha256": self._hash_json(state),
            "attempts": attempts,
        }
        return selected_equation, evidence

    def _write_progress_state(
        self,
        model,
        *,
        iteration: int,
        loss: float,
        equation: str | None,
        fidelity: dict[str, Any],
        first_discovered_attempt: int,
        first_discovered_minute: int,
        first_discovered_elapsed_seconds: float,
    ) -> None:
        if not self._progress_state_path or model is None:
            return
        model_state = model._state_dict() if hasattr(model, "_state_dict") else None
        payload = {
            "equation": equation,
            "loss": loss,
            "internal_loss": loss,
            "complexity": self._estimate_complexity(equation),
            "iteration": int(iteration),
            "first_discovered_attempt": int(first_discovered_attempt),
            "first_discovered_minute": int(first_discovered_minute),
            "first_discovered_elapsed_seconds": round(
                float(first_discovered_elapsed_seconds), 6
            ),
            "source": self._CANDIDATE_SOURCE,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fidelity": fidelity,
            "model_state": model_state,
        }
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            temporary_path = f"{self._progress_state_path}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self._progress_state_path)
        except Exception:
            pass

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="JAXSRRegressor.fit",
        )
        try:
            from jaxsr import SymbolicRegressor
        except Exception as err:  # pragma: no cover - exercised in integration env
            raise ImportError(
                "JAXSRRegressor 需要安装 jax、jaxlib、jaxsr；请先创建/激活 sim_jaxsr 环境。"
            ) from err

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        library = self._build_basis_library(int(X_arr.shape[1]))
        self._export_equation = None
        self._fidelity_evidence = None
        started_at = time.time()
        best_model = None
        best_loss = float("inf")
        iteration = 0
        while True:
            model = self._build_model(SymbolicRegressor, library, iteration)
            model.fit(X_arr, y_arr)
            loss = self._training_mse(model, X_arr, y_arr)
            if best_model is None or loss < best_loss:
                best_model = model
                best_loss = loss
                first_discovered_attempt = iteration + 1
                first_discovered_elapsed_seconds = max(0.0, time.time() - started_at)
                first_discovered_minute = max(
                    1,
                    int(math.ceil(first_discovered_elapsed_seconds / 60.0)),
                )
                pending_fidelity = {
                    "version": self._FIDELITY_PROBE_VERSION,
                    "status": "pending",
                    "reason": "当前 native best 正在执行 export fidelity 校验",
                    "model_state_sha256": self._hash_json(
                        best_model._state_dict() if hasattr(best_model, "_state_dict") else None
                    ),
                }
                self._export_equation = None
                self._fidelity_evidence = pending_fidelity
                self._write_progress_state(
                    best_model,
                    iteration=iteration + 1,
                    loss=best_loss,
                    equation=None,
                    fidelity=pending_fidelity,
                    first_discovered_attempt=first_discovered_attempt,
                    first_discovered_minute=first_discovered_minute,
                    first_discovered_elapsed_seconds=first_discovered_elapsed_seconds,
                )
                try:
                    export_equation, fidelity = self._prepare_model_export(best_model, X_arr)
                except Exception as err:
                    export_equation = None
                    fidelity = {
                        "version": self._FIDELITY_PROBE_VERSION,
                        "status": "failed",
                        "reason": f"{err.__class__.__name__}: {err}",
                        "model_state_sha256": self._hash_json(
                            best_model._state_dict()
                            if hasattr(best_model, "_state_dict")
                            else None
                        ),
                    }
                self._export_equation = export_equation
                self._fidelity_evidence = fidelity
                self._write_progress_state(
                    best_model,
                    iteration=iteration + 1,
                    loss=best_loss,
                    equation=export_equation,
                    fidelity=fidelity,
                    first_discovered_attempt=first_discovered_attempt,
                    first_discovered_minute=first_discovered_minute,
                    first_discovered_elapsed_seconds=first_discovered_elapsed_seconds,
                )
            iteration += 1
            if self._timeout_in_seconds is None:
                break
            elapsed = time.time() - started_at
            if elapsed >= max(0.0, self._timeout_in_seconds - self._timeout_guard_seconds):
                break
        self.model = best_model
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)

    def serialize(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if not hasattr(self.model, "_state_dict"):
            raise RuntimeError("JAXSR export fidelity fail-closed: 模型不支持状态冻结")
        params = {k: v for k, v in self.params.items() if k != "basis_library"}
        return json.dumps(
            {
                "mode": "jaxsr_state",
                "serialization_version": 2,
                "params": params,
                "contract": {
                    "n_features": self._contract_n_features,
                    "feature_names": self._contract_feature_names,
                    "target_name": self._contract_target_name,
                },
                "state": self.model._state_dict(),
                "equation": self._export_equation,
                "fidelity": self._fidelity_evidence,
            },
            ensure_ascii=False,
        )

    @classmethod
    def deserialize(cls, payload):
        try:
            obj = json.loads(payload)
        except Exception:
            return BaseWrapper.deserialize(payload)
        if not isinstance(obj, dict) or obj.get("mode") != "jaxsr_state":
            return BaseWrapper.deserialize(payload)
        contract = dict(obj.get("contract") or {})
        inst = cls(
            n_features=contract.get("n_features"),
            feature_names=contract.get("feature_names"),
            target_name=contract.get("target_name"),
            **dict(obj.get("params") or {}),
        )
        from jaxsr import SymbolicRegressor

        inst.model = SymbolicRegressor._from_dict(obj["state"])
        inst._export_equation = obj.get("equation")
        inst._fidelity_evidence = obj.get("fidelity")
        if inst._export_equation is not None:
            evidence = inst._fidelity_evidence
            definition = evidence.get("probe_definition") if isinstance(evidence, dict) else None
            values = definition.get("values") if isinstance(definition, dict) else None
            if not isinstance(values, list) or not values:
                raise RuntimeError(
                    "JAXSR export fidelity fail-closed: 序列化模型缺少可重放探针"
                )
            probe = np.asarray(values, dtype=float)
            restored_attempt = cls._compare_equation_to_native(
                inst.model,
                inst._export_equation,
                probe,
                source="deserialize_replay",
            )
            evidence_integrity_ok = all(
                (
                    evidence.get("probe_input_sha256") == cls._hash_float_array(probe),
                    evidence.get("equation_sha256")
                    == hashlib.sha256(inst._export_equation.encode("utf-8")).hexdigest(),
                    evidence.get("model_state_sha256") == cls._hash_json(obj["state"]),
                )
            )
            if restored_attempt.get("status") != "verified" or not evidence_integrity_ok:
                raise RuntimeError(
                    "JAXSR export fidelity fail-closed: 反序列化后的预测或证据哈希不一致"
                )
        return inst

    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if not isinstance(self._export_equation, str) or not self._export_equation.strip():
            raise RuntimeError(
                "JAXSR export fidelity fail-closed: native predict 无可验证的忠实符号表达式"
            )
        return self._export_equation

    def get_total_equations(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        equations = [self.get_optimal_equation()]
        pareto = getattr(self.model, "pareto_front_", None)
        if pareto:
            for item in pareto:
                expr = getattr(item, "expression", None)
                if callable(expr):
                    try:
                        equations.append(str(expr()))
                    except Exception:
                        pass
        return [eq for idx, eq in enumerate(equations) if eq and eq not in equations[:idx]]

    def export_canonical_symbolic_program(self):
        artifact = normalize_external_infix_artifact(
            self.get_optimal_equation(),
            tool_name="jaxsr",
            expected_n_features=self._contract_n_features,
            shift_one_based=False,
        )
        artifact["fidelity_check"] = deepcopy(self._fidelity_evidence or {})
        return artifact
