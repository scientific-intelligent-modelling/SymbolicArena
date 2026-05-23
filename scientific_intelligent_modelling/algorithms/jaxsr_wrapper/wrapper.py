from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_external_infix_artifact


class JAXSRRegressor(BaseWrapper):
    """JAXSR sparse basis-library 符号回归适配层。"""

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
        self.params = self._validate_and_normalize_params(raw_kwargs)
        self.model = None

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
        self.model = SymbolicRegressor(
            basis_library=library,
            max_terms=self.params["max_terms"],
            strategy=self.params["strategy"],
            information_criterion=self.params["information_criterion"],
            cv_folds=self.params["cv_folds"],
            regularization=self.params["regularization"],
            random_state=self.params.get("random_state"),
        )
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)

    def serialize(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if not hasattr(self.model, "_state_dict"):
            return super().serialize()
        params = {k: v for k, v in self.params.items() if k != "basis_library"}
        return json.dumps(
            {
                "mode": "jaxsr_state",
                "params": params,
                "contract": {
                    "n_features": self._contract_n_features,
                    "feature_names": self._contract_feature_names,
                    "target_name": self._contract_target_name,
                },
                "state": self.model._state_dict(),
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
        return inst

    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if hasattr(self.model, "to_sympy"):
            return str(self.model.to_sympy())
        return str(self.model.expression_)

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
        return normalize_external_infix_artifact(
            self.get_optimal_equation(),
            tool_name="jaxsr",
            expected_n_features=self._contract_n_features,
            shift_one_based=False,
        )
