"""RAG-SR wrapper backed by EvolutionaryForestRegressor.

RAG-SR note note `evolutionary_forest` note 
default note `rag_sr.py` notecurrent note
benchmark note 
"""

from __future__ import annotations

import base64
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_ragsr_artifact


class RAGSRRegressor(BaseWrapper):
    """RAG-SR benchmark wrapper using EvolutionaryForestRegressor."""

    _DEFAULT_PARAMS = {
        "n_gen": 100,
        "n_pop": 200,
        "select": "AutomaticLexicase",
        "cross_pb": 0.9,
        "mutation_pb": 0.1,
        "max_height": 10,
        "ensemble_size": 1,
        "initial_tree_size": "0-6",
        "gene_num": 10,
        "basic_primitives": "Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg",
        "base_learner": "RidgeCV",
        "ridge_alphas": "Auto",
        "verbose": False,
        "boost_size": None,
        "normalize": "MinMax",
        "external_archive": 1,
        "max_trees": 10000,
        "library_clustering_mode": "Worst",
        "pool_addition_mode": "Smallest~Auto",
        "pool_hard_instance_interval": 10,
        "random_order_replacement": True,
        "pool_based_addition": True,
        "semantics_length": 50,
        "change_semantic_after_deletion": True,
        "include_subtree_to_lib": True,
        "library_updating_mode": "Recent",
        # note RAG-SR default note Target encoding note fit note
        # categorical_features=np.zeros(X.shape[1]) current benchmark note
        # note note wrapper note fit note False note 
        "categorical_encoding": "Target",
        "root_crossover": True,
        "scaling_before_replacement": False,
        "score_func": "R2",
        "number_of_invokes": 0,
        "mutation_scheme": "uniform-plus",
        "environmental_selection": None,
        "record_training_data": False,
        "complementary_replacement": False,
        "validation_size": 0,
        "constant_type": "Float",
        "full_scaling_after_replacement": False,
        "neural_pool": 0.1,
        "neural_pool_num_of_functions": 5,
        "weight_of_contrastive_learning": 0.05,
        "neural_pool_dropout": 0.1,
        "neural_pool_transformer_layer": 1,
        "neural_pool_hidden_size": 64,
        "neural_pool_mlp_layers": 3,
        "selective_retrain": True,
        "negative_data_augmentation": True,
        "negative_local_search": False,
        "time_limit": None,
    }
    _META_PARAMS = {
        "cpu_num_threads",
        "exp_name",
        "exp_path",
        "problem_name",
        "seed",
        "n_features",
        "feature_names",
        "target_name",
        "timeout_in_seconds",
        "progress_snapshot_interval_seconds",
        "task_label",
        "task_global_index",
        "expected_dataset_rel",
        "expected_dataset_dir",
        "benchmark_variant",
        "component_notes",
    }
    _ALLOWED_PARAMS = set(_DEFAULT_PARAMS) | {"random_state", "categorical_features"}
    _INT_PARAMS = {
        "n_gen",
        "n_pop",
        "max_height",
        "ensemble_size",
        "gene_num",
        "external_archive",
        "max_trees",
        "pool_hard_instance_interval",
        "semantics_length",
        "number_of_invokes",
        "validation_size",
        "neural_pool_num_of_functions",
        "neural_pool_transformer_layer",
        "neural_pool_hidden_size",
        "neural_pool_mlp_layers",
        "random_state",
    }
    _FLOAT_PARAMS = {
        "cross_pb",
        "mutation_pb",
        "neural_pool",
        "weight_of_contrastive_learning",
        "neural_pool_dropout",
        "time_limit",
    }

    def __init__(self, **kwargs: Any):
        raw_kwargs = dict(kwargs or {})
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._seed = raw_kwargs.get("seed")
        self._cpu_num_threads = self._as_positive_int(raw_kwargs.get("cpu_num_threads"), default=4)
        self._timeout_in_seconds = self._as_positive_float(raw_kwargs.get("timeout_in_seconds"))
        self._experiment_dir = self._resolve_experiment_dir(raw_kwargs)
        self.params, self._fit_kwargs = self._validate_and_normalize_params(raw_kwargs)
        if self._timeout_in_seconds is not None and self.params.get("time_limit") is None:
            # note EvolutionaryForest note note 
            self.params["time_limit"] = max(1.0, self._timeout_in_seconds - 5.0)
        self.model = None
        self._equation = None
        self._canonical_artifact = None

    @staticmethod
    def _as_positive_float(value: Any) -> float | None:
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

    @staticmethod
    def _as_positive_int(value: Any, *, default: int) -> int:
        try:
            value = int(value)
        except Exception:
            return int(default)
        return value if value > 0 else int(default)

    @staticmethod
    def _configure_cpu_threads(threads: int) -> None:
        value = str(max(1, int(threads)))
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = value
        try:
            import torch

            torch.set_num_threads(int(value))
            try:
                torch.set_num_interop_threads(int(value))
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _threadpool_limits_context(threads: int):
        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            return nullcontext()
        return threadpool_limits(limits=max(1, int(threads)))

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
    def _validate_and_normalize_params(cls, raw_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        params = dict(raw_params or {})
        seed = params.get("seed")
        for key in cls._META_PARAMS:
            params.pop(key, None)

        if seed is not None and "random_state" not in params:
            params["random_state"] = int(seed)

        fit_kwargs = {}
        if "categorical_features" in params:
            fit_kwargs["categorical_features"] = params.pop("categorical_features")

        for key, value in cls._DEFAULT_PARAMS.items():
            params.setdefault(key, deepcopy(value))

        unknown = sorted(set(params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "RAG-SR note: {} current note: {}".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        for key in cls._INT_PARAMS:
            if key in params and params[key] is not None:
                params[key] = int(params[key])
        for key in cls._FLOAT_PARAMS:
            if key in params and params[key] is not None:
                params[key] = float(params[key])
        return params, fit_kwargs

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="RAGSRRegressor.fit",
        )
        self._configure_cpu_threads(self._cpu_num_threads)
        with self._threadpool_limits_context(self._cpu_num_threads):
            from evolutionary_forest.forest import EvolutionaryForestRegressor

            x_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            self.model = EvolutionaryForestRegressor(**self.params)
            self._install_current_best_callback()
            fit_kwargs = dict(self._fit_kwargs)
            if self.params.get("categorical_encoding") is not None:
                fit_kwargs.setdefault("categorical_features", [False] * x_arr.shape[1])
            self.model.fit(x_arr, y_arr, **fit_kwargs)
        self._equation = self._extract_model_expression()
        self._write_current_best_snapshot(source="final")
        return self

    def _install_current_best_callback(self) -> None:
        if self.model is None or self._experiment_dir is None:
            return
        original_callback = getattr(self.model, "callback", None)
        if not callable(original_callback):
            return

        def _callback_with_snapshot():
            result = original_callback()
            self._write_current_best_snapshot(source="callback")
            return result

        setattr(self.model, "callback", _callback_with_snapshot)

    def _write_current_best_snapshot(self, *, source: str) -> None:
        if self.model is None or self._experiment_dir is None:
            return
        try:
            equation = self._extract_model_expression()
        except Exception:
            return
        if not equation:
            return
        payload = {
            "tool": "ragsr",
            "equation": str(equation),
            "iteration": getattr(self.model, "current_gen", None),
            "source": source,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            hof = getattr(self.model, "hof", None)
            if hof:
                fitness = getattr(hof[0], "fitness", None)
                values = getattr(fitness, "values", None)
                if values:
                    payload["score"] = float(values[0])
        except Exception:
            pass
        try:
            path = self._experiment_dir / ".ragsr_current_best.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return

    def predict(self, X):
        if self.model is not None:
            return np.asarray(self.model.predict(np.asarray(X, dtype=float))).reshape(-1)
        if self._equation:
            return self._predict_from_equation(X)
        else:
            raise RuntimeError("RAG-SR note")

    def _predict_from_equation(self, X):
        """note note pickle note EF note """
        artifact = self.export_canonical_symbolic_program()
        expression = artifact.get("normalized_expression")
        variables = artifact.get("variables") or []
        if not expression:
            raise RuntimeError("RAG-SR canonical note note")

        try:
            import sympy as sp
        except ModuleNotFoundError as exc:
            raise RuntimeError("RAG-SR note sympy") from exc

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        ordered_variables = sorted(
            variables,
            key=lambda name: int(name[1:]) if isinstance(name, str) and name.startswith("x") and name[1:].isdigit() else 0,
        )
        if not ordered_variables:
            value = float(sp.sympify(expression))
            return np.full((X_arr.shape[0],), value, dtype=float)

        def _broadcast_maximum(*args):
            arrays = np.broadcast_arrays(*args)
            return np.maximum.reduce(arrays)

        def _broadcast_minimum(*args):
            arrays = np.broadcast_arrays(*args)
            return np.minimum.reduce(arrays)

        def _safe_amax(values, axis=None):
            if isinstance(values, (tuple, list)):
                return _broadcast_maximum(*values)
            return np.amax(values, axis=axis)

        def _safe_amin(values, axis=None):
            if isinstance(values, (tuple, list)):
                return _broadcast_minimum(*values)
            return np.amin(values, axis=axis)

        symbols = [sp.Symbol(name) for name in ordered_variables]
        parsed_expression = sp.sympify(expression)
        func = sp.lambdify(
            symbols,
            parsed_expression,
            modules=[
                {
                    "Max": _broadcast_maximum,
                    "Min": _broadcast_minimum,
                    "amax": _safe_amax,
                    "amin": _safe_amin,
                },
                "numpy",
            ],
        )
        args = [X_arr[:, int(name[1:])] for name in ordered_variables]
        try:
            pred = np.asarray(func(*args), dtype=float)
        except ValueError:
            scalar_func = sp.lambdify(
                symbols,
                parsed_expression,
                modules=[{"Max": max, "Min": min, "Abs": abs}, "math"],
            )
            values = []
            for row in X_arr:
                row_args = [float(row[int(name[1:])]) for name in ordered_variables]
                values.append(float(scalar_func(*row_args)))
            pred = np.asarray(values, dtype=float)
        if pred.ndim == 0:
            pred = np.full((X_arr.shape[0],), float(pred), dtype=float)
        return pred.reshape(-1)

    def _extract_model_expression(self) -> str:
        if self.model is None:
            raise RuntimeError("RAG-SR note")
        if hasattr(self.model, "model"):
            value = self.model.model()
            if value is not None:
                return str(value)
        raise RuntimeError("RAG-SR note")

    def get_optimal_equation(self):
        if self._equation is None:
            self._equation = self._extract_model_expression()
        return self._equation

    def get_total_equations(self):
        equation = self.get_optimal_equation()
        return [equation] if equation else []

    def export_canonical_symbolic_program(self):
        if self._canonical_artifact is None:
            self._canonical_artifact = normalize_ragsr_artifact(
                self.get_optimal_equation(),
                expected_n_features=self._contract_n_features,
            )
        return deepcopy(self._canonical_artifact)

    def serialize(self):
        """note note EvolutionaryForestRegressor note pickle """
        state = {
            "params": self.params,
            "fit_kwargs": self._fit_kwargs,
            "equation": self.get_optimal_equation(),
            "canonical_artifact": self.export_canonical_symbolic_program(),
            "contract_n_features": self._contract_n_features,
            "contract_feature_names": self._contract_feature_names,
            "contract_target_name": self._contract_target_name,
            "seed": self._seed,
        }
        return base64.b64encode(pickle.dumps(state)).decode("utf-8")

    @classmethod
    def deserialize(cls, instance_b64):
        state = pickle.loads(base64.b64decode(instance_b64))
        instance = cls(
            n_features=state.get("contract_n_features"),
            feature_names=state.get("contract_feature_names"),
            target_name=state.get("contract_target_name"),
            seed=state.get("seed"),
            **(state.get("params") or {}),
        )
        instance._fit_kwargs = state.get("fit_kwargs") or {}
        instance._equation = state.get("equation")
        instance._canonical_artifact = state.get("canonical_artifact")
        instance.model = None
        return instance
