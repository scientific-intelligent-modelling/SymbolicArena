"""uDSR trunk wrapper.

current note uDSR note DSO/GP/LINEAR note note full uDSR note
AIF note LSPT note encoder-controller 
"""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor
from scientific_intelligent_modelling.benchmarks.normalizers import annotate_udsr_trunk_artifact


class UDSRRegressor(DSORegressor):
    """uDSR-trunk benchmark wrapper built on the DSO/uDSR source tree."""

    _PROGRESS_STATE_FILENAME = ".udsr_current_best.json"
    _COMPONENT_METADATA_KEYS = {"benchmark_variant", "component_flags", "component_notes"}
    _DEFAULT_TASK = {
        "task_type": "regression",
        "function_set": [
            "add",
            "sub",
            "mul",
            "div",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            1.0,
            "const",
            "poly",
        ],
        "metric": "inv_nrmse",
        "metric_params": [1.0],
        "threshold": 1e-12,
        "protected": False,
        "poly_optimizer_params": {
            "degree": 3,
            "coef_tol": 1e-6,
            "regressor": "dso_least_squares",
            "regressor_params": {
                "cutoff_p_value": 1.0,
                "n_max_terms": None,
                "coef_tol": 1e-6,
            },
        },
    }
    _DEFAULT_TRAINING = {
        "batch_size": 1000,
        "n_samples": 2000000,
        "epsilon": 0.05,
        "baseline": "R_e",
        "n_cores_batch": 1,
    }
    _DEFAULT_POLICY_OPTIMIZER = {
        "policy_optimizer_type": "pg",
        "learning_rate": 0.0005,
        "entropy_weight": 0.03,
        "entropy_gamma": 0.7,
    }
    _DEFAULT_PRIOR = {
        "length": {"min_": 4, "max_": 100, "on": True},
        "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
        "inverse": {"on": True},
        "trig": {"on": True},
        "const": {"on": True},
        "no_inputs": {"on": True},
        "uniform_arity": {"on": True},
        "soft_length": {"loc": 10, "scale": 5, "on": True},
        "domain_range": {"on": True},
    }
    _DEFAULT_GP_MELD = {
        "run_gp_meld": True,
        "population_size": 100,
        "generations": 20,
        "crossover_operator": "cxOnePoint",
        "p_crossover": 0.5,
        "mutation_operator": "multi_mutate",
        "p_mutate": 0.5,
        "tournament_size": 5,
        "train_n": 50,
        "mutate_tree_max": 3,
        "verbose": False,
        "parallel_eval": False,
    }
    _TASK_KEYS = DSORegressor._TASK_KEYS | {"poly_optimizer_params"}
    _TRAINING_KEYS = DSORegressor._TRAINING_KEYS | {"baseline"}

    @classmethod
    def _build_config(cls, raw_kwargs):
        sanitized_kwargs = dict(raw_kwargs or {})
        for key in cls._COMPONENT_METADATA_KEYS:
            sanitized_kwargs.pop(key, None)
        config = super()._build_config(sanitized_kwargs)
        gp_meld = deepcopy(cls._DEFAULT_GP_MELD)
        if isinstance(config.get("gp_meld"), dict):
            gp_meld.update(config["gp_meld"])
        config["gp_meld"] = gp_meld
        return config

    @staticmethod
    def _build_fit_config(base_config: Dict[str, Any], X, y) -> Dict[str, Any]:
        """note CSV note uDSR note GP-meld note """
        config = deepcopy(base_config)
        experiment = config.setdefault("experiment", {})
        logdir = experiment.get("logdir")
        if not logdir:
            logdir = tempfile.mkdtemp(prefix="udsr-fit-")
            experiment["logdir"] = logdir
        os.makedirs(logdir, exist_ok=True)
        exp_name = experiment.get("exp_name") or "udsr_regression"
        dataset_path = UDSRRegressor._short_dataset_path(str(exp_name), "udsr_regression")
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        stacked = np.concatenate([x_arr, y_arr], axis=1)
        np.savetxt(dataset_path, stacked, delimiter=",")
        config.setdefault("task", {})
        config["task"]["dataset"] = dataset_path
        gp_meld = config.get("gp_meld") or {}
        gp_meld.setdefault("run_gp_meld", True)
        gp_meld.setdefault("parallel_eval", False)
        config["gp_meld"] = gp_meld
        return config

    def export_canonical_symbolic_program(self):
        artifact = super().export_canonical_symbolic_program()
        artifact["tool_name"] = "udsr"
        artifact["normalization_mode"] = "udsr_dso_sympy_expr"
        return annotate_udsr_trunk_artifact(artifact)
