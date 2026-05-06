# algorithms/llmsr_wrapper/wrapper.py
"""
LLMSR note SIM note 

- note `llmsr_regressor.LLMSRRegressor` note 
- note (X, y) note CSV note LLMSRRegressor note 
- LLMSRRegressor note `exp_path/exp_name` note meta.json samples/top*.json note  
- note wrapper note`note + note` note `existing_exp_dir`
  note LLMSRRegressor note `predict` note 

note 
- note samples note note LLM note note 
- note problem_name / exp_path / exp_name / llm_config_path note 
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import glob
from typing import Any, Dict, Optional, List
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_llmsr_artifact


def _llmsr_root_dir() -> str:
    """note llmsr note """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llmsr")


def _default_llm_config_path() -> str:
    """
    default note llm.config note 

    default note benchmark note note llmsr / drsr note
    LLM note 
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(
        repo_root,
        "exp-planning",
        "02.e1_selection_validation",
        "llm_configs",
        "benchmark_llm.config",
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _import_core_regressor():
    """
    note LLMSRRegressor 

    note noteenvironment note note
    note import note 
    """
    import sys

    root = _llmsr_root_dir()
    if root not in sys.path:
        sys.path.insert(0, root)
    from llmsr_regressor import LLMSRRegressor as CoreLLMSRRegressor  # type: ignore

    return CoreLLMSRRegressor


def _is_single_line_formula_function(func_source: Any) -> bool:
    """note return note equation note """
    if not isinstance(func_source, str) or not func_source.strip():
        return False
    try:
        tree = ast.parse(func_source)
    except Exception:
        return False

    equation_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            equation_func = node
            break
    if equation_func is None:
        return False

    body = list(equation_func.body)
    while (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return False
    return True


def _infer_n_features_from_function_signature(func_source: Any) -> int | None:
    """note equation note """
    if not isinstance(func_source, str) or not func_source.strip():
        return None
    try:
        tree = ast.parse(func_source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            arg_names = [arg.arg for arg in node.args.args]
            feature_names = [name for name in arg_names if name != "params"]
            return len(feature_names)
    return None


class LLMSRRegressor(BaseWrapper):
    """
    note SIM note LLMSR note 

    note 
    - `fit(X, y)` note CSV note LLMSRRegressor note `fit()` 
    - `predict(X)` note exp_dir note LLMSRRegressor note `predict(X)` 
    - `serialize()/deserialize()` note note
      LLMSRRegressor note 

    note note SymbolicRegressor(..., **params) note  
    - problem_name: note/note note note default note 
    - background:   note note prompt
    - llm_config_path: note llm.config note default note llm.config 
    - exp_path:     note default ./experiments 
    - exp_name:     note note note 
    - max_params:   note default 10 
    - niterations:  note default 2500 
    - samples_per_iteration: note default 4 
    - seed:         note note note 
    """

    def __init__(self, **kwargs: Any):
        # note note
        self.params: Dict[str, Any] = dict(kwargs) if kwargs else {}
        self.params.setdefault("timeout_in_seconds", 3600)
        self.params.setdefault("max_params", 10)
        self.params.setdefault("niterations", 100000)
        self.params.setdefault("samples_per_iteration", 4)
        self.params.setdefault("inject_prompt_semantics", False)
        self.params.setdefault(
            "background",
            "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
        )
        self.params.setdefault("persist_all_samples", False)
        self.params.setdefault("llm_config_path", _default_llm_config_path())

        # note note 
        self._core: Optional[Any] = None

        # note
        self._exp_dir: Optional[str] = self.params.pop("existing_exp_dir", None) or self.params.pop("exp_dir", None)
        self._problem_name: Optional[str] = self.params.get("problem_name")
        self._n_features: Optional[int] = self.params.pop("n_features", None)
        self._feature_names: Optional[list[str]] = self.params.pop("feature_names", None)
        self._target_name: Optional[str] = self.params.pop("target_name", None)
        self._prompt_feature_names: Optional[list[str]] = self.params.pop("prompt_feature_names", None)
        self._prompt_target_name: Optional[str] = self.params.pop("prompt_target_name", None)

    # ------------------------------------------------------------------
    # note / note note
    # ------------------------------------------------------------------
    def serialize(self) -> str:
        """
        notecurrent wrapper note JSON note 

        note 
        - params: note
        - exp_dir: note LLMSRRegressor note
        - problem_name: note note core 
        """
        state = {
            "params": self.params,
            "exp_dir": self._exp_dir,
            "problem_name": self._problem_name,
            "n_features": self._n_features,
            "feature_names": self._feature_names,
            "target_name": self._target_name,
            "prompt_feature_names": self._prompt_feature_names,
            "prompt_target_name": self._prompt_target_name,
        }
        return json.dumps(state, ensure_ascii=False)

    @classmethod
    def deserialize(cls, payload: str) -> "LLMSRRegressor":
        """
        note JSON note wrapper 

        note note note 
        note note exp_dir note core note existing_exp_dir
        note`note` 
        """
        obj = json.loads(payload)
        inst = cls(**obj.get("params", {}))
        inst._exp_dir = obj.get("exp_dir")
        inst._problem_name = obj.get("problem_name") or inst.params.get("problem_name")
        inst._n_features = obj.get("n_features")
        inst._feature_names = obj.get("feature_names")
        inst._target_name = obj.get("target_name")
        inst._prompt_feature_names = obj.get("prompt_feature_names")
        inst._prompt_target_name = obj.get("prompt_target_name")
        return inst

    # ------------------------------------------------------------------
    # note
    # ------------------------------------------------------------------
    def _resolve_prompt_columns(self, n_features: int) -> tuple[list[str], str]:
        """note LLMSR note CSV note 

        `feature_names` / `target_name` note note
        `prompt_feature_names` / `prompt_target_name` note LLMSR note DRSR
        note prompt note 
        """
        prompt_feature_names = self._prompt_feature_names
        if not isinstance(prompt_feature_names, list) or len(prompt_feature_names) != n_features:
            prompt_feature_names = [f"x{i}" for i in range(n_features)]
        prompt_target_name = self._prompt_target_name
        if not isinstance(prompt_target_name, str) or not prompt_target_name.strip():
            prompt_target_name = "y"
        return list(prompt_feature_names), prompt_target_name.strip()

    def _can_reuse_existing_experiment(self) -> bool:
        """note """
        if not self._exp_dir:
            return False
        exp_dir = os.path.abspath(self._exp_dir)
        return (
            os.path.isdir(exp_dir)
            and os.path.isfile(os.path.join(exp_dir, "meta.json"))
            and os.path.isdir(os.path.join(exp_dir, "samples"))
        )

    def fit(self, X, y):
        """
        note LLMSR note 

        note 
        1. note (X, y) note CSV 
        2. note LLMSRRegressor note fit() 
        3. note exp_dir note serialize()/deserialize() note 
        """
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._n_features,
            feature_names=self._feature_names,
            target_name=self._target_name,
            context="LLMSRRegressor.fit",
        )
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).reshape(-1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X note y note: X.shape={X_arr.shape}, y.shape={y_arr.shape}")

        # note existing_exp_dir note 
        if self._can_reuse_existing_experiment():
            self._exp_dir = os.path.abspath(self._exp_dir)
            self._n_features = int(X_arr.shape[1])
            self._core = None
            return self

        # note problem_name note 
        problem_name = (self._problem_name or "").strip()
        if not problem_name:
            problem_name = "llmsr_problem"
        self._problem_name = problem_name

        # 1) note CSV note 
        tmp_dir = tempfile.mkdtemp(prefix="llmsr_data_")
        try:
            n_features = X_arr.shape[1]
            self._n_features = int(n_features)
            prompt_feature_names, prompt_target_name = self._resolve_prompt_columns(n_features)
            columns = prompt_feature_names + [prompt_target_name]
            data = np.column_stack([X_arr, y_arr])
            df = pd.DataFrame(data, columns=columns)
            csv_path = os.path.join(tmp_dir, f"{problem_name}.csv")
            df.to_csv(csv_path, index=False)

            # 2) note LLMSRRegressor
            Core = _import_core_regressor()

            llm_config_path = self.params.get("llm_config_path") or _default_llm_config_path()
            exp_path = self.params.get("exp_path") or os.path.join(os.getcwd(), "experiments")
            exp_name = self.params.get("exp_name")

            max_params = int(self.params.get("max_params", 10))
            niterations = int(self.params.get("niterations", 2500))
            samples_per_iter = int(self.params.get("samples_per_iteration", 4))
            seed = self.params.get("seed")

            wandb_cfg = None
            if self.params.get("use_wandb"):
                # note note 
                dataset_path = self.params.get("train_path")
                dataset_name = self.params.get("dataset_name")
                prompts_type = self.params.get("prompts_type")
                wandb_cfg = {
                    "project": self.params.get("wandb_project"),
                    "entity": self.params.get("wandb_entity"),
                    "name": self.params.get("wandb_name"),
                    "group": self.params.get("wandb_group"),
                    "tags": self.params.get("wandb_tags"),
                }
                # note notecurrent note prompts note/note note WandB note
                if prompts_type is not None:
                    wandb_cfg["prompts_type"] = prompts_type
                if dataset_path is not None:
                    wandb_cfg["dataset_path"] = dataset_path
                if dataset_name is not None:
                    wandb_cfg["dataset_name"] = dataset_name

            core = Core(
                problem_name=problem_name,
                data_csv=csv_path,
                llm_config_path=llm_config_path,
                background=self.params.get("background", "") or "",
                exp_path=exp_path,
                exp_name=exp_name,
                max_params=max_params,
                niterations=niterations,
                samples_per_iteration=samples_per_iter,
                persist_all_samples=bool(self.params.get("persist_all_samples", False)),
                timeout_in_seconds=self.params.get("timeout_in_seconds"),
                seed=seed,
                metadata_path=self.params.get("metadata_path"),
                feature_descriptions=self.params.get("feature_descriptions"),
                target_description=self.params.get("target_description"),
                # wrapper note prompt CSV note note 
                anonymize=False,
                wandb_config=wandb_cfg,
            )
            core.fit()

            # note note
            self._core = core
            self._exp_dir = getattr(core, "exp_dir_", None)
            if not self._exp_dir:
                # note core.fit() note exp_dir_ note
                self._exp_dir = os.path.join(exp_path, exp_name or problem_name)

        finally:
            # note fit note note core note exp_path note
            import shutil

            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        return self

    # ------------------------------------------------------------------
    # note
    # ------------------------------------------------------------------
    def _ensure_core(self) -> Any:
        """
        note self._core note 

        - notecurrent note note 
        - note note exp_dir note
          `existing_exp_dir` note LLMSRRegressor note 
        """
        if self._core is not None:
            return self._core

        if not self._exp_dir:
            raise RuntimeError("LLMSRRegressor: note exp_dir note")

        Core = _import_core_regressor()

        llm_config_path = self.params.get("llm_config_path") or _default_llm_config_path()
        problem_name = (self._problem_name or self.params.get("problem_name") or "llmsr_problem").strip()

        # note data_csv note LLMSRRegressor note meta.json note
        core = Core(
            problem_name=problem_name,
            data_csv="",
            llm_config_path=llm_config_path,
            background=self.params.get("background", "") or "",
            exp_path=os.path.dirname(self._exp_dir),
            exp_name=os.path.basename(self._exp_dir),
            max_params=int(self.params.get("max_params", 10)),
            niterations=int(self.params.get("niterations", 2500)),
            samples_per_iteration=int(self.params.get("samples_per_iteration", 4)),
            persist_all_samples=bool(self.params.get("persist_all_samples", False)),
            timeout_in_seconds=self.params.get("timeout_in_seconds"),
            seed=self.params.get("seed"),
            metadata_path=self.params.get("metadata_path"),
            feature_descriptions=self.params.get("feature_descriptions"),
            target_description=self.params.get("target_description"),
            existing_exp_dir=self._exp_dir,
        )
        # existing_exp_dir note LLMSRRegressor note 
        # note predict/note note 
        try:
            if os.path.isfile(os.path.join(self._exp_dir, "meta.json")):
                with open(os.path.join(self._exp_dir, "meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if core.feature_names_ is None and meta.get("feature_names"):
                    core.feature_names_ = meta.get("feature_names")
                if core.target_name_ is None and meta.get("target_name"):
                    core.target_name_ = meta.get("target_name")
        except Exception:
            pass
        # note/note
        core.is_fitted_ = False
        self._core = core
        return core

    def predict(self, X):
        """note LLMSR note """
        core = self._ensure_core()
        X_arr = np.asarray(X)
        return core.predict(X_arr)

    # ------------------ note note samples/top*.json note ------------------
    def _load_best_sample(self) -> Optional[Dict[str, Any]]:
        """
        note samples note note LLMSRRegressor note  

        note:
            note 'function' 'params' 'nmse'/'mse' notefieldnote note None 
        """
        if not self._exp_dir:
            return None

        samples_dir = os.path.join(self._exp_dir, "samples")
        if not os.path.isdir(samples_dir):
            return None

        candidates: List[str] = glob.glob(os.path.join(samples_dir, "top01_*.json"))
        if not candidates:
            candidates = glob.glob(os.path.join(samples_dir, "top*.json"))
        if not candidates:
            return None

        best_key: Optional[float] = None
        best_data: Optional[Dict[str, Any]] = None

        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            func = d.get("function")
            if not _is_single_line_formula_function(func):
                continue

            key_val: Optional[float] = None
            nmse = d.get("nmse")
            mse = d.get("mse")
            if isinstance(nmse, (int, float)):
                key_val = float(nmse)
            elif isinstance(mse, (int, float)):
                key_val = float(mse)
            else:
                score = d.get("score")
                if isinstance(score, (int, float)):
                    key_val = -float(score)

            if key_val is None:
                continue
            if best_key is None or key_val < best_key:
                best_key = key_val
                best_data = d

        return best_data

    def get_optimal_equation(self):
        """
        note note top note 'function' field  
        """
        best = self._load_best_sample()
        if not best:
            return ""
        func = best.get("function") or ""
        return str(func)

    def get_total_equations(self, n: Optional[int] = None):
        """
        note top*.json note noteRanking  
        """
        if not self._exp_dir:
            return []

        samples_dir = os.path.join(self._exp_dir, "samples")
        if not os.path.isdir(samples_dir):
            return []

        paths = glob.glob(os.path.join(samples_dir, "top*.json"))
        items: List[Dict[str, Any]] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            key_val: Optional[float] = None
            nmse = d.get("nmse")
            mse = d.get("mse")
            if isinstance(nmse, (int, float)):
                key_val = float(nmse)
            elif isinstance(mse, (int, float)):
                key_val = float(mse)
            else:
                score = d.get("score")
                if isinstance(score, (int, float)):
                    key_val = -float(score)
            if key_val is None:
                continue
            d["_sort_key"] = key_val
            items.append(d)

        # noteRanking
        items.sort(key=lambda d: d.get("_sort_key", float("inf")))

        if n is not None:
            try:
                n = int(n)
                items = items[: max(0, n)]
            except Exception:
                pass

        eqs: List[str] = []
        for d in items:
            func = d.get("function")
            if _is_single_line_formula_function(func):
                eqs.append(func)
        return eqs

    def export_canonical_symbolic_program(self):
        best = self._load_best_sample()
        if not best:
            raise ValueError("note LLMSR note")
        func = best.get("function") or ""
        if not _is_single_line_formula_function(func):
            raise ValueError("LLMSR current note")
        params = best.get("params")
        if not isinstance(params, list):
            params = None
        expected_n_features = self._n_features
        if expected_n_features is None:
            expected_n_features = _infer_n_features_from_function_signature(func)
        return normalize_llmsr_artifact(
            str(func),
            parameter_values=params,
            expected_n_features=expected_n_features,
        )
