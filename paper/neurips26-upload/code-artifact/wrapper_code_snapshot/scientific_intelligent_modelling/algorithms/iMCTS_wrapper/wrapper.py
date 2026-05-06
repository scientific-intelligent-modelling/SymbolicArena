"""
iMCTS note

note MCTS-4-SR note note BaseWrapper note 
- fit(X, y): note
- predict(X): note
- get_optimal_equation(): note note 
- get_total_equations(): note note 

note 
- iMCTS.Regrssor note (n_features, n_samples) note (n_samples, n_features) note
- Regressor.fit() note (simplified_expr, vec_expr, eval_count, path)
- note eval('lambda x: {vec_expr}') note numpy note f(x)
- note/note note note vec_expr note
"""

import os
import sys
import json
import time
from typing import Any, Dict, Optional, List

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_imcts_artifact


def _default_eval_context(user_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """note eval note note  
    iMCTS default note regressor note sin/cos/exp/log/tanh note 
    """
    ctx = {
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'exp': np.exp,
        'log': np.log,
        'tanh': np.tanh,
    }
    if isinstance(user_ctx, dict):
        ctx.update(user_ctx)
    return ctx


class iMCTSRegressor(BaseWrapper):
    """iMCTS note """
    _PROGRESS_STATE_FILENAME = ".imcts_current_best.json"

    def __init__(self, **kwargs):
        # note note iMCTS.Regressor
        self.params: Dict[str, Any] = dict(kwargs) if kwargs else {}
        self.params.setdefault("ops", ["+", "-", "*", "/", "sin", "cos", "exp", "log", "R"])
        self.params.setdefault("max_depth", 6)
        self.params.setdefault("K", 500)
        self.params.setdefault("c", 4.0)
        self.params.setdefault("gamma", 0.5)
        self.params.setdefault("gp_rate", 0.2)
        self.params.setdefault("mutation_rate", 0.1)
        self.params.setdefault("exploration_rate", 0.2)
        self.params.setdefault("max_single_arity_ops", 999)
        self.params.setdefault("max_constants", 10)
        self.params.setdefault("max_expressions", 2000000)
        self.params.setdefault("verbose", False)
        self.params.setdefault("optimization_method", "LN_NELDERMEAD")
        self._exp_path = self.params.get('exp_path')
        self._exp_name = self.params.get('exp_name')
        self._contract_n_features = self.params.pop("n_features", None)
        self._contract_feature_names = self.params.pop("feature_names", None)
        self._contract_target_name = self.params.pop("target_name", None)

        # note
        self._best_expr_simplified: Optional[str] = None
        self._best_expr_vector: Optional[str] = None
        self._eval_count: Optional[int] = None
        self._best_path: Optional[int] = None
        self._n_features: Optional[int] = None

        # note note 
        self._eval_context: Dict[str, Any] = _default_eval_context(self.params.get('context'))

        # note fit note note note 
        self._runtime_regressor = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

    @classmethod
    def _resolve_progress_state_path(cls, exp_path: Optional[str], exp_name: Optional[str]) -> Optional[str]:
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(
            os.path.abspath(exp_path.strip()),
            exp_name.strip(),
            cls._PROGRESS_STATE_FILENAME,
        )

    def _write_progress_state(self, *, equation: str, score: Optional[float], evaluations: Optional[int]):
        if not self._progress_state_path:
            return
        if not isinstance(equation, str) or not equation.strip():
            return
        payload = {
            "equation": equation,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "evaluations": int(evaluations) if isinstance(evaluations, (int, float)) else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            with open(self._progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ============ note API ============
    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="iMCTSRegressor.fit",
        )
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2:
            raise ValueError("iMCTS note (n_samples, n_features)")
        self._n_features = int(X.shape[1])

        # note note sys.path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(base_dir, 'MCTS-4-SR')
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

        # note iMCTS
        from iMCTS.regressor import Regressor as _MCTSRegressor

        # iMCTS note (n_features, n_samples)
        x_train = X.T
        y_train = y

        # note iMCTS note
        allowed_keys = {
            'ops', 'arity_dict', 'context', 'max_depth', 'K', 'c', 'gamma',
            'gp_rate', 'mutation_rate', 'exploration_rate', 'max_single_arity_ops',
            'max_constants', 'max_expressions', 'verbose', 'reward_func',
            'optimization_method'
        }
        mcts_kwargs = {k: v for k, v in self.params.items() if k in allowed_keys}

        # note
        reg = _MCTSRegressor(
            x_train=x_train,
            y_train=y_train,
            progress_callback=self._write_progress_state if self._progress_state_path else None,
            **mcts_kwargs,
        )
        self._runtime_regressor = reg
        simplified_expr, vec_expr, eval_count, path = reg.fit(seed=self.params.get('seed'))

        # note
        self._best_expr_simplified = simplified_expr
        self._best_expr_vector = vec_expr
        self._eval_count = int(eval_count) if eval_count is not None else None
        if path is not None and not isinstance(path, (list, tuple)):
            self._best_path = int(path)
        else:
            # iMCTS note path note note
            self._best_path = path

        self._write_progress_state(
            equation=self._best_expr_simplified or self._best_expr_vector or "",
            score=None,
            evaluations=self._eval_count,
        )

        return self

    def predict(self, X):
        if not isinstance(self._best_expr_vector, str) or not self._best_expr_vector:
            raise ValueError("note")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("iMCTS note (n_samples, n_features)")

        # iMCTS note (n_features, n_samples)
        XT = X.T

        # note note predict note 
        if self._runtime_regressor is not None:
            return self._runtime_regressor.predict(XT, self._best_expr_vector)

        # note
        try:
            func = eval(f'lambda x: {self._best_expr_vector}', self._eval_context)
            y_pred = func(XT)
            return np.asarray(y_pred)
        except Exception as e:
            raise RuntimeError(f"iMCTS note: {e}")

    def get_optimal_equation(self):
        # note note/note 
        return self._best_expr_simplified or ""

    def get_total_equations(self):
        # current note
        return [self._best_expr_simplified] if self._best_expr_simplified else []

    # ============ note / note ============
    def serialize(self):
        state = {
            'params': self.params,
            'expr_simplified': self._best_expr_simplified,
            'expr_vector': self._best_expr_vector,
            'eval_count': self._eval_count,
            'best_path': self._best_path,
            'n_features': self._n_features,
            # note note default note note 
            'context_keys': list((self.params.get('context') or {}).keys())
        }
        return json.dumps(state, ensure_ascii=False)

    @classmethod
    def deserialize(cls, payload: str):
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst._best_expr_simplified = obj.get('expr_simplified')
        inst._best_expr_vector = obj.get('expr_vector')
        inst._eval_count = obj.get('eval_count')
        inst._best_path = obj.get('best_path')
        inst._n_features = obj.get('n_features')
        # note note+note
        inst._runtime_regressor = None
        return inst

    def export_canonical_symbolic_program(self):
        equation = self.get_optimal_equation()
        if not equation:
            raise ValueError("iMCTS current note")
        return normalize_imcts_artifact(
            equation,
            expected_n_features=self._n_features,
        )

    def __str__(self) -> str:
        lines: List[str] = ["iMCTSRegressor(tool='iMCTS')"]
        if self._best_expr_simplified:
            lines.append(f"note: {self._best_expr_simplified}")
        if self._eval_count is not None:
            lines.append(f"note: {self._eval_count}")
        return "\n".join(lines)
