"""QLattice note note 

note 
- note srkit/subprocess_runner note fit/predict/get_optimal_equation/get_total_equations
- note feyn.QLattice() note 
- note JSON note note
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from typing import List, Optional

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_qlattice_artifact
from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError


class QLatticeRegressor(BaseWrapper):
    """QLattice note note  

    note note  
    - n_epochs: int = 100 note
    - kind: str = 'regression' note
    - signif: int = 4 note note sympify note 
    - note QLattice.auto_run note
    """
    _PROGRESS_STATE_FILENAME = ".qlattice_current_best.json"

    def __init__(self, **kwargs) -> None:
        self.params = dict(kwargs)
        self.params.setdefault("n_epochs", 100)
        self.params.setdefault("kind", "regression")
        self.params.setdefault("criterion", "bic")
        self.params.setdefault("signif", 4)
        self.params.setdefault("threads", 4)
        self._contract_n_features = self.params.pop("n_features", None)
        self._contract_feature_names = self.params.pop("feature_names", None)
        self._contract_target_name = self.params.pop("target_name", None)
        self.model = None
        self._exp_path = self.params.get("exp_path")
        self._exp_name = self.params.get("exp_name")

        # QLattice note
        self._ql = None
        self._models = []
        self._best_model = None

        # note/note
        self._expr_str: Optional[str] = None  # note
        self._input_vars: List[str] = []
        self._output_name: str = 'y'
        self._lambdified = None
        # note note 
        self._equations: List[str] = []
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

    @classmethod
    def _resolve_progress_state_path(cls, exp_path, exp_name) -> Optional[str]:
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(
            os.path.abspath(exp_path.strip()),
            exp_name.strip(),
            cls._PROGRESS_STATE_FILENAME,
        )

    @staticmethod
    def _model_equation(model, signif: int) -> Optional[str]:
        try:
            return str(model.sympify(signif=signif))
        except Exception:
            try:
                return str(model)
            except Exception:
                return None

    def _criterion_name(self) -> Optional[str]:
        criterion = self.params.get("criterion", "bic")
        if criterion is None:
            return None
        return str(criterion)

    def _criterion_value(self, model, criterion_name: Optional[str]) -> Optional[float]:
        if criterion_name:
            try:
                value = getattr(model, criterion_name, None)
            except Exception:
                value = None
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        for fallback in ("bic", "aic"):
            try:
                value = getattr(model, fallback, None)
            except Exception:
                value = None
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return None

    def _estimate_complexity(self, equation: str) -> Optional[int]:
        try:
            import sympy as sp

            expr = sp.sympify(equation)
            return int(sum(1 for _ in sp.preorder_traversal(expr)))
        except Exception:
            return None

    def _select_best_model(self, models, criterion_name: Optional[str]):
        if not models:
            return None
        scored = []
        for model in models:
            value = self._criterion_value(model, criterion_name)
            if value is not None:
                scored.append((value, model))
        if scored:
            scored.sort(key=lambda item: item[0])
            return scored[0][1]
        return models[0]

    def _write_progress_state_from_models(self, models, *, epoch: int, signif: int, criterion_name: Optional[str]) -> None:
        if not self._progress_state_path or not models:
            return
        best_model = self._select_best_model(models, criterion_name)
        if best_model is None:
            return
        equation = self._model_equation(best_model, signif)
        if not isinstance(equation, str) or not equation.strip():
            return
        payload = {
            "equation": equation,
            "loss": self._criterion_value(best_model, criterion_name),
            "criterion": criterion_name,
            "complexity": self._estimate_complexity(equation),
            "epoch": int(epoch),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            with open(self._progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------- note ----------------
    def fit(self, X, y):
        """note QLattice note """
        import feyn
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="QLatticeRegressor.fit",
        )

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # note DataFrame
        n_features = X.shape[1]
        self._input_vars = [f"x{i}" for i in range(n_features)]
        self._output_name = self.params.get('output_name', self._contract_target_name or 'y')
        df = pd.DataFrame(X, columns=self._input_vars)
        df[self._output_name] = y

        # note QLattice note 
        self._ql = feyn.QLattice()

        # note auto_run note
        auto_args = {
            'data': df,
            'output_name': self._output_name,
            'kind': self.params.get('kind', 'regression'),
            'n_epochs': int(self.params.get('n_epochs', 100)),
        }
        # note QLattice.auto_run note note 
        for k in [
            'stypes', 'threads', 'max_complexity', 'query_string',
            'loss_function', 'criterion', 'sample_weights',
            'function_names', 'starting_models'
        ]:
            if k in self.params:
                auto_args[k] = self.params[k]

        total_epochs = int(auto_args.get('n_epochs', 100))
        if total_epochs < 1:
            raise ValueError('n_epochs note >= 1')
        signif = int(self.params.get('signif', 4))
        criterion_name = self._criterion_name()

        if self._progress_state_path:
            running_models = auto_args.get('starting_models')
            models = []
            epoch_args = dict(auto_args)
            epoch_args['n_epochs'] = 1
            for epoch in range(1, total_epochs + 1):
                if running_models:
                    epoch_args['starting_models'] = running_models
                else:
                    epoch_args.pop('starting_models', None)
                epoch_models = list(self._ql.auto_run(**epoch_args))
                # feyn note epoch note note 
                # note note epoch note 
                if not epoch_models:
                    continue
                models = epoch_models
                running_models = epoch_models
                self._write_progress_state_from_models(
                    models,
                    epoch=epoch,
                    signif=signif,
                    criterion_name=criterion_name,
                )
            if not models:
                raise NoValidOutputError('QLattice.auto_run note ')
        else:
            models = list(self._ql.auto_run(**auto_args))
            if not models:
                raise NoValidOutputError('QLattice.auto_run note ')

        self._models = models
        self._best_model = self._select_best_model(models, criterion_name)
        self.model = True

        # note
        self._expr_str = self._model_equation(self._best_model, signif)

        equations: List[str] = []
        for m in self._models:
            eq = self._model_equation(m, signif)
            if eq is not None:
                equations.append(eq)
        self._equations = equations

        # note lambdify note note 
        self._build_lambdify()
        return self

    def predict(self, X):
        """note """
        if self.model is None:
            raise ValueError('note note fit note ')

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self._input_vars):
            raise ValueError(f'note note {len(self._input_vars)} note note {X.shape[1]} note ')

        # prefernote feyn note note  note lambdify
        if self._best_model is not None:
            import pandas as pd
            df = pd.DataFrame(X, columns=self._input_vars)
            return np.asarray(self._best_model.predict(data=df))

        if self._lambdified is not None:
            cols = [X[:, i] for i in range(X.shape[1])]
            y_pred = self._lambdified(*cols)
            return np.asarray(y_pred)

        raise RuntimeError('note note QLattice note  ')

    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError('note note fit note ')
        if self._expr_str:
            return self._expr_str
        try:
            signif = int(self.params.get('signif', 4))
            return str(self._best_model.sympify(signif=signif))
        except Exception:
            return 'note'

    def get_total_equations(self, n: int | None = None):
        """note note  

        note:
            n: note note None note note 
        """
        if self.model is None:
            raise ValueError('note note fit note ')
        results: List[str] = []
        signif = int(self.params.get('signif', 4))
        if self._models:
            for m in self._models:
                try:
                    results.append(str(m.sympify(signif=signif)))
                except Exception:
                    results.append(str(m))
        elif self._equations:
            results = list(self._equations)
        elif self._expr_str:
            results = [self._expr_str]
        if isinstance(n, int) and n > 0:
            return results[:n]
        return results

    def export_canonical_symbolic_program(self):
        if self.model is None:
            raise ValueError('note note fit note ')
        return normalize_qlattice_artifact(
            self.get_optimal_equation(),
            expected_n_features=len(self._input_vars or []),
        )

    # ---------------- note/note ----------------
    def serialize(self):
        state = {
            'params': self.params,
            'expr': self._expr_str,
            'input_vars': self._input_vars,
            'output_name': self._output_name,
            'equations': self._equations,
        }
        return json.dumps(state, ensure_ascii=False)

    @classmethod
    def deserialize(cls, payload: str) -> 'QLatticeRegressor':
        obj = json.loads(payload)
        inst = cls(**obj.get('params', {}))
        inst.model = True
        inst._expr_str = obj.get('expr')
        inst._input_vars = obj.get('input_vars') or []
        inst._output_name = obj.get('output_name') or 'y'
        inst._equations = obj.get('equations') or []
        inst._build_lambdify()
        return inst

    # ---------------- note ----------------
    def _build_lambdify(self):
        if not self._expr_str or not self._input_vars:
            self._lambdified = None
            return
        try:
            import sympy as sp
            syms = sp.symbols(self._input_vars)
            expr = sp.sympify(self._expr_str)
            self._lambdified = sp.lambdify(syms, expr, modules='numpy')
        except Exception:
            self._lambdified = None

__all__ = ["QLatticeRegressor"]
