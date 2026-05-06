from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
import base64
from typing import Sequence

import numpy as np


class BaseWrapper(ABC):
    """note"""
    
    @abstractmethod
    def fit(self, X, y):
        """note"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """note"""
        pass

    @abstractmethod
    def get_optimal_equation(self):
        """note"""
        pass

    @abstractmethod
    def get_total_equations(self):
        """note"""
        pass

    def _infer_tool_name(self):
        """note """
        module_name = getattr(self.__class__, "__module__", "") or ""
        marker = ".algorithms."
        if marker in module_name and "_wrapper" in module_name:
            try:
                suffix = module_name.split(marker, 1)[1]
                return suffix.split("_wrapper", 1)[0]
            except Exception:
                pass
        return self.__class__.__name__

    def export_canonical_symbolic_program(self):
        """note default note 

        Phase 1 note note 
        note wrapper note note 
        """
        from scientific_intelligent_modelling.benchmarks.artifact_schema import (
            build_canonical_symbolic_program,
        )

        raw_equation = self.get_optimal_equation()
        parameter_values = None
        if hasattr(self, "get_fitted_params"):
            try:
                parameter_values = self.get_fitted_params()
            except Exception:
                parameter_values = None
        return build_canonical_symbolic_program(
            tool_name=self._infer_tool_name(),
            raw_equation=raw_equation,
            parameter_values=parameter_values,
            normalization_mode="wrapper_raw",
        )

    def _validate_explicit_dataset_contract(
        self,
        X,
        *,
        n_features: int | None = None,
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        context: str | None = None,
    ) -> int:
        """note runner note 

        note 
        - `n_features` note notecurrent `X.shape[1]` note 
        - `feature_names` note note note 
        - `target_name` note note 
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            inferred_n_features = 1
        elif X_arr.ndim == 2:
            inferred_n_features = int(X_arr.shape[1])
        else:
            raise ValueError(
                f"{context or self.__class__.__name__}: note X note current ndim={X_arr.ndim}"
            )

        if n_features is not None:
            try:
                expected_n_features = int(n_features)
            except Exception as err:
                raise TypeError(
                    f"{context or self.__class__.__name__}: n_features note note: {err}"
                ) from err
            if expected_n_features != inferred_n_features:
                raise ValueError(
                    f"{context or self.__class__.__name__}: note "
                    f"n_features={expected_n_features}, note={inferred_n_features}"
                )

        if feature_names is not None:
            if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
                raise TypeError(
                    f"{context or self.__class__.__name__}: feature_names note"
                )
            feature_names_list = list(feature_names)
            if len(feature_names_list) != inferred_n_features:
                raise ValueError(
                    f"{context or self.__class__.__name__}: feature_names note "
                    f"len(feature_names)={len(feature_names_list)}, note={inferred_n_features}"
                )
            bad_names = [name for name in feature_names_list if not isinstance(name, str) or not name.strip()]
            if bad_names:
                raise ValueError(
                    f"{context or self.__class__.__name__}: feature_names note: {bad_names!r}"
                )

        if target_name is not None and (not isinstance(target_name, str) or not target_name.strip()):
            raise ValueError(
                f"{context or self.__class__.__name__}: target_name note"
            )

        return inferred_n_features

    def serialize(self):
        """note"""
        # notepicklenote
        instance_bytes = pickle.dumps(self)
        # notebase64note
        instance_b64 = base64.b64encode(instance_bytes).decode('utf-8')
        return instance_b64

    @classmethod
    def deserialize(cls, instance_b64):
        """note"""
        # notebase64note
        instance_bytes = base64.b64decode(instance_b64)
        # notepicklenote
        instance = pickle.loads(instance_bytes)
        return instance
        
    def __str__(self):
        """note"""
        return f"{self.__class__.__name__}()"
