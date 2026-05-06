from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
import base64
from typing import Sequence

import numpy as np


class BaseWrapper(ABC):
    """所有估计器的基类"""
    
    @abstractmethod
    def fit(self, X, y):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """使用模型进行预测"""
        pass

    @abstractmethod
    def get_optimal_equation(self):
        """获得最优方程"""
        pass

    @abstractmethod
    def get_total_equations(self):
        """获得所有方程"""
        pass

    def _infer_tool_name(self):
        """从模块路径推断工具名。"""
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
        """导出统一符号工件的默认实现。

        Phase 1 只保证导出最小工件，不在这里做重度归一化。
        具体算法可在各自 wrapper 中覆写该方法，补更高保真度的信息。
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
        """校验 runner 显式注入的数据契约与真实输入是否一致。

        说明：
        - `n_features` 若给定，必须与当前 `X.shape[1]` 一致；
        - `feature_names` 若给定，长度必须与特征维度一致，且每项都应为非空字符串；
        - `target_name` 若给定，必须是非空字符串。
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            inferred_n_features = 1
        elif X_arr.ndim == 2:
            inferred_n_features = int(X_arr.shape[1])
        else:
            raise ValueError(
                f"{context or self.__class__.__name__}: 输入 X 必须是一维或二维数组，当前 ndim={X_arr.ndim}"
            )

        if n_features is not None:
            try:
                expected_n_features = int(n_features)
            except Exception as err:
                raise TypeError(
                    f"{context or self.__class__.__name__}: n_features 类型非法，无法转成整数: {err}"
                ) from err
            if expected_n_features != inferred_n_features:
                raise ValueError(
                    f"{context or self.__class__.__name__}: 显式维度契约不一致，"
                    f"n_features={expected_n_features}, 实际输入维度={inferred_n_features}"
                )

        if feature_names is not None:
            if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
                raise TypeError(
                    f"{context or self.__class__.__name__}: feature_names 必须是字符串序列"
                )
            feature_names_list = list(feature_names)
            if len(feature_names_list) != inferred_n_features:
                raise ValueError(
                    f"{context or self.__class__.__name__}: feature_names 长度与输入维度不一致，"
                    f"len(feature_names)={len(feature_names_list)}, 实际输入维度={inferred_n_features}"
                )
            bad_names = [name for name in feature_names_list if not isinstance(name, str) or not name.strip()]
            if bad_names:
                raise ValueError(
                    f"{context or self.__class__.__name__}: feature_names 中存在空值或非字符串项: {bad_names!r}"
                )

        if target_name is not None and (not isinstance(target_name, str) or not target_name.strip()):
            raise ValueError(
                f"{context or self.__class__.__name__}: target_name 必须是非空字符串"
            )

        return inferred_n_features

    def serialize(self):
        """将整个类实例序列化为字符串"""
        # 使用pickle序列化整个类实例
        instance_bytes = pickle.dumps(self)
        # 将字节转换为base64编码的字符串
        instance_b64 = base64.b64encode(instance_bytes).decode('utf-8')
        return instance_b64

    @classmethod
    def deserialize(cls, instance_b64):
        """从序列化的字符串反序列化为类实例"""
        # 将base64编码的字符串转换回字节
        instance_bytes = base64.b64decode(instance_b64)
        # 使用pickle从字节重建类实例
        instance = pickle.loads(instance_bytes)
        return instance
        
    def __str__(self):
        """返回模型的字符串表示"""
        return f"{self.__class__.__name__}()"
