"""QLattice 回归器包装器（原生实现）

说明：
- 适配 srkit/subprocess_runner 的统一调用：fit/predict/get_optimal_equation/get_total_equations
- 训练期依赖在线 feyn.QLattice()（社区版）
- 序列化采用稳健 JSON：保存最优表达式与候选方程字符串，反序列化后可离线预测
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
    """QLattice 的回归器封装（回归任务）。

    参数（常用）：
    - n_epochs: int = 100，自动搜索轮数
    - kind: str = 'regression'，任务类型
    - signif: int = 4，表达式输出的有效数字（用于 sympify 展示）
    - 其他 QLattice.auto_run 支持的参数可透传
    """
    _PROGRESS_STATE_FILENAME = ".qlattice_current_best.json"

    def __init__(self, **kwargs) -> None:
        self.params = dict(kwargs)
        self.params.setdefault("n_epochs", 100)
        self.params.setdefault("kind", "regression")
        self.params.setdefault("criterion", "bic")
        self.params.setdefault("signif", 4)
        self.params.setdefault("threads", 4)
        self._timeout_in_seconds = self._as_positive_float(self.params.pop("timeout_in_seconds", None))
        self._timeout_guard_seconds = self._as_positive_float(self.params.pop("timeout_guard_seconds", None)) or 5.0
        self._target_standardize = self.params.pop("target_standardize", "auto")
        self._target_standardize_threshold = (
            self._as_positive_float(self.params.pop("target_standardize_threshold", None)) or 1.0e6
        )
        self._constant_fallback_on_empty = self._as_bool(self.params.pop("constant_fallback_on_empty", True))
        self._contract_n_features = self.params.pop("n_features", None)
        self._contract_feature_names = self.params.pop("feature_names", None)
        self._contract_target_name = self.params.pop("target_name", None)
        self.model = None
        self._exp_path = self.params.get("exp_path")
        self._exp_name = self.params.get("exp_name")

        # QLattice 相关缓存
        self._ql = None
        self._models = []
        self._best_model = None

        # 表达式/预测相关缓存
        self._expr_str: Optional[str] = None  # 最优表达式字符串
        self._input_vars: List[str] = []
        self._output_name: str = 'y'
        self._lambdified = None
        self._target_offset = 0.0
        self._target_scale = 1.0
        self._target_was_standardized = False
        self._constant_prediction: Optional[float] = None
        self._fallback_reason: Optional[str] = None
        # 候选方程字符串列表（便于序列化后仍可获取多个解）
        self._equations: List[str] = []
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

    @staticmethod
    def _as_positive_float(value) -> Optional[float]:
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

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
    def _raw_model_equation(model, signif: int) -> Optional[str]:
        try:
            return str(model.sympify(signif=signif))
        except Exception:
            try:
                return str(model)
            except Exception:
                return None

    def _model_equation(self, model, signif: int) -> Optional[str]:
        raw_equation = self._raw_model_equation(model, signif)
        return self._equation_on_original_target_scale(raw_equation)

    @staticmethod
    def _format_float(value: float) -> str:
        return format(float(value), ".17g")

    def _equation_on_original_target_scale(self, equation: Optional[str]) -> Optional[str]:
        if not isinstance(equation, str) or not equation.strip():
            return None
        if not self._target_was_standardized:
            return equation
        offset = self._format_float(self._target_offset)
        scale = self._format_float(self._target_scale)
        return f"({offset}) + ({scale})*({equation})"

    def _should_standardize_target(self, y: np.ndarray) -> bool:
        mode = str(self._target_standardize).strip().lower()
        if mode in {"0", "false", "no", "off", "never"}:
            return False
        finite_y = y[np.isfinite(y)]
        if finite_y.size == 0:
            return False
        scale = float(np.std(finite_y))
        if not np.isfinite(scale) or scale <= 0:
            return False
        if mode in {"1", "true", "yes", "on", "always"}:
            return True
        max_abs = float(np.max(np.abs(finite_y)))
        return max(scale, max_abs) >= self._target_standardize_threshold

    def _prepare_target_for_fit(self, y: np.ndarray) -> np.ndarray:
        self._target_offset = 0.0
        self._target_scale = 1.0
        self._target_was_standardized = False
        if not self._should_standardize_target(y):
            return y
        finite_y = y[np.isfinite(y)]
        offset = float(np.mean(finite_y))
        scale = float(np.std(finite_y))
        if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
            return y
        self._target_offset = offset
        self._target_scale = scale
        self._target_was_standardized = True
        return (y - offset) / scale

    def _restore_target_scale(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self._target_was_standardized:
            arr = self._target_offset + self._target_scale * arr
        return arr

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
                    value = float(value)
                    if np.isfinite(value):
                        return value
                except Exception:
                    pass
        for fallback in ("bic", "aic"):
            try:
                value = getattr(model, fallback, None)
            except Exception:
                value = None
            if value is not None:
                try:
                    value = float(value)
                    if np.isfinite(value):
                        return value
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

    def _write_progress_state_from_equation(
        self,
        equation: Optional[str],
        *,
        epoch: int,
        loss: Optional[float],
        criterion_name: Optional[str],
    ) -> None:
        if not self._progress_state_path:
            return
        if not isinstance(equation, str) or not equation.strip():
            return
        payload = {
            "equation": equation,
            "loss": loss,
            "internal_loss": loss,
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

    # ---------------- 公共接口 ----------------
    def fit(self, X, y):
        """训练 QLattice 模型并缓存最优与候选表达式。"""
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
        y_for_fit = self._prepare_target_for_fit(y)

        # 构造 DataFrame
        n_features = X.shape[1]
        self._input_vars = [f"x{i}" for i in range(n_features)]
        self._output_name = self.params.get('output_name', self._contract_target_name or 'y')
        df = pd.DataFrame(X, columns=self._input_vars)
        df[self._output_name] = y_for_fit

        # 连接 QLattice（社区版需要联网）
        self._ql = feyn.QLattice()

        # 组装 auto_run 参数
        auto_args = {
            'data': df,
            'output_name': self._output_name,
            'kind': self.params.get('kind', 'regression'),
            'n_epochs': int(self.params.get('n_epochs', 100)),
        }
        # 尽量覆盖 QLattice.auto_run 的可选参数（按需透传）
        for k in [
            'stypes', 'threads', 'max_complexity', 'query_string',
            'loss_function', 'criterion', 'sample_weights',
            'function_names', 'starting_models'
        ]:
            if k in self.params:
                auto_args[k] = self.params[k]

        total_epochs = int(auto_args.get('n_epochs', 100))
        if total_epochs < 1:
            raise ValueError('n_epochs 必须 >= 1')
        signif = int(self.params.get('signif', 4))
        criterion_name = self._criterion_name()

        if self._progress_state_path:
            running_models = auto_args.get('starting_models')
            models = []
            global_best_model = None
            global_best_loss = None
            global_best_epoch = 0
            epoch_args = dict(auto_args)
            epoch_args['n_epochs'] = 1
            started_at = time.time()
            for epoch in range(1, total_epochs + 1):
                if self._timeout_in_seconds is not None:
                    elapsed = time.time() - started_at
                    if elapsed >= max(0.0, self._timeout_in_seconds - self._timeout_guard_seconds):
                        break
                if running_models:
                    epoch_args['starting_models'] = running_models
                else:
                    epoch_args.pop('starting_models', None)
                epoch_models = list(self._ql.auto_run(**epoch_args))
                # feyn 在单 epoch 增量调用时，部分数据集可能偶发返回空列表；
                # 这不等价于整个搜索失败，继续给后续 epoch 机会即可。
                if not epoch_models:
                    continue
                models = epoch_models
                running_models = epoch_models
                epoch_best_model = self._select_best_model(epoch_models, criterion_name)
                epoch_best_loss = self._criterion_value(epoch_best_model, criterion_name)
                if global_best_model is None or (
                    epoch_best_loss is not None
                    and (global_best_loss is None or epoch_best_loss < global_best_loss)
                ):
                    global_best_model = epoch_best_model
                    global_best_loss = epoch_best_loss
                    global_best_epoch = epoch
                self._write_progress_state_from_equation(
                    self._model_equation(global_best_model, signif),
                    epoch=global_best_epoch,
                    loss=global_best_loss,
                    criterion_name=criterion_name,
                )
            if not models:
                return self._fit_constant_fallback(y, reason='QLattice.auto_run 未返回任何模型。')
        else:
            models = list(self._ql.auto_run(**auto_args))
            if not models:
                return self._fit_constant_fallback(y, reason='QLattice.auto_run 未返回任何模型。')
            global_best_model = self._select_best_model(models, criterion_name)

        self._best_model = global_best_model
        self._models = [self._best_model]
        self._models.extend(model for model in models if model is not self._best_model)
        self.model = True

        # 提取最优与候选表达式
        self._expr_str = self._model_equation(self._best_model, signif)

        equations: List[str] = []
        for m in self._models:
            eq = self._model_equation(m, signif)
            if eq is not None:
                equations.append(eq)
        self._equations = equations

        # 构建 lambdify 预测器（便于序列化回放）
        self._build_lambdify()
        return self

    def _fit_constant_fallback(self, y: np.ndarray, *, reason: str):
        if not self._constant_fallback_on_empty:
            raise NoValidOutputError(reason)
        finite_y = y[np.isfinite(y)]
        if finite_y.size == 0:
            raise NoValidOutputError(reason)
        value = float(np.mean(finite_y))
        if not np.isfinite(value):
            raise NoValidOutputError(reason)
        self._models = []
        self._best_model = None
        self.model = True
        self._constant_prediction = value
        self._fallback_reason = reason
        self._expr_str = self._format_float(value)
        self._equations = [self._expr_str]
        self._build_lambdify()
        self._write_progress_state_from_equation(
            self._expr_str,
            epoch=0,
            loss=None,
            criterion_name=self._criterion_name(),
        )
        return self

    def predict(self, X):
        """使用模型进行预测。"""
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self._input_vars):
            raise ValueError(f'特征维度不匹配：期望 {len(self._input_vars)} 列，实际 {X.shape[1]} 列。')

        if self._constant_prediction is not None:
            return np.full(X.shape[0], float(self._constant_prediction), dtype=float)

        # 优先使用原生 feyn 模型（更稳健），其次使用 lambdify
        if self._best_model is not None:
            import pandas as pd
            df = pd.DataFrame(X, columns=self._input_vars)
            return self._restore_target_scale(self._best_model.predict(data=df))

        if self._lambdified is not None:
            cols = [X[:, i] for i in range(X.shape[1])]
            y_pred = self._lambdified(*cols)
            y_pred = np.asarray(y_pred, dtype=float)
            if y_pred.ndim == 0:
                y_pred = np.full(X.shape[0], float(y_pred), dtype=float)
            return y_pred

        raise RuntimeError('未找到可用的预测器（表达式或 QLattice 模型）。')

    def get_optimal_equation(self):
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')
        if self._expr_str:
            return self._expr_str
        try:
            signif = int(self.params.get('signif', 4))
            return str(self._best_model.sympify(signif=signif))
        except Exception:
            return '未找到可用的方程'

    def get_total_equations(self, n: int | None = None):
        """返回候选模型的表达式列表（字符串）。

        参数:
            n: 返回的方程数量上限；若为 None 或无效，则返回全部。
        """
        if self.model is None:
            raise ValueError('模型尚未训练，请先调用 fit 方法。')
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
            raise ValueError('模型尚未训练，请先调用 fit 方法。')
        return normalize_qlattice_artifact(
            self.get_optimal_equation(),
            expected_n_features=len(self._input_vars or []),
        )

    # ---------------- 序列化/反序列化 ----------------
    def serialize(self):
        state = {
            'params': self.params,
            'expr': self._expr_str,
            'input_vars': self._input_vars,
            'output_name': self._output_name,
            'equations': self._equations,
            'constant_prediction': self._constant_prediction,
            'fallback_reason': self._fallback_reason,
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
        constant_prediction = obj.get('constant_prediction')
        try:
            inst._constant_prediction = float(constant_prediction) if constant_prediction is not None else None
        except Exception:
            inst._constant_prediction = None
        inst._fallback_reason = obj.get('fallback_reason')
        inst._build_lambdify()
        return inst

    # ---------------- 内部工具 ----------------
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
