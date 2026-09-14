"""
iMCTS 包装器

将外部仓库 MCTS-4-SR 集成到本框架，提供统一的 BaseWrapper 接口：
- fit(X, y): 训练并发现最佳表达式
- predict(X): 使用找到的向量表达式进行预测
- get_optimal_equation(): 返回与 native predict 一致的向量表达式（字符串）
- get_total_equations(): 返回候选表达式列表（此处仅返回最优表达式）

实现要点：
- iMCTS.Regrssor 期望输入形状为 (n_features, n_samples)，本框架使用 (n_samples, n_features)，需转置
- Regressor.fit() 返回 (simplified_expr, vec_expr, eval_count, path)
- 预测通过 eval('lambda x: {vec_expr}') 并在 numpy 上下文下调用 f(x)
- 为了在子进程序列化/反序列化后仍可预测，本包装器不依赖底层类状态进行预测，而是持久化 vec_expr 并在需要时重建可调用函数
"""

import os
import sys
import json
import math
import time
import hashlib
from typing import Any, Dict, Optional, List

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_imcts_artifact


def _default_eval_context(user_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构建用于 eval 的安全上下文（仅暴露必要数学函数）。
    iMCTS 默认上下文参考其 regressor 实现：sin/cos/exp/log/tanh 等。
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
    """iMCTS 的统一包装器。"""
    _PROGRESS_STATE_FILENAME = ".imcts_current_best.json"

    def __init__(self, **kwargs):
        # 存储用户传入参数，部分会透传给 iMCTS.Regressor
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
        self._timeout_in_seconds = self._as_positive_float(self.params.pop("timeout_in_seconds", None))
        self._timeout_guard_seconds = self._as_positive_float(self.params.pop("timeout_guard_seconds", None)) or 5.0
        self._exp_path = self.params.get('exp_path')
        self._exp_name = self.params.get('exp_name')
        self._contract_n_features = self.params.pop("n_features", None)
        self._contract_feature_names = self.params.pop("feature_names", None)
        self._contract_target_name = self.params.pop("target_name", None)

        # 训练所得的表达式
        self._best_expr_simplified: Optional[str] = None
        self._best_expr_vector: Optional[str] = None
        self._eval_count: Optional[int] = None
        self._best_path: Optional[int] = None
        self._n_features: Optional[int] = None

        # 预测时的上下文（可由用户覆盖）
        self._eval_context: Dict[str, Any] = _default_eval_context(self.params.get('context'))

        # 运行时（fit 阶段）引用的底层回归器（仅在同一进程内可用）
        self._runtime_regressor = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)
        self._fit_started_at: Optional[float] = None

    @staticmethod
    def _as_positive_float(value) -> Optional[float]:
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

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
        observed_at = time.time()
        existing = None
        if os.path.isfile(self._progress_state_path):
            try:
                with open(self._progress_state_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except Exception:
                existing = None

        existing_score = existing.get("score") if isinstance(existing, dict) else None
        existing_score = (
            float(existing_score)
            if isinstance(existing_score, (int, float))
            and not isinstance(existing_score, bool)
            and np.isfinite(float(existing_score))
            else None
        )
        current_score = (
            float(score)
            if isinstance(score, (int, float))
            and not isinstance(score, bool)
            and np.isfinite(float(score))
            else None
        )
        # 原生 score 是 reward（越大越好）。同一 best、较差 chunk 及 fit 收尾时
        # 的无排名结果都不能覆盖首次发现证据。
        if current_score is None:
            return
        if existing_score is not None and current_score <= existing_score:
            return

        started_at = self._fit_started_at
        elapsed_seconds = (
            max(0.0, observed_at - started_at)
            if isinstance(started_at, (int, float)) and np.isfinite(float(started_at))
            else 0.0
        )
        payload = {
            "equation": equation,
            "expression_vector": equation,
            "score": current_score,
            "reward": current_score,
            "internal_objective": "native_reward",
            "objective_direction": "max",
            "evaluations": int(evaluations) if isinstance(evaluations, (int, float)) else None,
            "first_discovered_elapsed_seconds": round(elapsed_seconds, 6),
            "first_discovered_minute": max(1, int(math.ceil(elapsed_seconds / 60.0))),
            "source": "imcts_native_reward",
            "source_timestamp_unix": float(observed_at),
            "candidate_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "expression_vector": equation,
                        "reward": current_score,
                        "evaluations": int(evaluations)
                        if isinstance(evaluations, (int, float))
                        else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            temporary_path = f"{self._progress_state_path}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self._progress_state_path)
        except Exception:
            pass

    # ============ 标准 API ============
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
            raise ValueError("iMCTS 训练需要二维输入数组 (n_samples, n_features)")
        self._n_features = int(X.shape[1])

        # 导入第三方代码：将子仓库加入 sys.path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = os.path.join(base_dir, 'MCTS-4-SR')
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

        # 延迟导入 iMCTS
        from iMCTS.regressor import Regressor as _MCTSRegressor

        # iMCTS 期望输入形状为 (n_features, n_samples)
        x_train = X.T
        y_train = y

        # 过滤仅 iMCTS 支持的关键字参数
        allowed_keys = {
            'ops', 'arity_dict', 'context', 'max_depth', 'K', 'c', 'gamma',
            'gp_rate', 'mutation_rate', 'exploration_rate', 'max_single_arity_ops',
            'max_constants', 'max_expressions', 'verbose', 'reward_func',
            'optimization_method', 'time_limit', 'disable_success_early_stop'
        }
        mcts_kwargs = {k: v for k, v in self.params.items() if k in allowed_keys}

        started_at = time.time()
        self._fit_started_at = started_at
        base_seed = self.params.get('seed')
        max_chunks = int(self.params.get("budget_chunks", 1) or 1)
        if self._timeout_in_seconds is not None and "budget_chunks" not in self.params:
            max_chunks = 1000000
        if self._timeout_in_seconds is not None:
            mcts_kwargs["disable_success_early_stop"] = True

        best_tuple = None
        best_score = None
        last_reg = None
        natural_completions = 0
        for chunk_index in range(max(1, max_chunks)):
            if self._timeout_in_seconds is not None:
                elapsed = time.time() - started_at
                remaining_budget = max(0.0, self._timeout_in_seconds - self._timeout_guard_seconds - elapsed)
                if remaining_budget <= 0.0:
                    break
                mcts_kwargs["time_limit"] = remaining_budget
            reg = _MCTSRegressor(
                x_train=x_train,
                y_train=y_train,
                progress_callback=self._write_progress_state if self._progress_state_path else None,
                **mcts_kwargs,
            )
            last_reg = reg
            try:
                seed = int(base_seed) + chunk_index if base_seed is not None else None
            except Exception:
                seed = base_seed
            simplified_expr, vec_expr, eval_count, path = reg.fit(seed=seed)
            score = getattr(reg, "last_best_reward", None)
            try:
                score = float(score)
            except (TypeError, ValueError, OverflowError):
                score = None
            if score is not None and not np.isfinite(score):
                score = None
            if best_tuple is None or (score is not None and (best_score is None or score > best_score)):
                best_tuple = (simplified_expr, vec_expr, eval_count, path, reg)
                best_score = score
            natural_completions += 1
            if self._timeout_in_seconds is None:
                break

        if best_tuple is None:
            raise ValueError("iMCTS 未产生可用表达式")
        simplified_expr, vec_expr, eval_count, path, best_reg = best_tuple
        self._runtime_regressor = best_reg or last_reg
        self._budget_chunks_run = natural_completions
        self._natural_completions = natural_completions
        self._budget_loop_exhausted = (
            self._timeout_in_seconds is not None
            and (time.time() - started_at) >= max(0.0, self._timeout_in_seconds - self._timeout_guard_seconds)
        )

        # 缓存结果
        self._best_expr_simplified = simplified_expr
        self._best_expr_vector = vec_expr
        self._eval_count = int(eval_count) if eval_count is not None else None
        if path is not None and not isinstance(path, (list, tuple)):
            self._best_path = int(path)
        else:
            # iMCTS 运行时返回 path 可能为路径列表，保留原始结构用于调试
            self._best_path = path

        self._write_progress_state(
            equation=self._best_expr_vector or self._best_expr_simplified or "",
            score=best_score,
            evaluations=self._eval_count,
        )

        return self

    def predict(self, X):
        if not isinstance(self._best_expr_vector, str) or not self._best_expr_vector:
            raise ValueError("模型尚未训练或未找到可用的表达式")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("iMCTS 预测需要二维输入数组 (n_samples, n_features)")

        # iMCTS 预测期望 (n_features, n_samples)
        XT = X.T

        # 若同进程存在底层回归器，直接复用其 predict（包含更多上下文）
        if self._runtime_regressor is not None:
            return self._runtime_regressor.predict(XT, self._best_expr_vector)

        # 否则根据持久化的表达式与上下文重建可调用函数
        try:
            func = eval(f'lambda x: {self._best_expr_vector}', self._eval_context)
            y_pred = func(XT)
            return np.asarray(y_pred)
        except Exception as e:
            raise RuntimeError(f"iMCTS 预测失败: {e}")

    @staticmethod
    def _score_vector_expression(reg, vec_expr, x_train, y_train) -> Optional[float]:
        if not isinstance(vec_expr, str) or not vec_expr.strip():
            return None
        try:
            y_pred = reg.predict(x_train, vec_expr)
        except Exception:
            return None
        try:
            pred = np.asarray(y_pred, dtype=float).reshape(-1)
            target = np.asarray(y_train, dtype=float).reshape(-1)
        except Exception:
            return None
        if pred.shape[0] != target.shape[0]:
            return None
        mask = np.isfinite(pred) & np.isfinite(target)
        if not np.any(mask):
            return None
        try:
            return float(np.mean((pred[mask] - target[mask]) ** 2))
        except Exception:
            return None

    def get_optimal_equation(self):
        # native predict 始终执行 vector expression。上游 simplified expression
        # 经过 expand_log(force=True)，在未知实数域可能改变语义，不能再把它
        # 当成正式导出公式。
        return self._best_expr_vector or self._best_expr_simplified or ""

    def get_total_equations(self):
        # 当前仅返回一个最优表达式
        equation = self.get_optimal_equation()
        return [equation] if equation else []

    # ============ 序列化 / 反序列化 ============
    def serialize(self):
        state = {
            'params': self.params,
            'expr_simplified': self._best_expr_simplified,
            'expr_vector': self._best_expr_vector,
            'eval_count': self._eval_count,
            'best_path': self._best_path,
            'n_features': self._n_features,
            # 仅持久化上下文的键名，值用默认可重建（避免不可序列化对象）
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
        # 运行时回归器不可恢复；预测走表达式+上下文路径
        inst._runtime_regressor = None
        return inst

    def export_canonical_symbolic_program(self):
        equation = self._best_expr_vector or self._best_expr_simplified
        if not equation:
            raise ValueError("iMCTS 当前没有可导出的最优方程")
        return normalize_imcts_artifact(
            equation,
            expected_n_features=self._n_features,
        )

    def __str__(self) -> str:
        lines: List[str] = ["iMCTSRegressor(tool='iMCTS')"]
        if self._best_expr_simplified:
            lines.append(f"最佳表达式: {self._best_expr_simplified}")
        if self._eval_count is not None:
            lines.append(f"评估表达式数: {self._eval_count}")
        return "\n".join(lines)
