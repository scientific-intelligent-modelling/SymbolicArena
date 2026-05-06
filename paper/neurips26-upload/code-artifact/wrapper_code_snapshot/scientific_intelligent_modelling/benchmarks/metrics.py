"""统一的 benchmark 指标实现。

当前优先覆盖三类 benchmark 的核心指标：
1. SRBench: R2、模型复杂度、symbolic solution。
2. SRSD: R2>0.999 风格准确率、solution rate、NED。
3. LLM-SRBench: NMSE、Acc_tau / Acc0.1、symbolic accuracy（标签聚合）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

try:
    import sympy as sp
except ModuleNotFoundError:  # pragma: no cover - 运行环境可能未安装 sympy
    sp = None


def safe_float(value: Any) -> float | None:
    """将结果归一化为普通 float；NaN/Inf 统一返回 None。"""
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def acc_within_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float | None:
    """LLM-SRBench 风格的 Acc_tau。

    按绝对误差是否小于阈值计数。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return None
    return safe_float(np.mean(np.abs(y_true - y_pred) <= float(threshold)))


def llm_srbench_acc_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
    *,
    eps: float = 1e-12,
) -> float | None:
    """LLM-SRBench 原生的 Acc_tau。

    论文定义采用：
        1( max_i |(y_hat_i - y_i) / y_i| <= tau )
    即对整条测试轨迹给出 0/1 判定，而不是逐点取平均。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return None
    denom = np.maximum(np.abs(y_true), float(eps))
    max_rel_err = float(np.max(np.abs(y_pred - y_true) / denom))
    return safe_float(1.0 if max_rel_err <= float(tau) else 0.0)


def llm_srbench_nmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """LLM-SRBench 原生 NMSE。

    根据论文附录 B.1：
        sum_i (y_hat_i - y_i)^2 / sum_i (y_i - y_bar)^2
    这与当前工具集里常用的 `mse / mean(y^2)` 不是同一口径。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return None
    numerator = float(np.sum(np.square(y_pred - y_true)))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    if denominator <= 0 or math.isnan(denominator):
        return None
    return safe_float(numerator / denominator)


def llm_srbench_numeric_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tau: float = 0.1,
) -> dict[str, float | None]:
    """LLM-SRBench 数值指标组合。"""
    return {
        "nmse": llm_srbench_nmse(y_true, y_pred),
        "acc_tau": llm_srbench_acc_tau(y_true, y_pred, tau=tau),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    acc_threshold: float | None = None,
) -> dict[str, float | None]:
    """统一回归指标。

    返回当前工具集最常用的一组值，后续 benchmark profile 只需按需取子集：
    `mse`, `rmse`, `mae`, `r2`, `nmse`, `acc_tau`。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return {
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "nmse": None,
            "acc_tau": None,
        }

    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_mean)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = float(np.mean(np.square(y_true))) if y_true.size else float("nan")
    nmse = float(mse / denom) if denom and not math.isnan(denom) else float("nan")
    acc_tau = (
        acc_within_threshold(y_true, y_pred, acc_threshold)
        if acc_threshold is not None
        else None
    )
    return {
        "mse": safe_float(mse),
        "rmse": safe_float(rmse),
        "mae": safe_float(mae),
        "r2": safe_float(r2),
        "nmse": safe_float(nmse),
        "acc_tau": safe_float(acc_tau),
    }


def _sympify_expression(expr: Any) -> sp.Expr:
    if sp is None:
        raise ModuleNotFoundError("sympy 未安装，无法计算符号类 benchmark 指标")
    if isinstance(expr, sp.Expr):
        return expr
    if isinstance(expr, (int, float)):
        return sp.Float(expr)
    if isinstance(expr, str):
        return sp.sympify(expr)
    raise TypeError(f"不支持的表达式类型: {type(expr)!r}")


def _is_constant_expr(expr: sp.Expr) -> bool:
    return len(expr.free_symbols) == 0


def srbench_symbolic_solution(
    predicted_expr: Any,
    true_expr: Any,
) -> dict[str, Any]:
    """SRBench 2021 的 symbolic solution 判定。

    论文定义：
    - 预测式不能退化成常数；
    - 满足 `phi* - phi_hat = a` 或 `phi* / phi_hat = b, b != 0`，
      其中 a / b 为常数。
    """
    pred = sp.simplify(_sympify_expression(predicted_expr))
    true = sp.simplify(_sympify_expression(true_expr))

    if _is_constant_expr(pred):
        return {
            "is_symbolic_solution": False,
            "relation": "constant_prediction",
        }

    diff = sp.simplify(true - pred)
    if _is_constant_expr(diff):
        return {
            "is_symbolic_solution": True,
            "relation": "additive_constant",
            "constant": str(diff),
        }

    try:
        ratio = sp.simplify(true / pred)
    except Exception:
        ratio = None
    if ratio is not None and _is_constant_expr(ratio) and ratio != 0:
        return {
            "is_symbolic_solution": True,
            "relation": "scalar_multiple",
            "constant": str(ratio),
        }

    return {
        "is_symbolic_solution": False,
        "relation": "mismatch",
    }


def _count_complexity(expr: sp.Expr) -> tuple[int, int, int]:
    """返回 (operators, features, constants)。"""
    if expr.is_Symbol:
        return 0, 1, 0
    if expr.is_Number:
        return 0, 0, 1

    operators = 1
    features = 0
    constants = 0
    for arg in expr.args:
        sub_ops, sub_features, sub_constants = _count_complexity(arg)
        operators += sub_ops
        features += sub_features
        constants += sub_constants
    return operators, features, constants


def srbench_model_size(expr: Any, *, simplify: bool = False) -> dict[str, Any]:
    """SRBench 风格复杂度。

    论文将复杂度定义为：
    数学运算符 + 特征 + 常数 的总数。
    """
    parsed = _sympify_expression(expr)
    if simplify:
        parsed = sp.simplify(parsed)
    operators, features, constants = _count_complexity(parsed)
    return {
        "operators": operators,
        "features": features,
        "constants": constants,
        "size": operators + features + constants,
        "expression": str(parsed),
    }


@dataclass(frozen=True)
class _TreeNode:
    label: str
    children: tuple["_TreeNode", ...]


def _expr_to_tree(expr: sp.Expr) -> _TreeNode:
    expr = sp.simplify(expr)
    if expr.is_Symbol:
        return _TreeNode(f"Symbol:{expr}", ())
    if expr.is_Number:
        # SRSD 明确指出系数数值本身不应成为重点，统一折叠成常数节点。
        return _TreeNode("Const", ())
    return _TreeNode(
        type(expr).__name__,
        tuple(_expr_to_tree(arg) for arg in expr.args),
    )


@lru_cache(maxsize=None)
def _tree_size(node: _TreeNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(_tree_size(child) for child in node.children)


@lru_cache(maxsize=None)
def _tree_edit_distance(a: _TreeNode | None, b: _TreeNode | None) -> int:
    if a is None:
        return _tree_size(b)
    if b is None:
        return _tree_size(a)

    label_cost = 0 if a.label == b.label else 1
    a_children = a.children
    b_children = b.children
    dp = [[0] * (len(b_children) + 1) for _ in range(len(a_children) + 1)]

    for i in range(1, len(a_children) + 1):
        dp[i][0] = dp[i - 1][0] + _tree_size(a_children[i - 1])
    for j in range(1, len(b_children) + 1):
        dp[0][j] = dp[0][j - 1] + _tree_size(b_children[j - 1])

    for i in range(1, len(a_children) + 1):
        for j in range(1, len(b_children) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + _tree_size(a_children[i - 1]),
                dp[i][j - 1] + _tree_size(b_children[j - 1]),
                dp[i - 1][j - 1] + _tree_edit_distance(a_children[i - 1], b_children[j - 1]),
            )

    return label_cost + dp[-1][-1]


def normalized_tree_edit_distance(predicted_expr: Any, true_expr: Any) -> dict[str, Any]:
    """SRSD 风格的 normalized edit distance (NED)。

    依据论文的式(3)：
        min(1, d(f_pred, f_true) / |f_true|)
    其中 d 为 Zhang-Shasha tree edit distance，|f_true| 为真值树节点数。

    这里实现的是同等语义的有序树编辑距离版本，常数节点统一折叠为 `Const`。
    """
    pred_tree = _expr_to_tree(_sympify_expression(predicted_expr))
    true_tree = _expr_to_tree(_sympify_expression(true_expr))
    distance = _tree_edit_distance(pred_tree, true_tree)
    true_size = _tree_size(true_tree)
    normalized = min(1.0, float(distance) / float(true_size)) if true_size > 0 else 1.0
    return {
        "tree_edit_distance": distance,
        "true_tree_size": true_size,
        "ned": normalized,
    }


def aggregate_symbolic_accuracy(labels: Iterable[bool | int | float]) -> float | None:
    """聚合 LLM-SRBench 风格的 symbolic accuracy。"""
    labels = list(labels)
    if not labels:
        return None
    values = [1.0 if bool(v) else 0.0 for v in labels]
    return safe_float(np.mean(values))
