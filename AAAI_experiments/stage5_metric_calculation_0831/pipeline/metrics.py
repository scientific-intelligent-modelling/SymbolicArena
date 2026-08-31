"""新六轴评测的纯确定性数学核心。

本模块不读取实验文件、不调用大模型，也不复用旧六轴聚合逻辑。调用方必须先完成
来源冻结，并明确区分算法无有效输出与评测产物缺失。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


class MetricContractError(ValueError):
    """输入不满足修订指标契约。"""


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _bounded(value: float, *, name: str) -> float:
    number = _finite_float(value)
    if number is None or number < 0.0 or number > 1.0:
        raise MetricContractError(f"{name} 必须是 [0, 1] 内的有限数值，实际为 {value!r}")
    return number


def phi_nmse(value: object) -> float:
    """将单次运行的 NMSE 映射到固定的 [0, 1] 数值质量。

    无效、负数、NaN 和 Inf 均按指标文档映射为 0。零 NMSE 通过 ``1e-12``
    下界得到满分。
    """

    nmse = _finite_float(value)
    if nmse is None or nmse < 0.0:
        return 0.0
    log_nmse = math.log10(max(nmse, 1.0e-12))
    clipped = min(2.0, max(-12.0, log_nmse))
    return 1.0 - (clipped + 12.0) / 14.0


def aggregate_quality(nmse_values: Iterable[object]) -> float:
    """逐 run 映射后取经验平均，并缩放到 0--100。"""

    qualities = [phi_nmse(value) for value in nmse_values]
    if not qualities:
        return 0.0
    return 100.0 * sum(qualities) / len(qualities)


def set_f1(predicted: set[str], expected: set[str]) -> float:
    """计算变量集或算子集 F1；两个空集合视为完全恢复。"""

    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def symbolic_fidelity_score(
    *,
    equivalent: bool,
    tree_similarity: float,
    variable_f1: float,
    operator_f1: float,
    valid: bool = True,
) -> float:
    """计算单个 clean task-seed 的 SYM 分量。"""

    if not valid:
        return 0.0
    tree = _bounded(tree_similarity, name="tree_similarity")
    variables = _bounded(variable_f1, name="variable_f1")
    operators = _bounded(operator_f1, name="operator_f1")
    if equivalent:
        return 1.0
    return 0.5 * (tree * variables * operators) ** (1.0 / 3.0)


def minimality_score(
    reference_complexity: int,
    predicted_complexity: int,
    *,
    valid: bool = True,
) -> float:
    """计算单个 clean task-seed 的 MIN 分量。"""

    if not valid:
        return 0.0
    if isinstance(reference_complexity, bool) or reference_complexity <= 0:
        raise MetricContractError("reference_complexity 必须是正整数")
    if isinstance(predicted_complexity, bool) or predicted_complexity <= 0:
        raise MetricContractError("predicted_complexity 必须是正整数")
    return min(1.0, float(reference_complexity) / float(predicted_complexity))


def efficiency_from_qualities(
    qualities: Sequence[float | None],
    *,
    horizon: int = 180,
) -> float:
    """计算单次运行的 EFF 分量。

    ``qualities`` 必须已经是完整、经来源审计的固定网格。``None`` 或长度不足表示
    遥测缺失，属于评测基础设施错误，而不是算法的零质量输出。
    """

    if horizon <= 0:
        raise MetricContractError("horizon 必须为正整数")
    if len(qualities) != horizon:
        raise MetricContractError(f"EFF 需要 {horizon} 个检查点，实际为 {len(qualities)}")
    normalized: list[float] = []
    for index, value in enumerate(qualities, start=1):
        number = _finite_float(value)
        if number is None:
            raise MetricContractError(f"EFF 第 {index} 个检查点缺失或不是有限数值")
        normalized.append(_bounded(number, name=f"quality[{index}]"))
    best = max(normalized, default=0.0)
    if best <= 0.0:
        return 0.0
    return sum(value / best for value in normalized) / horizon


@dataclass(frozen=True)
class RunQuality:
    """一个 seed 的最终数值质量与输出有效性。"""

    id_quality: float
    ood_quality: float
    valid: bool

    def validated(self) -> "RunQuality":
        return RunQuality(
            id_quality=_bounded(self.id_quality, name="id_quality"),
            ood_quality=_bounded(self.ood_quality, name="ood_quality"),
            valid=bool(self.valid),
        )


@dataclass(frozen=True)
class StabilityResult:
    numerical_consistency: float
    validity: float
    structural_consistency: float
    score: float


def numerical_consistency(runs: Sequence[RunQuality]) -> float:
    """按三个 seed pair 计算一个算法-任务的 Numerical Consistency。"""

    if len(runs) != 3:
        raise MetricContractError(f"STAB 需要恰好 3 个 seed，实际为 {len(runs)}")
    checked = [run.validated() for run in runs]
    disagreements: list[float] = []
    for left, right in combinations(checked, 2):
        disagreements.append(
            (abs(left.id_quality - right.id_quality) + abs(left.ood_quality - right.ood_quality))
            / 2.0
        )
    return 1.0 - sum(disagreements) / len(disagreements)


def stability_score(
    runs: Sequence[RunQuality],
    *,
    structural_pair_results: Sequence[bool],
) -> StabilityResult:
    """计算不含任何性能修正项的新 STAB。"""

    if len(runs) != 3:
        raise MetricContractError(f"STAB 需要恰好 3 个 seed，实际为 {len(runs)}")
    if len(structural_pair_results) != 3:
        raise MetricContractError(
            f"三个 seed 应产生恰好 3 个结构 pair，实际为 {len(structural_pair_results)}"
        )
    checked = [run.validated() for run in runs]
    numerical = numerical_consistency(checked)
    validity = sum(1.0 for run in checked if run.valid) / 3.0
    structural = sum(1.0 for result in structural_pair_results if bool(result)) / 3.0
    score = (numerical * validity * structural) ** (1.0 / 3.0)
    return StabilityResult(
        numerical_consistency=numerical,
        validity=validity,
        structural_consistency=structural,
        score=score,
    )

