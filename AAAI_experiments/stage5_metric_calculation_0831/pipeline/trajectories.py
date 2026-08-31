"""180 分钟 best-so-far 轨迹的严格重建。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .metrics import phi_nmse


class TrajectoryContractError(ValueError):
    """分钟轨迹缺失、冲突或包含未来信息。"""


@dataclass(frozen=True)
class TrajectoryPoint:
    minute: int
    id_quality: float
    ood_quality: float
    quality: float
    expression: str
    source: str
    valid_output: bool


@dataclass(frozen=True)
class _Candidate:
    minute: int
    expression: str
    id_nmse: float
    ood_nmse: float


def _finite_nonnegative(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def canonical_expression(payload: Mapping[str, Any]) -> str:
    """选择实际参与预测的公式，拟合后常数优先。"""

    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    candidates = (
        artifact.get("instantiated_expression"),
        artifact.get("normalized_expression"),
        artifact.get("return_expression_source"),
        payload.get("equation"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _split_nmse(payload: Mapping[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        return None
    return _finite_nonnegative(block.get("nmse"))


def _candidate(payload: Mapping[str, Any], minute: int) -> _Candidate | None:
    expression = canonical_expression(payload)
    id_nmse = _split_nmse(payload, "id_test")
    ood_nmse = _split_nmse(payload, "ood_test")
    if not expression or id_nmse is None or ood_nmse is None:
        return None
    return _Candidate(
        minute=minute,
        expression=expression,
        id_nmse=id_nmse,
        ood_nmse=ood_nmse,
    )


def _validate_checkpoint(payload: Mapping[str, Any], minute: int, record_type: str) -> None:
    index = payload.get("checkpoint_index")
    if record_type == "final_best" and minute == 180 and index == "final":
        return
    try:
        parsed = int(index)
    except (TypeError, ValueError):
        raise TrajectoryContractError(
            f"minute_{minute:04d} 的 checkpoint_index 无效: {index!r}"
        ) from None
    if parsed != minute:
        raise TrajectoryContractError(
            f"minute_{minute:04d} 的 checkpoint_index={parsed} 不匹配"
        )


def _point_from_candidate(candidate: _Candidate, minute: int, source: str) -> TrajectoryPoint:
    id_quality = phi_nmse(candidate.id_nmse)
    ood_quality = phi_nmse(candidate.ood_nmse)
    return TrajectoryPoint(
        minute=minute,
        id_quality=id_quality,
        ood_quality=ood_quality,
        quality=(id_quality + ood_quality) / 2.0,
        expression=candidate.expression,
        source=source,
        valid_output=True,
    )


def _empty_point(minute: int) -> TrajectoryPoint:
    return TrajectoryPoint(
        minute=minute,
        id_quality=0.0,
        ood_quality=0.0,
        quality=0.0,
        expression="",
        source="explicit_no_valid_output",
        valid_output=False,
    )


def reconstruct_trajectory(
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    horizon: int = 180,
) -> list[TrajectoryPoint]:
    """从完整检查点映射重建固定网格，拒绝缺失和 future backfill。"""

    if horizon <= 0:
        raise TrajectoryContractError("horizon 必须为正整数")
    unexpected = sorted(set(snapshots) - set(range(1, horizon + 1)))
    if unexpected:
        raise TrajectoryContractError(f"轨迹包含预算外检查点: {unexpected[:10]}")

    latest: _Candidate | None = None
    points: list[TrajectoryPoint] = []
    allowed = {"periodic_best", "periodic_heartbeat", "periodic_backfill", "final_best"}
    for minute in range(1, horizon + 1):
        payload = snapshots.get(minute)
        if payload is None:
            raise TrajectoryContractError(f"轨迹缺失 minute_{minute:04d}.json")
        if not isinstance(payload, Mapping):
            raise TrajectoryContractError(f"minute_{minute:04d} 不是 JSON object")
        record_type = payload.get("record_type")
        if record_type not in allowed:
            raise TrajectoryContractError(
                f"minute_{minute:04d} 的 record_type 无效: {record_type!r}"
            )
        record_type = str(record_type)
        _validate_checkpoint(payload, minute, record_type)

        if record_type == "periodic_backfill":
            try:
                source_minute = int(payload.get("backfilled_from_minute"))
            except (TypeError, ValueError):
                raise TrajectoryContractError(
                    f"minute_{minute:04d} 的 backfilled_from_minute 无效"
                ) from None
            if source_minute > minute:
                raise TrajectoryContractError(
                    f"minute_{minute:04d} 使用 minute_{source_minute:04d} 的未来信息"
                )

        current = _candidate(payload, minute)
        if current is not None:
            latest = current
            prefix = "final" if record_type == "final_best" else "snapshot"
            points.append(_point_from_candidate(current, minute, f"{prefix}:{minute}"))
            continue
        if latest is not None:
            points.append(_point_from_candidate(latest, minute, f"carry_forward:{latest.minute}"))
        else:
            points.append(_empty_point(minute))
    return points

