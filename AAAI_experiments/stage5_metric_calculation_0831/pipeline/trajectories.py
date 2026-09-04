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
    id_nmse: float | None
    ood_nmse: float | None


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


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _internal_objective_policy(algorithm: str | None) -> tuple[str, str] | None:
    tool = str(algorithm or "").strip().lower()
    if tool in {
        "gplearn",
        "jaxsr",
        "jaxsr_wrapper",
        "qlattice",
        "qlattice_wrapper",
    }:
        return "source_loss", "min"
    if tool in {
        "drsr",
        "drsr_wrapper",
        "dso",
        "imcts",
        "imcts_wrapper",
    }:
        return "source_score", "max"
    if tool in {"symbolfit", "symbolfit_wrapper"}:
        return "source_internal_loss", "min"
    return None


def _is_objective_improvement(current: float, previous: float, *, direction: str) -> bool:
    if direction == "min":
        return current < previous
    if direction == "max":
        return current > previous
    raise TrajectoryContractError(f"未知内部目标方向: {direction!r}")


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


def _candidate(
    payload: Mapping[str, Any],
    minute: int,
    *,
    retain_invalid_metrics: bool = False,
) -> _Candidate | None:
    expression = canonical_expression(payload)
    id_nmse = _split_nmse(payload, "id_test")
    ood_nmse = _split_nmse(payload, "ood_test")
    if not expression:
        return None
    if not retain_invalid_metrics and (id_nmse is None or ood_nmse is None):
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
    if candidate.id_nmse is None or candidate.ood_nmse is None:
        return TrajectoryPoint(
            minute=minute,
            id_quality=0.0,
            ood_quality=0.0,
            quality=0.0,
            expression=candidate.expression,
            source=source,
            valid_output=False,
        )
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


def _empty_point(minute: int, *, source: str = "explicit_no_valid_output") -> TrajectoryPoint:
    return TrajectoryPoint(
        minute=minute,
        id_quality=0.0,
        ood_quality=0.0,
        quality=0.0,
        expression="",
        source=source,
        valid_output=False,
    )


def reconstruct_trajectory(
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    horizon: int = 180,
    algorithm: str | None = None,
) -> list[TrajectoryPoint]:
    """从完整检查点重建固定网格，并按算法内部目标维护历史最优。"""

    if horizon <= 0:
        raise TrajectoryContractError("horizon 必须为正整数")
    unexpected = sorted(set(snapshots) - set(range(1, horizon + 1)))
    if unexpected:
        raise TrajectoryContractError(f"轨迹包含预算外检查点: {unexpected[:10]}")

    latest: _Candidate | None = None
    latest_objective: float | None = None
    objective_policy = _internal_objective_policy(algorithm)
    points: list[TrajectoryPoint] = []
    allowed = {
        "periodic_best",
        "periodic_heartbeat",
        "periodic_backfill",
        "final_best",
        "recovered_final",
        "budget_end_internal_best",
        "audited_carry_forward",
    }
    internal_endpoint_types = {
        "final_best",
        "recovered_final",
        "budget_end_internal_best",
    }
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
                prefix = f"future_backfill_ignored:{source_minute}"
                if latest is not None:
                    points.append(
                        _point_from_candidate(
                            latest,
                            minute,
                            f"{prefix};carry_forward:{latest.minute}",
                        )
                    )
                else:
                    points.append(
                        _empty_point(
                            minute,
                            source=f"{prefix};explicit_no_valid_output",
                        )
                    )
                continue

        current_objective: float | None = None
        if objective_policy is not None:
            objective_field, _ = objective_policy
            current_objective = _finite(payload.get(objective_field))
            if (
                record_type in internal_endpoint_types
                and current_objective is None
                and latest is not None
            ):
                points.append(
                    _point_from_candidate(
                        latest,
                        minute,
                        f"internal_best_carry_forward:{latest.minute}",
                    )
                )
                continue

        current = _candidate(
            payload,
            minute,
            retain_invalid_metrics=objective_policy is not None,
        )
        if current is not None:
            if objective_policy is not None:
                objective_field, direction = objective_policy
                if current_objective is None:
                    raise TrajectoryContractError(
                        f"minute_{minute:04d} 缺少 {algorithm} 内部目标 {objective_field}"
                    )
                if (
                    latest is not None
                    and latest_objective is not None
                    and not _is_objective_improvement(
                        current_objective,
                        latest_objective,
                        direction=direction,
                    )
                ):
                    points.append(
                        _point_from_candidate(
                            latest,
                            minute,
                            f"internal_best_carry_forward:{latest.minute}",
                        )
                    )
                    continue
                latest_objective = current_objective
            latest = current
            if record_type == "audited_carry_forward":
                provenance = payload.get("recovery_provenance")
                if not isinstance(provenance, Mapping):
                    raise TrajectoryContractError(
                        f"minute_{minute:04d} 缺少 recovery_provenance"
                    )
                source_minute = provenance.get("source_minute")
                source_label = f"audited_repair:{source_minute}"
            else:
                if record_type == "budget_end_internal_best":
                    source_label = f"budget_end_internal_best:{minute}"
                else:
                    prefix = (
                        "final"
                        if record_type in {"final_best", "recovered_final"}
                        else "snapshot"
                    )
                    source_label = f"{prefix}:{minute}"
            points.append(_point_from_candidate(current, minute, source_label))
            continue
        expression = canonical_expression(payload)
        if (
            objective_policy is not None
            and latest is not None
            and not expression
        ):
            points.append(
                _point_from_candidate(
                    latest,
                    minute,
                    f"internal_best_carry_forward:{latest.minute}",
                )
            )
            continue
        if expression or payload.get("status") in {"error", "failed", "invalid"}:
            latest = None
            latest_objective = None
            prefix = "final" if record_type == "final_best" else "snapshot"
            points.append(
                _empty_point(minute, source=f"{prefix}_evaluator_error:{minute}")
            )
            continue
        if latest is not None:
            points.append(_point_from_candidate(latest, minute, f"carry_forward:{latest.minute}"))
        else:
            points.append(_empty_point(minute))
    return points
