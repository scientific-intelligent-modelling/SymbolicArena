"""为正式六轴评测准备 clean 条件下的运行级 EFF 输入。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .freeze_binding import FreezeBindingContractError, validate_freeze_binding_summary
from .metrics import MetricContractError, phi_nmse
from .performance_replay import (
    EVALUATION_PATH,
    FormulaRecoveryManifest,
    PerformanceReplayCache,
    PerformanceReplayError,
    load_formula_recovery_manifest,
    replay_payload_performance,
)
from .trajectories import (
    TrajectoryContractError,
    TrajectoryPoint,
    canonical_expression,
    reconstruct_trajectory,
)
from .trajectory_repairs import (
    TrajectoryRepairContractError,
    apply_repair_manifest,
    load_repair_manifest,
)


HORIZON = 180
NOISE_TAG = "clean"
EXPECTED_HOSTS = 8
EXPECTED_TASKS = 2250
EXPECTED_POINTS = 405000
EXPECTED_EXISTING_POINTS = 404985
EXPECTED_MISSING_POINTS = 15
EXPECTED_AUDITED_REPAIR_POINTS = 15
EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS = 34
EXPECTED_CHECKPOINT_NORMALIZATION_POINTS = 19
EXPECTED_OVERLAY_REPLACEMENTS = 225
OVERLAY_SCHEMA_VERSION = "clean_rerun_eff_overlay_v1"
OVERLAY_SCOPE = "mixed_clean_rerun_overlay"
TRAJECTORY_EVIDENCE_SCHEMA_VERSION = "clean_selected_trajectory_evidence.v1"
NATIVE_EFF_SCHEMA_VERSION = "algorithm_native_internal_best_so_far.v1"


class EffPreparationContractError(ValueError):
    """冻结 bundle、修复清单或 EFF 准备过程不满足正式契约。"""


@dataclass(frozen=True)
class NativeObjectiveAdapter:
    """一个算法的原生 incumbent 选择合同。"""

    algorithm: str
    objective_options: tuple[tuple[str, str], ...]
    native_rule: str
    tie_break: str = "keep_earliest_incumbent"


@dataclass(frozen=True)
class NativeTrajectoryResult:
    """严格原生选择后的轨迹及其逐分钟选择证据。"""

    points: tuple[TrajectoryPoint, ...]
    objective_fields: tuple[str | None, ...]
    objective_values: tuple[float | None, ...]
    incumbent_source_minutes: tuple[int | None, ...]
    q_star: float
    m_eff: float


def _adapter(
    algorithm: str,
    *objective_options: tuple[str, str],
    native_rule: str,
) -> NativeObjectiveAdapter:
    return NativeObjectiveAdapter(
        algorithm=algorithm,
        objective_options=tuple(objective_options),
        native_rule=native_rule,
    )


NATIVE_OBJECTIVE_ADAPTERS: dict[str, NativeObjectiveAdapter] = {
    "drsr": _adapter("drsr", ("source_score", "max"), native_rule="maximum native program score"),
    "dso": _adapter("dso", ("source_score", "max"), native_rule="maximum native reward"),
    "e2esr": _adapter("e2esr", ("source_score", "max"), native_rule="maximum native tree score"),
    "fepysr": _adapter("fepysr", ("source_score", "min"), native_rule="minimum training MSE"),
    "gplearn": _adapter("gplearn", ("source_loss", "min"), native_rule="minimum native fitness loss"),
    "imcts": _adapter("imcts", ("source_score", "max"), native_rule="maximum native MCTS reward"),
    "jaxsr": _adapter(
        "jaxsr",
        ("source_internal_loss", "min"),
        ("source_loss", "min"),
        native_rule="minimum frozen native loss channel",
    ),
    "llmsr": _adapter(
        "llmsr",
        ("source_loss", "min"),
        ("source_score", "max"),
        native_rule="native top-sample rule: NMSE/MSE minimum, otherwise score maximum",
    ),
    "pyoperon": _adapter("pyoperon", ("source_loss", "min"), native_rule="minimum native model-selection loss"),
    "pysr": _adapter("pysr", ("source_loss", "min"), native_rule="minimum native hall-of-fame loss"),
    "qlattice": _adapter("qlattice", ("source_loss", "min"), native_rule="minimum BIC"),
    "ragsr": _adapter("ragsr", ("source_score", "max"), native_rule="maximum native hall-of-fame fitness"),
    "symbolfit": _adapter("symbolfit", ("source_internal_loss", "min"), native_rule="minimum active PySR internal loss"),
    "tpsr": _adapter("tpsr", ("source_score", "max"), native_rule="maximum native planning reward"),
    "udsr": _adapter("udsr", ("source_score", "max"), native_rule="maximum native reward"),
}


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _payload_nmse(payload: Mapping[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        return None
    value = _finite_number(block.get("nmse"))
    return value if value is not None and value >= 0.0 else None


def _native_point(
    payload: Mapping[str, Any],
    *,
    minute: int,
    expression: str,
    source: str,
) -> TrajectoryPoint:
    id_nmse = _payload_nmse(payload, "id_test")
    ood_nmse = _payload_nmse(payload, "ood_test")
    if id_nmse is None or ood_nmse is None:
        return TrajectoryPoint(minute, 0.0, 0.0, 0.0, expression, source, False)
    id_quality = phi_nmse(id_nmse)
    ood_quality = phi_nmse(ood_nmse)
    return TrajectoryPoint(
        minute,
        id_quality,
        ood_quality,
        (id_quality + ood_quality) / 2.0,
        expression,
        source,
        True,
    )


def _native_empty_point(minute: int, source: str) -> TrajectoryPoint:
    return TrajectoryPoint(minute, 0.0, 0.0, 0.0, "", source, False)


def _required_evidence_action(algorithm: str, reason: str) -> str:
    tool = str(algorithm).strip().lower().replace("-", "")
    if "缺少可审计原生目标" in reason:
        if tool in {"e2esr", "llmsr"}:
            return "rerun_with_native_objective_telemetry"
        return "recollect_native_objective_evidence_or_rerun"
    if "统一评估" in reason:
        return "recover_canonical_execution_evidence_or_rerun"
    return "audit_source_evidence_before_rerun"


def reconstruct_native_incumbent_trajectory(
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    algorithm: str,
    horizon: int = HORIZON,
) -> NativeTrajectoryResult:
    """仅按原生目标选择 incumbent；ID/OOD 只在选择后用于事后评分。"""

    tool = str(algorithm).strip().lower().replace("-", "")
    adapter = NATIVE_OBJECTIVE_ADAPTERS.get(tool)
    if adapter is None:
        raise EffPreparationContractError(f"{algorithm} 没有冻结的原生 EFF adapter")
    if horizon <= 0:
        raise EffPreparationContractError("horizon 必须为正整数")
    unexpected = sorted(set(snapshots) - set(range(1, horizon + 1)))
    if unexpected:
        raise EffPreparationContractError(f"轨迹包含预算外分钟: {unexpected[:10]}")

    incumbent_payload: Mapping[str, Any] | None = None
    incumbent_expression = ""
    incumbent_objective: float | None = None
    incumbent_minute: int | None = None
    selected_field: str | None = None
    selected_direction: str | None = None
    points: list[TrajectoryPoint] = []
    objective_fields: list[str | None] = []
    objective_values: list[float | None] = []
    source_minutes: list[int | None] = []
    endpoint_types = {"final_best", "recovered_final", "budget_end_internal_best"}

    def append_current(minute: int, source: str) -> None:
        if incumbent_payload is None or incumbent_minute is None:
            points.append(_native_empty_point(minute, source))
            objective_fields.append(None)
            objective_values.append(None)
            source_minutes.append(None)
            return
        points.append(
            _native_point(
                incumbent_payload,
                minute=minute,
                expression=incumbent_expression,
                source=source,
            )
        )
        objective_fields.append(selected_field)
        objective_values.append(incumbent_objective)
        source_minutes.append(incumbent_minute)

    for minute in range(1, horizon + 1):
        payload = snapshots.get(minute)
        if not isinstance(payload, Mapping):
            raise EffPreparationContractError(
                f"minute_{minute:04d} 缺少可审计快照证据"
            )
        record_type = str(payload.get("record_type") or "")
        if record_type == "periodic_backfill":
            try:
                source_minute = int(payload.get("backfilled_from_minute"))
            except (TypeError, ValueError):
                raise EffPreparationContractError(
                    f"minute_{minute:04d} backfilled_from_minute 无效"
                ) from None
            if source_minute > minute:
                append_current(
                    minute,
                    f"future_backfill_ignored:{source_minute};"
                    + (
                        f"native_carry_forward:{incumbent_minute}"
                        if incumbent_minute is not None
                        else "pre_discovery_zero"
                    ),
                )
                continue

        expression = canonical_expression(payload)
        if record_type in endpoint_types and incumbent_payload is not None:
            objective_at_endpoint = (
                _finite_number(payload.get(selected_field)) if selected_field else None
            )
            if objective_at_endpoint is None:
                append_current(minute, f"native_endpoint_carry_forward:{incumbent_minute}")
                continue
        if not expression:
            if any(
                _finite_number(payload.get(field)) is not None
                for field, _ in adapter.objective_options
            ):
                raise EffPreparationContractError(
                    f"minute_{minute:04d} 有原生目标但缺少可审计表达式"
                )
            append_current(
                minute,
                f"native_carry_forward:{incumbent_minute}"
                if incumbent_minute is not None
                else "pre_discovery_zero",
            )
            continue

        objective_field = selected_field
        objective_direction = selected_direction
        objective_value = (
            _finite_number(payload.get(selected_field)) if selected_field else None
        )
        if selected_field is None:
            for candidate_field, candidate_direction in adapter.objective_options:
                candidate_value = _finite_number(payload.get(candidate_field))
                if candidate_value is not None:
                    objective_field = candidate_field
                    objective_direction = candidate_direction
                    objective_value = candidate_value
                    break
        if objective_value is None:
            if incumbent_payload is not None and expression == incumbent_expression:
                append_current(minute, f"native_carry_forward:{incumbent_minute}")
                continue
            expected = "/".join(field for field, _ in adapter.objective_options)
            raise EffPreparationContractError(
                f"minute_{minute:04d} 新表达式缺少可审计原生目标 {expected}"
            )

        assert objective_field is not None and objective_direction is not None
        improves = incumbent_objective is None
        if incumbent_objective is not None:
            improves = (
                objective_value < incumbent_objective
                if objective_direction == "min"
                else objective_value > incumbent_objective
            )
        if improves:
            replay_error = str(payload.get("canonical_replay_error") or "").strip()
            if replay_error:
                raise EffPreparationContractError(
                    f"minute_{minute:04d} 原生 incumbent 缺少可审计统一评估: "
                    f"{replay_error}"
                )
            incumbent_payload = payload
            incumbent_expression = expression
            incumbent_objective = objective_value
            incumbent_minute = minute
            selected_field = objective_field
            selected_direction = objective_direction
            points.append(
                _native_point(
                    payload,
                    minute=minute,
                    expression=expression,
                    source=f"native_incumbent:{minute}:{objective_field}",
                )
            )
        else:
            points.append(
                _native_point(
                    incumbent_payload,
                    minute=minute,
                    expression=incumbent_expression,
                    source=f"native_carry_forward:{incumbent_minute}",
                )
            )
        objective_fields.append(selected_field)
        objective_values.append(incumbent_objective)
        source_minutes.append(incumbent_minute)

    qualities = [point.quality for point in points]
    q_star = max(qualities, default=0.0)
    m_eff = (
        sum(quality / q_star for quality in qualities) / horizon
        if q_star > 0.0
        else 0.0
    )
    return NativeTrajectoryResult(
        points=tuple(points),
        objective_fields=tuple(objective_fields),
        objective_values=tuple(objective_values),
        incumbent_source_minutes=tuple(source_minutes),
        q_star=q_star,
        m_eff=m_eff,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage5_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _raise(message: str) -> None:
    raise EffPreparationContractError(message)


def _resolve_path(base: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (base / path).resolve()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EffPreparationContractError(f"{label} 不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        _raise(f"{label} 顶层必须是 JSON object: {path}")
    return payload


def _logical_key(source: Mapping[str, Any]) -> str:
    try:
        seed = int(source["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EffPreparationContractError(f"source.seed 无效: {source!r}") from exc
    return (
        f"{source.get('algorithm')}::{source.get('dataset_id')}::"
        f"s{seed}::{source.get('noise_tag')}"
    )


def _stable_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["algorithm"]),
        str(row["dataset_id"]),
        int(row["seed"]),
        str(row["task_id"]),
        str(row["host"]),
    )


def load_freeze_binding_report(
    path: Path,
    *,
    repo_root: Path,
    expected_hosts: int | None,
    expected_tasks: int | None,
    expected_points: int | None,
    expected_existing_points: int | None,
    expected_missing_points: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _resolve_path(repo_root, path)
    payload = _load_json_object(resolved, label="freeze_binding report")
    try:
        validate_freeze_binding_summary(
            payload,
            expected_hosts=expected_hosts,
            expected_tasks=expected_tasks,
            expected_points=expected_points,
            expected_existing_points=expected_existing_points,
            expected_missing_points=expected_missing_points,
        )
    except FreezeBindingContractError as exc:
        raise EffPreparationContractError(f"freeze_binding report 契约失败: {exc}") from exc
    if payload.get("noise_tag") != NOISE_TAG:
        _raise(f"freeze_binding report noise_tag 必须为 {NOISE_TAG}")
    if int(payload.get("horizon", 0)) != HORIZON:
        _raise(f"freeze_binding report horizon 必须为 {HORIZON}")
    report_info = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    return payload, report_info


def _verified_input_files(
    summary: Mapping[str, Any],
    *,
    repo_root: Path,
    section_name: str,
) -> dict[str, dict[str, Any]]:
    input_files = summary.get("input_files")
    if not isinstance(input_files, Mapping):
        _raise("freeze_binding report 缺少 input_files")
    entries = input_files.get(section_name)
    if not isinstance(entries, list) or not entries:
        _raise(f"freeze_binding report 缺少 {section_name}")
    verified: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            _raise(f"{section_name} 条目必须是 object")
        host = str(item.get("host") or "")
        raw_path = item.get("path")
        expected_sha = str(item.get("sha256") or "")
        if not host or not raw_path or not expected_sha:
            _raise(f"{section_name} 条目缺少 host/path/sha256")
        resolved = _resolve_path(repo_root, raw_path)
        if not resolved.is_file():
            _raise(f"{section_name} 文件不存在: {resolved}")
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            _raise(f"{section_name} SHA 不匹配: {resolved}")
        if host in verified:
            _raise(f"{section_name} 出现重复 host: {host}")
        verified[host] = {
            "host": host,
            "path": str(resolved),
            "sha256": actual_sha,
            "size_bytes": resolved.stat().st_size,
        }
    return dict(sorted(verified.items()))


def _iter_bundle_records(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EffPreparationContractError(
                    f"bundle {path.name}:{line_number} 不是合法 JSON"
                ) from exc
            if not isinstance(payload, dict):
                _raise(f"bundle {path.name}:{line_number} 顶层必须是 JSON object")
            yield payload


def _strict_overlay_snapshots(
    record: Mapping[str, Any], *, logical_key: str
) -> dict[int, dict[str, Any]]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        _raise(f"{logical_key} overlay 缺少 source")
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != HORIZON:
        _raise(f"{logical_key} overlay 必须恰有 {HORIZON} 个快照")
    parsed: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            _raise(f"{logical_key} overlay snapshot 必须是 object")
        try:
            minute = int(snapshot.get("minute"))
        except (TypeError, ValueError) as exc:
            raise EffPreparationContractError(
                f"{logical_key} overlay minute 无效"
            ) from exc
        if minute in parsed:
            _raise(f"{logical_key} overlay minute_{minute:04d} 重复")
        if snapshot.get("status") != "ok" or snapshot.get("conflict") is True:
            _raise(f"{logical_key} overlay minute_{minute:04d} 不可用")
        raw_text = snapshot.get("raw_text")
        selected_sha256 = snapshot.get("selected_sha256")
        if not isinstance(raw_text, str) or not isinstance(selected_sha256, str):
            _raise(f"{logical_key} overlay minute_{minute:04d} 缺少 raw/SHA")
        if _sha256_bytes(raw_text.encode("utf-8")) != selected_sha256:
            _raise(f"{logical_key} overlay minute_{minute:04d} raw SHA 不匹配")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise EffPreparationContractError(
                f"{logical_key} overlay minute_{minute:04d} raw 不是 JSON"
            ) from exc
        if not isinstance(payload, dict):
            _raise(f"{logical_key} overlay minute_{minute:04d} raw 顶层不是 object")
        if payload.get("checkpoint_index") != minute:
            _raise(f"{logical_key} overlay minute_{minute:04d} checkpoint 身份不匹配")
        if str(payload.get("tool", "")).lower() != str(
            source.get("algorithm", "")
        ).lower():
            _raise(f"{logical_key} overlay minute_{minute:04d} algorithm 身份不匹配")
        if str(payload.get("dataset", source.get("dataset_id"))) != str(
            source.get("dataset_id")
        ) or int(payload.get("seed", source.get("seed", -1))) != int(source.get("seed", -1)):
            _raise(f"{logical_key} overlay minute_{minute:04d} dataset/seed 身份不匹配")
        if minute == HORIZON:
            if payload.get("record_type") != "budget_end_internal_best":
                _raise(f"{logical_key} overlay minute_0180 不是 budget_end_internal_best")
        elif payload.get("record_type") not in {
            "periodic_best",
            "periodic_heartbeat",
            "periodic_backfill",
        }:
            _raise(f"{logical_key} overlay minute_{minute:04d} record_type 无效")
        parsed[minute] = payload
    expected_minutes = set(range(1, HORIZON + 1))
    if set(parsed) != expected_minutes:
        _raise(f"{logical_key} overlay 分钟网格不是严格 1..{HORIZON}")
    return parsed


def load_rerun_overlay_manifest(
    path: Path,
    *,
    repo_root: Path,
    expected_replacements: int,
) -> tuple[
    dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]],
    dict[str, Any],
]:
    resolved_manifest = _resolve_path(repo_root, path)
    manifest = _load_json_object(resolved_manifest, label="rerun overlay manifest")
    if manifest.get("schema_version") != OVERLAY_SCHEMA_VERSION:
        _raise("rerun overlay manifest schema_version 不匹配")
    if manifest.get("status") != "passed" or manifest.get("scope") != OVERLAY_SCOPE:
        _raise("rerun overlay manifest status/scope 不满足正式契约")
    if int(manifest.get("horizon_minutes", 0)) != HORIZON:
        _raise(f"rerun overlay horizon 必须为 {HORIZON}")
    declared_keys = manifest.get("eff_replacement_keys")
    if not isinstance(declared_keys, list) or not all(
        isinstance(key, str) for key in declared_keys
    ):
        _raise("rerun overlay manifest eff_replacement_keys 无效")
    if len(set(declared_keys)) != len(declared_keys):
        _raise("rerun overlay manifest eff_replacement_keys 重复")
    declared_count = int(manifest.get("eff_replacement_count", -1))
    if declared_count != expected_replacements or len(declared_keys) != declared_count:
        _raise(
            "rerun overlay replacement 数量不符: "
            f"{declared_count}/{len(declared_keys)} != {expected_replacements}"
        )
    if int(manifest.get("replacement_count", -1)) != declared_count:
        _raise("rerun overlay manifest replacement_count 不一致")
    if int(manifest.get("overlay_unique_keys", -1)) != declared_count:
        _raise("rerun overlay manifest overlay_unique_keys 不一致")
    checkpoint_identity = manifest.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping) or (
        checkpoint_identity.get("all_verified") is not True
        or int(checkpoint_identity.get("expected_per_run", 0)) != HORIZON
        or int(checkpoint_identity.get("verified_runs", 0)) != declared_count
        or int(checkpoint_identity.get("verified_points", 0))
        != declared_count * HORIZON
    ):
        _raise("rerun overlay manifest checkpoint_identity 不完整")
    outputs = manifest.get("outputs")
    bundle_info = outputs.get("overlay_bundle") if isinstance(outputs, Mapping) else None
    if not isinstance(bundle_info, Mapping):
        _raise("rerun overlay manifest 缺少 overlay_bundle")
    bundle_path = _resolve_path(repo_root, bundle_info.get("path"))
    if not bundle_path.is_file():
        _raise(f"rerun overlay bundle 不存在: {bundle_path}")
    expected_sha = str(bundle_info.get("sha256") or "")
    actual_sha = sha256_file(bundle_path)
    if actual_sha != expected_sha:
        _raise(f"rerun overlay bundle SHA 不匹配: {bundle_path}")
    if int(bundle_info.get("size_bytes", -1)) != bundle_path.stat().st_size:
        _raise("rerun overlay bundle size_bytes 不匹配")
    if int(bundle_info.get("rows", -1)) != declared_count:
        _raise("rerun overlay bundle rows 不匹配")

    records: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    algorithm_counts: Counter[str] = Counter()
    for record in _iter_bundle_records(bundle_path):
        source = record.get("source")
        overlay = record.get("overlay")
        if not isinstance(source, Mapping) or not isinstance(overlay, Mapping):
            _raise("rerun overlay record 缺少 source/overlay")
        logical_key = _logical_key(source)
        if logical_key in records:
            _raise(f"overlay logical_key 重复: {logical_key}")
        if source.get("noise_tag") != NOISE_TAG:
            _raise(f"{logical_key} overlay 不是 clean")
        if overlay.get("schema_version") != OVERLAY_SCHEMA_VERSION:
            _raise(f"{logical_key} overlay schema_version 不一致")
        if overlay.get("logical_key") != logical_key:
            _raise(f"{logical_key} overlay.logical_key 不一致")
        if overlay.get("scope") != OVERLAY_SCOPE or overlay.get(
            "replacement_scope"
        ) not in {"final_and_eff", "eff_only"}:
            _raise(f"{logical_key} overlay scope 无效")
        result = record.get("result")
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            _raise(f"{logical_key} overlay result 无效")
        result_raw = result.get("raw_text")
        if not isinstance(result_raw, str) or _sha256_bytes(
            result_raw.encode("utf-8")
        ) != result.get("sha256"):
            _raise(f"{logical_key} overlay result raw SHA 不匹配")
        try:
            result_payload = json.loads(result_raw)
        except json.JSONDecodeError as exc:
            raise EffPreparationContractError(
                f"{logical_key} overlay result raw 不是 JSON"
            ) from exc
        if not isinstance(result_payload, Mapping) or result_payload.get("status") != "ok":
            _raise(f"{logical_key} overlay result payload 无效")
        if str(result_payload.get("tool", "")).lower() != str(
            source.get("algorithm", "")
        ).lower() or str(result_payload.get("dataset", "")) != str(
            source.get("dataset_id", "")
        ) or int(result_payload.get("seed", -1)) != int(source.get("seed", -1)):
            _raise(f"{logical_key} overlay result 身份不匹配")
        parsed = _strict_overlay_snapshots(record, logical_key=logical_key)
        records[logical_key] = (record, parsed)
        algorithm_counts[str(source.get("algorithm", "")).lower()] += 1
    if set(records) != set(declared_keys):
        _raise("rerun overlay bundle keys 与 eff_replacement_keys 不一致")
    declared_algorithm_counts = manifest.get("algorithm_counts")
    if isinstance(declared_algorithm_counts, Mapping) and dict(
        sorted(algorithm_counts.items())
    ) != {str(key).lower(): int(value) for key, value in declared_algorithm_counts.items()}:
        _raise("rerun overlay algorithm_counts 不一致")
    info = {
        "path": str(resolved_manifest),
        "sha256": sha256_file(resolved_manifest),
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "scope": OVERLAY_SCOPE,
        "replacement_count": len(records),
        "bundle_path": str(bundle_path),
        "bundle_sha256": actual_sha,
        "bundle_size_bytes": bundle_path.stat().st_size,
    }
    return records, info


def _count_prefixed_sources(sources: Iterable[str], prefix: str) -> int:
    return sum(1 for source in sources if str(source).startswith(prefix))


def _future_backfill_minutes(raw_snapshots: list[Mapping[str, Any]], target_minute: int) -> list[int]:
    matched: list[int] = []
    for minute, snapshot in enumerate(raw_snapshots, start=1):
        if snapshot.get("status") != "ok":
            continue
        if snapshot.get("record_type") != "periodic_backfill":
            continue
        try:
            backfilled_from_minute = int(snapshot.get("backfilled_from_minute"))
        except (TypeError, ValueError):
            continue
        if backfilled_from_minute == target_minute and minute < target_minute:
            matched.append(minute)
    return matched


def _normalize_checkpoint_drift(
    parsed_snapshots: dict[int, dict[str, Any]],
    *,
    raw_snapshots: list[Mapping[str, Any]],
    logical_key: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    normalized = {minute: dict(payload) for minute, payload in parsed_snapshots.items()}
    audit: list[dict[str, Any]] = []
    for minute, payload in normalized.items():
        record_type = payload.get("record_type")
        index = payload.get("checkpoint_index")
        if record_type == "final_best" and minute == HORIZON and index == "final":
            continue
        try:
            parsed_index = int(index)
        except (TypeError, ValueError) as exc:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} checkpoint_index 无效: {index!r}"
            ) from exc
        if parsed_index == minute:
            continue
        if record_type != "periodic_best" or parsed_index > minute:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} 的 checkpoint_index={parsed_index} 不匹配"
            )
        leaked_minutes = _future_backfill_minutes(raw_snapshots, minute)
        if not leaked_minutes:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} 的 checkpoint_index={parsed_index} 缺少 future backfill 佐证"
            )
        payload["checkpoint_index_original"] = parsed_index
        payload["checkpoint_index"] = minute
        payload["checkpoint_index_normalization"] = {
            "reason": "future_backfill_source_minute",
            "future_backfill_minutes": leaked_minutes,
        }
        audit.append(
            {
                "minute": minute,
                "original_checkpoint_index": parsed_index,
                "normalized_checkpoint_index": minute,
                "future_backfill_minutes": leaked_minutes,
            }
        )
    return normalized, audit


def _replay_trajectory_payloads(
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    algorithm: str,
    repo_root: Path,
    cache: PerformanceReplayCache,
    recovery_manifest: FormulaRecoveryManifest | None = None,
    task_id: str | None = None,
    condition: str = NOISE_TAG,
    result_sha256: str | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """让所有可用候选经同一 canonical 执行器评分，失败候选显式记为零质量。"""

    replayed: dict[int, dict[str, Any]] = {}
    counts = {
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "invalid_output": 0,
        "artifact_rebuilt": 0,
    }
    for minute, source_payload in snapshots.items():
        payload = dict(source_payload)
        expression = canonical_expression(payload)
        if not expression:
            replayed[minute] = payload
            continue
        if payload.get("record_type") == "periodic_backfill":
            try:
                source_minute = int(payload.get("backfilled_from_minute"))
            except (TypeError, ValueError):
                source_minute = minute
            if source_minute > minute:
                replayed[minute] = payload
                continue

        counts["attempted"] += 1
        native_evaluation = {
            "id_test": payload.get("id_test"),
            "ood_test": payload.get("ood_test"),
        }
        try:
            replay = replay_payload_performance(
                payload,
                algorithm=algorithm,
                repo_root=repo_root,
                cache=cache,
                recovery_manifest=recovery_manifest,
                task_id=task_id,
                condition=condition,
                result_sha256=result_sha256,
            )
        except PerformanceReplayError as exc:
            counts["failed"] += 1
            payload["status"] = "invalid"
            payload["canonical_replay_error"] = str(exc)
            payload["native_evaluation"] = native_evaluation
            payload["id_test"] = None
            payload["ood_test"] = None
        else:
            counts["succeeded"] += 1
            counts["artifact_rebuilt"] += int(bool(replay["artifact_rebuilt"]))
            payload["evaluation_path"] = EVALUATION_PATH
            payload["canonical_artifact"] = replay["canonical_artifact"]
            payload["canonical_artifact_sha256"] = replay["canonical_artifact_sha256"]
            payload["native_evaluation"] = native_evaluation
            if bool(replay["valid_output"]):
                payload["id_test"] = replay["id_test"]
                payload["ood_test"] = replay["ood_test"]
            else:
                invalid_reason = str(replay.get("invalid_reason") or "").strip()
                if not invalid_reason:
                    counts["failed"] += 1
                    counts["succeeded"] -= 1
                    payload["status"] = "invalid"
                    payload["canonical_replay_error"] = (
                        "canonical replay 的无效输出缺少 invalid_reason"
                    )
                else:
                    counts["invalid_output"] += 1
                    payload["status"] = "invalid"
                    payload["canonical_replay_invalid_reason"] = invalid_reason
                payload["id_test"] = None
                payload["ood_test"] = None
        replayed[minute] = payload
    return replayed, counts


def _trajectory_evidence(
    trajectory: Sequence[TrajectoryPoint], *, logical_key: str
) -> dict[str, list[Any]]:
    """保留同一所选 canonical 候选的表达式、数值分量和有效性。"""

    qualities: list[float] = []
    id_qualities: list[float] = []
    ood_qualities: list[float] = []
    expressions: list[str | None] = []
    valid_outputs: list[bool] = []
    sources: list[str] = []
    for point in trajectory:
        quality = float(point.quality)
        id_quality = float(point.id_quality)
        ood_quality = float(point.ood_quality)
        combined = (id_quality + ood_quality) / 2.0
        if not math.isclose(
            quality, combined, rel_tol=1.0e-15, abs_tol=1.0e-15
        ):
            raise EffPreparationContractError(
                f"{logical_key} minute_{point.minute:04d} 的 canonical "
                "ID/OOD 分量与 combined quality 不一致"
            )
        if not point.valid_output and not (
            quality == 0.0 and id_quality == 0.0 and ood_quality == 0.0
        ):
            raise EffPreparationContractError(
                f"{logical_key} minute_{point.minute:04d} 的无效候选没有显式零分"
            )
        qualities.append(quality)
        id_qualities.append(id_quality)
        ood_qualities.append(ood_quality)
        expressions.append(point.expression if point.expression else None)
        valid_outputs.append(bool(point.valid_output))
        sources.append(str(point.source))
    return {
        "quality_trajectory": qualities,
        "id_quality_trajectory": id_qualities,
        "ood_quality_trajectory": ood_qualities,
        "selected_expression_trajectory": expressions,
        "valid_output_trajectory": valid_outputs,
        "trajectory_sources": sources,
    }


def _minute_source_evidence(
    raw_snapshots: Sequence[Mapping[str, Any]],
    *,
    incumbent_source_minutes: Sequence[int | None],
    repair_audit: Mapping[str, Any],
    logical_key: str,
) -> tuple[list[str], list[str]]:
    """把每分钟 incumbent 绑定到冻结快照路径与 SHA256。"""

    frozen: dict[int, tuple[str, str]] = {}
    for fallback_minute, snapshot in enumerate(raw_snapshots, start=1):
        try:
            minute = int(snapshot.get("minute", fallback_minute))
        except (TypeError, ValueError):
            minute = fallback_minute
        path = str(snapshot.get("selected_path") or snapshot.get("outer_path") or "")
        sha256 = str(snapshot.get("selected_sha256") or snapshot.get("outer_sha256") or "")
        if path and len(sha256) == 64:
            frozen[minute] = (path, sha256)

    repair_source = None
    applied = repair_audit.get("applied_minutes")
    if isinstance(applied, list) and applied:
        candidates = {
            int(value)
            for value in incumbent_source_minutes
            if value is not None and int(value) not in set(int(item) for item in applied)
        }
        if candidates:
            repair_source = max(candidates)

    paths: list[str] = []
    hashes: list[str] = []
    for minute, incumbent_minute in enumerate(incumbent_source_minutes, start=1):
        evidence_minute = incumbent_minute if incumbent_minute is not None else minute
        evidence = frozen.get(int(evidence_minute))
        if evidence is None and repair_source is not None:
            evidence = frozen.get(repair_source)
        if evidence is None:
            raise EffPreparationContractError(
                f"{logical_key} minute_{minute:04d} 缺少来源路径/SHA256"
            )
        paths.append(evidence[0])
        hashes.append(evidence[1])
    return paths, hashes


def _csv_row_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "logical_key": record["logical_key"],
        "algorithm": record["algorithm"],
        "dataset_id": record["dataset_id"],
        "seed": record["seed"],
        "task_id": record["task_id"],
        "host": record["host"],
        "noise_tag": record["noise_tag"],
        "m_eff": f"{float(record['m_eff']):.17g}",
        "best_quality": f"{float(record['best_quality']):.17g}",
        "audited_repair_points": record["audited_repair_points"],
        "future_backfill_ignored_points": record["future_backfill_ignored_points"],
        "checkpoint_normalization_points": len(record["checkpoint_normalizations"]),
        "bundle_sha256": record["bundle_sha256"],
        "bundle_report_sha256": record["bundle_report_sha256"],
        "freeze_binding_report_sha256": record["freeze_binding_report_sha256"],
        "repair_manifest_sha256": record["repair_manifest_sha256"],
        "evaluation_path": record["evaluation_path"],
        "canonical_replay_attempted_points": record["canonical_replay_counts"]["attempted"],
        "canonical_replay_succeeded_points": record["canonical_replay_counts"]["succeeded"],
        "canonical_replay_failed_points": record["canonical_replay_counts"]["failed"],
        "canonical_replay_invalid_output_points": record["canonical_replay_counts"][
            "invalid_output"
        ],
        "internal_best_carry_points": record["internal_best_carry_points"],
        "trajectory_evidence_schema_version": record[
            "trajectory_evidence_schema_version"
        ],
        "native_eff_schema_version": record["native_eff_schema_version"],
        "native_rule": record["native_adapter"]["native_rule"],
        "native_tie_break": record["native_adapter"]["tie_break"],
    }
    for minute, value in enumerate(record["quality_trajectory"], start=1):
        row[f"q_{minute:04d}"] = f"{float(value):.17g}"
    for minute, value in enumerate(record["id_quality_trajectory"], start=1):
        row[f"id_q_{minute:04d}"] = f"{float(value):.17g}"
    for minute, value in enumerate(record["ood_quality_trajectory"], start=1):
        row[f"ood_q_{minute:04d}"] = f"{float(value):.17g}"
    for minute, value in enumerate(
        record["selected_expression_trajectory"], start=1
    ):
        row[f"expression_{minute:04d}"] = value if value is not None else ""
    for minute, value in enumerate(record["valid_output_trajectory"], start=1):
        row[f"valid_output_{minute:04d}"] = "true" if value else "false"
    for minute, value in enumerate(record["trajectory_sources"], start=1):
        row[f"trajectory_source_{minute:04d}"] = value
    for minute, value in enumerate(record["objective_field_trajectory"], start=1):
        row[f"objective_field_{minute:04d}"] = value or ""
    for minute, value in enumerate(record["objective_value_trajectory"], start=1):
        row[f"objective_value_{minute:04d}"] = (
            f"{float(value):.17g}" if value is not None else ""
        )
    for minute, value in enumerate(
        record["incumbent_source_minute_trajectory"], start=1
    ):
        row[f"incumbent_source_minute_{minute:04d}"] = (
            str(int(value)) if value is not None else ""
        )
    for minute, value in enumerate(record["source_path_trajectory"], start=1):
        row[f"source_path_{minute:04d}"] = value
    for minute, value in enumerate(record["source_sha256_trajectory"], start=1):
        row[f"source_sha256_{minute:04d}"] = value
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _raise("不允许写空 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_csv_row_from_record(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plain_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _raise(f"不允许写空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_eff_revision_artifacts(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """写出严格原生 EFF 的运行、曲线、覆盖率和 adapter 交付件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_path = output_dir / "run_trajectory_wide.csv"
    _write_csv(wide_path, [dict(row) for row in rows])

    unavailable_rows = [
        dict(item)
        for item in report.get("unresolved", [])
        if isinstance(item, Mapping) and "::" in str(item.get("logical_key", ""))
    ]
    available_run_eff_rows = [
        {
            "logical_key": row["logical_key"],
            "condition": row["noise_tag"],
            "algorithm": row["algorithm"],
            "dataset_id": row["dataset_id"],
            "seed": row["seed"],
            "q_star": f"{float(row['best_quality']):.17g}",
            "m_eff": f"{float(row['m_eff']):.17g}",
            "eff_score": f"{100.0 * float(row['m_eff']):.17g}",
            "availability_status": "available",
            "unavailable_reason": "",
            "required_action": "",
            "native_eff_schema_version": row["native_eff_schema_version"],
            "evaluation_path": row["evaluation_path"],
            "task_id": row["task_id"],
            "host": row["host"],
            "bundle_path": row["bundle_path"],
            "bundle_sha256": row["bundle_sha256"],
            "freeze_binding_report_path": row["freeze_binding_report_path"],
            "freeze_binding_report_sha256": row[
                "freeze_binding_report_sha256"
            ],
        }
        for row in rows
    ]
    unavailable_run_eff_rows = []
    for item in unavailable_rows:
        parts = str(item["logical_key"]).split("::")
        algorithm = str(item.get("algorithm") or parts[0])
        unavailable_run_eff_rows.append(
            {
                "logical_key": item["logical_key"],
                "condition": item.get("condition") or parts[-1],
                "algorithm": algorithm,
                "dataset_id": item.get("dataset_id") or parts[1],
                "seed": item.get("seed") or parts[2].removeprefix("s"),
                "q_star": "",
                "m_eff": "",
                "eff_score": "",
                "availability_status": "unavailable",
                "unavailable_reason": item["reason"],
                "required_action": item.get("required_action")
                or _required_evidence_action(algorithm, str(item["reason"])),
                "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
                "evaluation_path": EVALUATION_PATH,
                "task_id": item.get("task_id", ""),
                "host": item.get("host", ""),
                "bundle_path": item.get("bundle_path", ""),
                "bundle_sha256": item.get("bundle_sha256", ""),
                "freeze_binding_report_path": item.get(
                    "freeze_binding_report_path", ""
                ),
                "freeze_binding_report_sha256": item.get(
                    "freeze_binding_report_sha256", ""
                ),
            }
        )
    run_eff_rows = sorted(
        [*available_run_eff_rows, *unavailable_run_eff_rows],
        key=lambda row: (
            str(row["algorithm"]).lower(),
            str(row["dataset_id"]),
            int(row["seed"]),
        ),
    )
    if report.get("summary", {}).get("full_contract_checked") and len(run_eff_rows) != 2250:
        _raise(f"完整 run_eff 网格应为 2250 行，实际 {len(run_eff_rows)}")
    run_eff_path = output_dir / "run_eff.csv"
    _write_plain_csv(run_eff_path, run_eff_rows)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)
    coverage_by_algorithm = {
        str(item["algorithm"]): item
        for item in report.get("algorithm_coverage", [])
        if isinstance(item, Mapping)
    }
    curve_rows: list[dict[str, Any]] = []
    algorithm_summary_rows: list[dict[str, Any]] = []
    for algorithm in sorted(coverage_by_algorithm):
        algorithm_rows = grouped.get(algorithm, [])
        formal_ready = bool(
            coverage_by_algorithm.get(algorithm, {}).get("formal_ready", False)
        )
        diagnostic_eff = (
            100.0
            * sum(float(row["m_eff"]) for row in algorithm_rows)
            / len(algorithm_rows)
            if algorithm_rows
            else None
        )
        algorithm_summary_rows.append(
            {
                "condition": NOISE_TAG,
                "algorithm": algorithm,
                "expected_run_count": 150,
                "available_run_count": len(algorithm_rows),
                "unavailable_run_count": 150 - len(algorithm_rows),
                "coverage_rate": f"{len(algorithm_rows) / 150.0:.17g}",
                "EFF": (
                    f"{diagnostic_eff:.17g}"
                    if formal_ready and diagnostic_eff is not None
                    else ""
                ),
                "availability_status": "available" if formal_ready else "unavailable",
                "diagnostic_available_run_eff": (
                    f"{diagnostic_eff:.17g}" if diagnostic_eff is not None else ""
                ),
                "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
                "formal_ready": str(formal_ready).lower(),
            }
        )
        if not algorithm_rows:
            for minute in range(1, HORIZON + 1):
                curve_rows.append(
                    {
                        "condition": NOISE_TAG,
                        "algorithm": algorithm,
                        "minute": minute,
                        "expected_run_count": 150,
                        "available_run_count": 0,
                        "coverage_rate": "0",
                        "mean_id_quality": "",
                        "mean_ood_quality": "",
                        "mean_quality": "",
                        "mean_relative_progress": "",
                        "cumulative_eff_score": "",
                        "diagnostic_available_mean_id_quality": "",
                        "diagnostic_available_mean_ood_quality": "",
                        "diagnostic_available_mean_quality": "",
                        "diagnostic_available_mean_relative_progress": "",
                        "diagnostic_available_cumulative_eff_score": "",
                        "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
                        "formal_ready": "false",
                    }
                )
            continue
        relative = []
        for row in algorithm_rows:
            q_star = float(row["best_quality"])
            relative.append(
                [
                    float(value) / q_star if q_star > 0.0 else 0.0
                    for value in row["quality_trajectory"]
                ]
            )
        cumulative = 0.0
        for minute in range(1, HORIZON + 1):
            mean_id = sum(
                float(row["id_quality_trajectory"][minute - 1])
                for row in algorithm_rows
            ) / len(algorithm_rows)
            mean_ood = sum(
                float(row["ood_quality_trajectory"][minute - 1])
                for row in algorithm_rows
            ) / len(algorithm_rows)
            mean_quality = sum(
                float(row["quality_trajectory"][minute - 1])
                for row in algorithm_rows
            ) / len(algorithm_rows)
            mean_relative = sum(run[minute - 1] for run in relative) / len(relative)
            cumulative += mean_relative
            curve_rows.append(
                {
                    "condition": NOISE_TAG,
                    "algorithm": algorithm,
                    "minute": minute,
                    "expected_run_count": 150,
                    "available_run_count": len(algorithm_rows),
                    "coverage_rate": f"{len(algorithm_rows) / 150.0:.17g}",
                    "mean_id_quality": f"{mean_id:.17g}" if formal_ready else "",
                    "mean_ood_quality": f"{mean_ood:.17g}" if formal_ready else "",
                    "mean_quality": f"{mean_quality:.17g}" if formal_ready else "",
                    "mean_relative_progress": (
                        f"{mean_relative:.17g}" if formal_ready else ""
                    ),
                    "cumulative_eff_score": (
                        f"{100.0 * cumulative / minute:.17g}"
                        if formal_ready
                        else ""
                    ),
                    "diagnostic_available_mean_id_quality": f"{mean_id:.17g}",
                    "diagnostic_available_mean_ood_quality": f"{mean_ood:.17g}",
                    "diagnostic_available_mean_quality": f"{mean_quality:.17g}",
                    "diagnostic_available_mean_relative_progress": f"{mean_relative:.17g}",
                    "diagnostic_available_cumulative_eff_score": f"{100.0 * cumulative / minute:.17g}",
                    "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
                    "formal_ready": str(formal_ready).lower(),
                }
            )
    curve_path = output_dir / "algorithm_180min.csv"
    _write_plain_csv(curve_path, curve_rows)
    algorithm_summary_path = output_dir / "algorithm_summary.csv"
    _write_plain_csv(algorithm_summary_path, algorithm_summary_rows)

    unavailable_path = output_dir / "unavailable.jsonl"
    _write_jsonl(unavailable_path, unavailable_rows)
    allowlist_rows = []
    for item in unavailable_rows:
        parts = str(item["logical_key"]).split("::")
        algorithm = str(item.get("algorithm") or parts[0])
        allowlist_rows.append(
            {
                "logical_key": item["logical_key"],
                "condition": item.get("condition") or parts[-1],
                "algorithm": algorithm,
                "dataset_id": item.get("dataset_id") or parts[1],
                "seed": item.get("seed") or parts[2].removeprefix("s"),
                "task_id": item.get("task_id", ""),
                "host": item.get("host", ""),
                "reason": item["reason"],
                "required_action": item.get("required_action")
                or _required_evidence_action(algorithm, str(item["reason"])),
            }
        )
    allowlist_path = output_dir / "rerun_or_recollect_allowlist.csv"
    if allowlist_rows:
        _write_plain_csv(allowlist_path, allowlist_rows)
    else:
        allowlist_path.write_text(
            "logical_key,condition,algorithm,dataset_id,seed,task_id,host,reason,required_action\n",
            encoding="utf-8",
        )
    coverage_path = output_dir / "coverage_manifest.json"
    _write_json(
        coverage_path,
        {
            "schema_version": NATIVE_EFF_SCHEMA_VERSION,
            "condition": NOISE_TAG,
            "horizon": HORIZON,
            "legacy_fallback_allowed": False,
            "formal_ready_rule": "exactly_150_auditable_runs_per_algorithm",
            "algorithm_coverage": report.get("algorithm_coverage", []),
            "unavailable_path": str(unavailable_path),
            "unavailable_sha256": sha256_file(unavailable_path),
            "rerun_or_recollect_allowlist": {
                "path": str(allowlist_path),
                "sha256": sha256_file(allowlist_path),
                "rows": len(allowlist_rows),
            },
            "algorithm_summary": {
                "path": str(algorithm_summary_path),
                "sha256": sha256_file(algorithm_summary_path),
                "rows": len(algorithm_summary_rows),
            },
        },
    )
    adapter_path = output_dir / "adapter_contract.json"
    _write_json(
        adapter_path,
        {
            "schema_version": NATIVE_EFF_SCHEMA_VERSION,
            "candidate_selection_uses_id_ood": False,
            "tie_break": "keep_earliest_incumbent",
            "adapters": report.get("native_adapter_contract", {}),
        },
    )
    artifact_paths = [
        wide_path,
        run_eff_path,
        curve_path,
        algorithm_summary_path,
        unavailable_path,
        allowlist_path,
        coverage_path,
        adapter_path,
    ]
    manifest = {
        "schema_version": NATIVE_EFF_SCHEMA_VERSION,
        "condition": NOISE_TAG,
        "status": "passed" if not report.get("unresolved") else "partial",
        "legacy_fallback_used": False,
        "candidate_selection_uses_id_ood": False,
        "auditable_trajectory_run_rows": len(rows),
        "run_rows": len(run_eff_rows),
        "curve_rows": len(curve_rows),
        "algorithm_summary_rows": len(algorithm_summary_rows),
        "unavailable_run_count": len(unavailable_rows),
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    checksum_paths = [*artifact_paths, manifest_path]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return manifest


def build_eff_preparation(
    *,
    freeze_binding_report: Path,
    repair_manifest: Path,
    formula_recovery_manifest: Path | None = None,
    rerun_overlay_manifest: Path | None = None,
    repo_root: Path | None = None,
    expected_hosts: int | None = EXPECTED_HOSTS,
    expected_tasks: int | None = EXPECTED_TASKS,
    expected_points: int | None = EXPECTED_POINTS,
    expected_existing_points: int | None = EXPECTED_EXISTING_POINTS,
    expected_missing_points: int | None = EXPECTED_MISSING_POINTS,
    expected_audited_repair_points: int | None = EXPECTED_AUDITED_REPAIR_POINTS,
    expected_future_backfill_ignored_points: int | None = EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS,
    expected_checkpoint_normalization_points: int | None = EXPECTED_CHECKPOINT_NORMALIZATION_POINTS,
    expected_overlay_replacements: int = EXPECTED_OVERLAY_REPLACEMENTS,
    limit_runs: int | None = None,
    replay_performance: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = repo_root.resolve() if repo_root is not None else _repo_root()
    binding_summary, binding_report_info = load_freeze_binding_report(
        freeze_binding_report,
        repo_root=repo_root,
        expected_hosts=expected_hosts,
        expected_tasks=expected_tasks,
        expected_points=expected_points,
        expected_existing_points=expected_existing_points,
        expected_missing_points=expected_missing_points,
    )
    freeze_bundles = _verified_input_files(
        binding_summary,
        repo_root=repo_root,
        section_name="freeze_records",
    )
    freeze_reports = _verified_input_files(
        binding_summary,
        repo_root=repo_root,
        section_name="freeze_reports",
    )
    if set(freeze_bundles) != set(freeze_reports):
        _raise("freeze bundle 与 bundle report host 集合不一致")

    manifest = load_repair_manifest(
        _resolve_path(repo_root, repair_manifest),
        repo_root=repo_root,
    )
    repair_manifest_info = {
        "path": str(_resolve_path(repo_root, repair_manifest)),
        "sha256": str(manifest["manifest_sha256"]),
    }
    recovery_path = _resolve_path(
        repo_root,
        formula_recovery_manifest
        if formula_recovery_manifest is not None
        else _stage5_root() / "manifests/formula_recovery.v1.json",
    )
    try:
        recovery_manifest = load_formula_recovery_manifest(
            recovery_path, expected_condition=NOISE_TAG
        )
    except PerformanceReplayError as exc:
        raise EffPreparationContractError(str(exc)) from exc
    recovery_manifest_info = {
        "path": str(recovery_manifest.path),
        "sha256": recovery_manifest.sha256,
        "condition": recovery_manifest.condition,
        "entry_count": len(recovery_manifest.entries),
    }
    if rerun_overlay_manifest is None:
        overlay_records: dict[
            str, tuple[dict[str, Any], dict[int, dict[str, Any]]]
        ] = {}
        overlay_manifest_info = None
    else:
        overlay_records, overlay_manifest_info = load_rerun_overlay_manifest(
            rerun_overlay_manifest,
            repo_root=repo_root,
            expected_replacements=expected_overlay_replacements,
        )

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    audited_repair_points = 0
    future_backfill_ignored_points = 0
    checkpoint_normalization_points = 0
    checkpoint_normalization_details: list[dict[str, Any]] = []
    missing_points_after_repairs = 0
    original_missing_points = 0
    processed_runs = 0
    replay_cache = PerformanceReplayCache()
    replay_totals = {
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "invalid_output": 0,
        "artifact_rebuilt": 0,
    }
    internal_best_carry_points = 0
    applied_overlay_keys: set[str] = set()

    for host, bundle_info in freeze_bundles.items():
        bundle_path = Path(bundle_info["path"])
        report_info = freeze_reports[host]
        for record in _iter_bundle_records(bundle_path):
            if limit_runs is not None and processed_runs >= limit_runs:
                break
            processed_runs += 1
            source = record.get("source")
            if not isinstance(source, Mapping):
                unresolved.append({"host": host, "reason": "record 缺少 source"})
                continue
            logical_key = _logical_key(source)
            if logical_key in seen_keys:
                unresolved.append({"logical_key": logical_key, "reason": "logical_key 重复"})
                continue
            seen_keys.add(logical_key)
            if str(source.get("host")) != host:
                unresolved.append({"logical_key": logical_key, "reason": "source.host 与 bundle host 不一致"})
                continue
            if str(source.get("noise_tag")) != NOISE_TAG:
                unresolved.append({"logical_key": logical_key, "reason": "仅允许 clean 记录"})
                continue
            overlay_entry = overlay_records.get(logical_key)
            if overlay_entry is not None:
                record, strict_overlay_snapshots = overlay_entry
                source = record["source"]
                applied_overlay_keys.add(logical_key)
            raw_snapshots = record.get("snapshots")
            if not isinstance(raw_snapshots, list):
                unresolved.append({"logical_key": logical_key, "reason": "record 缺少 snapshots 数组"})
                continue
            original_missing_points += sum(
                1 for snapshot in raw_snapshots if isinstance(snapshot, Mapping) and snapshot.get("status") == "missing"
            )
            try:
                if overlay_entry is None:
                    repaired_snapshots, repair_audit = apply_repair_manifest(
                        record,
                        manifest=manifest,
                        repo_root=repo_root,
                    )
                else:
                    repaired_snapshots = strict_overlay_snapshots
                    repair_audit = {"repair_applied": False, "applied_minutes": []}
                repaired_snapshots, checkpoint_normalizations = _normalize_checkpoint_drift(
                    repaired_snapshots,
                    raw_snapshots=raw_snapshots,
                    logical_key=logical_key,
                )
                algorithm = str(source.get("algorithm"))
                if replay_performance:
                    frozen_result = record.get("result")
                    frozen_result_sha256 = (
                        str(frozen_result.get("sha256"))
                        if isinstance(frozen_result, Mapping)
                        and isinstance(frozen_result.get("sha256"), str)
                        else None
                    )
                    repaired_snapshots, replay_counts = _replay_trajectory_payloads(
                        repaired_snapshots,
                        algorithm=algorithm,
                        repo_root=repo_root,
                        cache=replay_cache,
                        recovery_manifest=recovery_manifest,
                        task_id=str(source.get("task_id")),
                        condition=NOISE_TAG,
                        result_sha256=frozen_result_sha256,
                    )
                else:
                    replay_counts = {
                        "attempted": 0,
                        "succeeded": 0,
                        "failed": 0,
                        "invalid_output": 0,
                        "artifact_rebuilt": 0,
                    }
                for field, value in replay_counts.items():
                    replay_totals[field] += value
                native_trajectory = reconstruct_native_incumbent_trajectory(
                    repaired_snapshots,
                    horizon=HORIZON,
                    algorithm=algorithm,
                )
                trajectory = native_trajectory.points
                trajectory_evidence = _trajectory_evidence(
                    trajectory, logical_key=logical_key
                )
                quality_trajectory = trajectory_evidence["quality_trajectory"]
                id_quality_trajectory = trajectory_evidence[
                    "id_quality_trajectory"
                ]
                ood_quality_trajectory = trajectory_evidence[
                    "ood_quality_trajectory"
                ]
                selected_expression_trajectory = trajectory_evidence[
                    "selected_expression_trajectory"
                ]
                valid_output_trajectory = trajectory_evidence[
                    "valid_output_trajectory"
                ]
                trajectory_sources = trajectory_evidence["trajectory_sources"]
                objective_field_trajectory = list(native_trajectory.objective_fields)
                objective_value_trajectory = list(native_trajectory.objective_values)
                incumbent_source_minute_trajectory = list(
                    native_trajectory.incumbent_source_minutes
                )
                source_path_trajectory, source_sha256_trajectory = (
                    _minute_source_evidence(
                        raw_snapshots,
                        incumbent_source_minutes=incumbent_source_minute_trajectory,
                        repair_audit=repair_audit,
                        logical_key=logical_key,
                    )
                )
                m_eff = float(native_trajectory.m_eff)
            except (
                EffPreparationContractError,
                MetricContractError,
                TrajectoryContractError,
                TrajectoryRepairContractError,
            ) as exc:
                reason = str(exc)
                unresolved.append(
                    {
                        "logical_key": logical_key,
                        "condition": str(source.get("noise_tag")),
                        "algorithm": str(source.get("algorithm")),
                        "dataset_id": str(source.get("dataset_id")),
                        "seed": int(source["seed"]),
                        "task_id": str(source.get("task_id")),
                        "host": str(source.get("host") or host),
                        "bundle_path": (
                            str(overlay_manifest_info["bundle_path"])
                            if overlay_entry is not None
                            else str(bundle_path)
                        ),
                        "bundle_sha256": (
                            overlay_manifest_info["bundle_sha256"]
                            if overlay_entry is not None
                            else bundle_info["sha256"]
                        ),
                        "freeze_binding_report_path": binding_report_info["path"],
                        "freeze_binding_report_sha256": binding_report_info["sha256"],
                        "reason": reason,
                        "required_action": _required_evidence_action(
                            algorithm, reason
                        ),
                    }
                )
                continue

            if not (0.0 <= m_eff <= 1.0):
                unresolved.append({"logical_key": logical_key, "reason": f"m_eff 越界: {m_eff}"})
                continue

            missing_after = HORIZON - len(trajectory)
            if missing_after != 0:
                missing_points_after_repairs += missing_after

            audited_count = len(list(repair_audit.get("applied_minutes", [])))
            future_ignored_count = _count_prefixed_sources(trajectory_sources, "future_backfill_ignored:")
            internal_best_count = _count_prefixed_sources(
                trajectory_sources,
                "native_carry_forward:",
            )
            audited_repair_points += audited_count
            future_backfill_ignored_points += future_ignored_count
            checkpoint_normalization_points += len(checkpoint_normalizations)
            internal_best_carry_points += internal_best_count
            checkpoint_normalization_details.extend(
                {"logical_key": logical_key, **item}
                for item in checkpoint_normalizations
            )

            row = {
                "logical_key": logical_key,
                "algorithm": str(source.get("algorithm")),
                "dataset_id": str(source.get("dataset_id")),
                "seed": int(source["seed"]),
                "task_id": str(source.get("task_id")),
                "host": str(source.get("host")) if overlay_entry is not None else host,
                "noise_tag": str(source.get("noise_tag")),
                "bundle_path": (
                    str(overlay_manifest_info["bundle_path"])
                    if overlay_entry is not None
                    else str(bundle_path)
                ),
                "bundle_sha256": (
                    overlay_manifest_info["bundle_sha256"]
                    if overlay_entry is not None
                    else bundle_info["sha256"]
                ),
                "bundle_report_path": (
                    str(overlay_manifest_info["path"])
                    if overlay_entry is not None
                    else str(report_info["path"])
                ),
                "bundle_report_sha256": (
                    overlay_manifest_info["sha256"]
                    if overlay_entry is not None
                    else report_info["sha256"]
                ),
                "freeze_binding_report_path": binding_report_info["path"],
                "freeze_binding_report_sha256": binding_report_info["sha256"],
                "repair_manifest_path": repair_manifest_info["path"],
                "repair_manifest_sha256": repair_manifest_info["sha256"],
                "quality_trajectory": quality_trajectory,
                "id_quality_trajectory": id_quality_trajectory,
                "ood_quality_trajectory": ood_quality_trajectory,
                "selected_expression_trajectory": selected_expression_trajectory,
                "valid_output_trajectory": valid_output_trajectory,
                "trajectory_sources": trajectory_sources,
                "objective_field_trajectory": objective_field_trajectory,
                "objective_value_trajectory": objective_value_trajectory,
                "incumbent_source_minute_trajectory": (
                    incumbent_source_minute_trajectory
                ),
                "source_path_trajectory": source_path_trajectory,
                "source_sha256_trajectory": source_sha256_trajectory,
                "native_adapter": asdict(
                    NATIVE_OBJECTIVE_ADAPTERS[
                        algorithm.strip().lower().replace("-", "")
                    ]
                ),
                "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
                "trajectory_evidence_schema_version": (
                    TRAJECTORY_EVIDENCE_SCHEMA_VERSION
                ),
                "best_quality": native_trajectory.q_star,
                "m_eff": m_eff,
                "audited_repair_points": audited_count,
                "future_backfill_ignored_points": future_ignored_count,
                "checkpoint_normalizations": checkpoint_normalizations,
                "repair_applied": bool(repair_audit.get("repair_applied")),
                "repair_minutes": list(repair_audit.get("applied_minutes", [])),
                "evaluation_path": EVALUATION_PATH if replay_performance else "frozen_metrics.v1",
                "canonical_replay_counts": replay_counts,
                "internal_best_carry_points": internal_best_count,
            }
            rows.append(row)
        if limit_runs is not None and processed_runs >= limit_runs:
            break

    rows.sort(key=_stable_row_sort_key)

    if limit_runs is None and set(overlay_records) != applied_overlay_keys:
        missing_overlay_keys = sorted(set(overlay_records) - applied_overlay_keys)
        _raise(
            "rerun overlay replacement keys 不属于完整 freeze 网格: "
            f"{missing_overlay_keys[:10]}"
        )

    full_contract_checked = limit_runs is None
    if limit_runs is None and expected_tasks is not None and len(rows) != expected_tasks:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": f"成功 run 数应为 {expected_tasks}，实际为 {len(rows)}",
            }
        )
    if limit_runs is None and expected_audited_repair_points is not None and audited_repair_points != expected_audited_repair_points:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    f"audited_repair_points 应为 {expected_audited_repair_points}，"
                    f"实际为 {audited_repair_points}"
                ),
            }
        )
    if (
        limit_runs is None
        and expected_future_backfill_ignored_points is not None
        and future_backfill_ignored_points != expected_future_backfill_ignored_points
    ):
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    f"future_backfill_ignored_points 应为 {expected_future_backfill_ignored_points}，"
                    f"实际为 {future_backfill_ignored_points}"
                ),
            }
        )
    if limit_runs is None and missing_points_after_repairs != 0:
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": f"修复后仍有缺失点: {missing_points_after_repairs}",
            }
        )
    if (
        limit_runs is None
        and expected_checkpoint_normalization_points is not None
        and checkpoint_normalization_points != expected_checkpoint_normalization_points
    ):
        unresolved.append(
            {
                "logical_key": "__global__",
                "reason": (
                    "checkpoint_normalization_points 应为 "
                    f"{expected_checkpoint_normalization_points}，实际为 "
                    f"{checkpoint_normalization_points}"
                ),
            }
        )

    formal_eff_ready = (
        full_contract_checked
        and expected_tasks is not None
        and len(rows) == expected_tasks
        and missing_points_after_repairs == 0
        and not unresolved
    )
    successful_by_algorithm = Counter(str(row["algorithm"]) for row in rows)
    unresolved_by_algorithm: Counter[str] = Counter()
    for item in unresolved:
        logical_key = str(item.get("logical_key") or "")
        if "::" in logical_key:
            unresolved_by_algorithm[logical_key.split("::", 1)[0]] += 1
    algorithm_coverage = []
    for adapter_key, adapter in sorted(NATIVE_OBJECTIVE_ADAPTERS.items()):
        display_name = next(
            (
                str(row["algorithm"])
                for row in rows
                if str(row["algorithm"]).strip().lower().replace("-", "")
                == adapter_key
            ),
            adapter.algorithm,
        )
        available = int(successful_by_algorithm.get(display_name, 0))
        unavailable = int(unresolved_by_algorithm.get(display_name, 0))
        algorithm_coverage.append(
            {
                "algorithm": display_name,
                "expected_run_count": 150,
                "available_run_count": available,
                "unavailable_run_count": unavailable,
                "coverage_rate": available / 150.0,
                "formal_ready": available == 150 and unavailable == 0,
                "adapter": asdict(adapter),
            }
        )
    report = {
        "condition": NOISE_TAG,
        "horizon": HORIZON,
        "inputs": {
            "freeze_binding_report": binding_report_info,
            "freeze_records": list(freeze_bundles.values()),
            "freeze_reports": list(freeze_reports.values()),
            "repair_manifest": repair_manifest_info,
            "formula_recovery_manifest": recovery_manifest_info,
            "rerun_overlay_manifest": overlay_manifest_info,
        },
        "summary": {
            "processed_run_count": processed_runs,
            "success_count": len(rows),
            "unresolved_run_count": len(unresolved),
            "full_contract_checked": full_contract_checked,
            "limit_runs": limit_runs,
            "original_missing_points": original_missing_points,
            "missing_points_after_repairs": missing_points_after_repairs,
            "audited_repair_points": audited_repair_points,
            "future_backfill_ignored_points": future_backfill_ignored_points,
            "checkpoint_normalization_points": checkpoint_normalization_points,
            "canonical_replay_attempted_points": replay_totals["attempted"],
            "canonical_replay_succeeded_points": replay_totals["succeeded"],
            "canonical_replay_failed_points": replay_totals["failed"],
            "canonical_replay_invalid_output_points": replay_totals["invalid_output"],
            "canonical_artifact_rebuilt_points": replay_totals["artifact_rebuilt"],
            "internal_best_carry_points": internal_best_carry_points,
            "overlay_replacement_count": len(applied_overlay_keys),
            "evaluation_path": EVALUATION_PATH if replay_performance else "frozen_metrics.v1",
            "trajectory_evidence_schema_version": (
                TRAJECTORY_EVIDENCE_SCHEMA_VERSION
            ),
            "native_eff_schema_version": NATIVE_EFF_SCHEMA_VERSION,
            "formal_eff_ready": formal_eff_ready,
            "m_eff_min": min((row["m_eff"] for row in rows), default=0.0),
            "m_eff_max": max((row["m_eff"] for row in rows), default=0.0),
        },
        "checkpoint_normalization_details": checkpoint_normalization_details,
        "algorithm_coverage": algorithm_coverage,
        "native_adapter_contract": {
            key: asdict(value) for key, value in sorted(NATIVE_OBJECTIVE_ADAPTERS.items())
        },
        "unresolved": unresolved,
    }
    return rows, report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    stage5_root = _stage5_root()
    parser = argparse.ArgumentParser(description="从冻结 clean 轨迹准备正式 EFF 运行级输入")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="仓库根目录，用于解析相对路径",
    )
    parser.add_argument(
        "--freeze-binding-report",
        type=Path,
        default=stage5_root / "reports/freeze_binding.json",
    )
    parser.add_argument(
        "--repair-manifest",
        type=Path,
        default=stage5_root / "manifests/trajectory_repairs.v1.json",
    )
    parser.add_argument(
        "--formula-recovery-manifest",
        type=Path,
        default=stage5_root / "manifests/formula_recovery.v1.json",
    )
    parser.add_argument(
        "--rerun-overlay-manifest",
        type=Path,
        default=None,
        help="可选 clean rerun overlay manifest；提供后严格替换其中 225 条 EFF 轨迹",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=stage5_root / "reports/eff_preparation_runs.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=stage5_root / "reports/eff_preparation_runs.csv",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=stage5_root / "reports/eff_preparation.json",
    )
    parser.add_argument(
        "--revision-output-dir",
        type=Path,
        default=None,
        help="可选：额外写出严格原生 EFF 的正式审计交付件",
    )
    parser.add_argument("--expected-hosts", type=int, default=EXPECTED_HOSTS)
    parser.add_argument("--expected-tasks", type=int, default=EXPECTED_TASKS)
    parser.add_argument("--expected-points", type=int, default=EXPECTED_POINTS)
    parser.add_argument("--expected-existing-points", type=int, default=EXPECTED_EXISTING_POINTS)
    parser.add_argument("--expected-missing-points", type=int, default=EXPECTED_MISSING_POINTS)
    parser.add_argument(
        "--expected-audited-repair-points",
        type=int,
        default=EXPECTED_AUDITED_REPAIR_POINTS,
    )
    parser.add_argument(
        "--expected-future-backfill-ignored-points",
        type=int,
        default=EXPECTED_FUTURE_BACKFILL_IGNORED_POINTS,
    )
    parser.add_argument(
        "--expected-checkpoint-normalization-points",
        type=int,
        default=EXPECTED_CHECKPOINT_NORMALIZATION_POINTS,
    )
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument(
        "--skip-canonical-replay",
        action="store_true",
        help="仅供旧冻结夹具审计使用；正式聚合不得跳过 canonical replay",
    )
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, report = build_eff_preparation(
            freeze_binding_report=args.freeze_binding_report,
            repair_manifest=args.repair_manifest,
            formula_recovery_manifest=args.formula_recovery_manifest,
            rerun_overlay_manifest=args.rerun_overlay_manifest,
            repo_root=args.repo_root,
            expected_hosts=args.expected_hosts,
            expected_tasks=args.expected_tasks,
            expected_points=args.expected_points,
            expected_existing_points=args.expected_existing_points,
            expected_missing_points=args.expected_missing_points,
            expected_audited_repair_points=args.expected_audited_repair_points,
            expected_future_backfill_ignored_points=args.expected_future_backfill_ignored_points,
            expected_checkpoint_normalization_points=args.expected_checkpoint_normalization_points,
            limit_runs=args.limit_runs,
            replay_performance=not args.skip_canonical_replay,
        )
        output_jsonl = args.output_jsonl.resolve()
        output_csv = args.output_csv.resolve()
        output_report = args.output_report.resolve()
        _write_jsonl(output_jsonl, rows)
        _write_csv(output_csv, rows)
        report = {
            **report,
            "status": "ok",
            "contract_ok": len(report["unresolved"]) == 0,
            "outputs": {
                "eff_jsonl": str(output_jsonl),
                "eff_jsonl_sha256": sha256_file(output_jsonl),
                "eff_jsonl_row_count": len(rows),
                "eff_csv": str(output_csv),
                "eff_csv_sha256": sha256_file(output_csv),
                "eff_csv_row_count": len(rows),
            },
        }
        if args.revision_output_dir is not None:
            revision_manifest = write_eff_revision_artifacts(
                args.revision_output_dir.resolve(), rows=rows, report=report
            )
            report["outputs"]["eff_revision_manifest"] = revision_manifest
        _write_json(output_report, report)
    except (
        EffPreparationContractError,
        FreezeBindingContractError,
        MetricContractError,
        TrajectoryContractError,
        TrajectoryRepairContractError,
    ) as exc:
        report = {
            "condition": NOISE_TAG,
            "horizon": HORIZON,
            "fatal_error": str(exc),
            "status": "error",
            "contract_ok": False,
            "summary": {
                "processed_run_count": 0,
                "success_count": 0,
                "unresolved_run_count": 1,
                "full_contract_checked": args.limit_runs is None,
                "limit_runs": args.limit_runs,
                "formal_eff_ready": False,
            },
            "unresolved": [{"logical_key": "__fatal__", "reason": str(exc)}],
        }
        _write_json(args.output_report.resolve(), report)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.print_summary:
            print(rendered)
        else:
            print(rendered)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.print_summary:
        print(rendered)
    else:
        print(rendered)
    if report["summary"]["unresolved_run_count"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
