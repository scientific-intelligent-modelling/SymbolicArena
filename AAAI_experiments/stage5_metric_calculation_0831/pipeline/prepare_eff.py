"""为正式六轴评测准备 clean 条件下的运行级 EFF 输入。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .freeze_binding import FreezeBindingContractError, validate_freeze_binding_summary
from .metrics import MetricContractError, efficiency_from_qualities
from .performance_replay import (
    EVALUATION_PATH,
    FormulaRecoveryManifest,
    PerformanceReplayCache,
    PerformanceReplayError,
    load_formula_recovery_manifest,
    replay_payload_performance,
)
from .trajectories import TrajectoryContractError, canonical_expression, reconstruct_trajectory
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


class EffPreparationContractError(ValueError):
    """冻结 bundle、修复清单或 EFF 准备过程不满足正式契约。"""


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
    }
    for minute, value in enumerate(record["quality_trajectory"], start=1):
        row[f"q_{minute:04d}"] = f"{float(value):.17g}"
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
                if replay_counts["failed"]:
                    raise EffPreparationContractError(
                        f"{logical_key} 有 {replay_counts['failed']} 个 canonical replay 证据错误"
                    )
                trajectory = reconstruct_trajectory(
                    repaired_snapshots,
                    horizon=HORIZON,
                    algorithm=algorithm,
                )
                quality_trajectory = [float(point.quality) for point in trajectory]
                trajectory_sources = [str(point.source) for point in trajectory]
                m_eff = float(efficiency_from_qualities(quality_trajectory, horizon=HORIZON))
            except (
                EffPreparationContractError,
                MetricContractError,
                TrajectoryContractError,
                TrajectoryRepairContractError,
            ) as exc:
                unresolved.append({"logical_key": logical_key, "reason": str(exc)})
                continue

            if not (0.0 <= m_eff <= 1.0):
                unresolved.append({"logical_key": logical_key, "reason": f"m_eff 越界: {m_eff}"})
                continue

            missing_after = HORIZON - len(trajectory)
            if missing_after != 0:
                missing_points_after_repairs += missing_after

            audited_count = _count_prefixed_sources(trajectory_sources, "audited_repair:")
            future_ignored_count = _count_prefixed_sources(trajectory_sources, "future_backfill_ignored:")
            internal_best_count = _count_prefixed_sources(
                trajectory_sources,
                "internal_best_carry_forward:",
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
                "trajectory_sources": trajectory_sources,
                "best_quality": max(quality_trajectory, default=0.0),
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
            "formal_eff_ready": formal_eff_ready,
            "m_eff_min": min((row["m_eff"] for row in rows), default=0.0),
            "m_eff_max": max((row["m_eff"] for row in rows), default=0.0),
        },
        "checkpoint_normalization_details": checkpoint_normalization_details,
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
