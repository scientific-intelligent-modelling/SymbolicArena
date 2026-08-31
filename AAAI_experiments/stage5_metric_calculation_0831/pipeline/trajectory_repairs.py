"""对已审计的 clean 轨迹缺口应用显式、无未来信息的修复。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

class TrajectoryRepairContractError(ValueError):
    """轨迹修复清单或冻结输入不满足审计契约。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _resolve(repo_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (repo_root / path).resolve()


def _logical_key(source: Mapping[str, Any]) -> str:
    try:
        seed = int(source["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrajectoryRepairContractError("source.seed 无效") from exc
    return (
        f"{source.get('algorithm')}::{source.get('dataset_id')}::"
        f"s{seed}::{source.get('noise_tag')}"
    )


def load_repair_manifest(path: Path, *, repo_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrajectoryRepairContractError(f"修复清单不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "trajectory_repairs.v1":
        raise TrajectoryRepairContractError("修复清单 schema_version 无效")
    if payload.get("condition") != "clean" or payload.get("horizon") != 180:
        raise TrajectoryRepairContractError("修复清单必须固定为 clean / 180 分钟")
    repairs = payload.get("repairs")
    if not isinstance(repairs, list):
        raise TrajectoryRepairContractError("修复清单 repairs 必须是数组")

    seen: set[str] = set()
    for repair in repairs:
        if not isinstance(repair, dict):
            raise TrajectoryRepairContractError("repair 必须是 JSON object")
        logical_key = str(repair.get("logical_key") or "")
        if not logical_key or logical_key in seen:
            raise TrajectoryRepairContractError(f"repair logical_key 缺失或重复: {logical_key!r}")
        seen.add(logical_key)
        if repair.get("rule") != "carry_forward_last_observed_best":
            raise TrajectoryRepairContractError(f"{logical_key} 使用了未知修复规则")
        minutes = repair.get("missing_minutes")
        if (
            not isinstance(minutes, list)
            or not minutes
            or any(isinstance(item, bool) or not isinstance(item, int) for item in minutes)
            or minutes != sorted(set(minutes))
            or any(item < 1 or item > 180 for item in minutes)
        ):
            raise TrajectoryRepairContractError(f"{logical_key} missing_minutes 无效")
        source_minute = repair.get("source_minute")
        if isinstance(source_minute, bool) or not isinstance(source_minute, int):
            raise TrajectoryRepairContractError(f"{logical_key} source_minute 无效")
        if source_minute >= min(minutes):
            raise TrajectoryRepairContractError(f"{logical_key} source_minute 不是严格过去信息")

        evidence = repair.get("supporting_evidence")
        if not isinstance(evidence, dict):
            raise TrajectoryRepairContractError(f"{logical_key} 缺少 supporting_evidence")
        for path_field, sha_field in (
            ("current_best_path", "current_best_sha256"),
            ("wrapper_path", "wrapper_sha256"),
        ):
            evidence_path = _resolve(repo_root, evidence.get(path_field))
            if not evidence_path.is_file():
                raise TrajectoryRepairContractError(f"{logical_key} 证据文件不存在: {evidence_path}")
            actual_sha = _sha256_file(evidence_path)
            if actual_sha != evidence.get(sha_field):
                raise TrajectoryRepairContractError(
                    f"{logical_key} {path_field} SHA 不匹配: {actual_sha}"
                )
    payload["manifest_path"] = str(path.resolve())
    payload["manifest_sha256"] = _sha256_bytes(raw)
    return payload


def _parse_frozen_snapshot(
    snapshot: Mapping[str, Any],
    *,
    logical_key: str,
    minute: int,
) -> dict[str, Any]:
    raw_text = snapshot.get("raw_text")
    selected_sha = snapshot.get("selected_sha256")
    if not isinstance(raw_text, str) or not isinstance(selected_sha, str):
        raise TrajectoryRepairContractError(
            f"{logical_key} minute_{minute:04d} 缺少冻结 raw_text/SHA"
        )
    actual_sha = _sha256_bytes(raw_text.encode("utf-8"))
    if actual_sha != selected_sha:
        raise TrajectoryRepairContractError(
            f"{logical_key} minute_{minute:04d} 冻结 raw SHA 不匹配"
        )
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TrajectoryRepairContractError(
            f"{logical_key} minute_{minute:04d} raw_text 不是合法 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise TrajectoryRepairContractError(
            f"{logical_key} minute_{minute:04d} raw_text 顶层不是 object"
        )
    return payload


def apply_repair_manifest(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    source = record.get("source")
    snapshots = record.get("snapshots")
    if not isinstance(source, Mapping) or not isinstance(snapshots, list):
        raise TrajectoryRepairContractError("冻结记录缺少 source 或 snapshots")
    horizon = int(manifest.get("horizon", 0))
    if len(snapshots) != horizon:
        raise TrajectoryRepairContractError(f"冻结记录 snapshots 应为 {horizon} 个")
    logical_key = _logical_key(source)
    repairs = {
        str(item["logical_key"]): item
        for item in manifest.get("repairs", [])
        if isinstance(item, Mapping) and "logical_key" in item
    }
    repair = repairs.get(logical_key)
    actual_missing = [
        minute
        for minute, snapshot in enumerate(snapshots, start=1)
        if isinstance(snapshot, Mapping) and snapshot.get("status") == "missing"
    ]
    if any(not isinstance(snapshot, Mapping) for snapshot in snapshots):
        raise TrajectoryRepairContractError(f"{logical_key} 含非 object snapshot")
    if repair is None and actual_missing:
        raise TrajectoryRepairContractError(f"{logical_key} 存在未列入修复清单的缺失分钟")
    if repair is not None and actual_missing != list(repair["missing_minutes"]):
        raise TrajectoryRepairContractError(
            f"{logical_key} 缺失分钟集合与修复清单不一致: {actual_missing}"
        )

    parsed: dict[int, dict[str, Any]] = {}
    for minute, snapshot in enumerate(snapshots, start=1):
        if snapshot.get("status") == "ok":
            parsed[minute] = _parse_frozen_snapshot(
                snapshot,
                logical_key=logical_key,
                minute=minute,
            )

    if repair is None:
        return parsed, {
            "logical_key": logical_key,
            "repair_applied": False,
            "applied_minutes": [],
        }

    if str(source.get("task_id")) != repair.get("task_id") or str(source.get("host")) != repair.get("host"):
        raise TrajectoryRepairContractError(f"{logical_key} task/host 与修复清单不一致")
    source_minute = int(repair["source_minute"])
    source_snapshot = snapshots[source_minute - 1]
    if source_snapshot.get("status") != "ok":
        raise TrajectoryRepairContractError(f"{logical_key} source minute 不是可用快照")
    if source_snapshot.get("selected_sha256") != repair.get("source_snapshot_sha256"):
        raise TrajectoryRepairContractError(f"{logical_key} source snapshot SHA 不匹配")
    source_payload = parsed[source_minute]
    if source_payload.get("record_type") != repair.get("source_record_type"):
        raise TrajectoryRepairContractError(f"{logical_key} source record_type 不匹配")
    try:
        elapsed_seconds = float(source_payload.get("elapsed_seconds"))
        expected_elapsed = float(repair.get("source_elapsed_seconds"))
    except (TypeError, ValueError) as exc:
        raise TrajectoryRepairContractError(f"{logical_key} source elapsed_seconds 无效") from exc
    if not math.isclose(elapsed_seconds, expected_elapsed, rel_tol=0.0, abs_tol=1e-9):
        raise TrajectoryRepairContractError(f"{logical_key} source elapsed_seconds 不匹配")

    future_minute = int(repair["excluded_future_minute"])
    future_snapshot = snapshots[future_minute - 1]
    if future_snapshot.get("selected_sha256") != repair.get("excluded_future_snapshot_sha256"):
        raise TrajectoryRepairContractError(f"{logical_key} excluded future SHA 不匹配")

    evidence = repair["supporting_evidence"]
    current_best_path = _resolve(repo_root, evidence["current_best_path"])
    current_best = json.loads(current_best_path.read_text(encoding="utf-8"))
    if not isinstance(current_best, dict):
        raise TrajectoryRepairContractError(f"{logical_key} current-best 证据无效")
    source_raw_equation = str(source_payload.get("equation") or "").strip()
    if str(current_best.get("equation") or "").strip() != source_raw_equation:
        raise TrajectoryRepairContractError(f"{logical_key} current-best 与 source 公式不一致")

    for minute in repair["missing_minutes"]:
        recovered = copy.deepcopy(source_payload)
        recovered["record_type"] = "audited_carry_forward"
        recovered["source_record_type"] = source_payload.get("record_type")
        recovered["checkpoint_index"] = minute
        recovered["elapsed_seconds"] = minute * 60
        recovered["elapsed_minutes"] = minute
        recovered["recovery_provenance"] = {
            "rule": repair["rule"],
            "source_minute": source_minute,
            "source_snapshot_sha256": repair["source_snapshot_sha256"],
            "repair_manifest_sha256": manifest.get("manifest_sha256"),
            "excluded_future_minute": future_minute,
        }
        parsed[int(minute)] = recovered

    return parsed, {
        "logical_key": logical_key,
        "repair_applied": True,
        "rule": repair["rule"],
        "source_minute": source_minute,
        "source_snapshot_sha256": repair["source_snapshot_sha256"],
        "applied_minutes": list(repair["missing_minutes"]),
        "excluded_future_minute": future_minute,
    }
