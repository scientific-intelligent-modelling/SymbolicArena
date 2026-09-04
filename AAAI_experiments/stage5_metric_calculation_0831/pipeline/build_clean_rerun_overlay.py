#!/usr/bin/env python3
"""构建仅用于 EFF 的 clean 重跑轨迹覆盖层。

该工具不会改写正式 clean final。它把 JAXSR/iMCTS 的远端冻结记录与
SymbolFit 的本地 outer 结果统一成 ``remote_snapshot`` 兼容记录，同时生成
一个仅供诊断的 clean source-runs 复合表。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "clean_rerun_eff_overlay_v1"
SCOPE = "mixed_clean_rerun_overlay"
FINAL_AND_EFF = "final_and_eff"
EFF_ONLY = "eff_only"
DEFAULT_EXPECTED_COUNTS = {"jaxsr": 35, "imcts": 40, "symbolfit": 150}
DEFAULT_STRICT_TASK_ID = "symbolfit_s521_clean_g0039"
_TASK_ID = re.compile(
    r"(?P<algorithm>[A-Za-z0-9_-]+)_s(?P<seed>\d+)_clean_g(?P<index>\d{4})$"
)


class CleanRerunOverlayError(ValueError):
    """覆盖层输入证据不完整或不一致。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_gzip_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gzip_handle:
                with io.TextIOWrapper(
                    gzip_handle, encoding="utf-8", newline="\n"
                ) as text_handle:
                    for row in rows:
                        text_handle.write(_canonical_json(row))
                        text_handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normal_algorithm(value: object) -> str:
    return str(value or "").strip().lower()


def _logical_key(algorithm: object, dataset: object, seed: object) -> str:
    try:
        parsed_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise CleanRerunOverlayError(f"seed 非法: {seed!r}") from exc
    return f"{algorithm}::{dataset}::s{parsed_seed}::clean"


def _finite_number(value: object, *, field: str, nonnegative: bool = False) -> float:
    if value is None or isinstance(value, bool):
        raise CleanRerunOverlayError(f"{field} 不是有限数值: {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CleanRerunOverlayError(f"{field} 不是有限数值: {value!r}") from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise CleanRerunOverlayError(f"{field} 不是有效有限数值: {value!r}")
    return parsed


def _read_json_bytes(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanRerunOverlayError(f"{context} 不是合法 UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CleanRerunOverlayError(f"{context} 的 JSON 顶层不是 object")
    return payload


def _read_json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise CleanRerunOverlayError(f"输入文件不存在: {path}")
    raw = path.read_bytes()
    return raw, _read_json_bytes(raw, context=str(path))


def _parse_checksum_manifest(path: Path) -> dict[str, list[tuple[str, str]]]:
    if not path.is_file():
        raise CleanRerunOverlayError(f"校验和清单不存在: {path}")
    entries: dict[str, list[tuple[str, str]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise CleanRerunOverlayError(f"{path}:{line_number} 校验和格式非法")
        declared_path = Path(parts[1].lstrip("*")).as_posix()
        entries.setdefault(Path(declared_path).name, []).append(
            (declared_path, parts[0].lower())
        )
    if not entries:
        raise CleanRerunOverlayError(f"校验和清单为空: {path}")
    return entries


def _verify_manifest_file(
    path: Path,
    *,
    data_root: Path,
    checksum_entries: Mapping[str, Sequence[tuple[str, str]]],
    resolved_path: Path | None = None,
    resolved_data_root: Path | None = None,
) -> str:
    resolved_path = path.resolve() if resolved_path is None else resolved_path
    resolved_root = (
        data_root.resolve() if resolved_data_root is None else resolved_data_root
    )
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise CleanRerunOverlayError(f"输入不在声明的数据根目录内: {path}") from exc
    absolute = resolved_path.as_posix()
    matches = {
        digest
        for declared_path, digest in checksum_entries.get(path.name, ())
        if declared_path == relative
        or declared_path.endswith(f"/{relative}")
        or declared_path == absolute
    }
    if not matches:
        raise CleanRerunOverlayError(f"校验和清单未覆盖输入文件: {relative}")
    if len(matches) != 1:
        raise CleanRerunOverlayError(f"校验和清单对输入文件给出冲突哈希: {relative}")
    actual = _sha256_file(path)
    expected = next(iter(matches))
    if actual != expected:
        raise CleanRerunOverlayError(
            f"输入文件 SHA256 不匹配: {relative}: {actual} != {expected}"
        )
    return actual


def _load_clean_source_rows(
    path: Path, *, expected_clean_rows: int | None
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    if not path.is_file():
        raise CleanRerunOverlayError(f"source_runs.csv 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader if row.get("noise_tag") == "clean"]
    required = {
        "batch",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "status",
        "seconds",
        "id_nmse",
        "ood_nmse",
        "id_r2",
        "ood_r2",
        "id_acc",
        "ood_acc",
        "path",
        "logical_key",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise CleanRerunOverlayError(f"source_runs.csv 缺少字段: {missing}")
    if expected_clean_rows is not None and len(rows) != expected_clean_rows:
        raise CleanRerunOverlayError(
            f"clean source 行数应为 {expected_clean_rows}，实际为 {len(rows)}"
        )
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        expected_key = _logical_key(row["algorithm"], row["dataset_id"], row["seed"])
        if row["logical_key"] != expected_key:
            raise CleanRerunOverlayError(
                f"source logical_key 不符合 canonical 格式: {row['logical_key']!r}"
            )
        if expected_key in by_key:
            raise CleanRerunOverlayError(f"source 出现重复 logical_key: {expected_key}")
        by_key[expected_key] = row
    return fieldnames, rows, by_key


def _metric(payload: Mapping[str, Any], split: str, field: str) -> float:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        raise CleanRerunOverlayError(f"result.{split} 缺失")
    return _finite_number(
        block.get(field), field=f"result.{split}.{field}", nonnegative=field == "nmse"
    )


def _validate_result_payload(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    dataset: str,
    seed: int,
    context: str,
) -> None:
    if payload.get("status") != "ok":
        raise CleanRerunOverlayError(f"{context} result status 不是 ok")
    if _normal_algorithm(payload.get("tool")) != _normal_algorithm(algorithm):
        raise CleanRerunOverlayError(f"{context} result tool 与逻辑键不一致")
    if str(payload.get("dataset")) != dataset or int(payload.get("seed", -1)) != seed:
        raise CleanRerunOverlayError(f"{context} result dataset/seed 与逻辑键不一致")
    if not isinstance(payload.get("equation"), str) or not payload["equation"].strip():
        raise CleanRerunOverlayError(f"{context} result 最终公式为空")
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("artifact_valid") is not True:
        raise CleanRerunOverlayError(f"{context} result canonical_artifact 无效")
    for split in ("id_test", "ood_test"):
        for field in ("nmse", "r2", "acc_0_1"):
            _metric(payload, split, field)


def _validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    dataset: str,
    seed: int,
    minute: int,
    horizon: int,
    context: str,
) -> None:
    if _normal_algorithm(payload.get("tool")) != _normal_algorithm(algorithm):
        raise CleanRerunOverlayError(f"{context} checkpoint tool 身份不一致")
    if str(payload.get("dataset")) != dataset or int(payload.get("seed", -1)) != seed:
        raise CleanRerunOverlayError(f"{context} checkpoint dataset/seed 身份不一致")
    if payload.get("checkpoint_index") != minute:
        raise CleanRerunOverlayError(
            f"{context} checkpoint_index={payload.get('checkpoint_index')!r}，预期 {minute}"
        )
    elapsed = _finite_number(
        payload.get("elapsed_minutes"), field=f"{context}.elapsed_minutes", nonnegative=True
    )
    if minute < horizon and elapsed > horizon + 1:
        raise CleanRerunOverlayError(f"{context} elapsed_minutes 超出正式预算")
    record_type = payload.get("record_type")
    if minute == horizon:
        if record_type != "budget_end_internal_best":
            raise CleanRerunOverlayError(f"{context} 预算终点不是 internal best")
    elif record_type not in {"periodic_best", "periodic_heartbeat", "periodic_backfill"}:
        raise CleanRerunOverlayError(f"{context} record_type 非法: {record_type!r}")
    if record_type != "periodic_heartbeat" and payload.get("status") != "ok":
        raise CleanRerunOverlayError(f"{context} 候选 checkpoint status 不是 ok")


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    expression = next(
        (
            str(value).strip()
            for value in (
                artifact.get("instantiated_expression"),
                artifact.get("normalized_expression"),
                artifact.get("return_expression_source"),
                payload.get("equation"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    return {
        "record_type": payload.get("record_type"),
        "payload_status": payload.get("status"),
        "checkpoint_index": payload.get("checkpoint_index"),
        "elapsed_minutes": payload.get("elapsed_minutes"),
        "backfilled_from_minute": payload.get("backfilled_from_minute"),
        "has_expression": bool(expression),
        "expression": expression,
        "id_nmse": (payload.get("id_test") or {}).get("nmse"),
        "ood_nmse": (payload.get("ood_test") or {}).get("nmse"),
    }


def _source_dict(
    *,
    batch: str,
    algorithm: str,
    dataset: str,
    host: str,
    seed: int,
    task_id: str,
    result_path: str,
) -> dict[str, Any]:
    source = {
        "batch": batch,
        "algorithm": algorithm,
        "host": host,
        "seed": seed,
        "noise_tag": "clean",
        "task_id": task_id,
        "dataset_id": dataset,
        "path": result_path,
    }
    source["source_row_sha256"] = _sha256_bytes(
        _canonical_json(source).encode("utf-8")
    )
    return source


def _validate_frozen_record(
    record: Mapping[str, Any], *, horizon: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise CleanRerunOverlayError("冻结记录缺少 source")
    algorithm = str(source.get("algorithm") or "")
    dataset = str(source.get("dataset_id") or "")
    seed = int(source.get("seed", -1))
    if source.get("noise_tag") != "clean":
        raise CleanRerunOverlayError("重跑冻结记录不是 clean")
    if _normal_algorithm(algorithm) not in {"jaxsr", "imcts"}:
        raise CleanRerunOverlayError(f"冻结记录算法不在允许集合: {algorithm!r}")
    result = record.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "ok":
        raise CleanRerunOverlayError("冻结记录 result 不可用")
    raw_result_text = result.get("raw_text")
    if not isinstance(raw_result_text, str):
        raise CleanRerunOverlayError("冻结记录 result 缺少 raw_text")
    raw_result = raw_result_text.encode("utf-8")
    if _sha256_bytes(raw_result) != result.get("sha256"):
        raise CleanRerunOverlayError("冻结记录 result raw SHA256 不匹配")
    result_payload = _read_json_bytes(raw_result, context="冻结 result.raw_text")
    _validate_result_payload(
        result_payload,
        algorithm=algorithm,
        dataset=dataset,
        seed=seed,
        context=_logical_key(algorithm, dataset, seed),
    )
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != horizon:
        raise CleanRerunOverlayError(f"冻结记录必须恰有 {horizon} 个快照")
    by_minute: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise CleanRerunOverlayError("冻结 snapshot 不是 object")
        minute = int(snapshot.get("minute", -1))
        if minute in by_minute:
            raise CleanRerunOverlayError(f"冻结记录分钟重复: {minute}")
        if snapshot.get("status") != "ok" or snapshot.get("conflict") is True:
            raise CleanRerunOverlayError(f"冻结记录 minute_{minute:04d} 不可用或冲突")
        raw_text = snapshot.get("raw_text")
        if not isinstance(raw_text, str):
            raise CleanRerunOverlayError(f"冻结记录 minute_{minute:04d} 缺少 raw_text")
        raw = raw_text.encode("utf-8")
        if _sha256_bytes(raw) != snapshot.get("selected_sha256"):
            raise CleanRerunOverlayError(f"冻结记录 minute_{minute:04d} raw SHA256 不匹配")
        payload = _read_json_bytes(raw, context=f"冻结 minute_{minute:04d}")
        _validate_checkpoint_payload(
            payload,
            algorithm=algorithm,
            dataset=dataset,
            seed=seed,
            minute=minute,
            horizon=horizon,
            context=_logical_key(algorithm, dataset, seed),
        )
        by_minute[minute] = snapshot
    if set(by_minute) != set(range(1, horizon + 1)):
        missing = sorted(set(range(1, horizon + 1)).difference(by_minute))
        raise CleanRerunOverlayError(f"冻结记录分钟网格不完整: {missing}")
    copied = dict(record)
    copied["snapshots"] = [by_minute[minute] for minute in range(1, horizon + 1)]
    return copied, result_payload


def _load_jax_imcts_records(
    freeze_dir: Path,
    *,
    horizon: int,
    input_files: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    checksum_path = freeze_dir.parent / "SHA256SUMS"
    checksum_entries = _parse_checksum_manifest(checksum_path)
    input_files.append(
        {
            "role": "jax_imcts_checksum_manifest",
            "path": str(checksum_path.resolve()),
            "sha256": _sha256_file(checksum_path),
            "size_bytes": checksum_path.stat().st_size,
        }
    )
    paths = sorted(freeze_dir.glob("*.jsonl.gz"))
    if not paths:
        raise CleanRerunOverlayError(f"未找到 JAXSR/iMCTS 冻结 bundle: {freeze_dir}")
    records: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for path in paths:
        digest = _verify_manifest_file(
            path,
            data_root=freeze_dir.parent,
            checksum_entries=checksum_entries,
        )
        input_files.append(
            {
                "role": "jax_imcts_freeze_bundle",
                "path": str(path.resolve()),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        try:
            handle = gzip.open(path, "rt", encoding="utf-8")
            with handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise CleanRerunOverlayError(
                            f"{path}:{line_number} JSON 顶层不是 object"
                        )
                    record, result_payload = _validate_frozen_record(
                        payload, horizon=horizon
                    )
                    records.append((record, result_payload, "jax_imcts_raw_freeze"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CleanRerunOverlayError(f"无法解析冻结 bundle {path}: {exc}") from exc
    return records


def _find_symbolfit_outer_result(task_dir: Path) -> Path:
    candidates = [
        path
        for path in task_dir.rglob("result.json")
        if "experiments" not in path.relative_to(task_dir).parts
        and (path.parent / "progress").is_dir()
    ]
    if len(candidates) != 1:
        raise CleanRerunOverlayError(
            f"{task_dir} 的 SymbolFit outer result 数量异常: {len(candidates)}"
        )
    return candidates[0]


def _discover_symbolfit_full(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for task_dir in sorted(root.glob("symbolfit/seed*/tasks/symbolfit_s*_clean_g*")):
        if not task_dir.is_dir() or not _TASK_ID.fullmatch(task_dir.name):
            continue
        if task_dir.name in discovered:
            raise CleanRerunOverlayError(f"SymbolFit full task_id 重复: {task_dir.name}")
        discovered[task_dir.name] = task_dir
    if not discovered:
        raise CleanRerunOverlayError(f"SymbolFit full root 中没有任务: {root}")
    return discovered


def _discover_strict_task(root: Path, *, strict_task_id: str) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and _find_task_id_from_path(path) == strict_task_id
        and any(
            child.name == "result.json" and (child.parent / "progress").is_dir()
            for child in path.glob("*/result.json")
        )
    ]
    if candidates:
        candidates.sort(key=lambda item: len(item.parts))
        return candidates[0]
    # 正式 strict root 没有 task_id 目录，直接从唯一 outer result 反推任务根。
    result_candidates = [
        path
        for path in root.rglob("result.json")
        if "experiments" not in path.relative_to(root).parts
        and (path.parent / "progress").is_dir()
    ]
    if len(result_candidates) != 1:
        raise CleanRerunOverlayError(
            f"strict root 的 outer result 数量异常: {len(result_candidates)}"
        )
    return result_candidates[0].parent


def _find_task_id_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if _TASK_ID.fullmatch(part):
            return part
    return None


def _symbolfit_record(
    task_dir: Path,
    *,
    task_id: str,
    batch: str,
    checksum_root: Path,
    checksum_entries: Mapping[str, Sequence[tuple[str, str]]],
    horizon: int,
    base_by_normal_key: Mapping[tuple[str, str, int], tuple[str, Mapping[str, str]]],
    input_files: list[dict[str, Any]],
    origin: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    match = _TASK_ID.fullmatch(task_id)
    if match is None or _normal_algorithm(match.group("algorithm")) != "symbolfit":
        raise CleanRerunOverlayError(f"SymbolFit task_id 非法: {task_id}")
    seed = int(match.group("seed"))
    resolved_checksum_root = checksum_root.resolve()
    result_path = _find_symbolfit_outer_result(task_dir)
    resolved_result_path = result_path.resolve()
    raw_result, result_payload = _read_json_file(result_path)
    dataset = str(result_payload.get("dataset") or "")
    normal_key = ("symbolfit", dataset, seed)
    if normal_key not in base_by_normal_key:
        raise CleanRerunOverlayError(
            f"SymbolFit 重跑不属于 base clean 网格: {normal_key}"
        )
    base_key, base_row = base_by_normal_key[normal_key]
    algorithm = base_row["algorithm"]
    _validate_result_payload(
        result_payload,
        algorithm=algorithm,
        dataset=dataset,
        seed=seed,
        context=base_key,
    )
    result_digest = _verify_manifest_file(
        result_path,
        data_root=checksum_root,
        checksum_entries=checksum_entries,
        resolved_path=resolved_result_path,
        resolved_data_root=resolved_checksum_root,
    )
    if result_digest != _sha256_bytes(raw_result):
        raise CleanRerunOverlayError(f"{base_key} result 读取期间发生变化")
    used_files = [
        {
            "role": "symbolfit_outer_result",
            "path": str(resolved_result_path),
            "sha256": result_digest,
            "size_bytes": len(raw_result),
            "origin": origin,
        }
    ]
    progress_dir = result_path.parent / "progress"
    resolved_progress_dir = progress_dir.resolve()
    snapshots: list[dict[str, Any]] = []
    previous_internal_loss = math.inf
    seen_candidate = False
    for minute in range(1, horizon + 1):
        path = progress_dir / f"minute_{minute:04d}.json"
        resolved_path = path.resolve()
        raw, payload = _read_json_file(path)
        digest = _verify_manifest_file(
            path,
            data_root=checksum_root,
            checksum_entries=checksum_entries,
            resolved_path=resolved_path,
            resolved_data_root=resolved_checksum_root,
        )
        if digest != _sha256_bytes(raw):
            raise CleanRerunOverlayError(f"{base_key} minute_{minute:04d} 读取期间发生变化")
        _validate_checkpoint_payload(
            payload,
            algorithm=algorithm,
            dataset=dataset,
            seed=seed,
            minute=minute,
            horizon=horizon,
            context=base_key,
        )
        record_type = payload.get("record_type")
        if record_type == "periodic_heartbeat":
            if seen_candidate:
                raise CleanRerunOverlayError(f"{base_key} 发现候选后退回 heartbeat")
            if any(
                payload.get(field) is not None
                for field in ("equation", "canonical_artifact", "source_internal_loss")
            ):
                raise CleanRerunOverlayError(f"{base_key} heartbeat 携带候选")
        else:
            seen_candidate = True
            loss = _finite_number(
                payload.get("source_internal_loss"),
                field=f"{base_key}.minute_{minute:04d}.source_internal_loss",
            )
            if loss > previous_internal_loss + 1e-12:
                raise CleanRerunOverlayError(f"{base_key} internal best loss 发生回退")
            previous_internal_loss = loss
            if payload.get("candidate_source") != "symbolfit_active_pysr_hall_of_fame":
                raise CleanRerunOverlayError(f"{base_key} checkpoint 不是内部 PySR 候选")
            artifact = payload.get("canonical_artifact")
            if (
                not isinstance(payload.get("equation"), str)
                or not payload["equation"].strip()
                or not isinstance(artifact, Mapping)
                or artifact.get("artifact_valid") is not True
            ):
                raise CleanRerunOverlayError(f"{base_key} checkpoint 候选或 artifact 无效")
        raw_text = raw.decode("utf-8")
        snapshots.append(
            {
                "minute": minute,
                "outer_path": str(resolved_path),
                "inner_path": None,
                "outer_status": "ok",
                "inner_status": None,
                "outer_sha256": digest,
                "inner_sha256": None,
                "duplicate_semantically_equal": False,
                "conflict": False,
                "status": "ok",
                "selected_path": str(resolved_path),
                "selected_sha256": digest,
                "selected_semantic_sha256": _sha256_bytes(
                    _canonical_json(payload).encode("utf-8")
                ),
                **_compact_payload(payload),
                "raw_text": raw_text,
            }
        )
        used_files.append(
            {
                "role": "symbolfit_outer_checkpoint",
                "path": str(resolved_path),
                "sha256": digest,
                "size_bytes": len(raw),
                "origin": origin,
            }
        )
    if not seen_candidate:
        raise CleanRerunOverlayError(f"{base_key} 没有内部候选")
    host = next(
        (part for part in result_path.parts if re.fullmatch(r"iaaccn\d+", part)),
        "strict-local" if origin == "symbolfit_strict_replacement" else "unknown",
    )
    source = _source_dict(
        batch=batch,
        algorithm=algorithm,
        dataset=dataset,
        host=host,
        seed=seed,
        task_id=task_id,
        result_path=str(resolved_result_path),
    )
    record = {
        "source": source,
        "result": {
            "path": str(resolved_result_path),
            "exists": True,
            "size_bytes": len(raw_result),
            "sha256": result_digest,
            "status": "ok",
            "semantic_sha256": _sha256_bytes(
                _canonical_json(result_payload).encode("utf-8")
            ),
            "payload_summary": _compact_payload(result_payload),
            "raw_text": raw_result.decode("utf-8"),
        },
        "outer_progress_dir": str(resolved_progress_dir),
        "inner_progress_dir": None,
        "snapshots": snapshots,
        "summary": {
            "expected_snapshots": horizon,
            "available_snapshots": horizon,
            "missing_snapshots": 0,
            "conflicting_snapshots": 0,
            "parse_errors": 0,
        },
    }
    input_files.extend(used_files)
    return record, result_payload, origin


def _replacement_row(
    base_row: Mapping[str, str],
    record: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> dict[str, str]:
    source = record["source"]
    row = dict(base_row)
    row.update(
        {
            "batch": str(source["batch"]),
            "algorithm": str(source["algorithm"]),
            "dataset_id": str(source["dataset_id"]),
            "seed": str(source["seed"]),
            "noise_tag": "clean",
            "task_id": str(source["task_id"]),
            "host": str(source["host"]),
            "status": "ok",
            "seconds": f"{_finite_number(result_payload.get('seconds'), field='result.seconds', nonnegative=True):.17g}",
            "id_nmse": f"{_metric(result_payload, 'id_test', 'nmse'):.17g}",
            "ood_nmse": f"{_metric(result_payload, 'ood_test', 'nmse'):.17g}",
            "id_r2": f"{_metric(result_payload, 'id_test', 'r2'):.17g}",
            "ood_r2": f"{_metric(result_payload, 'ood_test', 'r2'):.17g}",
            "id_acc": f"{_metric(result_payload, 'id_test', 'acc_0_1'):.17g}",
            "ood_acc": f"{_metric(result_payload, 'ood_test', 'acc_0_1'):.17g}",
            "path": str(source["path"]),
            "logical_key": str(base_row["logical_key"]),
        }
    )
    return row


def build_clean_rerun_overlay(
    *,
    base_source_runs_csv: Path,
    jax_imcts_freeze_dir: Path,
    symbolfit_full_root: Path,
    symbolfit_strict_root: Path,
    output_bundle: Path,
    output_manifest: Path,
    output_composite_csv: Path,
    horizon: int = 180,
    expected_clean_rows: int | None = 2250,
    expected_counts: Mapping[str, int] | None = None,
    strict_task_id: str = DEFAULT_STRICT_TASK_ID,
) -> dict[str, Any]:
    """验证全部重跑证据并原子构建 225 条 EFF 轨迹覆盖层。"""

    if horizon < 1:
        raise CleanRerunOverlayError("horizon 必须为正整数")
    counts_expected = {
        _normal_algorithm(key): int(value)
        for key, value in (expected_counts or DEFAULT_EXPECTED_COUNTS).items()
    }
    if any(value < 0 for value in counts_expected.values()):
        raise CleanRerunOverlayError("expected_counts 不允许负数")

    fieldnames, clean_rows, base_by_key = _load_clean_source_rows(
        base_source_runs_csv, expected_clean_rows=expected_clean_rows
    )
    base_by_normal_key: dict[tuple[str, str, int], tuple[str, Mapping[str, str]]] = {}
    for key, row in base_by_key.items():
        normal_key = (_normal_algorithm(row["algorithm"]), row["dataset_id"], int(row["seed"]))
        if normal_key in base_by_normal_key:
            raise CleanRerunOverlayError(f"base clean 规范化逻辑键重复: {normal_key}")
        base_by_normal_key[normal_key] = (key, row)

    input_files: list[dict[str, Any]] = [
        {
            "role": "base_source_runs_csv",
            "path": str(base_source_runs_csv.resolve()),
            "sha256": _sha256_file(base_source_runs_csv),
            "size_bytes": base_source_runs_csv.stat().st_size,
        }
    ]
    loaded = _load_jax_imcts_records(
        jax_imcts_freeze_dir, horizon=horizon, input_files=input_files
    )

    full_checksum_path = symbolfit_full_root / "SHA256SUMS"
    full_checksums = _parse_checksum_manifest(full_checksum_path)
    strict_checksum_path = symbolfit_strict_root.parent / f"{symbolfit_strict_root.name}.SHA256SUMS"
    if not strict_checksum_path.is_file():
        strict_checksum_path = symbolfit_strict_root.parent / "collected.SHA256SUMS"
    strict_checksums = _parse_checksum_manifest(strict_checksum_path)
    for role, path in (
        ("symbolfit_full_checksum_manifest", full_checksum_path),
        ("symbolfit_strict_checksum_manifest", strict_checksum_path),
    ):
        input_files.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    full_tasks = _discover_symbolfit_full(symbolfit_full_root)
    if strict_task_id not in full_tasks:
        raise CleanRerunOverlayError(
            f"full150 中缺少必须由 strict 替换的任务: {strict_task_id}"
        )
    full_replaced_result = _find_symbolfit_outer_result(full_tasks[strict_task_id])
    full_replaced_sha = _verify_manifest_file(
        full_replaced_result,
        data_root=symbolfit_full_root,
        checksum_entries=full_checksums,
    )
    for task_id, task_dir in sorted(full_tasks.items()):
        if task_id == strict_task_id:
            continue
        loaded.append(
            _symbolfit_record(
                task_dir,
                task_id=task_id,
                batch="symbolfit_internal_progress_v1_full",
                checksum_root=symbolfit_full_root,
                checksum_entries=full_checksums,
                horizon=horizon,
                base_by_normal_key=base_by_normal_key,
                input_files=input_files,
                origin="symbolfit_full150",
            )
        )

    strict_task_dir = _discover_strict_task(
        symbolfit_strict_root, strict_task_id=strict_task_id
    )
    strict_record = _symbolfit_record(
        strict_task_dir,
        task_id=strict_task_id,
        batch="symbolfit_internal_progress_v1_strict_g0039",
        checksum_root=symbolfit_strict_root,
        checksum_entries=strict_checksums,
        horizon=horizon,
        base_by_normal_key=base_by_normal_key,
        input_files=input_files,
        origin="symbolfit_strict_replacement",
    )
    loaded.append(strict_record)

    overlay: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    actual_counts: Counter[str] = Counter()
    for record, result_payload, origin in loaded:
        source = record["source"]
        normal_key = (
            _normal_algorithm(source["algorithm"]),
            str(source["dataset_id"]),
            int(source["seed"]),
        )
        base_match = base_by_normal_key.get(normal_key)
        if base_match is None:
            raise CleanRerunOverlayError(f"覆盖记录不属于 base clean 网格: {normal_key}")
        key, base_row = base_match
        if key in overlay:
            raise CleanRerunOverlayError(f"覆盖记录出现重复 logical_key: {key}")
        source = _source_dict(
            batch=str(source["batch"]),
            algorithm=base_row["algorithm"],
            dataset=base_row["dataset_id"],
            host=str(source["host"]),
            seed=int(base_row["seed"]),
            task_id=str(source["task_id"]),
            result_path=str(source["path"]),
        )
        record["source"] = source
        replacement_scope = EFF_ONLY if normal_key[0] == "symbolfit" else FINAL_AND_EFF
        record["overlay"] = {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "replacement_scope": replacement_scope,
            "logical_key": key,
            "origin": origin,
            "base_final_preserved": replacement_scope == EFF_ONLY,
        }
        overlay[key] = (record, result_payload, origin)
        actual_counts[_normal_algorithm(source["algorithm"])] += 1

    if dict(sorted(actual_counts.items())) != dict(sorted(counts_expected.items())):
        raise CleanRerunOverlayError(
            f"覆盖算法计数不符: actual={dict(actual_counts)}, expected={counts_expected}"
        )
    expected_total = sum(counts_expected.values())
    if len(overlay) != expected_total:
        raise CleanRerunOverlayError(
            f"覆盖 logical_key 应为 {expected_total} 个，实际为 {len(overlay)}"
        )

    composite_by_key = {row["logical_key"]: dict(row) for row in clean_rows}
    replacement_details: list[dict[str, Any]] = []
    for key in sorted(overlay):
        record, result_payload, origin = overlay[key]
        old_row = composite_by_key[key]
        replacement_scope = str(record["overlay"]["replacement_scope"])
        new_row = _replacement_row(old_row, record, result_payload)
        if replacement_scope == FINAL_AND_EFF:
            composite_by_key[key] = new_row
        replacement_details.append(
            {
                "logical_key": key,
                "algorithm": new_row["algorithm"],
                "task_id": new_row["task_id"],
                "origin": origin,
                "replacement_scope": replacement_scope,
                "base_final_preserved": replacement_scope == EFF_ONLY,
                "old_source_row_sha256": _sha256_bytes(
                    _canonical_json(old_row).encode("utf-8")
                ),
                "replacement_source_row_sha256": _sha256_bytes(
                    _canonical_json(new_row).encode("utf-8")
                ),
                "result_sha256": record["result"]["sha256"],
                "checkpoint_sha256": [
                    snapshot["selected_sha256"] for snapshot in record["snapshots"]
                ],
                "checkpoint_identity_verified": True,
            }
        )

    composite_rows = [composite_by_key[key] for key in sorted(composite_by_key)]
    if len(composite_rows) != len(base_by_key) or len(
        {row["logical_key"] for row in composite_rows}
    ) != len(composite_rows):
        raise CleanRerunOverlayError("复合 clean source-runs 不再保持唯一完整网格")

    overlay_rows = [overlay[key][0] for key in sorted(overlay)]
    _atomic_write_gzip_jsonl(output_bundle, overlay_rows)
    _atomic_write_csv(output_composite_csv, fieldnames, composite_rows)

    strict_key, _ = base_by_normal_key[("symbolfit", strict_record[1]["dataset"], int(strict_record[1]["seed"]))]
    strict_output_record = overlay[strict_key][0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "scope": SCOPE,
        "formal_replacement_policy": {
            "final_and_eff_algorithms": ["imcts", "jaxsr"],
            "eff_only_algorithms": ["symbolfit"],
            "symbolfit_final_preserved": True,
            "purpose": "JAXSR/iMCTS 替换 final 与 EFF；SymbolFit 仅替换 EFF 内部搜索轨迹",
        },
        "horizon_minutes": horizon,
        "base_clean_rows": len(clean_rows),
        "overlay_unique_keys": len(overlay_rows),
        "replacement_count": len(replacement_details),
        "eff_replacement_keys": sorted(overlay),
        "eff_replacement_count": len(overlay),
        "final_replacement_keys": sorted(
            key
            for key, (record, _, _) in overlay.items()
            if record["overlay"]["replacement_scope"] == FINAL_AND_EFF
        ),
        "final_replacement_count": sum(
            record["overlay"]["replacement_scope"] == FINAL_AND_EFF
            for record, _, _ in overlay.values()
        ),
        "algorithm_counts": dict(sorted(actual_counts.items())),
        "expected_algorithm_counts": dict(sorted(counts_expected.items())),
        "checkpoint_identity": {
            "expected_per_run": horizon,
            "verified_runs": len(overlay_rows),
            "verified_points": len(overlay_rows) * horizon,
            "all_verified": True,
        },
        "strict_replacement": {
            "required_task_id": strict_task_id,
            "logical_key": strict_key,
            "forced": True,
            "full150_candidate_excluded": True,
            "full150_result_path": str(full_replaced_result.resolve()),
            "full150_result_sha256": full_replaced_sha,
            "strict_result_path": strict_output_record["result"]["path"],
            "strict_result_sha256": strict_output_record["result"]["sha256"],
            "different_result_evidence": (
                full_replaced_sha != strict_output_record["result"]["sha256"]
            ),
        },
        "inputs": {
            "files": sorted(input_files, key=lambda item: (item["role"], item["path"])),
            "all_sha256_verified": True,
        },
        "outputs": {
            "overlay_bundle": {
                "path": str(output_bundle.resolve()),
                "sha256": _sha256_file(output_bundle),
                "size_bytes": output_bundle.stat().st_size,
                "rows": len(overlay_rows),
            },
            "diagnostic_composite_source_runs_csv": {
                "path": str(output_composite_csv.resolve()),
                "sha256": _sha256_file(output_composite_csv),
                "size_bytes": output_composite_csv.stat().st_size,
                "rows": len(composite_rows),
                "replaced_final_rows": sum(
                    record["overlay"]["replacement_scope"] == FINAL_AND_EFF
                    for record, _, _ in overlay.values()
                ),
                "symbolfit_rows_preserved": True,
            },
            "manifest_self_hash": None,
            "manifest_self_hash_exclusion": "manifest cannot contain its own SHA256",
        },
        "replacements": replacement_details,
    }
    _atomic_write_text(
        output_manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    stage_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-source-runs-csv",
        type=Path,
        default=stage_root / "manifests/source_runs.csv",
    )
    parser.add_argument(
        "--jax-imcts-freeze-dir",
        type=Path,
        default=stage_root / "work/clean_rerun_trajectory_freeze_v1/collected",
    )
    parser.add_argument(
        "--symbolfit-full-root",
        type=Path,
        default=stage_root
        / "reruns/symbolfit_internal_progress_v1/collected/full150",
    )
    parser.add_argument(
        "--symbolfit-strict-root",
        type=Path,
        default=stage_root
        / "reruns/symbolfit_internal_progress_v1/strict_rerun_g0039/collected",
    )
    parser.add_argument(
        "--output-bundle",
        type=Path,
        default=stage_root / "work/clean_rerun_overlay_v1/clean_eff_overlay.jsonl.gz",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=stage_root / "work/clean_rerun_overlay_v1/manifest.json",
    )
    parser.add_argument(
        "--output-composite-csv",
        type=Path,
        default=stage_root
        / "work/clean_rerun_overlay_v1/source_runs.clean.diagnostic.csv",
    )
    args = parser.parse_args()
    report = build_clean_rerun_overlay(
        base_source_runs_csv=args.base_source_runs_csv,
        jax_imcts_freeze_dir=args.jax_imcts_freeze_dir,
        symbolfit_full_root=args.symbolfit_full_root,
        symbolfit_strict_root=args.symbolfit_strict_root,
        output_bundle=args.output_bundle,
        output_manifest=args.output_manifest,
        output_composite_csv=args.output_composite_csv,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "overlay_unique_keys": report["overlay_unique_keys"],
                "algorithm_counts": report["algorithm_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
