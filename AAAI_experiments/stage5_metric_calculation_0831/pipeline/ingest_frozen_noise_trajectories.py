#!/usr/bin/env python3
"""把冻结 noise 快照收敛为严格 1..180 分钟的 best-so-far 输入。"""

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
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, Mapping, TextIO

from .trajectories import TrajectoryContractError, reconstruct_trajectory


HORIZON = 180
SUPPORTED_CONDITIONS = {"noise001", "noise005"}
SUPPORTED_SEEDS = (520, 521, 522)
SCHEMA_VERSION = "stage5.frozen_noise_best_so_far.v1"
SOURCE_IDENTITY_FIELDS = (
    "batch",
    "algorithm",
    "dataset_id",
    "host",
    "noise_tag",
    "path",
    "seed",
    "task_id",
)
SNAPSHOT_RECORD_TYPES = {
    "periodic_best",
    "periodic_heartbeat",
    "periodic_backfill",
    "final_best",
    "recovered_final",
    "budget_end_internal_best",
    "audited_carry_forward",
}
TASK_ID_RE = re.compile(
    r"^(?P<algorithm>[a-z0-9]+)_s(?P<seed>520|521|522)_"
    r"(?P<condition>noise001|noise005)_g(?P<dataset_index>\d{4})$"
)
CONDITION_SIGMA = {"noise001": 0.01, "noise005": 0.05}


class FrozenNoiseTrajectoryError(ValueError):
    """冻结包无法可靠转换为严格分钟轨迹。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_allowed_seeds(values: Iterable[int]) -> tuple[int, ...]:
    seeds: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise FrozenNoiseTrajectoryError(f"allowed seed 无效: {value!r}")
        try:
            seed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenNoiseTrajectoryError(f"allowed seed 无效: {value!r}") from exc
        if isinstance(value, float) and not value.is_integer():
            raise FrozenNoiseTrajectoryError(f"allowed seed 无效: {value!r}")
        seeds.add(seed)
    if not seeds:
        raise FrozenNoiseTrajectoryError("allowed_seeds 不能为空")
    return tuple(sorted(seeds))


def _parse_seed(value: object, *, context: str, allowed_seeds: Iterable[int]) -> int:
    if isinstance(value, bool):
        raise FrozenNoiseTrajectoryError(f"{context} 无效: {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise FrozenNoiseTrajectoryError(f"{context} 无效: {value!r}")
    try:
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(text)
        seed = int(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FrozenNoiseTrajectoryError(f"{context} 无效: {value!r}") from exc
    allowed = set(allowed_seeds)
    if seed not in allowed:
        raise FrozenNoiseTrajectoryError(
            f"{context}={seed} 不属于允许种子集合 {sorted(allowed)}"
        )
    return seed


def _source_identity(
    source: Mapping[str, Any], *, allowed_seeds: Iterable[int]
) -> dict[str, Any]:
    missing = [field for field in SOURCE_IDENTITY_FIELDS if field not in source]
    if missing:
        raise FrozenNoiseTrajectoryError(f"source 缺少身份字段: {missing}")
    seed = _parse_seed(
        source.get("seed"), context="source.seed", allowed_seeds=allowed_seeds
    )
    identity = {
        field: str(source.get(field) or "").strip()
        for field in SOURCE_IDENTITY_FIELDS
        if field != "seed"
    }
    identity["seed"] = seed
    if not identity["algorithm"] or not identity["dataset_id"]:
        raise FrozenNoiseTrajectoryError("source 缺少 algorithm/dataset_id")
    if identity["noise_tag"] not in SUPPORTED_CONDITIONS:
        raise FrozenNoiseTrajectoryError(
            f"仅允许 noise001/noise005，实际为 {identity['noise_tag']!r}"
        )
    if not identity["task_id"]:
        raise FrozenNoiseTrajectoryError("source.task_id 不能为空")
    if not identity["batch"] or not identity["host"] or not identity["path"]:
        raise FrozenNoiseTrajectoryError("source 缺少 batch/host/path")
    match = TASK_ID_RE.fullmatch(identity["task_id"])
    if match is None:
        raise FrozenNoiseTrajectoryError(
            f"source.task_id 不是 canonical noise 任务身份: {identity['task_id']!r}"
        )
    if (
        match.group("seed") != str(seed)
        or match.group("condition") != identity["noise_tag"]
    ):
        raise FrozenNoiseTrajectoryError(
            f"source.task_id 与 seed/noise_tag 不一致: {identity['task_id']!r}"
        )
    if match.group("algorithm") != identity["algorithm"].casefold():
        raise FrozenNoiseTrajectoryError(
            f"source.task_id 与 algorithm 不一致: {identity['task_id']!r}"
        )
    return identity


def _canonical_source_row(
    source: Mapping[str, Any], *, allowed_seeds: Iterable[int]
) -> dict[str, Any]:
    """统一采用 source CSV 能重建的身份列，seed 规范化为整数。"""

    identity = _source_identity(source, allowed_seeds=allowed_seeds)
    return {field: identity[field] for field in SOURCE_IDENTITY_FIELDS}


def _source_row_sha256(
    source: Mapping[str, Any], *, allowed_seeds: Iterable[int]
) -> str:
    return _sha256_bytes(
        _canonical_json(
            _canonical_source_row(source, allowed_seeds=allowed_seeds)
        ).encode("utf-8")
    )


def _logical_key(
    source: Mapping[str, Any], *, allowed_seeds: Iterable[int] = SUPPORTED_SEEDS
) -> str:
    identity = _source_identity(source, allowed_seeds=allowed_seeds)
    return (
        f"{identity['algorithm']}::{identity['dataset_id']}::"
        f"s{identity['seed']}::{identity['noise_tag']}"
    )


def _load_source_rows(
    path: Path,
    *,
    allowed_seeds: Iterable[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FrozenNoiseTrajectoryError(f"source_runs.csv 不存在: {path}")
    required = set(SOURCE_IDENTITY_FIELDS) | {"logical_key"}
    rows: dict[str, dict[str, Any]] = {}
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise FrozenNoiseTrajectoryError(f"无法打开 source_runs.csv: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise FrozenNoiseTrajectoryError(f"source_runs.csv 缺少字段: {missing}")
        for line_number, raw_row in enumerate(reader, start=2):
            raw = dict(raw_row)
            noise_tag = str(raw.get("noise_tag") or "").strip()
            if noise_tag not in SUPPORTED_CONDITIONS:
                continue
            identity = _canonical_source_row(raw, allowed_seeds=allowed_seeds)
            logical_key = (
                f"{identity['algorithm']}::{identity['dataset_id']}::"
                f"s{identity['seed']}::{identity['noise_tag']}"
            )
            if raw.get("logical_key") != logical_key:
                raise FrozenNoiseTrajectoryError(
                    f"source_runs.csv:{line_number}.logical_key 不符合 canonical: {raw.get('logical_key')!r}"
                )
            if raw.get("status") not in (None, "", "ok"):
                raise FrozenNoiseTrajectoryError(
                    f"source_runs.csv:{line_number}.status 必须为 ok"
                )
            if logical_key in rows:
                raise FrozenNoiseTrajectoryError(
                    f"source_runs.csv 出现重复逻辑键: {logical_key}"
                )
            rows[logical_key] = {
                "line_number": line_number,
                "row": raw,
                "identity": identity,
                "source_row_sha256": _sha256_bytes(
                    _canonical_json(identity).encode("utf-8")
                ),
            }
    if not rows:
        raise FrozenNoiseTrajectoryError("source_runs.csv 没有 noise001/noise005 身份行")
    return rows, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
    }


def _validate_source_against_csv(
    source: Mapping[str, Any],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    allowed_seeds: Iterable[int],
    context: str,
) -> tuple[str, dict[str, Any]]:
    identity = _canonical_source_row(source, allowed_seeds=allowed_seeds)
    logical_key = (
        f"{identity['algorithm']}::{identity['dataset_id']}::"
        f"s{identity['seed']}::{identity['noise_tag']}"
    )
    csv_record = source_rows.get(logical_key)
    if csv_record is None:
        raise FrozenNoiseTrajectoryError(f"{context} 未命中 source CSV: {logical_key}")
    expected_identity = csv_record["identity"]
    if identity != expected_identity:
        differences = [
            field
            for field in SOURCE_IDENTITY_FIELDS
            if identity.get(field) != expected_identity.get(field)
        ]
        raise FrozenNoiseTrajectoryError(
            f"{context} source 身份与 source CSV 不一致: {differences}"
        )
    source_sha = source.get("source_row_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", source_sha
    ):
        raise FrozenNoiseTrajectoryError(f"{context}.source_row_sha256 必须为 SHA256")
    expected_sha = str(csv_record["source_row_sha256"])
    if source_sha.lower() != expected_sha:
        raise FrozenNoiseTrajectoryError(
            f"{context}.source_row_sha256 与 source CSV canonical row 不一致"
        )
    return logical_key, dict(csv_record)


def _task_id_in_path(value: object, task_id: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip().replace("\\", "/")
    return f"/tasks/{task_id}/" in text or text.endswith(f"/tasks/{task_id}")


def _validate_payload_identity(
    payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    logical_key: str,
    context: str,
    allowed_seeds: Iterable[int],
) -> None:
    expected = _canonical_source_row(source, allowed_seeds=allowed_seeds)
    tool = payload.get("tool")
    dataset = payload.get("dataset")
    if not isinstance(tool, str) or not tool.strip():
        raise FrozenNoiseTrajectoryError(f"{logical_key} {context} 缺少 tool")
    if tool.casefold() != str(expected["algorithm"]).casefold():
        raise FrozenNoiseTrajectoryError(f"{logical_key} {context}.tool 身份不一致")
    if not isinstance(dataset, str) or dataset.strip() != expected["dataset_id"]:
        raise FrozenNoiseTrajectoryError(f"{logical_key} {context}.dataset 身份不一致")
    payload_seed = _parse_seed(
        payload.get("seed"),
        context=f"{logical_key} {context}.seed",
        allowed_seeds=allowed_seeds,
    )
    if payload_seed != expected["seed"]:
        raise FrozenNoiseTrajectoryError(f"{logical_key} {context}.seed 身份不一致")

    payload_task_id = payload.get("task_id")
    if payload_task_id not in (None, ""):
        if str(payload_task_id).strip() != expected["task_id"]:
            raise FrozenNoiseTrajectoryError(f"{logical_key} {context}.task_id 身份不一致")
    elif not _task_id_in_path(payload.get("experiment_dir"), str(expected["task_id"])):
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} {context} 缺少可验证的 task_id/experiment_dir 身份"
        )

    payload_condition = payload.get("noise_tag")
    if (
        payload_condition not in (None, "")
        and str(payload_condition) != expected["noise_tag"]
    ):
        raise FrozenNoiseTrajectoryError(f"{logical_key} {context}.noise_tag 身份不一致")
    noise = payload.get("train_label_noise")
    if noise is not None:
        if not isinstance(noise, Mapping):
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} {context}.train_label_noise 无效"
            )
        if noise.get("enabled") is not True or noise.get("requested") is not True:
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} {context} 噪声标记与 condition 不一致"
            )
        try:
            sigma = float(noise.get("sigma"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} {context}.sigma 无效"
            ) from exc
        if not math.isfinite(sigma) or not math.isclose(
            sigma,
            CONDITION_SIGMA[str(expected["noise_tag"])],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} {context}.sigma 与 condition 不一致"
            )


def _validate_result(
    record: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    logical_key: str,
    allowed_seeds: Iterable[int],
) -> str:
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise FrozenNoiseTrajectoryError(f"{logical_key} 缺少 result object")
    raw_text = result.get("raw_text")
    expected_sha = result.get("sha256")
    if not isinstance(raw_text, str) or not raw_text:
        raise FrozenNoiseTrajectoryError(f"{logical_key} result 缺少 raw_text")
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha
    ):
        raise FrozenNoiseTrajectoryError(f"{logical_key} result.sha256 必须为 SHA256")
    actual_sha = _sha256_bytes(raw_text.encode("utf-8"))
    if actual_sha != expected_sha:
        raise FrozenNoiseTrajectoryError(f"{logical_key} result.raw_text SHA256 不匹配")
    if result.get("status") not in (None, "", "ok"):
        raise FrozenNoiseTrajectoryError(f"{logical_key} result.status 必须为 ok")
    if result.get("path") not in (None, "", source.get("path")):
        raise FrozenNoiseTrajectoryError(f"{logical_key} result.path 与 source.path 不一致")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} result.raw_text 不是合法 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FrozenNoiseTrajectoryError(f"{logical_key} result.raw_text 顶层不是 object")
    _validate_payload_identity(
        payload,
        source=source,
        logical_key=logical_key,
        context="result",
        allowed_seeds=allowed_seeds,
    )
    if result.get("status") == "ok" and payload.get("status") != "ok":
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} result.status 与 raw payload 不一致"
        )
    return actual_sha


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        handle = gzip.open(path, "rt", encoding="utf-8")
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FrozenNoiseTrajectoryError(
                        f"{path}:{line_number} 不是合法 JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise FrozenNoiseTrajectoryError(
                        f"{path}:{line_number} 顶层不是 JSON object"
                    )
                yield line_number, value
    except (OSError, EOFError) as exc:
        raise FrozenNoiseTrajectoryError(f"无法读取 gzip 冻结包 {path}: {exc}") from exc


def _parse_minute(value: object, *, logical_key: str) -> int:
    if isinstance(value, bool):
        raise FrozenNoiseTrajectoryError(f"{logical_key} snapshot.minute 无效: {value!r}")
    try:
        minute = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} snapshot.minute 无效: {value!r}"
        ) from exc
    if str(value).strip() != str(minute) and not isinstance(value, int):
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} snapshot.minute 非整数: {value!r}"
        )
    if minute <= 0:
        raise FrozenNoiseTrajectoryError(f"{logical_key} snapshot.minute 必须为正整数")
    return minute


def _decode_snapshot(
    snapshot: Mapping[str, Any],
    *,
    minute: int,
    logical_key: str,
    source: Mapping[str, Any],
    allowed_seeds: Iterable[int],
) -> tuple[dict[str, Any] | None, str | None]:
    if snapshot.get("conflict") is True or snapshot.get("status") == "conflict":
        return None, "conflict"
    status = str(snapshot.get("status") or "")
    if status != "ok":
        return None, status or "unusable"
    raw_text = snapshot.get("raw_text")
    selected_sha256 = snapshot.get("selected_sha256")
    if not isinstance(raw_text, str) or not isinstance(selected_sha256, str):
        return None, "missing_raw_evidence"
    actual_sha256 = _sha256_bytes(raw_text.encode("utf-8"))
    if actual_sha256 != selected_sha256:
        return None, "raw_sha256_mismatch"
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, "raw_parse_error"
    if not isinstance(payload, dict):
        return None, "raw_not_object"

    _validate_payload_identity(
        payload,
        source=source,
        logical_key=logical_key,
        context=f"snapshot minute_{minute:04d}",
        allowed_seeds=allowed_seeds,
    )
    record_type = payload.get("record_type")
    if record_type not in SNAPSHOT_RECORD_TYPES:
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} minute_{minute:04d} snapshot.record_type 无效: {record_type!r}"
        )
    checkpoint_index = payload.get("checkpoint_index")
    if (
        record_type == "final_best"
        and minute == HORIZON
        and checkpoint_index == "final"
    ):
        pass
    else:
        if isinstance(checkpoint_index, bool):
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} minute_{minute:04d} snapshot.checkpoint_index 无效"
            )
        try:
            parsed_index = int(checkpoint_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} minute_{minute:04d} snapshot.checkpoint_index 无效"
            ) from exc
        if parsed_index != minute:
            raise FrozenNoiseTrajectoryError(
                f"{logical_key} minute_{minute:04d} snapshot.checkpoint_index 身份不一致"
            )

    # wrapper 分钟与 payload checkpoint_index 都必须绑定；不能只依赖 trajectories.py 的后验校验。
    if snapshot.get("minute") != minute:
        raise FrozenNoiseTrajectoryError(
            f"{logical_key} minute_{minute:04d} wrapper 身份在解析时漂移"
        )
    return payload, None


@contextmanager
def _atomic_gzip_jsonl_writer(path: Path) -> Iterator[TextIO]:
    """逐行写确定性 gzip，只有完整成功后才替换目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
                    yield text
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    os.replace(temporary, path)


def _index_sources(
    input_paths: Iterable[Path],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    allowed_seeds: Iterable[int],
) -> tuple[dict[str, list[dict[str, Any]]], int, dict[str, int], dict[str, int],]:
    """第一遍只保留轻量 source 位置，确保重复 key 的全部记录都可排除。"""

    locations: dict[str, list[dict[str, Any]]] = {}
    record_count = 0
    by_condition: dict[str, int] = {}
    by_algorithm: dict[str, int] = {}
    for input_path in input_paths:
        for line_number, record in _iter_records(input_path):
            record_count += 1
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise FrozenNoiseTrajectoryError(
                    f"{input_path}:{line_number} 缺少 source object"
                )
            logical_key, _ = _validate_source_against_csv(
                source,
                source_rows=source_rows,
                allowed_seeds=allowed_seeds,
                context=f"{input_path}:{line_number}",
            )
            _validate_result(
                record,
                source=source,
                logical_key=logical_key,
                allowed_seeds=allowed_seeds,
            )
            condition = str(source["noise_tag"])
            algorithm = str(source["algorithm"])
            by_condition[condition] = by_condition.get(condition, 0) + 1
            by_algorithm[algorithm] = by_algorithm.get(algorithm, 0) + 1
            locations.setdefault(logical_key, []).append(
                {
                    "input_bundle_path": str(input_path),
                    "input_line_number": line_number,
                }
            )
    return locations, record_count, by_condition, by_algorithm


def ingest_frozen_noise_trajectories(
    *,
    input_paths: Iterable[Path],
    output_jsonl_gz: Path,
    manifest_json: Path,
    source_runs_csv: Path,
    expected_tasks: int | None = None,
    allowed_seeds: Iterable[int] = SUPPORTED_SEEDS,
) -> dict[str, Any]:
    """解析冻结证据；缺失分钟只进入审计，不以未来信息或插值补齐。"""

    normalized_seeds = _normalize_allowed_seeds(allowed_seeds)
    source_runs_csv = Path(source_runs_csv).resolve()
    source_rows, source_csv_info = _load_source_rows(
        source_runs_csv,
        allowed_seeds=normalized_seeds,
    )
    resolved_inputs = sorted({Path(path).resolve() for path in input_paths}, key=str)
    if not resolved_inputs:
        raise FrozenNoiseTrajectoryError("至少需要一个冻结包")
    for path in resolved_inputs:
        if not path.is_file():
            raise FrozenNoiseTrajectoryError(f"冻结包不存在: {path}")
    output_jsonl_gz = output_jsonl_gz.resolve()
    manifest_json = manifest_json.resolve()
    if (
        output_jsonl_gz in resolved_inputs
        or manifest_json in resolved_inputs
        or source_runs_csv in resolved_inputs
    ):
        raise FrozenNoiseTrajectoryError("输出路径不能覆盖输入冻结包")
    if output_jsonl_gz == manifest_json or output_jsonl_gz == source_runs_csv:
        raise FrozenNoiseTrajectoryError("JSONL 输出与 manifest 不能是同一路径")
    if manifest_json == source_runs_csv:
        raise FrozenNoiseTrajectoryError("manifest 不能覆盖 source_runs.csv")
    if expected_tasks is not None and expected_tasks < 0:
        raise FrozenNoiseTrajectoryError("expected_tasks 不能为负数")

    inputs = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in resolved_inputs
    ]
    locations, record_count, by_condition, by_algorithm = _index_sources(
        resolved_inputs,
        source_rows=source_rows,
        allowed_seeds=normalized_seeds,
    )
    duplicate_locations = {
        logical_key: occurrences
        for logical_key, occurrences in locations.items()
        if len(occurrences) > 1
    }
    unresolved: list[dict[str, Any]] = [
        {
            "logical_key": logical_key,
            "reason": "duplicate_logical_key",
            "occurrences": occurrences,
        }
        for logical_key, occurrences in sorted(duplicate_locations.items())
    ]
    stats = {
        "input_bundle_count": len(resolved_inputs),
        "record_count": record_count,
        "ready_task_count": 0,
        "unresolved_task_count": 0,
        "duplicate_logical_key_count": len(duplicate_locations),
        "duplicate_record_count": sum(map(len, duplicate_locations.values())),
        "within_budget_snapshots_seen": 0,
        "over_budget_snapshots_truncated": 0,
        "missing_snapshot_count": 0,
        "conflicting_snapshot_count": 0,
        "parse_or_evidence_error_snapshot_count": 0,
        "internal_best_carry_points": 0,
    }
    input_info_by_path = {Path(item["path"]): item for item in inputs}
    ready_count = 0
    with _atomic_gzip_jsonl_writer(output_jsonl_gz) as output_handle:
        for input_path in resolved_inputs:
            input_info = input_info_by_path[input_path]
            for line_number, record in _iter_records(input_path):
                source = record.get("source")
                if not isinstance(source, Mapping):
                    raise FrozenNoiseTrajectoryError(
                        f"{input_path}:{line_number} 缺少 source object"
                    )
                logical_key, _ = _validate_source_against_csv(
                    source,
                    source_rows=source_rows,
                    allowed_seeds=normalized_seeds,
                    context=f"{input_path}:{line_number}",
                )
                if logical_key in duplicate_locations:
                    continue
                algorithm = str(source["algorithm"])

                frozen_snapshots = record.get("snapshots")
                if not isinstance(frozen_snapshots, list):
                    raise FrozenNoiseTrajectoryError(f"{logical_key} 缺少 snapshots 数组")
                within_budget: dict[int, Mapping[str, Any]] = {}
                duplicate_minutes: list[int] = []
                for snapshot in frozen_snapshots:
                    if not isinstance(snapshot, Mapping):
                        raise FrozenNoiseTrajectoryError(
                            f"{logical_key} snapshot 不是 object"
                        )
                    minute = _parse_minute(
                        snapshot.get("minute"), logical_key=logical_key
                    )
                    if minute > HORIZON:
                        stats["over_budget_snapshots_truncated"] += 1
                        continue
                    stats["within_budget_snapshots_seen"] += 1
                    if minute in within_budget:
                        duplicate_minutes.append(minute)
                        continue
                    within_budget[minute] = snapshot

                missing_minutes = sorted(
                    set(range(1, HORIZON + 1)) - set(within_budget)
                )
                unusable_minutes: list[int] = []
                parsed_snapshots: dict[int, dict[str, Any]] = {}
                snapshot_hashes: dict[str, str] = {}
                for minute in sorted(within_budget):
                    snapshot = within_budget[minute]
                    payload, failure = _decode_snapshot(
                        snapshot,
                        minute=minute,
                        logical_key=logical_key,
                        source=source,
                        allowed_seeds=normalized_seeds,
                    )
                    if failure is not None:
                        unusable_minutes.append(minute)
                        if failure == "conflict":
                            stats["conflicting_snapshot_count"] += 1
                        elif failure not in {"missing", "unusable"}:
                            stats["parse_or_evidence_error_snapshot_count"] += 1
                        continue
                    assert payload is not None
                    parsed_snapshots[minute] = payload
                    snapshot_hashes[str(minute)] = str(snapshot["selected_sha256"])

                unavailable = sorted(
                    set(missing_minutes + unusable_minutes + duplicate_minutes)
                )
                stats["missing_snapshot_count"] += len(unavailable)
                if unavailable:
                    unresolved.append(
                        {
                            "logical_key": logical_key,
                            "reason": "missing_or_unusable_snapshots",
                            "missing_minutes": unavailable,
                        }
                    )
                    continue

                try:
                    trajectory = reconstruct_trajectory(
                        parsed_snapshots,
                        horizon=HORIZON,
                        algorithm=algorithm,
                    )
                except TrajectoryContractError as exc:
                    unresolved.append(
                        {
                            "logical_key": logical_key,
                            "reason": f"trajectory_contract_error: {exc}",
                            "missing_minutes": [],
                        }
                    )
                    continue
                points = [asdict(point) for point in trajectory]
                internal_carry = sum(
                    str(point["source"]).startswith("internal_best_carry_forward:")
                    for point in points
                )
                stats["internal_best_carry_points"] += internal_carry
                output_row = {
                    "schema_version": SCHEMA_VERSION,
                    "logical_key": logical_key,
                    "source": dict(source),
                    "horizon": HORIZON,
                    "snapshots": {
                        str(minute): parsed_snapshots[minute]
                        for minute in range(1, HORIZON + 1)
                    },
                    "snapshot_sha256": snapshot_hashes,
                    "best_so_far_trajectory": points,
                    "provenance": {
                        "input_bundle_path": str(input_path),
                        "input_bundle_sha256": input_info["sha256"],
                        "input_line_number": line_number,
                        "source_row_sha256": source.get("source_row_sha256"),
                        "result_sha256": (
                            record.get("result", {}).get("sha256")
                            if isinstance(record.get("result"), Mapping)
                            else None
                        ),
                        "over_budget_policy": "truncate_minutes_greater_than_180",
                        "trajectory_policy": "pipeline.trajectories.reconstruct_trajectory",
                    },
                }
                output_handle.write(_canonical_json(output_row) + "\n")
                ready_count += 1

    unresolved.sort(key=lambda row: str(row["logical_key"]))
    stats["ready_task_count"] = ready_count
    stats["unresolved_task_count"] = len(unresolved)
    expected_count_ok = (
        expected_tasks is None or stats["record_count"] == expected_tasks
    )
    stats["strict_1_to_180_ready"] = bool(
        expected_count_ok
        and stats["ready_task_count"] == stats["record_count"]
        and not unresolved
    )

    output_info = {
        "path": str(output_jsonl_gz),
        "sha256": _sha256_file(output_jsonl_gz),
        "size_bytes": output_jsonl_gz.stat().st_size,
        "record_count": ready_count,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "horizon": HORIZON,
        "conditions_allowed": sorted(SUPPORTED_CONDITIONS),
        "seeds_allowed": list(normalized_seeds),
        "expected_task_count": expected_tasks,
        "inputs": inputs,
        "source_runs_csv": source_csv_info,
        "outputs": {"jsonl_gz": output_info},
        "summary": stats,
        "counts_by_condition": dict(sorted(by_condition.items())),
        "counts_by_algorithm": dict(sorted(by_algorithm.items())),
        "unresolved": unresolved,
    }
    _atomic_write_json(manifest_json, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    stage5_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="将冻结 noise trajectory 转换为严格 1..180 分钟 best-so-far 输入"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="noise_freeze_*.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-runs-csv",
        type=Path,
        default=stage5_root / "manifests/source_runs.csv",
        help="用于校验 source_row_sha256 和任务身份的 source_runs.csv",
    )
    parser.add_argument(
        "--allowed-seeds",
        type=int,
        nargs="+",
        default=list(SUPPORTED_SEEDS),
        help="允许的随机种子；正式默认必须为 520 521 522",
    )
    parser.add_argument("--expected-tasks", type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = ingest_frozen_noise_trajectories(
        input_paths=args.inputs,
        output_jsonl_gz=args.output,
        manifest_json=args.manifest,
        source_runs_csv=args.source_runs_csv,
        expected_tasks=args.expected_tasks,
        allowed_seeds=args.allowed_seeds,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if report["summary"]["strict_1_to_180_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
