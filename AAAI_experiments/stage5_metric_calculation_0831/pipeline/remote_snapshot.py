#!/usr/bin/env python3
"""在结果所在主机上只读扫描并冻结 Stage 4 最终结果与分钟轨迹。

该文件只依赖 Python 标准库，可以单独复制到远端持久化实验目录执行。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


class SnapshotIdentityError(ValueError):
    """快照无法与当前调度任务建立唯一、同代的身份绑定。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _nmse(payload: Mapping[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        return None
    return _finite_nonnegative(block.get("nmse"))


def _expression(payload: Mapping[str, Any]) -> str:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    for value in (
        artifact.get("instantiated_expression"),
        artifact.get("normalized_expression"),
        artifact.get("return_expression_source"),
        payload.get("equation"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _read_json(path: Path, *, freeze_raw: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        record["status"] = "missing"
        return record
    try:
        stat = path.stat()
        raw = path.read_bytes()
    except OSError as exc:
        record.update(
            status="read_error",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return record
    record.update(size_bytes=len(raw), sha256=_sha256(raw), mtime_ns=stat.st_mtime_ns)
    try:
        raw_text = raw.decode("utf-8")
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise TypeError("JSON 顶层不是 object")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        record.update(
            status="parse_error",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        if freeze_raw:
            record["raw_text"] = raw.decode("utf-8", errors="replace")
        return record
    record.update(
        status="ok",
        semantic_sha256=_sha256(_canonical_json(payload).encode("utf-8")),
        payload=payload,
    )
    if freeze_raw:
        record["raw_text"] = raw_text
    return record


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_type": payload.get("record_type"),
        "payload_status": payload.get("status"),
        "checkpoint_index": payload.get("checkpoint_index"),
        "elapsed_minutes": payload.get("elapsed_minutes"),
        "backfilled_from_minute": payload.get("backfilled_from_minute"),
        "tool": payload.get("tool"),
        "dataset": payload.get("dataset"),
        "dataset_dir": payload.get("dataset_dir"),
        "seed": payload.get("seed"),
        "task_global_index": payload.get("task_global_index"),
        "has_expression": bool(_expression(payload)),
        "expression": _expression(payload),
        "id_nmse": _nmse(payload, "id_test"),
        "ood_nmse": _nmse(payload, "ood_test"),
    }


def _normalize_dataset_path(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text.rstrip("/")


def _parse_timestamp(value: object, *, context: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise SnapshotIdentityError(f"{context} 不是合法时间戳")
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if math.isfinite(timestamp):
            return timestamp
        raise SnapshotIdentityError(f"{context} 不是有限时间戳")
    text = str(value).strip()
    try:
        timestamp = float(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SnapshotIdentityError(f"{context} 不是合法 ISO 时间戳: {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    if not math.isfinite(timestamp):
        raise SnapshotIdentityError(f"{context} 不是有限时间戳")
    return timestamp


def _expected_global_index(task: Mapping[str, Any]) -> int | None:
    for key in ("task_global_index", "global_index"):
        value = task.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise SnapshotIdentityError(f"task.{key} 不是合法整数: {value!r}") from exc
    match = re.search(r"_g(?P<index>\d{4})$", str(task.get("task_id") or ""))
    return int(match.group("index")) if match else None


def _validate_payload_identity(
    task: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    path: Path,
    mtime_ns: int | None,
    record_kind: str,
) -> dict[str, Any]:
    checked: list[str] = []
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []

    def compare(
        field: str,
        actual: object,
        expected: object,
        normalize=lambda value: value,
    ) -> None:
        if expected in (None, ""):
            return
        if actual in (None, ""):
            missing.append(field)
            return
        checked.append(field)
        try:
            actual_normalized = normalize(actual)
            expected_normalized = normalize(expected)
        except (TypeError, ValueError):
            actual_normalized = actual
            expected_normalized = expected
        if actual_normalized != expected_normalized:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_normalized,
                    "actual": actual_normalized,
                }
            )

    compare(
        "tool",
        payload.get("tool"),
        task.get("algorithm"),
        lambda value: str(value).strip().lower(),
    )
    compare(
        "dataset",
        payload.get("dataset"),
        task.get("dataset_id"),
        lambda value: str(value).strip(),
    )
    compare("seed", payload.get("seed"), task.get("seed"), int)
    compare(
        "task_global_index",
        payload.get("task_global_index"),
        _expected_global_index(task),
        int,
    )
    expected_dataset_dir = (
        task.get("expected_dataset_rel")
        or task.get("expected_dataset_dir")
        or task.get("dataset_dir")
    )
    compare(
        "dataset_dir",
        payload.get("dataset_dir"),
        expected_dataset_dir,
        _normalize_dataset_path,
    )
    identity_check = payload.get("dataset_identity_check")
    if isinstance(identity_check, Mapping) and identity_check.get("match") is False:
        mismatches.append(
            {
                "field": "dataset_identity_check.match",
                "expected": True,
                "actual": False,
            }
        )

    assignment_started_at = _parse_timestamp(
        task.get("assignment_started_at") or task.get("started_at"),
        context="task.started_at",
    )
    payload_started_at = _parse_timestamp(
        payload.get("run_started_at") or payload.get("started_at"),
        context=f"{record_kind}.started_at",
    )
    if assignment_started_at is not None:
        if payload_started_at is not None:
            checked.append("started_at")
            if payload_started_at < assignment_started_at:
                mismatches.append(
                    {
                        "field": "started_at",
                        "expected_minimum": assignment_started_at,
                        "actual": payload_started_at,
                    }
                )
        if mtime_ns is None:
            missing.append("mtime_ns")
        else:
            checked.append("mtime_ns")
            mtime_seconds = mtime_ns / 1_000_000_000
            if mtime_seconds < assignment_started_at:
                mismatches.append(
                    {
                        "field": "mtime_ns",
                        "expected_minimum": assignment_started_at,
                        "actual": mtime_seconds,
                    }
                )

    if mismatches:
        raise SnapshotIdentityError(
            f"{task.get('task_id')} {record_kind} 身份或运行代际不一致: "
            f"{_canonical_json(mismatches)}"
        )
    task_id = str(task.get("task_id") or "").strip()
    path_bound = bool(task_id and f"/tasks/{task_id}/" in path.as_posix())
    return {
        "status": "match" if not missing else "partial",
        "checked_fields": sorted(set(checked)),
        "missing_fields": sorted(set(missing)),
        "task_path_bound": path_bound,
    }


def _strip_payload(record: dict[str, Any], *, freeze_raw: bool) -> dict[str, Any]:
    payload = record.pop("payload", None)
    if isinstance(payload, Mapping):
        record["payload_summary"] = _compact_payload(payload)
        if freeze_raw and "raw_text" not in record:
            record["raw_text"] = _canonical_json(payload)
    return record


def _scan_snapshot_pair(
    outer_path: Path,
    inner_path: Path | None,
    *,
    minute: int,
    freeze_raw: bool,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    outer = _read_json(outer_path, freeze_raw=freeze_raw)
    inner = (
        _read_json(inner_path, freeze_raw=freeze_raw)
        if inner_path is not None and inner_path != outer_path
        else None
    )
    available = [record for record in (outer, inner) if record and record.get("status") == "ok"]
    result: dict[str, Any] = {
        "minute": minute,
        "outer_path": str(outer_path),
        "inner_path": str(inner_path) if inner_path is not None else None,
        "outer_status": outer.get("status"),
        "inner_status": inner.get("status") if inner else None,
        "outer_sha256": outer.get("sha256"),
        "inner_sha256": inner.get("sha256") if inner else None,
        "duplicate_semantically_equal": False,
        "conflict": False,
    }
    if not available:
        statuses = {str(record.get("status")) for record in (outer, inner) if record}
        result["status"] = "parse_error" if "parse_error" in statuses else "missing"
        result["selected_path"] = None
        return result

    for record in available:
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            record["identity_check"] = _validate_payload_identity(
                task,
                payload,
                path=Path(str(record["path"])),
                mtime_ns=record.get("mtime_ns"),
                record_kind=f"minute_{minute:04d}",
            )

    if len(available) == 2:
        if available[0].get("semantic_sha256") != available[1].get("semantic_sha256"):
            result.update(status="conflict", conflict=True, selected_path=None)
            return result
        result["duplicate_semantically_equal"] = True

    selected = outer if outer.get("status") == "ok" else available[0]
    payload = selected.get("payload")
    result.update(
        status="ok",
        selected_path=selected.get("path"),
        selected_sha256=selected.get("sha256"),
        selected_semantic_sha256=selected.get("semantic_sha256"),
        selected_mtime_ns=selected.get("mtime_ns"),
        identity_check=selected.get("identity_check"),
    )
    if isinstance(payload, Mapping):
        result.update(_compact_payload(payload))
    if freeze_raw:
        result["raw_text"] = selected.get("raw_text")
    return result


def scan_task(
    task: Mapping[str, Any],
    *,
    horizon: int = 180,
    freeze_raw: bool = False,
    verify_inner: bool = True,
) -> dict[str, Any]:
    """扫描一个 selected run；任何冲突只报告，不做静默选择。"""

    result_path = Path(str(task["path"]))
    result_record = _read_json(result_path, freeze_raw=freeze_raw)
    result_payload = result_record.get("payload")
    inner_progress: Path | None = None
    if isinstance(result_payload, Mapping):
        result_record["identity_check"] = _validate_payload_identity(
            task,
            result_payload,
            path=result_path,
            mtime_ns=result_record.get("mtime_ns"),
            record_kind="result",
        )
        local_inner_progress = task.get("inner_progress_dir")
        if isinstance(local_inner_progress, str) and local_inner_progress.strip():
            inner_progress = Path(local_inner_progress)
            if not inner_progress.is_dir():
                raise FileNotFoundError(f"本地 inner progress 目录不存在: {inner_progress}")
        else:
            experiment_dir = result_payload.get("experiment_dir")
            if isinstance(experiment_dir, str) and experiment_dir.strip():
                inner_progress = Path(experiment_dir) / "progress"
    outer_progress = result_path.parent / "progress"

    snapshots: list[dict[str, Any]] = []
    for minute in range(1, horizon + 1):
        filename = f"minute_{minute:04d}.json"
        snapshots.append(
            _scan_snapshot_pair(
                outer_progress / filename,
                inner_progress / filename
                if verify_inner and inner_progress is not None
                else None,
                minute=minute,
                freeze_raw=freeze_raw,
                task=task,
            )
        )

    missing = sum(item["status"] == "missing" for item in snapshots)
    conflicts = sum(bool(item["conflict"]) for item in snapshots)
    parse_errors = sum(item["status"] == "parse_error" for item in snapshots)
    available = sum(item["status"] == "ok" for item in snapshots)
    source = {
        key: task.get(key)
        for key in (
            "batch",
            "algorithm",
            "host",
            "seed",
            "noise_tag",
            "task_id",
            "dataset_id",
            "global_index",
            "dataset_dir",
            "expected_dataset_rel",
            "expected_dataset_dir",
            "started_at",
            "assignment_started_at",
            "execution_set",
            "run_id",
            "path",
        )
    }
    source["source_row_sha256"] = _sha256(_canonical_json(source).encode("utf-8"))
    return {
        "source": source,
        "result": _strip_payload(result_record, freeze_raw=freeze_raw),
        "outer_progress_dir": str(outer_progress),
        "inner_progress_dir": str(inner_progress) if inner_progress is not None else None,
        "snapshots": snapshots,
        "summary": {
            "expected_snapshots": horizon,
            "available_snapshots": available,
            "missing_snapshots": missing,
            "conflicting_snapshots": conflicts,
            "parse_errors": parse_errors,
        },
    }


def _open_text(path: Path, mode: str) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _load_tasks(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path, "r") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict):
                raise TypeError(f"任务清单第 {line_number} 行不是 JSON object")
            yield task


def main() -> None:
    parser = argparse.ArgumentParser(description="只读扫描并冻结 Stage5 selected runs")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--freeze-raw", action="store_true")
    parser.add_argument(
        "--outer-only",
        action="store_true",
        help="只冻结 selected outer；仅可在独立 inner 一致性预检通过后使用",
    )
    args = parser.parse_args()

    totals = {
        "tasks": 0,
        "result_missing_or_invalid": 0,
        "expected_snapshots": 0,
        "available_snapshots": 0,
        "missing_snapshots": 0,
        "conflicting_snapshots": 0,
        "parse_errors": 0,
        "freeze_raw": bool(args.freeze_raw),
        "verify_inner": not bool(args.outer_only),
    }
    with _open_text(args.output, "w") as output:
        for task in _load_tasks(args.tasks):
            record = scan_task(
                task,
                horizon=args.horizon,
                freeze_raw=args.freeze_raw,
                verify_inner=not args.outer_only,
            )
            output.write(_canonical_json(record) + "\n")
            totals["tasks"] += 1
            if record["result"].get("status") != "ok":
                totals["result_missing_or_invalid"] += 1
            for key in (
                "expected_snapshots",
                "available_snapshots",
                "missing_snapshots",
                "conflicting_snapshots",
                "parse_errors",
            ):
                totals[key] += int(record["summary"][key])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(totals, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
