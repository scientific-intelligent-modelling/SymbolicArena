#!/usr/bin/env python3
"""在结果所在主机上只读扫描并冻结 Stage 4 最终结果与分钟轨迹。

该文件只依赖 Python 标准库，可以单独复制到远端 ``/tmp`` 执行。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


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
        raw = path.read_bytes()
    except OSError as exc:
        record.update(
            status="read_error",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return record
    record.update(size_bytes=len(raw), sha256=_sha256(raw))
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
        "has_expression": bool(_expression(payload)),
        "expression": _expression(payload),
        "id_nmse": _nmse(payload, "id_test"),
        "ood_nmse": _nmse(payload, "ood_test"),
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
) -> dict[str, Any]:
    """扫描一个 selected run；任何冲突只报告，不做静默选择。"""

    result_path = Path(str(task["path"]))
    result_record = _read_json(result_path, freeze_raw=freeze_raw)
    result_payload = result_record.get("payload")
    inner_progress: Path | None = None
    if isinstance(result_payload, Mapping):
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
                inner_progress / filename if inner_progress is not None else None,
                minute=minute,
                freeze_raw=freeze_raw,
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
    }
    with _open_text(args.output, "w") as output:
        for task in _load_tasks(args.tasks):
            record = scan_task(task, horizon=args.horizon, freeze_raw=args.freeze_raw)
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
