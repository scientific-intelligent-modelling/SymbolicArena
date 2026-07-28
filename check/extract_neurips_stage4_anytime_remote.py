#!/usr/bin/env python3
"""从远端 Stage-4 原始实验目录提取 clean best-so-far 快照。

输入 JSONL 由本地归档生成，每行对应一个 algorithm × dataset × seed。
脚本只读取该行明确记录的 ``experiment_dir``，避免在重复实验目录中误配。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any


def finite_nmse(payload: dict[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    try:
        value = float(block.get("nmse"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def canonical_expression(payload: dict[str, Any]) -> str:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, dict):
        artifact = {}
    candidates = (
        artifact.get("normalized_expression"),
        artifact.get("instantiated_expression"),
        artifact.get("return_expression_source"),
        payload.get("equation"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def usable_candidate(payload: dict[str, Any]) -> bool:
    return (
        bool(canonical_expression(payload))
        and finite_nmse(payload, "id_test") is not None
        and finite_nmse(payload, "ood_test") is not None
    )


def is_future_backfill(payload: dict[str, Any], minute: int) -> bool:
    if payload.get("record_type") != "periodic_backfill":
        return False
    try:
        source_minute = int(payload.get("backfilled_from_minute"))
    except (TypeError, ValueError):
        return True
    return source_minute > minute


def checkpoint_record(
    task: dict[str, Any],
    minute: int,
    payload: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    payload = payload or {}
    id_nmse = finite_nmse(payload, "id_test")
    ood_nmse = finite_nmse(payload, "ood_test")
    valid = bool(
        payload
        and id_nmse is not None
        and ood_nmse is not None
        and canonical_expression(payload)
    )
    return {
        "algorithm": task["algorithm"],
        "gid": task["gid"],
        "dataset": task["dataset"],
        "seed": int(task["seed"]),
        "host": task["host"],
        "minute": minute,
        "source": source,
        "record_type": payload.get("record_type"),
        "snapshot_elapsed_minutes": payload.get("elapsed_minutes"),
        "valid_output": valid,
        "metric_complete": id_nmse is not None and ood_nmse is not None,
        "id_test_nmse": id_nmse,
        "ood_test_nmse": ood_nmse,
        "expression_canonical": canonical_expression(payload) if valid else "",
        "status": payload.get("status") or ("ok" if valid else "no_valid_output"),
    }


def extract_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_dir = Path(task["experiment_dir"])
    progress_dir = experiment_dir / "progress"
    final_payload = task.get("final_payload")
    if not isinstance(final_payload, dict):
        final_payload = {}
    try:
        final_seconds = float(task.get("final_seconds"))
    except (TypeError, ValueError):
        final_seconds = math.inf

    latest: dict[str, Any] | None = None
    latest_source = "missing"
    rows: list[dict[str, Any]] = []
    for minute in range(1, 61):
        path = progress_dir / f"minute_{minute:04d}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if (
                isinstance(payload, dict)
                and not is_future_backfill(payload, minute)
                and usable_candidate(payload)
            ):
                latest = payload
                latest_source = f"snapshot:{minute}"

        if final_seconds <= 60.0 * minute and usable_candidate(final_payload):
            selected = final_payload
            source = "final_carry_forward"
        else:
            selected = latest
            source = latest_source
        rows.append(checkpoint_record(task, minute, selected, source))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl-gz", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_jsonl_gz, "wt", encoding="utf-8") as handle:
        for task in tasks:
            for row in extract_task(task):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
