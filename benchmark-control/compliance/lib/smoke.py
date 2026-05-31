from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

SMOKE_MIN_RUNTIME_RATIO = 0.9


def prepare_smoke_batch(*, batch_dir: Path) -> dict[str, int]:
    smoke_dir = batch_dir / "smoke"
    smoke_queue_path = batch_dir / "queues" / "smoke_2datasets_source.csv"
    smoke_rows = _read_csv(smoke_queue_path)
    smoke_dataset_ids = {row["dataset_id"] for row in smoke_rows}
    tasks = [
        row
        for row in _read_csv(batch_dir / "manifest" / "tasks.csv")
        if row.get("dataset_id") in smoke_dataset_ids
    ]
    tasks = _with_smoke_budget(tasks, params_smoke_dir=batch_dir / "params_smoke")
    algorithms = {row["algorithm"] for row in tasks}

    manifest_dir = smoke_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(manifest_dir / "tasks.csv", tasks)
    _copy_if_exists(batch_dir / "manifest" / "algorithms.json", manifest_dir / "algorithms.json")
    _copy_if_exists(batch_dir / "manifest" / "budget.json", manifest_dir / "budget.json")
    _copy_if_exists(batch_dir / "manifest" / "git_revision.txt", manifest_dir / "git_revision.txt")
    _write_csv(
        manifest_dir / "datasets.csv",
        [
            {"dataset_id": row["dataset_id"], "dataset_dir": row["dataset_dir"]}
            for row in smoke_rows
        ],
    )

    queue_dir = smoke_dir / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smoke_queue_path, queue_dir / "smoke_2datasets_source.csv")

    return {
        "total_tasks": len(tasks),
        "total_datasets": len(smoke_rows),
        "total_algorithms": len(algorithms),
    }


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def _with_smoke_budget(tasks: list[dict[str, str]], *, params_smoke_dir: Path) -> list[dict[str, str]]:
    if not params_smoke_dir.exists():
        return tasks
    smoke_timeouts = _read_smoke_timeouts(params_smoke_dir)
    if not smoke_timeouts:
        return tasks
    uniform_timeout = next(iter(set(smoke_timeouts.values()))) if len(set(smoke_timeouts.values())) == 1 else None
    updated: list[dict[str, str]] = []
    for task in tasks:
        params_key = f"{_normalize_params_name(task.get('algorithm', ''))}__{task.get('noise_tag') or 'clean'}"
        timeout = smoke_timeouts.get(params_key, uniform_timeout)
        if timeout is None:
            updated.append(task)
            continue
        row = dict(task)
        row["timeout_in_seconds"] = str(timeout)
        row["min_runtime_seconds"] = str(_smoke_min_runtime_seconds(timeout))
        updated.append(row)
    return updated


def _read_smoke_timeouts(params_smoke_dir: Path) -> dict[str, int]:
    timeouts: dict[str, int] = {}
    for path in sorted(params_smoke_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_timeout = payload.get("timeout_in_seconds")
        if raw_timeout is None:
            continue
        timeouts[path.stem] = int(raw_timeout)
    return timeouts


def _normalize_params_name(name: str) -> str:
    return name.strip().lower()


def _smoke_min_runtime_seconds(timeout_in_seconds: int) -> int:
    return max(1, int(timeout_in_seconds * SMOKE_MIN_RUNTIME_RATIO))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
