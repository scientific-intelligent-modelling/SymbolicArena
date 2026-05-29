from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any


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
