from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from models import (
    STAGE1_PROGRESS_INTERVAL_SECONDS,
    STAGE1_SEED,
    STAGE1_TIMEOUT_SECONDS,
)


def _read_tool_mapping(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("tool_mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"toolbox config missing tool_mapping: {path}")
    return {
        str(name): {
            "env": str(config.get("env", "")),
            "regressor": str(config.get("regressor", "")),
        }
        for name, config in sorted(mapping.items())
    }


def _read_datasets(ssr50_root: Path) -> list[dict[str, str]]:
    datasets: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for metadata in sorted(ssr50_root.rglob("metadata.yaml")):
        dataset_dir = metadata.parent
        dataset_id = dataset_dir.name
        if dataset_id in seen_ids:
            raise ValueError(f"duplicate SSR50 dataset_id {dataset_id!r} under {ssr50_root}")
        seen_ids.add(dataset_id)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_dir": str(dataset_dir),
            }
        )
    if len(datasets) != 50:
        raise ValueError(f"expected 50 SSR50 datasets, found {len(datasets)} under {ssr50_root}")
    return datasets


def _dataset_output_path(dataset_dir: str, dataset_dir_base: Path | None) -> str:
    if dataset_dir_base is None:
        return dataset_dir
    return str(Path(dataset_dir).relative_to(dataset_dir_base))


def generate_manifest(
    *,
    toolbox_config_path: Path,
    ssr50_root: Path,
    batch_dir: Path,
    git_revision: str,
    dataset_dir_base: Path | None = None,
) -> dict[str, int]:
    algorithms = _read_tool_mapping(toolbox_config_path)
    if len(algorithms) != 15:
        raise ValueError(f"expected 15 algorithms, found {len(algorithms)}")
    datasets = _read_datasets(ssr50_root)
    manifest_dir = batch_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = [
        {
            "dataset_id": dataset["dataset_id"],
            "dataset_dir": _dataset_output_path(dataset["dataset_dir"], dataset_dir_base),
        }
        for dataset in datasets
    ]

    (manifest_dir / "algorithms.json").write_text(
        json.dumps(algorithms, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "datasets.csv").write_text(
        _to_csv(["dataset_id", "dataset_dir"], dataset_rows),
        encoding="utf-8",
    )
    budget = {
        "seed": STAGE1_SEED,
        "timeout_in_seconds": STAGE1_TIMEOUT_SECONDS,
        "progress_snapshot_interval_seconds": STAGE1_PROGRESS_INTERVAL_SECONDS,
    }
    (manifest_dir / "budget.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "git_revision.txt").write_text(git_revision.strip() + "\n", encoding="utf-8")

    tasks: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for dataset in datasets:
            dataset_id = dataset["dataset_id"]
            tasks.append(
                {
                    "task_id": f"{algorithm}__seed{STAGE1_SEED}__{dataset_id}",
                    "algorithm": algorithm,
                    "dataset_id": dataset_id,
                    "dataset_dir": _dataset_output_path(dataset["dataset_dir"], dataset_dir_base),
                    "seed": STAGE1_SEED,
                    "timeout_in_seconds": STAGE1_TIMEOUT_SECONDS,
                    "progress_snapshot_interval_seconds": STAGE1_PROGRESS_INTERVAL_SECONDS,
                }
            )
    (manifest_dir / "tasks.csv").write_text(
        _to_csv(
            [
                "task_id",
                "algorithm",
                "dataset_id",
                "dataset_dir",
                "seed",
                "timeout_in_seconds",
                "progress_snapshot_interval_seconds",
            ],
            tasks,
        ),
        encoding="utf-8",
    )
    return {
        "total_algorithms": len(algorithms),
        "total_datasets": len(datasets),
        "total_tasks": len(tasks),
    }


def _to_csv(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
