from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_DEPLOY_SCRIPTS = [
    "00_sync_code_and_batch_to_iaaccn22.sh",
    "01_preflight_from_iaaccn22.sh",
    "02_smoke_dispatch_from_iaaccn22.sh",
    "03_full_dispatch_from_iaaccn22.sh",
]


def check_readiness(*, batch_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    tasks = _read_csv_if_exists(batch_dir / "manifest" / "tasks.csv", issues, "manifest/tasks.csv")
    datasets = _read_csv_if_exists(batch_dir / "manifest" / "datasets.csv", issues, "manifest/datasets.csv")
    algorithms = _read_algorithms(batch_dir / "manifest" / "algorithms.json", issues)
    ssr50_rows = _read_csv_if_exists(batch_dir / "queues" / "ssr50_source.csv", issues, "queues/ssr50_source.csv")
    smoke_rows = _read_csv_if_exists(
        batch_dir / "queues" / "smoke_2datasets_source.csv",
        issues,
        "queues/smoke_2datasets_source.csv",
    )

    _expect_count("manifest/tasks.csv", tasks, 750, issues)
    _expect_count("manifest/datasets.csv", datasets, 50, issues)
    _expect_count("manifest/algorithms.json", algorithms, 15, issues)
    _expect_count("queues/ssr50_source.csv", ssr50_rows, 50, issues)
    _expect_count("queues/smoke_2datasets_source.csv", smoke_rows, 2, issues)
    _check_task_contract(tasks, issues)
    _check_deploy_scripts(batch_dir, issues)

    summary = {
        "ready": not issues,
        "total_tasks": len(tasks),
        "total_datasets": len(datasets),
        "total_algorithms": len(algorithms),
        "ssr50_queue_rows": len(ssr50_rows),
        "smoke_queue_rows": len(smoke_rows),
        "issues": issues,
    }
    readiness_dir = batch_dir / "readiness"
    readiness_dir.mkdir(parents=True, exist_ok=True)
    (readiness_dir / "readiness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_csv_if_exists(path: Path, issues: list[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        issues.append(f"{label} missing")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_algorithms(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append("manifest/algorithms.json missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"manifest/algorithms.json invalid json: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append("manifest/algorithms.json must be an object")
        return {}
    return payload


def _expect_count(label: str, rows: object, expected: int, issues: list[str]) -> None:
    actual = len(rows) if hasattr(rows, "__len__") else 0
    if actual != expected:
        issues.append(f"{label} expected {expected}, got {actual}")


def _check_task_contract(tasks: list[dict[str, str]], issues: list[str]) -> None:
    seeds = {row.get("seed") for row in tasks}
    timeouts = {row.get("timeout_in_seconds") for row in tasks}
    intervals = {row.get("progress_snapshot_interval_seconds") for row in tasks}
    if tasks and seeds != {"520"}:
        issues.append(f"manifest/tasks.csv seed set must be {{520}}, got {sorted(seeds)}")
    if tasks and timeouts != {"3600"}:
        issues.append(f"manifest/tasks.csv timeout must be 3600, got {sorted(timeouts)}")
    if tasks and intervals != {"60"}:
        issues.append(f"manifest/tasks.csv progress interval must be 60, got {sorted(intervals)}")
    task_ids = [row.get("task_id", "") for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        issues.append("manifest/tasks.csv contains duplicate task_id")


def _check_deploy_scripts(batch_dir: Path, issues: list[str]) -> None:
    for script in REQUIRED_DEPLOY_SCRIPTS:
        path = batch_dir / "deploy" / script
        if not path.exists():
            issues.append(f"deploy/{script} missing")
            continue
        content = path.read_text(encoding="utf-8")
        if script.startswith("00_sync"):
            if "rsync -aR" not in content:
                issues.append(f"deploy/{script} does not call rsync")
        elif "run_e1_candidate200_12alg_load_queue.py" not in content:
            issues.append(f"deploy/{script} does not call load queue scheduler")
