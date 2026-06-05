from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECOVERY_FIELDS = [
    "task_id",
    "scheduler_task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "noise_sigma",
    "status",
    "source_path",
    "source_host",
    "target_dir",
    "reason",
]


def recover_snapshots(
    *,
    batch_dir: Path,
    source_roots: list[Path],
    source_batch: str,
    snapshot_name: str = "minute_0180.json",
) -> dict[str, int]:
    tasks = _read_csv(batch_dir / "manifest" / "tasks.csv")
    queue_index = _read_queue_index(batch_dir / "queues")
    recovery_dir = batch_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    recovered_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for task in tasks:
        scheduler_task_id = _scheduler_task_id(task, queue_index)
        candidates = _find_snapshot_candidates(
            source_roots=source_roots,
            task=task,
            scheduler_task_id=scheduler_task_id,
            snapshot_name=snapshot_name,
        )
        base_row = _row(task, scheduler_task_id=scheduler_task_id)
        if not candidates:
            missing_rows.append({**base_row, "status": "missing", "reason": "snapshot not found"})
            continue

        snapshot_path = _choose_candidate(candidates)
        payload = _read_json(snapshot_path)
        valid, reason = _validate_snapshot(payload)
        if not valid:
            invalid_row = {
                **base_row,
                "status": "invalid",
                "source_path": str(snapshot_path),
                "source_host": _source_host(snapshot_path, scheduler_task_id),
                "reason": reason,
            }
            invalid_rows.append(invalid_row)
            missing_rows.append(invalid_row)
            continue

        target_dir = _target_dir(batch_dir, task)
        _write_recovered_result(
            task=task,
            source_batch=source_batch,
            snapshot_path=snapshot_path,
            payload=payload,
            target_dir=target_dir,
        )
        recovered_rows.append(
            {
                **base_row,
                "status": "recovered",
                "source_path": str(snapshot_path),
                "source_host": _source_host(snapshot_path, scheduler_task_id),
                "target_dir": str(target_dir),
            }
        )

    _write_csv(recovery_dir / "recovered_tasks.csv", recovered_rows)
    _write_csv(recovery_dir / "missing_tasks.csv", missing_rows)
    _write_csv(recovery_dir / "invalid_snapshot_tasks.csv", invalid_rows)
    summary = {
        "total_tasks": len(tasks),
        "recovered": len(recovered_rows),
        "missing": len(missing_rows),
        "invalid": len(invalid_rows),
    }
    (recovery_dir / "recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_queue_index(queue_dir: Path) -> dict[str, dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in (queue_dir / "ssr50_source.csv", queue_dir / "smoke_2datasets_source.csv"):
        if path.exists():
            rows.extend(_read_csv(path))
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset_id = row.get("dataset_id") or row.get("dataset_name")
        if dataset_id:
            index.setdefault(dataset_id, row)
    return index


def _scheduler_task_id(task: dict[str, str], queue_index: dict[str, dict[str, str]]) -> str:
    dataset_id = task["dataset_id"]
    queue_row = queue_index.get(dataset_id)
    if queue_row is None:
        raise ValueError(f"dataset missing from queue csv: {dataset_id}")
    global_index = int(queue_row["global_index"])
    tool_key = _scheduler_tool_key(task["algorithm"])
    noise_tag = task.get("noise_tag") or "clean"
    return f"{tool_key}_s{task['seed']}_{noise_tag}_g{global_index:04d}"


def _scheduler_tool_key(algorithm: str) -> str:
    return {
        "QLattice": "qlattice",
        "iMCTS": "imcts",
    }.get(algorithm, algorithm.lower())


def _find_snapshot_candidates(
    *,
    source_roots: list[Path],
    task: dict[str, str],
    scheduler_task_id: str,
    snapshot_name: str,
) -> list[Path]:
    tool_key = _scheduler_tool_key(task["algorithm"])
    seed_dir = f"seed{task['seed']}"
    candidates: list[Path] = []
    for source_root in source_roots:
        direct_task_dir = source_root / tool_key / seed_dir / "tasks" / scheduler_task_id
        task_dirs = [direct_task_dir] if direct_task_dir.exists() else []
        task_dirs.extend(
            path
            for path in source_root.glob(f"**/{tool_key}/{seed_dir}/tasks/{scheduler_task_id}")
            if path not in task_dirs
        )
        for task_dir in task_dirs:
            candidates.extend(task_dir.glob(f"**/progress/{snapshot_name}"))
    return candidates


def _choose_candidate(candidates: list[Path]) -> Path:
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot must be a JSON object: {path}")
    return payload


def _validate_snapshot(payload: dict[str, Any]) -> tuple[bool, str]:
    status = str(payload.get("status", "")).strip().lower()
    if status != "ok":
        return False, f"snapshot status is {status or 'missing'}"
    runtime = _runtime_seconds(payload)
    if runtime <= 0.0:
        return False, "snapshot runtime missing"
    if not _metrics_valid(payload):
        return False, "snapshot metrics invalid"
    if not _artifact_valid(payload):
        return False, "snapshot artifact invalid"
    return True, ""


def _runtime_seconds(payload: dict[str, Any]) -> float:
    value = payload.get("runtime_seconds", payload.get("seconds", payload.get("elapsed_seconds", 0.0)))
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric_value if math.isfinite(numeric_value) else 0.0


def _metrics_valid(payload: dict[str, Any]) -> bool:
    for split in ("valid", "id_test", "ood_test"):
        metrics = payload.get(split)
        if not isinstance(metrics, dict):
            return False
        nmse = metrics.get("nmse")
        if not isinstance(nmse, (int, float)) or isinstance(nmse, bool):
            return False
        if not math.isfinite(float(nmse)):
            return False
    return True


def _artifact_valid(payload: dict[str, Any]) -> bool:
    artifact = payload.get("canonical_artifact")
    if isinstance(artifact, dict) and any(value for value in artifact.values()):
        return True
    equation = payload.get("equation")
    return isinstance(equation, str) and bool(equation.strip())


def _target_dir(batch_dir: Path, task: dict[str, str]) -> Path:
    noise_tag = task.get("noise_tag") or "clean"
    return batch_dir / "runs" / task["algorithm"] / f"seed{task['seed']}" / noise_tag / task["dataset_id"]


def _write_recovered_result(
    *,
    task: dict[str, str],
    source_batch: str,
    snapshot_path: Path,
    payload: dict[str, Any],
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_host = _source_host(snapshot_path, snapshot_path.parent.parent.parent.parent.name)
    runtime = _runtime_seconds(payload)
    result_payload = dict(payload)
    result_payload.setdefault("runtime_seconds", runtime)
    result_payload["recovered_from_24h"] = True
    result_payload["source_batch"] = source_batch
    result_payload["source_host"] = source_host
    result_payload["source_path"] = str(snapshot_path)
    result_payload["recovered_at"] = datetime.now(timezone.utc).isoformat()
    result_payload["formal3h_task_id"] = task["task_id"]
    (target_dir / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress_dir = target_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    shutil.copy2(snapshot_path, progress_dir / snapshot_path.name)
    recovery_source = {
        "recovered_from_24h": True,
        "source_batch": source_batch,
        "source_host": source_host,
        "source_path": str(snapshot_path),
        "elapsed_seconds": runtime,
        "recovered_at": result_payload["recovered_at"],
    }
    (target_dir / "recovery_source.json").write_text(
        json.dumps(recovery_source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_host(path: Path, scheduler_task_id: str) -> str:
    parts = path.parts
    if scheduler_task_id in parts:
        task_index = parts.index(scheduler_task_id)
        if task_index + 1 < len(parts):
            return parts[task_index + 1]
    for part in parts:
        if part.startswith("iaaccn"):
            return part
    return ""


def _row(task: dict[str, str], *, scheduler_task_id: str) -> dict[str, str]:
    return {
        "task_id": task["task_id"],
        "scheduler_task_id": scheduler_task_id,
        "algorithm": task["algorithm"],
        "dataset_id": task["dataset_id"],
        "seed": task["seed"],
        "noise_tag": task.get("noise_tag") or "clean",
        "noise_sigma": task.get("noise_sigma") or "0.0",
        "status": "",
        "source_path": "",
        "source_host": "",
        "target_dir": "",
        "reason": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOVERY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
