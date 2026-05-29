from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import HEARTBEAT_PHASES


def write_heartbeat(
    *,
    batch_dir: Path,
    phase: str,
    latest_rerun_queue: str = "",
) -> dict[str, Any]:
    if phase not in HEARTBEAT_PHASES:
        raise ValueError(f"invalid heartbeat phase: {phase}")

    failures_path = batch_dir / "audit" / "failure_cases.csv"
    task_audit_path = batch_dir / "audit" / "task_audit.csv"
    failures = _read_csv(failures_path)
    task_rows = _read_csv(batch_dir / "manifest" / "tasks.csv")
    task_audit_rows = _read_csv(task_audit_path)
    audit_exists = failures_path.exists()
    total_tasks = len(task_rows)
    audit_complete = _audit_complete(total_tasks=total_tasks, audit_rows=len(task_audit_rows))
    finished = _finished_count(
        total_tasks=total_tasks,
        failed=len(failures),
        audit_exists=audit_exists,
        audit_complete=audit_complete,
    )
    needs_codex = (not audit_exists) or (not audit_complete) or bool(failures)
    payload = {
        "batch_id": batch_dir.resolve().name,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": total_tasks,
        "pending": 0,
        "running": 0,
        "finished": finished,
        "failed": len(failures),
        "stale": 0,
        "needs_codex": needs_codex,
        "codex_reason": _codex_reason(
            failures=failures,
            audit_exists=audit_exists,
            audit_complete=audit_complete,
        ),
        "latest_audit": "audit/failure_cases.csv" if audit_exists else "",
        "latest_rerun_queue": latest_rerun_queue,
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "heartbeat.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_complete(*, total_tasks: int, audit_rows: int) -> bool:
    if total_tasks <= 0:
        return False
    return audit_rows == total_tasks


def _finished_count(*, total_tasks: int, failed: int, audit_exists: bool, audit_complete: bool) -> int:
    if not audit_exists or not audit_complete:
        return 0
    if total_tasks <= 0:
        return 0
    return max(0, total_tasks - failed)


def _codex_reason(*, failures: list[dict[str, str]], audit_exists: bool, audit_complete: bool) -> str:
    if not audit_exists:
        return "audit/failure_cases.csv missing"
    if not audit_complete:
        return "audit/task_audit.csv missing or incomplete"
    if not failures:
        return ""

    counts: dict[str, int] = {}
    for row in failures:
        key = row.get("failure_class", "unknown") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
