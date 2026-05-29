from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import HEARTBEAT_PHASES


def write_heartbeat(*, batch_dir: Path, phase: str) -> dict[str, Any]:
    if phase not in HEARTBEAT_PHASES:
        raise ValueError(f"invalid heartbeat phase: {phase}")

    failures = _read_csv(batch_dir / "audit" / "failure_cases.csv")
    task_rows = _read_csv(batch_dir / "manifest" / "tasks.csv")
    payload = {
        "batch_id": batch_dir.name,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(task_rows),
        "pending": 0,
        "running": 0,
        "finished": max(0, len(task_rows) - len(failures)) if task_rows else 0,
        "failed": len(failures),
        "stale": 0,
        "needs_codex": bool(failures),
        "codex_reason": _codex_reason(failures),
        "latest_audit": "audit/failure_cases.csv",
        "latest_rerun_queue": "",
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


def _codex_reason(failures: list[dict[str, str]]) -> str:
    if not failures:
        return ""

    counts: dict[str, int] = {}
    for row in failures:
        key = row.get("failure_class", "unknown") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
