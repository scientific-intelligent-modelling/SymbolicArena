from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def check_audit_success(
    *,
    batch_dir: Path,
    expected_total_tasks: int | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    tasks = _read_csv(batch_dir / "manifest" / "tasks.csv", issues, "manifest/tasks.csv")
    task_audit = _read_csv(batch_dir / "audit" / "task_audit.csv", issues, "audit/task_audit.csv")
    failures = _read_csv(batch_dir / "audit" / "failure_cases.csv", issues, "audit/failure_cases.csv")
    budget_rows = _read_csv(
        batch_dir / "audit" / "budget_compliance_summary.csv",
        issues,
        "audit/budget_compliance_summary.csv",
    )
    expected = expected_total_tasks if expected_total_tasks is not None else len(tasks)
    failure_rows = [row for row in failures if any(str(value).strip() for value in row.values())]
    if len(tasks) != expected:
        issues.append(f"manifest/tasks.csv expected {expected}, got {len(tasks)}")
    if len(task_audit) != expected:
        issues.append(f"audit/task_audit.csv expected {expected}, got {len(task_audit)}")
    if failure_rows:
        issues.append(f"audit/failure_cases.csv has {len(failure_rows)} failure rows")
    for row in task_audit:
        if row.get("failure_class"):
            issues.append(f"audit/task_audit.csv task {row.get('task_id')} failed: {row.get('failure_class')}")
    for row in budget_rows:
        if str(row.get("compliant", "")).lower() != "true":
            issues.append(f"audit/budget_compliance_summary.csv algorithm {row.get('algorithm')} not compliant")

    summary = {
        "audit_passed": not issues,
        "expected_total_tasks": expected,
        "task_audit_rows": len(task_audit),
        "failure_rows": len(failure_rows),
        "issues": issues,
    }
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_csv(path: Path, issues: list[str], label: str) -> list[dict[str, str]]:
    if not path.exists():
        issues.append(f"{label} missing")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
