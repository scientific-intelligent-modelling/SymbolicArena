from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from models import FAILURE_CLASSES, STAGE1_MIN_RUNTIME_SECONDS


TASK_AUDIT_FIELDS = [
    "task_id",
    "algorithm",
    "dataset_id",
    "seed",
    "status",
    "runtime_seconds",
    "has_result",
    "has_progress",
    "metrics_valid",
    "artifact_valid",
    "failure_class",
    "reason",
]


def audit_batch(*, batch_dir: Path) -> dict[str, int]:
    tasks = _read_tasks(batch_dir / "manifest" / "tasks.csv")
    audit_dir = batch_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows = [_audit_task(batch_dir, task) for task in tasks]
    failures = [row for row in rows if row["failure_class"]]

    _write_csv(audit_dir / "task_audit.csv", TASK_AUDIT_FIELDS, rows)
    _write_csv(audit_dir / "failure_cases.csv", TASK_AUDIT_FIELDS, failures)
    _write_csv(
        audit_dir / "early_stop_cases.csv",
        TASK_AUDIT_FIELDS,
        [row for row in failures if row["failure_class"] == "early_stop"],
    )
    _write_summary(audit_dir / "budget_compliance_summary.csv", rows)
    return {
        "total_tasks": len(rows),
        "failed": len(failures),
        "passed": len(rows) - len(failures),
    }


def _audit_task(batch_dir: Path, task: dict[str, str]) -> dict[str, Any]:
    task_dir = batch_dir / "runs" / task["algorithm"] / f"seed{task['seed']}" / task["dataset_id"]
    result_path = task_dir / "result.json"
    progress_dir = task_dir / "progress"
    has_progress = progress_dir.exists() and any(progress_dir.glob("minute_*.json"))

    if _dispatch_failure_marker(task_dir):
        return _row(
            task=task,
            status="failed",
            runtime=0.0,
            has_result=False,
            has_progress=has_progress,
            metrics_valid=False,
            artifact_valid=False,
            failure_class="dispatch_failure",
            reason="dispatch failure marker found",
        )

    if not result_path.exists():
        return _row(
            task=task,
            status="failed",
            runtime=0.0,
            has_result=False,
            has_progress=has_progress,
            metrics_valid=False,
            artifact_valid=False,
            failure_class="missing_result",
            reason="result.json not found",
        )

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _row(
            task=task,
            status="failed",
            runtime=0.0,
            has_result=True,
            has_progress=has_progress,
            metrics_valid=False,
            artifact_valid=False,
            failure_class="artifact_invalid",
            reason=f"invalid result json: {exc}",
        )

    runtime = _runtime_seconds(result)
    metrics_valid = _metrics_valid(result)
    artifact_valid = _artifact_valid(result)
    raw_status = str(result.get("status", "")).strip().lower()
    failure_class, reason = _classify_failure(
        result=result,
        runtime=runtime,
        has_progress=has_progress,
        metrics_valid=metrics_valid,
        artifact_valid=artifact_valid,
        raw_status=raw_status,
    )

    status = "ok" if not failure_class else "failed"
    return _row(
        task=task,
        status=status,
        runtime=runtime,
        has_result=True,
        has_progress=has_progress,
        metrics_valid=metrics_valid,
        artifact_valid=artifact_valid,
        failure_class=failure_class,
        reason=reason,
    )


def _dispatch_failure_marker(task_dir: Path) -> bool:
    return any(
        (task_dir / filename).exists()
        for filename in ("dispatch_failure.json", "dispatch_failure.txt")
    )


def _classify_failure(
    *,
    result: dict[str, Any],
    runtime: float,
    has_progress: bool,
    metrics_valid: bool,
    artifact_valid: bool,
    raw_status: str,
) -> tuple[str, str]:
    if _is_dispatch_failure_result(result, raw_status):
        return "dispatch_failure", _status_reason(result, fallback="dispatch failure")
    if raw_status == "error":
        return "runtime_crash", _status_reason(result, fallback="runtime error")
    if _is_timeout_unrecovered(result, raw_status):
        return "timeout_unrecovered", _status_reason(result, fallback="timeout without recovered output")
    if runtime < STAGE1_MIN_RUNTIME_SECONDS:
        return "early_stop", f"runtime_seconds {runtime:.3f} < {STAGE1_MIN_RUNTIME_SECONDS}"
    if not has_progress:
        return "missing_progress", "progress/minute_*.json not found"
    if not metrics_valid:
        return "metric_invalid", "valid/id_test/ood_test metrics missing or non-finite"
    if not artifact_valid:
        return "artifact_invalid", "canonical artifact or equation missing"
    if raw_status in {"", "ok"}:
        return "", ""
    return "unknown", _status_reason(result, fallback=f"unclassified status: {raw_status}")


def _is_dispatch_failure_result(result: dict[str, Any], raw_status: str) -> bool:
    if raw_status == "dispatch_failure":
        return True
    termination_reason = str(result.get("termination_reason", "")).strip().lower()
    if termination_reason == "dispatch_failure":
        return True
    error_text = str(result.get("error", "")).strip().lower()
    return "dispatch" in error_text and "fail" in error_text


def _is_timeout_unrecovered(result: dict[str, Any], raw_status: str) -> bool:
    if raw_status != "timed_out":
        return False
    if bool(result.get("recovered_from_timeout")):
        return False
    timeout_type = str(result.get("timeout_type", "")).strip().lower()
    return timeout_type != "budget_exhausted_with_output"


def _status_reason(result: dict[str, Any], *, fallback: str) -> str:
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    termination_reason = result.get("termination_reason")
    if isinstance(termination_reason, str) and termination_reason.strip():
        return termination_reason.strip()
    return fallback


def _runtime_seconds(result: dict[str, Any]) -> float:
    value = result.get("runtime_seconds", result.get("seconds", 0.0))
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric_value if math.isfinite(numeric_value) else 0.0


def _metrics_valid(result: dict[str, Any]) -> bool:
    for split in ("valid", "id_test", "ood_test"):
        metrics = result.get(split)
        if not isinstance(metrics, dict):
            return False
        nmse = metrics.get("nmse")
        if not isinstance(nmse, (int, float)) or isinstance(nmse, bool):
            return False
        if not math.isfinite(float(nmse)):
            return False
    return True


def _artifact_valid(result: dict[str, Any]) -> bool:
    artifact = result.get("canonical_artifact")
    if isinstance(artifact, dict) and any(value for value in artifact.values()):
        return True
    equation = result.get("equation")
    return isinstance(equation, str) and bool(equation.strip())


def _row(
    *,
    task: dict[str, str],
    status: str,
    runtime: float,
    has_result: bool,
    has_progress: bool,
    metrics_valid: bool,
    artifact_valid: bool,
    failure_class: str,
    reason: str,
) -> dict[str, str]:
    if failure_class and failure_class not in FAILURE_CLASSES:
        raise ValueError(f"unsupported failure class: {failure_class}")
    return {
        "task_id": task["task_id"],
        "algorithm": task["algorithm"],
        "dataset_id": task["dataset_id"],
        "seed": task["seed"],
        "status": status,
        "runtime_seconds": f"{runtime:.3f}",
        "has_result": str(has_result).lower(),
        "has_progress": str(has_progress).lower(),
        "metrics_valid": str(metrics_valid).lower(),
        "artifact_valid": str(artifact_valid).lower(),
        "failure_class": failure_class,
        "reason": reason,
    }


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_algorithm: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_algorithm.setdefault(row["algorithm"], {"total": 0, "passed": 0, "failed": 0})
        item["total"] += 1
        if row["failure_class"]:
            item["failed"] += 1
        else:
            item["passed"] += 1

    summary_rows = []
    for algorithm in sorted(by_algorithm):
        item = by_algorithm[algorithm]
        summary_rows.append(
            {
                "algorithm": algorithm,
                "total": item["total"],
                "passed": item["passed"],
                "failed": item["failed"],
                "compliant": str(item["failed"] == 0).lower(),
            }
        )
    _write_csv(
        path,
        ["algorithm", "total", "passed", "failed", "compliant"],
        summary_rows,
    )
