"""为 lease_expired 尝试补写不伪造模型输出的审计文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import CONTRACT_EFFORT, CONTRACT_MODEL, canonical_json
from .run_claude_plan import PlanContractError, load_plan_jsonl


JsonDict = dict[str, object]


class RecoveredAttemptAuditError(RuntimeError):
    """lease 恢复记录无法形成严格审计闭环。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path, *, context: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveredAttemptAuditError(f"{context} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecoveredAttemptAuditError(f"{context} 必须是 JSON object")
    return payload


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise RecoveredAttemptAuditError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _task_kind(task_type: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if task_type.endswith("simplify"):
        return "simplify"
    if task_type.endswith("equivalence"):
        return "equivalence"
    if task_type.endswith("structure"):
        return "structure"
    raise RecoveredAttemptAuditError(f"无法推断 task_kind: {task_type!r}")


def _require_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveredAttemptAuditError(f"{context} 必须是数值")
    return float(value)


def _expected_attempt_payload(
    *,
    row: sqlite3.Row,
    event: sqlite3.Row,
    plan_entry: object,
    plan_sha256: str,
) -> JsonDict:
    evaluation_key = str(row["evaluation_key"])
    attempt_number = int(row["attempt_number"])
    attempt_id = str(row["attempt_id"])
    expected_attempt_id = f"{evaluation_key}.a{attempt_number:02d}"
    if attempt_id != expected_attempt_id:
        raise RecoveredAttemptAuditError(
            f"{attempt_id} 不符合 evaluation_key + attempt_number 契约"
        )
    if row["status"] != "failed" or row["error_class"] != "lease_expired":
        raise RecoveredAttemptAuditError(f"{attempt_id} 不是 failed/lease_expired")
    if row["retryable"] != 1:
        raise RecoveredAttemptAuditError(f"{attempt_id}.retryable 必须为 1")
    reserved_at = _require_float(row["reserved_at"], context=f"{attempt_id}.reserved_at")
    lease_expires_at = _require_float(
        row["lease_expires_at"],
        context=f"{attempt_id}.lease_expires_at",
    )
    finished_at = _require_float(row["finished_at"], context=f"{attempt_id}.finished_at")
    event_at = _require_float(event["event_at"], context=f"{attempt_id}.event_at")
    if finished_at < lease_expires_at or event_at != finished_at:
        raise RecoveredAttemptAuditError(f"{attempt_id} 的 lease/event 时间闭环不一致")
    try:
        details = json.loads(str(event["details_json"]))
    except json.JSONDecodeError as exc:
        raise RecoveredAttemptAuditError(f"{attempt_id}.details_json 非法") from exc
    if not isinstance(details, dict) or details.get("next_state") not in {
        "retry_wait",
        "exhausted",
    }:
        raise RecoveredAttemptAuditError(f"{attempt_id}.lease_expired 事件详情非法")

    definition = plan_entry.definition
    task_type = definition.task_spec.task_type
    return {
        "attempt_id": attempt_id,
        "evaluation_key": evaluation_key,
        "logical_id": plan_entry.logical_id,
        "task_type": task_type,
        "task_kind": _task_kind(task_type, definition.task_kind),
        "request": dict(definition.request),
        "prompt": None,
        "command": None,
        "stdout": None,
        "stderr": None,
        "envelope": None,
        "structured_output": None,
        "validation": {
            "ok": False,
            "error_class": "lease_expired",
            "error_message": (
                "attempt lease 已过期，原执行进程未能完成审计文件；"
                "模型输出不可恢复且未作任何推断。"
            ),
            "structured_output": None,
        },
        "metadata": {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "evaluation_key": evaluation_key,
            "logical_id": plan_entry.logical_id,
            "task_type": task_type,
            "task_kind": _task_kind(task_type, definition.task_kind),
            "condition": definition.task_spec.condition,
            "requested_model": CONTRACT_MODEL,
            "requested_effort": CONTRACT_EFFORT,
            "reserved_at": reserved_at,
            "lease_expires_at": lease_expires_at,
            "finished_at": finished_at,
            "error_class": "lease_expired",
            "retryable": True,
            "recovered_lease_expired": True,
            "model_output_available": False,
            "model_invocation_status": "unknown_after_process_termination",
            "original_attempt_artifact_missing": True,
            "recovery_event_id": int(event["event_id"]),
            "recovery_event_at": event_at,
            "recovery_event_details": details,
            "plan_sha256": plan_sha256,
        },
    }


def materialize_recovered_attempts(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    report_json: str | Path,
) -> JsonDict:
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise RecoveredAttemptAuditError(f"plan 契约失败: {exc}") from exc
    by_key = {entry.evaluation_key: entry for entry in loaded_plan.entries}
    state_path = Path(state_db).resolve()
    attempts_path = Path(attempts_dir).resolve()
    connection = _connect_read_only(state_path)
    try:
        rows = connection.execute(
            """SELECT a.*, t.logical_id, t.task_type, t.condition_name, t.spec_json
               FROM attempts a
               JOIN tasks t ON t.evaluation_key = a.evaluation_key
               WHERE a.status='failed' AND a.error_class='lease_expired'
               ORDER BY a.evaluation_key, a.attempt_number"""
        ).fetchall()
        artifacts: list[JsonDict] = []
        created_count = 0
        reused_count = 0
        for row in rows:
            evaluation_key = str(row["evaluation_key"])
            entry = by_key.get(evaluation_key)
            if entry is None:
                raise RecoveredAttemptAuditError(
                    f"lease_expired attempt 不在 plan 中: {row['attempt_id']}"
                )
            definition = entry.definition
            if row["logical_id"] != entry.logical_id:
                raise RecoveredAttemptAuditError(f"{row['attempt_id']}.logical_id 漂移")
            if row["task_type"] != definition.task_spec.task_type:
                raise RecoveredAttemptAuditError(f"{row['attempt_id']}.task_type 漂移")
            if row["condition_name"] != definition.task_spec.condition:
                raise RecoveredAttemptAuditError(f"{row['attempt_id']}.condition 漂移")
            if row["spec_json"] != definition.task_spec.canonical_json():
                raise RecoveredAttemptAuditError(f"{row['attempt_id']}.task spec 漂移")
            event_rows = connection.execute(
                """SELECT event_id, event_at, details_json
                   FROM events
                   WHERE attempt_id=? AND evaluation_key=? AND event_type='lease_expired'
                   ORDER BY event_id""",
                (row["attempt_id"], evaluation_key),
            ).fetchall()
            if len(event_rows) != 1:
                raise RecoveredAttemptAuditError(
                    f"{row['attempt_id']} 必须恰有一个 lease_expired event"
                )
            expected_payload = _expected_attempt_payload(
                row=row,
                event=event_rows[0],
                plan_entry=entry,
                plan_sha256=loaded_plan.plan_sha256,
            )
            attempt_path = attempts_path / f"{row['attempt_id']}.json"
            if attempt_path.exists():
                actual_payload = _read_json_object(
                    attempt_path,
                    context=f"既有恢复 attempt {row['attempt_id']}",
                )
                if canonical_json(actual_payload) != canonical_json(expected_payload):
                    raise RecoveredAttemptAuditError(
                        f"既有恢复 attempt 与当前状态证据冲突: {attempt_path}"
                    )
                reused_count += 1
            else:
                _atomic_write_json(attempt_path, expected_payload)
                created_count += 1
            artifacts.append(
                {
                    "attempt_id": row["attempt_id"],
                    "evaluation_key": evaluation_key,
                    "logical_id": entry.logical_id,
                    "attempt_number": int(row["attempt_number"]),
                    "attempt_path": str(attempt_path),
                    "attempt_sha256": _sha256_file(attempt_path),
                    "recovery_event_id": int(event_rows[0]["event_id"]),
                }
            )
    finally:
        connection.close()

    report: JsonDict = {
        "status": "ok",
        "model_invoked": False,
        "attempt_count_delta": 0,
        "plan_jsonl": str(loaded_plan.plan_path.resolve()),
        "plan_sha256": loaded_plan.plan_sha256,
        "state_db": str(state_path),
        "attempts_dir": str(attempts_path),
        "eligible_count": len(artifacts),
        "created_count": created_count,
        "reused_count": reused_count,
        "artifacts": artifacts,
    }
    report_path = Path(report_json).resolve()
    _atomic_write_json(report_path, report)
    report["report_json"] = str(report_path)
    report["report_sha256"] = _sha256_file(report_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补全 lease_expired attempt 审计文件")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialize_recovered_attempts(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            report_json=args.report_json,
        )
    except RecoveredAttemptAuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
