"""把 clean pred 冻结审计失败项替换为可重新调用的版本化 successor。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from .revise_exhausted_pred_plan import _read_plan_rows, _successor_row
from .run_claude_plan import (
    PlanContractError,
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from .state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskStateStore,
    TaskSupersession,
)


JsonDict = dict[str, object]


class AuditRepairError(RuntimeError):
    """审计证据、successor 计划或状态迁移不满足约束。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditRepairError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditRepairError(f"{path} 顶层不是 JSON object")
    return payload


def _read_audit_failures(
    *,
    audit_jsonl: Path,
    audit_report_json: Path,
    plan_sha256: str,
    expected_plan_count: int,
) -> dict[str, str]:
    report = _read_json_object(audit_report_json)
    if report.get("plan_sha256") != plan_sha256:
        raise AuditRepairError("audit report 的 plan_sha256 与 predecessor plan 不一致")
    if report.get("output_sha256") != _sha256_file(audit_jsonl):
        raise AuditRepairError("audit report 的 output_sha256 与 audit JSONL 不一致")
    rows: list[JsonDict] = []
    with audit_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditRepairError(f"audit JSONL 第 {line_number} 行解析失败: {exc}") from exc
            if not isinstance(payload, dict):
                raise AuditRepairError(f"audit JSONL 第 {line_number} 行不是 object")
            rows.append(payload)
    if len(rows) != expected_plan_count:
        raise AuditRepairError(
            f"audit 行数不符: {len(rows)} != {expected_plan_count}"
        )
    failures: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        logical_id = row.get("logical_id")
        status = row.get("status")
        if not isinstance(logical_id, str) or not logical_id or logical_id in seen:
            raise AuditRepairError("audit logical_id 缺失或重复")
        seen.add(logical_id)
        if status == "failed":
            failure_class = row.get("failure_class")
            if not isinstance(failure_class, str) or not failure_class:
                raise AuditRepairError(f"失败审计缺少 failure_class: {logical_id}")
            failures[logical_id] = failure_class
        elif status != "passed":
            raise AuditRepairError(f"未知 audit status: {logical_id}={status!r}")
    declared_failed = report.get("failed_count")
    if isinstance(declared_failed, bool) or not isinstance(declared_failed, int):
        raise AuditRepairError("audit report.failed_count 无效")
    if declared_failed != len(failures) or not failures:
        raise AuditRepairError(
            f"audit failed 数量不闭合: report={declared_failed}, rows={len(failures)}"
        )
    return failures


def _create_backup(source_path: Path, backup_path: Path) -> dict[str, object]:
    if backup_path.exists():
        raise AuditRepairError(f"备份路径已存在，拒绝覆盖: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
        destination.commit()
        integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = destination.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        destination.close()
        source.close()
    if integrity != "ok" or foreign_keys:
        raise AuditRepairError("SQLite 备份完整性检查失败")
    return {
        "path": str(backup_path.resolve()),
        "sha256": _sha256_file(backup_path),
        "integrity_check": integrity,
        "foreign_key_violation_count": len(foreign_keys),
    }


def repair_audited_pred_simplifications(
    *,
    predecessor_plan_jsonl: str | Path,
    audit_jsonl: str | Path,
    audit_report_json: str | Path,
    state_db: str | Path,
    output_plan_jsonl: str | Path,
    backup_state_db: str | Path,
    report_json: str | Path,
    predecessor_attempt_manifest: str | Path | None = None,
    logical_id_suffix: str = "v2",
    expected_plan_count: int = 2250,
    expected_failed_count: int | None = None,
    now: float | None = None,
) -> JsonDict:
    started_at = time.time() if now is None else float(now)
    predecessor_path = Path(predecessor_plan_jsonl).resolve()
    try:
        predecessor = load_plan_jsonl(predecessor_path)
    except PlanContractError as exc:
        raise AuditRepairError(f"predecessor plan 契约失败: {exc}") from exc
    if len(predecessor.entries) != expected_plan_count:
        raise AuditRepairError(
            f"predecessor plan 行数不符: {len(predecessor.entries)} != {expected_plan_count}"
        )
    raw_rows = _read_plan_rows(predecessor_path)
    row_by_logical_id = {str(row["logical_id"]): row for row in raw_rows}
    if len(row_by_logical_id) != len(raw_rows):
        raise AuditRepairError("predecessor plan logical_id 不唯一")
    failures = _read_audit_failures(
        audit_jsonl=Path(audit_jsonl).resolve(),
        audit_report_json=Path(audit_report_json).resolve(),
        plan_sha256=predecessor.plan_sha256,
        expected_plan_count=expected_plan_count,
    )
    if expected_failed_count is not None and len(failures) != expected_failed_count:
        raise AuditRepairError(
            f"失败任务数量不符: {len(failures)} != {expected_failed_count}"
        )

    state_path = Path(state_db).resolve()
    if not state_path.is_file():
        raise AuditRepairError(f"状态库不存在: {state_path}")
    with sqlite3.connect(f"{state_path.as_uri()}?mode=ro", uri=True) as connection:
        task_states = dict(
            connection.execute(
                "SELECT logical_id, state FROM tasks"
            ).fetchall()
        )
        frozen_keys = {
            str(row[0])
            for row in connection.execute("SELECT evaluation_key FROM frozen_results")
        }

    output_rows: list[JsonDict] = []
    provisional: list[tuple[object, JsonDict, str]] = []
    for entry in predecessor.entries:
        row = row_by_logical_id[entry.logical_id]
        if entry.logical_id not in failures:
            output_rows.append(dict(row))
            continue
        if task_states.get(entry.logical_id) != "frozen" or entry.evaluation_key not in frozen_keys:
            raise AuditRepairError(f"审计失败 predecessor 未冻结: {entry.logical_id}")
        successor = _successor_row(
            row,
            prompt_path=Path(str(row["prompt_path"])).resolve(),
            prompt_template=str(row["prompt_template"]),
            prompt_sha256=str(row["prompt_sha256"]),
            logical_id_suffix=logical_id_suffix,
            condition=entry.definition.task_spec.condition,
        )
        output_rows.append(successor)
        provisional.append((entry, successor, failures[entry.logical_id]))

    output_rows.sort(key=lambda row: (int(row["priority"]), str(row["logical_id"])))
    output_path = Path(output_plan_jsonl).resolve()
    _atomic_write_text(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in output_rows),
    )
    try:
        successor_plan = load_plan_jsonl(output_path)
    except PlanContractError as exc:
        raise AuditRepairError(f"successor plan 回读失败: {exc}") from exc
    successor_by_logical = {entry.logical_id: entry for entry in successor_plan.entries}

    bindings: list[TaskSupersession] = []
    binding_report: list[JsonDict] = []
    for predecessor_entry, successor_row, failure_class in provisional:
        successor_logical_id = str(successor_row["logical_id"])
        successor_entry = successor_by_logical[successor_logical_id]
        if predecessor_entry.definition.request != successor_entry.definition.request:
            raise AuditRepairError(f"successor request 漂移: {successor_logical_id}")
        binding = TaskSupersession(
            predecessor_evaluation_key=predecessor_entry.evaluation_key,
            successor=successor_entry.definition.task_spec,
            identity=f"pred_audit_repair::{predecessor_entry.logical_id}",
            reason=f"frozen simplification audit failed: {failure_class}",
            predecessor_plan_sha256=predecessor.plan_sha256,
            successor_plan_sha256=successor_plan.plan_sha256,
        )
        bindings.append(binding)
        binding_report.append(
            {
                "predecessor_logical_id": predecessor_entry.logical_id,
                "predecessor_evaluation_key": predecessor_entry.evaluation_key,
                "successor_logical_id": successor_logical_id,
                "successor_evaluation_key": successor_entry.evaluation_key,
                "failure_class": failure_class,
            }
        )

    backup = _create_backup(state_path, Path(backup_state_db).resolve())
    manifest: PredecessorAttemptManifest | None = None
    manifest_report: JsonDict | None = None
    if predecessor_attempt_manifest is not None:
        loaded_manifest = _load_predecessor_attempt_manifest(predecessor_attempt_manifest)
        manifest = PredecessorAttemptManifest(
            path=str(loaded_manifest.path),
            sha256=loaded_manifest.sha256,
            attempt_count=loaded_manifest.attempt_count,
        )
        manifest_report = {
            "path": str(loaded_manifest.path),
            "sha256": loaded_manifest.sha256,
            "attempt_count": loaded_manifest.attempt_count,
        }
    try:
        store = TaskStateStore(
            state_path,
            predecessor_attempt_manifest=manifest,
        )
        store.register_supersession_batch(bindings, now=started_at)
    except StateContractError as exc:
        raise AuditRepairError(f"successor 状态注册失败: {exc}") from exc
    successor_states = {
        binding.successor.evaluation_key: store.task_state(binding.successor.evaluation_key)
        for binding in bindings
    }
    predecessor_states = {
        binding.predecessor_evaluation_key: store.task_state(binding.predecessor_evaluation_key)
        for binding in bindings
    }
    if set(successor_states.values()) != {"pending"}:
        raise AuditRepairError("successor 注册后未全部进入 pending")
    if set(predecessor_states.values()) != {"superseded"}:
        raise AuditRepairError("predecessor 注册后未全部进入 superseded")

    report: JsonDict = {
        "status": "ok",
        "model_invoked": False,
        "state_db_mutated": True,
        "predecessor_plan_jsonl": str(predecessor_path),
        "predecessor_plan_sha256": predecessor.plan_sha256,
        "audit_jsonl": str(Path(audit_jsonl).resolve()),
        "audit_jsonl_sha256": _sha256_file(Path(audit_jsonl).resolve()),
        "audit_report_json": str(Path(audit_report_json).resolve()),
        "failed_count": len(failures),
        "logical_id_suffix": logical_id_suffix,
        "output_plan_jsonl": str(output_path),
        "output_plan_sha256": successor_plan.plan_sha256,
        "output_plan_count": len(successor_plan.entries),
        "state_db": str(state_path),
        "backup": backup,
        "predecessor_attempt_manifest": manifest_report,
        "bindings": binding_report,
        "state_summary": store.state_summary(),
        "started_at": started_at,
        "finished_at": time.time() if now is None else float(now),
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-plan-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--audit-report-json", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-plan-jsonl", type=Path, required=True)
    parser.add_argument("--backup-state-db", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path)
    parser.add_argument("--logical-id-suffix", default="v2")
    parser.add_argument("--expected-plan-count", type=int, default=2250)
    parser.add_argument("--expected-failed-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repair_audited_pred_simplifications(
            predecessor_plan_jsonl=args.predecessor_plan_jsonl,
            audit_jsonl=args.audit_jsonl,
            audit_report_json=args.audit_report_json,
            state_db=args.state_db,
            output_plan_jsonl=args.output_plan_jsonl,
            backup_state_db=args.backup_state_db,
            report_json=args.report_json,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            logical_id_suffix=args.logical_id_suffix,
            expected_plan_count=args.expected_plan_count,
            expected_failed_count=args.expected_failed_count,
        )
    except AuditRepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
