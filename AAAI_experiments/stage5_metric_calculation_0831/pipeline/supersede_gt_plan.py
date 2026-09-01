"""用受审计的一对一绑定把旧 Ground Truth 化简计划替换为新版计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit_frozen_simplifications import _load_audit_plan
from .run_claude_plan import (
    PlanContractError,
    PlannedDefinition,
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from .state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskStateStore,
    TaskSupersession,
)


class GroundTruthSupersessionError(RuntimeError):
    """Ground Truth 计划替代契约不成立。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise GroundTruthSupersessionError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _dataset_id(entry: PlannedDefinition) -> str:
    value = entry.definition.request.get("dataset_id")
    if not isinstance(value, str) or not value:
        raise GroundTruthSupersessionError(
            f"任务 {entry.logical_id!r} 缺少非空 request.dataset_id"
        )
    return value


def _index_plan(
    entries: Sequence[PlannedDefinition],
    *,
    label: str,
) -> dict[str, PlannedDefinition]:
    indexed: dict[str, PlannedDefinition] = {}
    for entry in entries:
        dataset_id = _dataset_id(entry)
        if dataset_id in indexed:
            raise GroundTruthSupersessionError(
                f"{label} 按 dataset_id 不唯一: {dataset_id!r}"
            )
        indexed[dataset_id] = entry
    return indexed


def _build_supersessions(
    *,
    predecessor_entries: Sequence[PlannedDefinition],
    successor_entries: Sequence[PlannedDefinition],
    predecessor_plan_sha256: str,
    successor_plan_sha256: str,
    expected_count: int,
) -> tuple[TaskSupersession, ...]:
    predecessors = _index_plan(predecessor_entries, label="predecessor plan")
    successors = _index_plan(successor_entries, label="successor plan")
    if len(predecessors) != expected_count or len(successors) != expected_count:
        raise GroundTruthSupersessionError(
            "Ground Truth 计划行数不符合预期: "
            f"predecessor={len(predecessors)}, successor={len(successors)}, "
            f"expected={expected_count}"
        )
    if predecessors.keys() != successors.keys():
        missing = sorted(predecessors.keys() - successors.keys())
        extra = sorted(successors.keys() - predecessors.keys())
        raise GroundTruthSupersessionError(
            f"Ground Truth 数据集集合漂移: missing={missing}, extra={extra}"
        )

    bindings: list[TaskSupersession] = []
    for dataset_id in sorted(predecessors):
        predecessor = predecessors[dataset_id]
        successor = successors[dataset_id]
        predecessor_spec = predecessor.definition.task_spec
        successor_spec = successor.definition.task_spec
        if predecessor.definition.request != successor.definition.request:
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的冻结 request/evidence 发生漂移"
            )
        expected_successor_logical_id = f"{predecessor.logical_id}::v2"
        if successor.logical_id != expected_successor_logical_id:
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的 successor logical_id 不合法: "
                f"{successor.logical_id!r} != {expected_successor_logical_id!r}"
            )
        stable_identity = (
            predecessor_spec.task_type,
            predecessor_spec.condition,
            predecessor_spec.priority,
            predecessor_spec.dependencies,
            predecessor_spec.schema_version,
        )
        successor_identity = (
            successor_spec.task_type,
            successor_spec.condition,
            successor_spec.priority,
            successor_spec.dependencies,
            successor_spec.schema_version,
        )
        if stable_identity != successor_identity:
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的任务身份或依赖发生漂移"
            )
        if predecessor_spec.task_type != "gt_simplify":
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的 predecessor 不是 gt_simplify"
            )
        if predecessor_spec.condition != "clean":
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的 predecessor 不是 clean 条件"
            )
        if predecessor_spec.prompt_version == successor_spec.prompt_version:
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的 prompt_version 没有变化"
            )
        if predecessor.evaluation_key == successor.evaluation_key:
            raise GroundTruthSupersessionError(
                f"数据集 {dataset_id!r} 的 evaluation_key 没有变化"
            )
        bindings.append(
            TaskSupersession(
                predecessor_evaluation_key=predecessor.evaluation_key,
                successor=successor_spec,
                identity=f"gt_simplify::{dataset_id}",
                reason="Ground Truth simplification prompt upgraded to simplify.v2",
                predecessor_plan_sha256=predecessor_plan_sha256,
                successor_plan_sha256=successor_plan_sha256,
            )
        )
    return tuple(bindings)


def _rows_as_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, object]]:
    return [{key: row[key] for key in row.keys()} for row in rows]


def _snapshot_predecessors(
    state_db: Path,
    bindings: Sequence[TaskSupersession],
) -> dict[str, object]:
    keys = [item.predecessor_evaluation_key for item in bindings]
    placeholders = ",".join("?" for _ in keys)
    with _connect_read_only(state_db) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_rows = _rows_as_dicts(connection.execute("PRAGMA foreign_key_check").fetchall())
        meta = dict(connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
        running_tasks = int(
            connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE state='running'"
            ).fetchone()[0]
        )
        running_attempts = int(
            connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE status='running'"
            ).fetchone()[0]
        )
        tasks = _rows_as_dicts(
            connection.execute(
                f"SELECT * FROM tasks WHERE evaluation_key IN ({placeholders}) "
                "ORDER BY evaluation_key",
                keys,
            ).fetchall()
        )
        attempts = _rows_as_dicts(
            connection.execute(
                f"SELECT * FROM attempts WHERE evaluation_key IN ({placeholders}) "
                "ORDER BY attempt_id",
                keys,
            ).fetchall()
        )
        frozen = _rows_as_dicts(
            connection.execute(
                f"SELECT * FROM frozen_results WHERE evaluation_key IN ({placeholders}) "
                "ORDER BY evaluation_key",
                keys,
            ).fetchall()
        )
        global_counts = {
            "tasks": int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]),
            "attempts": int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
            "supersessions": int(
                connection.execute("SELECT COUNT(*) FROM task_supersessions").fetchone()[0]
            )
            if "task_supersessions"
            in {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            else 0,
        }
        valid_superseded_count = int(
            connection.execute(
                """SELECT COUNT(*)
                   FROM task_supersessions ts
                   JOIN tasks predecessor
                     ON predecessor.evaluation_key = ts.predecessor_evaluation_key
                    AND predecessor.logical_id = ts.predecessor_logical_id
                   JOIN frozen_results predecessor_frozen
                     ON predecessor_frozen.evaluation_key = ts.predecessor_evaluation_key
                   JOIN tasks successor
                     ON successor.evaluation_key = ts.successor_evaluation_key
                    AND successor.logical_id = ts.successor_logical_id
                   WHERE predecessor.state='superseded'"""
            ).fetchone()[0]
        ) if global_counts["supersessions"] else 0
        global_counts["active_tasks"] = global_counts["tasks"] - valid_superseded_count
    if integrity != "ok" or foreign_key_rows:
        raise GroundTruthSupersessionError(
            f"状态库完整性检查失败: integrity={integrity!r}, foreign_keys={foreign_key_rows}"
        )
    if running_tasks or running_attempts:
        raise GroundTruthSupersessionError(
            f"状态库仍有 running 数据: tasks={running_tasks}, attempts={running_attempts}"
        )
    if len(tasks) != len(keys) or len(frozen) != len(keys):
        raise GroundTruthSupersessionError(
            f"predecessor 状态/冻结绑定不完整: tasks={len(tasks)}, frozen={len(frozen)}"
        )
    if any(row["state"] not in {"frozen", "superseded"} for row in tasks):
        states = sorted({str(row["state"]) for row in tasks})
        raise GroundTruthSupersessionError(f"predecessor 状态非法: {states}")
    frozen_files: list[dict[str, str]] = []
    for row in frozen:
        result_path = Path(str(row["result_path"]))
        if not result_path.is_file():
            raise GroundTruthSupersessionError(f"冻结文件不存在: {result_path}")
        actual_sha256 = _sha256_file(result_path)
        if actual_sha256 != row["result_sha256"]:
            raise GroundTruthSupersessionError(f"冻结文件 SHA256 漂移: {result_path}")
        frozen_files.append({"path": str(result_path), "sha256": actual_sha256})
    immutable_tasks = [
        {
            key: value
            for key, value in row.items()
            if key not in {"state", "lease_expires_at", "last_error_class", "updated_at"}
        }
        for row in tasks
    ]
    immutable_payload = {
        "tasks": immutable_tasks,
        "attempts": attempts,
        "frozen_results": frozen,
        "frozen_files": frozen_files,
    }
    return {
        "schema_version": meta.get("schema_version"),
        "meta": meta,
        "global_counts": global_counts,
        "task_states": {str(row["evaluation_key"]): str(row["state"]) for row in tasks},
        "immutable_sha256": _sha256_json(immutable_payload),
        "immutable_payload": immutable_payload,
    }


def _create_sqlite_backup(source_path: Path, backup_path: Path) -> dict[str, object]:
    if backup_path.exists():
        raise GroundTruthSupersessionError(f"备份路径已存在，拒绝覆盖: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
        destination.commit()
        integrity = str(destination.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_rows = destination.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        destination.close()
        source.close()
    if integrity != "ok" or foreign_key_rows:
        raise GroundTruthSupersessionError(
            f"SQLite 备份完整性检查失败: integrity={integrity!r}, foreign_keys={foreign_key_rows}"
        )
    return {
        "path": str(backup_path.resolve()),
        "sha256": _sha256_file(backup_path),
        "integrity_check": integrity,
        "foreign_key_violation_count": len(foreign_key_rows),
    }


def supersede_ground_truth_plan(
    *,
    predecessor_plan_jsonl: str | Path,
    successor_plan_jsonl: str | Path,
    state_db: str | Path,
    predecessor_attempt_manifest: str | Path,
    backup_path: str | Path,
    report_json: str | Path,
    expected_count: int = 50,
    now: float | None = None,
) -> dict[str, object]:
    started_at = time.time() if now is None else float(now)
    predecessor_plan, _, predecessor_mode, predecessor_prompt_mismatches = _load_audit_plan(
        predecessor_plan_jsonl,
        allow_archived_rendered_prompt=True,
    )
    successor_plan = load_plan_jsonl(successor_plan_jsonl)
    bindings = _build_supersessions(
        predecessor_entries=predecessor_plan.entries,
        successor_entries=successor_plan.entries,
        predecessor_plan_sha256=predecessor_plan.plan_sha256,
        successor_plan_sha256=successor_plan.plan_sha256,
        expected_count=expected_count,
    )
    state_path = Path(state_db).resolve()
    before = _snapshot_predecessors(state_path, bindings)
    backup = _create_sqlite_backup(state_path, Path(backup_path).resolve())
    backup_snapshot = _snapshot_predecessors(Path(str(backup["path"])), bindings)
    if (
        backup_snapshot["immutable_sha256"] != before["immutable_sha256"]
        or backup_snapshot["global_counts"] != before["global_counts"]
        or backup_snapshot["schema_version"] != before["schema_version"]
    ):
        raise GroundTruthSupersessionError("SQLite 备份与迁移前逻辑快照不一致")
    loaded_manifest = _load_predecessor_attempt_manifest(predecessor_attempt_manifest)
    manifest = PredecessorAttemptManifest(
        path=str(loaded_manifest.path),
        sha256=loaded_manifest.sha256,
        attempt_count=loaded_manifest.attempt_count,
    )
    store = TaskStateStore(
        state_path,
        predecessor_attempt_manifest=manifest,
        allow_schema_upgrade=True,
    )
    store.register_supersession_batch(bindings, now=started_at)
    after = _snapshot_predecessors(state_path, bindings)
    if before["immutable_sha256"] != after["immutable_sha256"]:
        raise GroundTruthSupersessionError(
            "supersession 后 predecessor 的任务身份、attempt、冻结绑定或文件发生漂移"
        )
    if set(after["task_states"].values()) != {"superseded"}:
        raise GroundTruthSupersessionError("supersession 后 predecessor 未全部进入 superseded")
    before_counts = before["global_counts"]
    after_counts = after["global_counts"]
    if after_counts["attempts"] != before_counts["attempts"]:
        raise GroundTruthSupersessionError("supersession 意外改变了物理 attempt 数")
    if after_counts["active_tasks"] != before_counts["active_tasks"]:
        raise GroundTruthSupersessionError("supersession 意外改变了 active logical task 数")
    allowed_historical_task_counts = {
        before_counts["tasks"],
        before_counts["tasks"] + expected_count,
    }
    if after_counts["tasks"] not in allowed_historical_task_counts:
        raise GroundTruthSupersessionError("supersession 后历史任务数变化不符合幂等契约")
    summary = store.state_summary()
    successor_states = {
        item.successor.evaluation_key: store.task_state(item.successor.evaluation_key)
        for item in bindings
    }
    finished_at = time.time() if now is None else float(now)
    report: dict[str, object] = {
        "status": "ok",
        "model_invoked": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "state_db": str(state_path),
        "expected_count": expected_count,
        "supersession_count": len(bindings),
        "predecessor_plan": {
            "path": str(predecessor_plan.plan_path.resolve()),
            "sha256": predecessor_plan.plan_sha256,
            "reader_mode": predecessor_mode,
            "archived_prompt_mismatch_count": predecessor_prompt_mismatches,
        },
        "successor_plan": {
            "path": str(successor_plan.plan_path.resolve()),
            "sha256": successor_plan.plan_sha256,
            "reader_mode": "current_renderer",
        },
        "predecessor_attempt_manifest": {
            "path": str(loaded_manifest.path),
            "sha256": loaded_manifest.sha256,
            "attempt_count": loaded_manifest.attempt_count,
        },
        "backup": backup,
        "before": {
            "schema_version": before["schema_version"],
            "global_counts": before_counts,
            "immutable_sha256": before["immutable_sha256"],
        },
        "after": {
            "schema_version": after["schema_version"],
            "global_counts": after_counts,
            "immutable_sha256": after["immutable_sha256"],
            "successor_state_counts": {
                state: list(successor_states.values()).count(state)
                for state in sorted(set(successor_states.values()))
            },
            "state_summary": summary,
        },
        "bindings": [
            {
                "identity": item.identity,
                "predecessor_evaluation_key": item.predecessor_evaluation_key,
                "successor_evaluation_key": item.successor.evaluation_key,
                "successor_logical_id": item.successor.logical_id,
            }
            for item in bindings
        ],
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-plan-jsonl", required=True)
    parser.add_argument("--successor-plan-jsonl", required=True)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--predecessor-attempt-manifest", required=True)
    parser.add_argument("--backup-path", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = supersede_ground_truth_plan(
            predecessor_plan_jsonl=args.predecessor_plan_jsonl,
            successor_plan_jsonl=args.successor_plan_jsonl,
            state_db=args.state_db,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            backup_path=args.backup_path,
            report_json=args.report_json,
            expected_count=args.expected_count,
        )
    except (GroundTruthSupersessionError, PlanContractError, StateContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
