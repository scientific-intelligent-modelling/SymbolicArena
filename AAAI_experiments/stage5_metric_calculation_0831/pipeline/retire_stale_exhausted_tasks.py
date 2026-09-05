"""审计化退休已有冻结替代项的旧 exhausted 任务。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .run_claude_plan import _load_predecessor_attempt_manifest, load_plan_jsonl
from .state import PredecessorAttemptManifest, TaskRetirement, TaskStateStore


class StaleExhaustedRetirementError(RuntimeError):
    """旧 exhausted 任务无法安全映射到当前冻结版本。"""


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


def _revision_base(logical_id: str) -> str:
    return re.sub(r"::v\d+$", "", logical_id)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _historical_plan_bindings(
    paths: Sequence[Path],
    *,
    task_type: str,
    condition: str,
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        loaded = load_plan_jsonl(path)
        for entry in loaded.entries:
            spec = entry.definition.task_spec
            if spec.task_type != task_type or spec.condition != condition:
                raise StaleExhaustedRetirementError(
                    f"历史 plan 与显式范围不一致: {entry.logical_id} "
                    f"({spec.task_type}/{spec.condition})"
                )
            bindings.setdefault(
                entry.evaluation_key,
                {
                    "plan_path": str(path),
                    "plan_sha256": loaded.plan_sha256,
                },
            )
    return bindings


def _backup_sqlite(source_path: Path, backup_path: Path) -> None:
    if backup_path.exists():
        raise StaleExhaustedRetirementError(f"备份文件已存在，拒绝覆盖: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)


def retire_stale_exhausted_tasks(
    *,
    state_db: Path,
    predecessor_attempt_manifest: Path,
    active_plan_jsonl: Path,
    historical_plan_jsonls: Sequence[Path],
    backup_sqlite: Path,
    manifest_jsonl: Path,
    report_json: Path,
    expected_count: int,
    task_type: str = "pred_simplify",
    condition: str = "clean",
) -> dict[str, Any]:
    if not task_type:
        raise StaleExhaustedRetirementError("task_type 不得为空")
    if condition not in {"clean", "noise001", "noise005"}:
        raise StaleExhaustedRetirementError(f"condition 无效: {condition!r}")
    active_plan = load_plan_jsonl(active_plan_jsonl.resolve())
    active_by_base: dict[str, Any] = {}
    for entry in active_plan.entries:
        spec = entry.definition.task_spec
        if spec.task_type != task_type or spec.condition != condition:
            raise StaleExhaustedRetirementError(
                f"当前 plan 与显式范围不一致: {entry.logical_id} "
                f"({spec.task_type}/{spec.condition})"
            )
        base = _revision_base(entry.logical_id)
        if base in active_by_base:
            raise StaleExhaustedRetirementError(f"当前计划 revision base 重复: {base}")
        active_by_base[base] = entry

    historical_bindings = _historical_plan_bindings(
        historical_plan_jsonls,
        task_type=task_type,
        condition=condition,
    )
    with sqlite3.connect(state_db) as connection:
        connection.row_factory = sqlite3.Row
        state_meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        exhausted_rows = connection.execute(
            """SELECT evaluation_key, logical_id, attempt_count, last_error_class
               FROM tasks
               WHERE task_type=?
                 AND condition_name=?
                 AND state='exhausted'
               ORDER BY logical_id""",
            (task_type, condition),
        ).fetchall()
        task_rows = {
            str(row["evaluation_key"]): row
            for row in connection.execute(
                """SELECT evaluation_key, logical_id, task_type, condition_name,
                          state, spec_json
                   FROM tasks"""
            ).fetchall()
        }
        frozen_keys = {
            str(row["evaluation_key"])
            for row in connection.execute(
                "SELECT evaluation_key FROM frozen_results"
            ).fetchall()
        }
    if len(exhausted_rows) != expected_count:
        raise StaleExhaustedRetirementError(
            f"旧 exhausted 数量漂移: {len(exhausted_rows)} != {expected_count}"
        )

    retirements: list[TaskRetirement] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in exhausted_rows:
        exhausted_key = str(row["evaluation_key"])
        exhausted_logical_id = str(row["logical_id"])
        base = _revision_base(exhausted_logical_id)
        replacement = active_by_base.get(base)
        if replacement is None:
            raise StaleExhaustedRetirementError(
                f"{exhausted_logical_id} 在当前计划中没有同 revision base 替代项"
            )
        if replacement.evaluation_key == exhausted_key:
            raise StaleExhaustedRetirementError(
                f"{exhausted_logical_id} 仍是当前计划任务，禁止退休"
            )
        replacement_state = task_rows.get(replacement.evaluation_key)
        if replacement_state is None:
            raise StaleExhaustedRetirementError(
                f"替代任务尚未注册: {replacement.logical_id}"
            )
        replacement_spec = replacement.definition.task_spec
        if (
            replacement_state["logical_id"] != replacement.logical_id
            or replacement_state["task_type"] != task_type
            or replacement_state["condition_name"] != condition
            or replacement_state["spec_json"] != replacement_spec.canonical_json()
        ):
            raise StaleExhaustedRetirementError(
                f"替代任务身份漂移: {replacement.logical_id}"
            )
        if (
            replacement_state["state"] != "frozen"
            or replacement.evaluation_key not in frozen_keys
        ):
            raise StaleExhaustedRetirementError(
                f"替代任务必须已 frozen 且有结果绑定: {replacement.logical_id}"
            )
        historical_binding = historical_bindings.get(exhausted_key)
        if historical_binding is None:
            raise StaleExhaustedRetirementError(
                f"{exhausted_logical_id} 缺少历史 plan 绑定"
            )
        identity_payload = {
            "schema_version": "stale_exhausted_retirement.v1",
            "exhausted_evaluation_key": exhausted_key,
            "replacement_evaluation_key": replacement.evaluation_key,
            "revision_base": base,
        }
        identity = _sha256_json(identity_payload)
        reason = "当前 Stage5 正式计划已有同逻辑且已冻结的替代版本"
        retirement = TaskRetirement(
            exhausted_evaluation_key=exhausted_key,
            replacement=replacement.definition.task_spec,
            identity=identity,
            reason=reason,
            exhausted_plan_sha256=historical_binding["plan_sha256"],
            replacement_plan_sha256=active_plan.plan_sha256,
        )
        retirements.append(retirement)
        manifest_rows.append(
            {
                **identity_payload,
                "identity": identity,
                "reason": reason,
                "exhausted_logical_id": exhausted_logical_id,
                "exhausted_attempt_count": int(row["attempt_count"]),
                "exhausted_last_error_class": row["last_error_class"],
                "exhausted_plan_path": historical_binding["plan_path"],
                "exhausted_plan_sha256": historical_binding["plan_sha256"],
                "replacement_logical_id": replacement.logical_id,
                "replacement_plan_path": str(active_plan.plan_path),
                "replacement_plan_sha256": active_plan.plan_sha256,
            }
        )

    _backup_sqlite(state_db.resolve(), backup_sqlite.resolve())
    loaded_predecessor_manifest = _load_predecessor_attempt_manifest(
        predecessor_attempt_manifest.resolve()
    )
    store = TaskStateStore(
        state_db.resolve(),
        attempt_cap=int(state_meta["attempt_cap"]),
        logical_task_cap=int(state_meta["logical_task_cap"]),
        max_attempts_per_task=int(state_meta["max_attempts_per_task"]),
        predecessor_attempt_manifest=PredecessorAttemptManifest(
            path=str(loaded_predecessor_manifest.path),
            sha256=loaded_predecessor_manifest.sha256,
            attempt_count=loaded_predecessor_manifest.attempt_count,
        ),
    )
    store.retire_exhausted_tasks(tuple(retirements))
    summary = store.state_summary()
    with sqlite3.connect(state_db) as connection:
        remaining_exhausted = int(
            connection.execute(
                """SELECT COUNT(*) FROM tasks
                   WHERE task_type=?
                     AND condition_name=?
                     AND state='exhausted'""",
                (task_type, condition),
            ).fetchone()[0]
        )
        retirement_count = int(
            connection.execute(
                """SELECT COUNT(*)
                   FROM task_retirements AS retirement
                   JOIN tasks AS exhausted
                     ON exhausted.evaluation_key=retirement.exhausted_evaluation_key
                   WHERE exhausted.task_type=? AND exhausted.condition_name=?""",
                (task_type, condition),
            ).fetchone()[0]
        )
        global_retirement_count = int(
            connection.execute("SELECT COUNT(*) FROM task_retirements").fetchone()[0]
        )
    if remaining_exhausted != 0 or retirement_count < expected_count:
        raise StaleExhaustedRetirementError(
            "retirement 落库后状态未闭合: "
            f"remaining_exhausted={remaining_exhausted}, retirement_count={retirement_count}"
        )

    _atomic_write_jsonl(manifest_jsonl.resolve(), manifest_rows)
    report = {
        "status": "ok",
        "schema_version": "stale_exhausted_retirement_report.v1",
        "task_type": task_type,
        "condition": condition,
        "state_db": str(state_db.resolve()),
        "predecessor_attempt_manifest": str(predecessor_attempt_manifest.resolve()),
        "predecessor_attempt_manifest_sha256": loaded_predecessor_manifest.sha256,
        "backup_sqlite": str(backup_sqlite.resolve()),
        "backup_sha256": _sha256_file(backup_sqlite.resolve()),
        "active_plan_jsonl": str(active_plan.plan_path),
        "active_plan_sha256": active_plan.plan_sha256,
        "historical_plan_jsonls": [str(path.resolve()) for path in historical_plan_jsonls],
        "expected_count": expected_count,
        "retired_count": len(retirements),
        "remaining_exhausted_count": remaining_exhausted,
        "task_retirement_table_count": retirement_count,
        "global_task_retirement_table_count": global_retirement_count,
        "manifest_jsonl": str(manifest_jsonl.resolve()),
        "manifest_sha256": _sha256_file(manifest_jsonl.resolve()),
        "state_summary": summary,
    }
    _atomic_write_json(report_json.resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path, required=True)
    parser.add_argument("--active-plan-jsonl", type=Path, required=True)
    parser.add_argument(
        "--historical-plan-jsonl",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--backup-sqlite", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--task-type", default="pred_simplify")
    parser.add_argument(
        "--condition",
        choices=("clean", "noise001", "noise005"),
        default="clean",
    )
    args = parser.parse_args(argv)
    report = retire_stale_exhausted_tasks(
        state_db=args.state_db,
        predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        active_plan_jsonl=args.active_plan_jsonl,
        historical_plan_jsonls=args.historical_plan_jsonl,
        backup_sqlite=args.backup_sqlite,
        manifest_jsonl=args.manifest_jsonl,
        report_json=args.report_json,
        expected_count=args.expected_count,
        task_type=args.task_type,
        condition=args.condition,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
