#!/usr/bin/env python3
"""淘汰旧 QLattice 口径的结果，并将全部 450 条任务原子重置为待运行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from check import run_e1_candidate200_12alg_load_queue as queue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reap_running_tasks(
    tasks_by_host: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, bool]]:
    results: dict[str, dict[str, bool]] = {}
    if not tasks_by_host:
        return results
    os.environ["SIM_QUEUE_CONTROLLER_IS_LOCAL"] = "1"
    with ThreadPoolExecutor(max_workers=len(tasks_by_host)) as executor:
        futures = {
            executor.submit(
                queue._reap_remote_task_processes_bulk,
                host,
                tasks,
                controller_host="iaaccn22",
                use_internal_ips=True,
                reason="qlattice_global_best_contract_superseded",
            ): host
            for host, tasks in tasks_by_host.items()
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as exc:
                results[host] = {"__error__": False, "__detail__": repr(exc)}  # type: ignore[dict-item]
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--fix-commit", required=True)
    parser.add_argument("--expected-task-count", type=int, default=450)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    wrapper_sha256 = _sha256(args.wrapper)
    if wrapper_sha256 != args.expected_wrapper_sha256:
        raise SystemExit(
            "QLattice wrapper SHA-256 不一致: "
            f"expected={args.expected_wrapper_sha256}, actual={wrapper_sha256}"
        )

    state = json.loads(args.state.read_text(encoding="utf-8"))
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        raise SystemExit("state.tasks 必须是 object")
    qlattice_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if isinstance(task, dict) and task.get("tool") == "qlattice"
    }
    if len(qlattice_tasks) != args.expected_task_count:
        raise SystemExit(
            f"QLattice 任务数错误: expected={args.expected_task_count}, actual={len(qlattice_tasks)}"
        )

    state_sha256_before = _sha256(args.state)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    backup = args.state.with_name(
        f"{args.state.stem}.pre_qlattice_global_best_fix.{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    backup.write_bytes(args.state.read_bytes())
    if _sha256(backup) != state_sha256_before:
        raise SystemExit("state 备份 SHA-256 校验失败")

    before_counts = Counter(str(task.get("state")) for task in qlattice_tasks.values())
    running_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, task in qlattice_tasks.items():
        if task.get("state") != "running":
            continue
        host = str(task.get("assigned_host") or "")
        session = str(task.get("session") or "")
        if host and session:
            running_by_host[host].append({**task, "task_id": task_id})
    reap_results = _reap_running_tasks(dict(running_by_host))

    reset_fields = {
        "assigned_host": None,
        "state": "pending",
        "attempts": 0,
        "status_counts": {},
        "required_runtime_code_sha256": args.expected_wrapper_sha256,
        "contract_requeued_at": timestamp,
        "contract_requeue_reason": "qlattice_global_best_internal_criterion_fix",
        "contract_fix_commit": args.fix_commit,
    }
    runtime_fields = (
        "host",
        "session",
        "session_name",
        "started_at",
        "ended_at",
        "error",
        "last_status_read_error",
        "last_status_read_error_at",
        "status_read_error_since",
        "host_unavailable_since",
        "host_unavailable_last_at",
    )
    for task_id, task in qlattice_tasks.items():
        history = task.setdefault("contract_requeue_history", [])
        history.append(
            {
                "recorded_at": timestamp,
                "reason": "qlattice_global_best_internal_criterion_fix",
                "fix_commit": args.fix_commit,
                "required_runtime_code_sha256": args.expected_wrapper_sha256,
                "superseded_state": task.get("state"),
                "superseded_attempts": task.get("attempts"),
                "superseded_assigned_host": task.get("assigned_host"),
                "superseded_session": task.get("session"),
                "superseded_started_at": task.get("started_at"),
                "superseded_ended_at": task.get("ended_at"),
            }
        )
        task.update(reset_fields)
        for field in runtime_fields:
            task.pop(field, None)
        task["assigned_host"] = None

    state["updated_at"] = timestamp
    _atomic_write_json(args.state, state)
    state_sha256_after = _sha256(args.state)
    after_counts = Counter(str(task.get("state")) for task in qlattice_tasks.values())
    if after_counts != {"pending": args.expected_task_count}:
        raise SystemExit(f"重置后的 QLattice 状态错误: {dict(after_counts)}")

    event = {
        "event": "qlattice_global_best_contract_requeue",
        "time": timestamp,
        "task_count": args.expected_task_count,
        "before_counts": dict(sorted(before_counts.items())),
        "after_counts": dict(sorted(after_counts.items())),
        "fix_commit": args.fix_commit,
        "required_runtime_code_sha256": args.expected_wrapper_sha256,
        "state_sha256_before": state_sha256_before,
        "state_sha256_after": state_sha256_after,
        "backup": str(backup),
    }
    args.events.parent.mkdir(parents=True, exist_ok=True)
    with args.events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    audit = {
        "schema": "all_15alg_fullcpu_v1.qlattice_global_best_requeue.v1",
        **event,
        "wrapper": str(args.wrapper),
        "wrapper_sha256": wrapper_sha256,
        "running_task_counts_by_host": {
            host: len(items) for host, items in sorted(running_by_host.items())
        },
        "reap_results": reap_results,
        "reap_failed_task_ids": sorted(
            task_id
            for host_results in reap_results.values()
            for task_id, ok in host_results.items()
            if not ok and not task_id.startswith("__")
        ),
    }
    _atomic_write_json(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
