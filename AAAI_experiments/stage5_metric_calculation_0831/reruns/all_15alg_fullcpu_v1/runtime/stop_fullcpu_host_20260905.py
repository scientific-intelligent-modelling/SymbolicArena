#!/usr/bin/env python3
"""按精确 task/session 身份终止单台主机上的 fullcpu 子进程。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _load_targets(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise SystemExit("targets 必须是列表")
    result: dict[str, str] = {}
    for item in targets:
        if not isinstance(item, dict):
            raise SystemExit("target 必须是对象")
        task_id = str(item.get("task_id") or "").strip()
        session = str(item.get("session") or "").strip()
        if not task_id or not session:
            raise SystemExit("target 缺少 task_id/session")
        if not session.startswith("all_conditions_cpu_v2_"):
            raise SystemExit(f"拒绝非 fullcpu session: {session}")
        result[task_id] = session
    if len(result) != len(targets):
        raise SystemExit("target task_id 重复")
    return result


def _tmux_sessions() -> set[str]:
    completed = subprocess.run(
        ["tmux", "list-sessions", "-F", "#S"],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _marked_processes(targets: dict[str, str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    own_pgrp = os.getpgrp()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        environment = _read_bytes(process_dir / "environ")
        if b"SIM_QUEUE_TASK_ID=" not in environment:
            continue
        entries = environment.split(b"\0")
        values: dict[bytes, bytes] = {}
        for entry in entries:
            if b"=" in entry:
                key, value = entry.split(b"=", 1)
                values[key] = value
        task_id = values.get(b"SIM_QUEUE_TASK_ID", b"").decode(errors="replace")
        session = values.get(b"SIM_QUEUE_SESSION", b"").decode(errors="replace")
        if targets.get(task_id) != session:
            continue
        try:
            stat = (process_dir / "stat").read_text(encoding="utf-8").split()
            pgrp = int(stat[4])
        except (OSError, ValueError, IndexError):
            continue
        if pgrp <= 1 or pgrp == own_pgrp:
            continue
        command = _read_bytes(process_dir / "cmdline").replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        matches.append(
            {
                "pid": int(process_dir.name),
                "pgrp": pgrp,
                "task_id": task_id,
                "session": session,
                "command": command,
            }
        )
    return matches


def _kill_groups(matches: list[dict[str, Any]], sig: signal.Signals) -> None:
    for pgrp in sorted({int(item["pgrp"]) for item in matches}):
        try:
            os.killpg(pgrp, sig)
        except (ProcessLookupError, PermissionError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = _load_targets(args.targets)
    sessions_before = _tmux_sessions().intersection(targets.values())
    processes_before = _marked_processes(targets)
    if not args.dry_run:
        for session_name in sorted(sessions_before):
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        _kill_groups(processes_before, signal.SIGTERM)
        time.sleep(5)
        _kill_groups(_marked_processes(targets), signal.SIGKILL)
        time.sleep(2)

    sessions_after = _tmux_sessions().intersection(targets.values())
    processes_after = _marked_processes(targets)
    if not args.dry_run:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(
            json.dumps(
                {
                    "schema": "all_15alg_fullcpu_v1.host_abort_marker.v1",
                    "batch_name": "all_15alg_fullcpu_v1_formal",
                    "disposition": "aborted_do_not_use",
                    "host": os.uname().nodename,
                    "aborted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    result = {
        "schema": "all_15alg_fullcpu_v1.host_stop.v1",
        "host": os.uname().nodename,
        "dry_run": bool(args.dry_run),
        "target_count": len(targets),
        "matched_session_count_before": len(sessions_before),
        "matched_process_count_before": len(processes_before),
        "remaining_sessions": sorted(sessions_after),
        "remaining_processes": processes_after,
        "ok": bool(args.dry_run) or (not sessions_after and not processes_after),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
