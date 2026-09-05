#!/usr/bin/env python3
"""终止本机所有带正式队列 QLattice 环境标记的旧进程和 tmux 会话。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


TASK_MARKER = b"SIM_QUEUE_TASK_ID=qlattice_"
SESSION_MARKER = b"SIM_QUEUE_SESSION=all_conditions_cpu_v2_qlattice_"
SESSION_PREFIX = "all_conditions_cpu_v2_qlattice_"


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _marked_processes() -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    own_pid = os.getpid()
    own_pgrp = os.getpgrp()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        environment = _read_bytes(process_dir / "environ")
        if TASK_MARKER not in environment and SESSION_MARKER not in environment:
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
        matches.append({"pid": int(process_dir.name), "pgrp": pgrp, "command": command})
    return matches


def _qlattice_tmux_sessions() -> list[str]:
    completed = subprocess.run(
        ["tmux", "list-sessions", "-F", "#S"],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(SESSION_PREFIX)
    )


def _process_alive(pid: int) -> bool:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(stat) > 2 and stat[2] != "Z"


def main() -> int:
    before = _marked_processes()
    sessions = _qlattice_tmux_sessions()
    for session in sessions:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    process_groups = sorted({int(item["pgrp"]) for item in before})
    for pgrp in process_groups:
        try:
            os.killpg(pgrp, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(5)
    remaining_after_term = [
        int(item["pid"]) for item in before if _process_alive(int(item["pid"]))
    ]
    if remaining_after_term:
        remaining_groups = {
            int(item["pgrp"])
            for item in before
            if int(item["pid"]) in remaining_after_term
        }
        for pgrp in remaining_groups:
            try:
                os.killpg(pgrp, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(2)

    remaining = [
        int(item["pid"]) for item in before if _process_alive(int(item["pid"]))
    ]
    payload = {
        "matched_process_count": len(before),
        "matched_process_groups": process_groups,
        "matched_tmux_sessions": sessions,
        "remaining_pids": remaining,
        "ok": not remaining and not _qlattice_tmux_sessions(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
