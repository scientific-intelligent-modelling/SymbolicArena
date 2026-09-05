#!/usr/bin/env python3
"""停止并隔离误启动的 all_15alg_fullcpu_v1_formal 批次。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


HOSTS = ("iaaccn22", "iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29")
BATCH_NAME = "all_15alg_fullcpu_v1_formal"
CONTROLLER_SESSION = "stage5_all_15alg_fullcpu_v1"
SESSION_PREFIX = "all_conditions_cpu_v2_"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            exc.stdout or "",
            exc.stderr or "timeout",
        )


def _controller_pids() -> list[int]:
    matches: list[int] = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except OSError:
            continue
        if "run_e1_candidate200_12alg_load_queue.py" in command and BATCH_NAME in command:
            matches.append(int(process_dir.name))
    return sorted(matches)


def _stop_controller(*, dry_run: bool) -> dict[str, Any]:
    before = _controller_pids()
    tmux_before = _run(["tmux", "has-session", "-t", CONTROLLER_SESSION], timeout=20).returncode == 0
    if not dry_run:
        _run(["tmux", "kill-session", "-t", CONTROLLER_SESSION], timeout=20)
        for pid in before:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + 10
        while _controller_pids() and time.time() < deadline:
            time.sleep(0.5)
        for pid in _controller_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1)
    after = _controller_pids()
    tmux_after = _run(["tmux", "has-session", "-t", CONTROLLER_SESSION], timeout=20).returncode == 0
    return {
        "pids_before": before,
        "pids_after": after,
        "tmux_before": tmux_before,
        "tmux_after": tmux_after,
        "ok": bool(dry_run) or (not after and not tmux_after),
    }


def _host_ip(host: str) -> str:
    return host if host == "iaaccn22" else f"10.10.100.{host.removeprefix('iaaccn')}"


def _run_host_reaper(
    host: str,
    *,
    host_script: Path,
    targets_file: Path,
    audit_dir: Path,
    experiment_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    if host == "iaaccn22":
        command = [
            "python3",
            str(host_script),
            "--targets",
            str(targets_file),
            "--marker",
            str(experiment_root / "ABORTED_DO_NOT_USE.json"),
        ]
        if dry_run:
            command.append("--dry-run")
        completed = _run(command, timeout=180)
    else:
        ip = _host_ip(host)
        remote_dir = audit_dir
        mkdir = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ip, f"mkdir -p '{remote_dir}'"],
            timeout=30,
        )
        if mkdir.returncode != 0:
            return {"host": host, "stage": "mkdir", "ok": False, "returncode": mkdir.returncode}
        copy = _run(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                str(host_script),
                str(targets_file),
                f"{ip}:{remote_dir}/",
            ],
            timeout=120,
        )
        if copy.returncode != 0:
            return {"host": host, "stage": "copy", "ok": False, "returncode": copy.returncode}
        command = [
            "python3",
            str(remote_dir / host_script.name),
            "--targets",
            str(remote_dir / targets_file.name),
            "--marker",
            str(experiment_root / "ABORTED_DO_NOT_USE.json"),
        ]
        if dry_run:
            command.append("--dry-run")
        remote_command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            ip,
            " ".join(f"'{part}'" for part in command),
        ]
        completed = _run(remote_command, timeout=240)
    parsed: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    return {
        "host": host,
        "stage": "reap",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "result": parsed,
        "ok": completed.returncode == 0 and bool(parsed and parsed.get("ok")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--controller-log", type=Path, required=True)
    parser.add_argument("--host-script", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--asset-marker", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--protected-state", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    tasks = state.get("tasks")
    if state.get("batch_name") != BATCH_NAME or not isinstance(tasks, dict):
        raise SystemExit("state 不属于目标 fullcpu batch")
    if len(tasks) != 6607:
        raise SystemExit(f"fullcpu state 任务数应为 6607，实际 {len(tasks)}")

    protected_before = {
        str(path): _sha256(path) for path in args.protected_state if path.is_file()
    }
    controller = _stop_controller(dry_run=args.dry_run)
    if not controller["ok"]:
        raise SystemExit(f"controller 未完全停止: {controller}")

    timestamp = _now()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    backup_files: list[dict[str, Any]] = []
    for source in (args.state, args.latest, args.events, args.controller_log):
        if not source.is_file():
            continue
        destination = args.audit_dir / "backup" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            shutil.copy2(source, destination)
        backup_files.append(
            {
                "source": str(source),
                "source_sha256": _sha256(source),
                "backup": str(destination),
                "backup_sha256": _sha256(destination) if destination.is_file() else None,
            }
        )

    targets_file = args.audit_dir / "targets_all_6607.json"
    targets = [
        {"task_id": task_id, "session": f"{SESSION_PREFIX}{task_id}"}
        for task_id in sorted(tasks)
    ]
    target_payload = {
        "schema": "all_15alg_fullcpu_v1.stop_targets.v1",
        "batch_name": BATCH_NAME,
        "created_at": timestamp,
        "targets": targets,
    }
    if not args.dry_run:
        _atomic_json(targets_file, target_payload)

    host_results: dict[str, Any] = {}
    if args.dry_run:
        host_results = {host: {"host": host, "dry_run": True, "ok": True} for host in HOSTS}
    else:
        for host in HOSTS:
            host_results[host] = _run_host_reaper(
                host,
                host_script=args.host_script,
                targets_file=targets_file,
                audit_dir=args.audit_dir,
                experiment_root=args.experiment_root,
                dry_run=False,
            )

    before_counts = Counter(str(task.get("state")) for task in tasks.values())
    if not args.dry_run:
        for task in tasks.values():
            previous_state = str(task.get("state"))
            task["batch_disposition"] = "aborted_do_not_use"
            if previous_state not in {"done", "failed", "cancelled"}:
                task["state_before_batch_abort"] = previous_state
                task["state"] = "cancelled"
                task["cancelled_at"] = timestamp
                task["cancel_reason"] = "user_rejected_unrequested_fullcpu_rerun"
        state["batch_disposition"] = "aborted_do_not_use"
        state["aborted_at"] = timestamp
        state["abort_reason"] = "user_rejected_unrequested_fullcpu_rerun"
        state["updated_at"] = timestamp
        _atomic_json(args.state, state)
        latest_payload = {
            "batch_name": BATCH_NAME,
            "time": timestamp,
            "batch_disposition": "aborted_do_not_use",
            "task_states": dict(
                sorted(Counter(str(task.get("state")) for task in tasks.values()).items())
            ),
            "by_tool": {
                tool: dict(sorted(counts.items()))
                for tool, counts in sorted(
                    (
                        tool,
                        Counter(
                            str(task.get("state"))
                            for task in tasks.values()
                            if str(task.get("tool")) == tool
                        ),
                    )
                    for tool in {str(task.get("tool")) for task in tasks.values()}
                )
            },
            "by_noise_tag": {
                noise_tag: dict(sorted(counts.items()))
                for noise_tag, counts in sorted(
                    (
                        noise_tag,
                        Counter(
                            str(task.get("state"))
                            for task in tasks.values()
                            if str(task.get("noise_tag")) == noise_tag
                        ),
                    )
                    for noise_tag in {str(task.get("noise_tag")) for task in tasks.values()}
                )
            },
        }
        _atomic_json(args.latest, latest_payload)
        event = {
            "time": timestamp,
            "event": "batch_aborted",
            "batch_name": BATCH_NAME,
            "reason": state["abort_reason"],
            "target_count": len(targets),
        }
        with args.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        marker = {
            "schema": "all_15alg_fullcpu_v1.abort_marker.v1",
            "batch_name": BATCH_NAME,
            "disposition": "aborted_do_not_use",
            "aborted_at": timestamp,
            "reason": state["abort_reason"],
            "state_path": str(args.state),
            "state_sha256": _sha256(args.state),
        }
        _atomic_json(args.asset_marker, marker)

    after_counts = Counter(str(task.get("state")) for task in tasks.values())
    protected_after = {
        str(path): _sha256(path) for path in args.protected_state if path.is_file()
    }
    protected_unchanged = protected_before == protected_after
    hosts_with_live_targets = {
        str(task.get("assigned_host"))
        for task in tasks.values()
        if task.get("state") in {"running", "dispatching"} and task.get("assigned_host")
    } if args.dry_run else {
        str(task.get("assigned_host"))
        for task in json.loads((args.audit_dir / "backup" / args.state.name).read_text(encoding="utf-8"))["tasks"].values()
        if task.get("state") in {"running", "dispatching"} and task.get("assigned_host")
    }
    required_host_results_ok = all(host_results.get(host, {}).get("ok") for host in hosts_with_live_targets)
    audit = {
        "schema": "all_15alg_fullcpu_v1.batch_abort.v1",
        "batch_name": BATCH_NAME,
        "dry_run": bool(args.dry_run),
        "time": timestamp,
        "controller": controller,
        "task_count": len(tasks),
        "task_states_before": dict(sorted(before_counts.items())),
        "task_states_after": dict(sorted(after_counts.items())),
        "hosts_with_live_targets": sorted(hosts_with_live_targets),
        "host_results": host_results,
        "backup_files": backup_files,
        "protected_state_sha256_before": protected_before,
        "protected_state_sha256_after": protected_after,
        "protected_states_unchanged": protected_unchanged,
        "required_host_results_ok": required_host_results_ok,
        "state_sha256_after": _sha256(args.state),
        "ok": controller["ok"] and protected_unchanged and required_host_results_ok,
    }
    audit_path = args.audit_dir / "abort_audit.json"
    if not args.dry_run:
        _atomic_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
