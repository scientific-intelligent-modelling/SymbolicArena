#!/usr/bin/env python3
"""Probe4 full-664 负载感知队列调度器。

设计目标：
- `iaaccn23` 作为中心调度节点，维护 pending/running/done 状态。
- 调度粒度是单个 dataset × tool × seed 任务，机器空闲时持续领取任务。
- `iaaccn23~29` 每台机器默认最多 100 个并发任务。
- 每轮 poll 每台机器默认最多新增 2 个任务，用慢启动避免瞬时打爆机器。
- 调度器根据 CPU load ratio、内存使用率、已有 probe4 session 数决定是否派发任务。
- `timed_out` 只表示预算耗尽，不等价于失败；调度完成判定以 launcher
  的任务状态是否收口为准。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1"
SOURCE_CSV = ASSET_ROOT / "full664_unified.csv"
REMOTE_ROOT = Path("/home/zhangziwen/projects/scientific-intelligent-modelling")

DEFAULT_HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29")
DEFAULT_SEEDS = (520, 521, 522)
DEFAULT_TOOLS = ("pyoperon", "imcts", "dso", "udsr")
DONE_STATUSES = {"ok", "timed_out", "no_valid_output"}

TOOL_CONFIG: dict[str, dict[str, Any]] = {
    "pyoperon": {
        "tool_arg": "pyoperon",
        "params": "pyoperon",
        "env": "sim_base",
        "workers": 1,
        "task_size": 1,
    },
    "imcts": {
        "tool_arg": "iMCTS",
        "params": "imcts",
        "env": "sim_iMCTS",
        "workers": 1,
        "task_size": 1,
    },
    "dso": {
        "tool_arg": "dso",
        "params": "dso",
        "env": "sim_dso",
        "workers": 1,
        "task_size": 1,
    },
    "udsr": {
        "tool_arg": "udsr",
        "params": "udsr",
        "env": "sim_dso",
        "workers": 1,
        "task_size": 1,
    },
}


@dataclass(frozen=True)
class QueueTask:
    task_id: str
    tool: str
    seed: int
    task_index: int
    rows: list[dict[str, str]]
    slice_path: Path

    @property
    def expected(self) -> int:
        return len(self.rows)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, _safe_text(exc.stdout), _safe_text(exc.stderr) or "timeout")


def _host_number(host: str) -> str | None:
    suffix = host.removeprefix("iaaccn")
    return suffix if suffix.isdigit() else None


def _target_for_host(host: str, *, controller_host: str, use_internal_ips: bool) -> str:
    if host == controller_host:
        return host
    number = _host_number(host)
    if use_internal_ips and number is not None:
        return f"10.10.100.{number}"
    return host


def _is_local_host(host: str, controller_host: str) -> bool:
    local_names = {socket.gethostname(), socket.getfqdn(), "localhost", "127.0.0.1"}
    if host == controller_host:
        local_names.add(controller_host)
    short_names = {name.split(".")[0] for name in local_names}
    return host in local_names or host in short_names


def _ssh(host: str, command: str, *, controller_host: str, use_internal_ips: bool, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    if _is_local_host(host, controller_host):
        return _run(["bash", "-lc", command], timeout=timeout)
    target = _target_for_host(host, controller_host=controller_host, use_internal_ips=use_internal_ips)
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            target,
            command,
        ],
        timeout=timeout,
    )


def _scp(local_path: Path, host: str, remote_path: Path, *, controller_host: str, use_internal_ips: bool, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    if _is_local_host(host, controller_host):
        if local_path.resolve() == remote_path.resolve():
            return subprocess.CompletedProcess(["cp", str(local_path), str(remote_path)], 0, "same file", "")
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        return _run(["cp", str(local_path), str(remote_path)], timeout=timeout)
    target = _target_for_host(host, controller_host=controller_host, use_internal_ips=use_internal_ips)
    return _run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{target}:{remote_path}",
        ],
        timeout=timeout,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 664:
        raise ValueError(f"期望 full664_unified.csv 有 664 行，实际 {len(rows)} 行: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_tasks(rows: list[dict[str, str]], *, tools: list[str], seeds: list[int]) -> list[QueueTask]:
    tasks: list[QueueTask] = []
    for seed in seeds:
        for tool in tools:
            task_size = int(TOOL_CONFIG[tool]["task_size"])
            for task_index, start in enumerate(range(0, len(rows), task_size), start=1):
                task_rows = rows[start : start + task_size]
                global_index = task_rows[0].get("global_index", str(task_index))
                task_id = f"{tool}_s{seed}_g{int(global_index):04d}"
                slice_path = ASSET_ROOT / "load_queue" / "slices" / tool / f"seed{seed}" / f"{task_id}.csv"
                tasks.append(
                    QueueTask(
                        task_id=task_id,
                        tool=tool,
                        seed=seed,
                        task_index=task_index,
                        rows=task_rows,
                        slice_path=slice_path,
                    )
                )
    return tasks


def _materialize_slices(tasks: list[QueueTask]) -> None:
    for task in tasks:
        if not task.slice_path.exists():
            _write_csv(task.slice_path, task.rows)


def _remote_support_script_path() -> Path:
    return ASSET_ROOT / "load_queue" / "remote" / "run_queue_task.sh"


def _write_remote_support_script() -> Path:
    path = _remote_support_script_path()
    content = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 10 ]; then
  echo "Usage: $0 <batch> <task_id> <tool_key> <tool_arg> <seed> <workers> <env> <slice_rel> <params_rel> <host_label> [retry]" >&2
  exit 2
fi

BATCH_NAME="$1"
TASK_ID="$2"
TOOL_KEY="$3"
TOOL_ARG="$4"
SEED="$5"
WORKERS="$6"
ENV_NAME="$7"
SLICE_REL="$8"
PARAMS_REL="$9"
HOST_LABEL="${{10}}"
RETRY_MODE="${{11:-}}"
REMOTE_ROOT="{REMOTE_ROOT}"
EXTRA_ARGS=()
if [ "$RETRY_MODE" = "retry" ]; then
  EXTRA_ARGS+=(--retry-failed)
fi

cd "$REMOTE_ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

conda run -n "$ENV_NAME" python check/launch_e1_benchmark.py run \\
  --tool "$TOOL_ARG" \\
  --slice-csv "$REMOTE_ROOT/$SLICE_REL" \\
  --params-json "$REMOTE_ROOT/$PARAMS_REL" \\
  --output-root "$REMOTE_ROOT/experiments/$BATCH_NAME/$TOOL_KEY/seed$SEED/tasks/$TASK_ID/$HOST_LABEL" \\
  --seed "$SEED" \\
  --workers "$WORKERS" \\
  "${{EXTRA_ARGS[@]}}"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _state_path(batch_name: str) -> Path:
    return ASSET_ROOT / "load_queue" / "state" / f"{batch_name}.state.json"


def _summary_path(batch_name: str) -> Path:
    return ASSET_ROOT / "load_queue" / "state" / f"{batch_name}.latest.json"


def _log_path(batch_name: str) -> Path:
    return ASSET_ROOT / "load_queue" / "state" / f"{batch_name}.events.jsonl"


def _initial_state(batch_name: str, tasks: list[QueueTask]) -> dict[str, Any]:
    return {
        "batch_name": batch_name,
        "created_at": _now(),
        "updated_at": _now(),
        "tasks": {
            task.task_id: {
                "task_id": task.task_id,
                "tool": task.tool,
                "seed": task.seed,
                "task_index": task.task_index,
                "expected": task.expected,
                "state": "pending",
                "attempts": 0,
                "assigned_host": None,
                "session": None,
                "started_at": None,
                "ended_at": None,
                "status_counts": {},
                "error": None,
            }
            for task in tasks
        },
    }


def _load_or_init_state(batch_name: str, tasks: list[QueueTask]) -> dict[str, Any]:
    path = _state_path(batch_name)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = {task.task_id for task in tasks}
        actual_ids = set(state.get("tasks", {}))
        if expected_ids != actual_ids:
            raise SystemExit(
                f"已有 state 与当前任务粒度不一致，避免混跑: {path}. "
                "请换 batch-name，或确认后手动删除旧 state。"
            )
        return state
    state = _initial_state(batch_name, tasks)
    _save_state(state)
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    path = _state_path(state["batch_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_event(batch_name: str, payload: dict[str, Any]) -> None:
    path = _log_path(batch_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": _now(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _remote_mkdir(host: str, path: Path, *, controller_host: str, use_internal_ips: bool) -> None:
    result = _ssh(
        host,
        f"mkdir -p {shlex.quote(str(path))}",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{host} mkdir 失败: {result.stderr or result.stdout}")


def _sync_support_to_host(host: str, *, controller_host: str, use_internal_ips: bool) -> None:
    support = _write_remote_support_script()
    remote_support = REMOTE_ROOT / support.relative_to(REPO_ROOT)
    remote_launcher = REMOTE_ROOT / "check/launch_e1_benchmark.py"
    local_launcher = REPO_ROOT / "check/launch_e1_benchmark.py"
    local_slices = ASSET_ROOT / "load_queue" / "slices"
    remote_slices = REMOTE_ROOT / local_slices.relative_to(REPO_ROOT)
    param_pairs = []
    for config in TOOL_CONFIG.values():
        local_param = REPO_ROOT / "exp-planning/02.E1选择验证/generated/params" / f"{config['params']}.json"
        remote_param = REMOTE_ROOT / local_param.relative_to(REPO_ROOT)
        param_pairs.append((local_param, remote_param))

    _remote_mkdir(host, remote_support.parent, controller_host=controller_host, use_internal_ips=use_internal_ips)
    _remote_mkdir(host, remote_launcher.parent, controller_host=controller_host, use_internal_ips=use_internal_ips)
    _remote_mkdir(host, remote_slices.parent, controller_host=controller_host, use_internal_ips=use_internal_ips)
    for _, remote_path in param_pairs:
        _remote_mkdir(host, remote_path.parent, controller_host=controller_host, use_internal_ips=use_internal_ips)
    for local_path, remote_path in ((support, remote_support), (local_launcher, remote_launcher), *param_pairs):
        result = _scp(local_path, host, remote_path, controller_host=controller_host, use_internal_ips=use_internal_ips, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"{host} 同步 {local_path} 失败: {result.stderr or result.stdout}")
    if _is_local_host(host, controller_host):
        if local_slices.resolve() != remote_slices.resolve():
            result = _run(["rsync", "-a", f"{local_slices}/", f"{remote_slices}/"], timeout=180)
        else:
            result = subprocess.CompletedProcess(["rsync", str(local_slices), str(remote_slices)], 0, "same dir", "")
    else:
        target = _target_for_host(host, controller_host=controller_host, use_internal_ips=use_internal_ips)
        result = _run(
            [
                "rsync",
                "-a",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                f"{local_slices}/",
                f"{target}:{remote_slices}/",
            ],
            timeout=300,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{host} 同步 load_queue slices 失败: {result.stderr or result.stdout}")
    chmod = _ssh(
        host,
        f"chmod +x {shlex.quote(str(remote_support))}",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=20,
    )
    if chmod.returncode != 0:
        raise RuntimeError(f"{host} chmod 失败: {chmod.stderr or chmod.stdout}")


def _sync_task_slice(host: str, task: QueueTask, *, controller_host: str, use_internal_ips: bool) -> str:
    del host, controller_host, use_internal_ips
    rel = task.slice_path.relative_to(REPO_ROOT)
    return str(rel)


def _probe_host(host: str, *, controller_host: str, use_internal_ips: bool) -> dict[str, Any]:
    script = r"""
import json
import os
import subprocess

def mem_info():
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
        total_gb = values.get("MemTotal", 0) / 1024 / 1024
        available_gb = values.get("MemAvailable", 0) / 1024 / 1024
        used_ratio = 1.0 - (available_gb / total_gb) if total_gb else None
        return {
            "mem_total_gb": total_gb,
            "mem_available_gb": available_gb,
            "mem_used_ratio": used_ratio,
        }
    except Exception:
        return {
            "mem_total_gb": None,
            "mem_available_gb": None,
            "mem_used_ratio": None,
        }

def session_count(pattern):
    proc = subprocess.run(["bash", "-lc", "tmux ls 2>/dev/null || true"], text=True, capture_output=True)
    return sum(1 for line in proc.stdout.splitlines() if pattern in line)

load1, load5, load15 = os.getloadavg()
cpu_count = os.cpu_count() or 1
memory = mem_info()
print(json.dumps({
    "load1": load1,
    "load5": load5,
    "load15": load15,
    "cpu_count": cpu_count,
    "load_ratio": load1 / cpu_count,
    **memory,
    "queue_sessions": session_count("probe4_full664_queue_"),
    "probe4_sessions": session_count("probe4_full664"),
}))
"""
    command = f"python - <<'PY'\n{script}\nPY"
    result = _ssh(host, command, controller_host=controller_host, use_internal_ips=use_internal_ips, timeout=25)
    if result.returncode != 0:
        return {"host": host, "ok": False, "error": (result.stderr or result.stdout).strip()}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"host": host, "ok": False, "error": f"无法解析 host probe: {exc}; stdout={result.stdout!r}"}
    return {"host": host, "ok": True, **payload}


def _host_can_accept(host_state: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if not host_state.get("ok"):
        return False, str(host_state.get("error") or "host probe failed")
    if int(host_state.get("queue_sessions") or 0) >= args.max_jobs_per_host:
        return False, "queue session 已达上限"
    # controller 机器上会常驻一个 `probe4_full664_queue` 调度器 tmux，会被
    # probe4_sessions 统计到，但它不是实际任务，不应阻止该机器领取任务。
    controller_session_allowance = 1 if str(host_state.get("host")) == args.controller_host else 0
    if (
        not args.allow_existing_probe4
        and int(host_state.get("probe4_sessions") or 0) > int(host_state.get("queue_sessions") or 0) + controller_session_allowance
    ):
        return False, "存在非队列 probe4 session"
    if float(host_state.get("load_ratio") or 99.0) >= args.max_load_ratio:
        return False, f"load_ratio>={args.max_load_ratio}"
    mem_used = host_state.get("mem_used_ratio")
    if mem_used is not None and float(mem_used) >= args.max_memory_used_ratio:
        return False, f"mem_used_ratio>={args.max_memory_used_ratio}"
    mem_available = host_state.get("mem_available_gb")
    if args.min_free_mem_gb > 0 and mem_available is not None and float(mem_available) < args.min_free_mem_gb:
        return False, f"mem_available_gb<{args.min_free_mem_gb}"
    return True, "ok"


def _read_task_status(task: dict[str, Any], *, controller_host: str, use_internal_ips: bool) -> dict[str, Any]:
    host = str(task["assigned_host"])
    tool = str(task["tool"])
    seed = int(task["seed"])
    task_id = str(task["task_id"])
    status_path = (
        REMOTE_ROOT
        / "experiments"
        / str(task["batch_name"])
        / tool
        / f"seed{seed}"
        / "tasks"
        / task_id
        / host
        / "__launcher__/task_status.jsonl"
    )
    result = _ssh(
        host,
        f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))} || true",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=45,
    )
    if result.returncode != 0:
        return {"read_error": (result.stderr or result.stdout).strip(), "seen": 0, "done": 0, "errors": 0, "counts": {}}

    latest: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_key = item.get("task_key")
        if isinstance(task_key, str):
            latest[task_key] = item
    counts = Counter(str(item.get("status") or "unknown") for item in latest.values())
    done = sum(count for status, count in counts.items() if status in DONE_STATUSES)
    return {
        "read_error": None,
        "seen": len(latest),
        "done": done,
        "errors": counts.get("error", 0),
        "counts": dict(sorted(counts.items())),
    }


def _session_running(host: str, session: str, *, controller_host: str, use_internal_ips: bool) -> bool:
    result = _ssh(
        host,
        f"tmux has-session -t {shlex.quote(session)} >/dev/null 2>&1",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=15,
    )
    return result.returncode == 0


def _start_task_on_host(
    task: QueueTask,
    host: str,
    state_task: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    config = TOOL_CONFIG[task.tool]
    session = f"probe4_full664_queue_{task.task_id}"
    support_rel = _remote_support_script_path().relative_to(REPO_ROOT)
    support_remote = REMOTE_ROOT / support_rel
    slice_rel = _sync_task_slice(host, task, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips)
    params_rel = f"exp-planning/02.E1选择验证/generated/params/{config['params']}.json"
    retry = "retry" if int(state_task.get("attempts") or 0) > 0 else "noretry"
    command = (
        f"cd {shlex.quote(str(REMOTE_ROOT))} && "
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"/bin/bash {shlex.quote(str(support_remote))} "
        f"{shlex.quote(args.batch_name)} "
        f"{shlex.quote(task.task_id)} "
        f"{shlex.quote(task.tool)} "
        f"{shlex.quote(str(config['tool_arg']))} "
        f"{shlex.quote(str(task.seed))} "
        f"{shlex.quote(str(config['workers']))} "
        f"{shlex.quote(str(config['env']))} "
        f"{shlex.quote(slice_rel)} "
        f"{shlex.quote(params_rel)} "
        f"{shlex.quote(host)} "
        f"{shlex.quote(retry)}"
    )
    result = _ssh(host, command, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"{host} 启动 {task.task_id} 失败: {result.stderr or result.stdout}")
    state_task.update(
        {
            "state": "running",
            "attempts": int(state_task.get("attempts") or 0) + 1,
            "assigned_host": host,
            "session": session,
            "started_at": _now(),
            "ended_at": None,
            "error": None,
            "batch_name": args.batch_name,
        }
    )


def _update_running_tasks(state: dict[str, Any], args: argparse.Namespace) -> None:
    for task_id, task in state["tasks"].items():
        if task.get("state") != "running":
            continue
        host = str(task["assigned_host"])
        session = str(task["session"])
        if _session_running(host, session, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips):
            continue
        status = _read_task_status(task, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips)
        task["status_counts"] = status["counts"]
        expected = int(task["expected"])
        missing = max(0, expected - int(status["seen"]))
        if status["read_error"]:
            task.update({"state": "pending", "error": status["read_error"]})
            _append_event(args.batch_name, {"event": "task_status_read_failed", "task_id": task_id, "host": host, "error": status["read_error"]})
        elif status["done"] == expected and status["errors"] == 0:
            task.update({"state": "done", "ended_at": _now(), "error": None})
            _append_event(args.batch_name, {"event": "task_done", "task_id": task_id, "host": host, "status_counts": status["counts"]})
        elif int(task.get("attempts") or 0) <= args.retry_limit:
            task.update({"state": "pending", "error": f"未完整收口: missing={missing}, errors={status['errors']}"})
            _append_event(args.batch_name, {"event": "task_retry_pending", "task_id": task_id, "host": host, "status": status})
        else:
            task.update({"state": "failed", "ended_at": _now(), "error": f"超过重试上限: missing={missing}, errors={status['errors']}"})
            _append_event(args.batch_name, {"event": "task_failed", "task_id": task_id, "host": host, "status": status})


def _summarize_state(state: dict[str, Any], host_states: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    counts = Counter(task["state"] for task in state["tasks"].values())
    by_tool = Counter()
    by_tool_state: dict[str, Counter[str]] = {}
    for task in state["tasks"].values():
        tool = str(task["tool"])
        by_tool[tool] += 1
        by_tool_state.setdefault(tool, Counter())[str(task["state"])] += 1
    return {
        "batch_name": state["batch_name"],
        "time": _now(),
        "task_states": dict(sorted(counts.items())),
        "by_tool": {tool: dict(sorted(counter.items())) for tool, counter in sorted(by_tool_state.items())},
        "hosts": host_states or [],
    }


def _write_summary(state: dict[str, Any], host_states: list[dict[str, Any]] | None = None) -> None:
    summary = _summarize_state(state, host_states)
    path = _summary_path(state["batch_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pending_task_ids(state: dict[str, Any]) -> list[str]:
    return [
        task_id
        for task_id, task in state["tasks"].items()
        if task.get("state") == "pending"
    ]


def _run_scheduler(tasks: list[QueueTask], args: argparse.Namespace) -> None:
    task_map = {task.task_id: task for task in tasks}
    state = _load_or_init_state(args.batch_name, tasks)

    if args.dry_run:
        print(json.dumps(_summarize_state(state), ensure_ascii=False, indent=2))
        return

    ready_hosts: list[str] = []
    for host in args.hosts:
        try:
            _sync_support_to_host(host, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips)
            _append_event(args.batch_name, {"event": "support_synced", "host": host})
            ready_hosts.append(host)
        except Exception as exc:
            _append_event(args.batch_name, {"event": "support_sync_failed", "host": host, "error": repr(exc)})
    if not ready_hosts:
        raise SystemExit("没有任何机器完成支持文件和队列切片同步，停止调度。")

    while True:
        _update_running_tasks(state, args)
        host_states = [
            _probe_host(host, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips)
            for host in ready_hosts
        ]

        pending_ids = _pending_task_ids(state)
        dispatched = 0
        for host_state in sorted(host_states, key=lambda item: float(item.get("load_ratio") or 99.0)):
            if not pending_ids:
                break
            host = str(host_state["host"])
            can_accept, reason = _host_can_accept(host_state, args)
            if not can_accept:
                host_state["dispatch_skip_reason"] = reason
                continue
            active_sessions = int(host_state.get("queue_sessions") or 0)
            available_slots = max(0, min(args.max_new_jobs_per_host_per_poll, args.max_jobs_per_host - active_sessions))
            host_dispatched = 0
            for _ in range(available_slots):
                if not pending_ids:
                    break
                task_id = pending_ids.pop(0)
                task = task_map[task_id]
                state_task = state["tasks"][task_id]
                try:
                    _start_task_on_host(task, host, state_task, args)
                    dispatched += 1
                    host_dispatched += 1
                    _append_event(args.batch_name, {"event": "task_started", "task_id": task_id, "host": host, "tool": task.tool, "seed": task.seed})
                except Exception as exc:
                    state_task["error"] = repr(exc)
                    _append_event(args.batch_name, {"event": "task_start_failed", "task_id": task_id, "host": host, "error": repr(exc)})
            host_state["dispatched"] = host_dispatched

        _save_state(state)
        _write_summary(state, host_states)
        summary = _summarize_state(state, host_states)
        print(json.dumps({"event": "queue_poll", "dispatched": dispatched, **{k: v for k, v in summary.items() if k != "hosts"}}, ensure_ascii=False), flush=True)

        counts = Counter(task["state"] for task in state["tasks"].values())
        if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
            if counts.get("failed", 0) > 0:
                raise SystemExit(f"队列结束但存在 failed task: {dict(counts)}")
            print(json.dumps({"event": "queue_done", "batch_name": args.batch_name, "time": _now()}, ensure_ascii=False), flush=True)
            break
        if args.once:
            break
        time.sleep(args.poll_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe4 full-664 负载感知中心队列调度器")
    parser.add_argument("--batch-name", default=f"probe4_full664_load_queue_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    parser.add_argument("--tools", nargs="+", default=list(DEFAULT_TOOLS), choices=sorted(TOOL_CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--controller-host", default="iaaccn23")
    parser.add_argument("--use-internal-ips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-limit", type=int, default=1)
    parser.add_argument("--max-jobs-per-host", type=int, default=100)
    parser.add_argument("--max-new-jobs-per-host-per-poll", type=int, default=2)
    parser.add_argument("--max-load-ratio", type=float, default=0.80)
    parser.add_argument("--max-memory-used-ratio", type=float, default=0.80)
    parser.add_argument("--min-free-mem-gb", type=float, default=0.0)
    parser.add_argument("--allow-existing-probe4", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    unknown_tools = sorted(set(args.tools) - set(TOOL_CONFIG))
    if unknown_tools:
        raise SystemExit(f"未知工具: {unknown_tools}")
    rows = _read_rows(SOURCE_CSV)
    tasks = _build_tasks(rows, tools=args.tools, seeds=args.seeds)
    _materialize_slices(tasks)
    _write_remote_support_script()
    print(
        json.dumps(
            {
                "event": "queue_start",
                "batch_name": args.batch_name,
                "hosts": args.hosts,
                "tools": args.tools,
                "seeds": args.seeds,
                "tasks": len(tasks),
                "tool_config": TOOL_CONFIG,
                "dry_run": args.dry_run,
                "time": _now(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    _run_scheduler(tasks, args)


if __name__ == "__main__":
    main()
