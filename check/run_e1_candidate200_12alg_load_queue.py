#!/usr/bin/env python3
"""E1 Candidate-200 / Core50 / 12 算法负载感知队列调度器。

默认只在 `iaaccn23~29` 上调度。设计继承 Probe4 full-664 队列调度器：

- 调度粒度为 `dataset x tool x seed`，E1 默认 12 x 200 x 1；
  Core50 可通过 `--source-csv/--expected-rows/--queue-root` 复用同一调度器。
- 每台机器按整机 CPU load ratio 与内存使用率决定是否继续领取任务。
- 默认派发梯度：load < 50% 每轮补 10 个，load < 70% 补 5 个，
  load < 80% 补 2 个。
- `timed_out` 表示预算耗尽并正常收口，不直接视为调度失败。

本脚本包含三种安全层级：

1. `--dry-run`：只在本地构建任务队列，不连远端。
2. `--preflight-only`：只读检查远端 SSH、代码、数据、参数和环境。
3. 默认模式：同步切片/支持脚本并开始派发 tmux 任务。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证"
GENERATED_ROOT = E1_ROOT / "generated"
SOURCE_CSV = GENERATED_ROOT / "candidate200_unified.csv"
QUEUE_ROOT = GENERATED_ROOT / "candidate200_12alg_load_queue"
PARAMS_ROOT = GENERATED_ROOT / "params"
REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
REMOTE_DATA_ROOT = Path("/home/zhangziwen/sim-datasets-data")

DEFAULT_HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29")
DEFAULT_SEEDS = (1314,)
DONE_STATUSES = {"ok", "timed_out", "no_valid_output"}


TOOL_CONFIG: dict[str, dict[str, Any]] = {
    "gplearn": {"tool_arg": "gplearn", "params": "gplearn", "env": "sim_base", "workers": 1, "task_size": 1},
    "llmsr": {"tool_arg": "llmsr", "params": "llmsr", "env": "sim_llm", "workers": 1, "task_size": 1, "llm": True},
    "pyoperon": {"tool_arg": "pyoperon", "params": "pyoperon", "env": "sim_base", "workers": 1, "task_size": 1},
    "drsr": {"tool_arg": "drsr", "params": "drsr", "env": "sim_llm", "workers": 1, "task_size": 1, "llm": True},
    "pysr": {"tool_arg": "pysr", "params": "pysr", "env": "sim_base", "workers": 1, "task_size": 1},
    "dso": {"tool_arg": "dso", "params": "dso", "env": "sim_dso", "workers": 1, "task_size": 1},
    "tpsr": {"tool_arg": "tpsr", "params": "tpsr", "env": "sim_tpsr", "workers": 1, "task_size": 1},
    "e2esr": {"tool_arg": "e2esr", "params": "e2esr", "env": "sim_e2esr", "workers": 1, "task_size": 1},
    "qlattice": {"tool_arg": "QLattice", "params": "qlattice", "env": "sim_qLattice", "workers": 1, "task_size": 1},
    "imcts": {"tool_arg": "iMCTS", "params": "imcts", "env": "sim_iMCTS", "workers": 1, "task_size": 1},
    "udsr": {"tool_arg": "udsr", "params": "udsr", "env": "sim_dso", "workers": 1, "task_size": 1},
    "ragsr": {"tool_arg": "ragsr", "params": "ragsr", "env": "sim_ragsr", "workers": 1, "task_size": 1},
}

DEFAULT_TOOLS = tuple(TOOL_CONFIG)


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
    short_names = {name.split(".")[0] for name in local_names}
    del controller_host
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


def _read_rows(path: Path, *, expected_rows: int | None = 200) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"期望任务表有 {expected_rows} 行，实际 {len(rows)} 行: {path}")
    for index, row in enumerate(rows, start=1):
        # Core50 manifest 使用 core50_index；旧 E1 launcher 依赖 global_index。
        if not row.get("global_index"):
            row["global_index"] = row.get("core50_index") or str(index)
        if not row.get("dataset_rel"):
            row["dataset_rel"] = row.get("dataset_dir") or ""
        if not row.get("dataset_name"):
            row["dataset_name"] = row.get("dataset_id") or row.get("basename") or f"dataset_{index}"
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"拒绝写空切片: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_tasks(rows: list[dict[str, str]], *, tools: list[str], seeds: list[int], queue_root: Path) -> list[QueueTask]:
    tasks: list[QueueTask] = []
    for seed in seeds:
        for tool in tools:
            task_size = int(TOOL_CONFIG[tool]["task_size"])
            for task_index, start in enumerate(range(0, len(rows), task_size), start=1):
                task_rows = rows[start : start + task_size]
                global_index = task_rows[0].get("global_index", str(task_index))
                task_id = f"{tool}_s{seed}_g{int(global_index):04d}"
                slice_path = queue_root / "slices" / tool / f"seed{seed}" / f"{task_id}.csv"
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


def _remote_support_script_path(queue_root: Path) -> Path:
    return queue_root / "remote" / "run_queue_task.sh"


def _write_remote_support_script(queue_root: Path) -> Path:
    path = _remote_support_script_path(queue_root)
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


def _state_path(batch_name: str, queue_root: Path) -> Path:
    return queue_root / "state" / f"{batch_name}.state.json"


def _summary_path(batch_name: str, queue_root: Path) -> Path:
    return queue_root / "state" / f"{batch_name}.latest.json"


def _log_path(batch_name: str, queue_root: Path) -> Path:
    return queue_root / "state" / f"{batch_name}.events.jsonl"


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


def _load_or_init_state(batch_name: str, tasks: list[QueueTask], queue_root: Path) -> dict[str, Any]:
    path = _state_path(batch_name, queue_root)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = {task.task_id for task in tasks}
        actual_ids = set(state.get("tasks", {}))
        if expected_ids != actual_ids:
            raise SystemExit(
                f"已有 state 与当前任务集合不一致，避免混跑: {path}. "
                "请换 batch-name，或确认后手动删除旧 state。"
            )
        return state
    state = _initial_state(batch_name, tasks)
    _save_state(state, queue_root)
    return state


def _save_state(state: dict[str, Any], queue_root: Path) -> None:
    state["updated_at"] = _now()
    path = _state_path(state["batch_name"], queue_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_event(batch_name: str, payload: dict[str, Any], queue_root: Path) -> None:
    path = _log_path(batch_name, queue_root)
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


def _selected_envs(tools: list[str]) -> list[str]:
    return sorted({str(TOOL_CONFIG[tool]["env"]) for tool in tools})


def _selected_params(tools: list[str], params_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for tool in tools:
        params_name = str(TOOL_CONFIG[tool]["params"])
        if params_name in seen:
            continue
        seen.add(params_name)
        local_param = params_root / f"{params_name}.json"
        remote_param = REMOTE_ROOT / local_param.relative_to(REPO_ROOT)
        pairs.append((local_param, remote_param))
    return pairs


def _sync_support_to_host(
    host: str,
    *,
    tools: list[str],
    queue_root: Path,
    params_root: Path,
    controller_host: str,
    use_internal_ips: bool,
) -> None:
    support = _write_remote_support_script(queue_root)
    remote_support = REMOTE_ROOT / support.relative_to(REPO_ROOT)
    remote_launcher = REMOTE_ROOT / "check/launch_e1_benchmark.py"
    local_launcher = REPO_ROOT / "check/launch_e1_benchmark.py"
    local_slices = queue_root / "slices"
    remote_slices = REMOTE_ROOT / local_slices.relative_to(REPO_ROOT)
    param_pairs = _selected_params(tools, params_root)

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
        raise RuntimeError(f"{host} 同步 queue slices 失败: {result.stderr or result.stdout}")

    chmod = _ssh(
        host,
        f"chmod +x {shlex.quote(str(remote_support))}",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=20,
    )
    if chmod.returncode != 0:
        raise RuntimeError(f"{host} chmod 失败: {chmod.stderr or chmod.stdout}")


def _sync_task_slice(task: QueueTask) -> str:
    return str(task.slice_path.relative_to(REPO_ROOT))


def _probe_host(host: str, *, controller_host: str, use_internal_ips: bool, session_prefix: str) -> dict[str, Any]:
    script = rf"""
import json
import os
import subprocess

def mem_info():
    try:
        values = {{}}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
        total_gb = values.get("MemTotal", 0) / 1024 / 1024
        available_gb = values.get("MemAvailable", 0) / 1024 / 1024
        used_ratio = 1.0 - (available_gb / total_gb) if total_gb else None
        return {{
            "mem_total_gb": total_gb,
            "mem_available_gb": available_gb,
            "mem_used_ratio": used_ratio,
        }}
    except Exception:
        return {{
            "mem_total_gb": None,
            "mem_available_gb": None,
            "mem_used_ratio": None,
        }}

def session_count(pattern):
    proc = subprocess.run(["bash", "-lc", "tmux ls 2>/dev/null || true"], text=True, capture_output=True)
    return sum(1 for line in proc.stdout.splitlines() if pattern in line)

load1, load5, load15 = os.getloadavg()
cpu_count = os.cpu_count() or 1
memory = mem_info()
prefix = {session_prefix!r}
print(json.dumps({{
    "load1": load1,
    "load5": load5,
    "load15": load15,
    "cpu_count": cpu_count,
    "load_ratio": load1 / cpu_count,
    **memory,
    "queue_sessions": session_count(prefix),
}}))
"""
    result = _ssh(host, f"python - <<'PY'\n{script}\nPY", controller_host=controller_host, use_internal_ips=use_internal_ips, timeout=25)
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
    if float(host_state.get("load_ratio") or 99.0) >= args.max_load_ratio:
        return False, f"load_ratio>={args.max_load_ratio}"
    mem_used = host_state.get("mem_used_ratio")
    if mem_used is not None and float(mem_used) >= args.max_memory_used_ratio:
        return False, f"mem_used_ratio>={args.max_memory_used_ratio}"
    mem_available = host_state.get("mem_available_gb")
    if args.min_free_mem_gb > 0 and mem_available is not None and float(mem_available) < args.min_free_mem_gb:
        return False, f"mem_available_gb<{args.min_free_mem_gb}"
    return True, "ok"


def _parse_load_tiers(raw: str) -> list[tuple[float, int]]:
    if not raw.strip():
        return []
    tiers: list[tuple[float, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"负载分段格式错误，缺少 ':': {item!r}")
        threshold_raw, jobs_raw = item.split(":", 1)
        threshold = float(threshold_raw)
        jobs = int(jobs_raw)
        if not 0 < threshold <= 1:
            raise ValueError(f"负载阈值必须在 (0, 1] 内: {threshold}")
        if jobs < 0:
            raise ValueError(f"新增任务数不能为负数: {jobs}")
        tiers.append((threshold, jobs))
    tiers.sort(key=lambda pair: pair[0])
    return tiers


def _max_new_jobs_for_host(host_state: dict[str, Any], args: argparse.Namespace) -> tuple[int, str]:
    active_sessions = int(host_state.get("queue_sessions") or 0)
    hard_slots = max(0, args.max_jobs_per_host - active_sessions)
    if hard_slots <= 0:
        return 0, "no_hard_slot"
    load_ratio = float(host_state.get("load_ratio") or 99.0)
    for threshold, jobs in args.load_tier_new_jobs_parsed:
        if load_ratio < threshold:
            return min(jobs, hard_slots), f"load<{threshold:g}:jobs={jobs}"
    if args.load_tier_new_jobs_parsed:
        return 0, "load_not_in_tiers"
    return min(args.max_new_jobs_per_host_per_poll, hard_slots), "fixed_max_new_jobs"


def _read_task_statuses_bulk(
    host: str,
    tasks: list[dict[str, Any]],
    *,
    controller_host: str,
    use_internal_ips: bool,
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    task_specs = [
        {
            "task_id": str(task["task_id"]),
            "tool": str(task["tool"]),
            "seed": int(task["seed"]),
            "host": str(task["assigned_host"]),
            "batch_name": str(task["batch_name"]),
        }
        for task in tasks
    ]
    script = f"""
import json
from collections import Counter
from pathlib import Path

REMOTE_ROOT = Path({str(REMOTE_ROOT)!r})
DONE_STATUSES = {sorted(DONE_STATUSES)!r}
TASKS = json.loads({json.dumps(task_specs, ensure_ascii=False)!r})
out = {{}}

for task in TASKS:
    status_path = (
        REMOTE_ROOT
        / "experiments"
        / task["batch_name"]
        / task["tool"]
        / f"seed{{task['seed']}}"
        / "tasks"
        / task["task_id"]
        / task["host"]
        / "__launcher__/task_status.jsonl"
    )
    latest = {{}}
    try:
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                task_key = item.get("task_key")
                if isinstance(task_key, str):
                    latest[task_key] = item
        counts = Counter(str(item.get("status") or "unknown") for item in latest.values())
        done = sum(count for status, count in counts.items() if status in DONE_STATUSES)
        out[task["task_id"]] = {{
            "read_error": None,
            "seen": len(latest),
            "done": done,
            "errors": counts.get("error", 0),
            "counts": dict(sorted(counts.items())),
        }}
    except Exception as exc:
        out[task["task_id"]] = {{
            "read_error": repr(exc),
            "seen": 0,
            "done": 0,
            "errors": 0,
            "counts": {{}},
        }}

print(json.dumps(out, ensure_ascii=False))
"""
    result = _ssh(
        host,
        f"python - <<'PY'\n{script}\nPY",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=max(60, 2 * len(tasks)),
    )
    if result.returncode != 0:
        return {
            str(task["task_id"]): {
                "read_error": (result.stderr or result.stdout).strip(),
                "seen": 0,
                "done": 0,
                "errors": 0,
                "counts": {},
            }
            for task in tasks
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            str(task["task_id"]): {
                "read_error": f"bulk status parse failed: {exc}; stdout={result.stdout[:500]!r}",
                "seen": 0,
                "done": 0,
                "errors": 0,
                "counts": {},
            }
            for task in tasks
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


def _list_queue_sessions(host: str, *, controller_host: str, use_internal_ips: bool, session_prefix: str) -> set[str] | None:
    result = _ssh(
        host,
        f"tmux ls 2>/dev/null | cut -d: -f1 | grep '^{shlex.quote(session_prefix)}' || true",
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
        timeout=25,
    )
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _start_task_on_host(task: QueueTask, host: str, state_task: dict[str, Any], args: argparse.Namespace) -> None:
    config = TOOL_CONFIG[task.tool]
    session = f"{args.session_prefix}{task.task_id}"
    support_rel = _remote_support_script_path(args.queue_root_path).relative_to(REPO_ROOT)
    support_remote = REMOTE_ROOT / support_rel
    slice_rel = _sync_task_slice(task)
    params_rel = str((args.params_root_path / f"{config['params']}.json").relative_to(REPO_ROOT))
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
    running_items = [(task_id, task) for task_id, task in state["tasks"].items() if task.get("state") == "running"]
    sessions_by_host: dict[str, set[str] | None] = {}
    for _, task in running_items:
        host = str(task["assigned_host"])
        if host not in sessions_by_host:
            sessions_by_host[host] = _list_queue_sessions(
                host,
                controller_host=args.controller_host,
                use_internal_ips=args.use_internal_ips,
                session_prefix=args.session_prefix,
            )

    finished_items: list[tuple[str, dict[str, Any]]] = []
    for task_id, task in running_items:
        host = str(task["assigned_host"])
        session = str(task["session"])
        host_sessions = sessions_by_host.get(host)
        if host_sessions is not None:
            if session in host_sessions:
                continue
        elif _session_running(host, session, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips):
            continue
        finished_items.append((task_id, task))

    finished_by_host: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for item in finished_items:
        finished_by_host[str(item[1]["assigned_host"])].append(item)

    statuses: dict[str, dict[str, Any]] = {}
    for host, items in finished_by_host.items():
        host_statuses = _read_task_statuses_bulk(
            host,
            [task for _, task in items],
            controller_host=args.controller_host,
            use_internal_ips=args.use_internal_ips,
        )
        statuses.update(host_statuses)

    for task_id, task in finished_items:
        host = str(task["assigned_host"])
        status = statuses.get(task_id) or {}
        task["status_counts"] = status.get("counts", {})
        expected = int(task["expected"])
        missing = max(0, expected - int(status.get("seen") or 0))
        errors = int(status.get("errors") or 0)
        if status.get("read_error"):
            task.update({"state": "pending", "error": status["read_error"]})
            _append_event(args.batch_name, {"event": "task_status_read_failed", "task_id": task_id, "host": host, "error": status["read_error"]}, args.queue_root_path)
        elif int(status.get("done") or 0) == expected and errors == 0:
            task.update({"state": "done", "ended_at": _now(), "error": None})
            _append_event(args.batch_name, {"event": "task_done", "task_id": task_id, "host": host, "status_counts": status.get("counts", {})}, args.queue_root_path)
        elif int(task.get("attempts") or 0) <= args.retry_limit:
            task.update({"state": "pending", "error": f"未完整收口: missing={missing}, errors={errors}"})
            _append_event(args.batch_name, {"event": "task_retry_pending", "task_id": task_id, "host": host, "status": status}, args.queue_root_path)
        else:
            task.update({"state": "failed", "ended_at": _now(), "error": f"超过重试上限: missing={missing}, errors={errors}"})
            _append_event(args.batch_name, {"event": "task_failed", "task_id": task_id, "host": host, "status": status}, args.queue_root_path)


def _summarize_state(state: dict[str, Any], host_states: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    counts = Counter(task["state"] for task in state["tasks"].values())
    by_tool_state: dict[str, Counter[str]] = {}
    for task in state["tasks"].values():
        tool = str(task["tool"])
        by_tool_state.setdefault(tool, Counter())[str(task["state"])] += 1
    return {
        "batch_name": state["batch_name"],
        "time": _now(),
        "task_states": dict(sorted(counts.items())),
        "by_tool": {tool: dict(sorted(counter.items())) for tool, counter in sorted(by_tool_state.items())},
        "hosts": host_states or [],
    }


def _write_summary(state: dict[str, Any], host_states: list[dict[str, Any]] | None = None) -> None:
    path = _summary_path(state["batch_name"], Path(state.get("queue_root") or QUEUE_ROOT))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_summarize_state(state, host_states), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pending_task_ids(state: dict[str, Any]) -> list[str]:
    return [task_id for task_id, task in state["tasks"].items() if task.get("state") == "pending"]


def _pending_task_ids_by_tool(state: dict[str, Any], tool: str) -> list[str]:
    return [
        task_id
        for task_id, task in state["tasks"].items()
        if task.get("state") == "pending" and task.get("tool") == tool
    ]


def _running_count_for_tool(state: dict[str, Any], tool: str) -> int:
    return sum(1 for task in state["tasks"].values() if task.get("state") == "running" and task.get("tool") == tool)


def _next_pending_task_id(state: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not args.round_robin_tools:
        pending = _pending_task_ids(state)
        return pending[0] if pending else None
    tools = list(args.tools)
    start = int(state.get("round_robin_cursor") or 0)
    for offset in range(len(tools)):
        idx = (start + offset) % len(tools)
        tool = tools[idx]
        max_running = int(TOOL_CONFIG[tool].get("max_running") or args.default_max_running_per_tool)
        if max_running > 0 and _running_count_for_tool(state, tool) >= max_running:
            continue
        pending = _pending_task_ids_by_tool(state, tool)
        if pending:
            state["round_robin_cursor"] = (idx + 1) % len(tools)
            return pending[0]
    return None


def _run_scheduler(tasks: list[QueueTask], args: argparse.Namespace) -> None:
    task_map = {task.task_id: task for task in tasks}
    state = _load_or_init_state(args.batch_name, tasks, args.queue_root_path)
    state["queue_root"] = str(args.queue_root_path)

    if args.dry_run:
        print(json.dumps(_summarize_state(state), ensure_ascii=False, indent=2))
        return

    ready_hosts: list[str] = []
    for host in args.hosts:
        try:
            _sync_support_to_host(
                host,
                tools=args.tools,
                queue_root=args.queue_root_path,
                params_root=args.params_root_path,
                controller_host=args.controller_host,
                use_internal_ips=args.use_internal_ips,
            )
            _append_event(args.batch_name, {"event": "support_synced", "host": host}, args.queue_root_path)
            ready_hosts.append(host)
        except Exception as exc:
            _append_event(args.batch_name, {"event": "support_sync_failed", "host": host, "error": repr(exc)}, args.queue_root_path)
    if not ready_hosts:
        raise SystemExit("没有任何机器完成支持文件和队列切片同步，停止调度。")

    while True:
        _update_running_tasks(state, args)
        host_states = [
            _probe_host(host, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips, session_prefix=args.session_prefix)
            for host in ready_hosts
        ]

        dispatched = 0
        for host_state in sorted(host_states, key=lambda item: float(item.get("load_ratio") or 99.0)):
            host = str(host_state["host"])
            can_accept, reason = _host_can_accept(host_state, args)
            if not can_accept:
                host_state["dispatch_skip_reason"] = reason
                continue
            available_slots, dispatch_limit_reason = _max_new_jobs_for_host(host_state, args)
            host_state["dispatch_limit_reason"] = dispatch_limit_reason
            host_state["available_slots"] = available_slots
            host_dispatched = 0
            for _ in range(available_slots):
                task_id = _next_pending_task_id(state, args)
                if task_id is None:
                    break
                task = task_map[task_id]
                state_task = state["tasks"][task_id]
                try:
                    _start_task_on_host(task, host, state_task, args)
                    dispatched += 1
                    host_dispatched += 1
                    _append_event(args.batch_name, {"event": "task_started", "task_id": task_id, "host": host, "tool": task.tool, "seed": task.seed}, args.queue_root_path)
                except Exception as exc:
                    state_task["error"] = repr(exc)
                    _append_event(args.batch_name, {"event": "task_start_failed", "task_id": task_id, "host": host, "error": repr(exc)}, args.queue_root_path)
                    # 启动失败不立即丢弃任务，下轮继续尝试。
                    break
            host_state["dispatched"] = host_dispatched

        _save_state(state, args.queue_root_path)
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


def _write_preflight_script() -> Path:
    path = QUEUE_ROOT / "preflight" / "remote_preflight.py"
    content = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
REMOTE_DATA_ROOT = Path("/home/zhangziwen/sim-datasets-data")
SOURCE_CSV = REMOTE_ROOT / "exp-planning/02.E1选择验证/generated/candidate200_unified.csv"
PARAMS_ROOT = REMOTE_ROOT / "exp-planning/02.E1选择验证/generated/params"

ENV_IMPORTS = {
    "sim_base": [
        "scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper",
    ],
    "sim_llm": [
        "scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper",
    ],
    "sim_dso": [
        "scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper",
        "scientific_intelligent_modelling.algorithms.udsr_wrapper.wrapper",
    ],
    "sim_tpsr": ["scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper"],
    "sim_e2esr": ["scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper"],
    "sim_qLattice": ["scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper"],
    "sim_iMCTS": ["scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper"],
    "sim_ragsr": ["scientific_intelligent_modelling.algorithms.ragsr_wrapper.wrapper"],
}


def run(cmd: str, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "stdout": str(exc.stdout or "")[-4000:], "stderr": str(exc.stderr or "timeout")[-4000:]}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidate_rows() -> tuple[list[dict], dict]:
    if not SOURCE_CSV.exists():
        return [], {"exists": False, "path": str(SOURCE_CSV)}
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = []
    for row in rows:
        dataset_dir = Path(row.get("dataset_dir") or "")
        if not dataset_dir.is_absolute():
            if dataset_dir.parts and dataset_dir.parts[0] == "sim-datasets-data":
                dataset_dir = REMOTE_DATA_ROOT.joinpath(*dataset_dir.parts[1:])
            else:
                dataset_dir = REMOTE_DATA_ROOT / dataset_dir
        required = ["metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv"]
        absent = [name for name in required if not (dataset_dir / name).exists()]
        if absent:
            missing.append({"dataset": row.get("dataset_name"), "dataset_dir": str(dataset_dir), "missing": absent})
    return rows, {
        "exists": True,
        "path": str(SOURCE_CSV),
        "row_count": len(rows),
        "dataset_missing_count": len(missing),
        "dataset_missing_examples": missing[:10],
    }


def remote_dataset_dir(dataset_rel: str) -> Path:
    path = Path(dataset_rel)
    if path.is_absolute():
        try:
            return REMOTE_DATA_ROOT / path.relative_to("/home/zhangziwen/sim-datasets-data")
        except ValueError:
            return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        return REMOTE_DATA_ROOT.joinpath(*path.parts[1:])
    return REMOTE_DATA_ROOT / path


def file_fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size": None, "sha256": None}
    return {"exists": True, "size": path.stat().st_size, "sha256": sha256(path)}


def check_dataset_sync(expected: list[dict]) -> dict:
    mismatches = []
    compared_files = 0
    missing_files = 0
    hash_mismatch_files = 0
    size_mismatch_files = 0
    for item in expected:
        dataset_rel = str(item.get("dataset_rel") or "")
        dataset_dir = remote_dataset_dir(dataset_rel)
        for name, expected_fp in (item.get("files") or {}).items():
            compared_files += 1
            actual = file_fingerprint(dataset_dir / name)
            problems = []
            if not actual["exists"]:
                missing_files += 1
                problems.append("missing")
            else:
                if actual["size"] != expected_fp.get("size"):
                    size_mismatch_files += 1
                    problems.append("size_mismatch")
                if actual["sha256"] != expected_fp.get("sha256"):
                    hash_mismatch_files += 1
                    problems.append("sha256_mismatch")
            if problems and len(mismatches) < 20:
                mismatches.append(
                    {
                        "dataset": item.get("dataset_name"),
                        "dataset_rel": dataset_rel,
                        "file": name,
                        "remote_path": str(dataset_dir / name),
                        "problems": problems,
                        "expected": expected_fp,
                        "actual": actual,
                    }
                )
    return {
        "expected_datasets": len(expected),
        "compared_files": compared_files,
        "missing_files": missing_files,
        "size_mismatch_files": size_mismatch_files,
        "hash_mismatch_files": hash_mismatch_files,
        "all_match": missing_files == 0 and size_mismatch_files == 0 and hash_mismatch_files == 0,
        "mismatch_examples": mismatches,
    }


def check_params(tools: list[str]) -> dict:
    out = {}
    for tool in tools:
        path = PARAMS_ROOT / f"{tool}.json"
        item = {"exists": path.exists(), "path": str(path)}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                item["json_ok"] = True
                if tool in {"llmsr", "drsr"}:
                    item["inject_prompt_semantics"] = payload.get("inject_prompt_semantics")
                    item["canonical_prompt_variables"] = payload.get("canonical_prompt_variables")
                    item["has_background_override"] = "background" in payload
                    llm_config = payload.get("llm_config_path")
                    item["llm_config_path"] = llm_config
                    item["llm_config_exists"] = bool(llm_config and Path(str(llm_config)).exists())
            except Exception as exc:
                item["json_ok"] = False
                item["error"] = repr(exc)
        out[tool] = item
    return out


def check_envs(envs: list[str]) -> dict:
    conda_envs = run("conda env list", timeout=30)
    out = {"conda_env_list_returncode": conda_envs["returncode"], "envs": {}}
    for env in envs:
        modules = ENV_IMPORTS.get(env, [])
        code = "import importlib\n" + "\n".join(f"importlib.import_module({m!r})" for m in modules) + "\nprint('import_ok')"
        tmp_path = Path(tempfile.gettempdir()) / f"e1_candidate200_import_check_{env}.py"
        tmp_path.write_text(code, encoding="utf-8")
        cmd = (
            f"cd {shlex.quote(str(REMOTE_ROOT))} && "
            f"PYTHONPATH=. conda run -n {shlex.quote(env)} python {shlex.quote(str(tmp_path))}"
        )
        try:
            result = run(cmd, timeout=120)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        out["envs"][env] = {
            "required_modules": modules,
            "returncode": result["returncode"],
            "ok": result["returncode"] == 0 and "import_ok" in result["stdout"],
            "stdout_tail": result["stdout"][-1000:],
            "stderr_tail": result["stderr"][-1000:],
        }
    return out


def main() -> None:
    request_arg = sys.argv[1]
    request_path = Path(request_arg)
    if request_path.exists():
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(request_arg)
    tools = payload["tools"]
    envs = payload["envs"]
    local_head = payload.get("local_head")
    local_files = payload.get("local_files", {})
    local_hashes = payload.get("local_hashes", {})
    expected_dataset_fingerprints = payload.get("dataset_fingerprints", [])

    rows, dataset_report = load_candidate_rows()
    git_head = run(f"git -C {REMOTE_ROOT} rev-parse HEAD", timeout=20)
    git_status = run(f"git -C {REMOTE_ROOT} status --short", timeout=30)
    remote_head = git_head["stdout"].strip().splitlines()[-1] if git_head["returncode"] == 0 and git_head["stdout"].strip() else None
    files = {}
    for label, rel_path in local_files.items():
        remote_sha = sha256(REMOTE_ROOT / rel_path)
        expected_sha = local_hashes.get(label)
        files[label] = {
            "path": str(REMOTE_ROOT / rel_path),
            "sha256": remote_sha,
            "expected_sha256": expected_sha,
            "matches_expected": bool(expected_sha and remote_sha == expected_sha),
        }

    print(json.dumps({
        "host": payload.get("host"),
        "remote_root": str(REMOTE_ROOT),
        "git": {
            "remote_head": remote_head,
            "local_head": local_head,
            "head_matches_local": bool(local_head and remote_head == local_head),
            "status_short_nonempty": bool(git_status["stdout"].strip()),
            "status_short_preview": git_status["stdout"].splitlines()[:20],
        },
        "files": files,
        "candidate200": dataset_report,
        "dataset_sync": check_dataset_sync(expected_dataset_fingerprints),
        "params": check_params(tools),
        "envs": check_envs(envs),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _local_git_head() -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], timeout=20)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _local_file_hashes() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for label, rel in _preflight_local_files().items():
        path = REPO_ROOT / rel
        if not path.exists():
            out[label] = None
            continue
        out[label] = _sha256_file(path)
    return out


def _preflight_local_files() -> dict[str, str]:
    rels = {
        "scheduler": "check/run_e1_candidate200_12alg_load_queue.py",
        "launcher": "check/launch_e1_benchmark.py",
        "runner": "scientific_intelligent_modelling/benchmarks/runner.py",
        "toolbox_config": "scientific_intelligent_modelling/config/toolbox_config.json",
        "candidate200": "exp-planning/02.E1选择验证/generated/candidate200_unified.csv",
    }
    for tool in sorted(TOOL_CONFIG):
        rels[f"{tool}_params"] = f"exp-planning/02.E1选择验证/generated/params/{TOOL_CONFIG[tool]['params']}.json"
    return rels


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_local_dataset_dir(row: dict[str, str]) -> Path:
    dataset_rel = row.get("dataset_rel") or row.get("dataset_dir") or ""
    path = Path(dataset_rel)
    if path.is_absolute():
        try:
            rel = path.relative_to("/home/zhangziwen/sim-datasets-data")
            return REPO_ROOT / "sim-datasets-data" / rel
        except ValueError:
            return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        return REPO_ROOT / path
    return REPO_ROOT / "sim-datasets-data" / path


def _local_candidate_data_fingerprints(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    required = ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        dataset_rel = row.get("dataset_rel") or row.get("dataset_dir") or ""
        dataset_dir = _resolve_local_dataset_dir(row)
        files: dict[str, dict[str, Any]] = {}
        for name in required:
            path = dataset_dir / name
            files[name] = {
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
                "sha256": _sha256_file(path) if path.exists() else None,
            }
        out.append(
            {
                "dataset_name": row.get("dataset_name") or row.get("dataset_id"),
                "dataset_rel": dataset_rel,
                "files": files,
            }
        )
    return out


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    script = _write_preflight_script()
    remote_script = Path("/tmp/e1_candidate200_12alg_preflight.py")
    rows = _read_rows(SOURCE_CSV)
    local_files = _preflight_local_files()
    request = {
        "tools": args.tools,
        "envs": _selected_envs(args.tools),
        "local_head": _local_git_head(),
        "local_hashes": _local_file_hashes(),
        "local_files": local_files,
        "dataset_fingerprints": _local_candidate_data_fingerprints(rows),
    }
    host_reports = []
    for host in args.hosts:
        copy = _scp(script, host, remote_script, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips, timeout=60)
        if copy.returncode != 0:
            host_reports.append({"host": host, "ok": False, "stage": "scp_preflight_script", "error": copy.stderr or copy.stdout})
            continue
        payload = {**request, "host": host}
        local_request = QUEUE_ROOT / "preflight" / f"request_{host}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        local_request.parent.mkdir(parents=True, exist_ok=True)
        local_request.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        remote_request = Path(f"/tmp/e1_candidate200_12alg_preflight_request_{host}.json")
        copy_request = _scp(
            local_request,
            host,
            remote_request,
            controller_host=args.controller_host,
            use_internal_ips=args.use_internal_ips,
            timeout=60,
        )
        if copy_request.returncode != 0:
            host_reports.append({"host": host, "ok": False, "stage": "scp_preflight_request", "error": copy_request.stderr or copy_request.stdout})
            continue
        command = f"python {shlex.quote(str(remote_script))} {shlex.quote(str(remote_request))}"
        result = _ssh(host, command, controller_host=args.controller_host, use_internal_ips=args.use_internal_ips, timeout=args.preflight_host_timeout)
        if result.returncode != 0:
            host_reports.append({"host": host, "ok": False, "stage": "run_preflight", "error": result.stderr or result.stdout})
            continue
        try:
            report = json.loads(result.stdout.strip().splitlines()[-1])
            report["ok"] = True
            host_reports.append(report)
        except Exception as exc:
            host_reports.append({"host": host, "ok": False, "stage": "parse_preflight", "error": repr(exc), "stdout_tail": result.stdout[-2000:]})
    summary = {
        "time": _now(),
        "hosts": host_reports,
        "requested_tools": args.tools,
        "requested_envs": _selected_envs(args.tools),
    }
    if args.preflight_report:
        report_path = Path(args.preflight_report)
    else:
        report_path = QUEUE_ROOT / "preflight" / f"preflight_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 Candidate-200 / 12 算法负载感知队列调度器")
    parser.add_argument("--batch-name", default=f"e1_candidate200_12alg_queue_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--source-csv", default=str(SOURCE_CSV), help="任务数据集清单 CSV，需包含 dataset_dir；Core50 可传 core50_datasets.csv。")
    parser.add_argument("--expected-rows", type=int, default=200, help="任务清单期望行数；传 0 表示不校验。")
    parser.add_argument("--queue-root", default=str(QUEUE_ROOT), help="本地/远端仓库内队列状态、切片和支持脚本目录。")
    parser.add_argument("--params-root", default=str(PARAMS_ROOT), help="参数 JSON 所在目录。")
    parser.add_argument("--hosts", nargs="+", default=list(DEFAULT_HOSTS))
    parser.add_argument("--tools", nargs="+", default=list(DEFAULT_TOOLS), choices=sorted(TOOL_CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--controller-host", default="iaaccn23")
    parser.add_argument("--use-internal-ips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-limit", type=int, default=1)
    parser.add_argument("--max-jobs-per-host", type=int, default=100)
    parser.add_argument("--default-max-running-per-tool", type=int, default=0, help="0 表示不限制单工具全局 running 数")
    parser.add_argument("--max-new-jobs-per-host-per-poll", type=int, default=2)
    parser.add_argument(
        "--load-tier-new-jobs",
        default="0.50:10,0.70:5,0.80:2",
        help="按整机 load_ratio 分段设置每台每轮新增任务数，例如 '0.50:10,0.70:5,0.80:2'。",
    )
    parser.add_argument("--max-load-ratio", type=float, default=0.80)
    parser.add_argument("--max-memory-used-ratio", type=float, default=0.80)
    parser.add_argument("--min-free-mem-gb", type=float, default=0.0)
    parser.add_argument("--session-prefix", default="e1_c200_12alg_queue_")
    parser.add_argument("--round-robin-tools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-report", default=None)
    parser.add_argument("--preflight-host-timeout", type=int, default=900)
    args = parser.parse_args()
    args.load_tier_new_jobs_parsed = _parse_load_tiers(args.load_tier_new_jobs)
    args.tools = [str(tool).strip().lower() for tool in args.tools]
    args.source_csv_path = Path(args.source_csv).expanduser()
    if not args.source_csv_path.is_absolute():
        args.source_csv_path = REPO_ROOT / args.source_csv_path
    args.queue_root_path = Path(args.queue_root).expanduser()
    if not args.queue_root_path.is_absolute():
        args.queue_root_path = REPO_ROOT / args.queue_root_path
    args.params_root_path = Path(args.params_root).expanduser()
    if not args.params_root_path.is_absolute():
        args.params_root_path = REPO_ROOT / args.params_root_path
    args.expected_rows_value = None if args.expected_rows == 0 else args.expected_rows
    unknown = sorted(set(args.tools) - set(TOOL_CONFIG))
    if unknown:
        raise SystemExit(f"未知工具: {unknown}")
    return args


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.source_csv_path, expected_rows=args.expected_rows_value)
    tasks = _build_tasks(rows, tools=args.tools, seeds=args.seeds, queue_root=args.queue_root_path)
    print(
        json.dumps(
            {
                "event": "queue_start",
                "batch_name": args.batch_name,
                "source_csv": str(args.source_csv_path),
                "queue_root": str(args.queue_root_path),
                "params_root": str(args.params_root_path),
                "hosts": args.hosts,
                "tools": args.tools,
                "seeds": args.seeds,
                "tasks": len(tasks),
                "tool_config": {tool: TOOL_CONFIG[tool] for tool in args.tools},
                "load_tier_new_jobs": args.load_tier_new_jobs,
                "dry_run": args.dry_run,
                "preflight_only": args.preflight_only,
                "time": _now(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.preflight_only:
        summary = _run_preflight(args)
        print(json.dumps({"event": "preflight_done", **summary}, ensure_ascii=False), flush=True)
        return
    if not args.dry_run:
        _materialize_slices(tasks)
        _write_remote_support_script(args.queue_root_path)
    _run_scheduler(tasks, args)


if __name__ == "__main__":
    main()
