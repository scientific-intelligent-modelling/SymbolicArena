#!/usr/bin/env python3
"""Probe4 full-664 顺序 seed orchestrator。

策略：
- 每个 seed 内同时启动 udsr/dso/imcts/pyoperon 四个工具。
- 等当前 seed 的所有远端 tmux 结束后，再进入下一个 seed。
- 重启同一 batch 时，底层 launcher 会跳过 ok/timed_out/no_valid_output，
  并在 retry 模式下重跑 error 或缺失任务。
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/zhangziwen/projects/scientific-intelligent-modelling")
ASSET_ROOT = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1"
MANIFEST = ASSET_ROOT / "probe4_full664_manifest.csv"
LAUNCH_SCRIPT = ASSET_ROOT / "launch/run_tool_seed.sh"

TOOLS = ["udsr", "dso", "imcts", "pyoperon"]
SEEDS = ["520", "521", "522"]
HOSTS = ["iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26", "iaaccn27", "iaaccn28", "iaaccn29"]
DONE_STATUSES = {"ok", "timed_out", "no_valid_output"}


def _run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "", exc.stderr or "timeout")


def _ssh(host: str, command: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            command,
        ],
        timeout=timeout,
    )


def _load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _manifest_index() -> dict[tuple[str, str, str], dict[str, str]]:
    return {(row["tool"], row["seed"], row["host"]): row for row in _load_manifest()}


def _session(tool: str, seed: str, host: str) -> str:
    return f"probe4_full664_{tool}_s{seed}_{host}"


def _tmux_running(host: str, session: str) -> bool:
    result = _ssh(host, f"tmux has-session -t {session!r}", timeout=15)
    return result.returncode == 0


def _start_manifest_job(row: dict[str, str], batch_name: str, *, retry: bool) -> dict[str, Any]:
    """启动单个 host/tool/seed job。

    不通过 `run_tool_seed.sh` 整波启动，原因是单台 SSH 抖动不应该阻断
    其它 host/tool。这里按 job 独立启动，已在跑的 tmux 会直接跳过。
    """

    retry_arg = "retry" if retry else "noretry"
    host = row["host"]
    session = row["tmux_session"]
    remote_job = REMOTE_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1" / row["remote_job"]
    cmd = (
        f"cd {shlex.quote(str(REMOTE_ROOT))} && "
        f"chmod +x {shlex.quote(str(remote_job))} && "
        f"if tmux has-session -t {shlex.quote(session)} >/dev/null 2>&1; then "
        f"echo ALREADY_RUNNING; "
        f"else "
        f"tmux new-session -d -s {shlex.quote(session)} "
        f"/bin/bash {shlex.quote(str(remote_job))} "
        f"{shlex.quote(batch_name)} {shlex.quote(row['workers'])} {shlex.quote(retry_arg)}; "
        f"echo STARTED; "
        f"fi"
    )
    result = _ssh(host, cmd, timeout=60)
    return {
        "tool": row["tool"],
        "seed": row["seed"],
        "host": host,
        "session": session,
        "retry": retry,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _start_seed_jobs(seed: str, batch_name: str, *, retry: bool, manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    jobs = [row for row in manifest_rows if row["seed"] == seed]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=14) as executor:
        future_map = {
            executor.submit(_start_manifest_job, row, batch_name, retry=retry): row
            for row in jobs
        }
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {
                    "tool": row["tool"],
                    "seed": row["seed"],
                    "host": row["host"],
                    "session": row["tmux_session"],
                    "retry": retry,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": repr(exc),
                }
            results.append(item)
            print(json.dumps({"event": "start_job", **item}, ensure_ascii=False), flush=True)
    failed = [item for item in results if item["returncode"] != 0]
    print(
        json.dumps(
            {
                "event": "start_seed_jobs_done",
                "seed": seed,
                "retry": retry,
                "total": len(results),
                "failed": len(failed),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return results


def _read_status(host: str, batch_name: str, tool: str, seed: str) -> list[dict[str, Any]]:
    path = REMOTE_ROOT / "experiments" / batch_name / tool / f"seed{seed}" / host / "__launcher__/task_status.jsonl"
    result = _ssh(host, f"test -f {str(path)!r} && cat {str(path)!r} || true", timeout=60)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(item)
    return rows


def _summarize(batch_name: str, seed: str, manifest: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, Any]:
    by_job: list[dict[str, Any]] = []
    total_expected = 0
    total_latest = 0
    total_done = 0
    total_error = 0
    running_sessions = 0
    status_counter: Counter[str] = Counter()

    for tool in TOOLS:
        for host in HOSTS:
            row = manifest[(tool, seed, host)]
            expected = int(row["tasks"])
            session = _session(tool, seed, host)
            running = _tmux_running(host, session)
            statuses = _read_status(host, batch_name, tool, seed)
            latest: dict[str, dict[str, Any]] = {}
            for item in statuses:
                key = item.get("task_key")
                if isinstance(key, str):
                    latest[key] = item

            counts = Counter(str(item.get("status") or "unknown") for item in latest.values())
            missing = max(0, expected - len(latest))
            done = sum(count for status, count in counts.items() if status in DONE_STATUSES)
            errors = counts.get("error", 0)

            total_expected += expected
            total_latest += len(latest)
            total_done += done
            total_error += errors
            running_sessions += int(running)
            status_counter.update(counts)
            if missing:
                status_counter["missing"] += missing

            by_job.append(
                {
                    "tool": tool,
                    "seed": seed,
                    "host": host,
                    "expected": expected,
                    "seen": len(latest),
                    "done": done,
                    "error": errors,
                    "missing": missing,
                    "running": running,
                    "status_counts": dict(sorted(counts.items())),
                }
            )

    return {
        "batch_name": batch_name,
        "seed": seed,
        "expected": total_expected,
        "seen": total_latest,
        "done": total_done,
        "error": total_error,
        "running_sessions": running_sessions,
        "status_counts": dict(sorted(status_counter.items())),
        "jobs": by_job,
        "time": datetime.now().isoformat(timespec="seconds"),
    }


def _write_summary(batch_name: str, payload: dict[str, Any]) -> None:
    out_dir = REPO_ROOT / "exp-planning/02.E1选择验证/generated/probe4_full664_v1/orchestrator_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{batch_name}.latest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 seed 顺序运行 Probe4 full-664")
    parser.add_argument("--batch-name", default=f"probe4_full664_v1_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--retry-limit", type=int, default=1)
    parser.add_argument("--seed-timeout-hours", type=float, default=10.0)
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    args = parser.parse_args()

    manifest_rows = _load_manifest()
    manifest = {(row["tool"], row["seed"], row["host"]): row for row in manifest_rows}
    print(
        json.dumps(
            {
                "event": "orchestrator_start",
                "batch_name": args.batch_name,
                "seeds": args.seeds,
                "tools": TOOLS,
                "hosts": HOSTS,
                "time": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for seed in args.seeds:
        for attempt in range(args.retry_limit + 1):
            retry = attempt > 0
            print(
                json.dumps(
                    {"event": "seed_attempt_start", "seed": seed, "attempt": attempt, "retry": retry},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _start_seed_jobs(seed, args.batch_name, retry=retry, manifest_rows=manifest_rows)

            deadline = time.time() + args.seed_timeout_hours * 3600
            while True:
                summary = _summarize(args.batch_name, seed, manifest)
                summary["event"] = "seed_poll"
                summary["attempt"] = attempt
                _write_summary(args.batch_name, summary)
                print(json.dumps({k: v for k, v in summary.items() if k != "jobs"}, ensure_ascii=False), flush=True)

                if summary["running_sessions"] == 0:
                    if summary["done"] == summary["expected"] and summary["error"] == 0:
                        print(
                            json.dumps(
                                {"event": "seed_done", "seed": seed, "attempt": attempt, "summary": {k: v for k, v in summary.items() if k != "jobs"}},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        break
                    if attempt < args.retry_limit:
                        print(
                            json.dumps(
                                {"event": "seed_needs_retry", "seed": seed, "attempt": attempt, "summary": {k: v for k, v in summary.items() if k != "jobs"}},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        break
                    raise SystemExit(f"seed{seed} 结束但仍有未完成或失败任务: {summary}")

                stale_jobs = [
                    job
                    for job in summary["jobs"]
                    if not job["running"] and job["done"] < job["expected"]
                ]
                if stale_jobs:
                    stale_keys = {(job["tool"], job["seed"], job["host"]) for job in stale_jobs}
                    stale_rows = [row for row in manifest_rows if (row["tool"], row["seed"], row["host"]) in stale_keys]
                    print(
                        json.dumps(
                            {
                                "event": "restart_stale_jobs",
                                "seed": seed,
                                "count": len(stale_rows),
                                "jobs": sorted([f"{row['tool']}:{row['host']}" for row in stale_rows]),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        for future in as_completed(
                            [executor.submit(_start_manifest_job, row, args.batch_name, retry=True) for row in stale_rows]
                        ):
                            print(json.dumps({"event": "restart_stale_job_result", **future.result()}, ensure_ascii=False), flush=True)

                if time.time() > deadline:
                    raise SystemExit(f"seed{seed} 超过 {args.seed_timeout_hours} 小时仍未结束")
                time.sleep(args.poll_seconds)

            if summary["done"] == summary["expected"] and summary["error"] == 0:
                break

    print(
        json.dumps(
            {"event": "orchestrator_done", "batch_name": args.batch_name, "time": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
