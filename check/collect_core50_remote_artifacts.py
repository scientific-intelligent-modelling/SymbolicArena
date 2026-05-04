#!/usr/bin/env python3
"""从远端机器拉取 Core50 clean/noise 实验的汇总所需产物。

只拉取后处理和画图需要的文件：
- result.json / *.report.json
- progress.json / progress/*.json / minute_*.json
- best_history/*.json / samples/top*.json
- __launcher__ 状态与日志

不会删除远端文件；本地目标目录可重复覆盖同步。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SSH_RSYNC = (
    "ssh -o BatchMode=yes -o ConnectTimeout=10 "
    "-o ServerAliveInterval=15 -o ServerAliveCountMax=2"
)
RSYNC_NETWORK_ARGS = ["--timeout=120", "-e", SSH_RSYNC]

HOSTS = [
    "iaaccn23",
    "iaaccn24",
    "iaaccn25",
    "iaaccn26",
    "iaaccn27",
    "iaaccn28",
    "iaaccn29",
    "iaaccn48",
    "iaaccn49",
    "iaaccn50",
    "iaaccn51",
    "iaaccn52",
    "iaaccn53",
    "iaaccn54",
    "iaaccn55",
]

ROOT_DEFAULT = "/home/zhangziwen/workplace/scientific-intelligent-modelling"
ROOT_OVERRIDES = {
    "iaaccn48": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn49": "/data3/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn50": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn51": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn52": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn53": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn54": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
    "iaaccn55": "/data1/zhangziwen/workplace/scientific-intelligent-modelling",
}

NOISE_BATCHES = [
    "core50_noise_sigma001_12alg_3seed_20260503-083512",
    "core50_noise_sigma005_12alg_3seed_20260503-083512",
    "core50_noise_sigma010_12alg_3seed_20260503-083512",
    "core50_noise_sigma001_12alg_seed34_20260504-015409",
    "core50_noise_sigma005_12alg_seed34_20260504-015409",
    "core50_noise_sigma010_12alg_seed34_20260504-015409",
]

CLEAN_BATCHES = [
    "core50_drsr_llmbudget_seed567_20260504-165500",
]

CONTROLLER_FILES = [
    "exp-planning/05.Core50噪声鲁棒性评测/PATHS_20260503-083512.md",
    "exp-planning/05.Core50噪声鲁棒性评测/generated/load_queue/sigma001/state/",
    "exp-planning/05.Core50噪声鲁棒性评测/generated/load_queue/sigma005/state/",
    "exp-planning/05.Core50噪声鲁棒性评测/generated/load_queue/sigma010/state/",
    "exp-planning/04.Core50正式全量评测/generated/core50_drsr_llmbudget_seed567_20260504/load_queue/state/",
    "experiments/core50_noise_12alg_3seed_20260503-083512/",
    "experiments/core50_noise_12alg_seed34_20260504-015409/",
    "experiments/core50_drsr_llmbudget_seed567_20260504-165500/",
]


def remote_root(host: str) -> str:
    return ROOT_OVERRIDES.get(host, ROOT_DEFAULT)


def run(cmd: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def run_rsync(cmd: list[str], *, timeout: int, retries: int) -> tuple[subprocess.CompletedProcess[str], int]:
    """对 SSH banner 超时等瞬时失败做有限重试，避免整批漏拉。"""
    last_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, retries + 2):
        proc = run(cmd, timeout=timeout)
        last_proc = proc
        if proc.returncode != 255:
            return proc, attempt
        if attempt <= retries:
            time.sleep(min(10, 2 * attempt))
    assert last_proc is not None
    return last_proc, retries + 1


def rsync_filtered(
    remote: str,
    local: Path,
    *,
    timeout: int = 900,
    retries: int = 2,
    include_task_logs: bool = False,
) -> dict[str, Any]:
    local.mkdir(parents=True, exist_ok=True)
    include_rules = [
        "*/",
        "result.json",
        "*.report.json",
        "report.json",
        "progress.json",
        "minute_*.json",
        "progress/*.json",
        "progress/**/*.json",
        "best_history/*.json",
        "best_history/**/*.json",
        "samples/top*.json",
        "samples/best*.json",
        "__launcher__/*.json",
        "__launcher__/*.jsonl",
        "controller*.log",
        "task_status.jsonl",
        "*.state.json",
        "*.latest.json",
        "*.events.jsonl",
    ]
    if include_task_logs:
        include_rules.append("__launcher__/logs/*.log")
    cmd = ["rsync", "-a", "--prune-empty-dirs", *RSYNC_NETWORK_ARGS]
    for rule in include_rules:
        cmd.extend(["--include", rule])
    cmd.extend(["--exclude", "*", remote, str(local) + "/"])
    proc, attempts = run_rsync(cmd, timeout=timeout, retries=retries)
    return {
        "cmd": cmd,
        "attempts": attempts,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def rsync_full(remote: str, local: Path, *, timeout: int = 300, retries: int = 2) -> dict[str, Any]:
    local.mkdir(parents=True, exist_ok=True)
    proc, attempts = run_rsync(
        ["rsync", "-a", *RSYNC_NETWORK_ARGS, remote, str(local) + "/"],
        timeout=timeout,
        retries=retries,
    )
    return {
        "attempts": attempts,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def collect_batch(
    host: str,
    batch: str,
    local_root: Path,
    kind: str,
    retries: int,
    include_task_logs: bool,
) -> dict[str, Any]:
    root = remote_root(host)
    remote = f"{host}:{root}/experiments/{batch}/"
    local = local_root / kind / batch / host
    result = rsync_filtered(remote, local, retries=retries, include_task_logs=include_task_logs)
    result.update({"host": host, "batch": batch, "kind": kind, "local": str(local), "remote": remote})
    return result


def collect_controller(local_root: Path, retries: int, include_task_logs: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    host = "iaaccn23"
    root = remote_root(host)
    for rel in CONTROLLER_FILES:
        remote = f"{host}:{root}/{rel}"
        local = local_root / "controller" / rel
        local_target = local.parent if rel.endswith("/") else local.parent
        if rel.endswith("/"):
            result = rsync_filtered(remote, local, retries=retries, include_task_logs=include_task_logs)
        else:
            result = rsync_full(remote, local_target, retries=retries)
        result.update({"host": host, "remote": remote, "local": str(local), "kind": "controller"})
        out.append(result)
    return out


def summarize_files(root: Path) -> dict[str, Any]:
    patterns = {
        "result_json": "**/result.json",
        "report_json": "**/*.report.json",
        "progress_json": "**/progress.json",
        "progress_dir_json": "**/progress/*.json",
        "minute_json": "**/minute_*.json",
        "best_history_json": "**/best_history/**/*.json",
        "sample_top_json": "**/samples/top*.json",
        "state_json": "**/*.state.json",
        "latest_json": "**/*.latest.json",
        "events_jsonl": "**/*.events.jsonl",
        "task_status_jsonl": "**/task_status.jsonl",
    }
    return {name: sum(1 for _ in root.glob(pattern)) for name, pattern in patterns.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Core50 remote result/progress artifacts")
    default_outdir = (
        REPO_ROOT
        / "exp-planning/05.Core50噪声鲁棒性评测/results"
        / f"remote_artifacts_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-controller", action="store_true")
    parser.add_argument("--hosts", nargs="*", default=HOSTS)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--include-task-logs",
        action="store_true",
        help="Also collect __launcher__/logs/*.log. Disabled by default because PySR logs can be huge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str]] = []
    if not args.skip_noise:
        for batch in NOISE_BATCHES:
            for host in args.hosts:
                jobs.append((host, batch, "noise"))
    if not args.skip_clean:
        for batch in CLEAN_BATCHES:
            for host in args.hosts:
                jobs.append((host, batch, "clean"))

    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                collect_batch,
                host,
                batch,
                outdir,
                kind,
                args.retries,
                args.include_task_logs,
            ): (host, batch, kind)
            for host, batch, kind in jobs
        }
        for fut in cf.as_completed(future_map):
            host, batch, kind = future_map[fut]
            try:
                item = fut.result()
            except Exception as exc:
                item = {"host": host, "batch": batch, "kind": kind, "returncode": -1, "error": repr(exc)}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False))

    controller_results: list[dict[str, Any]] = []
    if not args.skip_controller:
        controller_results = collect_controller(outdir, args.retries, args.include_task_logs)
        for item in controller_results:
            print(json.dumps(item, ensure_ascii=False))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outdir": str(outdir),
        "hosts": args.hosts,
        "noise_batches": [] if args.skip_noise else NOISE_BATCHES,
        "clean_batches": [] if args.skip_clean else CLEAN_BATCHES,
        "results": results,
        "controller_results": controller_results,
        "file_counts": summarize_files(outdir),
    }
    (outdir / "collection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Core50 remote artifact collection",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Output directory: `{outdir}`",
        "",
        "## File counts",
        "",
    ]
    for key, value in summary["file_counts"].items():
        lines.append(f"- `{key}`: {value}")
    failed = [r for r in results + controller_results if r.get("returncode") not in (0, 23, 24)]
    lines.extend(["", "## Failed transfers", ""])
    if failed:
        for item in failed:
            lines.append(f"- `{item.get('kind')}` `{item.get('host')}` `{item.get('batch') or item.get('remote')}` rc={item.get('returncode')} err={item.get('error') or item.get('stderr','')[:200]}")
    else:
        lines.append("- none")
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "file_counts": summary["file_counts"], "failed": len(failed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
