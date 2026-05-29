from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


DEFAULT_HOSTS = [
    "iaaccn22",
    "iaaccn23",
    "iaaccn24",
    "iaaccn25",
    "iaaccn26",
    "iaaccn27",
    "iaaccn28",
    "iaaccn29",
]
DEFAULT_REMOTE_ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
SSH_RSYNC = (
    "ssh -o BatchMode=yes -o ConnectTimeout=10 "
    "-o ServerAliveInterval=15 -o ServerAliveCountMax=2"
)
INCLUDE_RULES = [
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


Runner = Callable[..., subprocess.CompletedProcess[str]]


def collect_remote_batch(
    *,
    batch_dir: Path,
    batch_id: str,
    hosts: list[str],
    remote_root: Path = DEFAULT_REMOTE_ROOT,
    controller_host: str = "iaaccn22",
    use_internal_ips: bool = True,
    include_task_logs: bool = False,
    timeout: int = 900,
    runner: Runner = subprocess.run,
) -> dict[str, int]:
    collect_dir = batch_dir / "collect"
    collect_dir.mkdir(parents=True, exist_ok=True)
    local_root = batch_dir / "remote-experiments"

    results: list[dict[str, Any]] = []
    for host in hosts:
        command = build_rsync_command(
            batch_id=batch_id,
            host=host,
            remote_root=remote_root,
            local_root=local_root,
            controller_host=controller_host,
            use_internal_ips=use_internal_ips,
            include_task_logs=include_task_logs,
        )
        (local_root / host).mkdir(parents=True, exist_ok=True)
        proc = runner(command, text=True, capture_output=True, timeout=timeout)
        results.append(
            {
                "host": host,
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
                "local_root": str(local_root / host),
            }
        )

    succeeded = sum(1 for item in results if item["returncode"] in (0, 23, 24))
    summary = {
        "total_hosts": len(hosts),
        "succeeded": succeeded,
        "failed": len(hosts) - succeeded,
    }
    payload = {
        "batch_id": batch_id,
        "remote_root": str(remote_root),
        "local_root": str(local_root),
        "summary": summary,
        "results": results,
    }
    (collect_dir / "collect_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_results_csv(collect_dir / "collect_results.csv", results)
    return summary


def build_rsync_command(
    *,
    batch_id: str,
    host: str,
    remote_root: Path,
    local_root: Path,
    controller_host: str,
    use_internal_ips: bool,
    include_task_logs: bool,
) -> list[str]:
    source = _remote_source(
        host=host,
        remote_root=remote_root,
        batch_id=batch_id,
        controller_host=controller_host,
        use_internal_ips=use_internal_ips,
    )
    command = [
        "rsync",
        "-a",
        "--prune-empty-dirs",
        "--timeout=120",
        "-e",
        SSH_RSYNC,
    ]
    include_rules = list(INCLUDE_RULES)
    if include_task_logs:
        include_rules.append("__launcher__/logs/*.log")
    for rule in include_rules:
        command.extend(["--include", rule])
    command.extend(["--exclude", "*", source, str(local_root / host) + "/"])
    return command


def _remote_source(
    *,
    host: str,
    remote_root: Path,
    batch_id: str,
    controller_host: str,
    use_internal_ips: bool,
) -> str:
    remote_path = f"{remote_root}/experiments/{batch_id}/"
    if host == controller_host:
        return remote_path
    endpoint = _host_endpoint(host, use_internal_ips=use_internal_ips)
    return f"{endpoint}:{remote_path}"


def _host_endpoint(host: str, *, use_internal_ips: bool) -> str:
    if not use_internal_ips:
        return host
    if host.startswith("iaaccn"):
        suffix = host.removeprefix("iaaccn")
        if suffix.isdigit() and 22 <= int(suffix) <= 29:
            return f"10.10.100.{int(suffix)}"
    return host


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = ["host", "returncode", "local_root", "stdout", "stderr"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})
