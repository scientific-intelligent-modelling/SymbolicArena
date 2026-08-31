from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REMOTE_HELPER = r"""#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def inspect_entry(entry: dict[str, object]) -> dict[str, object]:
    result_path = Path(str(entry["result_path"]))
    parent = result_path.parent
    exists = result_path.exists()
    outer_progress_dir = parent / "progress"
    outer_count = len(list(outer_progress_dir.glob("minute_*.json"))) if outer_progress_dir.is_dir() else 0
    inner_counts: list[int] = []
    experiments_dir = parent / "experiments"
    if experiments_dir.is_dir():
        for progress_dir in experiments_dir.glob("*/progress"):
            if progress_dir.is_dir():
                inner_counts.append(len(list(progress_dir.glob("minute_*.json"))))
    progress_max_count = max([outer_count] + inner_counts) if (outer_count or inner_counts) else 0
    has_progress = progress_max_count > 0
    return {
        "logical_key": entry["logical_key"],
        "task_id": entry["task_id"],
        "dataset_id": entry["dataset_id"],
        "algorithm": entry["algorithm"],
        "seed": entry["seed"],
        "noise_tag": entry["noise_tag"],
        "host": entry["host"],
        "result_path": str(result_path),
        "result_exists": exists,
        "outer_progress_count": outer_count,
        "inner_progress_counts": inner_counts,
        "progress_max_count": progress_max_count,
        "has_progress": has_progress,
    }


def main() -> int:
    import sys
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    rows = [inspect_entry(entry) for entry in payload["entries"]]
    summary = {
        "host": payload["host"],
        "entry_count": len(rows),
        "result_exists": sum(1 for row in rows if row["result_exists"]),
        "has_progress": sum(1 for row in rows if row["has_progress"]),
    }
    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_source_rows(source_runs_csv: Path, *, noise_tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["noise_tag"] != noise_tag:
                continue
            rows.append(
                {
                    "logical_key": row["logical_key"],
                    "task_id": row["task_id"],
                    "dataset_id": row["dataset_id"],
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "noise_tag": row["noise_tag"],
                    "host": row["host"],
                    "result_path": row["path"],
                }
            )
    return rows


def build_host_payloads(
    rows: list[dict[str, Any]],
    *,
    limit_per_host: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["host"], []).append(row)
    for host, host_rows in grouped.items():
        host_rows.sort(key=lambda item: (item["algorithm"], item["dataset_id"], item["seed"], item["task_id"]))
        if limit_per_host is not None:
            grouped[host] = host_rows[:limit_per_host]
        else:
            grouped[host] = host_rows
    return dict(sorted(grouped.items()))


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )


def _scp_to_host(local_path: Path, host: str, remote_path: str) -> None:
    cmd = [
        "timeout",
        "60",
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        str(local_path),
        f"{host}:{remote_path}",
    ]
    completed = _run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"SCP 到 {host} 失败: {completed.stderr.strip() or completed.stdout.strip()}")


def _ssh_run_json(host: str, remote_helper: str, remote_payload: str) -> dict[str, Any]:
    cmd = [
        "timeout",
        "120",
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        f"python3 {remote_helper} {remote_payload}",
    ]
    completed = _run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"SSH 执行 {host} 失败: {completed.stderr.strip() or completed.stdout.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{host} 返回 JSON 解析失败: {completed.stdout[:300]!r}") from exc


def execute_remote_preflight(
    source_runs_csv: Path,
    *,
    noise_tag: str,
    limit_per_host: int | None = None,
) -> dict[str, Any]:
    rows = load_source_rows(source_runs_csv, noise_tag=noise_tag)
    payloads = build_host_payloads(rows, limit_per_host=limit_per_host)
    local_tmp = Path(tempfile.mkdtemp(prefix="stage5_remote_preflight_"))
    remote_root = f"/tmp/stage5_remote_preflight_{os.getpid()}"
    helper_path = local_tmp / "remote_helper.py"
    helper_path.write_text(REMOTE_HELPER, encoding="utf-8")

    host_reports: list[dict[str, Any]] = []
    try:
        for host, host_rows in payloads.items():
            payload_path = local_tmp / f"{host}.json"
            payload_path.write_text(
                json.dumps({"host": host, "entries": host_rows}, ensure_ascii=False),
                encoding="utf-8",
            )
            mkdir_cmd = [
                "timeout",
                "30",
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                f"mkdir -p {remote_root}",
            ]
            mkdir_completed = _run(mkdir_cmd)
            if mkdir_completed.returncode != 0:
                raise RuntimeError(
                    f"远端目录创建失败 {host}: {mkdir_completed.stderr.strip() or mkdir_completed.stdout.strip()}"
                )
            remote_helper = f"{remote_root}/remote_helper.py"
            remote_payload = f"{remote_root}/{host}.json"
            _scp_to_host(helper_path, host, remote_helper)
            _scp_to_host(payload_path, host, remote_payload)
            host_reports.append(_ssh_run_json(host, remote_helper, remote_payload))
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)

    combined_rows = [row for host_report in host_reports for row in host_report["rows"]]
    missing_results = [row for row in combined_rows if not row["result_exists"]]
    missing_progress = [row for row in combined_rows if not row["has_progress"]]
    per_host = {
        host_report["summary"]["host"]: host_report["summary"] for host_report in host_reports
    }
    return {
        "noise_tag": noise_tag,
        "source_runs_csv": str(source_runs_csv),
        "requested_rows": len(rows),
        "checked_rows": len(combined_rows),
        "limit_per_host": limit_per_host,
        "per_host": per_host,
        "summary": {
            "result_exists": len(combined_rows) - len(missing_results),
            "result_missing": len(missing_results),
            "progress_exists": len(combined_rows) - len(missing_progress),
            "progress_missing": len(missing_progress),
        },
        "missing_results": missing_results[:50],
        "missing_progress": missing_progress[:50],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="只读远端预检 clean/noise 的 result/progress 覆盖")
    parser.add_argument(
        "--source-runs-csv",
        type=Path,
        default=repo_root / "AAAI_experiments/stage5_metric_calculation_0831/manifests/source_runs.csv",
    )
    parser.add_argument("--noise-tag", choices=("clean", "noise001", "noise005"), default="clean")
    parser.add_argument("--limit-per-host", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "AAAI_experiments/stage5_metric_calculation_0831/reports/clean_remote_preflight.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = execute_remote_preflight(
        args.source_runs_csv.resolve(),
        noise_tag=args.noise_tag,
        limit_per_host=args.limit_per_host,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

