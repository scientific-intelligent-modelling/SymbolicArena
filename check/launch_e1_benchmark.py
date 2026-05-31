#!/usr/bin/env python3
"""E1 通用 benchmark launcher。

支持任意工具按切片 CSV 跑本地固定 worker 池，并将状态落到:

- `<output_root>/__launcher__/task_status.jsonl`
- `<output_root>/__launcher__/logs/*.log`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


DONE_STATUSES = {"ok", "timed_out", "no_valid_output"}


def _load_tasks(slice_csv: Path) -> list[dict[str, str]]:
    with open(slice_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"任务切片为空: {slice_csv}")
    return rows


def _load_completed(status_path: Path) -> dict[str, dict[str, Any]]:
    if not status_path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with open(status_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            task_key = item.get("task_key")
            if isinstance(task_key, str):
                completed[task_key] = item
    return completed


def _append_status(status_path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with open(status_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _task_key(row: dict[str, str], seed: int) -> str:
    dataset_dir = row.get("dataset_dir") or row.get("dataset") or ""
    return f"{dataset_dir}|seed={seed}"


def _task_label(row: dict[str, str]) -> str:
    dataset_name = row.get("dataset_name") or row.get("dataset") or row.get("basename") or "dataset"
    try:
        return f"g{int(row['global_index']):04d}_{dataset_name}"
    except Exception:
        return f"g{row.get('global_index', 'x')}_{dataset_name}"


def _resolve_dataset_dir(dataset_dir: str) -> str:
    path = Path(dataset_dir)
    candidates: list[Path] = []
    env_data_root = Path(os.environ["SIM_DATA_ROOT"]).expanduser() if os.environ.get("SIM_DATA_ROOT") else None
    if path.is_absolute():
        candidates.append(path)
    else:
        if path.parts and path.parts[0] == "sim-datasets-data":
            if env_data_root is not None:
                candidates.append(env_data_root.joinpath(*path.parts[1:]))
            candidates.append(Path.home() / path)
            candidates.append(Path.cwd() / path)
        else:
            candidates.append(Path.cwd() / path)
            if env_data_root is not None:
                candidates.append(env_data_root / path)
            candidates.append(Path.home() / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return dataset_dir


def _should_skip(
    row: dict[str, str],
    seed: int,
    completed: dict[str, dict[str, Any]],
    retry_failed: bool,
    force_rerun: bool,
) -> bool:
    if force_rerun:
        return False
    record = completed.get(_task_key(row, seed))
    if not record:
        return False
    status = str(record.get("status") or "")
    if status in DONE_STATUSES:
        return True
    if not retry_failed:
        return True
    return False


def _run_single_task(
    row: dict[str, str],
    *,
    tool_name: str,
    params_json: Path,
    output_root: Path,
    seed: int,
    logs_dir: Path,
) -> dict[str, Any]:
    dataset_name = row.get("dataset_name") or row.get("dataset") or row.get("basename") or f"task_{row.get('global_index', 'x')}"
    report_path = logs_dir / f"{row['global_index']}_{dataset_name}.report.json"
    log_path = logs_dir / f"{row['global_index']}_{dataset_name}.log"
    cmd = [
        sys.executable,
        __file__,
        "run-task",
        "--tool",
        tool_name,
        "--dataset-dir",
        row["dataset_dir"],
        "--output-root",
        str(output_root),
        "--seed",
        str(seed),
        "--params-json",
        str(params_json),
        "--task-label",
        _task_label(row),
        "--task-global-index",
        str(row["global_index"]),
        "--expected-dataset-rel",
        row.get("dataset_rel") or "",
        "--expected-dataset-dir",
        row.get("dataset_dir") or "",
        "--report-json",
        str(report_path),
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    if not report_path.exists():
        return {
            "task_key": _task_key(row, seed),
            "dataset_name": dataset_name,
            "dataset_dir": row["dataset_dir"],
            "global_index": int(row["global_index"]),
            "status": "error",
            "error": f"缺少报告文件: {report_path}",
            "log_path": str(log_path),
        }

    with open(report_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    record.update(
        {
            "task_key": _task_key(row, seed),
            "dataset_name": dataset_name,
            "dataset_dir": row["dataset_dir"],
            "global_index": int(row["global_index"]),
            "log_path": str(log_path),
        }
    )
    return record


def _controller(args: argparse.Namespace) -> None:
    slice_csv = Path(args.slice_csv).resolve()
    params_json = Path(args.params_json).resolve()
    output_root = Path(args.output_root).resolve()
    launcher_dir = output_root / "__launcher__"
    logs_dir = launcher_dir / "logs"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = launcher_dir / "task_status.jsonl"

    tasks = _load_tasks(slice_csv)
    completed = _load_completed(status_path)
    pending = [
        row
        for row in tasks
        if not _should_skip(row, args.seed, completed, args.retry_failed, args.force_rerun)
    ]

    print(f"tool: {args.tool}")
    print(f"切片文件: {slice_csv}")
    print(f"参数文件: {params_json}")
    print(f"总任务数: {len(tasks)}")
    print(f"待运行任务数: {len(pending)}")
    print(f"并发数: {args.workers}")

    lock = threading.Lock()
    finished = 0
    total = len(pending)
    if total == 0:
        print("没有待运行任务，直接退出。")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                _run_single_task,
                row,
                tool_name=args.tool,
                params_json=params_json,
                output_root=output_root,
                seed=args.seed,
                logs_dir=logs_dir,
            ): row
            for row in pending
        }
        while future_map:
            done, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done:
                row = future_map.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    dataset_name = row.get("dataset_name") or row.get("dataset") or row.get("basename") or f"task_{row.get('global_index', 'x')}"
                    record = {
                        "task_key": _task_key(row, args.seed),
                        "dataset_name": dataset_name,
                        "dataset_dir": row["dataset_dir"],
                        "global_index": int(row["global_index"]),
                        "status": "error",
                        "error": repr(exc),
                    }
                record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _append_status(status_path, record, lock)
                finished += 1
                print(
                    f"[{finished}/{total}] {record.get('dataset_name')} -> {record.get('status')} "
                    f"(seconds={record.get('seconds')})"
                )


def _run_task(args: argparse.Namespace) -> None:
    # 直接从 runner 模块导入，避免远端 benchmarks/__init__.py 未及时同步时
    # 无法从包入口 re-export `run_benchmark_task`。
    from scientific_intelligent_modelling.benchmarks.runner import run_benchmark_task

    output_root = Path(args.output_root).resolve()
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    params.update(
        {
            "task_label": args.task_label,
            "task_global_index": args.task_global_index,
            "expected_dataset_rel": args.expected_dataset_rel,
            "expected_dataset_dir": args.expected_dataset_dir,
        }
    )
    try:
        result_path = run_benchmark_task(
            tool_name=args.tool,
            dataset_dir=_resolve_dataset_dir(args.dataset_dir),
            output_root=str(output_root),
            seed=args.seed,
            params_override=params,
        )
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        report = {
            "result_path": str(result_path),
            "status": result.get("status"),
            "error": result.get("error"),
            "budget_exhausted": result.get("budget_exhausted"),
            "timeout_type": result.get("timeout_type"),
            "termination_reason": result.get("termination_reason"),
            "recovered_from_timeout": result.get("recovered_from_timeout"),
            "seconds": result.get("seconds"),
            "experiment_dir": result.get("experiment_dir"),
            "task_label": result.get("task_label"),
            "task_global_index": result.get("task_global_index"),
            "expected_dataset_rel": result.get("expected_dataset_rel"),
            "expected_dataset_dir": result.get("expected_dataset_dir"),
            "dataset_identity_check": result.get("dataset_identity_check"),
        }
    except Exception as exc:
        report = {
            "result_path": None,
            "status": "error",
            "error": repr(exc),
            "seconds": None,
            "experiment_dir": None,
        }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 通用 benchmark launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="按切片文件启动本地固定 worker 池")
    run_parser.add_argument("--tool", required=True, help="工具名")
    run_parser.add_argument("--slice-csv", required=True, help="本机切片 CSV")
    run_parser.add_argument("--params-json", required=True, help="工具参数 JSON 文件")
    run_parser.add_argument("--output-root", required=True, help="本机结果输出根目录")
    run_parser.add_argument("--seed", type=int, default=1314, help="随机种子")
    run_parser.add_argument("--workers", type=int, default=8, help="并发 worker 数")
    run_parser.add_argument("--retry-failed", action="store_true", help="是否重试 error 任务")
    run_parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="忽略已有 task_status.jsonl 记录，强制重算切片内任务",
    )

    task_parser = subparsers.add_parser("run-task", help="执行单个 benchmark 任务")
    task_parser.add_argument("--tool", required=True, help="工具名")
    task_parser.add_argument("--dataset-dir", required=True, help="数据集目录")
    task_parser.add_argument("--params-json", required=True, help="工具参数 JSON 文件")
    task_parser.add_argument("--output-root", required=True, help="本机结果输出根目录")
    task_parser.add_argument("--seed", type=int, default=1314, help="随机种子")
    task_parser.add_argument("--task-label", default=None, help="唯一任务标签，用于结果目录隔离")
    task_parser.add_argument("--task-global-index", default=None, help="Candidate-200 全局编号")
    task_parser.add_argument("--expected-dataset-rel", default=None, help="期望数据集相对身份路径")
    task_parser.add_argument("--expected-dataset-dir", default=None, help="期望数据集目录")
    task_parser.add_argument("--report-json", required=True, help="单任务报告 JSON 输出路径")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        _controller(args)
        return
    if args.command == "run-task":
        _run_task(args)
        return
    parser.error(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
