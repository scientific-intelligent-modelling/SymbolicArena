#!/usr/bin/env python3
"""PySR 双探针实验 launcher。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


PYSR_PARAMS = {
    "timeout_in_seconds": 3600,
    "niterations": 10_000_000,
    "population_size": 64,
    "populations": 8,
    "ncycles_per_iteration": 500,
    "maxsize": 30,
    "maxdepth": 10,
    "parsimony": 1e-3,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "cube", "exp", "log", "sin", "cos"],
    "constraints": {
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 7,
        "log": 7,
        "sin": 9,
        "cos": 9,
    },
    "nested_constraints": {
        "exp": {"exp": 0, "log": 1},
        "log": {"exp": 0, "log": 0},
        "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0},
    },
    "complexity_of_operators": {
        "/": 2,
        "square": 2,
        "cube": 3,
        "sin": 2,
        "cos": 2,
        "exp": 3,
        "log": 3,
    },
    "complexity_of_constants": 2,
    "complexity_of_variables": 1,
    "max_evals": None,
    "early_stop_condition": None,
    "precision": 32,
    "deterministic": True,
    "parallelism": "serial",
    "model_selection": "best",
    "progress": True,
    "verbosity": 1,
    "procs": 1,
}

DONE_STATUSES = {"success", "timed_out"}


def _load_tasks(slice_csv: Path) -> list[dict[str, str]]:
    with open(slice_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    return f"{row['dataset_dir']}|seed={seed}"


def _resolve_dataset_dir(dataset_dir: str) -> str:
    path = Path(dataset_dir)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if path.parts and path.parts[0] == "sim-datasets-data":
            # 远端真实数据根优先放在 ~/sim-datasets-data，仓库内同名目录可能只是残缺镜像。
            candidates.append(Path.home() / path)
            candidates.append(Path.cwd() / path)
        else:
            candidates.append(Path.cwd() / path)
            candidates.append(Path.home() / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return dataset_dir


def _should_skip(row: dict[str, str], seed: int, completed: dict[str, dict[str, Any]], retry_failed: bool) -> bool:
    record = completed.get(_task_key(row, seed))
    if not record:
        return False
    status = str(record.get("status") or "")
    if status in DONE_STATUSES:
        return True
    if not retry_failed:
        return True
    return False


def _run_single_task(row: dict[str, str], *, output_root: Path, seed: int, logs_dir: Path) -> dict[str, Any]:
    report_path = logs_dir / f"{row['global_index']}_{row['dataset_name']}.report.json"
    log_path = logs_dir / f"{row['global_index']}_{row['dataset_name']}.log"
    cmd = [
        sys.executable,
        __file__,
        "run-task",
        "--dataset-dir",
        row["dataset_dir"],
        "--output-root",
        str(output_root),
        "--seed",
        str(seed),
        "--report-json",
        str(report_path),
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    if not report_path.exists():
        return {
            "task_key": _task_key(row, seed),
            "dataset_name": row["dataset_name"],
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
            "dataset_name": row["dataset_name"],
            "dataset_dir": row["dataset_dir"],
            "global_index": int(row["global_index"]),
            "log_path": str(log_path),
        }
    )
    return record


def _controller(args: argparse.Namespace) -> None:
    slice_csv = Path(args.slice_csv).resolve()
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
        if not _should_skip(row, args.seed, completed, args.retry_failed)
    ]

    print(f"切片文件: {slice_csv}")
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
            executor.submit(_run_single_task, row, output_root=output_root, seed=args.seed, logs_dir=logs_dir): row
            for row in pending
        }
        while future_map:
            done, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done:
                row = future_map.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "task_key": _task_key(row, args.seed),
                        "dataset_name": row["dataset_name"],
                        "dataset_dir": row["dataset_dir"],
                        "global_index": int(row["global_index"]),
                        "status": "error",
                        "error": repr(exc),
                    }
                record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _append_status(status_path, record, lock)
                finished += 1
                print(
                    f"[{finished}/{total}] {row['dataset_name']} -> {record.get('status')} "
                    f"(seconds={record.get('seconds')})"
                )


def _run_task(args: argparse.Namespace) -> None:
    from scientific_intelligent_modelling.benchmarks import run_benchmark_task

    output_root = Path(args.output_root).resolve()
    try:
        result_path = run_benchmark_task(
            tool_name="pysr",
            dataset_dir=_resolve_dataset_dir(args.dataset_dir),
            output_root=str(output_root),
            seed=args.seed,
            params_override=dict(PYSR_PARAMS),
        )
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        report = {
            "result_path": str(result_path),
            "status": result.get("status"),
            "error": result.get("error"),
            "seconds": result.get("seconds"),
            "experiment_dir": result.get("experiment_dir"),
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
    parser = argparse.ArgumentParser(description="启动 PySR 双探针实验。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="按切片文件启动本地固定 worker 池。")
    run_parser.add_argument("--slice-csv", required=True, help="本机对应的切片 CSV")
    run_parser.add_argument("--output-root", required=True, help="本机结果输出根目录")
    run_parser.add_argument("--seed", type=int, default=1314, help="第一阶段默认 seed")
    run_parser.add_argument("--workers", type=int, default=64, help="固定 worker 数")
    run_parser.add_argument("--retry-failed", action="store_true", help="是否重跑历史失败任务")
    run_parser.set_defaults(func=_controller)

    task_parser = subparsers.add_parser("run-task", help="内部子进程入口。")
    task_parser.add_argument("--dataset-dir", required=True)
    task_parser.add_argument("--output-root", required=True)
    task_parser.add_argument("--seed", type=int, required=True)
    task_parser.add_argument("--report-json", required=True)
    task_parser.set_defaults(func=_run_task)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
