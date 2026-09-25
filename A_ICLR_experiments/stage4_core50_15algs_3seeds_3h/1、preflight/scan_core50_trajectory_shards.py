import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[3]
SCANNER = "AAAI_experiments.stage5_metric_calculation_0831.pipeline.remote_snapshot"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.candidate")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(path)


def inspect_completed(tasks_path, output_path, report_path, expected_rows):
    if not output_path.is_file() or not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("tasks") != expected_rows
            or report.get("expected_snapshots") != expected_rows * 180
            or report.get("freeze_raw") is not True):
        raise ValueError(f"completed shard contract mismatch: {report_path}")
    return {
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "tasks": expected_rows,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "available_snapshots": int(report["available_snapshots"]),
        "missing_snapshots": int(report["missing_snapshots"]),
        "conflicting_snapshots": int(report["conflicting_snapshots"]),
        "parse_errors": int(report["parse_errors"]),
        "result_missing_or_invalid": int(report["result_missing_or_invalid"]),
    }


def scan_shard(condition, host, record, output_root):
    tasks_path = Path(record["path"])
    expected_rows = int(record["rows"])
    output_path = output_root / condition / f"{host}.jsonl.gz"
    report_path = output_root / condition / f"{host}.report.json"
    if output_path.exists() or report_path.exists():
        completed = inspect_completed(tasks_path, output_path, report_path, expected_rows)
        if completed is None:
            raise FileExistsError(f"partial shard output needs a new output root: {condition}/{host}")
        return condition, host, completed

    command = [
        sys.executable,
        "-m",
        SCANNER,
        "--tasks",
        str(tasks_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--horizon",
        "180",
        "--freeze-raw",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scanner failed for {condition}/{host}: {result.stderr[-2000:]}")
    completed = inspect_completed(tasks_path, output_path, report_path, expected_rows)
    if completed is None:
        raise FileNotFoundError(f"scanner outputs missing for {condition}/{host}")
    return condition, host, completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    manifest_path = args.tasks_root / "manifest.json"
    task_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = {
        "schema": "core50.trajectory_scan_progress.v1",
        "task_manifest_path": str(manifest_path.resolve()),
        "task_manifest_sha256": sha256_file(manifest_path),
        "workers": args.workers,
        "freeze_raw": True,
        "completed": {},
    }
    lock = threading.Lock()
    work = []
    for condition, detail in task_manifest["conditions"].items():
        for host, record in detail["hosts"].items():
            output_path = args.output_root / condition / f"{host}.jsonl.gz"
            report_path = args.output_root / condition / f"{host}.report.json"
            if output_path.exists() or report_path.exists():
                completed = inspect_completed(Path(record["path"]), output_path, report_path, int(record["rows"]))
                if completed is None:
                    raise FileExistsError(f"partial shard output needs a new output root: {condition}/{host}")
                progress["completed"][f"{condition}/{host}"] = completed
            else:
                work.append((condition, host, record))
    write_json(progress_path, progress)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(scan_shard, condition, host, record, args.output_root): (condition, host)
            for condition, host, record in work
        }
        for future in as_completed(futures):
            condition, host, completed = future.result()
            with lock:
                progress["completed"][f"{condition}/{host}"] = completed
                write_json(progress_path, progress)
    print(json.dumps({"completed_shards": len(progress["completed"]), "run_count": {
        condition: sum(int(item["tasks"]) for key, item in progress["completed"].items() if key.startswith(f"{condition}/"))
        for condition in task_manifest["conditions"]
    }}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
