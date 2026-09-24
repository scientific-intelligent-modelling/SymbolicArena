import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = HERE.parent
GOAL_WORK = ROOT / ".agent/work/GOAL-CORE50"
QUEUES = {"core50_new15_nonllm_v2": GOAL_WORK / "queue_nonllm_v2",
          "core50_new15_llm_v1": GOAL_WORK / "queue_llm_v1"}
PENDING = HERE / "pending_training.csv"
MANIFEST = HERE / "core50_training_collection.csv"
SUMMARY = HERE / "core50_training_collection_summary.json"
ALGORITHM_CONFIG = ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/experiment_config.json"
CONTROLLER_HOST = "iaaccn22"
HOST_ROOTS = {"iaaccn48": "/data1/zhangziwen/sim-runtime/code",
              "iaaccn49": "/data3/zhangziwen/sim-runtime/code",
              "iaaccn50": "/data1/zhangziwen/sim-runtime/code",
              "iaaccn51": "/data1/zhangziwen/sim-runtime/code",
              "iaaccn52": "/data1/zhangziwen/sim-runtime/code",
              "iaaccn53": "/data1/zhangziwen/sim-runtime/code",
              "iaaccn55": "/data1/zhangziwen/sim-runtime/code"}
REMOTE_ROOT = "/home/zhangziwen/workplace/scientific-intelligent-modelling"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def host_target(host):
    if host == CONTROLLER_HOST:
        return host, []
    number = host.removeprefix("iaaccn")
    return f"zhangziwen@10.10.100.{number}", ["-o", f"ProxyJump={CONTROLLER_HOST}"]


def run_capture(command, timeout):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def read_manifest():
    with PENDING.open(encoding="utf-8", newline="") as source:
        pending = list(csv.DictReader(source))
    with ALGORITHM_CONFIG.open(encoding="utf-8") as source:
        config = json.load(source)
    if len(pending) != 2025:
        raise ValueError(f"pending_training.csv项数错误: {len(pending)}")
    if len({(r["noise"], r["algorithm"], r["dataset_id"], int(r["seed"])) for r in pending}) != 2025:
        raise ValueError("待训练清单含重复运行")
    return pending, config


def controller_state():
    return json.loads((QUEUE / "state" / f"{BATCH}.state.json").read_text())


def initial_records(pending):
    rows = {(r["noise"], r["algorithm"].lower(), r["dataset_id"], int(r["seed"])): dict(r)
            for r in pending}
    for row in rows.values():
        row.update(collection_state="awaiting_dispatch", controller_task_id="", assigned_host="",
                   controller_attempts=0, controller_state="", source_run_dir="", result_sha256="",
                   result_status="", equation_available=False, numeric_available=False,
                   progress_file_count=0, imported_file_count=0, imported_bytes=0, collection_error="")
    return rows


def sync_one(task, record, remote_root, batch):
    host = str(task["assigned_host"])
    task_id = task["task_id"]
    record.update(controller_task_id=task_id, assigned_host=host,
                  controller_attempts=int(task.get("attempts") or 0),
                  controller_state=task["state"])
    dataset = str(record["dataset_name"])
    index = int(record["dataset_id"][1:])
    source_dir = (Path(remote_root) / "experiments" / batch / task["tool"]
                  / f"seed{task['seed']}" / "tasks" / task_id / host / task["tool"]
                  / f"g{index:04d}_{dataset}")
    destination = (STAGE / "2、experiments" / task["noise_tag"] / task["tool"]
                  / dataset / str(task["seed"]))
    record["source_run_dir"] = str(source_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target, jump = host_target(host)
    ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR",
                   *jump, target, "sha256sum", "--", str(source_dir / "result.json")]
    remote_result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=30)
    remote_sha = remote_result.stdout.split(maxsplit=1)[0] if remote_result.returncode == 0 else ""
    existing_result = destination / "result.json"
    existing_binding = destination / "import_binding.json"
    if existing_result.exists():
        existing_sha = digest(existing_result)
        if not remote_sha or existing_sha != remote_sha:
            record.update(collection_state="result_conflict",
                          collection_error="destination result differs from the terminal remote result; preserved without overwrite")
            return record
        if existing_binding.exists():
            old_binding = json.loads(existing_binding.read_text())
            if old_binding.get("result_sha256") == remote_sha:
                record.update(collection_state="collected", result_sha256=remote_sha,
                              result_status=old_binding.get("result_status", ""),
                              equation_available=bool(old_binding.get("equation_available")),
                              numeric_available=bool(old_binding.get("numeric_available")))
                return record
    transport = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR"
    if jump:
        transport += " " + " ".join(shlex.quote(part) for part in jump)
    source = f"{target}:{shlex.quote(str(source_dir))}/"
    subprocess.run(["rsync", "-a", "--partial", "--protect-args", "-e", transport,
                    source, str(destination) + "/"], check=True, timeout=900)
    result_path = destination / "result.json"
    if not result_path.is_file():
        record.update(collection_state="terminal_no_result", collection_error="result.json missing after rsync")
    else:
        result_sha = digest(result_path)
        if remote_sha and result_sha != remote_sha:
            record.update(collection_state="result_conflict",
                          collection_error="copied result SHA256 differs from terminal remote result")
            return record
        result = json.loads(result_path.read_text())
        assert int(result["task_global_index"]) == index
        assert result["expected_dataset_rel"] == record["dataset_rel"]
        assert str(result["tool"]).lower() == task["tool"].lower()
        assert int(result["seed"]) == int(task["seed"])
        assert result["dataset_identity_check"]["match"] is True
        equation = result.get("equation") or (result.get("canonical_artifact") or {}).get("instantiated_expression")
        numeric = all(isinstance(result.get(split), dict)
                      and isinstance(result[split].get("nmse"), (float, int))
                      and not isinstance(result[split]["nmse"], bool)
                      and math.isfinite(result[split]["nmse"])
                      for split in ["id_test", "ood_test"])
        record.update(collection_state="collected", result_sha256=result_sha,
                      result_status=result.get("status", ""), equation_available=bool(equation),
                      numeric_available=numeric)
    files = [Path(directory) / name for directory, _, names in os.walk(destination) for name in names]
    binding = {"schema": "core50.run_collection.v1", "batch": batch,
               "algorithm": task["tool"], "dataset_id": record["dataset_id"],
               "dataset_rel": record["dataset_rel"], "noise": record["noise"],
               "seed": int(task["seed"]), "source_host": host, "source_run_dir": str(source_dir),
               "destination_run_dir": str(destination), "controller_task_id": task_id,
               "controller_attempts": int(task.get("attempts") or 0),
               "controller_state": task["state"], "result_sha256": record["result_sha256"],
               "collection_state": record["collection_state"],
               "progress_files": sum(path.parent.name == "progress" and path.name.startswith("minute_") for path in files),
               "file_count": len(files), "total_bytes": sum(path.stat().st_size for path in files),
               "error": record["collection_error"]}
    binding_path = destination / "import_binding.json"
    if binding_path.exists():
        previous = json.loads(binding_path.read_text())
        if previous.get("result_sha256") not in {"", record["result_sha256"]}:
            record.update(collection_state="result_conflict",
                          collection_error="destination already binds a different result SHA256")
            return record
    binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record.update(progress_file_count=binding["progress_files"], imported_file_count=binding["file_count"],
                  imported_bytes=binding["total_bytes"])
    return record


def persist(records):
    fields = list(next(iter(records.values())).keys())
    path = MANIFEST.with_suffix(".pending")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records.values())
    path.replace(MANIFEST)
    counts = Counter(record["collection_state"] for record in records.values())
    summary = {"schema": "core50_training_collection.v1", "batches": list(QUEUES),
               "total_training_runs": len(records), "collection_counts": dict(counts),
               "imported_result_count": sum(r["collection_state"] == "collected" for r in records.values()),
               "formula_available_count": sum(r["equation_available"] for r in records.values()),
               "numeric_available_count": sum(r["numeric_available"] for r in records.values()),
               "imported_progress_files": sum(r["progress_file_count"] for r in records.values()),
               "run_collection_manifest_sha256": digest(MANIFEST),
               "training_launched": any(row["controller_state"] in {"running", "done", "failed"}
                                         for row in records.values())}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def one_cycle(records, states, workers, max_per_poll):
    by_task_id = {}
    for batch, state in states.items():
        for task_id, source_task in state["tasks"].items():
            task = dict(source_task, batch_name=batch)
            index = int(task_id.rsplit("g", 1)[1])
            key = (task["noise_tag"], task["tool"].lower(), f"g{index:04d}", int(task["seed"]))
            record = records[key]
            record.update(controller_task_id=task_id, controller_attempts=int(task.get("attempts") or 0),
                          controller_state=task["state"], assigned_host=task.get("assigned_host") or "",
                          queue_batch=batch)
            if task["state"] == "pending" and record["collection_state"] == "awaiting_dispatch":
                record["collection_state"] = "queued"
            elif task["state"] in {"dispatching", "running"}:
                record["collection_state"] = "training"
            if task["state"] not in {"done", "failed"}:
                continue
            if not task.get("assigned_host"):
                record.update(collection_state="terminal_without_host",
                              collection_error="controller ended without an assigned host")
                continue
            if record["collection_state"] in {"collected", "terminal_no_result", "result_conflict", "terminal_without_host"}:
                continue
            by_task_id[(batch, task_id)] = (task, record, key)
    selected = list(by_task_id.items())[:max_per_poll]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sync_one, task, record,
                               REMOTE_ROOTS.get(str(task["assigned_host"]), REMOTE_HOME_ROOT), batch): (batch, task_id)
                   for (batch, task_id), (task, record, key) in selected
                   if task.get("assigned_host") and key}
        for future in as_completed(futures):
            batch, task_id = futures[future]
            task, record, key = by_task_id[batch, task_id]
            try:
                records[key] = future.result()
            except Exception as error:
                record.update(collection_state="sync_error", collection_error=repr(error))
    return len(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-per-poll", type=int, default=32)
    args = parser.parse_args()
    if min(args.interval_seconds, args.workers, args.max_per_poll) <= 0:
        parser.error("采集参数必须为正数")
    pending, _ = read_manifest()
    records = initial_records(pending)
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8", newline="") as source:
            saved = {(r["noise"], r["algorithm"].lower(), r["dataset_id"], int(r["seed"])): r
                     for r in csv.DictReader(source)}
        assert set(saved) == set(records)
        for row in saved.values():
            for key in ["equation_available", "numeric_available"]:
                row[key] = row[key].strip().lower() == "true"
            for key in ["controller_attempts", "progress_file_count", "imported_file_count", "imported_bytes"]:
                row[key] = int(row[key] or 0)
        records.update(saved)
    while True:
        states = {batch: json.loads((queue / "state" / f"{batch}.state.json").read_text())
                  for batch, queue in QUEUES.items()
                  if (queue / "state" / f"{batch}.state.json").is_file()}
        if states:
            one_cycle(records, states, args.workers, args.max_per_poll)
        persist(records)
        if args.once:
            return
        all_terminal = bool(states) and all(task["state"] in {"done", "failed"}
                                            for state in states.values() for task in state["tasks"].values())
        all_imported = all(row["collection_state"] in {"collected", "terminal_no_result",
                                                        "result_conflict", "terminal_without_host"}
                           for row in records.values())
        if all_terminal and all_imported:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    REMOTE_HOME_ROOT = "/home/zhangziwen/workplace/scientific-intelligent-modelling"
    REMOTE_ROOTS = {"iaaccn48": "/data1/zhangziwen/sim-runtime/code",
                    "iaaccn49": "/data3/zhangziwen/sim-runtime/code",
                    "iaaccn50": "/data1/zhangziwen/sim-runtime/code",
                    "iaaccn51": "/data1/zhangziwen/sim-runtime/code",
                    "iaaccn52": "/data1/zhangziwen/sim-runtime/code",
                    "iaaccn53": "/data1/zhangziwen/sim-runtime/code",
                    "iaaccn55": "/data1/zhangziwen/sim-runtime/code"}
    main()
