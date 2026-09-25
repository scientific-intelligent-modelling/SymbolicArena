import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import full664_10m as original
import collect_full664_bulk as bulk


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = original.CONFIG / "recovery49"
WORK = ROOT / ".agent/work/FULL664-10M/recovery49"
QUEUE = WORK / "queue"
BATCH = "full664_missing49_s1314_600s_20260926"
HOSTS = [f"iaaccn{i}" for i in [*range(22, 30), 48, *range(50, 54), 55]]


def queue_args():
    return [str(HERE / "run_core50_queue.py"), "--batch-name", BATCH,
        "--source-csv", str(original.CONFIG / "datasets.csv"), "--expected-rows", "664",
        "--queue-root", str(QUEUE), "--params-root", str(original.CONFIG / "params"),
        "--task-id-allowlist-csv", str(CONFIG / "tasks.csv"),
        "--hosts", *HOSTS, "--tools", *original.TOOLS, "--seeds", "1314", "--noise-sigmas", "0",
        "--remote-root", original.REMOTE, "--remote-data-root", "/home/zhangziwen/sim-datasets-data",
        "--host-remote-root-overrides", ",".join(f"{h}={original.roots(h)[0]}" for h in HOSTS if int(h[6:]) >= 48),
        "--host-remote-data-root-overrides", ",".join(f"{h}={original.roots(h)[1]}" for h in HOSTS if int(h[6:]) >= 48),
        "--max-jobs-per-host", "24", "--max-new-jobs-per-host-per-poll", "18",
        "--max-cpu-used-ratio", "1.0", "--max-load-ratio", "1.0", "--load-tier-new-jobs", "1.0:18",
        "--max-memory-used-ratio", "0.92", "--min-free-mem-gb", "8", "--poll-seconds", "15",
        "--retry-limit", "1", "--host-unavailable-grace-seconds", "900",
        "--llm-model-assignment", "from-params", "--llm-default-bucket", "turbo",
        "--llm-model-bucket-limits", "base:0,turbo:0", "--no-prioritize-llm",
        "--controller-host", "iaaccn22", "--use-internal-ips", "--skip-support-sync",
        "--session-prefix", "full664_recovery49_", "--host-session-count-prefix", "full664_recovery49_"]


def prepare():
    CONFIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (original.CONFIG / "collection.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row for row in rows if row["collection_state"] != "collected"]
    assert len(missing) == 236 and all(row["assigned_host"] == "iaaccn49" for row in missing)
    assert all(not list(Path(row["destination"]).glob("*")) for row in missing)
    selected = [{"task_id": f"{row['algorithm']}_s1314_clean_{row['dataset_id']}"} for row in missing]
    with (CONFIG / "tasks.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    bulk.dump(CONFIG / "preserved_results.json", {row["destination"]: row["result_sha256"] for row in rows if row["collection_state"] == "collected"})
    bulk.dump(CONFIG / "request.json", {"original_batch": original.BATCH, "rerun_batch": BATCH,
        "tasks": 236, "seed": 1314, "budget_seconds": 600, "condition": "clean", "hosts": HOSTS,
        "algorithms": dict(Counter(row["algorithm"] for row in missing)), "existing_results_preserved": 9724,
        "params_sha256": {path.name: bulk.sha(path) for path in (original.CONFIG / "params").glob("*.json")}})
    shutil.copy2(original.CONFIG / "collection.csv", CONFIG / "input_collection.csv")
    module = original.queue_module()
    sys.argv = queue_args()
    args = module._parse_args()
    tasks = module._build_tasks(module._read_rows(args.source_csv_path, expected_rows=664),
        tools=args.tools, seeds=args.seeds, noise_sigmas=args.noise_sigmas, queue_root=args.queue_root_path,
        params_root=args.params_root_path, llm_model_assignment=args.llm_model_assignment,
        llm_model_buckets=args.llm_model_buckets_parsed, llm_default_bucket=args.llm_default_bucket)
    tasks = module._filter_tasks_by_allowlist(tasks, CONFIG / "tasks.csv")
    assert len(tasks) == 236 and {task.task_id for task in tasks} == {row["task_id"] for row in selected}
    module._materialize_slices(tasks)
    module._write_remote_support_script(QUEUE)
    with tarfile.open(WORK / "support.tar.gz", "w:gz") as archive:
        for path in (QUEUE / "slices", QUEUE / "remote"):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    print(json.dumps({"tasks": len(tasks), "algorithms": dict(Counter(task.tool for task in tasks))}), flush=True)


def install(host):
    repo = Path(original.roots(host)[0])
    archive = Path(original.REMOTE) / ".agent/work/FULL664-10M/recovery49/support.tar.gz"
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(repo)], check=True)
    (repo / ".agent/work/FULL664-10M/recovery49/queue/logs").mkdir(parents=True, exist_ok=True)


def deploy():
    archive = Path(original.REMOTE) / ".agent/work/FULL664-10M/recovery49/support.tar.gz"
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR"]
    def one(host):
        if host == "iaaccn22":
            install(host)
            return {"host": host, "synced": True}
        target = "zhangziwen@10.10.100." + host[6:]
        command = ["python3", str(Path(__file__).resolve()), "install", "--host", host]
        for args in (["ssh", *options, target, "mkdir", "-p", str(archive.parent), str(HERE)],
                     ["scp", *options, str(archive), target + ":" + str(archive)],
                     ["scp", *options, str(Path(__file__).resolve()), str(HERE / "full664_10m.py"), str(HERE / "collect_full664_bulk.py"), target + ":" + str(HERE) + "/"],
                     ["ssh", *options, target, shlex.join(command)]):
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=40)
            except subprocess.TimeoutExpired as error:
                return {"host": host, "synced": False, "error": str(error)}
            if result.returncode:
                return {"host": host, "synced": False, "error": result.stderr[-800:]}
        return {"host": host, "synced": True}
    reports = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, host): host for host in HOSTS}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            bulk.dump(archive.parent / "deployment.json", reports)
            print(json.dumps(report), flush=True)


def run():
    global HOSTS
    deployment = json.loads((CONFIG / "deployment.json").read_text())
    HOSTS = [row["host"] for row in deployment if row["synced"] and row["host"] in HOSTS]
    assert HOSTS
    request = json.loads((CONFIG / "request.json").read_text())
    for filename, digest in request["params_sha256"].items():
        assert bulk.sha(original.CONFIG / "params" / filename) == digest
    bulk.dump(CONFIG / "active_hosts.json", HOSTS)
    module = original.queue_module()
    sys.argv = queue_args()
    module.main()
    state_path = QUEUE / "state" / f"{BATCH}.state.json"
    state = json.loads(state_path.read_text())
    assert len(state["tasks"]) == 236 and all(task["state"] in {"done", "failed"} for task in state["tasks"].values())
    bulk.dump(original.CONFIG / "delivery_overrides.json", {"original_batch": original.BATCH, "rerun_batch": BATCH,
        "queue_state_sha256": bulk.sha(state_path), "tasks": state["tasks"]})
    bulk.WORK = WORK / "bulk"
    bulk.EVIDENCE = CONFIG
    counts = bulk.collect()
    bulk.seal()
    shutil.copy2(state_path, CONFIG / "terminal_queue.json")
    preserved = json.loads((CONFIG / "preserved_results.json").read_text())
    assert len(preserved) == 9724
    for directory, digest in preserved.items():
        assert bulk.sha(Path(directory) / "result.json") == digest, directory
    bulk.dump(CONFIG / "completion.json", {"time": time.time(), "rerun_states": dict(Counter(task["state"] for task in state["tasks"].values())),
        "collection_counts": counts, "preserved_result_hashes_verified": len(preserved)})
    print(json.dumps({"complete": counts.get("collected") == 9960, "counts": counts}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "deploy", "install", "run"))
    parser.add_argument("--host")
    args = parser.parse_args()
    if args.mode == "install":
        install(args.host)
    else:
        {"prepare": prepare, "deploy": deploy, "run": run}[args.mode]()
