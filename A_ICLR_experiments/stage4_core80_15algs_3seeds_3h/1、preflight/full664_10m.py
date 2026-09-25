import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = ROOT / ".agent/work/FULL664-10M"
CONFIG = HERE / "full664_10m"
BATCH = "full664_15alg_clean_s1314_600s_20260926"
HOSTS = [f"iaaccn{i}" for i in [*range(22, 30), *range(48, 56)]]
TOOLS = ["gplearn", "llmsr", "pyoperon", "drsr", "pysr", "dso", "tpsr", "e2esr", "fepysr", "jaxsr", "qlattice", "imcts", "udsr", "ragsr", "symbolfit"]
REMOTE = "/home/zhangziwen/workplace/scientific-intelligent-modelling"
REMOTE_AUDIT = "/home/zhangziwen/sim-runtime/full664_10m"


def roots(host):
    number = int(host.removeprefix("iaaccn"))
    if number < 48:
        return REMOTE, "/home/zhangziwen/sim-datasets-data"
    base = "/data3/zhangziwen" if number == 49 else "/data1/zhangziwen"
    return base + "/sim-runtime/code", base + "/sim-datasets-data"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    CONFIG.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source = ROOT / "A_ICLR_experiments/stage3_664dats_4probes_3seeds_1h/probe4_postprocess_dataset_level.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 664 and len({r["dataset_rel"] for r in rows}) == 664
    names = Counter(row["dataset_name"] for row in rows)
    core_names = {}
    for filename in ("reuse_manifest.csv", "pending_training.csv"):
        with (HERE / filename).open(newline="") as handle:
            for record in csv.DictReader(handle):
                core_names[record["dataset_rel"]] = record["dataset_name"]
    datasets = []
    for row in rows:
        name = row["dataset_name"]
        destination = core_names.get(row["dataset_rel"], name if names[name] == 1 else row["dataset_id"] + "_" + name)
        datasets.append({key: row[key] for key in ("dataset_id", "global_index", "dataset_name", "dataset_rel")} | {
            "dataset_dir": row["dataset_rel"], "destination_name": destination,
            "required_files": ["metadata.yaml", "train.csv", "valid.csv", "id_test.csv"] + (["ood_test.csv"] if int(row["ood_test_samples"]) else []),
        })
    assert len({r["destination_name"] for r in datasets}) == 664
    with (CONFIG / "datasets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "global_index", "dataset_name", "dataset_rel", "dataset_dir", "destination_name"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(datasets)
    dump(CONFIG / "datasets.json", datasets)
    params_root = CONFIG / "params"
    params_root.mkdir(exist_ok=True)
    for tool in TOOLS:
        source_params = ROOT / f".agent/work/EXP-001/controller/params/{tool}__clean.json"
        params = json.loads(source_params.read_text())
        params["timeout_in_seconds"] = 600
        params["progress_snapshot_interval_seconds"] = 60
        assert params["train_label_noise_enabled"] is False
        dump(params_root / f"{tool}__clean.json", params)
    dump(CONFIG / "experiment.json", {"batch": BATCH, "tasks": 9960, "datasets": 664, "algorithms": TOOLS,
        "seed": 1314, "condition": "clean", "budget_seconds": 600, "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "destination_root": str(HERE.parent / "2.1 full set experimets"),
        "smoke_tests": False, "opus": False, "retry_limit": 1})
    print(json.dumps({"prepared": 9960, "dataset_names_disambiguated": sum(r["destination_name"] != r["dataset_name"] for r in datasets)}), flush=True)


def audit_node(host, manifest):
    repo, data = roots(host)
    rows = json.loads(Path(manifest).read_text())
    missing = []
    for row in rows:
        directory = Path(data) / row["dataset_rel"].removeprefix("sim-datasets-data/")
        for filename in row["required_files"]:
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append({"dataset_rel": row["dataset_rel"], "file": filename, "reason": "missing_or_empty"})
                continue
            with path.open("rb") as handle:
                if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    missing.append({"dataset_rel": row["dataset_rel"], "file": filename, "reason": "lfs_pointer"})
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(value.split()[0]) * 1024
    print(json.dumps({"host": host, "ready": not missing and Path(repo).is_dir(), "datasets": len(rows),
        "missing": missing, "repo_exists": Path(repo).is_dir(), "cpu_count": os.cpu_count(),
        "load": list(os.getloadavg()), "memory": memory, "disk_free": shutil.disk_usage(Path(data).parent).free}), flush=True)


def audit_cluster():
    base = Path(REMOTE_AUDIT)
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "LogLevel=ERROR"]
    def one(host):
        command = ["python3", str(base / "full664_10m.py"), "audit-node", "--host", host, "--manifest", str(base / "datasets.json")]
        if host == "iaaccn22":
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        else:
            target = "zhangziwen@10.10.100." + host.removeprefix("iaaccn")
            result = subprocess.run(["ssh", *options, target, "mkdir", "-p", str(base)], capture_output=True, text=True, timeout=12)
            if result.returncode:
                return {"host": host, "ready": False, "error": result.stderr[-1000:]}
            result = subprocess.run(["scp", *options, str(base / "full664_10m.py"), str(base / "datasets.json"), target + ":" + str(base) + "/"], capture_output=True, text=True, timeout=20)
            if result.returncode:
                return {"host": host, "ready": False, "error": result.stderr[-1000:]}
            result = subprocess.run(["ssh", *options, target, shlex.join(command)], capture_output=True, text=True, timeout=60)
        if result.returncode:
            return {"host": host, "ready": False, "error": result.stderr[-1000:]}
        return json.loads(result.stdout)
    reports = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(one, host): host for host in HOSTS}
        for future in as_completed(futures):
            host = futures[future]
            try:
                report = future.result()
            except subprocess.TimeoutExpired as error:
                report = {"host": host, "ready": False, "error": str(error)}
            reports.append(report)
            dump(base / "audit.json", reports)
            print(json.dumps({k: v for k, v in report.items() if k != "missing"} | {"missing_files": len(report.get("missing", []))}), flush=True)


def queue_module():
    path = HERE / "run_core50_queue.py"
    spec = importlib.util.spec_from_file_location("full664_existing_queue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install(host):
    repo, _ = roots(host)
    subprocess.run(["tar", "-xzf", REMOTE_AUDIT + "/support.tgz", "--skip-old-files", "-C", repo], check=True)
    (Path(repo) / ".agent/work/FULL664-10M/queue/logs").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"host": host, "synced": True}), flush=True)


def distribute():
    base = Path(REMOTE_AUDIT)
    reports = json.loads((base / "audit.json").read_text())
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR"]
    def one(host):
        command = ["python3", str(base / "full664_10m.py"), "install", "--host", host]
        if host == "iaaccn22":
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        else:
            target = "zhangziwen@10.10.100." + host[6:]
            result = subprocess.run(["scp", *options, str(base / "full664_10m.py"), str(base / "support.tgz"), target + ":" + str(base) + "/"], capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                result = subprocess.run(["ssh", *options, target, shlex.join(command)], capture_output=True, text=True, timeout=90)
        return {"host": host, "synced": result.returncode == 0, "error": result.stderr[-1000:]}
    completed = []
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(one, report["host"]): report["host"] for report in reports if report["ready"]}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            dump(base / "support_sync.json", completed)
            print(json.dumps(result), flush=True)


def launch():
    reports = json.loads((CONFIG / "host_data_audit.json").read_text())
    hosts = [row["host"] for row in reports if row["ready"]]
    synced = json.loads((CONFIG / "support_sync.json").read_text())
    assert set(hosts) == {row["host"] for row in synced if row["synced"]}
    assert hosts
    module = queue_module()
    sys.argv = [str(HERE / "run_core50_queue.py"), "--batch-name", BATCH,
        "--source-csv", str(CONFIG / "datasets.csv"), "--expected-rows", "664",
        "--queue-root", str(WORK / "queue"), "--params-root", str(CONFIG / "params"),
        "--hosts", *hosts, "--tools", *TOOLS, "--seeds", "1314", "--noise-sigmas", "0",
        "--remote-root", REMOTE, "--remote-data-root", "/home/zhangziwen/sim-datasets-data",
        "--host-remote-root-overrides", ",".join(f"{h}={roots(h)[0]}" for h in hosts if int(h[6:]) >= 48),
        "--host-remote-data-root-overrides", ",".join(f"{h}={roots(h)[1]}" for h in hosts if int(h[6:]) >= 48),
        "--max-jobs-per-host", "256", "--max-new-jobs-per-host-per-poll", "64",
        "--max-cpu-used-ratio", "1.0", "--max-load-ratio", "1.0",
        "--load-tier-new-jobs", "0.65:64,0.85:32,0.95:16,1.0:8",
        "--max-memory-used-ratio", "0.92", "--min-free-mem-gb", "8", "--poll-seconds", "15",
        "--retry-limit", "1", "--host-unavailable-grace-seconds", "900",
        "--llm-model-assignment", "from-params", "--llm-default-bucket", "turbo",
        "--llm-model-bucket-limits", "base:0,turbo:0", "--no-prioritize-llm",
        "--controller-host", "iaaccn22", "--use-internal-ips",
        "--session-prefix", "full664_10m_", "--host-session-count-prefix", "full664_10m_", "--skip-support-sync"]
    module.main()


def collect():
    path = HERE / "collect_core50_new15_results.py"
    spec = importlib.util.spec_from_file_location("full664_existing_collector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.QUEUES = {BATCH: WORK / "queue"}
    module.MANIFEST = CONFIG / "collection.csv"
    module.SUMMARY = CONFIG / "collection_summary.json"
    module.REMOTE_HOME_ROOT = REMOTE
    module.REMOTE_ROOTS = {host: roots(host)[0] for host in HOSTS}
    datasets = json.loads((CONFIG / "datasets.json").read_text())
    pending = []
    for tool in TOOLS:
        for row in datasets:
            name = {"imcts": "iMCTS", "qlattice": "QLattice"}.get(tool, tool)
            destination = HERE.parent / "2.1 full set experimets/clean" / name / row["destination_name"] / "1314"
            pending.append({"noise": "clean", "algorithm": tool, "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"], "dataset_rel": row["dataset_rel"],
                "seed": "1314", "training_budget_seconds": "600", "destination": str(destination)})
    records = module.initial_records(pending)
    if module.MANIFEST.exists():
        with module.MANIFEST.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("equation_available", "numeric_available"):
                    row[key] = row[key].lower() == "true"
                for key in ("controller_attempts", "progress_file_count", "imported_file_count", "imported_bytes"):
                    row[key] = int(row[key] or 0)
                records[(row["noise"], row["algorithm"], row["dataset_id"], int(row["seed"]))] = row
    state_path = WORK / "queue/state" / f"{BATCH}.state.json"
    while True:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            selected = module.one_cycle(records, {BATCH: state}, workers=12, max_per_poll=192)
            module.persist(records)
            counts = Counter(row["collection_state"] for row in records.values())
            print(json.dumps({"time": time.time(), "collection": dict(counts)}, ensure_ascii=False), flush=True)
            if all(task["state"] in {"done", "failed"} for task in state["tasks"].values()) and all(
                row["collection_state"] in {"collected", "terminal_no_result", "result_conflict", "terminal_without_host"}
                for row in records.values()
            ):
                return
            if selected:
                continue
        time.sleep(20)


def start():
    WORK.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["tmux", "has-session", "-t", "full664_10m_controller"], capture_output=True)
    if result.returncode == 0:
        raise RuntimeError("本轮调度器已经启动")
    for name, mode in (("full664_10m_controller", "launch"), ("full664_10m_collector", "collect")):
        if subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0:
            continue
        subprocess.run(["tmux", "new-session", "-d", "-s", name, sys.executable, "-u", str(Path(__file__).resolve()), mode], check=True)
        subprocess.run(["tmux", "pipe-pane", "-o", "-t", name, "cat >> " + shlex.quote(str(WORK / f"{mode}.log"))], check=True)
    print(json.dumps({"controller": "full664_10m_controller", "collector": "full664_10m_collector"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "audit-node", "audit-cluster", "launch", "collect", "start", "install", "distribute"])
    parser.add_argument("--host")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.mode == "audit-node":
        audit_node(args.host, args.manifest)
    elif args.mode == "install":
        install(args.host)
    else:
        {"prepare": prepare, "audit-cluster": audit_cluster, "launch": launch, "collect": collect, "start": start, "distribute": distribute}[args.mode]()
