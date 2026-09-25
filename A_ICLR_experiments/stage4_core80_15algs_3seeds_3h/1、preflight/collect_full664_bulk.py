import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "full664_10m"
WORK = ROOT / ".agent/work/FULL664-10M/bulk"
BATCH = "full664_15alg_clean_s1314_600s_20260926"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def roots(host):
    number = int(host[6:])
    if number < 48:
        return "/home/zhangziwen/workplace/scientific-intelligent-modelling"
    return ("/data3" if number == 49 else "/data1") + "/zhangziwen/sim-runtime/code"


def pack(plan_path, repo, output):
    plan_path, repo, output = Path(plan_path), Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive_path, manifest_path = output / "runs.tar.gz", output / "manifest.json"
    plan_sha = sha(plan_path)
    if archive_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["plan_sha256"] == plan_sha, "已有打包清单与请求不一致"
        print(json.dumps({"archive_bytes": archive_path.stat().st_size, "runs": len(manifest["runs"])}), flush=True)
        return
    manifest = {"plan_sha256": plan_sha, "runs": []}
    with tarfile.open(output / "runs.tar.gz.pending", "w:gz", compresslevel=1, dereference=True) as archive:
        for task in json.loads(plan_path.read_text()):
            directory = repo / task["source_rel"]
            files = []
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(directory))
                files.append({"path": relative, "size": path.stat().st_size, "sha256": sha(path)})
                archive.add(path, arcname=task["task_id"] + "/" + relative, recursive=False)
            launcher = repo / task["launcher_rel"]
            for path in sorted(launcher.rglob("*")):
                if path.is_file():
                    relative = "__launcher_evidence/" + str(path.relative_to(launcher))
                    files.append({"path": relative, "size": path.stat().st_size, "sha256": sha(path)})
                    archive.add(path, arcname=task["task_id"] + "/" + relative, recursive=False)
            manifest["runs"].append(dict(task, files=files, result_present=(directory / "result.json").is_file()))
    (output / "runs.tar.gz.pending").replace(archive_path)
    manifest["archive_sha256"] = sha(archive_path)
    manifest["archive_bytes"] = archive_path.stat().st_size
    dump(manifest_path, manifest)
    print(json.dumps({"archive_bytes": manifest["archive_bytes"], "runs": len(manifest["runs"])}), flush=True)


def transport(host):
    common = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR",
              "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
              "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
              "-o", "ControlPath=.agent/work/FULL664-10M/bulk/c-%C"]
    if host == "iaaccn22":
        return common, host
    proxy = shlex.join(["ssh", *common, "-W", "%h:%p", "iaaccn22"]).replace("%C", "%%C")
    return common + ["-o", "ProxyCommand=" + proxy], "zhangziwen@10.10.100." + host[6:]


def transfer(host, tasks):
    local = WORK / host
    local.mkdir(parents=True, exist_ok=True)
    plan = local / "plan.json"
    if plan.exists():
        saved = {task["task_id"]: task for task in json.loads(plan.read_text())}
        assert all(saved.get(task["task_id"]) == task for task in tasks)
    else:
        dump(plan, tasks)
    remote = roots(host) + "/.agent/work/FULL664-10M/bulk"
    options, target = transport(host)
    with (local / "transfer.log").open("a") as log:
        commands = [
            (["ssh", *options, target, "mkdir -p " + shlex.quote(remote)], 30),
            (["scp", *options, str(Path(__file__).resolve()), str(plan), target + ":" + remote + "/"], 45),
            (["ssh", *options, target, shlex.join(["python3", remote + "/collect_full664_bulk.py", "pack", "--plan", remote + "/plan.json", "--repo", roots(host), "--output", remote])], 600),
            (["rsync", "-a", "--partial", "--append-verify", "--timeout=90", "-e", shlex.join(["ssh", *options]),
              target + ":" + remote + "/runs.tar.gz", target + ":" + remote + "/manifest.json", str(local) + "/"], 1200),
        ]
        for command, timeout in commands:
            subprocess.run(command, check=True, stdout=log, stderr=log, timeout=timeout, cwd=ROOT)
    manifest = json.loads((local / "manifest.json").read_text())
    assert manifest["plan_sha256"] == sha(plan)
    assert manifest["archive_sha256"] == sha(local / "runs.tar.gz")
    extracted = local / "extracted"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(local / "runs.tar.gz", "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    for task in manifest["runs"]:
        for item in task["files"]:
            path = extracted / task["task_id"] / item["path"]
            assert path.stat().st_size == item["size"] and sha(path) == item["sha256"], path
    dump(CONFIG / "bulk_manifests" / f"{host}.json", manifest)
    return manifest


def valid_result(path, record):
    result = json.loads(path.read_text())
    assert int(result["task_global_index"]) == int(record["dataset_id"][1:])
    assert result["expected_dataset_rel"] == record["dataset_rel"]
    assert result["tool"].lower() == record["algorithm"].lower()
    assert int(result["seed"]) == 1314
    assert result["dataset_identity_check"]["match"] is True
    assert result["params"]["timeout_in_seconds"] == 600
    return result


def result_fields(result):
    expression = result.get("equation") or (result.get("canonical_artifact") or {}).get("instantiated_expression")
    numeric = all(isinstance(result.get(split), dict) and isinstance(result[split].get("nmse"), (int, float))
                  and math.isfinite(result[split]["nmse"]) for split in ("id_test", "ood_test"))
    return {"result_status": result.get("status", ""), "equation_available": bool(expression), "numeric_available": numeric}


def publish_task(task, record, host):
    source = WORK / host / "extracted" / task["task_id"]
    destination = Path(record["destination"])
    conflicts = [item["path"] for item in task["files"] if (destination / item["path"]).exists()
                 and sha(destination / item["path"]) != item["sha256"]]
    if conflicts:
        record.update(collection_state="result_conflict", collection_error="different existing files: " + ",".join(conflicts[:8]))
        return
    if task["result_present"]:
        result = valid_result(source / "result.json", record)
    elif task["controller_state"] != "failed":
        record.update(collection_state="terminal_no_result", collection_error="remote result.json missing")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in task["files"]:
        path = destination / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item["path"], path)
    record.update(source_run_dir=roots(host) + "/" + task["source_rel"],
                  imported_file_count=len(task["files"]), imported_bytes=sum(f["size"] for f in task["files"]),
                  progress_file_count=sum(Path(f["path"]).parent.name == "progress" and Path(f["path"]).name.startswith("minute_") for f in task["files"]))
    if task["result_present"]:
        record.update(collection_state="collected", collection_error="", result_sha256=sha(destination / "result.json"), **result_fields(result))
    else:
        record.update(collection_state="terminal_no_result", collection_error="failed training; original launcher evidence collected", result_sha256="", result_status="error", equation_available=False, numeric_available=False)
    binding = {"schema": "full664.bulk_collection.v1", "batch": BATCH, "source_host": host,
               "source_run_dir": record["source_run_dir"], "result_sha256": record["result_sha256"],
               "file_count": record["imported_file_count"], "total_bytes": record["imported_bytes"],
               "progress_files": record["progress_file_count"], "controller_task_id": task["task_id"],
               "manifest_path": str(CONFIG / "bulk_manifests" / f"{host}.json"), **{k: record[k] for k in ("result_status", "equation_available", "numeric_available")}}
    binding_path = destination / "import_binding.json"
    if not binding_path.exists():
        dump(binding_path, binding)
    else:
        assert json.loads(binding_path.read_text())["result_sha256"] == record["result_sha256"]


def collect():
    WORK.mkdir(parents=True, exist_ok=True)
    state = json.loads((ROOT / ".agent/work/FULL664-10M/queue/state" / f"{BATCH}.state.json").read_text())
    assert all(task["state"] in {"done", "failed"} for task in state["tasks"].values())
    original = WORK / "input_collection.csv"
    if not original.exists():
        shutil.copy2(CONFIG / "collection.csv", original)
        shutil.copy2(ROOT / ".agent/work/FULL664-10M/queue/state" / f"{BATCH}.state.json", WORK / "input_queue.json")
    with (CONFIG / "collection.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    grouped = defaultdict(list)
    for row in rows:
        for field in ("equation_available", "numeric_available"):
            row[field] = row[field].lower() == "true"
        for field in ("controller_attempts", "progress_file_count", "imported_file_count", "imported_bytes"):
            row[field] = int(row[field] or 0)
        task_id = f"{row['algorithm']}_s1314_clean_{row['dataset_id']}"
        task = state["tasks"][task_id]
        row.update(controller_state=task["state"], controller_task_id=task_id, assigned_host=task["assigned_host"], controller_attempts=task["attempts"])
        records[task_id] = row
        destination = Path(row["destination"])
        result, binding = destination / "result.json", destination / "import_binding.json"
        if result.is_file() and binding.is_file():
            saved = json.loads(binding.read_text())
            if saved.get("source_host") == task["assigned_host"] and saved.get("result_sha256") == sha(result):
                parsed = valid_result(result, row)
                row.update(collection_state="collected", collection_error="", result_sha256=saved["result_sha256"], **result_fields(parsed))
                continue
        row["collection_state"] = "awaiting_bulk_sync"
        tool_dir = {"imcts": "iMCTS", "qlattice": "QLattice"}.get(task["tool"], task["tool"])
        base = f"experiments/{BATCH}/{task['tool']}/seed1314/tasks/{task_id}/{task['assigned_host']}"
        grouped[task["assigned_host"]].append({"task_id": task_id, "controller_state": task["state"],
            "source_rel": base + f"/{tool_dir}/{row['dataset_id']}_{row['dataset_name']}", "launcher_rel": base + "/__launcher__"})
    spec = importlib.util.spec_from_file_location("bulk_existing_collector", HERE / "collect_core50_new15_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MANIFEST = CONFIG / "collection.csv"
    module.SUMMARY = CONFIG / "collection_summary.json"
    module.QUEUES = {BATCH: None}
    module.persist(records)
    print(json.dumps({"remaining_by_host": {host: len(tasks) for host, tasks in grouped.items()}}), flush=True)
    report_path = CONFIG / "bulk_sync_status.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(transfer, host, tasks): host for host, tasks in sorted(grouped.items(), key=lambda item: item[0] == "iaaccn49")}
        for future in as_completed(futures):
            host = futures[future]
            try:
                manifest = future.result()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                report[host] = {"state": "transport_error", "error": str(error)}
                for task in grouped[host]:
                    records[task["task_id"]].update(collection_state="sync_error", collection_error=str(error))
            else:
                for task in manifest["runs"]:
                    publish_task(task, records[task["task_id"]], host)
                report[host] = {"state": "verified", "runs": len(manifest["runs"]), "archive_bytes": manifest["archive_bytes"]}
            module.persist(records)
            dump(CONFIG / "bulk_sync_status.json", report)
            print(json.dumps({"host": host, **report[host], "counts": dict(Counter(row["collection_state"] for row in records.values()))}), flush=True)
    return dict(Counter(row["collection_state"] for row in records.values()))


def seal():
    for path in WORK.glob("iaaccn*/manifest.json"):
        manifest = json.loads(path.read_text())
        assert sha(path.parent / "runs.tar.gz") == manifest["archive_sha256"]
        dump(CONFIG / "bulk_manifests" / (path.parent.name + ".json"), manifest)
    for source, target in ((WORK / "input_queue.json", CONFIG / "bulk_input_queue.json"),
                           (WORK / "input_collection.csv", CONFIG / "bulk_input_collection.csv")):
        if target.exists():
            assert sha(source) == sha(target)
        else:
            shutil.copy2(source, target)


def watch():
    while True:
        with (CONFIG / "collection.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        hosts = sorted({row["assigned_host"] for row in rows if row["collection_state"] == "sync_error"})
        if not hosts:
            return
        time.sleep(60)
        reachable = []
        errors = {}
        for host in hosts:
            options, target = transport(host)
            try:
                result = subprocess.run(["ssh", *options, target, "hostname"], capture_output=True, text=True, timeout=20, cwd=ROOT)
            except subprocess.TimeoutExpired as error:
                errors[host] = str(error)
                continue
            if result.returncode == 0:
                reachable.append(host)
            else:
                errors[host] = result.stderr.strip()
        dump(CONFIG / "bulk_retry_status.json", {"time": time.time(), "waiting_hosts": hosts, "reachable_hosts": reachable, "transport_errors": errors})
        print(json.dumps({"waiting_hosts": hosts, "reachable_hosts": reachable}), flush=True)
        if reachable:
            collect()
            seal()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pack", "collect", "seal", "watch"))
    parser.add_argument("--plan")
    parser.add_argument("--repo")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "pack":
        pack(args.plan, args.repo, args.output)
    elif args.mode == "collect":
        collect()
    elif args.mode == "seal":
        seal()
    else:
        watch()
