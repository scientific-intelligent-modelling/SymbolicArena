#!/usr/bin/env python3
"""在 iaaccn22 上并行构建并汇集 clean165 + noise309 host bundles。"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/zhangziwen/workplace/scientific-intelligent-modelling")
RELEASE = ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/final_release_20260913/release_v2"
OUTPUT = RELEASE / "eff_native_raw_bundles_20260914"
COLLECTOR = ROOT / "check/collect_eff_native_host_bundle.py"
BATCHES = {
    "eff_native_missing165_20260913": RELEASE / "eff_native_missing165_20260913/formal_queue/state/eff_native_missing165_20260913.state.json",
    "eff_native_noise_missing309_20260914": RELEASE / "eff_native_noise_missing309_20260914/formal_queue/state/eff_native_noise_missing309_20260914.state.json",
}


def run(command, timeout=1800):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout)


def main():
    grouped = defaultdict(list)
    for batch, state_path in BATCHES.items():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for task in state["tasks"].values():
            item = dict(task)
            item["experiment_root"] = str(ROOT / "experiments" / batch)
            item["source_batch"] = batch
            grouped[task["assigned_host"]].append(item)
    if sum(map(len, grouped.values())) != 474:
        raise RuntimeError("combined task count is not 474")
    request_dir = OUTPUT / "requests"
    bundle_dir = OUTPUT / "host_bundles"
    request_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    def collect(host, tasks):
        request = request_dir / f"{host}.json"
        request.write_text(
            json.dumps({"host": host, "tasks": tasks}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output = bundle_dir / f"{host}.jsonl.gz"
        if host == "iaaccn22":
            proc = run(["python", str(COLLECTOR), "--request", str(request), "--output", str(output)])
        else:
            target = "10.10.100." + host.replace("iaaccn", "")
            run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, f"mkdir -p {shlex.quote(str(request_dir))} {shlex.quote(str(bundle_dir))}"], 60)
            copy = run(["rsync", "-a", str(request), f"{target}:{request}"], 120)
            if copy.returncode:
                return host, False, copy.stderr
            proc = run([
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target,
                shlex.join(["python", str(COLLECTOR), "--request", str(request), "--output", str(output)]),
            ])
            if proc.returncode == 0:
                for remote_path in (str(output), f"{output}.report.json"):
                    copy = run(
                        ["rsync", "-a", f"{target}:{remote_path}", str(bundle_dir) + "/"],
                        600,
                    )
                    if copy.returncode:
                        return host, False, copy.stderr
        return host, proc.returncode == 0, proc.stderr or proc.stdout

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(collect, host, tasks): host for host, tasks in grouped.items()}
        for future in as_completed(futures):
            results.append(future.result())
    failures = [item for item in results if not item[1]]
    report = {
        "status": "passed" if not failures else "failed",
        "task_count": 474,
        "hosts": sorted({host: len(grouped[host]) for host in grouped}.items()),
        "results": sorted(results),
    }
    (OUTPUT / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
