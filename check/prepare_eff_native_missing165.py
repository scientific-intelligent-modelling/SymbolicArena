#!/usr/bin/env python3
"""冻结 EFF native telemetry 精确 165 条 clean 重跑资产。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/final_release_20260913/release_v2"
AUDIT_ALLOWLIST = RELEASE_ROOT / "eff_revision/rerun_or_recollect_allowlist.csv"
CORE50 = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
OUTPUT_ROOT = RELEASE_ROOT / "eff_native_missing165_20260913"
BASE_PARAMS = REPO_ROOT / "exp-planning/02.E1选择验证/generated/params"
EXPECTED = {"e2esr": 150, "imcts": 12, "pysr": 3}
CODE_FILES = (
    "scientific_intelligent_modelling/benchmarks/runner.py",
    "scientific_intelligent_modelling/algorithms/e2esr_wrapper/wrapper.py",
    "scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/model_wrapper.py",
    "scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/sklearn_wrapper.py",
    "scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/symbolicregression/model/transformer.py",
    "scientific_intelligent_modelling/algorithms/iMCTS_wrapper/wrapper.py",
    "scientific_intelligent_modelling/algorithms/iMCTS_wrapper/MCTS-4-SR/iMCTS/regressor.py",
    "check/run_e1_candidate200_12alg_load_queue.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    core = _read(CORE50)
    by_name = {row["dataset_name"]: row for row in core}
    rows = []
    for source in _read(AUDIT_ALLOWLIST):
        algorithm = source["algorithm"].strip().lower()
        if algorithm not in EXPECTED:
            continue
        if source["condition"] != "clean" or int(source["seed"]) not in {520, 521, 522}:
            raise ValueError(f"非正式 clean identity: {source}")
        dataset = by_name.get(source["dataset_id"])
        if dataset is None:
            raise ValueError(f"未命中 Core50: {source['dataset_id']}")
        expected_task_id = f"{algorithm}_s{source['seed']}_clean_g{int(dataset['core50_index']):04d}"
        rows.append(
            {
                "scheduler_task_id": expected_task_id,
                "task_id": expected_task_id,
                "source_audit_task_id": source["task_id"],
                "algorithm": algorithm,
                "dataset_id": source["dataset_id"],
                "core50_index": dataset["core50_index"],
                "seed": source["seed"],
                "condition": "clean",
                "source_audit_host": source["host"],
                "source_audit_reason": source["reason"],
                "required_action": "rerun_with_algorithm_native_incumbent_telemetry",
            }
        )
    counts = Counter(row["algorithm"] for row in rows)
    if counts != Counter(EXPECTED) or len(rows) != 165:
        raise ValueError(f"精确范围漂移: {dict(counts)}, total={len(rows)}")
    if len({row["task_id"] for row in rows}) != 165:
        raise ValueError("allowlist task_id 重复")
    rows.sort(key=lambda row: (row["algorithm"], int(row["seed"]), int(row["core50_index"])))
    output_root.mkdir(parents=True, exist_ok=True)
    allowlist = output_root / "allowlist_165.csv"
    with allowlist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    params_root = output_root / "params"
    params_root.mkdir(parents=True, exist_ok=True)
    params_hashes = {}
    for tool in EXPECTED:
        payload = json.loads((BASE_PARAMS / f"{tool}.json").read_text(encoding="utf-8"))
        payload["timeout_in_seconds"] = 10800
        payload["progress_snapshot_interval_seconds"] = 60
        path = params_root / f"{tool}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        clean_path = params_root / f"{tool}__clean.json"
        clean_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        params_hashes[tool] = _sha256(path)
    manifest = {
        "schema_version": "eff_native_missing165.v1",
        "batch_name": "eff_native_missing165_20260913",
        "condition": "clean",
        "seeds": [520, 521, 522],
        "expected_counts": EXPECTED,
        "run_count": len(rows),
        "allowlist": str(allowlist),
        "allowlist_sha256": _sha256(allowlist),
        "params_sha256": params_hashes,
        "timeout_in_seconds": 10800,
        "snapshot_interval_seconds": 60,
        "source_audit": str(AUDIT_ALLOWLIST),
        "source_audit_sha256": _sha256(AUDIT_ALLOWLIST),
        "formal_output_isolated": True,
        "code_sha256": {name: _sha256(REPO_ROOT / name) for name in CODE_FILES},
        "scheduler_limits": {
            "max_load_ratio": 0.95,
            "max_memory_used_ratio": 0.95,
            "min_free_mem_gb": 16,
            "max_cpu_used_ratio": 0.95,
        },
        "remote_batch_root": (
            "/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/"
            "eff_native_missing165_20260913"
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True))
