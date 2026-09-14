#!/usr/bin/env python3
"""冻结 noise001/noise005 native EFF 精确 309 条重跑资产。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831/work/final_release_20260913/release_v2"
SOURCE = RELEASE / "eff_noise_revision/unavailable_all_conditions.csv"
CORE50 = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
OUTPUT = RELEASE / "eff_native_noise_missing309_20260914"
NOISE_PARAMS = REPO_ROOT / "exp-planning/05.Core50噪声鲁棒性评测/generated/params"
EXPECTED = {
    ("noise001", "gplearn"): 1,
    ("noise001", "e2esr"): 150,
    ("noise001", "pysr"): 3,
    ("noise005", "e2esr"): 150,
    ("noise005", "llmsr"): 1,
    ("noise005", "pyoperon"): 1,
    ("noise005", "pysr"): 3,
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path = OUTPUT) -> dict[str, object]:
    core = {row["dataset_name"]: row for row in _read(CORE50)}
    rows = []
    for source in _read(SOURCE):
        condition = source["condition"]
        algorithm = source["algorithm"].lower()
        if (condition, algorithm) not in EXPECTED:
            continue
        dataset = core.get(source["dataset_id"])
        if dataset is None:
            raise ValueError(f"未命中 Core50: {source['dataset_id']}")
        seed = int(source["seed"])
        if seed not in {520, 521, 522}:
            raise ValueError(f"非法 seed: {seed}")
        scheduler_id = (
            f"{algorithm}_s{seed}_{condition}_g{int(dataset['core50_index']):04d}"
        )
        rows.append(
            {
                "scheduler_task_id": scheduler_id,
                "task_id": scheduler_id,
                "source_audit_task_id": source["task_id"],
                "condition": condition,
                "algorithm": algorithm,
                "dataset_id": source["dataset_id"],
                "core50_index": dataset["core50_index"],
                "seed": str(seed),
                "source_audit_host": source["host"],
                "source_audit_reason": source["reason"],
                "required_action": "rerun_with_algorithm_native_incumbent_telemetry",
            }
        )
    counts = Counter((row["condition"], row["algorithm"]) for row in rows)
    if counts != Counter(EXPECTED) or len(rows) != 309:
        raise ValueError(f"exact309 漂移: {dict(counts)}, total={len(rows)}")
    if len({row["task_id"] for row in rows}) != 309:
        raise ValueError("allowlist task_id 重复")
    rows.sort(
        key=lambda row: (
            row["condition"], row["algorithm"], int(row["seed"]), int(row["core50_index"])
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    allowlist = output / "allowlist_309.csv"
    with allowlist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    params_root = output / "params"
    params_root.mkdir(parents=True, exist_ok=True)
    param_hashes = {}
    for condition, algorithm in EXPECTED:
        sigma_dir = "sigma001" if condition == "noise001" else "sigma005"
        source_name = "llmsr_turbo.json" if algorithm == "llmsr" else f"{algorithm}.json"
        payload = json.loads((NOISE_PARAMS / sigma_dir / source_name).read_text(encoding="utf-8"))
        payload["timeout_in_seconds"] = 10800
        payload["progress_snapshot_interval_seconds"] = 60
        path = params_root / f"{algorithm}__{condition}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        param_hashes[f"{condition}/{algorithm}"] = _sha(path)

    llm_params = json.loads((params_root / "llmsr__noise005.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "eff_native_noise_missing309.v1",
        "batch_name": "eff_native_noise_missing309_20260914",
        "run_count": 309,
        "counts": {f"{condition}/{algorithm}": count for (condition, algorithm), count in EXPECTED.items()},
        "allowlist": str(allowlist),
        "allowlist_sha256": _sha(allowlist),
        "params_sha256": param_hashes,
        "source_audit": str(SOURCE),
        "source_audit_sha256": _sha(SOURCE),
        "timeout_in_seconds": 10800,
        "snapshot_interval_seconds": 60,
        "candidate_selection_uses_id_ood": False,
        "llmsr_external_generation_upper_bound": int(llm_params["niterations"])
        * int(llm_params["samples_per_iteration"]),
        "llmsr_model_assignment": llm_params.get("llm_model_assignment"),
        "llmsr_config_path": llm_params.get("llm_config_path"),
        "secret_material_recorded": False,
        "scheduler_limits": {
            "max_load_ratio": 0.95,
            "max_memory_used_ratio": 0.95,
            "min_free_mem_gb": 16,
            "max_cpu_used_ratio": 0.95,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True))
