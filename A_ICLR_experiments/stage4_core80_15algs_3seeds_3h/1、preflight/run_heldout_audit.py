from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys
import time

from scipy.stats import wasserstein_distance

import select_response_balance as base


HERE = Path(__file__).resolve().parent
WORK = base.ROOT / ".agent/work/METHOD-005"
SELECTOR = HERE / "select_heldout_response.py"


def run_fold(probe):
    output = HERE / f"heldout_{probe}_core50"
    if (output / "summary.json").exists():
        return {"probe": probe, "reused": True}
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(WORK))
    command = [sys.executable, str(SELECTOR), "--held-out", probe, "--seconds", "170",
               "--output", str(output)]
    started = time.monotonic()
    with (WORK / f"{probe}.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=180)
    return {"probe": probe, "reused": False, "command": command,
            "elapsed_seconds": time.monotonic()-started, "thread_environment": {
                key: env[key] for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}}


def main():
    destination = HERE / "heldout_loo_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    WORK.mkdir(parents=True, exist_ok=True)
    # 每个求解进程独立执行170秒预算，已有通过的结果保留。
    with ThreadPoolExecutor(max_workers=3) as executor:
        execution = list(executor.map(run_fold, base.PROBES))
    groups = defaultdict(list)
    with base.SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 7968
    for row in raw:
        groups[row["dataset_id"], row["method"]].append(row)
    scores = {probe: {dataset: base.response(rows) for (dataset, method), rows in groups.items()
                      if method == probe} for probe in base.PROBES}
    metadata_path = base.SOURCE.parent / "probe4_postprocess_dataset_level.csv"
    with metadata_path.open(newline="") as handle:
        metadata = {row["dataset_id"]: row for row in csv.DictReader(handle)}
    family_counts = Counter(row["family"] for row in metadata.values())
    expected_hashes = {"selector_sha256": base.digest(SELECTOR), "source_sha256": base.digest(base.SOURCE),
                       "model_helper_sha256": base.digest(Path(base.__file__)),
                       "scoring_helper_sha256": base.digest(HERE / "evaluate_core50.py"),
                       "metadata_sha256": base.digest(metadata_path)}
    folds = []
    for heldout in base.PROBES:
        output = HERE / f"heldout_{heldout}_core50"
        report = json.loads((output / "summary.json").read_text())
        config = json.loads((output / "configuration.json").read_text())
        assert all(config[key] == value for key, value in expected_hashes.items())
        assert config["held_out"] == heldout and config["min_mean_info"] == 0
        construction = [p for p in base.PROBES if p != heldout]
        assert config["construction_probes"] == config["eligibility_probes"] == construction
        assert config["seconds_limit"] == 170 and not config["heldout_used_for_selection"]
        path = output / "core50.csv"
        assert report["selection_sha256"] == base.digest(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["dataset_id"] for row in rows]
        assert len(ids) == len(set(ids)) == 50 and ids == report["dataset_ids"]
        for column in ["semantic_duplicate_group", "basename"]:
            assert len({metadata[i][column] for i in ids}) == 50
        assert max(Counter(metadata[i]["subgroup"] for i in ids).values()) <= 8
        selected_families = Counter(metadata[i]["family"] for i in ids)
        for family, count in family_counts.items():
            target = 50*(0.6*count/664 + 0.4/len(family_counts))
            assert max(1, math.floor(target-1)) <= selected_families[family] <= math.ceil(target+2)
        for i in ids:
            valid_rate = sum(int(row["result_valid_output"]) for p in construction for row in groups[i, p])/9
            assert valid_rate >= 0.5
        errors, distances = {}, {}
        for p in base.PROBES:
            population = list(scores[p].values())
            subset = [scores[p][i] for i in ids]
            assert len(population) == 664
            errors[p] = abs(mean(subset)-mean(population))
            distances[p] = wasserstein_distance(subset, population)/pstdev(population)
        construction_mae = mean(errors[p] for p in construction)
        assert abs(errors[heldout]-report["heldout_absolute_error"]) < 1e-12
        assert abs(construction_mae-report["construction_mae"]) < 1e-12
        assert abs(mean(distances[p] for p in construction)-report["mean_standardized_w1"]) < 1e-12
        folds.append({"held_out": heldout, "construction_mae": construction_mae,
                      "heldout_absolute_error": errors[heldout],
                      "heldout_signed_error": mean(scores[heldout][i] for i in ids)-mean(scores[heldout].values()),
                      "heldout_full_mean": mean(scores[heldout].values()),
                      "heldout_core_mean": mean(scores[heldout][i] for i in ids),
                      "solver_gap": report["mip_gap"], "solver_status": report["solver_status"],
                      "excluded_count": len(report["excluded_ids"]),
                      "selection_sha256": base.digest(path), "output": str(output.relative_to(base.ROOT)),
                      "verified": True})
    report = {"method": "pure_W1_leave_one_probe_out", "fold_count": 4, "k": 50,
              "min_mean_info": 0, "folds": folds, "execution": execution,
              "heldout_mae": mean(f["heldout_absolute_error"] for f in folds),
              "construction_mae": mean(f["construction_mae"] for f in folds),
              "worst_heldout_error": max(f["heldout_absolute_error"] for f in folds),
              "folds_at_most_015": sum(f["heldout_absolute_error"] <= 0.15 for f in folds),
              "score_version": "probe_consensus_response_v1", "source_hashes": expected_hashes,
              "audit_script_sha256": base.digest(Path(__file__)),
              "prior_method_development_used_all_four_probes": True, "formal_ready": False}
    base.write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
