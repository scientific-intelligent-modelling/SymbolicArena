import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean
import time

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp
from scipy.stats import wasserstein_distance

import select_response_balance as base


def read_runs(probes):
    groups = defaultdict(list)
    with base.SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in probes:
                groups[row["dataset_id"], row["method"]].append(row)
    assert len(groups) == 664 * len(probes)
    return groups


def prepare(probes, k):
    path = base.SOURCE.parent / "probe4_postprocess_dataset_level.csv"
    columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
               "semantic_duplicate_group", "basename"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(({c: row[c] for c in columns} for row in csv.DictReader(handle)),
                      key=lambda row: row["dataset_id"])
    ids = [row["dataset_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 664
    groups = read_runs(probes)
    assert set(groups) == {(dataset, probe) for dataset in ids for probe in probes}
    r = np.array([[base.response(groups[dataset, probe]) for probe in probes] for dataset in ids])
    valid_rates = np.array([sum(int(row["result_valid_output"]) for probe in probes
                               for row in groups[dataset, probe]) / (3 * len(probes)) for dataset in ids])
    for row, valid_rate in zip(rows, valid_rates):
        row["construction_valid_rate"] = float(valid_rate)
        row["completion_rate"] = 1.0
    # 所有资格信息来自construction运行；元信息只读取身份及结构字段。
    eligible = valid_rates >= 0.5
    attributes = ["family", "subgroup", "semantic_duplicate_group", "basename"]
    family_counts = Counter(row["family"] for row in rows)
    hc = np.empty((len(ids), len(attributes)), dtype=int)
    metadata, lower, upper = [], [], []
    for j, attribute in enumerate(attributes):
        for label in sorted({row[attribute] for row in rows}):
            assert label
            members = [i for i, row in enumerate(rows) if row[attribute] == label]
            hc[members, j] = len(metadata)
            lo, hi = 0, 1
            if attribute == "family":
                target = k * (0.6 * family_counts[label] / 664 + 0.4 / len(family_counts))
                lo, hi = max(1, math.floor(target-k/50)), math.ceil(target+2*k/50)
            elif attribute == "subgroup":
                hi = math.ceil(8*k/50)
            lower.append(lo)
            upper.append(hi)
            metadata.append({"attribute": attribute, "label": label, "minimum": lo, "maximum": hi})
    mu, sigma = r.mean(axis=0), r.std(axis=0, ddof=0)
    assert np.isfinite(r).all() and (sigma > 0).all()
    p = {"ids": ids, "k": k, "rows": rows, "eligible": eligible,
         "hc": hc, "constraint_meta": metadata, "lower": np.array(lower), "upper": np.array(upper)}
    return p, r, (r-mu)/sigma, {"probes": probes, "mean": mu.tolist(), "std": sigma.tolist()}, columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", choices=base.PROBES, default="udsr")
    parser.add_argument("--seconds", type=float, default=170)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 170:
        parser.error("求解预算必须位于(0,170]")
    args.output.mkdir(parents=True, exist_ok=False)
    probes = [probe for probe in base.PROBES if probe != args.held_out]
    p, r, z, normalization, columns = prepare(probes, 50)
    config = {"held_out": args.held_out, "construction_probes": probes, "k": 50,
              "objective": "mean_construction_probe_standardized_W1", "min_mean_info": 0,
              "metadata_selection_columns": columns, "eligibility_probes": probes,
              "min_construction_valid_rate": 0.5, "family_proportional_weight": 0.6,
              "family_min_margin": 1, "family_max_margin": 2, "subgroup_max": 8, "duplicate_cap": 1,
              "score_version": "probe_consensus_response_v1", "normalization": normalization,
              "seconds_limit": args.seconds, "source_sha256": base.digest(base.SOURCE),
              "metadata_sha256": base.digest(base.SOURCE.parent / "probe4_postprocess_dataset_level.csv"),
              "selector_sha256": base.digest(Path(__file__)), "model_helper_sha256": base.digest(Path(base.__file__)),
              "scoring_helper_sha256": base.digest(Path(__file__).with_name("evaluate_core50.py")),
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": base.scipy.__version__},
              "heldout_used_for_selection": False, "prior_method_development_used_all_four_probes": True,
              "formal_ready": False}
    base.write_json(args.output / "configuration.json", config)
    cost, integrality, bounds, matrices, lbs, ubs, _, constraints, eligible = base.build_model(
        p, z, include_diagnostics=False)
    matrix = base.vstack(matrices, format="csc")
    low, high = np.concatenate(lbs), np.concatenate(ubs)
    started = time.monotonic()
    result = milp(cost, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(matrix, low, high),
                  options={"time_limit": args.seconds, "mip_rel_gap": 0., "presolve": True})
    elapsed = time.monotonic()-started
    if result.x is None:
        base.write_json(args.output / "failure.json", {"status": int(result.status), "message": result.message})
        raise RuntimeError(result.message)
    x = result.x[:664]
    assert np.max(np.abs(x-np.rint(x))) <= 1e-7
    selected = np.flatnonzero(x > .5)
    assert len(selected) == 50 and eligible[selected].all()
    actual = matrix @ result.x
    assert np.all(actual >= low-1e-6) and np.all(actual <= high+1e-6)
    assert np.all(result.x >= bounds.lb-1e-6) and np.all(result.x <= bounds.ub+1e-6)
    checks = []
    for item in constraints:
        count = len(set(selected) & set(item["members"]))
        assert item["minimum"] <= count <= item["maximum"]
        checks.append({key: value for key, value in item.items() if key != "members"} | {"actual": count})
    distances = [float(wasserstein_distance(z[selected, j], z[:, j])) for j in range(len(probes))]
    np.testing.assert_allclose(mean(distances), result.fun/base.SCALE, atol=1e-9, rtol=0)
    ids = [p["ids"][i] for i in selected]
    selection_path = args.output / "core50.csv"
    pd.DataFrame([p["rows"][i] for i in selected]).to_csv(selection_path, index=False)
    selection_sha = base.digest(selection_path)
    base.write_json(args.output / "constraints.json", checks)
    base.write_json(args.output / "solver_solution.json", {"solution": result.x.tolist(), "indices": selected.tolist()})
    # 名单写出之后才读取并评估held-out response。
    heldout_groups = read_runs([args.held_out])
    heldout_scores = {dataset: base.response(rows) for (dataset, _), rows in heldout_groups.items()}
    assert set(heldout_scores) == set(p["ids"])
    full, subset = mean(heldout_scores.values()), mean(heldout_scores[i] for i in ids)
    construction = [{"probe": probe, "full_mean": float(r[:, j].mean()),
                     "core_mean": float(r[selected, j].mean()),
                     "absolute_error": abs(float(r[selected, j].mean()-r[:, j].mean())),
                     "standardized_w1": distances[j]} for j, probe in enumerate(probes)]
    report = {"held_out": args.held_out, "heldout_full_mean": full, "heldout_core_mean": subset,
              "heldout_absolute_error": abs(subset-full), "construction_results": construction,
              "construction_mae": mean(item["absolute_error"] for item in construction),
              "mean_standardized_w1": mean(distances), "solver_status": int(result.status),
              "mip_gap": float(result.mip_gap), "dual_bound": float(result.mip_dual_bound)/base.SCALE,
              "seconds": elapsed, "dataset_ids": ids, "selection_sha256": selection_sha,
              "constraints_passed": True, "excluded_ids": [p["ids"][i] for i in np.flatnonzero(~eligible)],
              "heldout_used_for_selection": False, "prior_method_development_used_all_four_probes": True,
              "formal_ready": False, "historical_native_replay_verified": False}
    assert selection_sha == base.digest(selection_path)
    base.write_json(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
