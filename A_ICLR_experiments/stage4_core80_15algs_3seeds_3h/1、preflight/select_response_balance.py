import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, diags, eye, hstack, vstack
from scipy.stats import wasserstein_distance

from evaluate_core50 import PROBES, ROOT, SOURCE, response


PACKAGE = ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/1、build_core30_80/core30_core80_all_sensitivity_package/core_nested_all_sensitivity"
sys.path.insert(0, str(PACKAGE))
from lib.engine import prepare

SCALE = 1e6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_data(k):
    config = json.loads((PACKAGE / "configs/core50.json").read_text())
    config.update(target_size=k, subgroup_max=int(np.ceil(8*k/50)),
                  family_min_margin=k/50, family_max_margin=2*k/50)
    source = SOURCE.parent
    p = prepare(source / "probe4_postprocess_dataset_level.csv",
                source / "probe4_postprocess_dataset_algorithm.csv", config)
    assert digest(SOURCE) == "1a588501a0b31b5b392592b05c3105489aee18ab19e3cffb190102e6bee5ddcb"
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 7968
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["dataset_id"], row["method"]].append(row)
    r = np.array([[response(grouped[dataset, probe]) for probe in PROBES] for dataset in p["ids"]])
    mu, sigma = r.mean(axis=0), r.std(axis=0, ddof=0)
    assert np.isfinite(r).all() and (sigma > 0).all()
    z = (r-mu)/sigma
    return p, r, z, config, {"mean": mu.tolist(), "std": sigma.tolist()}


def build_model(p, z, include_diagnostics=True):
    n, k = len(p["ids"]), p["k"]
    g = len(p["coeff"]) if include_diagnostics else 0
    supports = []
    for column in range(z.shape[1]):
        levels = np.unique(z[:, column])
        levels = levels[:-1]
        cumulative = csc_matrix((z[:, column][None, :] <= levels[:, None]).astype(float))
        gaps = np.diff(np.unique(z[:, column]))
        supports.append((cumulative, gaps, k*np.asarray(cumulative.sum(axis=1)).ravel()/n))
    widths = [len(gaps) for _, gaps, _ in supports]
    offsets = np.cumsum([n+g] + widths)
    c_index, i_index, tc_index, ti_index = range(int(offsets[-1]), int(offsets[-1])+4)
    nv = ti_index+1
    cost = np.zeros(nv)
    lower = np.zeros(nv)
    upper = np.ones(nv)
    upper[:n] = p["eligible"]
    lower[[c_index, i_index]] = 1e-12
    lower[[tc_index, ti_index]], upper[[tc_index, ti_index]] = -np.inf, 0
    cost[[tc_index, ti_index]] = -SCALE
    if not include_diagnostics:
        lower[[c_index, i_index, tc_index, ti_index]] = 0
        upper[[c_index, i_index, tc_index, ti_index]] = 0
        cost[[tc_index, ti_index]] = 0
    matrices, lbs, ubs = [], [], []

    def append(matrix, lo, hi):
        matrices.append(matrix)
        lbs.append(np.broadcast_to(lo, matrix.shape[0]).astype(float))
        ubs.append(np.broadcast_to(hi, matrix.shape[0]).astype(float))

    # 结构性数量约束共享同一组规则，响应平衡由连续分布距离计算。
    active_constraints = []
    for j, meta in enumerate(p["constraint_meta"]):
        if meta["attribute"] not in {"family", "subgroup", "semantic_duplicate_group", "basename"}:
            continue
        members = np.flatnonzero((p["hc"] == j).any(axis=1))
        matrix = csc_matrix((np.ones(len(members)), (np.zeros(len(members)), members)), shape=(1, nv))
        append(matrix, p["lower"][j], p["upper"][j])
        active_constraints.append({**meta, "members": members.tolist()})
    append(csc_matrix((np.ones(n), (np.zeros(n), np.arange(n))), shape=(1, nv)), k, k)
    eligible = p["eligible"].copy()
    if not all(float(row["completion_rate"]) == 1 for row in p["rows"]):
        raise ValueError("输入包含未完成运行")

    for j in range(g):
        members = np.flatnonzero((p["codes"] == j).any(axis=1))
        indices = np.r_[members, n+j]
        append(csc_matrix((np.r_[-np.ones(len(members)), 1.0],
                           (np.zeros(len(indices)), indices)), shape=(1, nv)), -np.inf, 0)
        count = len(members)
        columns = np.column_stack([members, np.full(count, n+j)]).ravel()
        append(csc_matrix((np.tile([1., -1.], count), (np.repeat(np.arange(count), 2), columns)),
                           shape=(count, nv)), -np.inf, 0)
    if include_diagnostics:
        append(csc_matrix((np.r_[-p["coeff"], 1.],
                            (np.zeros(g+1), np.r_[np.arange(n, n+g), c_index])), shape=(1, nv)), 0, 0)
        append(csc_matrix((np.r_[-p["info"]/k, 1.],
                            (np.zeros(n+1), np.r_[np.arange(n), i_index])), shape=(1, nv)), 0, 0)

    for column, (cumulative, gaps, target) in enumerate(supports):
        start, end = offsets[column:column+2]
        size = len(gaps)
        identity = eye(size, format="csc")
        cost[start:end] = gaps/k/z.shape[1]*SCALE
        upper[start:end] = k
        def embed(left, right):
            return hstack([left, csc_matrix((size, start-n)), right,
                           csc_matrix((size, nv-end))], format="csc")
        append(embed(cumulative, -identity), -np.inf, target)
        append(embed(-cumulative, -identity), -np.inf, -target)
        floor = np.floor(target)
        fraction = target-floor
        slope = 1-2*fraction
        intercept = fraction-slope*floor
        counts = np.arange(k+1)[None, :]
        if np.any(slope[:, None]*counts+intercept[:, None] > np.abs(counts-target[:, None])+1e-11):
            raise AssertionError("整数CDF约束检查失败")
        append(embed(diags(slope) @ cumulative, -identity), -np.inf, -intercept)
    integrality = np.r_[np.ones(n, dtype=int), np.zeros(nv-n, dtype=int)]
    return cost, integrality, Bounds(lower, upper), matrices, lbs, ubs, (c_index, i_index, tc_index, ti_index), active_constraints, eligible


def terms(p, z, selected):
    counts = np.bincount(p["codes"][selected].ravel(), minlength=len(p["coeff"]))
    coverage = float(p["coeff"][counts > 0].sum())
    info = float(p["info"][selected].mean())
    distances = [float(wasserstein_distance(z[selected, j], z[:, j])) for j in range(len(PROBES))]
    distance = float(np.mean(distances))
    log_objective = float(np.log(coverage)+np.log(info)-distance)
    return {"coverage": coverage, "mean_info": info, "per_probe_standardized_w1": distances,
            "mean_standardized_w1": distance, "response_balance": float(np.exp(-distance)),
            "log_objective": log_objective, "geometric_objective": float(np.exp(log_objective/3))}


def solve(p, z, seconds):
    started = time.monotonic()
    cost, integrality, bounds, matrices, lbs, ubs, indices, constraints, eligible = build_model(p, z)
    ci, ii, tc, ti = indices
    tangents = [set(), set()]
    def tangent(component, proxy, value, number):
        value = float(value)
        if value <= 0 or value in tangents[number]:
            return
        tangents[number].add(value)
        matrices.append(csc_matrix(([1., -1/value], ([0, 0], [proxy, component])), shape=(1, len(cost))))
        lbs.append(np.array([-np.inf]))
        ubs.append(np.array([np.log(value)-1]))
    for j, (component, proxy) in enumerate([(ci, tc), (ii, ti)]):
        for value in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            tangent(component, proxy, value, j)
    best_selected, best_terms, upper_bound = None, None, float("inf")
    rounds = []
    for round_id in range(30):
        remaining = seconds-(time.monotonic()-started)
        if remaining <= 0:
            break
        matrix = vstack(matrices, format="csc")
        low, high = np.concatenate(lbs), np.concatenate(ubs)
        result = milp(cost, integrality=integrality, bounds=bounds,
                      constraints=LinearConstraint(matrix, low, high),
                      options={"time_limit": remaining, "mip_rel_gap": 0., "presolve": True})
        if result.x is None:
            raise RuntimeError(result.message)
        n = len(p["ids"])
        x = result.x[:n]
        if np.max(np.abs(x-np.rint(x))) > 1e-7:
            raise AssertionError("整数性核验失败")
        selected = np.flatnonzero(x > 0.5)
        assert len(selected) == p["k"] and eligible[selected].all()
        for item in constraints:
            count = len(set(selected) & set(item["members"]))
            assert item["minimum"] <= count <= item["maximum"], item
        evaluated = terms(p, z, selected)
        if best_terms is None or evaluated["log_objective"] > best_terms["log_objective"]:
            best_selected, best_terms = selected.copy(), evaluated
        dual = result.get("mip_dual_bound")
        if dual is not None and np.isfinite(dual):
            upper_bound = min(upper_bound, -float(dual)/SCALE)
        gap = upper_bound-best_terms["log_objective"]
        rounds.append({"round": round_id+1, "solver_status": int(result.status),
                       "best_log_objective": best_terms["log_objective"],
                       "global_upper_log_bound": upper_bound if np.isfinite(upper_bound) else None,
                       "gap_log": gap if np.isfinite(gap) else None})
        print(json.dumps(rounds[-1]), flush=True)
        if gap <= 1e-7:
            break
        tangent(ci, tc, evaluated["coverage"], 0)
        tangent(ii, ti, evaluated["mean_info"], 1)
        tangent(ci, tc, result.x[ci], 0)
        tangent(ii, ti, result.x[ii], 1)
    if best_selected is None:
        raise RuntimeError("未产生可行名单")
    gap = upper_bound-best_terms["log_objective"]
    return best_selected, {"terms": best_terms, "seconds": time.monotonic()-started,
                          "rounds": rounds, "certified_within_log_tolerance": bool(gap <= 1e-7),
                          "gap_log": gap if np.isfinite(gap) else None,
                          "constraints_passed": True, "historical_members_used_in_optimization": False,
                          "mae_used_in_optimization": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "response_balance_core50")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise FileExistsError("已有候选结果，须使用独立目录")
    p, r, z, config, normalization = load_data(args.size)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"score_version": "probe_consensus_response_v1",
                "method": "geometric_coverage_info_marginal_response_balance",
                "objective": "(Coverage * MeanInfo * exp(-mean_probe_standardized_W1)) ** (1/3)",
                "k": args.size, "seconds_limit": args.seconds, "normalization": normalization,
                "source_sha256": digest(SOURCE), "metadata_sha256": p["validation"]["dataset_sha256"],
                "aggregate_sha256": p["validation"]["aggregate_sha256"],
                "selector_sha256": digest(Path(__file__)),
                "scoring_helper_sha256": digest(Path(__file__).with_name("evaluate_core50.py")),
                "metadata_helper_sha256": digest(PACKAGE / "lib/engine.py"),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                "family_proportional_weight": config["family_proportional_weight"],
                "family_min_margin": config["family_min_margin"], "family_max_margin": config["family_max_margin"],
                "subgroup_max": config["subgroup_max"], "min_dataset_valid_rate": config["min_dataset_valid_rate"],
                "duplicate_cap": 1, "attributes": config["attributes"],
                "historical_native_replay_verified": False, "formal_ready": False}
    write_json(args.output / "configuration.json", manifest)
    selected, report = solve(p, z, args.seconds)
    probe_results = []
    for j, probe in enumerate(PROBES):
        full, subset = float(r[:, j].mean()), float(r[selected, j].mean())
        probe_results.append({"probe": probe, "full_mean_response": full, "selected_mean_response": subset,
                              "signed_error": subset-full, "absolute_error": abs(subset-full)})
    report.update(k=args.size, probe_results=probe_results,
                  mae=float(np.mean([row["absolute_error"] for row in probe_results])),
                  dataset_ids=[p["ids"][i] for i in selected], formal_ready=False, loo_executed=False,
                  historical_native_replay_verified=False)
    columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
               "semantic_duplicate_group", "basename", "dataset_valid_rate", "info_score"]
    destination = args.output / f"core{args.size}.csv"
    pd.DataFrame([p["rows"][i] for i in selected])[columns].to_csv(destination, index=False)
    report["selection_sha256"] = digest(destination)
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
