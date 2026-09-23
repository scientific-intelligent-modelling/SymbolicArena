import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import select_response_balance as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--min-info", type=float, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "response_distribution_core50")
    args = parser.parse_args()
    if not 1 <= args.size <= 664 or args.seconds <= 0:
        parser.error("规模或预算不合法")
    if not 0 <= args.min_info <= 1:
        parser.error("Info下限必须位于[0,1]")
    if (args.output / "summary.json").exists():
        raise FileExistsError("已有候选结果，须使用独立目录")
    p, r, z, config, normalization = base.load_data(args.size)
    args.output.mkdir(parents=True, exist_ok=True)
    cost, integrality, original_bounds, matrices, lbs, ubs, indices, constraints, eligible = base.build_model(p, z)
    ci, ii, tc, ti = indices
    cost[[tc, ti]] = 0
    lower, upper = original_bounds.lb.copy(), original_bounds.ub.copy()
    lower[[ci, ii, tc, ti]] = 0
    lower[ii] = args.min_info
    upper[[tc, ti]] = 0
    bounds = Bounds(lower, upper)
    manifest = {"method": "mean_standardized_probe_response_w1",
                "objective": "mean_p W1(z_response_subset_p, z_response_full664_p)",
                "score_version": "probe_consensus_response_v1",
                "k": args.size, "seconds_limit": args.seconds, "normalization": normalization,
                "min_mean_info": args.min_info,
                "source_sha256": base.digest(base.SOURCE), "metadata_sha256": p["validation"]["dataset_sha256"],
                "aggregate_sha256": p["validation"]["aggregate_sha256"],
                "selector_sha256": base.digest(Path(__file__)),
                "model_helper_sha256": base.digest(Path(base.__file__)),
                "scoring_helper_sha256": base.digest(Path(__file__).with_name("evaluate_core50.py")),
                "metadata_helper_sha256": base.digest(base.PACKAGE / "lib/engine.py"),
                "objective_scale": base.SCALE, "mip_rel_gap": 0,
                "family_proportional_weight": config["family_proportional_weight"],
                "family_min_margin": config["family_min_margin"], "family_max_margin": config["family_max_margin"],
                "subgroup_max": config["subgroup_max"], "min_dataset_valid_rate": config["min_dataset_valid_rate"],
                "duplicate_cap": 1, "historical_members_used_in_optimization": False,
                "mae_used_in_optimization": False, "formal_ready": False,
                "historical_native_replay_verified": False,
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": base.scipy.__version__}}
    base.write_json(args.output / "configuration.json", manifest)
    matrix = base.vstack(matrices, format="csc")
    low, high = np.concatenate(lbs), np.concatenate(ubs)
    started = time.monotonic()
    result = milp(cost, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(matrix, low, high),
                  options={"time_limit": args.seconds, "mip_rel_gap": 0., "presolve": True})
    seconds = time.monotonic()-started
    if result.x is None:
        base.write_json(args.output / "failure.json", {"status": int(result.status), "message": result.message})
        raise RuntimeError(result.message)
    n = len(p["ids"])
    x = result.x[:n]
    assert np.max(np.abs(x-np.rint(x))) <= 1e-7
    selected = np.flatnonzero(x > .5)
    assert len(selected) == args.size and eligible[selected].all()
    checks = []
    for item in constraints:
        count = len(set(selected) & set(item["members"]))
        assert item["minimum"] <= count <= item["maximum"], item
        checks.append({key: value for key, value in item.items() if key != "members"} | {"actual": count})
    actual = matrix @ result.x
    assert np.all(actual >= low-1e-6) and np.all(actual <= high+1e-6)
    assert np.all(result.x >= lower-1e-6) and np.all(result.x <= upper+1e-6)
    measured = base.terms(p, z, selected)
    assert measured["mean_info"] >= args.min_info-1e-9
    objective = measured["mean_standardized_w1"]
    np.testing.assert_allclose(objective, result.fun/base.SCALE, atol=1e-9, rtol=0)
    bound = float(result.mip_dual_bound)/base.SCALE
    gap = abs(objective-bound)
    probe_results = []
    for j, probe in enumerate(base.PROBES):
        full, subset = float(r[:, j].mean()), float(r[selected, j].mean())
        probe_results.append({"probe": probe, "full_mean_response": full, "selected_mean_response": subset,
                              "signed_error": subset-full, "absolute_error": abs(subset-full)})
    report = {"k": args.size, "method": manifest["method"], "terms": measured, "seconds": seconds,
              "min_mean_info": args.min_info,
              "solver_status": int(result.status), "solver_message": str(result.message),
              "mip_gap": float(result.mip_gap), "objective": objective, "dual_bound": bound,
              "absolute_gap": gap, "certified_within_tolerance": bool(result.status == 0 and gap <= 1e-9),
              "probe_results": probe_results, "mae": float(np.mean([row["absolute_error"] for row in probe_results])),
              "dataset_ids": [p["ids"][i] for i in selected], "constraints_passed": True,
              "formal_ready": False, "loo_executed": False, "historical_native_replay_verified": False,
              "historical_members_used_in_optimization": False, "mae_used_in_optimization": False}
    columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
               "semantic_duplicate_group", "basename", "dataset_valid_rate", "info_score"]
    path = args.output / f"core{args.size}.csv"
    pd.DataFrame([p["rows"][i] for i in selected])[columns].to_csv(path, index=False)
    report["selection_sha256"] = base.digest(path)
    base.write_json(args.output / "constraints.json", checks)
    base.write_json(args.output / "solver_solution.json", {"solution": result.x.tolist(), "indices": selected.tolist()})
    base.write_json(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
