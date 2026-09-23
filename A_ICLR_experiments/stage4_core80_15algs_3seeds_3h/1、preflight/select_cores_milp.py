import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, diags, eye, hstack, vstack
from scipy.stats import wasserstein_distance

import select_cores_hillclimb as inputs


OBJECTIVE_SCALE = 1e6
VERIFY_TOLERANCE = 1e-9


def finite_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def build_problem(frame, k):
    values = frame.difficulty.to_numpy()
    n = len(frame)
    positions = np.flatnonzero(np.diff(values) > 0)
    gaps = np.diff(values)[positions]
    m = len(positions)
    cumulative = csc_matrix((np.arange(n)[None, :] <= positions[:, None]).astype(float))
    auxiliary = eye(m, format="csc")
    target = k * (positions + 1) / n
    matrices = [hstack([cumulative, -auxiliary]), hstack([-cumulative, -auxiliary])]
    upper = [target, -target]
    lower = [np.full(m, -np.inf), np.full(m, -np.inf)]

    # 累计任务数为整数，在相邻整数处连接绝对值，可提高连续松弛的下界。
    floor = np.floor(target)
    fraction = target - floor
    slope = 1 - 2 * fraction
    intercept = fraction - slope * floor
    integer_counts = np.arange(k + 1)[None, :]
    cut_values = slope[:, None] * integer_counts + intercept[:, None]
    exact_values = np.abs(integer_counts - target[:, None])
    if np.any(cut_values > exact_values + 1e-11):
        raise AssertionError("整数CDF加强约束验证失败")
    matrices.append(hstack([diags(slope) @ cumulative, -auxiliary]))
    lower.append(np.full(m, -np.inf))
    upper.append(-intercept)
    matrices.append(csc_matrix(np.r_[np.ones(n), np.zeros(m)].reshape(1, -1)))
    lower.append(np.array([k], dtype=float))
    upper.append(np.array([k], dtype=float))
    duplicate_constraints = {}
    for name in ("semantic_duplicate_group", "basename"):
        codes = inputs.group_codes(frame, name)
        groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
        groups = [group for group in groups if len(group) > 1]
        duplicate_constraints[name] = len(groups)
        if groups:
            row_ids = np.concatenate([np.full(len(group), j) for j, group in enumerate(groups)])
            columns = np.concatenate(groups)
            matrix = csc_matrix((np.ones(len(columns)), (row_ids, columns)), shape=(len(groups), n + m))
            matrices.append(matrix)
            lower.append(np.full(len(groups), -np.inf))
            upper.append(np.ones(len(groups)))
    matrix = vstack(matrices, format="csc")
    objective = np.r_[np.zeros(n), gaps / k * OBJECTIVE_SCALE]
    bounds = Bounds(np.zeros(n + m), np.r_[frame.eligible.to_numpy(dtype=float), np.full(m, k)])
    constraints = LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper))
    integrality = np.r_[np.ones(n, dtype=int), np.zeros(m, dtype=int)]
    evidence = {"binary_variables": n, "continuous_variables": m,
                "constraint_rows": matrix.shape[0], "duplicate_constraints": duplicate_constraints,
                "integer_cdf_cuts_verified": True, "integer_cut_check_count": m * (k + 1)}
    return objective, integrality, bounds, constraints, evidence


def solve(frame, k, seconds):
    objective, integrality, bounds, constraints, evidence = build_problem(frame, k)
    began = time.monotonic()
    result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                  options={"time_limit": seconds, "mip_rel_gap": 0.0, "presolve": True})
    report = {"k": k, "method": "scipy_highs_milp_integer_cdf",
              "solver_status": int(result.status), "solver_message": str(result.message),
              "seconds": round(time.monotonic() - began, 6), "time_limit_seconds": seconds,
              "mip_gap": finite_number(result.get("mip_gap")),
              "mip_node_count": int(result.get("mip_node_count") or 0),
              "objective_scale": OBJECTIVE_SCALE, "verification_tolerance": VERIFY_TOLERANCE,
              "optimal_within_numerical_tolerance": False, **evidence}
    objective_value = finite_number(result.fun)
    dual = finite_number(result.get("mip_dual_bound"))
    report["objective_w1"] = objective_value / OBJECTIVE_SCALE if objective_value is not None else None
    report["dual_bound_w1"] = dual / OBJECTIVE_SCALE if dual is not None else None
    if result.x is None:
        return None, report
    membership = result.x[:len(frame)]
    if np.max(np.abs(membership - np.rint(membership))) > 1e-7:
        raise AssertionError("求解结果不满足整数要求")
    selected_indices = np.flatnonzero(membership > 0.5)
    selected = frame.iloc[selected_indices].copy()
    if len(selected) != k or not selected.eligible.all():
        raise AssertionError("名单规模或资格校验失败")
    for name in ("semantic_duplicate_group", "basename"):
        if len(np.unique(inputs.group_codes(frame, name)[selected_indices])) != k:
            raise AssertionError(f"去重约束失败: {name}")
    values = frame.difficulty.to_numpy()
    w1 = float(wasserstein_distance(values[selected_indices], values))
    cdf_w1, _ = inputs.cdf_distance(values, selected_indices)
    np.testing.assert_allclose(w1, cdf_w1, atol=1e-11, rtol=0)
    np.testing.assert_allclose(w1, report["objective_w1"], atol=VERIFY_TOLERANCE, rtol=0)
    primal = constraints.A @ result.x
    if np.any(primal < constraints.lb - 1e-6) or np.any(primal > constraints.ub + 1e-6):
        raise AssertionError("MILP线性约束校验失败")
    gap = w1 - report["dual_bound_w1"] if report["dual_bound_w1"] is not None else None
    report.update(w1=w1, absolute_gap_w1=gap, constraints_passed=True, scipy_w1_verified=True,
                  optimal_within_numerical_tolerance=bool(
                      result.status == 0 and result.success and gap is not None and abs(gap) <= VERIFY_TOLERANCE),
                  selected_indices=selected_indices.tolist(),
                  solver_solution=result.x.tolist())
    return selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results_milp")
    parser.add_argument("--seconds-per-size", type=float, default=180)
    args = parser.parse_args()
    if args.seconds_per_size <= 0:
        parser.error("每个规模的时间上限必须为正")
    output = args.output_dir
    if (output / "summary.json").exists():
        raise FileExistsError("输出目录已有汇总，请使用独立运行目录")
    metadata, responses, audit = inputs.load_inputs()
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"method": "scipy_highs_milp_integer_cdf", "sizes": list(inputs.SIZES),
                     "score_version": "probe_consensus_response_v1", "probes": list(inputs.PROBES),
                     "source_sha256": audit["source_sha256"], "script_sha256": inputs.sha(Path(__file__)),
                     "input_helpers_sha256": inputs.sha(Path(inputs.__file__)),
                     "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                     "seconds_per_size": args.seconds_per_size, "mip_rel_gap": 0,
                     "objective_scale": OBJECTIVE_SCALE, "verification_tolerance": VERIFY_TOLERANCE,
                     "nested": False, "random_restarts": False, "formal_ready": False,
                     "historical_native_replay_verified": False}
    inputs.write_json(output / "configuration.json", configuration)
    inputs.write_json(output / "input_audit.json", audit)
    responses.to_csv(output / "responses.csv", index=False)
    metadata.to_csv(output / "difficulty.csv", index=False)
    frame = metadata.sort_values(["difficulty", "dataset_id"], kind="stable").reset_index(drop=True)
    reports, diagnostics, selections = [], [], {}
    for k in inputs.SIZES:
        selected, report = solve(frame, k, args.seconds_per_size)
        if selected is not None:
            columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
                       "basename", "semantic_duplicate_group", "difficulty", "dataset_valid_rate"]
            path = output / f"core{k}.csv"
            selected[columns].to_csv(path, index=False)
            report["selection_sha256"] = inputs.sha(path)
            selections[k] = set(selected.dataset_id)
            errors = []
            for probe in inputs.PROBES:
                group = responses[responses.probe == probe]
                full_mean = float(group.response.mean())
                core_mean = float(group[group.dataset_id.isin(selections[k])].response.mean())
                error = core_mean - full_mean
                diagnostics.append({"k": k, "probe": probe, "full_mean_response": full_mean,
                                    "core_mean_response": core_mean, "signed_error": error,
                                    "absolute_error": abs(error), "role": "construction_diagnostic"})
                errors.append(abs(error))
            report["construction_mae"] = float(np.mean(errors))
        inputs.write_json(output / f"solver_core{k}.json", report)
        compact = {key: value for key, value in report.items() if key not in {"solver_solution", "selected_indices"}}
        reports.append(compact)
        inputs.write_json(output / "progress.json", {"completed_sizes": [row["k"] for row in reports], "results": reports})
        print(json.dumps(compact), flush=True)
    overlaps = [{"left": a, "right": b, "intersection": len(selections[a] & selections[b]),
                 "jaccard": len(selections[a] & selections[b]) / len(selections[a] | selections[b])}
                for a, b in zip(inputs.SIZES, inputs.SIZES[1:]) if a in selections and b in selections]
    pd.DataFrame(diagnostics).to_csv(output / "construction_diagnostics.csv", index=False)
    all_optimal = all(row["optimal_within_numerical_tolerance"] for row in reports)
    inputs.write_json(output / "summary.json", {
        "status": "optimal_for_historical_inputs" if all_optimal else "incomplete_optimality",
        "all_sizes_optimal": all_optimal, "formal_ready": False, "loo_executed": False,
        "input_audit": audit, "results": reports, "adjacent_overlaps": overlaps,
        "configuration_sha256": inputs.sha(output / "configuration.json")})
    if not all_optimal:
        raise RuntimeError("部分规模尚未完成最优性证明，已保留求解结果")


if __name__ == "__main__":
    main()
