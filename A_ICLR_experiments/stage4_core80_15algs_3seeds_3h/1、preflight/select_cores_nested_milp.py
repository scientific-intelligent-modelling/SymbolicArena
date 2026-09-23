import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csc_matrix, vstack
from scipy.stats import wasserstein_distance

import select_cores_milp as independent


inputs = independent.inputs


def build_nested_problem(frame):
    problems = [independent.build_problem(frame, k) for k in inputs.SIZES]
    widths = [len(problem[0]) for problem in problems]
    offsets = np.cumsum([0] + widths)
    objective = np.concatenate([problem[0] / len(problems) for problem in problems])
    integrality = np.concatenate([problem[1] for problem in problems])
    bounds = Bounds(np.concatenate([problem[2].lb for problem in problems]),
                    np.concatenate([problem[2].ub for problem in problems]))
    diagonal = block_diag([problem[3].A for problem in problems], format="csc")
    n = len(frame)
    count = (len(problems) - 1) * n
    row_indices = np.repeat(np.arange(count), 2)
    columns, coefficients = [], []
    for level in range(len(problems) - 1):
        for task in range(n):
            columns.extend([offsets[level] + task, offsets[level + 1] + task])
            coefficients.extend([1.0, -1.0])
    nesting = csc_matrix((coefficients, (row_indices, columns)), shape=(count, len(objective)))
    constraints = LinearConstraint(vstack([diagonal, nesting], format="csc"),
        np.r_[np.concatenate([problem[3].lb for problem in problems]), np.full(count, -np.inf)],
        np.r_[np.concatenate([problem[3].ub for problem in problems]), np.zeros(count)])
    return objective, integrality, bounds, constraints, offsets, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results_nested_milp")
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("联合求解时间上限必须为正")
    output = args.output_dir
    if (output / "summary.json").exists():
        raise FileExistsError("输出目录已有嵌套试算结果")
    metadata, responses, audit = inputs.load_inputs()
    frame = metadata.sort_values(["difficulty", "dataset_id"], kind="stable").reset_index(drop=True)
    archive = inputs.ROOT / "A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h"
    old_config_path = archive / "experiment_config.json"
    old_config = json.loads(old_config_path.read_text())
    previous = old_config["dataset_selection"]
    reference = archive / Path(previous["reference"]["path"]).relative_to(
        "A_ICLR_experiments/stage4_core80_15algs_3seeds_3h")
    if inputs.sha(reference) != previous["reference"]["sha256"]:
        raise ValueError("旧Core80参考名单哈希不符")
    old_paths = {row["dataset_rel"] for row in previous["datasets"]}
    if len(old_paths) != 80:
        raise ValueError("旧Core80任务数量不符")
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "method": "joint_nested_milp", "objective": "equal_mean_of_six_exact_w1",
        "sizes": list(inputs.SIZES), "weights": [1 / len(inputs.SIZES)] * len(inputs.SIZES),
        "score_version": "probe_consensus_response_v1", "source_sha256": audit["source_sha256"],
        "seconds": args.seconds, "mip_rel_gap": 0, "presolve": True,
        "objective_scale": independent.OBJECTIVE_SCALE,
        "script_sha256": inputs.sha(Path(__file__)),
        "model_helpers_sha256": inputs.sha(Path(independent.__file__)),
        "input_helpers_sha256": inputs.sha(Path(inputs.__file__)),
        "old_core80_configuration_sha256": inputs.sha(old_config_path),
        "old_core80_reference_sha256": inputs.sha(reference),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "solver_calls_planned": 1, "formal_ready": False,
        "historical_native_replay_verified": False, "loo_executed": False,
    }
    inputs.write_json(output / "configuration.json", configuration)
    inputs.write_json(output / "input_audit.json", audit)
    responses.to_csv(output / "responses.csv", index=False)
    metadata.to_csv(output / "difficulty.csv", index=False)
    objective, integrality, bounds, constraints, offsets, problems = build_nested_problem(frame)
    print(json.dumps({"event": "single_joint_solve_started", "variables": len(objective),
                      "constraints": constraints.A.shape[0], "seconds_limit": args.seconds}), flush=True)
    began = time.monotonic()
    result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                  options={"time_limit": args.seconds, "mip_rel_gap": 0.0, "presolve": True})
    seconds = time.monotonic() - began
    fun = independent.finite_number(result.fun)
    dual = independent.finite_number(result.get("mip_dual_bound"))
    report = {
        "solver_calls": 1, "solver_status": int(result.status), "solver_message": str(result.message),
        "seconds": round(seconds, 6), "mip_gap": independent.finite_number(result.get("mip_gap")),
        "mip_node_count": int(result.get("mip_node_count") or 0),
        "objective_mean_w1": fun / independent.OBJECTIVE_SCALE if fun is not None else None,
        "dual_bound_mean_w1": dual / independent.OBJECTIVE_SCALE if dual is not None else None,
        "optimal_within_numerical_tolerance": False,
        "variables": len(objective), "constraints": constraints.A.shape[0],
        "per_size_model_checks": [problem[4] for problem in problems],
    }
    if result.x is None:
        inputs.write_json(output / "joint_solver.json", report)
        raise RuntimeError("本次联合求解未返回可行解，已保存状态")
    primal = constraints.A @ result.x
    if np.any(primal < constraints.lb - 1e-6) or np.any(primal > constraints.ub + 1e-6):
        raise AssertionError("联合线性约束校验失败")
    selections, results, diagnostics = {}, [], []
    for level, k in enumerate(inputs.SIZES):
        membership = result.x[offsets[level]:offsets[level] + len(frame)]
        if np.max(np.abs(membership - np.rint(membership))) > 1e-7:
            raise AssertionError("联合解整数性校验失败")
        chosen = np.flatnonzero(membership > 0.5)
        selected = frame.iloc[chosen].copy()
        if len(selected) != k or not selected.eligible.all():
            raise AssertionError("集合规模或资格校验失败")
        for name in ("semantic_duplicate_group", "basename"):
            if len(np.unique(inputs.group_codes(frame, name)[chosen])) != k:
                raise AssertionError(f"集合去重校验失败: {name}")
        paths = set(selected.dataset_rel)
        selections[k] = paths
        if level and not selections[inputs.SIZES[level - 1]].issubset(paths):
            raise AssertionError("逐级嵌套校验失败")
        w1 = float(wasserstein_distance(selected.difficulty, frame.difficulty))
        cdf_w1, _ = inputs.cdf_distance(frame.difficulty.to_numpy(), chosen)
        np.testing.assert_allclose(w1, cdf_w1, atol=1e-11, rtol=0)
        local_objective = problems[level][0] @ result.x[offsets[level]:offsets[level + 1]]
        np.testing.assert_allclose(w1, local_objective / independent.OBJECTIVE_SCALE, atol=1e-9, rtol=0)
        columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
                   "basename", "semantic_duplicate_group", "difficulty", "dataset_valid_rate"]
        destination = output / f"core{k}.csv"
        selected[columns].to_csv(destination, index=False)
        errors = []
        for probe in inputs.PROBES:
            group = responses[responses.probe == probe]
            full_mean = float(group.response.mean())
            core_mean = float(group[group.dataset_id.isin(selected.dataset_id)].response.mean())
            error = core_mean - full_mean
            errors.append(abs(error))
            diagnostics.append({"k": k, "probe": probe, "full_mean_response": full_mean,
                                "core_mean_response": core_mean, "signed_error": error,
                                "absolute_error": abs(error), "role": "construction_diagnostic"})
        results.append({"k": k, "w1": w1, "construction_mae": float(np.mean(errors)),
                        "old_core80_overlap": len(paths & old_paths), "new_to_old_core80": len(paths - old_paths),
                        "selection_sha256": inputs.sha(destination), "constraints_passed": True})
    combined = set.union(*selections.values())
    common = set.intersection(*selections.values())
    if len(combined) != 80 or len(common) != 30:
        raise AssertionError("嵌套集合的并集或交集数量错误")
    recomputed = float(np.mean([row["w1"] for row in results]))
    np.testing.assert_allclose(recomputed, report["objective_mean_w1"], atol=1e-9, rtol=0)
    gap = abs(recomputed - report["dual_bound_mean_w1"]) if dual is not None else None
    report.update(recomputed_mean_w1=recomputed, absolute_gap_w1=gap,
                  optimal_within_numerical_tolerance=bool(result.status == 0 and result.success
                                                         and gap is not None and gap <= 1e-9),
                  constraints_passed=True, nesting_verified=True, scipy_w1_verified=True,
                  solver_solution=result.x.tolist())
    inputs.write_json(output / "joint_solver.json", report)
    pd.DataFrame(diagnostics).to_csv(output / "construction_diagnostics.csv", index=False)
    summary = {
        "status": "optimal_for_historical_inputs" if report["optimal_within_numerical_tolerance"] else "feasible_candidate",
        "formal_ready": False, "historical_native_replay_verified": False, "loo_executed": False,
        "joint_solver": {key: value for key, value in report.items() if key != "solver_solution"},
        "results": results,
        "overlap": {"identity": "dataset_rel", "union_count": len(combined), "intersection_count": len(common),
                    "old_core80_count": len(old_paths), "union_overlap_old_core80": len(combined & old_paths),
                    "union_new_to_old_core80": len(combined - old_paths),
                    "old_core80_absent_from_union": len(old_paths - combined)},
        "configuration_sha256": inputs.sha(output / "configuration.json"),
    }
    inputs.write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "joint_solver"}), flush=True)
    print(json.dumps({key: value for key, value in report.items() if key not in {"solver_solution", "per_size_model_checks"}}), flush=True)


if __name__ == "__main__":
    main()
