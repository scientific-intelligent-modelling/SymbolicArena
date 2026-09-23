import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance


PROBES = ("dso", "pyoperon", "imcts", "udsr")
SEEDS = {520, 521, 522}
SIZES = (30, 40, 50, 60, 70, 80)
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "A_ICLR_experiments/stage3_664dats_4probes_3seeds_1h"
EXPECTED_HASHES = {
    "probe4_current_run_level_raw_digest_7968.csv": "1a588501a0b31b5b392592b05c3105489aee18ab19e3cffb190102e6bee5ddcb",
    "probe4_postprocess_dataset_level.csv": "f88a594cd678ef7497f91301675a7c7f3e8169f14e6fd7dc9dbcf640821f7a13",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs():
    hashes = {name: sha(SOURCE / name) for name in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise ValueError("输入文件哈希与本次确认的历史版本不一致")
    raw = pd.read_csv(SOURCE / "probe4_current_run_level_raw_digest_7968.csv")
    metadata = pd.read_csv(SOURCE / "probe4_postprocess_dataset_level.csv")
    if len(raw) != 7968 or len(metadata) != 664 or metadata.dataset_id.nunique() != 664:
        raise ValueError("Full664输入数量不符")
    if raw.duplicated(["dataset_id", "method", "seed"]).any():
        raise ValueError("存在重复运行身份")
    if set(raw.method) != set(PROBES) or set(raw.seed) != SEEDS:
        raise ValueError("probe或seed集合不符")
    if set(raw.dataset_id) != set(metadata.dataset_id):
        raise ValueError("原始结果与数据集元信息不一致")
    if not raw.state.isin(["done", "failed"]).all() or not raw.result_dataset_identity_match.eq(1).all():
        raise ValueError("存在未完成运行或数据集身份错误")
    if not raw.result_valid_output.isin([0, 1]).all():
        raise ValueError("有效性标记缺失")
    metrics = raw[["result_id_test_nmse", "result_ood_test_nmse"]].to_numpy(dtype=float)
    paired = np.isfinite(metrics).all(axis=1) & (metrics >= 0).all(axis=1)
    valid = raw.result_valid_output.eq(1).to_numpy()
    if np.any(valid & ~paired):
        raise ValueError("有效输出标记与原始ID/OOD指标冲突")
    raw = raw.assign(paired_valid=valid & paired)
    rows = []
    for (dataset_id, probe), group in raw.groupby(["dataset_id", "method"], sort=True):
        if len(group) != 3 or set(group.seed) != SEEDS:
            raise ValueError(f"三个seed不完整: {dataset_id}/{probe}")
        accepted = group[group.paired_valid]
        count = len(accepted)
        mean_id = mean_ood = None
        if count:
            logs = np.clip(np.log10(np.maximum(
                accepted[["result_id_test_nmse", "result_ood_test_nmse"]].to_numpy(dtype=float), 1e-12)), -12, 12)
            mean_id, mean_ood = map(float, logs.mean(axis=0))
            response = max(-12.0, -(mean_id + mean_ood) / 2 - 2 * (1 - count / 3))
        else:
            response = -12.0
        rows.append({"dataset_id": dataset_id, "probe": probe, "valid_seed_count": count,
                     "valid_seeds": ";".join(map(str, sorted(accepted.seed))),
                     "valid_rate": count / 3, "mean_log_id_nmse": mean_id,
                     "mean_log_ood_nmse": mean_ood, "response": response})
    responses = pd.DataFrame(rows)
    metadata = metadata.sort_values("dataset_id").reset_index(drop=True)
    matrix = responses.pivot(index="dataset_id", columns="probe", values="response").loc[metadata.dataset_id, list(PROBES)]
    values = matrix.to_numpy(dtype=float)
    means, stds = values.mean(axis=0), values.std(axis=0, ddof=0)
    if not np.isfinite(values).all() or np.any(stds == 0):
        raise ValueError("response存在非有限值或probe标准差为零")
    z = (values - means) / stds
    metadata["difficulty"] = -z.mean(axis=1)
    for column, probe in enumerate(PROBES):
        metadata[f"z_{probe}"] = z[:, column]
    valid_rates = responses.groupby("dataset_id").valid_seed_count.sum() / 12
    metadata["dataset_valid_rate"] = metadata.dataset_id.map(valid_rates)
    metadata["eligible"] = metadata.dataset_valid_rate.ge(0.5)
    audit = {
        "source_sha256": hashes, "runs": len(raw), "dataset_probe_groups": len(responses),
        "valid_runs": int(raw.paired_valid.sum()), "all_invalid_groups": int(responses.valid_seed_count.eq(0).sum()),
        "eligible_count": int(metadata.eligible.sum()),
        "excluded": metadata.loc[~metadata.eligible, ["dataset_id", "dataset_name", "dataset_valid_rate"]].to_dict("records"),
        "historical_state_counts": {str(k): int(v) for k, v in raw.state.value_counts().items()},
        "historical_native_replay_verified": False,
        "status": "historical_input_trial",
        "pending": ["历史算法修复对这些原生预测与指标的影响尚未重新回放核验"],
        "normalization": {probe: {"mean": float(means[j]), "std": float(stds[j])} for j, probe in enumerate(PROBES)},
    }
    return metadata, responses, audit


def group_codes(frame, field):
    keys = [(field, str(value)) if pd.notna(value) and str(value).strip()
            else ("unique", dataset_id) for value, dataset_id in zip(frame[field], frame.dataset_id)]
    mapping = {}
    return np.array([mapping.setdefault(key, len(mapping)) for key in keys], dtype=int)


def cdf_distance(values, selected):
    counts = np.zeros(len(values), dtype=int)
    counts[selected] = 1
    difference = np.cumsum(counts)[:-1] / len(selected) - np.arange(1, len(values)) / len(values)
    return float(np.dot(np.diff(values), np.abs(difference))), difference


def swap_deltas(values, selected, candidates):
    score, difference = cdf_distance(values, selected)
    gaps = np.diff(values)
    # 一次替换只改变两个任务位置之间的CDF，用前缀和计算所有替换的精确W1变化。
    plus = np.concatenate(([0.0], np.cumsum(gaps * (np.abs(difference + 1 / len(selected)) - np.abs(difference)))))
    minus = np.concatenate(([0.0], np.cumsum(gaps * (np.abs(difference - 1 / len(selected)) - np.abs(difference)))))
    deltas = np.where(candidates[None, :] < selected[:, None],
                      plus[selected, None] - plus[candidates][None, :],
                      minus[candidates][None, :] - minus[selected, None])
    return score, deltas


def feasible_swaps(selected, candidates, semantic, basename):
    sem_allowed = (~np.isin(semantic[candidates], semantic[selected]))[None, :] | (semantic[selected, None] == semantic[candidates][None, :])
    base_allowed = (~np.isin(basename[candidates], basename[selected]))[None, :] | (basename[selected, None] == basename[candidates][None, :])
    return sem_allowed & base_allowed


def random_start(k, eligible, semantic, basename, rng):
    for _ in range(100):
        chosen, used_semantic, used_basename = [], set(), set()
        for candidate in rng.permutation(np.flatnonzero(eligible)):
            if semantic[candidate] in used_semantic or basename[candidate] in used_basename:
                continue
            chosen.append(candidate)
            used_semantic.add(semantic[candidate])
            used_basename.add(basename[candidate])
            if len(chosen) == k:
                return np.sort(np.array(chosen, dtype=int))
    raise ValueError(f"随机初始化未能构造Core{k}可行集合")


def verify_deltas(values, selected, eligible, semantic, basename):
    candidates = np.setdiff1d(np.flatnonzero(eligible), selected)
    score, deltas = swap_deltas(values, selected, candidates)
    np.testing.assert_allclose(score, wasserstein_distance(values[selected], values), atol=1e-11, rtol=0)
    pairs = np.argwhere(feasible_swaps(selected, candidates, semantic, basename))[:20]
    for removed, added in pairs:
        trial = selected.copy()
        trial[removed] = candidates[added]
        expected = wasserstein_distance(values[trial], values)
        np.testing.assert_allclose(score + deltas[removed, added], expected, atol=1e-11, rtol=0)
    return len(pairs)


def search(k, frame, restarts, seconds, base_seed):
    values = frame.difficulty.to_numpy()
    eligible = frame.eligible.to_numpy()
    semantic = group_codes(frame, "semantic_duplicate_group")
    basename = group_codes(frame, "basename")
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, k]))
    began = time.monotonic()
    best_score, best_selected, best_restart = float("inf"), None, None
    records = []
    verification_count = 0
    for restart in range(restarts):
        if restart and time.monotonic() - began >= seconds:
            break
        selected = random_start(k, eligible, semantic, basename, rng)
        if restart == 0:
            verification_count = verify_deltas(values, selected, eligible, semantic, basename)
        initial_score, _ = cdf_distance(values, selected)
        moves, local_optimum = 0, False
        while time.monotonic() - began < seconds:
            candidates = np.setdiff1d(np.flatnonzero(eligible), selected)
            current, deltas = swap_deltas(values, selected, candidates)
            deltas[~feasible_swaps(selected, candidates, semantic, basename)] = np.inf
            improvement = float(np.min(deltas)) if deltas.size else float("inf")
            if improvement >= -1e-12:
                local_optimum = True
                break
            ties = np.argwhere(deltas <= improvement + 1e-14)
            removed, added = ties[rng.integers(len(ties))]
            selected[removed] = candidates[added]
            selected.sort()
            updated, _ = cdf_distance(values, selected)
            if updated >= current - 1e-13:
                raise RuntimeError("替换后的真实W1未改善")
            np.testing.assert_allclose(updated, current + deltas[removed, added], atol=1e-11, rtol=0)
            moves += 1
        score, _ = cdf_distance(values, selected)
        records.append({"restart": restart, "initial_w1": initial_score, "final_w1": score,
                        "accepted_swaps": moves, "one_swap_local_optimum": local_optimum})
        if score < best_score:
            best_score, best_selected, best_restart = score, selected.copy(), restart
    if best_selected is None:
        raise RuntimeError("未产生可行候选")
    if len(set(best_selected)) != k or not eligible[best_selected].all():
        raise RuntimeError("最终名单的规模或资格检查失败")
    if len(set(semantic[best_selected])) != k or len(set(basename[best_selected])) != k:
        raise RuntimeError("最终名单重复组检查失败")
    scipy_score = float(wasserstein_distance(values[best_selected], values))
    np.testing.assert_allclose(scipy_score, best_score, atol=1e-11, rtol=0)
    report = {"k": k, "w1": scipy_score, "base_seed": base_seed, "seed_spawn_key": [base_seed, k],
              "restarts_completed": len(records), "best_restart": best_restart,
              "seconds": round(time.monotonic() - began, 6), "time_limit_seconds": seconds,
              "search": "random_restart_steepest_one_swap_hillclimb",
              "global_optimality_proven": False,
              "best_one_swap_local_optimum": records[best_restart]["one_swap_local_optimum"],
              "constraints_passed": True, "scipy_w1_verified": True,
              "real_swap_delta_checks": verification_count, "restarts": records}
    return frame.iloc[best_selected].copy(), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results_hillclimb")
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--seconds-per-size", type=float, default=180)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.restarts < 1 or args.seconds_per_size <= 0:
        parser.error("重启次数和时间上限必须为正")
    output = args.output_dir
    if (output / "summary.json").exists():
        raise FileExistsError("该目录已有候选汇总，请使用独立的运行目录")
    metadata, responses, audit = load_inputs()
    output.mkdir(parents=True, exist_ok=True)
    config = {"score_version": "probe_consensus_response_v1", "probes": list(PROBES),
              "sizes": list(SIZES), "seeds": sorted(SEEDS), "base_seed": args.seed,
              "restarts": args.restarts, "seconds_per_size": args.seconds_per_size,
              "log_base": 10, "nmse_floor": 1e-12, "log_clip": [-12, 12],
              "seed_aggregation": "arithmetic_mean_valid_paired_seeds", "failure_penalty": 2,
              "response_floor": -12, "all_invalid_response": -12, "std_ddof": 0,
              "eligibility_min_valid_rate": 0.5, "improvement_tolerance": 1e-12,
              "source_sha256": audit["source_sha256"], "script_sha256": sha(Path(__file__)),
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
              "formal_ready": False, "historical_native_replay_verified": False}
    write_json(output / "configuration.json", config)
    write_json(output / "input_audit.json", audit)
    responses.to_csv(output / "responses.csv", index=False)
    metadata.to_csv(output / "difficulty.csv", index=False)
    frame = metadata.sort_values(["difficulty", "dataset_id"], kind="stable").reset_index(drop=True)
    summaries, selections, diagnostics = [], {}, []
    for k in SIZES:
        selected, report = search(k, frame, args.restarts, args.seconds_per_size, args.seed)
        columns = ["dataset_id", "dataset_name", "dataset_rel", "family", "subgroup",
                   "basename", "semantic_duplicate_group", "difficulty", "dataset_valid_rate"]
        path = output / f"core{k}.csv"
        selected[columns].to_csv(path, index=False)
        report["selection_sha256"] = sha(path)
        write_json(output / f"search_core{k}.json", report)
        selections[k] = set(selected.dataset_id)
        for probe in PROBES:
            group = responses[responses.probe == probe]
            full_mean = float(group.response.mean())
            subset_mean = float(group[group.dataset_id.isin(selections[k])].response.mean())
            diagnostics.append({"k": k, "probe": probe, "full_mean_response": full_mean,
                                "core_mean_response": subset_mean, "signed_error": subset_mean - full_mean,
                                "absolute_error": abs(subset_mean - full_mean), "role": "construction_diagnostic"})
        summary = {key: value for key, value in report.items() if key != "restarts"}
        summaries.append(summary)
        write_json(output / "progress.json", {"completed_sizes": [row["k"] for row in summaries], "results": summaries})
        print(json.dumps(summary), flush=True)
    overlaps = [{"left": a, "right": b, "intersection": len(selections[a] & selections[b]),
                 "jaccard": len(selections[a] & selections[b]) / len(selections[a] | selections[b])}
                for a, b in zip(SIZES, SIZES[1:])]
    pd.DataFrame(diagnostics).to_csv(output / "construction_diagnostics.csv", index=False)
    write_json(output / "summary.json", {"status": "candidate", "formal_ready": False,
               "input_audit": audit, "results": summaries, "adjacent_overlaps": overlaps,
               "loo_executed": False, "configuration_sha256": sha(output / "configuration.json")})


if __name__ == "__main__":
    main()
