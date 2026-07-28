#!/usr/bin/env python3
"""整理 NeurIPS Stage-4 的 5/10/30/60 分钟六轴结果。"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE4 = REPO_ROOT / "A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h"
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "A_Neurips_experiments/rebuttal/05_stage4_anytime_12algs_5seeds_clean_1h"
)
DEFAULT_PARAMS = (
    REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/"
    "symf_formal_judge_params_20260504/symf_formal_judge_parameters.csv"
)
TARGET_MINUTES = (5, 10, 30, 60)
ALGORITHM_ORDER = (
    "udsr",
    "imcts",
    "pysr",
    "dso",
    "drsr",
    "llmsr",
    "gplearn",
    "qlattice",
    "tpsr",
    "pyoperon",
    "e2esr",
    "ragsr",
)
DISPLAY_NAMES = {
    "udsr": "uDSR",
    "imcts": "iMCTS",
    "pysr": "PySR",
    "dso": "DSO",
    "drsr": "DRSR",
    "llmsr": "LLM-SR",
    "gplearn": "gplearn",
    "qlattice": "QLattice",
    "tpsr": "TPSR",
    "pyoperon": "PyOperon",
    "e2esr": "E2ESR",
    "ragsr": "RAG-SR",
}
METRIC_ORDER = (
    ("ID-Q", "ID-Q"),
    ("OOD-G", "OOD-G"),
    ("SYM-F (proxy)", "SYM-F"),
    ("EFF", "EFF"),
    ("ROBU", "ROBU"),
    ("STAB (proxy)", "STAB"),
)


def normalize_algorithm(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {
        "qlattice_wrapper": "qlattice",
        "imcts_wrapper": "imcts",
    }.get(text, text)


def phi(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 1e2
    if not math.isfinite(number) or number < 0:
        number = 1e2
    clipped = np.clip(np.log10(max(number, 1e-12)), -12.0, 2.0)
    return float(1.0 - (clipped + 12.0) / 14.0)


def log_nmse(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 1e2
    if not math.isfinite(number) or number < 0:
        number = 1e2
    return float(np.clip(np.log10(max(number, 1e-12)), -12.0, 2.0))


def retention(id_nmse: float, ood_nmse: float) -> float:
    gap = max(0.0, math.log10((ood_nmse + 1e-12) / (id_nmse + 1e-12)))
    return 1.0 - float(np.clip(gap / 4.0, 0.0, 1.0))


def result_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def compact_final_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "status",
            "equation",
            "canonical_artifact",
            "id_test",
            "ood_test",
        )
    }


def prepare_manifests(outdir: Path) -> None:
    manifest = pd.read_csv(STAGE4 / "run_archive/manifest.csv")
    target = outdir / "source_manifests"
    target.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    counts: dict[str, int] = {}
    try:
        for _, row in manifest.iterrows():
            payload = json.loads(
                result_path(row["result_json"]).read_text(encoding="utf-8")
            )
            algorithm = normalize_algorithm(payload.get("tool") or row.get("tool"))
            gid_value = payload.get("task_global_index")
            gid = (
                f"g{int(gid_value):04d}"
                if gid_value is not None
                else str(row["task_id"])
            )
            host = str(row.get("host") or "").strip()
            if not host:
                for candidate in range(23, 30):
                    token = f"iaaccn{candidate}"
                    if token in str(payload.get("experiment_dir")):
                        host = token
                        break
            if not host:
                raise ValueError(f"无法确定运行主机: {row['task_id']}")
            if host not in handles:
                handles[host] = (target / f"{host}.jsonl").open("w", encoding="utf-8")
                counts[host] = 0
            record = {
                "algorithm": algorithm,
                "gid": gid,
                "dataset": payload.get("dataset"),
                "seed": int(payload.get("seed")),
                "host": host,
                "experiment_dir": payload.get("experiment_dir"),
                "final_seconds": payload.get("seconds"),
                "final_payload": compact_final_payload(payload),
            }
            handles[host].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[host] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (target / "manifest_summary.json").write_text(
        json.dumps(
            {"total": sum(counts.values()), "by_host": counts},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_remote_rows(remote_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(remote_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"未找到远端快照: {remote_dir}")
    frame["algorithm"] = frame["algorithm"].map(normalize_algorithm)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["minute"] = pd.to_numeric(frame["minute"], errors="raise").astype(int)
    expected = 12 * 50 * 5 * 60
    if len(frame) != expected:
        raise ValueError(f"快照行数应为 {expected}，实际 {len(frame)}")
    duplicate = frame.duplicated(["algorithm", "gid", "seed", "minute"])
    if duplicate.any():
        raise ValueError(f"存在 {int(duplicate.sum())} 条重复快照")
    return frame


def prepare_symf_input(rows: pd.DataFrame, outdir: Path) -> Path:
    target = rows[rows["minute"].isin(TARGET_MINUTES)].copy()
    target["status"] = target["status"].fillna("no_valid_output")
    target["result_path"] = ""
    target["valid_output"] = target["valid_output"].astype(bool)
    target["metric_complete"] = target["metric_complete"].astype(bool)
    columns = [
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "minute",
        "status",
        "valid_output",
        "metric_complete",
        "result_path",
        "expression_canonical",
        "id_test_nmse",
        "ood_test_nmse",
    ]
    path = outdir / "checkpoint_runs_for_symf.csv"
    target[columns].to_csv(path, index=False)
    return path


def load_symf_module():
    path = REPO_ROOT / "check/generate_symf_formal_metrics.py"
    spec = importlib.util.spec_from_file_location("symf_formal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_symf(
    run_csv: Path,
    params_csv: Path,
    output_csv: Path,
    *,
    minute: int | None = None,
    algorithm: str | None = None,
    seed: int | None = None,
) -> None:
    module = load_symf_module()
    runs = pd.read_csv(run_csv)
    params = pd.read_csv(params_csv)
    minutes = (minute,) if minute is not None else TARGET_MINUTES
    parts: list[pd.DataFrame] = []
    for target_minute in minutes:
        subset = runs[runs["minute"] == target_minute].copy()
        if algorithm is not None:
            subset = subset[
                subset["algorithm"].map(normalize_algorithm)
                == normalize_algorithm(algorithm)
            ]
        if seed is not None:
            subset = subset[pd.to_numeric(subset["seed"], errors="coerce") == seed]
        metrics = module.run_formal_metrics(
            params,
            subset,
            prefer_run_level_expression=True,
            require_frozen_formula_source=False,
        )
        metrics["minute"] = target_minute
        parts.append(metrics)
    pd.concat(parts, ignore_index=True).to_csv(output_csv, index=False)


def set_f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def structure_signature(expression: sp.Expr | None) -> str:
    if expression is None:
        return ""

    def visit(node: sp.Expr) -> str:
        if getattr(node, "is_Number", False):
            return "C"
        if getattr(node, "is_Symbol", False):
            return "X"
        name = getattr(getattr(node, "func", None), "__name__", "op").lower()
        children = [visit(child) for child in getattr(node, "args", ())]
        if name in {"add", "mul"}:
            children.sort()
        return f"{name}({','.join(children)})"

    try:
        return visit(expression)
    except Exception:
        return ""


def compute_symf_proxy(
    run_csv: Path,
    params_csv: Path,
    output_csv: Path,
) -> None:
    module = load_symf_module()
    runs = pd.read_csv(run_csv)
    params = pd.read_csv(params_csv)
    params_by_gid = {module.stable_gid(row): row for _, row in params.iterrows()}
    gt_cache: dict[str, tuple[Any, set[str], set[str]]] = {}
    rows: list[dict[str, Any]] = []
    for run in runs.to_dict("records"):
        gid = module.stable_gid(run)
        param = params_by_gid.get(gid)
        if param is None:
            pred_expr = None
            pred_cleaned = ""
            gt_expr = None
            gt_vars: set[str] = set()
            gt_ops: set[str] = set()
        else:
            feature_count = int(param["feature_count"])
            if gid not in gt_cache:
                gt_expr, _, _ = module.parse_sympy(
                    param.get("gt_expression_x"),
                    feature_count=feature_count,
                    feature_to_x_map=None,
                )
                gt_cache[gid] = (
                    gt_expr,
                    module.variable_set(gt_expr),
                    module.operator_set(gt_expr),
                )
            gt_expr, gt_vars, gt_ops = gt_cache[gid]
            feature_map = module.json_loads(param.get("feature_to_x_map"), {})
            pred_expr, _, pred_cleaned = module.parse_sympy(
                run.get("expression_canonical"),
                feature_count=feature_count,
                feature_to_x_map=feature_map,
            )
        pred_vars = module.variable_set(pred_expr)
        pred_ops = module.operator_set(pred_expr)
        var_f1 = set_f1(pred_vars, gt_vars)
        op_f1 = set_f1(pred_ops, gt_ops)
        sof1 = 0.5 * (var_f1 + op_f1)
        tree_proxy = set_jaccard(pred_vars | pred_ops, gt_vars | gt_ops)
        try:
            id_nmse = float(run.get("id_test_nmse"))
            ood_nmse = float(run.get("ood_test_nmse"))
        except (TypeError, ValueError):
            id_nmse = math.inf
            ood_nmse = math.inf
        valid = bool(run.get("valid_output")) and bool(run.get("metric_complete"))
        equivalent_proxy = bool(
            valid
            and pred_expr is not None
            and gt_expr is not None
            and id_nmse <= 1e-10
            and ood_nmse <= 1e-10
        )
        score = 1.0 if equivalent_proxy else 0.3 * tree_proxy + 0.2 * sof1
        if not valid or pred_expr is None or gt_expr is None:
            score = 0.0
        rows.append(
            {
                "algorithm": normalize_algorithm(run.get("algorithm")),
                "gid": gid,
                "dataset": run.get("dataset"),
                "seed": int(run["seed"]),
                "minute": int(run["minute"]),
                "sym_f_formal": float(np.clip(score, 0.0, 1.0)),
                "pred_expression_cleaned": pred_cleaned,
                "pred_structure_signature": structure_signature(pred_expr),
                "equivalent_proxy": equivalent_proxy,
                "tree_similarity_proxy": tree_proxy,
                "var_f1": var_f1,
                "op_f1": op_f1,
                "symf_method": "lightweight_proxy",
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def merge_symf_shards(shard_dir: Path, output_csv: Path) -> None:
    paths = sorted(shard_dir.glob("minute*_*.csv"))
    merged = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected = len(TARGET_MINUTES) * len(ALGORITHM_ORDER) * 50 * 5
    if len(merged) != expected:
        raise ValueError(f"SYM-F 行数应为 {expected}，实际 {len(merged)}")
    keys = ["algorithm", "gid", "dataset", "seed", "minute"]
    if merged.duplicated(keys).any():
        raise ValueError("SYM-F shard 合并后存在重复运行")
    merged.to_csv(output_csv, index=False)


def pairwise_structure_consistency(group: pd.DataFrame) -> float:
    signatures = list(group.sort_values("seed")["pred_structure_signature"].fillna(""))
    if len(signatures) < 2:
        return 0.0
    values: list[float] = []
    for left, right in combinations(signatures, 2):
        values.append(float(bool(left) and left == right))
    return float(np.mean(values)) if values else 0.0


def compute_tables(
    rows: pd.DataFrame,
    symf: pd.DataFrame,
    outdir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = rows.copy()
    valid = work["valid_output"].astype(bool) & work["metric_complete"].astype(bool)
    work["id_used"] = np.where(valid, work["id_test_nmse"], 1e2).astype(float)
    work["ood_used"] = np.where(valid, work["ood_test_nmse"], 1e2).astype(float)
    work["q_id_run"] = work["id_used"].map(phi)
    work["q_ood_run"] = work["ood_used"].map(phi)
    work["retention_run"] = [
        retention(i, o) for i, o in zip(work["id_used"], work["ood_used"])
    ]
    work["q_ood_g_run"] = 0.7 * work["q_ood_run"] + 0.3 * work["retention_run"]
    work["q_time"] = 0.5 * work["q_id_run"] + 0.5 * work["q_ood_run"]
    work["auc_to_checkpoint"] = (
        work.sort_values("minute")
        .groupby(["algorithm", "gid", "dataset", "seed"], dropna=False)["q_time"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1, 2, 3], drop=True)
        .sort_index()
    )

    symf = symf.copy()
    symf["algorithm"] = symf["algorithm"].map(normalize_algorithm)
    target = work[work["minute"].isin(TARGET_MINUTES)].merge(
        symf[
            [
                "algorithm",
                "gid",
                "dataset",
                "seed",
                "minute",
                "sym_f_formal",
                "pred_expression_cleaned",
                "pred_structure_signature",
            ]
        ],
        on=["algorithm", "gid", "dataset", "seed", "minute"],
        how="left",
        validate="one_to_one",
    )
    target["sym_f_formal"] = target["sym_f_formal"].fillna(0.0)

    protocol_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for minute in TARGET_MINUTES:
        at_time = target[target["minute"] == minute]
        for (algorithm, gid, dataset), group in at_time.groupby(
            ["algorithm", "gid", "dataset"], dropna=False
        ):
            med_id = float(np.median(group["id_used"]))
            med_ood = float(np.median(group["ood_used"]))
            q_id_protocol = phi(med_id)
            q_ood_protocol = phi(med_ood)
            q_ood_g_protocol = 0.7 * q_ood_protocol + 0.3 * retention(med_id, med_ood)
            q_sym = float(group["sym_f_formal"].mean())
            id_iqr = float(
                np.percentile(group["id_used"].map(log_nmse), 75)
                - np.percentile(group["id_used"].map(log_nmse), 25)
            )
            ood_iqr = float(
                np.percentile(group["ood_used"].map(log_nmse), 75)
                - np.percentile(group["ood_used"].map(log_nmse), 25)
            )
            numeric_stability = 0.5 * (
                1.0 - float(np.clip(id_iqr / 3.0, 0.0, 1.0))
            ) + 0.5 * (1.0 - float(np.clip(ood_iqr / 3.0, 0.0, 1.0)))
            valid_rate = float(
                (
                    group["valid_output"].astype(bool)
                    & group["metric_complete"].astype(bool)
                ).mean()
            )
            structure = pairwise_structure_consistency(group)
            pure_stab = 0.4 * numeric_stability + 0.3 * valid_rate + 0.3 * structure
            q_perf = 0.4 * q_id_protocol + 0.3 * q_ood_g_protocol + 0.3 * q_sym
            stab = pure_stab * math.sqrt(max(0.0, q_perf))
            dataset_rows.append(
                {
                    "algorithm": algorithm,
                    "gid": gid,
                    "dataset": dataset,
                    "minute": minute,
                    "seeds": int(group["seed"].nunique()),
                    "ID_Q_protocol": q_id_protocol,
                    "OOD_G_protocol": q_ood_g_protocol,
                    "SYM_F": q_sym,
                    "EFF_protocol": float(group["auc_to_checkpoint"].median()),
                    "STAB": stab,
                    "ID_Q_seed_mean": float(group["q_id_run"].mean()),
                    "OOD_G_seed_mean": float(group["q_ood_g_run"].mean()),
                    "EFF_seed_mean": float(group["auc_to_checkpoint"].mean()),
                    "valid_seed_rate": valid_rate,
                    "numeric_stability": numeric_stability,
                    "structural_consistency": structure,
                }
            )

    dataset_frame = pd.DataFrame(dataset_rows)
    robu = pd.read_csv(STAGE4 / "paper_tables/table13_noise_robustness_summary.csv")
    robu["algorithm"] = robu["Algorithm"].map(normalize_algorithm)
    robu_map = dict(
        zip(robu["algorithm"], pd.to_numeric(robu["ROBU"], errors="coerce"))
    )
    for algorithm in ALGORITHM_ORDER:
        alg = dataset_frame[dataset_frame["algorithm"] == algorithm]
        for minute in TARGET_MINUTES:
            subset = alg[alg["minute"] == minute]
            rob_value = (
                float(robu_map.get(algorithm, math.nan)) if minute == 60 else math.nan
            )
            protocol_rows.append(
                {
                    "algorithm": algorithm,
                    "minute": minute,
                    "ID-Q": 100 * float(subset["ID_Q_protocol"].mean()),
                    "OOD-G": 100 * float(subset["OOD_G_protocol"].mean()),
                    "SYM-F": 100 * float(subset["SYM_F"].mean()),
                    "EFF": 100 * float(subset["EFF_protocol"].mean()),
                    "ROBU": rob_value,
                    "STAB": 100 * float(subset["STAB"].mean()),
                }
            )
            mean_rows.append(
                {
                    "algorithm": algorithm,
                    "minute": minute,
                    "ID-Q": 100 * float(subset["ID_Q_seed_mean"].mean()),
                    "OOD-G": 100 * float(subset["OOD_G_seed_mean"].mean()),
                    "SYM-F": 100 * float(subset["SYM_F"].mean()),
                    "EFF": 100 * float(subset["EFF_seed_mean"].mean()),
                    "ROBU": rob_value,
                    "STAB": 100 * float(subset["STAB"].mean()),
                }
            )
    return pd.DataFrame(protocol_rows), pd.DataFrame(mean_rows), dataset_frame


def image_style_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHM_ORDER:
        subset = scores[scores["algorithm"] == algorithm].set_index("minute")
        for display_metric, score_column in METRIC_ORDER:
            row: dict[str, Any] = {
                "算法": DISPLAY_NAMES[algorithm],
                "指标": display_metric,
            }
            for minute in TARGET_MINUTES:
                value = subset.loc[minute, score_column]
                row[f"{minute}min"] = "NA" if pd.isna(value) else f"{float(value):.2f}"
            rows.append(row)
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Algorithm & Metric & 5min & 10min & 30min & 60min \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        values = [str(row[f"{minute}min"]) for minute in TARGET_MINUTES]
        lines.append(f"{row['算法']} & {row['指标']} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    rows: pd.DataFrame,
    protocol: pd.DataFrame,
    seed_mean: pd.DataFrame,
    dataset_frame: pd.DataFrame,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        outdir / "clean_minute_run_level.csv.gz", index=False, compression="gzip"
    )
    dataset_frame.to_csv(outdir / "dataset_axis_components.csv", index=False)
    protocol.to_csv(outdir / "checkpoint_scores_protocol.csv", index=False)
    seed_mean.to_csv(outdir / "checkpoint_scores_seed_mean.csv", index=False)
    protocol_table = image_style_table(protocol)
    mean_table = image_style_table(seed_mean)
    protocol_table.to_csv(outdir / "table_protocol.csv", index=False)
    mean_table.to_csv(outdir / "table_seed_mean.csv", index=False)
    (outdir / "table_protocol.md").write_text(
        protocol_table.to_markdown(index=False, disable_numparse=True) + "\n",
        encoding="utf-8",
    )
    (outdir / "table_seed_mean.md").write_text(
        mean_table.to_markdown(index=False, disable_numparse=True) + "\n",
        encoding="utf-8",
    )
    write_latex(protocol_table, outdir / "table_protocol.tex")
    write_latex(mean_table, outdir / "table_seed_mean.tex")

    source_counts = (
        rows.groupby(["algorithm", "minute", "source"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    source_counts.to_csv(outdir / "snapshot_source_audit.csv", index=False)
    coverage = (
        rows[rows["minute"].isin(TARGET_MINUTES)]
        .groupby(["algorithm", "minute"], dropna=False)
        .agg(
            runs=("seed", "size"),
            datasets=("gid", "nunique"),
            seeds=("seed", "nunique"),
            valid_runs=("valid_output", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(outdir / "checkpoint_coverage.csv", index=False)
    formal_reference = pd.read_csv(
        STAGE4 / "formal_analysis/symbolic_metrics_formal_algorithm_summary.csv"
    )
    formal_reference.to_csv(
        outdir / "formal_60min_symf_reference.csv",
        index=False,
    )
    summary = {
        "clean_run_minute_rows": int(len(rows)),
        "clean_runs": int(
            rows[["algorithm", "gid", "dataset", "seed"]].drop_duplicates().shape[0]
        ),
        "algorithms": int(rows["algorithm"].nunique()),
        "datasets": int(rows["gid"].nunique()),
        "seeds": sorted(int(value) for value in rows["seed"].unique()),
        "target_minutes": list(TARGET_MINUTES),
        "checkpoint_valid_rates": {
            str(int(row.minute)): float(row.valid_runs / row.runs)
            for row in coverage.groupby("minute", as_index=False)[
                ["runs", "valid_runs"]
            ]
            .sum()
            .itertuples()
        },
        "seed_mean_policy": "mean run-level scores across five seeds, then mean across datasets",
        "protocol_policy": (
            "paper-defined seed median for ID-Q/OOD-G/EFF; "
            "seed mean for lightweight SYM-F proxy"
        ),
        "robu_policy": "NA at 5/10/30; formal final noisy-track score at 60",
        "future_backfill_used": False,
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# NeurIPS Stage-4 anytime 六轴整理

- 数据：NeurIPS Stage 4 clean，`12 algorithms × 50 datasets × 5 seeds`。
- 时间点：5、10、30、60 分钟；每分钟使用当时可用的 best-so-far。
- `table_seed_mean.*`：按用户要求，先在种子层面取均值，再对 50 个任务取均值。
- `table_protocol.*`：论文聚合规则对照；ID-Q/OOD-G/EFF 在任务内对种子取
  中位数，SYM-F proxy 对种子取均值，STAB proxy 由五种子共同定义。
- 时间点 SYM-F 使用轻量 proxy：`ID/OOD NMSE <= 1e-10` 的有效公式记为近等价，
  否则使用变量/算子 F1 与变量-算子集合 Jaccard。它用于搜索动态诊断，
  不能替代 Stage 4 已归档的 60 分钟 formal CAS+dense-probe+NED SYM-F。
- 时间点 STAB 使用数值 IQR、有效 seed rate、结构签名一致性和性能校正，
  因结构一致性是轻量签名比较，同样标为 proxy。
- `ROBU`：5/10/30 分钟为 `NA`。Stage 4 归档没有 noisy minute snapshots；
  60 分钟直接引用完整 noisy track 的正式 ROBU，不从 clean 结果估计。
- 失败或当时尚无可评估候选的 run 按协议记 `NMSE=1e2`、质量为 0。
- `final_carry_forward` 仅用于在该时间点之前真实完成的运行，不使用未来快照。

## 产物

- `table_seed_mean.md/csv/tex`：与示意图同结构的主表。
- `table_protocol.md/csv/tex`：论文种子聚合规则对照表；符号和结构部分为
  明确标注的 proxy。
- `checkpoint_scores_*.csv`：算法 × 时间点宽指标。
- `checkpoint_symbolic_metrics_proxy.csv`：时间点 SYM-F proxy 的运行级明细。
- `formal_60min_symf_reference.csv`：Stage 4 已归档的正式 60 分钟
  CAS+dense-probe+NED SYM-F 对照，不与时间点 proxy 混算。
- `dataset_axis_components.csv`：算法 × 任务 × 时间点分项。
- `clean_minute_run_level.csv.gz`：180,000 条 run-minute 记录。
- `checkpoint_coverage.csv`、`snapshot_source_audit.csv`：覆盖率与来源审计。

## 本地中间文件

- `source_manifests/`、`remote_extracts/`：远端只读提取的输入和压缩回传。
- `checkpoint_runs_for_symf.csv`：SYM-F 后处理输入。
- `symf_shards/`：曾尝试的 formal CAS/NED 分片；因复杂表达式成本过高而
  中止，分片不完整且不参与本目录任何汇总。正式表只读取
  `checkpoint_symbolic_metrics_proxy.csv`。

## 复算

先用 `extract_neurips_stage4_anytime_remote.py` 在各 manifest 指定主机上
生成 `remote_extracts/*.jsonl.gz`，再从仓库根目录运行：

```bash
python check/analyze_neurips_stage4_anytime.py prepare-symf
python check/analyze_neurips_stage4_anytime.py compute-symf-proxy
python check/analyze_neurips_stage4_anytime.py aggregate
pytest -q tests/test_neurips_stage4_anytime.py
```
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare-manifests",
            "prepare-symf",
            "compute-symf",
            "compute-symf-proxy",
            "merge-symf",
            "aggregate",
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--remote-dir", type=Path)
    parser.add_argument("--params-csv", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--minute", type=int, choices=TARGET_MINUTES)
    parser.add_argument("--algorithm", choices=ALGORITHM_ORDER)
    parser.add_argument("--seed", type=int, choices=range(5))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    remote_dir = args.remote_dir or (args.outdir / "remote_extracts")
    if args.command == "prepare-manifests":
        prepare_manifests(args.outdir)
        return
    rows = load_remote_rows(remote_dir)
    if args.command == "prepare-symf":
        prepare_symf_input(rows, args.outdir)
        return
    symf_input = args.outdir / "checkpoint_runs_for_symf.csv"
    symf_output = args.outdir / "checkpoint_symbolic_metrics_formal.csv"
    symf_proxy_output = args.outdir / "checkpoint_symbolic_metrics_proxy.csv"
    if args.command == "compute-symf-proxy":
        compute_symf_proxy(symf_input, args.params_csv, symf_proxy_output)
        return
    if args.command == "compute-symf":
        if (args.minute is None) != (args.algorithm is None):
            raise ValueError("--minute 与 --algorithm 必须同时提供")
        if args.seed is not None and args.algorithm is None:
            raise ValueError("--seed 必须与 --minute/--algorithm 同时提供")
        if args.minute is None:
            output = symf_output
        else:
            shard_dir = args.outdir / "symf_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if args.seed is None else f"_seed{args.seed}"
            output = shard_dir / f"minute{args.minute:02d}_{args.algorithm}{suffix}.csv"
        compute_symf(
            symf_input,
            args.params_csv,
            output,
            minute=args.minute,
            algorithm=args.algorithm,
            seed=args.seed,
        )
        return
    if args.command == "merge-symf":
        merge_symf_shards(args.outdir / "symf_shards", symf_output)
        return
    symf = pd.read_csv(symf_proxy_output)
    protocol, seed_mean, dataset_frame = compute_tables(rows, symf, args.outdir)
    write_outputs(rows, protocol, seed_mean, dataset_frame, args.outdir)


if __name__ == "__main__":
    main()
