#!/usr/bin/env python3
"""生成 Core-50 六轴指标 v0 图表。

当前本地 clean Core50 汇总具备完整 final-run NMSE 和 result.json，
但尚未本地收齐 minute snapshots、formal symbolic judge 与 noise track。
因此本脚本明确区分：

- formal: ID-Q / OOD-G，严格按固定 NMSE 映射计算。
- proxy: SYM-F / EFF / STAB，用现有表达式与 final runtime 构造可追溯占位分。
- pending: ROB，等待噪声实验汇总后再计算。

输出目录默认：
exp-planning/04.Core50正式全量评测/analysis/hexagon_v0_20260504
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE50_ROOT = REPO_ROOT / "exp-planning/04.Core50正式全量评测"
DEFAULT_RUNS_CSV = (
    CORE50_ROOT
    / "generated/core50_12alg_5seed_final_results/analysis/core50_12alg_run_level_log_nmse_20260503.csv"
)
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments/core50_12alg_5seed_all_20260502-065700"
DEFAULT_CORE50_CSV = CORE50_ROOT / "core50_datasets.csv"
DEFAULT_OUTDIR = CORE50_ROOT / "analysis/hexagon_v0_20260504"


AXES = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
DISPLAY_AXIS_NAMES = {"ID_Q": "ID-Q", "OOD_G": "OOD-G", "SYM_F": "SYM-F"}


def phi_from_nmse(value: Any) -> float:
    """固定 NMSE -> quality 映射，失败值由调用方传入 1e2。"""
    try:
        x = float(value)
    except Exception:
        x = 1e2
    if not math.isfinite(x) or x < 0:
        x = 1e2
    log_value = np.clip(np.log10(max(x, 1e-12)), -12, 2)
    return float(1.0 - (log_value + 12.0) / 14.0)


def clipped_log_nmse(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 1e2
    if not math.isfinite(x) or x < 0:
        x = 1e2
    return float(np.clip(np.log10(max(x, 1e-12)), -12, 2))


def safe_nmse(row: pd.Series, column: str) -> float:
    metric_complete = bool(row.get("metric_complete", False))
    valid_output = bool(row.get("valid_output", False))
    if not metric_complete or not valid_output:
        return 1e2
    value = row.get(column)
    try:
        value = float(value)
    except Exception:
        return 1e2
    if not math.isfinite(value) or value < 0:
        return 1e2
    return value


def f1_score(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sympy_locals() -> dict[str, Any]:
    if sp is None:
        return {}
    out: dict[str, Any] = {}
    for i in range(128):
        out[f"x{i}"] = sp.Symbol(f"x{i}")
        out[f"c{i}"] = sp.Symbol(f"c{i}")
    out.update(
        {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "Abs": sp.Abs,
            "abs": sp.Abs,
            "pow": lambda x, y: x**y,
            "div": lambda x, y: x / y,
            "pi": sp.pi,
        }
    )
    return out


def sanitize_expr(expr: str) -> str:
    text = str(expr or "").strip()
    text = text.replace("np.", "").replace("numpy.", "").replace("math.", "")
    text = re.sub(r"\bx_(\d+)\b", lambda m: f"x{m.group(1)}", text)
    text = re.sub(r"\bx\[(\d+)\]", lambda m: f"x{m.group(1)}", text)
    text = text.replace("^", "**")
    return text


def parse_expr(expr: str):
    if sp is None or not expr:
        return None
    try:
        return sp.sympify(sanitize_expr(expr), locals=sympy_locals())
    except Exception:
        return None


def operator_set(expr) -> set[str]:
    if sp is None or expr is None:
        return set()
    ops: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if getattr(node, "is_Symbol", False) or getattr(node, "is_Number", False):
            continue
        name = getattr(getattr(node, "func", None), "__name__", "")
        if name:
            ops.add(name.lower())
    return ops


def variable_set(expr) -> set[str]:
    if sp is None or expr is None:
        return set()
    return {str(s) for s in expr.free_symbols}


def skeleton(expr) -> str | None:
    """生成粗粒度结构骨架，用于 STAB proxy 的 seed 间结构一致性。"""
    if sp is None or expr is None:
        return None

    def rec(node) -> str:
        if getattr(node, "is_Number", False):
            return "C"
        if getattr(node, "is_Symbol", False):
            return "X"
        name = getattr(getattr(node, "func", None), "__name__", "op").lower()
        children = [rec(arg) for arg in getattr(node, "args", ())]
        if name in {"add", "mul"}:
            children = sorted(children)
        return f"{name}({','.join(children)})"

    try:
        return rec(expr)
    except Exception:
        return None


def extract_formula_return(formula_path: Path, target_name: str | None) -> str | None:
    if not formula_path.exists():
        return None
    try:
        tree = ast.parse(formula_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            candidates.append(node)
    ordered = sorted(
        candidates,
        key=lambda n: 0 if target_name and n.name == target_name else (1 if n.name in {"target", "y"} else 2),
    )
    for fn in ordered:
        for stmt in fn.body:
            if isinstance(stmt, ast.Return):
                try:
                    return ast.unparse(stmt.value)
                except Exception:
                    return None
    return None


def load_core50_metadata(core50_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    df = pd.read_csv(core50_csv)
    for _, row in df.iterrows():
        dataset = row.get("dataset_id") or row.get("dataset") or row.get("dataset_name")
        dataset_dir_raw = row.get("dataset_dir")
        dataset_dir = Path(str(dataset_dir_raw))
        if not dataset_dir.is_absolute():
            dataset_dir = REPO_ROOT / dataset_dir
        metadata_path = dataset_dir / "metadata.yaml"
        target_name = None
        gt_expr = None
        if metadata_path.exists():
            try:
                meta = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                target_name = (((meta.get("dataset") or {}).get("target") or {}).get("name"))
                formula_file = (((meta.get("dataset") or {}).get("ground_truth_formula") or {}).get("file")) or "formula.py"
                gt_expr = extract_formula_return(dataset_dir / formula_file, target_name)
            except Exception:
                pass
        parsed = parse_expr(gt_expr or "")
        rows.append(
            {
                "gid": row.get("gid"),
                "dataset": dataset,
                "dataset_dir": str(dataset_dir),
                "target_name": target_name,
                "ground_truth_expression": gt_expr,
                "gt_parse_ok": parsed is not None,
                "gt_variables": sorted(variable_set(parsed)),
                "gt_operators": sorted(operator_set(parsed)),
                "gt_skeleton": skeleton(parsed),
            }
        )
    return pd.DataFrame(rows)


def iter_result_jsons(experiment_root: Path) -> list[Path]:
    return sorted(experiment_root.glob("**/result.json"))


def extract_result_records(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_result_jsons(experiment_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact = payload.get("canonical_artifact") or {}
        expr = (
            artifact.get("normalized_expression")
            or artifact.get("instantiated_expression")
            or artifact.get("return_expression_source")
            or payload.get("equation")
        )
        parsed = parse_expr(expr or "")
        rows.append(
            {
                "algorithm": payload.get("tool"),
                "gid": f"g{int(payload.get('task_global_index')):04d}" if str(payload.get("task_global_index", "")).isdigit() else None,
                "dataset": payload.get("dataset"),
                "seed": payload.get("seed"),
                "status_json": payload.get("status"),
                "result_path": str(path),
                "experiment_dir": payload.get("experiment_dir"),
                "expression_raw": payload.get("equation"),
                "expression_canonical": expr,
                "expression_parse_ok": parsed is not None,
                "pred_variables": sorted(variable_set(parsed)),
                "pred_operators": sorted(operator_set(parsed)),
                "pred_skeleton": skeleton(parsed),
            }
        )
    return pd.DataFrame(rows)


def compute_run_and_dataset_metrics(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = runs.copy()
    df["id_nmse_used"] = df.apply(lambda r: safe_nmse(r, "id_test_nmse"), axis=1)
    df["ood_nmse_used"] = df.apply(lambda r: safe_nmse(r, "ood_test_nmse"), axis=1)
    df["id_log_used"] = df["id_nmse_used"].map(clipped_log_nmse)
    df["ood_log_used"] = df["ood_nmse_used"].map(clipped_log_nmse)
    df["id_quality_run"] = df["id_nmse_used"].map(phi_from_nmse)
    df["ood_quality_run"] = df["ood_nmse_used"].map(phi_from_nmse)
    df["valid_for_metric"] = df["valid_output"].astype(bool) & df["metric_complete"].astype(bool)

    grouped_rows: list[dict[str, Any]] = []
    for (alg, gid, dataset), g in df.groupby(["algorithm", "gid", "dataset"], dropna=False):
        med_id = float(np.median(g["id_nmse_used"]))
        med_ood = float(np.median(g["ood_nmse_used"]))
        q_id = phi_from_nmse(med_id)
        q_ood = phi_from_nmse(med_ood)
        delta = max(0.0, math.log10((med_ood + 1e-12) / (med_id + 1e-12)))
        ood_retention = 1.0 - float(np.clip(delta / 4.0, 0.0, 1.0))
        q_ood_g = 0.7 * q_ood + 0.3 * ood_retention
        final_quality = 0.5 * q_id + 0.5 * q_ood
        seconds = pd.to_numeric(g["seconds"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med_seconds = float(seconds.median()) if seconds.notna().any() else 3600.0
        time_bonus = 1.0 - float(np.clip(med_seconds / 3600.0, 0.0, 1.0))
        # proxy：没有 minute AUC 时，用 final quality 和完成时间给一个保守效率占位。
        eff_proxy = final_quality * (0.35 + 0.65 * time_bonus)
        id_iqr = float(np.percentile(g["id_log_used"], 75) - np.percentile(g["id_log_used"], 25))
        ood_iqr = float(np.percentile(g["ood_log_used"], 75) - np.percentile(g["ood_log_used"], 25))
        u_id = 1.0 - float(np.clip(id_iqr / 3.0, 0.0, 1.0))
        u_ood = 1.0 - float(np.clip(ood_iqr / 3.0, 0.0, 1.0))
        numeric_stability = 0.5 * u_id + 0.5 * u_ood
        valid_rate = float(g["valid_for_metric"].mean())
        grouped_rows.append(
            {
                "algorithm": alg,
                "gid": gid,
                "dataset": dataset,
                "n_runs": int(len(g)),
                "valid_rate": valid_rate,
                "median_id_nmse": med_id,
                "median_ood_nmse": med_ood,
                "q_id": q_id,
                "q_ood": q_ood,
                "ood_retention": ood_retention,
                "q_ood_g": q_ood_g,
                "final_quality": final_quality,
                "median_seconds": med_seconds,
                "eff_proxy": eff_proxy,
                "id_log_iqr": id_iqr,
                "ood_log_iqr": ood_iqr,
                "numeric_stability": numeric_stability,
            }
        )
    return df, pd.DataFrame(grouped_rows)


def compute_symbolic_proxy(run_df: pd.DataFrame, result_df: pd.DataFrame, core_meta: pd.DataFrame) -> pd.DataFrame:
    keys = ["algorithm", "gid", "dataset", "seed"]
    merged = run_df.merge(result_df, on=keys, how="left").merge(core_meta, on=["gid", "dataset"], how="left")

    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        pred_vars = set(r.get("pred_variables") if isinstance(r.get("pred_variables"), list) else [])
        pred_ops = set(r.get("pred_operators") if isinstance(r.get("pred_operators"), list) else [])
        gt_vars = set(r.get("gt_variables") if isinstance(r.get("gt_variables"), list) else [])
        gt_ops = set(r.get("gt_operators") if isinstance(r.get("gt_operators"), list) else [])
        parse_ok = bool(r.get("expression_parse_ok", False))
        numeric_equiv_proxy = bool(
            parse_ok
            and safe_nmse(r, "id_test_nmse") <= 1e-10
            and safe_nmse(r, "ood_test_nmse") <= 1e-10
        )
        var_f1 = f1_score(pred_vars, gt_vars) if parse_ok else 0.0
        op_f1 = f1_score(pred_ops, gt_ops) if parse_ok else 0.0
        sof1 = 0.5 * var_f1 + 0.5 * op_f1
        tree_similarity_proxy = jaccard(pred_ops | pred_vars, gt_ops | gt_vars) if parse_ok else 0.0
        sym_score = 1.0 if numeric_equiv_proxy else (0.3 * tree_similarity_proxy + 0.2 * sof1)
        rows.append(
            {
                "algorithm": r.get("algorithm"),
                "gid": r.get("gid"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "expression_parse_ok": parse_ok,
                "numeric_equiv_proxy": numeric_equiv_proxy,
                "tree_similarity_proxy": tree_similarity_proxy,
                "var_f1": var_f1,
                "op_f1": op_f1,
                "sof1": sof1,
                "sym_f_proxy_run": float(np.clip(sym_score, 0.0, 1.0)),
                "pred_skeleton": r.get("pred_skeleton"),
            }
        )
    return pd.DataFrame(rows)


def structural_consistency(group: pd.DataFrame) -> float:
    skeletons = list(group.sort_values("seed")["pred_skeleton"])
    if len(skeletons) < 2:
        return 0.0
    total = 0
    ok = 0
    for a, b in combinations(skeletons, 2):
        total += 1
        if isinstance(a, str) and isinstance(b, str) and a and b and a == b:
            ok += 1
    return ok / total if total else 0.0


def compute_algorithm_scores(dataset_df: pd.DataFrame, symbolic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym_dataset = (
        symbolic_df.groupby(["algorithm", "gid", "dataset"], dropna=False)
        .agg(
            q_sym_proxy=("sym_f_proxy_run", "mean"),
            parse_rate=("expression_parse_ok", "mean"),
            numeric_equiv_proxy_rate=("numeric_equiv_proxy", "mean"),
        )
        .reset_index()
    )
    consistency = (
        symbolic_df.groupby(["algorithm", "gid", "dataset"], dropna=False)
        .apply(structural_consistency)
        .reset_index(name="structural_consistency_proxy")
    )
    full = dataset_df.merge(sym_dataset, on=["algorithm", "gid", "dataset"], how="left").merge(
        consistency, on=["algorithm", "gid", "dataset"], how="left"
    )
    full["q_sym_proxy"] = full["q_sym_proxy"].fillna(0.0)
    full["structural_consistency_proxy"] = full["structural_consistency_proxy"].fillna(0.0)
    full["q_perf_proxy"] = 0.4 * full["q_id"] + 0.3 * full["q_ood_g"] + 0.3 * full["q_sym_proxy"]
    full["pure_stab_proxy"] = (
        0.4 * full["numeric_stability"]
        + 0.3 * full["valid_rate"]
        + 0.3 * full["structural_consistency_proxy"]
    )
    full["stab_proxy"] = full["pure_stab_proxy"] * np.sqrt(full["q_perf_proxy"].clip(lower=0.0))

    alg = (
        full.groupby("algorithm", dropna=False)
        .agg(
            datasets=("dataset", "nunique"),
            ID_Q=("q_id", lambda x: 100 * float(np.mean(x))),
            OOD_G=("q_ood_g", lambda x: 100 * float(np.mean(x))),
            SYM_F=("q_sym_proxy", lambda x: 100 * float(np.mean(x))),
            EFF=("eff_proxy", lambda x: 100 * float(np.mean(x))),
            STAB=("stab_proxy", lambda x: 100 * float(np.mean(x))),
            valid_rate=("valid_rate", "mean"),
            median_seconds=("median_seconds", "median"),
            parse_rate=("parse_rate", "mean"),
            numeric_equiv_proxy_rate=("numeric_equiv_proxy_rate", "mean"),
        )
        .reset_index()
    )
    alg["ROB"] = np.nan
    for col in AXES:
        alg[col] = alg[col].clip(lower=0, upper=100)
    # ROB 缺失时，HexaScore 只作为 clean/proxy 五轴几何平均。
    clean_proxy_axes = ["ID_Q", "OOD_G", "SYM_F", "EFF", "STAB"]
    alg["HexaScore_clean_proxy"] = 100 * np.exp(
        np.mean(np.log(np.maximum(alg[clean_proxy_axes].to_numpy() / 100.0, 1e-6)), axis=1)
    )
    alg = alg.sort_values("HexaScore_clean_proxy", ascending=False)
    return full, alg


def bootstrap_ci(full: pd.DataFrame, n: int = 500, seed: int = 20260504) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datasets = sorted(full["dataset"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for alg, g_alg in full.groupby("algorithm"):
        values = {axis: [] for axis in ["ID_Q", "OOD_G", "SYM_F", "EFF", "STAB"]}
        for _ in range(n):
            sample = rng.choice(datasets, size=len(datasets), replace=True)
            sample_df = pd.concat([g_alg[g_alg["dataset"] == d] for d in sample], ignore_index=True)
            values["ID_Q"].append(100 * sample_df["q_id"].mean())
            values["OOD_G"].append(100 * sample_df["q_ood_g"].mean())
            values["SYM_F"].append(100 * sample_df["q_sym_proxy"].mean())
            values["EFF"].append(100 * sample_df["eff_proxy"].mean())
            values["STAB"].append(100 * sample_df["stab_proxy"].mean())
        row = {"algorithm": alg}
        for axis, vals in values.items():
            row[f"{axis}_ci_low"] = float(np.percentile(vals, 2.5))
            row[f"{axis}_ci_high"] = float(np.percentile(vals, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def save_heatmap(scores: pd.DataFrame, out: Path) -> None:
    plot_cols = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
    data = scores.set_index("algorithm")[plot_cols].rename(
        columns={"ID_Q": "ID-Q", "OOD_G": "OOD-G", "SYM_F": "SYM-F"}
    )
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(data))))
    mat = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(masked, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=0)
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = mat[i, j]
            text = "N/A" if np.isnan(value) else f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111111")
    ax.set_title("Core-50 Six-Axis Scores v0 (ROB pending; SYM/EFF/STAB are proxies)")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_radar(scores: pd.DataFrame, out: Path, top_k: int = 4) -> None:
    cols = ["ID_Q", "OOD_G", "SYM_F", "EFF", "STAB"]
    labels = ["ID-Q", "OOD-G", "SYM-F*", "EFF*", "STAB*"]
    selected = scores.head(top_k)
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for _, row in selected.iterrows():
        values = [float(row[c]) for c in cols]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["algorithm"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Core-50 Radar v0 (asterisk axes are proxies)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_scatter(scores: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(scores[x], scores[y], s=55, color="#2f6f8f", alpha=0.85)
    for _, row in scores.iterrows():
        ax.annotate(str(row["algorithm"]), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x.replace("_", "-"))
    ax.set_ylabel(y.replace("_", "-"))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_metric_bar(scores: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    data = scores.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(4.5, 0.42 * len(data))))
    ax.barh(data["algorithm"], data[metric], color="#386641")
    ax.set_xlim(0, 100)
    ax.set_xlabel(metric.replace("_", "-"))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_report(outdir: Path, scores: pd.DataFrame) -> None:
    lines = [
        "# Core-50 hexagon v0 figures",
        "",
        "This directory contains a first executable version of the Core-50 six-axis plotting pipeline.",
        "",
        "## Metric status",
        "",
        "- `ID-Q`: formal, computed from median seed clean ID NMSE and the fixed phi mapping.",
        "- `OOD-G`: formal, computed from clean OOD quality plus ID-to-OOD retention.",
        "- `SYM-F`: proxy, using parseability, variable/operator F1, operator-variable Jaccard, and ID/OOD ultra-low-NMSE numeric-equivalence proxy.",
        "- `EFF`: proxy, because local clean Core50 results do not contain minute snapshots. The proxy combines final ID/OOD quality and median runtime.",
        "- `ROB`: pending, requires noisy-training results.",
        "- `STAB`: proxy, using numeric IQR, valid rate, and exact skeleton consistency; formal symbolic consistency can replace it later.",
        "",
        "## Files",
        "",
        "- `clean_final_runs.csv`: run-level clean metrics with failure-as-zero NMSE policy.",
        "- `core50_result_expressions.csv`: extracted final expressions and parse metadata from result.json.",
        "- `symbolic_metrics_proxy.csv`: seed-level symbolic proxy metrics.",
        "- `dataset_axis_components.csv`: dataset x algorithm components.",
        "- `hexagon_scores.csv`: algorithm-level scores.",
        "- `hexagon_scores_with_ci.csv`: algorithm-level scores plus bootstrap CI over datasets.",
        "- `fig_hexagon_heatmap.png`: 12 algorithms x 6 axes heatmap.",
        "- `fig_radar_top4.png`: radar chart for top-4 by clean/proxy HexaScore.",
        "- `fig_tradeoff_idq_oodg.png`, `fig_tradeoff_idq_symf.png`, `fig_tradeoff_eff_stab.png`: trade-off scatters.",
        "",
        "## Current ranking by clean/proxy HexaScore",
        "",
        scores[["algorithm", "ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB", "HexaScore_clean_proxy"]]
        .to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Core50 six-axis metrics v0")
    parser.add_argument("--runs-csv", default=str(DEFAULT_RUNS_CSV))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--core50-csv", default=str(DEFAULT_CORE50_CSV))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs_csv)
    run_df, dataset_df = compute_run_and_dataset_metrics(runs)
    result_df = extract_result_records(Path(args.experiment_root))
    core_meta = load_core50_metadata(Path(args.core50_csv))
    symbolic_df = compute_symbolic_proxy(run_df, result_df, core_meta)
    dataset_components, scores = compute_algorithm_scores(dataset_df, symbolic_df)
    ci = bootstrap_ci(dataset_components, n=args.bootstrap)
    scores_ci = scores.merge(ci, on="algorithm", how="left")

    run_df.to_csv(outdir / "clean_final_runs.csv", index=False)
    result_df.to_csv(outdir / "core50_result_expressions.csv", index=False)
    symbolic_df.to_csv(outdir / "symbolic_metrics_proxy.csv", index=False)
    dataset_components.to_csv(outdir / "dataset_axis_components.csv", index=False)
    scores.to_csv(outdir / "hexagon_scores.csv", index=False)
    scores_ci.to_csv(outdir / "hexagon_scores_with_ci.csv", index=False)

    save_heatmap(scores, outdir / "fig_hexagon_heatmap.png")
    save_radar(scores, outdir / "fig_radar_top4.png")
    save_scatter(scores, "ID_Q", "OOD_G", outdir / "fig_tradeoff_idq_oodg.png", "ID-Q vs OOD-G")
    save_scatter(scores, "ID_Q", "SYM_F", outdir / "fig_tradeoff_idq_symf.png", "ID-Q vs SYM-F proxy")
    save_scatter(scores, "EFF", "STAB", outdir / "fig_tradeoff_eff_stab.png", "EFF proxy vs STAB proxy")
    save_metric_bar(scores, "HexaScore_clean_proxy", outdir / "fig_clean_proxy_hexascore_bar.png", "Clean/proxy HexaScore")
    write_report(outdir, scores)
    print(json.dumps({"outdir": str(outdir), "algorithms": len(scores), "rows": len(run_df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
