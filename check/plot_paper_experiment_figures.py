#!/usr/bin/env python3
"""补齐论文实验叙事图表。

输出目录：
`exp-planning/04.Core50正式全量评测/analysis/paper_figures_20260505`

该脚本把现有 01/02/03/04/05 阶段资产串起来，生成论文 Q&A 实验叙事需要的
pipeline、数据池覆盖、双探针、Probe-4、Core-50 代表性、leaderboard、
symbolic / noise / stability / ablation 等图表。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/paper_figures_20260505"

RUNNABLE_664 = REPO_ROOT / "exp-planning/01.双探针实验/datasets_runnable.csv"
DUAL_PROBE = REPO_ROOT / "experiment-results/benchmark_formal200_20260417/one_seed_probe_dataset_compare.csv"
STAGE1_CANDIDATE = REPO_ROOT / "experiment-results/benchmark_selection_dossier_20260422/tables/stage1_candidate200_flat.csv"
E1_12 = REPO_ROOT / "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv"
PROBE4_ALG = REPO_ROOT / "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/probe4_selection_nmse_only/probe4_algorithm_scores_nmse_only.csv"
PROBE4_COMBO = REPO_ROOT / "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/probe4_selection_nmse_only/probe4_combo_scores_nmse_only.csv"
PROBE4_PAIR = REPO_ROOT / "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/probe4_selection_nmse_only/probe4_pairwise_complementarity_nmse_only.csv"
PROBE4_DATASET = REPO_ROOT / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_dataset_level.csv"
PROBE4_DATASET_ALG = REPO_ROOT / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_dataset_algorithm.csv"
PROBE4_RUN = REPO_ROOT / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_run_level.csv"
CORE50 = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
HEXAGON_V2 = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v2_formal_20260505"
HEXAGON_SCORES = HEXAGON_V2 / "hexagon_scores_formal.csv"
HEXAGON_SCORES_CI = HEXAGON_V2 / "hexagon_scores_formal_with_ci.csv"
HEXAGON_COMPONENTS = HEXAGON_V2 / "dataset_axis_components_formal.csv"
SYMF_RUN = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/symf_formal_metrics_20260504/symbolic_metrics_formal.csv"
CLEAN_RUNS = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/clean_final_runs_updated.csv"
NOISE_RUNS = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/noise_final_runs.csv"
NOISE_COMPLETION = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/noise_completion_summary.csv"
ROBUSTNESS = REPO_ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/robustness_components.csv"

PROBE4_SELECTED = ["dso", "pyoperon", "imcts", "udsr"]
AXES = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
DISPLAY = {"ID_Q": "ID-Q", "OOD_G": "OOD-G", "SYM_F": "SYM-F", "EFF": "EFF", "ROB": "ROBU", "STAB": "STAB"}
COLORS = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#577590",
    "#8ab17d",
    "#6d597a",
    "#bc6c25",
    "#43aa8b",
    "#f94144",
    "#277da1",
]


@dataclass
class FigureRecord:
    figure: str
    title: str
    status: str
    files: str
    note: str = ""


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "text.antialiased": True,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def log_nmse(value: Any, lo: float = -12.0, hi: float = 12.0) -> float:
    num = finite(value)
    if num is None or num < 0:
        return hi
    return float(np.clip(math.log10(max(num, 1e-12)), lo, hi))


def phi(value: Any) -> float:
    clipped = float(np.clip(log_nmse(value, -12, 2), -12, 2))
    return 1.0 - (clipped + 12.0) / 14.0


def entropy_score(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return 0.0
    p = counts.to_numpy(dtype=float) / counts.sum()
    h = float(-(p * np.log(p + 1e-12)).sum())
    return h / math.log(len(p)) if len(p) > 1 else 0.0


def tv_similarity(sub: pd.Series, full: pd.Series) -> float:
    sub_counts = sub.dropna().astype(str).value_counts(normalize=True)
    full_counts = full.dropna().astype(str).value_counts(normalize=True)
    keys = sorted(set(sub_counts.index) | set(full_counts.index))
    tv = 0.5 * sum(abs(float(sub_counts.get(k, 0.0)) - float(full_counts.get(k, 0.0))) for k in keys)
    return float(np.clip(1.0 - tv, 0.0, 1.0))


def save(fig: plt.Figure, out: Path) -> list[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=360, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [out.name, pdf.name]


def safe_name(text: str, max_len: int = 42) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def normalize_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        lo, hi = vals.quantile(0.05), vals.quantile(0.95)
        if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
            out[f"norm_{col}"] = 0.0
        else:
            out[f"norm_{col}"] = ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0.0)
    return out


def pca2(df: pd.DataFrame) -> np.ndarray:
    mat = df.to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = mat - mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    mat = mat / std
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    return mat @ vt[:2].T


def rankdata(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], abs_tol=1e-12):
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    ar = rankdata({k: a[k] for k in keys})
    br = rankdata({k: b[k] for k in keys})
    return pearson([ar[k] for k in keys], [br[k] for k in keys])


def kendall(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    c = d = 0
    for x, y in combinations(keys, 2):
        ax = (a[x] > a[y]) - (a[x] < a[y])
        bx = (b[x] > b[y]) - (b[x] < b[y])
        if ax == 0 or bx == 0:
            continue
        if ax == bx:
            c += 1
        else:
            d += 1
    return (c - d) / (c + d) if c + d else float("nan")


def pairwise_agreement(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    total = agree = 0
    for x, y in combinations(keys, 2):
        ax = (a[x] > a[y]) - (a[x] < a[y])
        bx = (b[x] > b[y]) - (b[x] < b[y])
        if ax == 0:
            continue
        total += 1
        agree += int(ax == bx)
    return agree / total if total else float("nan")


def method_scores_for_subset(dataset_alg: pd.DataFrame, dataset_ids: set[str]) -> dict[str, float]:
    sub = dataset_alg[dataset_alg["dataset_id"].isin(dataset_ids)].copy()
    scores: dict[str, float] = {}
    for method, group in sub.groupby("method_norm"):
        values = []
        for _, row in group.iterrows():
            idv = finite(row.get("median_log_id_nmse"))
            oodv = finite(row.get("median_log_ood_nmse"))
            if idv is None or oodv is None:
                values.append(12.0)
            else:
                values.append(0.5 * idv + 0.5 * oodv)
        scores[str(method)] = float(np.mean(values)) if values else 12.0
    return scores


def subset_metrics(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, subset_ids: set[str], label: str, full_scores: dict[str, float]) -> dict[str, float | str]:
    sub = dataset_level[dataset_level["dataset_id"].isin(subset_ids)].copy()
    if sub.empty:
        return {"subset": label}
    scores = method_scores_for_subset(dataset_alg, subset_ids)
    row = {
        "subset": label,
        "size": len(sub),
        "coverage": 0.35 * entropy_score(sub["family"]) + 0.25 * entropy_score(sub["subgroup"]) + 0.2 * entropy_score(sub["operator_group"]) + 0.2 * entropy_score(sub["feature_count_bin"]),
        "mean_info": float(pd.to_numeric(sub["info_score"], errors="coerce").fillna(0).mean()),
        "rank_fidelity": spearman(full_scores, scores),
        "kendall": kendall(full_scores, scores),
        "pairwise_win_agreement": pairwise_agreement(full_scores, scores),
        "aggregate_error": float(np.mean([abs(full_scores[k] - scores[k]) for k in sorted(set(full_scores) & set(scores))])),
        "stability": float(pd.to_numeric(sub["stability_score"], errors="coerce").fillna(0).mean()),
        "non_redundancy": float(sub["semantic_duplicate_group"].nunique() / max(1, len(sub))) if "semantic_duplicate_group" in sub else 1.0,
        "difficulty_balance": tv_similarity(sub["difficulty_bin"], dataset_level["difficulty_bin"]),
        "family_match": tv_similarity(sub["family"], dataset_level["family"]),
        "failure_match": tv_similarity(sub["failure_mode"], dataset_level["failure_mode"]),
    }
    row.update(selection_score_terms(dataset_level, subset_ids, row))
    return row


def core_ids_from_manifest(dataset_level: pd.DataFrame, core50: pd.DataFrame) -> set[str]:
    """精确使用冻结 manifest 中的 dataset_dir，避免跨来源同名任务误匹配。"""
    if "dataset_dir" in core50 and "dataset_rel" in dataset_level:
        core_dirs = set(core50["dataset_dir"].astype(str))
        core_ids = set(dataset_level[dataset_level["dataset_rel"].astype(str).isin(core_dirs)]["dataset_id"])
        if len(core_ids) >= 45:
            return core_ids
    core_names = set(core50["dataset_name"].astype(str)) if "dataset_name" in core50 else set()
    core_ids = set(dataset_level[dataset_level["dataset_name"].astype(str).isin(core_names)]["dataset_id"])
    if len(core_ids) < 45 and "basename" in dataset_level:
        core_ids = set(dataset_level[dataset_level["basename"].astype(str).isin(core_names)]["dataset_id"])
    return core_ids


def family_quota_bounds(dataset_level: pd.DataFrame) -> dict[str, tuple[int, int]]:
    counts = dataset_level["family"].astype(str).value_counts()
    total = len(dataset_level)
    n_families = len(counts)
    bounds: dict[str, tuple[int, int]] = {}
    for family, count in counts.items():
        target = 50.0 * (0.6 * float(count) / total + 0.4 / n_families)
        bounds[str(family)] = (max(1, math.floor(target - 1.0)), math.ceil(target + 2.0))
    return bounds


def hard_constraint_audit(dataset_level: pd.DataFrame, subset_ids: set[str]) -> dict[str, float]:
    sub = dataset_level[dataset_level["dataset_id"].isin(subset_ids)].copy()
    family_bounds = family_quota_bounds(dataset_level)

    size_violation = abs(len(sub) - 50)
    semantic_duplicate_excess = int((sub["semantic_duplicate_group"].astype(str).value_counts() - 1).clip(lower=0).sum()) if "semantic_duplicate_group" in sub else 0
    basename_duplicate_excess = int((sub["basename"].astype(str).value_counts() - 1).clip(lower=0).sum()) if "basename" in sub else 0

    family_quota_violation = 0
    for family, (lo, hi) in family_bounds.items():
        observed = int((sub["family"].astype(str) == family).sum())
        if observed < lo:
            family_quota_violation += lo - observed
        elif observed > hi:
            family_quota_violation += observed - hi

    subgroup_cap = 5
    subgroup_cap_violation = int((sub["subgroup"].astype(str).value_counts() - subgroup_cap).clip(lower=0).sum()) if "subgroup" in sub else 0

    hard_constraint_violations = (
        int(size_violation > 0)
        + int(semantic_duplicate_excess > 0)
        + int(basename_duplicate_excess > 0)
        + int(family_quota_violation > 0)
        + int(subgroup_cap_violation > 0)
    )
    hard_constraint_excess = size_violation + semantic_duplicate_excess + basename_duplicate_excess + family_quota_violation + subgroup_cap_violation
    return {
        "hard_constraint_violations": float(hard_constraint_violations),
        "hard_constraint_excess": float(hard_constraint_excess),
        "semantic_duplicate_excess": float(semantic_duplicate_excess),
        "basename_duplicate_excess": float(basename_duplicate_excess),
        "family_quota_violation": float(family_quota_violation),
        "subgroup_cap_violation": float(subgroup_cap_violation),
    }


def selection_balance(dataset_level: pd.DataFrame, subset_ids: set[str]) -> float:
    sub = dataset_level[dataset_level["dataset_id"].isin(subset_ids)].copy()
    balance_cols = [
        "family",
        "subgroup",
        "operator_group",
        "feature_count_bin",
        "sample_count_bin",
        "complexity_bin",
        "difficulty_bin",
        "failure_mode",
        "winner_probe",
        "eligible_class",
    ]
    scores = [tv_similarity(sub[col], dataset_level[col]) for col in balance_cols if col in sub and col in dataset_level]
    return float(np.mean(scores)) if scores else 0.0


def selection_score_terms(dataset_level: pd.DataFrame, subset_ids: set[str], row: dict[str, Any]) -> dict[str, float]:
    balance = selection_balance(dataset_level, subset_ids)
    raw_score = 0.45 * float(row.get("coverage", 0.0)) + 0.35 * float(row.get("mean_info", 0.0)) + 0.20 * balance
    audit = hard_constraint_audit(dataset_level, subset_ids)
    feasible_score = raw_score if audit["hard_constraint_violations"] == 0 else 0.0
    return {
        "selection_balance": balance,
        "raw_selection_score": raw_score,
        "feasible_selection_score": feasible_score,
        **audit,
    }


def build_core_baselines(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, core50: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    full_ids = set(dataset_level["dataset_id"])
    full_scores = method_scores_for_subset(dataset_alg, full_ids)
    core_ids = core_ids_from_manifest(dataset_level, core50)

    top_info_ids = set(dataset_level.sort_values(["info_score", "stability_score"], ascending=False).head(50)["dataset_id"])
    difficulty_ids: set[str] = set()
    for _, group in dataset_level.groupby("difficulty_bin"):
        n = max(1, round(50 * len(group) / len(dataset_level)))
        difficulty_ids.update(group.sort_values("info_score", ascending=False).head(n)["dataset_id"])
    difficulty_ids = set(dataset_level[dataset_level["dataset_id"].isin(difficulty_ids)].sort_values("info_score", ascending=False).head(50)["dataset_id"])

    meta = dataset_level.copy()
    meta["meta_score"] = (
        pd.to_numeric(meta["feature_count"], errors="coerce").rank(pct=True).fillna(0)
        + pd.to_numeric(meta["formula_operator_count"], errors="coerce").rank(pct=True).fillna(0)
        + pd.to_numeric(meta["total_samples"], errors="coerce").rank(pct=True).fillna(0)
    )
    metadata_ids: set[str] = set()
    for _, group in meta.groupby("family"):
        n = max(1, round(50 * len(group) / len(meta)))
        metadata_ids.update(group.sort_values("meta_score", ascending=False).head(n)["dataset_id"])
    metadata_ids = set(meta[meta["dataset_id"].isin(metadata_ids)].head(50)["dataset_id"])

    random_rows = []
    strat_rows = []
    all_ids = np.asarray(sorted(full_ids))
    family_groups = {fam: group["dataset_id"].to_numpy() for fam, group in dataset_level.groupby("family")}
    for idx in range(80):
        random_ids = set(rng.choice(all_ids, size=50, replace=False))
        random_rows.append(subset_metrics(dataset_level, dataset_alg, random_ids, f"random-{idx}", full_scores))
        strat_ids: set[str] = set()
        for fam, ids in family_groups.items():
            n = max(1, round(50 * len(ids) / len(dataset_level)))
            strat_ids.update(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())
        if len(strat_ids) > 50:
            strat_ids = set(rng.choice(np.asarray(sorted(strat_ids)), size=50, replace=False))
        strat_rows.append(subset_metrics(dataset_level, dataset_alg, strat_ids, f"family-random-{idx}", full_scores))

    named = {
        "Core-50": core_ids,
        "top-info-50": top_info_ids,
        "metadata-diverse-50": metadata_ids,
        "difficulty-balanced-50": difficulty_ids,
    }
    rows = [subset_metrics(dataset_level, dataset_alg, ids, label, full_scores) for label, ids in named.items()]
    avg_random = pd.DataFrame(random_rows).drop(columns=["subset"]).mean(numeric_only=True).to_dict()
    avg_random["subset"] = "random-50 avg"
    avg_strat = pd.DataFrame(strat_rows).drop(columns=["subset"]).mean(numeric_only=True).to_dict()
    avg_strat["subset"] = "family-stratified random-50 avg"
    rows.extend([avg_random, avg_strat])
    return pd.DataFrame(rows), named | {"random_example": set(rng.choice(all_ids, size=50, replace=False))}


def plot_pipeline(record: list[FigureRecord]) -> None:
    boxes = [
        ("800+ raw datasets", "collected SR sources"),
        ("664 GT datasets", "ground-truth + runnable splits"),
        ("Candidate-200", "PySR + LLM-SR dual probe"),
        ("Probe-4", "DSO + PyOperon + iMCTS + uDSR"),
        ("Core-50", "distilled evaluation set"),
        ("Leaderboard", "12 algorithms × 5 seeds + noise"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(boxes))
    for i, (x, (title, subtitle)) in enumerate(zip(xs, boxes, strict=False)):
        ax.text(x, 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.45", fc="#e8f3f1", ec="#2a9d8f", lw=1.6))
        ax.text(x, 0.34, subtitle, ha="center", va="center", fontsize=8.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.62), xytext=(x + 0.065, 0.62), arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))
    files = save(fig, OUTDIR / "figure01_pipeline.png")
    record.append(FigureRecord("Figure 1", "Pipeline", "generated", ";".join(files)))


def plot_reservoir_composition(dataset_level: pd.DataFrame, record: list[FigureRecord]) -> None:
    title_pad = 12

    def wrap_label(text: Any, width: int = 13) -> str:
        label = re.sub(r"[_/]+", " ", str(text))
        label = re.sub(r"\s+", " ", label).strip()
        words = label.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines[:2]) if lines else label

    def add_bar_labels(ax: plt.Axes, values: list[float] | np.ndarray) -> None:
        ymax = max(values) if len(values) else 0
        for idx, value in enumerate(values):
            ax.text(idx, value + ymax * 0.025, f"{int(value)}", ha="center", va="bottom", fontsize=8.0, fontweight="medium", color="#222222")

    # 正文里这张图会被压到单页文本宽度。保持 1x4 结构，并压缩高度，
    # 避免在论文中占用过多纵向空间。
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.25))
    family = dataset_level["family"].value_counts()
    family_labels = ["firstprinciples" if str(x) == "srbench2025/firstprinciples" else x for x in family.index]
    family_y = np.arange(len(family))
    axes[0].barh(family_y, family.values, color="#2a9d8f")
    axes[0].set_yticks(family_y)
    axes[0].set_yticklabels([wrap_label(x, 14) for x in family_labels])
    axes[0].invert_yaxis()
    family_xmax = max(family.values) if len(family.values) else 0
    axes[0].set_xlim(0, family_xmax * 1.18)
    for idx, value in enumerate(family.values):
        axes[0].text(value + family_xmax * 0.025, idx, f"{int(value)}", ha="left", va="center", fontsize=8.2, fontweight="medium", color="#222222")

    op = dataset_level["operator_group"].fillna("unknown").value_counts()
    op_label_map = {
        "trigonometric": "trig.",
        "exponential_log": "exp/log",
        "mixed_elementary": "mixed",
    }
    op_labels = [op_label_map.get(str(x), x) for x in op.index]
    op_y = np.arange(len(op))
    axes[1].barh(op_y, op.values, color="#f4a261")
    axes[1].set_yticks(op_y)
    axes[1].set_yticklabels([wrap_label(x, 14) for x in op_labels])
    axes[1].invert_yaxis()
    op_xmax = max(op.values) if len(op.values) else 0
    axes[1].set_xlim(0, op_xmax * 1.18)
    for idx, value in enumerate(op.values):
        axes[1].text(value + op_xmax * 0.025, idx, f"{int(value)}", ha="left", va="center", fontsize=8.2, fontweight="medium", color="#222222")

    complexity_order = ["simple", "moderate", "complex"]
    complexity = dataset_level["complexity_bin"].fillna("unknown").value_counts().reindex(complexity_order + ["unknown"]).dropna()
    axes[2].bar(np.arange(len(complexity)), complexity.values, color="#e9c46a")
    axes[2].set_xticks(np.arange(len(complexity)))
    axes[2].set_xticklabels([wrap_label(x, 12) for x in complexity.index], rotation=0)
    add_bar_labels(axes[2], complexity.values)

    difficulty_order = ["easy", "medium", "hard", "extreme"]
    difficulty = dataset_level["difficulty_bin"].fillna("unknown").value_counts().reindex(difficulty_order + ["unknown"]).dropna()
    axes[3].bar(np.arange(len(difficulty)), difficulty.values, color="#577590")
    axes[3].set_xticks(np.arange(len(difficulty)))
    axes[3].set_xticklabels([wrap_label(x, 12) for x in difficulty.index], rotation=0)
    add_bar_labels(axes[3], difficulty.values)

    for ax in axes:
        ax.grid(axis="x" if ax in axes[:2] else "y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.8)
        ax.tick_params(axis="y", labelsize=8.8)

    fig.subplots_adjust(left=0.070, right=0.985, top=0.70, bottom=0.11, wspace=0.38)
    files = save(fig, OUTDIR / "figure02_reservoir_composition.png")
    record.append(FigureRecord("Figure 2", "GT-Reservoir composition", "generated", ";".join(files)))

    stats = (
        dataset_level.groupby("family", dropna=False)
        .agg(
            datasets=("dataset_id", "nunique"),
            subgroups=("subgroup", "nunique"),
            median_variables=("feature_count", "median"),
            median_complexity=("formula_operator_count", "median"),
            median_train_samples=("train_samples", "median"),
            ood_available=("ood_test_samples", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
        )
        .reset_index()
        .sort_values("datasets", ascending=False)
    )
    stats.to_csv(OUTDIR / "table01_dataset_statistics.csv", index=False)
    (OUTDIR / "table01_dataset_statistics.md").write_text(stats.to_markdown(index=False), encoding="utf-8")


def plot_dual_probe(dual: pd.DataFrame, record: list[FigureRecord]) -> None:
    df = dual.copy()
    df["selected"] = df["is_formal200_candidate"].astype(bool)
    for col in ["pysr_id_nmse", "llmsr_id_nmse", "pysr_ood_nmse", "llmsr_ood_nmse"]:
        df[f"log_{col}"] = df[col].map(log_nmse)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, split in zip(axes, ["id", "ood"], strict=False):
        x = f"log_pysr_{split}_nmse"
        y = f"log_llmsr_{split}_nmse"
        ax.scatter(df.loc[~df["selected"], x], df.loc[~df["selected"], y], s=13, alpha=0.38, color="#aaaaaa", label="not selected")
        ax.scatter(df.loc[df["selected"], x], df.loc[df["selected"], y], s=18, alpha=0.82, color="#e76f51", label="Candidate-200")
        lim = [min(df[x].min(), df[y].min()) - 0.4, max(df[x].max(), df[y].max()) + 0.4]
        ax.plot(lim, lim, "--", color="#333333", linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"PySR {split.upper()} log10 NMSE")
        ax.set_ylabel(f"LLM-SR {split.upper()} log10 NMSE")
        ax.grid(alpha=0.45)
    axes[0].legend(frameon=False)
    files = save(fig, OUTDIR / "figure03_dual_probe_scatter.png")
    record.append(FigureRecord("Figure 3", "PySR vs LLM-SR gap scatter", "generated", ";".join(files)))

    df["gap_score"] = 0.5 * (df["log_pysr_id_nmse"] - df["log_llmsr_id_nmse"]).abs() + 0.5 * (df["log_pysr_ood_nmse"] - df["log_llmsr_ood_nmse"]).abs()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(df.loc[~df["selected"], "gap_score"], bins=36, alpha=0.62, label="not selected", color="#999999")
    ax.hist(df.loc[df["selected"], "gap_score"], bins=36, alpha=0.72, label="Candidate-200", color="#e76f51")
    ax.set_xlabel("overall gap score")
    ax.set_ylabel("# datasets")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.45)
    files = save(fig, OUTDIR / "figure04_gap_score_distribution.png")
    record.append(FigureRecord("Figure 4", "overall_gap_score distribution", "generated", ";".join(files)))


def plot_candidate_composition(candidate: pd.DataFrame, record: list[FigureRecord]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    advantage_col = "candidate_advantage_side" if "candidate_advantage_side" in candidate.columns else "advantage_side"
    for ax, col, title, color in [
        (axes[0], "pool", "Candidate pools", "#2a9d8f"),
        (axes[1], "selection_mode", "Selection modes", "#e9c46a"),
        (axes[2], advantage_col, "Advantage side", "#e76f51"),
    ]:
        vc = candidate[col].fillna("unknown").value_counts()
        ax.bar(vc.index, vc.values, color=color)
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vc.values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=8)
    files = save(fig, OUTDIR / "figure05_candidate200_composition_modes.png")

    fam = candidate["family"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(fam.index, fam.values, color="#577590")
    ax.set_ylabel("# datasets")
    ax.tick_params(axis="x", rotation=35)
    files += save(fig, OUTDIR / "figure05_candidate200_family_distribution.png")
    record.append(FigureRecord("Figure 5", "Candidate-200 composition", "generated", ";".join(files)))


def plot_probe4_selection(e1: pd.DataFrame, alg: pd.DataFrame, combos: pd.DataFrame, pair: pd.DataFrame, record: list[FigureRecord]) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    df = alg.sort_values("available_metric_score", ascending=False)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["finite_id_ood_rate"], width=0.2, label="finite ID/OOD", color="#2a9d8f")
    ax.bar(x, 1 - df["id_ood_explosion_rate_gt_100"], width=0.2, label="non-explosion", color="#e9c46a")
    ax.bar(x + 0.2, df["available_metric_score"], width=0.2, label="health/score", color="#e76f51")
    ax.set_xticks(x, df["algorithm"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.45)
    files = save(fig, OUTDIR / "figure06_algorithm_health.png")
    record.append(FigureRecord("Figure 6", "12 algorithm health", "generated", ";".join(files)))

    pivot = e1.pivot_table(index="dataset_id", columns="algorithm", values="combined_log_id_ood_nmse", aggfunc="mean")
    corr = pivot.corr(method="spearman").fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    files = save(fig, OUTDIR / "figure07_algorithm_behavior_correlation.png")
    record.append(FigureRecord("Figure 7", "algorithm behavior correlation", "generated", ";".join(files)))

    top = combos.head(10).copy().iloc[::-1]
    fields = [
        ("mean_operational_stability", "Health"),
        ("mean_discrimination", "Discrimination"),
        ("mean_pairwise_complementarity", "Complementarity"),
        ("mean_family_subgroup_coverage", "Coverage"),
        ("taxonomy_diversity_score", "Taxonomy"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    left = np.zeros(len(top))
    y = np.arange(len(top))
    for idx, (field, label) in enumerate(fields):
        vals = pd.to_numeric(top[field], errors="coerce").fillna(0).to_numpy()
        ax.barh(y, vals, left=left, label=label, color=COLORS[idx])
        left += vals
    ax.set_yticks(y, [safe_name(c, 46) for c in top["combo"]])
    ax.set_xlabel("score components (not re-normalized)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="x", alpha=0.45)
    files = save(fig, OUTDIR / "figure08_probe4_score_decomposition.png")
    record.append(FigureRecord("Figure 8", "Probe-4 selection score decomposition", "generated", ";".join(files)))

    info = compute_e1_info_scores(e1)
    fig, ax = plt.subplots(figsize=(6.7, 5.6))
    ax.scatter(info["teacher_info"], info["probe4_info"], s=24, alpha=0.72, color="#2a9d8f", edgecolor="white", linewidth=0.35)
    rho = info[["teacher_info", "probe4_info"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("I^12(dataset)")
    ax.set_ylabel("I^Probe4(dataset)")
    ax.grid(alpha=0.45)
    files = save(fig, OUTDIR / "figure09_panel_fidelity_scatter.png")
    info.to_csv(OUTDIR / "figure09_panel_fidelity_points.csv", index=False)
    record.append(FigureRecord("Figure 9", "Panel Fidelity scatter", "generated", ";".join(files)))


def compute_e1_info_scores(e1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, group in e1.groupby("dataset_id"):
        row = {"dataset_id": dataset_id}
        for label, algs in [("teacher", sorted(group["algorithm"].unique())), ("probe4", PROBE4_SELECTED)]:
            sub = group[group["algorithm"].isin(algs)]
            vals = pd.to_numeric(sub["combined_log_id_ood_nmse"], errors="coerce").dropna()
            valid_rate = float(pd.to_numeric(sub["finite_id_ood"], errors="coerce").fillna(0).mean()) if len(sub) else 0.0
            p = np.clip(valid_rate, 1e-9, 1 - 1e-9)
            valid_entropy = float(-(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2))
            row[f"{label}_info"] = float(vals.var(ddof=0) if len(vals) >= 2 else 0.0) + 0.25 * valid_entropy
        rows.append(row)
    return pd.DataFrame(rows)


def plot_core50_validity(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, core50: pd.DataFrame, record: list[FigureRecord]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    core_ids = core_ids_from_manifest(dataset_level, core50)
    candidate = read_csv(STAGE1_CANDIDATE)
    candidate_names = set(candidate["dataset"].astype(str)) if "dataset" in candidate else set()

    features = dataset_level[
        [
            "feature_count",
            "train_samples",
            "valid_samples",
            "id_test_samples",
            "ood_test_samples",
            "formula_operator_count",
            "formula_char_count",
            "info_score",
            "difficulty_score",
            "stability_score",
        ]
    ].copy()
    for col in features:
        features[col] = np.log1p(pd.to_numeric(features[col], errors="coerce").fillna(0)) if "samples" in col or "count" in col else pd.to_numeric(features[col], errors="coerce").fillna(0)
    coords = pca2(features)
    dataset_level = dataset_level.copy()
    dataset_level["pc1"] = coords[:, 0]
    dataset_level["pc2"] = coords[:, 1]
    dataset_level["is_core50"] = dataset_level["dataset_id"].astype(str).isin(core_ids)
    dataset_level["is_candidate200"] = dataset_level["dataset_name"].astype(str).isin(candidate_names) | dataset_level["basename"].astype(str).isin(candidate_names)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.scatter(dataset_level["pc1"], dataset_level["pc2"], s=14, color="#bbbbbb", alpha=0.45, label="GT-Reservoir-664")
    cand = dataset_level[dataset_level["is_candidate200"]]
    core = dataset_level[dataset_level["is_core50"]]
    ax.scatter(cand["pc1"], cand["pc2"], s=20, color="#5dade2", alpha=0.6, label="Candidate-200")
    ax.scatter(core["pc1"], core["pc2"], s=42, color="#e76f51", alpha=0.95, label="Core-50")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.35)
    files = save(fig, OUTDIR / "figure10_core50_coverage_map.png")
    record.append(FigureRecord("Figure 10", "Core-50 coverage map", "generated", ";".join(files), "PCA over structural + Probe4 response features."))

    rng = np.random.default_rng(20260505)
    baseline, subsets = build_core_baselines(dataset_level, dataset_alg, core50, rng)
    # This is only the unweighted radar-axis mean for plotting diagnostics.
    # The actual selector objective is feasible_selection_score.
    baseline["radar_mean_for_plot_only"] = baseline[["coverage", "mean_info", "rank_fidelity", "stability", "non_redundancy", "difficulty_balance"]].mean(axis=1)
    baseline.to_csv(OUTDIR / "core50_baseline_quality_metrics.csv", index=False)

    radar_cols = ["coverage", "mean_info", "rank_fidelity", "stability", "non_redundancy", "difficulty_balance"]
    fig = plt.figure(figsize=(8.2, 7.5))
    ax = fig.add_subplot(111, polar=True)
    labels = ["Coverage", "MeanInfo", "RankFid", "Stability", "NonRedund", "DiffBal"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    for i, (_, row) in enumerate(baseline.iterrows()):
        values = [float(row[c]) for c in radar_cols]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.9, label=row["subset"], color=COLORS[i % len(COLORS)])
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)
    files = save(fig, OUTDIR / "figure11_core50_vs_baselines_radar.png")
    record.append(FigureRecord("Figure 11", "Core-50 vs baselines quality radar", "generated", ";".join(files)))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    metrics = ["rank_fidelity", "kendall", "pairwise_win_agreement", "aggregate_error"]
    plot_df = baseline.set_index("subset")[metrics].copy()
    plot_df["aggregate_error"] = 1 - plot_df["aggregate_error"].clip(0, 1)
    x = np.arange(len(plot_df))
    width = 0.19
    for i, col in enumerate(metrics):
        label = "1-aggregate_error" if col == "aggregate_error" else col
        ax.bar(x + (i - 1.5) * width, plot_df[col], width, label=label)
    ax.set_xticks(x, [safe_name(s, 20) for s in plot_df.index], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.45)
    files = save(fig, OUTDIR / "figure12_rank_fidelity_comparison.png")
    record.append(FigureRecord("Figure 12", "Rank fidelity comparison", "generated", ";".join(files)))

    distribution_plots(dataset_level, subsets, record)
    k_scaling(dataset_level, dataset_alg, record)
    return baseline, subsets


def distribution_plots(dataset_level: pd.DataFrame, subsets: dict[str, set[str]], record: list[FigureRecord]) -> None:
    groups = {"Full-664": set(dataset_level["dataset_id"])}
    groups.update({k: v for k, v in subsets.items() if k in {"Core-50", "top-info-50", "metadata-diverse-50", "difficulty-balanced-50"}})
    fields = [("family", "Family"), ("difficulty_bin", "Difficulty"), ("failure_mode", "Failure mode")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (field, title) in zip(axes, fields, strict=False):
        cats = dataset_level[field].fillna("unknown").value_counts().head(8).index.tolist()
        bottom = np.zeros(len(groups))
        x = np.arange(len(groups))
        for i, cat in enumerate(cats):
            vals = []
            for ids in groups.values():
                sub = dataset_level[dataset_level["dataset_id"].isin(ids)]
                vals.append(float((sub[field].fillna("unknown") == cat).mean()))
            ax.bar(x, vals, bottom=bottom, label=safe_name(cat, 16), color=COLORS[i % len(COLORS)])
            bottom += np.asarray(vals)
        ax.set_xticks(x, [safe_name(k, 14) for k in groups], rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.35)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    files = save(fig, OUTDIR / "figure13_distribution_matching.png")
    record.append(FigureRecord("Figure 13", "distribution matching", "generated", ";".join(files)))


def k_scaling(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, record: list[FigureRecord]) -> None:
    full_scores = method_scores_for_subset(dataset_alg, set(dataset_level["dataset_id"]))
    ks = [20, 30, 40, 50, 60, 80, 100]
    rows = []
    rng = np.random.default_rng(20260506)
    all_ids = np.asarray(sorted(dataset_level["dataset_id"]))
    for k in ks:
        top_ids = set(dataset_level.sort_values(["info_score", "stability_score"], ascending=False).head(k)["dataset_id"])
        row = subset_metrics(dataset_level, dataset_alg, top_ids, f"top-info-{k}", full_scores)
        row["K"] = k
        row["selector"] = "top-info"
        rows.append(row)
        rand_metrics = []
        for _ in range(40):
            ids = set(rng.choice(all_ids, size=k, replace=False))
            rand_metrics.append(subset_metrics(dataset_level, dataset_alg, ids, f"random-{k}", full_scores))
        avg = pd.DataFrame(rand_metrics).mean(numeric_only=True).to_dict()
        avg["K"] = k
        avg["selector"] = "random avg"
        rows.append(avg)
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "figure14_k_scaling_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for selector, group in df.groupby("selector"):
        group = group.sort_values("K")
        ax.plot(group["K"], group["coverage"], marker="o", label=f"{selector}: coverage")
        ax.plot(group["K"], group["pairwise_win_agreement"], marker="s", linestyle="--", label=f"{selector}: pairwise")
        ax.plot(group["K"], group["stability"], marker="^", linestyle=":", label=f"{selector}: stability")
    ax.plot(ks, np.asarray(ks) / max(ks), color="#444", linewidth=1.2, label="relative cost")
    ax.axvline(50, color="#e76f51", linestyle="--", linewidth=1)
    ax.set_xlabel("K")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.45)
    files = save(fig, OUTDIR / "figure14_k_scaling_curve.png")
    record.append(FigureRecord("Figure 14", "K-scaling curve", "generated", ";".join(files)))


def copy_v2_figure(src_name: str, dst_stem: str, figure: str, title: str, record: list[FigureRecord]) -> None:
    files = []
    for suffix in [".png", ".pdf"]:
        src = HEXAGON_V2 / f"{src_name}{suffix}"
        if src.exists():
            dst = OUTDIR / f"{dst_stem}{suffix}"
            shutil.copy2(src, dst)
            files.append(dst.name)
    record.append(FigureRecord(figure, title, "copied_from_v2" if files else "missing", ";".join(files), str(HEXAGON_V2 / src_name)))


def plot_tradeoffs_and_symbolic(record: list[FigureRecord]) -> None:
    scores = read_csv(HEXAGON_SCORES)
    sym = read_csv(SYMF_RUN)
    clean = read_csv(CLEAN_RUNS)
    copy_v2_figure("fig_hexagon_heatmap_formal", "figure15_hexagon_heatmap_formal", "Figure 15", "12 algorithms x 6 axes heatmap", record)
    copy_v2_figure("fig_radar_top4_formal", "figure16_radar_top4_formal", "Figure 16", "Top-4 radar", record)
    copy_v2_figure("fig_tradeoff_idq_symf_formal", "figure17_idq_vs_symf", "Figure 17", "ID-Q vs SYM-F", record)

    fig, ax = plt.subplots(figsize=(6.7, 5.6))
    for idx, row in scores.iterrows():
        ax.scatter(row["SYM_F"], row["OOD_G"], s=62, color=COLORS[idx % len(COLORS)], edgecolor="white")
        ax.text(row["SYM_F"] + 0.7, row["OOD_G"] + 0.7, row["algorithm"], fontsize=8)
    ax.set_xlabel("SYM-F")
    ax.set_ylabel("OOD-G")
    ax.set_xlim(0, max(100, float(scores["SYM_F"].max()) * 1.15))
    ax.set_ylim(0, max(100, float(scores["OOD_G"].max()) * 1.15))
    ax.grid(alpha=0.45)
    files = save(fig, OUTDIR / "figure18_symf_vs_oodg.png")
    record.append(FigureRecord("Figure 18", "OOD-G vs SYM-F scatter", "generated", ";".join(files)))

    merged = sym.merge(clean[["algorithm", "gid", "dataset", "seed", "id_test_nmse", "ood_test_nmse"]], on=["algorithm", "gid", "dataset", "seed"], how="left")
    cases = merged[
        (merged["equiv_final"] == False)
        & (pd.to_numeric(merged["id_test_nmse"], errors="coerce") < 1e-4)
        & (pd.to_numeric(merged["ood_test_nmse"], errors="coerce") < 1e-2)
    ].copy()
    cases["ood_test_nmse"] = pd.to_numeric(cases["ood_test_nmse"], errors="coerce")
    cases = cases.sort_values("ood_test_nmse").head(8)
    cases_out = cases[
        [
            "algorithm",
            "dataset",
            "seed",
            "id_test_nmse",
            "ood_test_nmse",
            "gt_expression_x",
            "pred_expression_cleaned",
            "sym_f_formal",
            "failure_reason",
        ]
    ].copy()
    cases_out.to_csv(OUTDIR / "figure19_low_nmse_non_equivalent_cases.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    table_data = [
        [
            r["algorithm"],
            safe_name(r["dataset"], 22),
            f"{r['id_test_nmse']:.1e}",
            f"{r['ood_test_nmse']:.1e}",
            safe_name(r["gt_expression_x"], 44),
            safe_name(r["pred_expression_cleaned"], 58),
        ]
        for _, r in cases_out.iterrows()
    ]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["alg", "dataset", "ID NMSE", "OOD NMSE", "ground truth", "prediction"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    tbl.scale(1, 1.6)
    files = save(fig, OUTDIR / "figure19_low_nmse_non_equivalent_cases.png")
    record.append(FigureRecord("Figure 19", "low-NMSE non-equivalent cases", "generated", ";".join(files)))

    thresholds = [1e-2, 1e-4, 1e-6, 1e-8]
    rows = []
    for alg, group in merged.groupby("algorithm"):
        for t in thresholds:
            eligible = group[pd.to_numeric(group["ood_test_nmse"], errors="coerce") <= t]
            rows.append(
                {
                    "algorithm": alg,
                    "threshold": t,
                    "runs_under_threshold": len(eligible),
                    "equiv_rate": float(eligible["equiv_final"].mean()) if len(eligible) else 0.0,
                }
            )
    eq = pd.DataFrame(rows)
    eq.to_csv(OUTDIR / "figure20_equivalence_rate_by_nmse_threshold.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for idx, (alg, group) in enumerate(eq.groupby("algorithm")):
        group = group.sort_values("threshold", ascending=False)
        ax.plot([f"1e{int(math.log10(t))}" for t in group["threshold"]], 100 * group["equiv_rate"], marker="o", label=alg, color=COLORS[idx % len(COLORS)])
    ax.set_ylabel("equivalence rate (%)")
    ax.set_xlabel("OOD NMSE threshold")
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    files = save(fig, OUTDIR / "figure20_equivalence_rate_by_nmse_threshold.png")
    record.append(FigureRecord("Figure 20", "equivalence rate by NMSE threshold", "generated", ";".join(files)))


def snapshot_rows(max_files: int = 120000) -> pd.DataFrame:
    roots = [
        REPO_ROOT / "exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_clean_modelsplit_slim_20260504-224415",
        REPO_ROOT / "exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_noise_cpu_20260504-184611",
        REPO_ROOT / "exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_noise_gpu_20260504-184619",
    ]
    wanted_minutes = {1, 2, 3, 5, 10, 20, 30, 45, 60}
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int, int, str]] = set()
    parsed = 0
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            if "/experiments/" in dirpath:
                continue
            if "progress" not in dirpath:
                continue
            for name in filenames:
                if not name.startswith("minute_") or not name.endswith(".json"):
                    continue
                m = re.search(r"minute_(\d+)", name)
                minute = int(m.group(1)) if m else 0
                if minute not in wanted_minutes:
                    continue
                path = Path(dirpath) / name
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                alg = str(obj.get("tool") or "").lower()
                dataset = str(obj.get("dataset") or "")
                seed = int(obj.get("seed") or -1)
                sigma = 0.0
                text = str(path)
                if "sigma001" in text:
                    sigma = 0.01
                elif "sigma005" in text:
                    sigma = 0.05
                elif "sigma010" in text:
                    sigma = 0.10
                key = (alg, dataset, seed, minute, f"{sigma:.2f}")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                id_nmse = split_nmse(obj, "id_test")
                ood_nmse = split_nmse(obj, "ood_test")
                valid = id_nmse is not None and ood_nmse is not None
                artifact = obj.get("canonical_artifact") if isinstance(obj.get("canonical_artifact"), dict) else {}
                complexity = finite(artifact.get("complexity") if isinstance(artifact, dict) else None)
                if complexity is None:
                    complexity = finite(obj.get("source_complexity"))
                rows.append(
                    {
                        "algorithm": alg,
                        "dataset": dataset,
                        "seed": seed,
                        "minute": minute,
                        "noise_sigma": sigma,
                        "id_nmse": id_nmse,
                        "ood_nmse": ood_nmse,
                        "quality": 0.5 * phi(id_nmse) + 0.5 * phi(ood_nmse) if valid else 0.0,
                        "complexity": complexity,
                        "valid": valid,
                    }
                )
                parsed += 1
                if parsed >= max_files:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def split_nmse(obj: dict[str, Any], split: str) -> float | None:
    block = obj.get(split)
    if not isinstance(block, dict):
        return None
    val = finite(block.get("nmse"))
    return val if val is not None and val >= 0 else None


def plot_search_dynamics(record: list[FigureRecord]) -> None:
    snap = snapshot_rows()
    snap.to_csv(OUTDIR / "minute_snapshot_sample_for_figures.csv", index=False)
    if snap.empty:
        for i in range(21, 25):
            record.append(FigureRecord(f"Figure {i}", "minute-level snapshot plot", "missing", "", "No local minute snapshots found."))
        return
    clean_or_low_noise = snap[snap["noise_sigma"].isin([0.0, 0.01])].copy()
    agg = clean_or_low_noise.groupby(["algorithm", "minute"], dropna=False).agg(quality=("quality", "median"), valid_rate=("valid", "mean"), complexity=("complexity", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for idx, (alg, group) in enumerate(agg.groupby("algorithm")):
        group = group.sort_values("minute")
        ax.plot(group["minute"], 100 * group["quality"], marker="o", linewidth=1.6, label=alg, color=COLORS[idx % len(COLORS)])
    ax.set_xlabel("minute")
    ax.set_ylabel("median best-so-far quality")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    files = save(fig, OUTDIR / "figure21_anytime_performance_curve.png")
    record.append(FigureRecord("Figure 21", "anytime performance curve", "generated_proxy", ";".join(files), "Uses locally available clean LLM and low-noise snapshots."))

    threshold = 0.5
    first_hit = (
        clean_or_low_noise[clean_or_low_noise["quality"] >= threshold]
        .groupby(["algorithm", "dataset", "seed"], dropna=False)["minute"]
        .min()
        .reset_index()
    )
    all_runs = clean_or_low_noise.groupby(["algorithm", "dataset", "seed"], dropna=False).size().reset_index()[["algorithm", "dataset", "seed"]]
    first_hit = all_runs.merge(first_hit, on=["algorithm", "dataset", "seed"], how="left")
    minutes = sorted(clean_or_low_noise["minute"].unique())
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for idx, (alg, group) in enumerate(first_hit.groupby("algorithm")):
        surv = [float(((group["minute"].isna()) | (group["minute"] > m)).mean()) for m in minutes]
        ax.plot(minutes, surv, marker="o", linewidth=1.6, label=alg, color=COLORS[idx % len(COLORS)])
    ax.set_xlabel("minute")
    ax.set_ylabel("fraction not yet reaching q_time >= 0.5")
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    files = save(fig, OUTDIR / "figure22_time_to_threshold_survival.png")
    record.append(FigureRecord("Figure 22", "time-to-threshold survival curve", "generated_proxy", ";".join(files)))

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    comp = agg.dropna(subset=["complexity"])
    for idx, (alg, group) in enumerate(comp.groupby("algorithm")):
        group = group.sort_values("minute")
        ax.plot(group["minute"], group["complexity"], marker="o", linewidth=1.6, label=alg, color=COLORS[idx % len(COLORS)])
    ax.set_xlabel("minute")
    ax.set_ylabel("median expression complexity")
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    files = save(fig, OUTDIR / "figure23_complexity_over_time.png")
    record.append(FigureRecord("Figure 23", "complexity over time", "generated_proxy", ";".join(files)))

    pivot = clean_or_low_noise[clean_or_low_noise["minute"].isin([5, 60])].pivot_table(index=["algorithm", "dataset", "seed"], columns="minute", values="quality", aggfunc="max").reset_index()
    if 5 in pivot.columns and 60 in pivot.columns:
        fig, ax = plt.subplots(figsize=(6.6, 5.5))
        for idx, (alg, group) in enumerate(pivot.groupby("algorithm")):
            ax.scatter(100 * group[5], 100 * group[60], s=28, alpha=0.68, label=alg, color=COLORS[idx % len(COLORS)])
        ax.plot([0, 100], [0, 100], "--", color="#777", linewidth=1)
        ax.set_xlabel("5-minute quality")
        ax.set_ylabel("60-minute quality")
        ax.grid(alpha=0.45)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        files = save(fig, OUTDIR / "figure24_early_vs_final_performance.png")
        record.append(FigureRecord("Figure 24", "early vs final performance", "generated_proxy", ";".join(files)))
    else:
        record.append(FigureRecord("Figure 24", "early vs final performance", "missing", "", "No minute 5 and 60 pair found."))


def plot_noise_stability_metric_design(record: list[FigureRecord]) -> None:
    copy_v2_figure("fig_noise_rob_by_sigma", "figure25_noise_robustness_curve", "Figure 25", "noise robustness curve", record)
    copy_v2_figure("fig_noise_quality_by_sigma_heatmap", "figure26_noise_quality_heatmap", "Figure 26", "ROBU heatmap", record)

    comp = read_csv(HEXAGON_COMPONENTS)
    agg = (
        comp.groupby("algorithm")
        .agg(
            numerical_stability=("numeric_stability", "mean"),
            valid_rate=("valid_rate", "mean"),
            structural_consistency=("structural_consistency_proxy", "mean"),
            performance_correction=("q_perf_proxy", "mean"),
            STAB=("STAB_component", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5.3))
    x = np.arange(len(agg))
    width = 0.18
    for i, col in enumerate(["numerical_stability", "valid_rate", "structural_consistency", "performance_correction"]):
        ax.bar(x + (i - 1.5) * width, 100 * agg[col], width, label=col)
    ax.plot(x, 100 * agg["STAB"], color="#111", marker="o", linewidth=1.8, label="final STAB")
    ax.set_xticks(x, agg["algorithm"], rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("component score")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.45)
    files = save(fig, OUTDIR / "figure27_stab_component_chart.png")
    record.append(FigureRecord("Figure 27", "STAB component chart", "generated", ";".join(files)))

    scores = read_csv(HEXAGON_SCORES)
    rng = np.random.default_rng(20260507)
    rows = []
    full_rank = scores.set_index("algorithm")["HexaScore_formal_with_ROB"].rank(ascending=False, pct=True)
    for _ in range(120):
        size = int(rng.integers(5, len(scores) + 1))
        subset = scores.sample(size, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        rank_score = subset.set_index("algorithm")["HexaScore_formal_with_ROB"].rank(ascending=False, pct=True)
        for alg in subset["algorithm"]:
            rows.append(
                {
                    "algorithm": alg,
                    "absolute_drift": 0.0,
                    "rank_based_drift": abs(float(rank_score[alg]) - float(full_rank[alg])),
                }
            )
    drift = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.boxplot([drift["absolute_drift"], drift["rank_based_drift"]], labels=["absolute score", "rank-normalized score"])
    ax.set_ylabel("score drift under leaderboard resampling")
    ax.grid(axis="y", alpha=0.45)
    files = save(fig, OUTDIR / "figure28_score_drift_comparison.png")
    record.append(FigureRecord("Figure 28", "score drift comparison", "generated", ";".join(files)))

    corr = scores[AXES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(AXES)), [DISPLAY[a] for a in AXES], rotation=35, ha="right")
    ax.set_yticks(range(len(AXES)), [DISPLAY[a] for a in AXES])
    for i in range(len(AXES)):
        for j in range(len(AXES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    files = save(fig, OUTDIR / "figure29_axis_correlation_matrix.png")
    record.append(FigureRecord("Figure 29", "axis correlation matrix", "generated", ";".join(files)))

    ci = read_csv(HEXAGON_SCORES_CI)
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), sharex=True)
    order = scores["algorithm"].tolist()
    for ax, axis in zip(axes.ravel(), AXES, strict=False):
        merged = scores[["algorithm", axis]].merge(ci[["algorithm", f"{axis}_ci_low", f"{axis}_ci_high"]], on="algorithm", how="left")
        merged = merged.set_index("algorithm").loc[order].reset_index()
        y = np.arange(len(merged))
        low = merged[axis] - merged[f"{axis}_ci_low"]
        high = merged[f"{axis}_ci_high"] - merged[axis]
        ax.errorbar(
            merged[axis],
            y,
            xerr=[low, high],
            fmt="o",
            color="#2a9d8f",
            ecolor="#555",
            capsize=4,
            markersize=6,
            elinewidth=1.35,
        )
        ax.set_yticks(y, merged["algorithm"])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="x", alpha=0.45)
        if axis not in {"ID_Q", "EFF"}:
            ax.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.16, hspace=0.28)
    files = save(fig, OUTDIR / "figure30_bootstrap_ci.png")
    record.append(FigureRecord("Figure 30", "bootstrap confidence interval", "generated", ";".join(files)))


def plot_ablation_summary(baseline: pd.DataFrame, record: list[FigureRecord]) -> None:
    if baseline.empty:
        record.append(FigureRecord("Figure 31", "ablation summary", "missing", "", "No baseline metrics."))
        return
    df = baseline.copy()
    if "feasible_selection_score" not in df:
        df["feasible_selection_score"] = 0.0
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    df = df.sort_values("feasible_selection_score", ascending=True)
    ax.barh(df["subset"], df["feasible_selection_score"], color="#2a9d8f")
    ax.set_xlim(0, 1)
    ax.set_xlabel("feasible weighted selection score")
    ax.grid(axis="x", alpha=0.45)
    files = save(fig, OUTDIR / "figure31_ablation_summary.png")
    record.append(FigureRecord("Figure 31", "ablation summary", "generated", ";".join(files), "Infeasible baselines receive zero feasible score after hard-constraint gating."))


def write_report(records: list[FigureRecord]) -> None:
    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_csv(OUTDIR / "figure_coverage.csv", index=False)
    lines = [
        "# Paper Experiment Figures 20260505",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- This directory supplements the Core-50 formal hexagon figures with distillation-validity and reviewer-facing experiment figures.",
        "- Figures marked `generated_proxy` are valid for exploratory analysis but should be described with their exact data scope in the paper.",
        "",
        "## Coverage",
        "",
        df.to_markdown(index=False),
        "",
    ]
    (OUTDIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outdir": str(OUTDIR.resolve()),
        "figures": [r.__dict__ for r in records],
        "png_count": len(list(OUTDIR.glob("figure*.png"))),
        "pdf_count": len(list(OUTDIR.glob("figure*.pdf"))),
    }
    (OUTDIR / "figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    global OUTDIR
    parser = argparse.ArgumentParser(description="Plot paper experiment figures")
    parser.add_argument("--outdir", default=str(OUTDIR))
    args = parser.parse_args()
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_style()

    records: list[FigureRecord] = []
    dataset_level = read_csv(PROBE4_DATASET)
    dataset_alg = read_csv(PROBE4_DATASET_ALG)
    dual = read_csv(DUAL_PROBE)
    candidate = read_csv(STAGE1_CANDIDATE)
    e1 = read_csv(E1_12)
    alg = read_csv(PROBE4_ALG)
    combos = read_csv(PROBE4_COMBO)
    pair = read_csv(PROBE4_PAIR)
    core50 = read_csv(CORE50)

    plot_pipeline(records)
    plot_reservoir_composition(dataset_level, records)
    plot_dual_probe(dual, records)
    plot_candidate_composition(candidate, records)
    plot_probe4_selection(e1, alg, combos, pair, records)
    baseline, _ = plot_core50_validity(dataset_level, dataset_alg, core50, records)
    plot_tradeoffs_and_symbolic(records)
    plot_search_dynamics(records)
    plot_noise_stability_metric_design(records)
    plot_ablation_summary(baseline, records)
    write_report(records)
    print(json.dumps({"outdir": str(OUTDIR.resolve()), "figures": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
