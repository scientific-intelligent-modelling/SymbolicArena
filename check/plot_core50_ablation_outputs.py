#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = REPO_ROOT / "check"
sys.path.insert(0, str(CHECK_ROOT))

import plot_paper_experiment_figures as base  # noqa: E402


OUTDIR = base.OUTDIR
PAPER_ROOT = REPO_ROOT / "paper" / "Paper-SRInfra"
PAPER_IMG_DIR = PAPER_ROOT / "imgs" / "Article"
PAPER_TABLE_DIR = PAPER_ROOT / "tables"


def _display_name(name: str) -> str:
    mapping = {
        "Core-50": "Core-50",
        "top-info-50": "Top-info",
        "metadata-diverse-50": "Metadata-diverse",
        "difficulty-balanced-50": "Difficulty-balanced",
        "response-kmedoids-50": "Response-medoids",
        "random-50 avg": "Random avg",
        "family-stratified random-50 avg": "Family-random avg",
    }
    return mapping.get(name, name)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _response_matrix(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    methods = sorted(dataset_alg["method_norm"].astype(str).unique())
    ids = sorted(dataset_level["dataset_id"].astype(str).unique())
    rows: list[list[float]] = []
    for dataset_id in ids:
        group = dataset_alg[dataset_alg["dataset_id"].astype(str) == dataset_id]
        by_method = {str(row["method_norm"]): row for _, row in group.iterrows()}
        values: list[float] = []
        for method in methods:
            row = by_method.get(method)
            if row is None:
                values.extend([12.0, 12.0, 0.0, 12.0])
                continue
            values.extend(
                [
                    _safe_float(row.get("median_log_id_nmse"), 12.0),
                    _safe_float(row.get("median_log_ood_nmse"), 12.0),
                    _safe_float(row.get("valid_rate"), 0.0),
                    _safe_float(row.get("iqr_log_ood_nmse"), 12.0),
                ]
            )
        rows.append(values)
    mat = np.asarray(rows, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=12.0, neginf=-12.0)
    std = mat.std(axis=0)
    std[std == 0] = 1.0
    mat = (mat - mat.mean(axis=0)) / std
    return ids, mat


def _pairwise_dist(mat: np.ndarray) -> np.ndarray:
    sq = np.sum(mat * mat, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * mat @ mat.T, 0.0)
    return np.sqrt(dist2)


def _response_kmedoids_ids(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, k: int = 50) -> set[str]:
    """Deterministic response-space medoids baseline.

    The initializer is farthest-first in the four-probe response space, followed by
    a small PAM-style swap refinement. This is a diagnostic baseline, not the
    constrained Core-50 selector.
    """
    ids, mat = _response_matrix(dataset_level, dataset_alg)
    dist = _pairwise_dist(mat)
    centroid = mat.mean(axis=0, keepdims=True)
    selected: list[int] = [int(np.argmin(np.linalg.norm(mat - centroid, axis=1)))]
    nearest = dist[selected[0]].copy()
    while len(selected) < k:
        idx = int(np.argmax(nearest))
        selected.append(idx)
        nearest = np.minimum(nearest, dist[idx])

    selected_set = set(selected)
    for _ in range(3):
        selected_arr = np.asarray(sorted(selected_set), dtype=int)
        dsel = dist[:, selected_arr]
        order = np.argsort(dsel, axis=1)
        nearest_idx = selected_arr[order[:, 0]]
        nearest_dist = dsel[np.arange(len(ids)), order[:, 0]]
        second_dist = dsel[np.arange(len(ids)), order[:, 1]] if len(selected_arr) > 1 else np.full(len(ids), np.inf)
        current_cost = float(nearest_dist.sum())
        best_gain = 0.0
        best_swap: tuple[int, int] | None = None
        non_selected = [i for i in range(len(ids)) if i not in selected_set]
        for old in selected_arr:
            base_without_old = np.where(nearest_idx == old, second_dist, nearest_dist)
            for new in non_selected:
                new_cost = float(np.minimum(base_without_old, dist[:, new]).sum())
                gain = current_cost - new_cost
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (int(old), int(new))
        if best_swap is None:
            break
        selected_set.remove(best_swap[0])
        selected_set.add(best_swap[1])
    return {ids[i] for i in selected_set}


def _load_or_build_subsets(dataset_level: pd.DataFrame, dataset_alg: pd.DataFrame, core50: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rng = np.random.default_rng(20260505)
    baseline, subsets = base.build_core_baselines(dataset_level, dataset_alg, core50, rng)
    full_scores = base.method_scores_for_subset(dataset_alg, set(dataset_level["dataset_id"].astype(str)))
    kmedoids = _response_kmedoids_ids(dataset_level, dataset_alg)
    krow = base.subset_metrics(dataset_level, dataset_alg, kmedoids, "response-kmedoids-50", full_scores)
    baseline = pd.concat([baseline, pd.DataFrame([krow])], ignore_index=True)
    subsets["response-kmedoids-50"] = kmedoids
    return baseline, subsets


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _membership_rows(dataset_level: pd.DataFrame, subsets: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, ids in sorted(subsets.items()):
        if subset == "random_example":
            continue
        sub = dataset_level[dataset_level["dataset_id"].astype(str).isin(ids)].copy()
        sub = sub.sort_values(["family", "subgroup", "dataset_name", "dataset_id"])
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            rows.append(
                {
                    "subset": subset,
                    "rank_in_subset_sorted": rank,
                    "dataset_id": row.get("dataset_id", ""),
                    "global_index": row.get("global_index", ""),
                    "dataset_name": row.get("dataset_name", ""),
                    "family": row.get("family", ""),
                    "subgroup": row.get("subgroup", ""),
                    "dataset_rel": row.get("dataset_rel", ""),
                    "semantic_duplicate_group": row.get("semantic_duplicate_group", ""),
                    "difficulty_bin": row.get("difficulty_bin", ""),
                    "failure_mode": row.get("failure_mode", ""),
                    "winner_probe": row.get("winner_probe", ""),
                    "info_score": row.get("info_score", ""),
                    "stability_score": row.get("stability_score", ""),
                    "difficulty_score": row.get("difficulty_score", ""),
                }
            )
    return rows


def _relative_rows(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    core = metrics[metrics["subset"] == "Core-50"].iloc[0]
    core_error = float(core["aggregate_error"])
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        err = float(row["aggregate_error"])
        if row["subset"] == "Core-50":
            reduction = 0.0
        else:
            reduction = (err - core_error) / err if err > 0 else 0.0
        rows.append(
            {
                "baseline": row["subset"],
                "aggregate_error": err,
                "core50_error_reduction_vs_baseline": reduction,
                "mean_info_delta_vs_core50": float(row["mean_info"]) - float(core["mean_info"]),
                "hard_constraint_violations": float(row.get("hard_constraint_violations", 0.0)),
            }
        )
    return rows


def _family_distribution(dataset_level: pd.DataFrame, subsets: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = ["Core-50", "top-info-50", "metadata-diverse-50", "response-kmedoids-50", "difficulty-balanced-50"]
    for subset in selected:
        ids = subsets[subset]
        sub = dataset_level[dataset_level["dataset_id"].astype(str).isin(ids)]
        counts = sub["family"].astype(str).value_counts()
        for family, count in counts.items():
            rows.append({"subset": subset, "family": family, "count": int(count), "fraction": float(count / len(sub))})
    return pd.DataFrame(rows)


def _plot_ablation(metrics: pd.DataFrame, family_dist: pd.DataFrame) -> list[str]:
    order = [
        "Core-50",
        "random-50 avg",
        "family-stratified random-50 avg",
        "metadata-diverse-50",
        "response-kmedoids-50",
        "top-info-50",
        "difficulty-balanced-50",
    ]
    plot_df = metrics[metrics["subset"].isin(order)].copy()
    plot_df["subset"] = pd.Categorical(plot_df["subset"], order, ordered=True)
    plot_df = plot_df.sort_values("subset")
    labels = [_display_name(str(x)) for x in plot_df["subset"]]
    colors = ["#d65f3a" if str(x) == "Core-50" else "#8aa6b2" for x in plot_df["subset"]]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), gridspec_kw={"height_ratios": [1.0, 1.05]})
    ax = axes[0, 0]
    ax.barh(labels, plot_df["aggregate_error"], color=colors, edgecolor="#333333", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate score MAE to full reservoir (lower is better)")
    ax.set_title("Fidelity error")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[0, 1]
    feasible = plot_df["hard_constraint_violations"].fillna(0) <= 0
    for ok, marker, label in [(True, "*", "feasible"), (False, "o", "violates hard constraints")]:
        sub = plot_df[feasible == ok]
        ax.scatter(
            sub["mean_info"],
            sub["aggregate_error"],
            s=170 if ok else 90,
            marker=marker,
            color="#d65f3a" if ok else "#8aa6b2",
            edgecolor="#222222",
            linewidth=0.8,
            label=label,
            alpha=0.95,
        )
        for _, row in sub.iterrows():
            ax.annotate(_display_name(str(row["subset"])), (row["mean_info"], row["aggregate_error"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Mean information")
    ax.set_ylabel("Aggregate score MAE")
    ax.set_title("Information vs. fidelity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    x = np.arange(len(plot_df))
    width = 0.37
    ax.bar(x - width / 2, plot_df["raw_selection_score"], width, label="raw", color="#6f95b8")
    ax.bar(x + width / 2, plot_df["feasible_selection_score"], width, label="feasible", color="#d65f3a")
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Selection score")
    ax.set_title("Hard-constraint gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    family_order = sorted(family_dist["family"].unique())
    bottom = np.zeros(5)
    subset_order = ["Core-50", "top-info-50", "metadata-diverse-50", "response-kmedoids-50", "difficulty-balanced-50"]
    family_colors = plt.cm.tab20(np.linspace(0, 1, len(family_order)))
    for color, family in zip(family_colors, family_order):
        vals = []
        for subset in subset_order:
            match = family_dist[(family_dist["subset"] == subset) & (family_dist["family"] == family)]
            vals.append(float(match["count"].iloc[0]) if not match.empty else 0.0)
        ax.bar([_display_name(s) for s in subset_order], vals, bottom=bottom, label=family, color=color, linewidth=0)
        bottom += np.asarray(vals)
    ax.set_ylabel("Datasets")
    ax.set_title("Family concentration")
    ax.tick_params(axis="x", rotation=24)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle("Core-50 ablation: feasible distillation beats single-axis selectors", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return base.save(fig, OUTDIR / "figure32_core50_ablation_tradeoff.png")


def _latex_table(metrics: pd.DataFrame) -> str:
    order = [
        "Core-50",
        "random-50 avg",
        "family-stratified random-50 avg",
        "metadata-diverse-50",
        "response-kmedoids-50",
        "top-info-50",
        "difficulty-balanced-50",
    ]
    df = metrics[metrics["subset"].isin(order)].copy()
    df["subset"] = pd.Categorical(df["subset"], order, ordered=True)
    df = df.sort_values("subset")
    best_info = df["mean_info"].max()
    best_stability = df["stability"].max()
    best_error = df["aggregate_error"].min()
    best_feasible = df["feasible_selection_score"].max()

    def fmt(value: float, best: float | None = None, higher: bool = True) -> str:
        text = f"{value:.3f}"
        if best is not None:
            if (higher and abs(value - best) < 5e-4) or ((not higher) and abs(value - best) < 5e-4):
                return "\\textbf{" + text + "}"
        return text

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Core-50 ablation against deterministic and random 50-task selectors. Lower is better for aggregate error and hard violations; higher is better for the remaining score columns.}",
        "\\label{tab:core50-ablation-summary}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Subset} & \\textbf{Coverage} & \\textbf{Mean info} & \\textbf{Stability} & \\textbf{Aggregate error} & \\textbf{Hard viol.} & \\textbf{Feasible score} \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        subset = "\\core{}" if row["subset"] == "Core-50" else _display_name(str(row["subset"]))
        lines.append(
            f"{subset} & "
            f"{float(row['coverage']):.3f} & "
            f"{fmt(float(row['mean_info']), best_info, True)} & "
            f"{fmt(float(row['stability']), best_stability, True)} & "
            f"{fmt(float(row['aggregate_error']), best_error, False)} & "
            f"{float(row['hard_constraint_violations']):.1f} & "
            f"{fmt(float(row['feasible_selection_score']), best_feasible, True)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _markdown_table(metrics: pd.DataFrame) -> str:
    order = [
        "Core-50",
        "random-50 avg",
        "family-stratified random-50 avg",
        "metadata-diverse-50",
        "response-kmedoids-50",
        "top-info-50",
        "difficulty-balanced-50",
    ]
    df = metrics[metrics["subset"].isin(order)].copy()
    df["subset"] = pd.Categorical(df["subset"], order, ordered=True)
    df = df.sort_values("subset")
    lines = [
        "# Core-50 ablation summary",
        "",
        "| Subset | Coverage | Mean info | Stability | Aggregate error | Hard viol. | Feasible score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {_display_name(str(row['subset']))} | "
            f"{float(row['coverage']):.3f} | "
            f"{float(row['mean_info']):.3f} | "
            f"{float(row['stability']):.3f} | "
            f"{float(row['aggregate_error']):.3f} | "
            f"{float(row['hard_constraint_violations']):.1f} | "
            f"{float(row['feasible_selection_score']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    dataset_level = pd.read_csv(base.PROBE4_DATASET)
    dataset_alg = pd.read_csv(base.PROBE4_DATASET_ALG)
    core50 = pd.read_csv(base.CORE50)
    metrics, subsets = _load_or_build_subsets(dataset_level, dataset_alg, core50)
    metrics = metrics.copy()
    metrics["radar_mean_for_plot_only"] = metrics[["coverage", "mean_info", "rank_fidelity", "stability", "non_redundancy", "difficulty_balance"]].mean(axis=1)
    metrics_path = OUTDIR / "core50_ablation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    membership_rows = _membership_rows(dataset_level, subsets)
    _write_csv(OUTDIR / "core50_ablation_subset_membership.csv", membership_rows)

    relative_rows = _relative_rows(metrics)
    _write_csv(OUTDIR / "core50_ablation_relative_improvement.csv", relative_rows)

    family_dist = _family_distribution(dataset_level, subsets)
    family_path = OUTDIR / "core50_ablation_family_distribution.csv"
    family_dist.to_csv(family_path, index=False)

    files = _plot_ablation(metrics, family_dist)
    for name in files:
        shutil.copy2(OUTDIR / name, PAPER_IMG_DIR / name)

    table_tex = _latex_table(metrics)
    table_path = PAPER_TABLE_DIR / "table19_core50_ablation_summary.tex"
    table_path.write_text(table_tex, encoding="utf-8")
    metrics.to_csv(PAPER_TABLE_DIR / "table19_core50_ablation_summary.csv", index=False)
    (PAPER_TABLE_DIR / "table19_core50_ablation_summary.md").write_text(_markdown_table(metrics), encoding="utf-8")

    md_lines = [
        "# Core-50 ablation outputs",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Membership: `{OUTDIR / 'core50_ablation_subset_membership.csv'}`",
        f"- Relative improvement: `{OUTDIR / 'core50_ablation_relative_improvement.csv'}`",
        f"- Family distribution: `{family_path}`",
        f"- Figure: `{OUTDIR / 'figure32_core50_ablation_tradeoff.png'}`",
        f"- Paper table: `{table_path}`",
        "",
        "Key point: Core-50 is not the maximum-information subset; it is the only compared deterministic subset here that passes the hard-constraint gate while retaining the lowest aggregate score error to the full reservoir.",
    ]
    (OUTDIR / "core50_ablation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print({"metrics": str(metrics_path), "figure_files": files, "paper_table": str(table_path)})


if __name__ == "__main__":
    main()
