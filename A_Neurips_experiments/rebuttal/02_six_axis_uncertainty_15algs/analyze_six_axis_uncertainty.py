#!/usr/bin/env python3
"""Generate public rebuttal uncertainty and paired-comparison artifacts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
PRIMARY_COMPONENTS_CSV = OUT_DIR / "six_axis_task_components_15algs.csv"
MATCHED3_COMPONENTS_CSV = OUT_DIR / "matched3_task_components_15algs.csv"

ADDED_ALGORITHMS = ("fepysr", "jaxsr", "symbolfit")
AXES = ("ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB")
MATCHED3_AXES = ("ID_Q", "OOD_G", "SYM_F", "EFF", "ROB")
COMPONENT_COLUMNS = {
    "ID_Q": "ID_Q_component",
    "OOD_G": "OOD_G_component",
    "SYM_F": "SYM_F_component",
    "EFF": "EFF_component",
    "ROB": "ROB_component",
    "STAB": "STAB_component",
}
AXIS_LABELS = {
    "ID_Q": "ID-Q",
    "OOD_G": "OOD-G",
    "SYM_F": "SYM-F",
    "EFF": "EFF",
    "ROB": "ROBU",
    "STAB": "STAB",
}
DISPLAY_NAMES = {
    "drsr": "DRSR",
    "dso": "DSO",
    "e2esr": "E2ESR",
    "fepysr": "FePySR",
    "gplearn": "gplearn",
    "imcts": "iMCTS",
    "jaxsr": "JAXSR",
    "llmsr": "LLM-SR",
    "pyoperon": "PyOperon",
    "pysr": "PySR",
    "qlattice": "QLattice",
    "ragsr": "RAG-SR",
    "symbolfit": "SymbolFit",
    "tpsr": "TPSR",
    "udsr": "uDSR",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_components(
    components: pd.DataFrame,
    axes: Iterable[str],
    *,
    expected_algorithms: int = 15,
    expected_datasets: int = 50,
) -> pd.DataFrame:
    output = components.copy()
    required = {
        "algorithm",
        "dataset",
        "cohort",
        "clean_seeds",
        "noise_levels",
        *[COMPONENT_COLUMNS[axis] for axis in axes],
    }
    missing = required - set(output.columns)
    if missing:
        raise ValueError(f"component columns missing: {sorted(missing)}")
    output["algorithm"] = output["algorithm"].astype(str).str.lower()
    output["cohort"] = output["algorithm"].map(
        lambda algorithm: (
            "Rebuttal-added" if algorithm in ADDED_ALGORITHMS else "Original"
        )
    )
    for axis in axes:
        column = COMPONENT_COLUMNS[axis]
        output[column] = pd.to_numeric(output[column], errors="raise")
        if not output[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains values outside [0, 1]")
    counts = output.groupby("algorithm")["dataset"].nunique()
    if len(counts) != expected_algorithms or not counts.eq(expected_datasets).all():
        raise ValueError(f"invalid algorithm/dataset grid: {counts.to_dict()}")
    if output["dataset"].nunique() != expected_datasets:
        raise ValueError("methods do not share the expected task set")
    if output.duplicated(["algorithm", "dataset"]).any():
        raise ValueError("duplicate algorithm/dataset component rows")
    return output


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_and_pairwise(
    components: pd.DataFrame,
    axes: Iterable[str],
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = tuple(axes)
    algorithms = sorted(components["algorithm"].unique())
    datasets = sorted(components["dataset"].unique())
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0,
        len(datasets),
        size=(n_bootstrap, len(datasets)),
    )
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_permutations, len(datasets)),
    )
    means_rows: dict[str, dict[str, Any]] = {
        algorithm: {
            "algorithm": algorithm,
            "display_name": DISPLAY_NAMES[algorithm],
            "cohort": (
                "Rebuttal-added" if algorithm in ADDED_ALGORITHMS else "Original"
            ),
        }
        for algorithm in algorithms
    }
    pairwise_rows: list[dict[str, Any]] = []

    for axis in axes:
        column = COMPONENT_COLUMNS[axis]
        matrix = (
            components.pivot(
                index="algorithm",
                columns="dataset",
                values=column,
            )
            .reindex(index=algorithms, columns=datasets)
            .to_numpy(dtype=float)
            * 100.0
        )
        if np.isnan(matrix).any():
            raise ValueError(f"{axis}: incomplete algorithm/task matrix")
        bootstrap_means = np.stack(
            [values[bootstrap_indices].mean(axis=1) for values in matrix]
        )
        for index, algorithm in enumerate(algorithms):
            means_rows[algorithm][axis] = float(matrix[index].mean())
            means_rows[algorithm][f"{axis}_ci_low"] = float(
                np.percentile(bootstrap_means[index], 2.5)
            )
            means_rows[algorithm][f"{axis}_ci_high"] = float(
                np.percentile(bootstrap_means[index], 97.5)
            )

        axis_rows: list[dict[str, Any]] = []
        p_values: list[float] = []
        for left_index, right_index in itertools.combinations(
            range(len(algorithms)),
            2,
        ):
            difference = matrix[left_index] - matrix[right_index]
            observed = float(difference.mean())
            bootstrap_difference = (
                bootstrap_means[left_index] - bootstrap_means[right_index]
            )
            ci_low = float(np.percentile(bootstrap_difference, 2.5))
            ci_high = float(np.percentile(bootstrap_difference, 97.5))
            null_distribution = signs @ difference / len(datasets)
            p_value = float(
                (np.count_nonzero(np.abs(null_distribution) >= abs(observed)) + 1)
                / (n_permutations + 1)
            )
            row = {
                "axis": axis,
                "algorithm_a": algorithms[left_index],
                "algorithm_b": algorithms[right_index],
                "difference_a_minus_b": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_a_gt_b": float(
                    np.mean(bootstrap_difference > 0)
                ),
                "paired_sign_flip_p": p_value,
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "tasks": len(datasets),
            }
            axis_rows.append(row)
            p_values.append(p_value)
        adjusted = holm_adjust(p_values)
        for row, p_adjusted in zip(axis_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = p_adjusted
            row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
            row["statistically_distinguishable"] = bool(
                row["ci_excludes_zero"] and p_adjusted < 0.05
            )
        pairwise_rows.extend(axis_rows)

    means = pd.DataFrame(means_rows.values())
    means["axis_mean"] = means[list(axes)].mean(axis=1)
    return means, pd.DataFrame(pairwise_rows)


def invert_shared_y_axis(axes: Any) -> None:
    """共享 y 轴只反转一次，避免子图重复调用相互抵消。"""
    axes.ravel()[0].invert_yaxis()


def save_figure_variants(figure: Any, output: Path) -> None:
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "SymbolicArena rebuttal analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        output.with_suffix(".jpg"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )


def plot_uncertainty(means: pd.DataFrame, output: Path) -> None:
    order = means.sort_values(
        ["axis_mean", "algorithm"],
        ascending=[False, True],
    )["algorithm"].tolist()
    indexed = means.set_index("algorithm").loc[order]
    y_positions = np.arange(len(order))
    original_color = "#236782"
    added_color = "#D26932"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 10.2),
        sharex=True,
        sharey=True,
    )
    for axis_name, axis_plot in zip(AXES, axes.ravel(), strict=True):
        for position, algorithm in enumerate(order):
            row = indexed.loc[algorithm]
            added = algorithm in ADDED_ALGORITHMS
            color = added_color if added else original_color
            marker = "D" if added else "o"
            value = float(row[axis_name])
            low = float(row[f"{axis_name}_ci_low"])
            high = float(row[f"{axis_name}_ci_high"])
            axis_plot.errorbar(
                value,
                position,
                xerr=[[value - low], [high - value]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=2.2,
                capthick=1.0,
                markersize=4.8 if added else 4.4,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
            )
        axis_plot.set_title(AXIS_LABELS[axis_name])
        axis_plot.set_xlim(0, 100)
        axis_plot.set_xticks([0, 20, 40, 60, 80, 100])
        axis_plot.set_yticks(
            y_positions,
            [DISPLAY_NAMES[algorithm] for algorithm in order],
        )
        axis_plot.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        axis_plot.set_axisbelow(True)
        axis_plot.set_xlabel("Score")
    invert_shared_y_axis(axes)

    figure.suptitle(
        "Six-axis task-bootstrap uncertainty across 15 SR methods",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=original_color,
                markeredgecolor="white",
                markersize=7,
                label="Original 12",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=added_color,
                markeredgecolor="white",
                markersize=7,
                label="Rebuttal-added 3",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    figure.text(
        0.5,
        0.021,
        "Points are Core-50 task means; bars are 95% intervals from 20,000 "
        "synchronized task-bootstrap resamples.",
        ha="center",
        fontsize=9,
    )
    figure.text(
        0.5,
        0.008,
        "Marginal intervals are uncertainty diagnostics; algorithm "
        "comparisons use paired differences reported separately.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.text(
        0.5,
        -0.005,
        "All scores use one-hour evaluations. These intervals quantify task "
        "sampling uncertainty, not seed-resampling uncertainty.",
        ha="center",
        fontsize=8.3,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.91,
        bottom=0.095,
        wspace=0.17,
        hspace=0.24,
    )
    save_figure_variants(figure, output)
    plt.close(figure)


def oriented_pair(
    pairwise: pd.DataFrame,
    axis: str,
    algorithm_a: str,
    algorithm_b: str,
) -> dict[str, Any]:
    selected = pairwise[
        (pairwise["axis"] == axis)
        & (
            (
                (pairwise["algorithm_a"] == algorithm_a)
                & (pairwise["algorithm_b"] == algorithm_b)
            )
            | (
                (pairwise["algorithm_a"] == algorithm_b)
                & (pairwise["algorithm_b"] == algorithm_a)
            )
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{axis}: pair {algorithm_a}/{algorithm_b} rows={len(selected)}"
        )
    row = selected.iloc[0]
    same_order = row["algorithm_a"] == algorithm_a
    difference = float(row["difference_a_minus_b"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    probability = float(row["bootstrap_probability_a_gt_b"])
    if not same_order:
        difference = -difference
        ci_low, ci_high = -ci_high, -ci_low
        probability = 1.0 - probability
    return {
        "axis": axis,
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "difference_a_minus_b": difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_probability_a_gt_b": probability,
        "paired_sign_flip_p": float(row["paired_sign_flip_p"]),
        "holm_adjusted_p": float(row["holm_adjusted_p"]),
        "ci_excludes_zero": bool(row["ci_excludes_zero"]),
        "statistically_distinguishable": bool(row["statistically_distinguishable"]),
        "tasks": int(row["tasks"]),
    }


def build_compact_pairwise_tables(
    means: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    for axis in AXES:
        ranking = means.sort_values(
            [axis, "algorithm"],
            ascending=[False, True],
        )["algorithm"].tolist()
        for rank_a, rank_b in itertools.combinations(range(5), 2):
            row = oriented_pair(
                pairwise,
                axis,
                ranking[rank_a],
                ranking[rank_b],
            )
            row.update(
                {
                    "rank_a": rank_a + 1,
                    "rank_b": rank_b + 1,
                    "display_a": DISPLAY_NAMES[ranking[rank_a]],
                    "display_b": DISPLAY_NAMES[ranking[rank_b]],
                    "conclusion": (
                        "distinguishable"
                        if row["statistically_distinguishable"]
                        else "not_distinguished"
                    ),
                }
            )
            top_rows.append(row)
        for rank_a in range(len(ranking) - 1):
            rank_b = rank_a + 1
            row = oriented_pair(
                pairwise,
                axis,
                ranking[rank_a],
                ranking[rank_b],
            )
            row.update(
                {
                    "rank_a": rank_a + 1,
                    "rank_b": rank_b + 1,
                    "display_a": DISPLAY_NAMES[ranking[rank_a]],
                    "display_b": DISPLAY_NAMES[ranking[rank_b]],
                    "conclusion": (
                        "distinguishable"
                        if row["statistically_distinguishable"]
                        else "not_distinguished"
                    ),
                }
            )
            adjacent_rows.append(row)
    return pd.DataFrame(top_rows), pd.DataFrame(adjacent_rows)


def plot_top5_matrix(
    means: pd.DataFrame,
    pairwise: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 9.8))
    color_map = ListedColormap(["#C96B54", "#E6E6E6", "#2C7F86"])
    for axis_name, axis_plot in zip(AXES, axes.ravel(), strict=True):
        ranking = means.sort_values(
            [axis_name, "algorithm"],
            ascending=[False, True],
        )[
            "algorithm"
        ].tolist()[:5]
        codes = np.zeros((5, 5), dtype=float)
        labels = np.full((5, 5), "", dtype=object)
        for row_index, row_algorithm in enumerate(ranking):
            for column_index, column_algorithm in enumerate(ranking):
                if row_index == column_index:
                    labels[row_index, column_index] = "--"
                    continue
                result = oriented_pair(
                    pairwise,
                    axis_name,
                    row_algorithm,
                    column_algorithm,
                )
                difference = result["difference_a_minus_b"]
                if result["statistically_distinguishable"]:
                    codes[row_index, column_index] = 1.0 if difference > 0 else -1.0
                    marker = "*"
                else:
                    codes[row_index, column_index] = 0.0
                    marker = ""
                labels[row_index, column_index] = f"{difference:+.1f}{marker}"
        axis_plot.imshow(
            codes,
            cmap=color_map,
            vmin=-1,
            vmax=1,
            aspect="equal",
        )
        display = [DISPLAY_NAMES[algorithm] for algorithm in ranking]
        axis_plot.set_xticks(range(5), display, rotation=42, ha="right")
        axis_plot.set_yticks(range(5), display)
        axis_plot.set_title(AXIS_LABELS[axis_name])
        axis_plot.tick_params(length=0)
        axis_plot.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        axis_plot.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        axis_plot.grid(
            which="minor",
            color="white",
            linewidth=1.4,
        )
        axis_plot.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(5):
            for column_index in range(5):
                color = (
                    "white" if abs(codes[row_index, column_index]) == 1 else "#222222"
                )
                axis_plot.text(
                    column_index,
                    row_index,
                    labels[row_index, column_index],
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=color,
                    fontweight=(
                        "bold" if abs(codes[row_index, column_index]) == 1 else "normal"
                    ),
                )
    figure.suptitle(
        "Top-5 paired score differences by evaluation axis",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=[
            Patch(facecolor="#2C7F86", label="Row method higher"),
            Patch(facecolor="#C96B54", label="Row method lower"),
            Patch(facecolor="#E6E6E6", label="Not distinguished"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        "Cells report row-minus-column score differences. * requires both a "
        "paired 95% bootstrap interval excluding zero",
        ha="center",
        fontsize=8.8,
    )
    figure.text(
        0.5,
        0.005,
        "and a Holm-adjusted two-sided task-level sign-flip p-value below "
        "0.05 across all 105 comparisons on that axis.",
        ha="center",
        fontsize=8.8,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.89,
        bottom=0.09,
        wspace=0.28,
        hspace=0.38,
    )
    save_figure_variants(figure, output)
    plt.close(figure)


def matched3_sensitivity(
    primary_means: pd.DataFrame,
    matched3_means: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    primary_indexed = primary_means.set_index("algorithm")
    matched_indexed = matched3_means.set_index("algorithm")
    algorithms = sorted(primary_indexed.index)
    for axis in MATCHED3_AXES:
        primary_values = primary_indexed.loc[algorithms, axis]
        matched_values = matched_indexed.loc[algorithms, axis]
        primary_ranks = primary_values.rank(
            ascending=False,
            method="min",
        )
        matched_ranks = matched_values.rank(
            ascending=False,
            method="min",
        )
        rank_correlation = float(np.corrcoef(primary_ranks, matched_ranks)[0, 1])
        deltas = matched_values - primary_values
        primary_top3 = set(primary_values.nlargest(3).index)
        matched_top3 = set(matched_values.nlargest(3).index)
        summary_rows.append(
            {
                "axis": axis,
                "rank_correlation": rank_correlation,
                "mean_absolute_score_change": float(deltas.abs().mean()),
                "max_absolute_score_change": float(deltas.abs().max()),
                "top1_primary": primary_values.idxmax(),
                "top1_matched3": matched_values.idxmax(),
                "top1_unchanged": bool(
                    primary_values.idxmax() == matched_values.idxmax()
                ),
                "top3_overlap": len(primary_top3 & matched_top3),
            }
        )
        for algorithm in algorithms:
            detail_rows.append(
                {
                    "axis": axis,
                    "algorithm": algorithm,
                    "display_name": DISPLAY_NAMES[algorithm],
                    "cohort": (
                        "Rebuttal-added"
                        if algorithm in ADDED_ALGORITHMS
                        else "Original"
                    ),
                    "primary_score": float(primary_values[algorithm]),
                    "matched3_score": float(matched_values[algorithm]),
                    "score_change": float(deltas[algorithm]),
                    "primary_rank": int(primary_ranks[algorithm]),
                    "matched3_rank": int(matched_ranks[algorithm]),
                    "rank_change": int(
                        primary_ranks[algorithm] - matched_ranks[algorithm]
                    ),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def plot_matched3_sensitivity(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.8))
    original_color = "#236782"
    added_color = "#D26932"
    for axis_name, axis_plot in zip(
        MATCHED3_AXES,
        axes.ravel(),
        strict=False,
    ):
        selected = detail[detail["axis"] == axis_name]
        for cohort, color, marker in (
            ("Original", original_color, "o"),
            ("Rebuttal-added", added_color, "D"),
        ):
            group = selected[selected["cohort"] == cohort]
            axis_plot.scatter(
                group["primary_score"],
                group["matched3_score"],
                s=34,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.6,
                label=cohort,
                zorder=3,
            )
        lower = float(
            min(
                selected["primary_score"].min(),
                selected["matched3_score"].min(),
            )
        )
        upper = float(
            max(
                selected["primary_score"].max(),
                selected["matched3_score"].max(),
            )
        )
        padding = max(2.0, 0.08 * (upper - lower))
        bounds = (max(0.0, lower - padding), min(100.0, upper + padding))
        axis_plot.plot(
            bounds,
            bounds,
            color="#777777",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )
        axis_plot.set_xlim(bounds)
        axis_plot.set_ylim(bounds)
        row = summary[summary["axis"] == axis_name].iloc[0]
        axis_plot.set_title(
            f"{AXIS_LABELS[axis_name]}  " f"(rank r={row['rank_correlation']:.3f})"
        )
        axis_plot.set_xlabel("Primary score")
        axis_plot.set_ylabel("Matched-three-run score")
        axis_plot.grid(color="#E2E2E2", linewidth=0.7)
        axis_plot.set_axisbelow(True)
    axes.ravel()[-1].axis("off")
    figure.suptitle(
        "Matched-three-run sensitivity across 15 SR methods",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=original_color,
                markeredgecolor="white",
                markersize=7,
                label="Original 12",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=added_color,
                markeredgecolor="white",
                markersize=7,
                label="Rebuttal-added 3",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.94, 0.17),
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        "All methods use three runs per task; ROBU uses the two common noise "
        "levels. STAB is excluded because its matched seed-level proxy is not "
        "part of this sensitivity artifact.",
        ha="center",
        fontsize=8.7,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.91,
        bottom=0.09,
        wspace=0.26,
        hspace=0.32,
    )
    save_figure_variants(figure, output)
    plt.close(figure)


def write_public_docs(
    means: pd.DataFrame,
    pairwise: pd.DataFrame,
    adjacent: pd.DataFrame,
    matched_summary: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
) -> None:
    significant = (
        pairwise.groupby("axis")["statistically_distinguishable"]
        .agg(["sum", "count"])
        .reset_index()
    )
    adjacent_counts = (
        adjacent.groupby("axis")["statistically_distinguishable"]
        .agg(["sum", "count"])
        .reset_index()
    )
    top_adjacent_count = int(
        adjacent.loc[
            adjacent["rank_a"] < 5,
            "statistically_distinguishable",
        ].sum()
    )
    score_table = means.sort_values(
        "axis_mean",
        ascending=False,
    )[
        ["display_name", *AXES, "axis_mean"]
    ].to_markdown(index=False, floatfmt=".2f")
    significance_table = significant.to_markdown(index=False)
    adjacent_table = adjacent_counts.to_markdown(index=False)
    sensitivity_table = matched_summary.assign(
        rank_correlation=lambda frame: frame["rank_correlation"].round(3),
        mean_absolute_score_change=lambda frame: frame[
            "mean_absolute_score_change"
        ].round(2),
        max_absolute_score_change=lambda frame: frame[
            "max_absolute_score_change"
        ].round(2),
    ).to_markdown(index=False)
    readme = f"""# Six-axis uncertainty for 15 algorithms

This directory reports task-bootstrap uncertainty and direct paired
algorithm comparisons for the 15-method Core-50 rebuttal evaluation.

## Protocol

- All scores use one-hour evaluations over the same 50 tasks.
- The primary analysis uses five runs for the original 12 methods and three
  runs for the three rebuttal-added methods.
- Marginal 95% intervals use {n_bootstrap:,} synchronized task-bootstrap
  resamples. They quantify task-sampling uncertainty and do not resample seeds.
- Pairwise inference uses the same resampled task set for both methods,
  {n_permutations:,} task-level sign-flip permutations, and Holm correction
  across all 105 method pairs within each axis.
- A pair is called statistically distinguishable only when both the paired
  95% bootstrap interval excludes zero and the Holm-adjusted p-value is below
  0.05.
- The matched-three-run sensitivity uses three runs for every method. ROBU
  uses the two common noise levels. STAB is excluded from this sensitivity
  because a matched seed-level structural proxy is not included in the
  sensitivity artifact.

Overlapping marginal intervals are diagnostics only. The paired-difference
tables and matrix are the authoritative outputs for algorithm comparisons.

## Primary scores

{score_table}

## Distinguishable pair counts

{significance_table}

## Adjacent-ranking result

{adjacent_table}

Only {top_adjacent_count} of the 24 adjacent comparisons across the six
axis-specific Top-5 rankings are statistically distinguishable.

## Matched-three-run sensitivity

{sensitivity_table}

## Public outputs

- `six_axis_uncertainty_15algs.png/.pdf/.jpg`
- `pairwise_significance_top5.png/.pdf/.jpg`
- `matched3_sensitivity.png/.pdf/.jpg`
- `six_axis_task_components_15algs.csv`
- `matched3_task_components_15algs.csv`
- `six_axis_means_ci_15algs.csv`
- `six_axis_pairwise_inference.csv`
- `top5_pairwise_inference.csv`
- `adjacent_pairwise_inference.csv`
- `matched3_means_ci_15algs.csv`
- `matched3_sensitivity_detail.csv`
- `matched3_sensitivity_summary.csv`
- `reviewer_p5tg_statistical_significance.md`
- `analysis_summary.json`
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    rank_min = float(matched_summary["rank_correlation"].min())
    rank_max = float(matched_summary["rank_correlation"].max())
    max_shift = float(matched_summary["max_absolute_score_change"].max())
    counts = {row["axis"]: int(row["sum"]) for _, row in significant.iterrows()}
    adjacent_by_axis = {
        row["axis"]: int(row["sum"]) for _, row in adjacent_counts.iterrows()
    }
    response = f"""# Response to Reviewer p5tG: statistical significance

## Rebuttal-ready English response

Thank you for pointing this out. We agree that point estimates and rankings
alone do not establish whether small score differences are meaningful. We
have therefore added uncertainty estimates based on {n_bootstrap:,}
synchronized bootstrap resamples of the 50 Core tasks. The new six-panel
figure reports task-level mean scores and marginal 95% bootstrap intervals
for all six axes. These intervals quantify task-sampling uncertainty rather
than seed-resampling uncertainty.

To directly assess algorithm-to-algorithm differences, we additionally
compute paired bootstrap intervals for

`Delta(a,b) = S(a) - S(b),`

using the same resampled task set for both methods. We complement these
intervals with two-sided task-level sign-flip tests ({n_permutations:,}
resamples) and apply Holm correction to all 105 method pairs within each
axis. We call a pair statistically distinguishable only when the paired 95%
interval excludes zero and the Holm-adjusted p-value is below 0.05.
Accordingly, only {min(counts.values())}--{max(counts.values())} of 105 pairs
are distinguishable on any axis (ID-Q: {counts['ID_Q']}, OOD-G:
{counts['OOD_G']}, SYM-F: {counts['SYM_F']}, EFF: {counts['EFF']}, ROBU:
{counts['ROB']}, STAB: {counts['STAB']}). We therefore avoid interpreting
the remaining small score gaps as definitive orderings.

This conclusion is even clearer for neighboring ranks: only
{sum(adjacent_by_axis.values())} of the 84 adjacent-ranking comparisons are
distinguishable after correction (ID-Q: {adjacent_by_axis['ID_Q']}, OOD-G:
{adjacent_by_axis['OOD_G']}, SYM-F: {adjacent_by_axis['SYM_F']}, EFF:
{adjacent_by_axis['EFF']}, ROBU: {adjacent_by_axis['ROB']}, STAB:
{adjacent_by_axis['STAB']}), and none of the adjacent comparisons within the
Top-5 are distinguishable. This directly confirms that small neighboring
score gaps should not be interpreted as statistically resolved rankings.

For readability, the rebuttal shows a Top-5 paired-difference matrix for each
axis and reports all adjacent-ranking comparisons in a companion table. The
complete 630-pair result is also provided. As a fairness check, we repeated
the analysis with exactly three runs per method on the five axes that admit a
matched reconstruction. Rank correlations between the primary and matched
analyses are {rank_min:.3f}--{rank_max:.3f}, and the largest absolute score
change is {max_shift:.2f} points. This sensitivity result supports the main
ordering while making clear where uncertainty remains. ROBU in this check
uses the two common noise levels; STAB is not included because its matched
seed-level structural proxy is not part of the sensitivity artifact.

## Figure caption

**Six-axis score uncertainty across 15 symbolic regression methods.** Points
denote mean scores over Core-50 tasks, and horizontal bars denote marginal
95% intervals obtained from {n_bootstrap:,} synchronized task-level
bootstrap resamples. These bars visualize task-sampling uncertainty;
statistical comparisons are determined separately using paired bootstrap
intervals of algorithm-score differences and Holm-adjusted task-level
sign-flip tests.

## 中文核对

- 主图置信区间只表示任务采样不确定性，不声称覆盖 seed 重采样方差。
- 两算法比较使用同一组重采样任务，保持严格配对。
- “可区分”同时要求差值区间排除 0 且 Holm 校正后 p < 0.05。
- 正文展示每轴 Top-5 矩阵和相邻排名比较，完整 630 组结果保留在 CSV。
- matched-three-run sensitivity 对所有方法统一使用 3 次运行；ROBU 使用两个
  共同噪声水平；STAB 不使用不可严格复算的近似值。
"""
    (OUT_DIR / "reviewer_p5tg_statistical_significance.md").write_text(
        response,
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> None:
    primary_components = validate_components(
        pd.read_csv(PRIMARY_COMPONENTS_CSV),
        AXES,
    )
    matched_components = validate_components(
        pd.read_csv(MATCHED3_COMPONENTS_CSV),
        MATCHED3_AXES,
    )
    primary_means, pairwise = bootstrap_and_pairwise(
        primary_components,
        AXES,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    matched_means, _ = bootstrap_and_pairwise(
        matched_components,
        MATCHED3_AXES,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    top5, adjacent = build_compact_pairwise_tables(
        primary_means,
        pairwise,
    )
    sensitivity_detail, sensitivity_summary = matched3_sensitivity(
        primary_means,
        matched_means,
    )

    primary_means.to_csv(
        OUT_DIR / "six_axis_means_ci_15algs.csv",
        index=False,
    )
    pairwise.to_csv(
        OUT_DIR / "six_axis_pairwise_inference.csv",
        index=False,
    )
    top5.to_csv(OUT_DIR / "top5_pairwise_inference.csv", index=False)
    adjacent.to_csv(
        OUT_DIR / "adjacent_pairwise_inference.csv",
        index=False,
    )
    matched_means.to_csv(
        OUT_DIR / "matched3_means_ci_15algs.csv",
        index=False,
    )
    sensitivity_detail.to_csv(
        OUT_DIR / "matched3_sensitivity_detail.csv",
        index=False,
    )
    sensitivity_summary.to_csv(
        OUT_DIR / "matched3_sensitivity_summary.csv",
        index=False,
    )

    main_figure = OUT_DIR / "six_axis_uncertainty_15algs.png"
    matrix_figure = OUT_DIR / "pairwise_significance_top5.png"
    sensitivity_figure = OUT_DIR / "matched3_sensitivity.png"
    plot_uncertainty(primary_means, main_figure)
    plot_top5_matrix(primary_means, pairwise, matrix_figure)
    plot_matched3_sensitivity(
        sensitivity_detail,
        sensitivity_summary,
        sensitivity_figure,
    )
    write_public_docs(
        primary_means,
        pairwise,
        adjacent,
        sensitivity_summary,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
    )

    summary = {
        "algorithms": int(primary_components["algorithm"].nunique()),
        "tasks": int(primary_components["dataset"].nunique()),
        "primary_task_component_rows": len(primary_components),
        "matched3_task_component_rows": len(matched_components),
        "bootstrap_resamples": args.bootstrap,
        "permutation_resamples": args.permutations,
        "random_seed": args.seed,
        "distinguishable_pairs": {
            axis: int(
                pairwise.loc[
                    pairwise["axis"] == axis,
                    "statistically_distinguishable",
                ].sum()
            )
            for axis in AXES
        },
        "matched3_rank_correlation": {
            row["axis"]: float(row["rank_correlation"])
            for _, row in sensitivity_summary.iterrows()
        },
        "input_sha256": {
            "primary_task_components": sha256_file(PRIMARY_COMPONENTS_CSV),
            "matched3_task_components": sha256_file(MATCHED3_COMPONENTS_CSV),
        },
        "output_sha256": {
            "means_ci": sha256_file(OUT_DIR / "six_axis_means_ci_15algs.csv"),
            "pairwise_inference": sha256_file(
                OUT_DIR / "six_axis_pairwise_inference.csv"
            ),
            "top5_pairwise": sha256_file(OUT_DIR / "top5_pairwise_inference.csv"),
            "adjacent_pairwise": sha256_file(
                OUT_DIR / "adjacent_pairwise_inference.csv"
            ),
            "main_figure_png": sha256_file(main_figure),
            "matrix_figure_png": sha256_file(matrix_figure),
            "sensitivity_figure_png": sha256_file(sensitivity_figure),
        },
        "scope_notes": [
            "Marginal intervals quantify synchronized task-bootstrap uncertainty only.",
            "Primary estimates use five runs for original methods and three runs for rebuttal-added methods.",
            "Matched-three-run sensitivity covers ID-Q, OOD-G, SYM-F, EFF, and ROBU.",
            "Matched-three-run ROBU uses the two common noise levels; STAB is excluded.",
            "EFF retains the archived proxy-component definition in the primary analysis.",
            "STAB uses the formal SYM-F in its performance correction (paper F.5) for the 12 "
            "original methods. The 3 rebuttal-added methods keep their archived STAB because "
            "their per-seed structural inputs were not preserved, so STAB is mixed-provenance "
            "across cohorts and cross-cohort STAB gaps should not be over-interpreted.",
        ],
    }
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate public six-axis rebuttal statistics",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
