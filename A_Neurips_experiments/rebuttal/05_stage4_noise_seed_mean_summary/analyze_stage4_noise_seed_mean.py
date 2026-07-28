#!/usr/bin/env python3
"""整理 NeurIPS Stage 4 clean/noise 多种子均值统计并生成 rebuttal 表格。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from check.plot_core50_hexagon_metrics import phi_from_nmse


REPO_ROOT = _REPO_ROOT_FOR_IMPORT
STAGE4_ROOT = REPO_ROOT / "A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h"
DEFAULT_CLEAN = (
    REPO_ROOT
    / "paper/neurips26-upload/results-artifact/clean-core50/clean_final_runs_updated.csv"
)
DEFAULT_NOISE = STAGE4_ROOT / "noise_robustness/noise_final_runs.csv"
DEFAULT_FORMAL = STAGE4_ROOT / "paper_tables/table13_noise_robustness_summary.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent

CONDITION_ORDER = ["clean", "noise_0.01", "noise_0.05", "noise_0.10"]
CONDITION_LABELS = {
    "clean": "Clean",
    "noise_0.01": "Noise 0.01",
    "noise_0.05": "Noise 0.05",
    "noise_0.10": "Noise 0.10",
}
METRIC_ORDER = [
    "Valid rate (%)",
    "ID-Q mean (%)",
    "OOD-Q mean (%)",
    "Joint-Q mean (%)",
    "Clean-relative retention (%)",
]
FAILURE_NMSE = 1e2
EXPECTED_ALGORITHMS = 12
EXPECTED_DATASETS = 50
EXPECTED_SEEDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--noise-csv", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--formal-robu-csv", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_runs(clean_path: Path, noise_path: Path) -> pd.DataFrame:
    clean = pd.read_csv(clean_path)
    noise = pd.read_csv(noise_path)
    clean["condition"] = "clean"
    clean["noise_sigma"] = 0.0
    noise["noise_sigma"] = pd.to_numeric(noise["noise_sigma"], errors="coerce")
    noise["condition"] = noise["noise_sigma"].map(
        {0.01: "noise_0.01", 0.05: "noise_0.05", 0.1: "noise_0.10"}
    )
    required = {
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "status",
        "valid_output",
        "metric_complete",
        "id_test_nmse",
        "ood_test_nmse",
        "condition",
        "noise_sigma",
    }
    for name, frame in [("clean", clean), ("noise", noise)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 数据缺少字段: {sorted(missing)}")

    runs = pd.concat([clean[list(required)], noise[list(required)]], ignore_index=True)
    runs["valid_output"] = as_bool(runs["valid_output"])
    runs["metric_complete"] = as_bool(runs["metric_complete"])
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    runs["id_test_nmse"] = pd.to_numeric(runs["id_test_nmse"], errors="coerce")
    runs["ood_test_nmse"] = pd.to_numeric(runs["ood_test_nmse"], errors="coerce")
    finite_id = np.isfinite(runs["id_test_nmse"]) & (runs["id_test_nmse"] >= 0)
    finite_ood = np.isfinite(runs["ood_test_nmse"]) & (runs["ood_test_nmse"] >= 0)
    runs["usable_metric"] = (
        runs["valid_output"] & runs["metric_complete"] & finite_id & finite_ood
    )
    runs["id_nmse_used"] = runs["id_test_nmse"].where(runs["usable_metric"], FAILURE_NMSE)
    runs["ood_nmse_used"] = runs["ood_test_nmse"].where(runs["usable_metric"], FAILURE_NMSE)
    runs["id_quality"] = runs["id_nmse_used"].map(phi_from_nmse)
    runs["ood_quality"] = runs["ood_nmse_used"].map(phi_from_nmse)
    runs["joint_quality"] = 0.5 * (runs["id_quality"] + runs["ood_quality"])
    return runs


def audit_grid(runs: pd.DataFrame) -> dict[str, Any]:
    duplicate_keys = int(
        runs.duplicated(["algorithm", "dataset", "seed", "condition"], keep=False).sum()
    )
    per_cell = runs.groupby(["algorithm", "dataset", "condition"]).size()
    audit = {
        "run_count": int(len(runs)),
        "algorithm_count": int(runs["algorithm"].nunique()),
        "dataset_count": int(runs["dataset"].nunique()),
        "seed_count": int(runs["seed"].nunique()),
        "condition_count": int(runs["condition"].nunique()),
        "condition_counts": {
            str(key): int(value) for key, value in runs["condition"].value_counts().items()
        },
        "status_counts": {
            str(key): int(value) for key, value in runs["status"].value_counts().items()
        },
        "valid_output_count": int(runs["valid_output"].sum()),
        "metric_complete_count": int(runs["metric_complete"].sum()),
        "usable_metric_count": int(runs["usable_metric"].sum()),
        "penalized_run_count": int((~runs["usable_metric"]).sum()),
        "duplicate_key_rows": duplicate_keys,
        "algorithm_dataset_condition_cells": int(len(per_cell)),
        "cells_with_exactly_five_seeds": int((per_cell == EXPECTED_SEEDS).sum()),
        "cells_without_exactly_five_seeds": int((per_cell != EXPECTED_SEEDS).sum()),
    }
    expected_runs = (
        EXPECTED_ALGORITHMS * EXPECTED_DATASETS * EXPECTED_SEEDS * len(CONDITION_ORDER)
    )
    failures = []
    if audit["run_count"] != expected_runs:
        failures.append(f"run_count={audit['run_count']}，预期 {expected_runs}")
    if audit["algorithm_count"] != EXPECTED_ALGORITHMS:
        failures.append(f"algorithm_count={audit['algorithm_count']}")
    if audit["dataset_count"] != EXPECTED_DATASETS:
        failures.append(f"dataset_count={audit['dataset_count']}")
    if audit["seed_count"] != EXPECTED_SEEDS:
        failures.append(f"seed_count={audit['seed_count']}")
    if set(runs["condition"]) != set(CONDITION_ORDER):
        failures.append(f"conditions={sorted(runs['condition'].unique())}")
    if duplicate_keys:
        failures.append(f"duplicate_key_rows={duplicate_keys}")
    if audit["cells_without_exactly_five_seeds"]:
        failures.append(
            f"cells_without_exactly_five_seeds={audit['cells_without_exactly_five_seeds']}"
        )
    if failures:
        raise ValueError("Stage 4 网格完整性校验失败: " + "; ".join(failures))
    return audit


def compute_condition_tables(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell = (
        runs.groupby(["algorithm", "dataset", "condition"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            valid_rate=("usable_metric", "mean"),
            id_quality=("id_quality", "mean"),
            ood_quality=("ood_quality", "mean"),
            joint_quality=("joint_quality", "mean"),
        )
    )
    clean_quality = (
        cell.loc[cell["condition"] == "clean", ["algorithm", "dataset", "joint_quality"]]
        .rename(columns={"joint_quality": "clean_joint_quality"})
        .copy()
    )
    cell = cell.merge(clean_quality, on=["algorithm", "dataset"], how="left", validate="many_to_one")
    cell["retention"] = (
        cell["joint_quality"] / cell["clean_joint_quality"].clip(lower=0.25)
    ).clip(0, 1)
    cell.loc[cell["condition"] == "clean", "retention"] = 1.0

    condition = (
        cell.groupby(["algorithm", "condition"], as_index=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            valid_rate=("valid_rate", "mean"),
            id_quality=("id_quality", "mean"),
            ood_quality=("ood_quality", "mean"),
            joint_quality=("joint_quality", "mean"),
            retention=("retention", "mean"),
        )
    )
    order = pd.CategoricalDtype(CONDITION_ORDER, ordered=True)
    condition["condition"] = condition["condition"].astype(order)
    condition = condition.sort_values(["algorithm", "condition"]).reset_index(drop=True)

    long_rows: list[dict[str, Any]] = []
    value_columns = {
        "Valid rate (%)": "valid_rate",
        "ID-Q mean (%)": "id_quality",
        "OOD-Q mean (%)": "ood_quality",
        "Joint-Q mean (%)": "joint_quality",
        "Clean-relative retention (%)": "retention",
    }
    for row in condition.itertuples(index=False):
        for metric, column in value_columns.items():
            long_rows.append(
                {
                    "algorithm": row.algorithm,
                    "condition": str(row.condition),
                    "metric": metric,
                    "value": 100.0 * float(getattr(row, column)),
                    "dataset_count": int(row.dataset_count),
                    "seed_count_per_dataset": EXPECTED_SEEDS,
                }
            )
    long = pd.DataFrame(long_rows)
    long["metric"] = pd.Categorical(long["metric"], METRIC_ORDER, ordered=True)
    long["condition"] = pd.Categorical(long["condition"], CONDITION_ORDER, ordered=True)
    long = long.sort_values(["algorithm", "metric", "condition"]).reset_index(drop=True)
    long["metric"] = long["metric"].astype(str)
    long["condition"] = long["condition"].astype(str)

    noisy_cells = cell[cell["condition"] != "clean"].copy()
    noisy_cells["robu_component"] = (
        0.7 * noisy_cells["joint_quality"] + 0.3 * noisy_cells["retention"]
    )
    robu = (
        noisy_cells.groupby("algorithm", as_index=False)
        .agg(
            robu_seed_mean=("robu_component", lambda x: 100.0 * float(np.mean(x))),
            noisy_dataset_sigma_cells=("robu_component", "size"),
        )
        .sort_values("robu_seed_mean", ascending=False)
        .reset_index(drop=True)
    )
    robu["robu_seed_mean_rank"] = (
        robu["robu_seed_mean"].rank(method="min", ascending=False).astype(int)
    )

    wide = long.pivot(index=["algorithm", "metric"], columns="condition", values="value")
    wide = wide.reset_index()
    wide.columns.name = None
    wide = wide.merge(
        robu[["algorithm", "robu_seed_mean", "robu_seed_mean_rank"]],
        on="algorithm",
        how="left",
        validate="many_to_one",
    )
    metric_type = pd.CategoricalDtype(METRIC_ORDER, ordered=True)
    wide["metric"] = wide["metric"].astype(metric_type)
    wide = wide.sort_values(["robu_seed_mean_rank", "algorithm", "metric"]).reset_index(drop=True)
    wide["metric"] = wide["metric"].astype(str)
    return cell, condition, long, wide


def compare_formal(robu: pd.DataFrame, formal_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    formal = pd.read_csv(formal_path)[["Algorithm", "ROBU"]].rename(
        columns={"Algorithm": "algorithm", "ROBU": "robu_formal_seed_median"}
    )
    comparison = robu.merge(formal, on="algorithm", validate="one_to_one")
    comparison["difference_seed_mean_minus_formal"] = (
        comparison["robu_seed_mean"] - comparison["robu_formal_seed_median"]
    )
    comparison["formal_rank"] = (
        comparison["robu_formal_seed_median"].rank(method="min", ascending=False).astype(int)
    )
    comparison["rank_difference"] = comparison["robu_seed_mean_rank"] - comparison["formal_rank"]
    rho = float(
        spearmanr(comparison["robu_seed_mean"], comparison["robu_formal_seed_median"]).statistic
    )
    summary = {
        "spearman_rank_correlation": rho,
        "mean_absolute_score_difference": float(
            comparison["difference_seed_mean_minus_formal"].abs().mean()
        ),
        "max_absolute_score_difference": float(
            comparison["difference_seed_mean_minus_formal"].abs().max()
        ),
    }
    comparison = comparison.sort_values("robu_seed_mean_rank").reset_index(drop=True)
    return comparison, summary


def completeness_table(runs: pd.DataFrame) -> pd.DataFrame:
    result = (
        runs.groupby(["algorithm", "condition"], as_index=False)
        .agg(
            observed_runs=("seed", "size"),
            valid_output_runs=("valid_output", "sum"),
            metric_complete_runs=("metric_complete", "sum"),
            usable_metric_runs=("usable_metric", "sum"),
        )
    )
    result["expected_runs"] = EXPECTED_DATASETS * EXPECTED_SEEDS
    result["penalized_runs"] = result["observed_runs"] - result["usable_metric_runs"]
    result["usable_metric_rate"] = result["usable_metric_runs"] / result["observed_runs"]
    result["condition"] = pd.Categorical(result["condition"], CONDITION_ORDER, ordered=True)
    result = result.sort_values(["algorithm", "condition"]).reset_index(drop=True)
    result["condition"] = result["condition"].astype(str)
    return result


def plot_table(wide: pd.DataFrame, output_base: Path) -> None:
    display = wide.copy()
    display["algorithm_label"] = display["algorithm"].str.upper()
    display["robu_label"] = display["robu_seed_mean"].map(lambda x: f"{x:.2f}")
    rows: list[list[str]] = []
    previous_algorithm = None
    for _, row in display.iterrows():
        first = row["algorithm"] != previous_algorithm
        rows.append(
            [
                row["algorithm_label"] if first else "",
                row["metric"],
                f"{row['clean']:.2f}",
                f"{row['noise_0.01']:.2f}",
                f"{row['noise_0.05']:.2f}",
                f"{row['noise_0.10']:.2f}",
                row["robu_label"] if first else "",
            ]
        )
        previous_algorithm = row["algorithm"]

    fig, ax = plt.subplots(figsize=(15, 24))
    ax.axis("off")
    headers = [
        "Algorithm",
        "Metric",
        "Clean",
        "Noise 0.01",
        "Noise 0.05",
        "Noise 0.10",
        "Aggregate: ROBU",
    ]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.10, 0.23, 0.125, 0.125, 0.125, 0.125, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 1.52)
    header_colors = ["#e8edf3", "#e8edf3", "#e8f4e3", "#e5f1fb", "#fff3cf", "#fde8d8", "#eee3f5"]
    for column, color in enumerate(header_colors):
        cell = table[(0, column)]
        cell.set_facecolor(color)
        cell.set_text_props(weight="bold", fontsize=9)
        cell.set_linewidth(0.8)

    for index, row in enumerate(rows, start=1):
        algorithm_block = (index - 1) // len(METRIC_ORDER)
        background = "#ffffff" if algorithm_block % 2 == 0 else "#f7f8fa"
        for column in range(len(headers)):
            table[(index, column)].set_facecolor(background)
            table[(index, column)].set_linewidth(0.35)
        if (index - 1) % len(METRIC_ORDER) == 0:
            for column in range(len(headers)):
                table[(index, column)].set_linewidth(0.9)
            table[(index, 0)].set_text_props(weight="bold")
            table[(index, 6)].set_text_props(weight="bold")

    ax.set_title(
        "NeurIPS Stage 4: Core-50 Clean/Noise Summary (mean over 5 seeds)",
        fontsize=15,
        weight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        0.012,
        "All values are percentages; higher is better. Failed/incomplete runs use NMSE=100. "
        "ROBU uses noisy conditions only.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_reproducibility(
    output_dir: Path,
    clean_path: Path,
    noise_path: Path,
    formal_path: Path,
    audit: dict[str, Any],
    comparison_summary: dict[str, float],
) -> None:
    payload = {
        "analysis_type": "local post-processing only; no algorithm rerun",
        "command": (
            "PYTHONPATH=. python "
            "A_Neurips_experiments/rebuttal/05_stage4_noise_seed_mean_summary/"
            "analyze_stage4_noise_seed_mean.py"
        ),
        "git_commit_before_generated_outputs": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "clean": {"path": str(clean_path.resolve()), "sha256": file_sha256(clean_path)},
            "noise": {"path": str(noise_path.resolve()), "sha256": file_sha256(noise_path)},
            "formal_robu": {"path": str(formal_path.resolve()), "sha256": file_sha256(formal_path)},
        },
        "definitions": {
            "seed_aggregation": "arithmetic mean of per-run quality over seeds 0..4",
            "failure_semantics": "invalid/incomplete/non-finite run uses ID NMSE=OOD NMSE=1e2",
            "quality_mapping": "project phi_from_nmse from check/plot_core50_hexagon_metrics.py",
            "joint_quality": "0.5 * ID-Q + 0.5 * OOD-Q",
            "retention": "clip(noisy_joint_quality / max(clean_joint_quality, 0.25), 0, 1)",
            "robu_seed_mean": "100 * mean_dataset,sigma(0.7 * joint_quality + 0.3 * retention)",
        },
        "grid_audit": audit,
        "formal_comparison": comparison_summary,
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary_report(
    output_dir: Path,
    completeness: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_summary: dict[str, float],
) -> None:
    by_condition = (
        completeness.groupby("condition", as_index=False)
        .agg(
            observed_runs=("observed_runs", "sum"),
            usable_metric_runs=("usable_metric_runs", "sum"),
            penalized_runs=("penalized_runs", "sum"),
        )
    )
    by_condition["condition"] = pd.Categorical(
        by_condition["condition"], CONDITION_ORDER, ordered=True
    )
    by_condition = by_condition.sort_values("condition")
    completeness_rows = [
        [
            CONDITION_LABELS[str(row.condition)],
            str(int(row.observed_runs)),
            str(int(row.usable_metric_runs)),
            str(int(row.penalized_runs)),
            f"{100.0 * row.usable_metric_runs / row.observed_runs:.2f}%",
        ]
        for row in by_condition.itertuples(index=False)
    ]
    robu_rows = [
        [
            str(int(row.robu_seed_mean_rank)),
            str(row.algorithm).upper(),
            f"{row.robu_seed_mean:.2f}",
            str(int(row.formal_rank)),
            f"{row.robu_formal_seed_median:.2f}",
            f"{row.difference_seed_mean_minus_formal:+.2f}",
        ]
        for row in comparison.itertuples(index=False)
    ]
    report = f"""# NeurIPS Stage 4 clean/noise 多种子均值结果

## 结论

本次整理只复用冻结的 Stage 4 run-level 数据，未重跑算法。完整网格为
`12 × 50 × 5 × 4 = 12000` 条记录，每个算法、任务、条件均恰好包含五个种子。

按用户要求对五个种子的逐 run 质量分取算术均值后，ROBU 前两名为
**DSO（{comparison.iloc[0]['robu_seed_mean']:.2f}）**和
**PySR（{comparison.iloc[1]['robu_seed_mean']:.2f}）**。seed-mean ROBU 与论文正式
seed-median ROBU 的算法排序 Spearman 相关系数为
**{comparison_summary['spearman_rank_correlation']:.3f}**；平均绝对分数差为
**{comparison_summary['mean_absolute_score_difference']:.3f}** 分，最大差为
**{comparison_summary['max_absolute_score_difference']:.3f}** 分。

这说明两种种子聚合口径总体一致，但不能互换：seed-mean 下的部分中游算法会发生名次变化。
论文和 rebuttal 的正式 ROBU 数字仍应使用 seed-median 原口径；本表适合回答“多个种子求均值”
的补充分析需求。

## 完整性

{markdown_table(
    ['Condition', 'Observed', 'Usable', 'Penalized', 'Usable rate'],
    completeness_rows,
)}

`Penalized` 包括失败、无有效输出、指标不完整或 ID/OOD NMSE 非有限的 run。
这些 run 按正式 failure semantics 使用 `NMSE=100`，没有从均值中删除。

## ROBU 对照

{markdown_table(
    ['Mean rank', 'Algorithm', 'ROBU seed-mean', 'Formal rank', 'ROBU formal median', 'Difference'],
    robu_rows,
)}

## 文件说明

- 逐算法、逐条件五项统计：`stage4_condition_metrics_seed_mean_wide.csv`
- 逐算法、逐任务、逐条件聚合：`stage4_dataset_condition_seed_mean.csv`
- 完整表图：`stage4_noise_seed_mean_table.png` 和 `.pdf`
- 正式口径对照：`stage4_robu_seed_mean_vs_formal.csv`
- 完整复现信息：`reproducibility.json`
"""
    (output_dir / "stage4_noise_seed_mean_summary.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = prepare_runs(args.clean_csv, args.noise_csv)
    audit = audit_grid(runs)
    cell, condition, long, wide = compute_condition_tables(runs)
    robu = (
        wide[["algorithm", "robu_seed_mean", "robu_seed_mean_rank"]]
        .drop_duplicates()
        .sort_values("robu_seed_mean_rank")
    )
    comparison, comparison_summary = compare_formal(robu, args.formal_robu_csv)
    completeness = completeness_table(runs)

    cell.to_csv(args.output_dir / "stage4_dataset_condition_seed_mean.csv", index=False)
    condition.to_csv(args.output_dir / "stage4_condition_summary.csv", index=False)
    long.to_csv(args.output_dir / "stage4_condition_metrics_seed_mean_long.csv", index=False)
    wide.to_csv(args.output_dir / "stage4_condition_metrics_seed_mean_wide.csv", index=False)
    completeness.to_csv(args.output_dir / "stage4_condition_completeness.csv", index=False)
    comparison.to_csv(args.output_dir / "stage4_robu_seed_mean_vs_formal.csv", index=False)
    plot_table(wide, args.output_dir / "stage4_noise_seed_mean_table")
    write_summary_report(args.output_dir, completeness, comparison, comparison_summary)
    write_reproducibility(
        args.output_dir,
        args.clean_csv,
        args.noise_csv,
        args.formal_robu_csv,
        audit,
        comparison_summary,
    )
    print(json.dumps({"grid_audit": audit, "formal_comparison": comparison_summary}, indent=2))


if __name__ == "__main__":
    main()
