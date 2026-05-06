#!/usr/bin/env python3
"""绘制 Core-50 formal 六轴图表。

这一版继承 `hexagon_v1_with_artifacts_20260504` 的 clean/noise 统计，
并用 `symf_formal_metrics_20260504` 替换旧的 SYM-F proxy。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE50_ROOT = REPO_ROOT / "exp-planning/04.Core50正式全量评测"
DEFAULT_V1_DIR = CORE50_ROOT / "analysis/hexagon_v1_with_artifacts_20260504"
DEFAULT_SYMF_DIR = CORE50_ROOT / "analysis/symf_formal_metrics_20260504"
DEFAULT_OUTDIR = CORE50_ROOT / "analysis/hexagon_v2_formal_20260505"

AXES = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
DISPLAY = {
    "ID_Q": "ID-Q",
    "OOD_G": "OOD-G",
    "SYM_F": "SYM-F",
    "EFF": "EFF",
    "ROB": "ROBU",
    "STAB": "STAB",
}
ALGORITHM_DISPLAY = {
    "dso": "DSO",
    "drsr": "DRSR",
    "e2esr": "E2ESR",
    "gplearn": "gplearn",
    "imcts": "iMCTS",
    "llmsr": "LLM-SR",
    "pyoperon": "PyOperon",
    "pysr": "PySR",
    "qlattice": "QLattice",
    "ragsr": "RAG-SR",
    "tpsr": "TPSR",
    "udsr": "uDSR",
}
PALETTE = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#8ab17d",
    "#577590",
    "#bc6c25",
    "#6d597a",
    "#43aa8b",
    "#f94144",
    "#277da1",
]


def safe_geomean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(100 * np.exp(np.mean(np.log(np.maximum(arr / 100.0, 1e-6)))))


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f2f2f",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 9,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )


def load_inputs(v1_dir: Path, symf_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "v1_scores": pd.read_csv(v1_dir / "hexagon_scores.csv"),
        "v1_scores_ci": pd.read_csv(v1_dir / "hexagon_scores_with_ci.csv"),
        "dataset_components": pd.read_csv(v1_dir / "dataset_axis_components.csv"),
        "robustness": pd.read_csv(v1_dir / "robustness_components.csv"),
        "noise_completion": pd.read_csv(v1_dir / "noise_completion_summary.csv"),
        "symf_alg": pd.read_csv(symf_dir / "symbolic_metrics_formal_algorithm_summary.csv"),
        "symf_dataset": pd.read_csv(symf_dir / "symbolic_metrics_formal_dataset_summary.csv"),
        "symf_run": pd.read_csv(symf_dir / "symbolic_metrics_formal.csv"),
    }


def build_scores(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1 = data["v1_scores"].copy()
    symf_alg = data["symf_alg"].copy()
    scores = v1.rename(columns={"SYM_F": "SYM_F_proxy", "HexaScore_with_ROB": "HexaScore_v1_proxy"}).merge(
        symf_alg[
            [
                "algorithm",
                "SYM_F_formal",
                "exact_equiv_rate",
                "cas_equiv_rate",
                "numeric_equiv_rate",
                "pred_parse_rate",
                "mean_tree_similarity",
                "mean_var_f1",
                "mean_op_f1",
            ]
        ],
        on="algorithm",
        how="left",
    )
    scores["SYM_F"] = scores["SYM_F_formal"].fillna(0.0)
    scores["HexaScore_formal_with_ROB"] = scores[AXES].apply(lambda row: safe_geomean(row), axis=1)
    clean_axes = ["ID_Q", "OOD_G", "SYM_F", "EFF", "STAB"]
    scores["HexaScore_formal_clean"] = scores[clean_axes].apply(lambda row: safe_geomean(row), axis=1)
    scores = scores.sort_values("HexaScore_formal_with_ROB", ascending=False).reset_index(drop=True)

    base = data["dataset_components"].copy()
    symf_dataset = data["symf_dataset"][["algorithm", "gid", "dataset", "sym_f_formal"]].copy()
    rob_ds = (
        data["robustness"]
        .groupby(["algorithm", "gid", "dataset"], dropna=False)
        .agg(rob_component=("rob_component", "mean"))
        .reset_index()
    )
    components = (
        base.merge(symf_dataset, on=["algorithm", "gid", "dataset"], how="left")
        .merge(rob_ds, on=["algorithm", "gid", "dataset"], how="left")
        .rename(
            columns={
                "q_id": "ID_Q_component",
                "q_ood_g": "OOD_G_component",
                "eff_proxy": "EFF_component",
                "stab_proxy": "STAB_component",
                "q_sym_proxy": "SYM_F_proxy_component",
                "sym_f_formal": "SYM_F_component",
                "rob_component": "ROB_component",
            }
        )
    )
    for col in [
        "ID_Q_component",
        "OOD_G_component",
        "SYM_F_component",
        "EFF_component",
        "ROB_component",
        "STAB_component",
    ]:
        components[col] = pd.to_numeric(components[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    components["HexaScore_formal_with_ROB_component"] = components[
        [
            "ID_Q_component",
            "OOD_G_component",
            "SYM_F_component",
            "EFF_component",
            "ROB_component",
            "STAB_component",
        ]
    ].apply(lambda row: safe_geomean(100 * row), axis=1)
    return scores, components


def bootstrap_ci(components: pd.DataFrame, n_boot: int, seed: int = 20260505) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    comp_cols = {
        "ID_Q": "ID_Q_component",
        "OOD_G": "OOD_G_component",
        "SYM_F": "SYM_F_component",
        "EFF": "EFF_component",
        "ROB": "ROB_component",
        "STAB": "STAB_component",
    }
    for alg, group in components.groupby("algorithm", dropna=False):
        group = group.reset_index(drop=True)
        n = len(group)
        draws: dict[str, list[float]] = {axis: [] for axis in AXES}
        draws["HexaScore_formal_with_ROB"] = []
        for _ in range(n_boot):
            sample = group.iloc[rng.integers(0, n, size=n)]
            axis_scores = {axis: 100 * float(sample[col].mean()) for axis, col in comp_cols.items()}
            for axis, value in axis_scores.items():
                draws[axis].append(value)
            draws["HexaScore_formal_with_ROB"].append(safe_geomean(axis_scores.values()))
        row: dict[str, float | str] = {"algorithm": alg}
        for axis, values in draws.items():
            row[f"{axis}_ci_low"] = float(np.percentile(values, 2.5))
            row[f"{axis}_ci_high"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap(data: pd.DataFrame, out: Path, title: str | None = None, cmap: str = "YlGnBu") -> None:
    fig, ax = plt.subplots(figsize=(10.4, max(4.9, 0.45 * len(data))))
    mat = data.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    xlabels = list(data.columns)
    ylabels = [ALGORITHM_DISPLAY.get(str(name).lower(), str(name)) for name in data.index]
    ax.set_xticks(range(data.shape[1]), xlabels, fontsize=14)
    ax.set_yticks(range(data.shape[0]), ylabels, fontsize=13)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if mat[i, j] >= 72 else "#111111"
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=11.5, color=color)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    cbar = fig.colorbar(im, ax=ax, label="Score", fraction=0.032, pad=0.035)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Score", fontsize=13)
    fig.tight_layout(pad=0.4)
    save_figure(fig, out)


def plot_hexagon_heatmap(scores: pd.DataFrame, out: Path) -> None:
    data = scores.set_index("algorithm")[AXES].rename(columns=DISPLAY)
    heatmap(data, out, None)


def plot_symbolic_components(symf_alg: pd.DataFrame, out: Path) -> None:
    df = symf_alg.sort_values("SYM_F_formal", ascending=False).set_index("algorithm")
    data = pd.DataFrame(
        {
            "SYM-F": df["SYM_F_formal"],
            "Exact Eq": 100 * df["exact_equiv_rate"],
            "CAS Eq": 100 * df["cas_equiv_rate"],
            "Numeric Eq": 100 * df["numeric_equiv_rate"],
            "Parse": 100 * df["pred_parse_rate"],
            "Tree Sim": 100 * df["mean_tree_similarity"],
            "Var F1": 100 * df["mean_var_f1"],
            "Op F1": 100 * df["mean_op_f1"],
        }
    )
    heatmap(data, out, "Formal Symbolic Components", cmap="crest" if False else "YlOrBr")


def plot_bar(scores: pd.DataFrame, metric: str, out: Path, title: str, xlabel: str = "Score") -> None:
    df = scores.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.42 * len(df))))
    bars = ax.barh(df["algorithm"], df[metric], color=PALETTE[: len(df)])
    ax.set_xlim(0, max(100, float(df[metric].max()) * 1.08))
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.55)
    for bar, value in zip(bars, df[metric], strict=False):
        ax.text(float(value) + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    save_figure(fig, out)


def plot_axis_small_multiples(scores: pd.DataFrame, out: Path) -> None:
    order = scores.sort_values("HexaScore_formal_with_ROB", ascending=False)["algorithm"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True)
    for ax, axis in zip(axes.ravel(), AXES, strict=False):
        df = scores.set_index("algorithm").loc[order].reset_index()
        ax.barh(df["algorithm"], df[axis], color="#2a9d8f")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.5)
        ax.invert_yaxis()
        if axis not in {"ID_Q", "EFF"}:
            ax.tick_params(labelleft=False)
    save_figure(fig, out)


def plot_radar(scores: pd.DataFrame, out: Path, top_k: int = 4) -> None:
    df = scores.head(top_k)
    labels = [DISPLAY[a] for a in AXES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, polar=True)
    for idx, row in df.iterrows():
        values = [float(row[a]) for a in AXES]
        values += values[:1]
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(angles, values, linewidth=2.2, label=row["algorithm"], color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), frameon=False)
    save_figure(fig, out)


def plot_scatter(scores: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 5.7))
    for idx, row in scores.iterrows():
        ax.scatter(row[x], row[y], s=58, color=PALETTE[idx % len(PALETTE)], edgecolor="white", linewidth=0.8)
        ax.text(row[x] + 0.8, row[y] + 0.8, row["algorithm"], fontsize=8)
    ax.set_xlabel(DISPLAY.get(x, x))
    ax.set_ylabel(DISPLAY.get(y, y))
    ax.set_xlim(0, max(100, float(scores[x].max()) * 1.08))
    ax.set_ylim(0, max(100, float(scores[y].max()) * 1.08))
    ax.grid(alpha=0.45)
    save_figure(fig, out)


def plot_proxy_vs_formal(scores: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for idx, row in scores.iterrows():
        ax.scatter(row["SYM_F_proxy"], row["SYM_F"], s=58, color=PALETTE[idx % len(PALETTE)], edgecolor="white")
        ax.text(row["SYM_F_proxy"] + 0.8, row["SYM_F"] + 0.8, row["algorithm"], fontsize=8)
    lim = max(100, float(max(scores["SYM_F_proxy"].max(), scores["SYM_F"].max())) * 1.05)
    ax.plot([0, lim], [0, lim], "--", color="#777777", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("SYM-F proxy")
    ax.set_ylabel("SYM-F formal")
    ax.grid(alpha=0.45)
    save_figure(fig, out)


def plot_equiv_rates(symf_alg: pd.DataFrame, out: Path) -> None:
    df = symf_alg.sort_values("SYM_F_formal", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(x - width, 100 * df["exact_equiv_rate"], width, label="Final Eq", color="#2a9d8f")
    ax.bar(x, 100 * df["cas_equiv_rate"], width, label="CAS Eq", color="#e9c46a")
    ax.bar(x + width, 100 * df["numeric_equiv_rate"], width, label="Numeric Eq", color="#e76f51")
    ax.set_xticks(x, df["algorithm"], rotation=35, ha="right")
    ax.set_ylim(0, max(50, 100 * float(df[["exact_equiv_rate", "cas_equiv_rate", "numeric_equiv_rate"]].max().max()) * 1.15))
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.45)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, out)


def plot_noise_line(robustness: pd.DataFrame, out: Path) -> None:
    agg = (
        robustness.groupby(["algorithm", "noise_sigma"], dropna=False)
        .agg(rob_score=("rob_component", lambda x: 100 * float(np.mean(x))))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for idx, (alg, group) in enumerate(agg.groupby("algorithm", sort=False)):
        group = group.sort_values("noise_sigma")
        ax.plot(
            group["noise_sigma"],
            group["rob_score"],
            marker="o",
            linewidth=1.8,
            label=alg,
            color=PALETTE[idx % len(PALETTE)],
        )
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("ROBU component")
    ax.set_xticks([0.01, 0.05, 0.10], ["1%", "5%", "10%"])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.45)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_figure(fig, out)


def plot_noise_heatmaps(robustness: pd.DataFrame, noise_completion: pd.DataFrame, outdir: Path) -> None:
    agg = (
        robustness.groupby(["algorithm", "noise_sigma"], dropna=False)
        .agg(q_noise=("q_noise", lambda x: 100 * float(np.mean(x))))
        .reset_index()
    )
    qmat = agg.pivot(index="algorithm", columns="noise_sigma", values="q_noise").fillna(0)
    qmat = qmat.loc[qmat.mean(axis=1).sort_values(ascending=False).index]
    qmat.columns = [f"{int(c * 100)}%" for c in qmat.columns]
    heatmap(qmat, outdir / "fig_noise_quality_by_sigma_heatmap.png", None)

    comp = noise_completion.copy()
    comp["valid_rate_pct"] = 100 * comp["valid_rate"].fillna(0)
    vmat = comp.pivot(index="algorithm", columns="noise_sigma", values="valid_rate_pct").fillna(0)
    vmat = vmat.loc[qmat.index]
    vmat.columns = [f"{int(c * 100)}%" for c in vmat.columns]
    heatmap(vmat, outdir / "fig_noise_valid_rate_by_sigma_heatmap.png", "Noise Run Valid Rate by Sigma", cmap="Greens")


def plot_valid_parse_rates(scores: pd.DataFrame, out: Path) -> None:
    cols = ["valid_rate", "pred_parse_rate", "noise_valid_rate"]
    labels = ["Clean valid", "Expression parse", "Noise valid"]
    data = scores.set_index("algorithm")[cols].fillna(0.0) * 100
    data = data.loc[scores["algorithm"]]
    fig, ax = plt.subplots(figsize=(11.4, 5.5))
    x = np.arange(len(data))
    width = 0.25
    for i, (col, label) in enumerate(zip(cols, labels, strict=False)):
        ax.bar(x + (i - 1) * width, data[col], width, label=label)
    ax.set_xticks(x, data.index, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.45)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, out)


def write_readme(outdir: Path, scores: pd.DataFrame, figure_files: list[str], warnings: list[str]) -> None:
    lines = [
        "# Core-50 hexagon v2 formal figures",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- `SYM-F` uses formal judge: CAS equivalence, independent probe numeric equivalence, variable/operator F1, and fast tree similarity.",
        "- `ID-Q`, `OOD-G`, `EFF`, `ROBU`, `STAB` inherit the v1 clean/noise artifacts.",
        "- `EFF` and `STAB` are still proxy axes until full clean minute-level AUC and formal seed-level structural consistency are wired.",
        "",
        "## Score Table",
        "",
        scores[
            [
                "algorithm",
                "ID_Q",
                "OOD_G",
                "SYM_F",
                "EFF",
                "ROB",
                "STAB",
                "HexaScore_formal_with_ROB",
                "SYM_F_proxy",
            ]
        ].to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{name}`" for name in figure_files)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Core-50 formal hexagon figures")
    parser.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    parser.add_argument("--symf-dir", default=str(DEFAULT_SYMF_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    set_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(Path(args.v1_dir), Path(args.symf_dir))
    scores, components = build_scores(data)
    ci = bootstrap_ci(components, n_boot=args.bootstrap)
    scores_ci = scores.merge(ci, on="algorithm", how="left")

    scores.to_csv(outdir / "hexagon_scores_formal.csv", index=False)
    scores_ci.to_csv(outdir / "hexagon_scores_formal_with_ci.csv", index=False)
    components.to_csv(outdir / "dataset_axis_components_formal.csv", index=False)

    figure_files = [
        "fig_hexagon_heatmap_formal.png",
        "fig_radar_top4_formal.png",
        "fig_hexascore_formal_bar.png",
        "fig_axis_bars_formal.png",
        "fig_tradeoff_idq_oodg.png",
        "fig_tradeoff_idq_symf_formal.png",
        "fig_tradeoff_oodg_rob.png",
        "fig_tradeoff_symf_eff.png",
        "fig_tradeoff_eff_stab.png",
        "fig_symf_proxy_vs_formal.png",
        "fig_symbolic_components_heatmap.png",
        "fig_symbolic_equiv_rates_bar.png",
        "fig_noise_rob_by_sigma.png",
        "fig_noise_quality_by_sigma_heatmap.png",
        "fig_noise_valid_rate_by_sigma_heatmap.png",
        "fig_valid_parse_rates.png",
    ]

    plot_hexagon_heatmap(scores, outdir / "fig_hexagon_heatmap_formal.png")
    plot_radar(scores, outdir / "fig_radar_top4_formal.png")
    plot_bar(scores, "HexaScore_formal_with_ROB", outdir / "fig_hexascore_formal_bar.png", "Formal HexaScore with ROBU")
    plot_axis_small_multiples(scores, outdir / "fig_axis_bars_formal.png")
    plot_scatter(scores, "ID_Q", "OOD_G", outdir / "fig_tradeoff_idq_oodg.png", "ID-Q vs OOD-G")
    plot_scatter(scores, "ID_Q", "SYM_F", outdir / "fig_tradeoff_idq_symf_formal.png", "ID-Q vs formal SYM-F")
    plot_scatter(scores, "OOD_G", "ROB", outdir / "fig_tradeoff_oodg_rob.png", "OOD-G vs ROBU")
    plot_scatter(scores, "SYM_F", "EFF", outdir / "fig_tradeoff_symf_eff.png", "Formal SYM-F vs EFF")
    plot_scatter(scores, "EFF", "STAB", outdir / "fig_tradeoff_eff_stab.png", "EFF vs STAB")
    plot_proxy_vs_formal(scores, outdir / "fig_symf_proxy_vs_formal.png")
    plot_symbolic_components(data["symf_alg"], outdir / "fig_symbolic_components_heatmap.png")
    plot_equiv_rates(data["symf_alg"], outdir / "fig_symbolic_equiv_rates_bar.png")
    plot_noise_line(data["robustness"], outdir / "fig_noise_rob_by_sigma.png")
    plot_noise_heatmaps(data["robustness"], data["noise_completion"], outdir)
    plot_valid_parse_rates(scores, outdir / "fig_valid_parse_rates.png")

    warnings: list[str] = []
    if len(components) != 600:
        warnings.append(f"`dataset_axis_components_formal.csv` row count is {len(components)}, expected 600.")
    if scores["SYM_F"].isna().any():
        warnings.append("Some algorithms are missing formal SYM-F.")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outdir": str(outdir.resolve()),
        "algorithms": int(scores["algorithm"].nunique()),
        "datasets_per_algorithm_expected": 50,
        "dataset_component_rows": int(len(components)),
        "figures_png": figure_files,
        "figures_pdf": [Path(name).with_suffix(".pdf").name for name in figure_files],
        "warnings": warnings,
    }
    (outdir / "figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(outdir, scores, figure_files, warnings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
