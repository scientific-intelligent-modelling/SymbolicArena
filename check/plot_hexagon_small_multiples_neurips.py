#!/usr/bin/env python3
"""绘制 NeurIPS 正文用 3x4 六芒星 small-multiples 图。

每个 panel 对应一个算法。panel 内绘制全部 12 个算法的六轴 profile：
- 所有算法作为低透明背景，alpha 约 0.12；
- 当前算法的六芒星边框高亮；
- 算法按正式 HexaScore 从高到低排序。
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v2_formal_20260505/hexagon_scores_formal.csv"
ANALYSIS_OUT = ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v2_formal_20260505"
PAPER_IMGS = ROOT / "paper/Paper-SRInfra/imgs"

AXES = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
AXIS_LABELS = ["ID-Q", "OOD-G", "SYM-F", "EFF", "ROBU", "STAB"]
SCORE_COL = "HexaScore_formal_with_ROB"
HIGHLIGHT_COLORS = [
    "#005f73",
    "#0a9396",
    "#ee9b00",
    "#ca6702",
    "#9b2226",
    "#3a0ca3",
    "#4361ee",
    "#2d6a4f",
    "#6c757d",
    "#7f4f24",
    "#bc4749",
    "#343a40",
]


def set_neurips_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polygon_points(values: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.clip(values / 100.0, 0.0, 1.0)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.r_[x, x[0]], np.r_[y, y[0]]


def draw_grid(ax: plt.Axes, angles: np.ndarray, *, show_axis_labels: bool = False) -> None:
    grid_color = "#d8d8d8"
    spoke_color = "#e5e5e5"
    for r in [0.25, 0.50, 0.75, 1.00]:
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color=grid_color, lw=0.45, zorder=0)
    for angle in angles:
        ax.plot([0, math.cos(angle)], [0, math.sin(angle)], color=spoke_color, lw=0.45, zorder=0)

    if not show_axis_labels:
        return

    for label, angle in zip(AXIS_LABELS, angles):
        x = 1.15 * math.cos(angle)
        y = 1.15 * math.sin(angle)
        ha = "center"
        if x < -0.2:
            ha = "right"
        elif x > 0.2:
            ha = "left"
        va = "center"
        if y > 0.7:
            va = "bottom"
        elif y < -0.7:
            va = "top"
        ax.text(x, y, label, ha=ha, va=va, fontsize=5.8, color="#4b4b4b")


def draw_axis_key(fig: plt.Figure, angles: np.ndarray) -> None:
    """在整张图顶部只画一次六轴方向说明，避免每个 panel 重复 6 个标签。"""
    key_ax = fig.add_axes([0.815, 0.890, 0.125, 0.090])
    draw_grid(key_ax, angles, show_axis_labels=True)
    key_ax.set_aspect("equal")
    key_ax.set_xlim(-1.45, 1.45)
    key_ax.set_ylim(-1.35, 1.35)
    key_ax.axis("off")


def render_small_multiples(df: pd.DataFrame, angles: np.ndarray, *, out_name: str, orientation: str) -> dict[str, object]:
    values = df[AXES].to_numpy(dtype=float)
    algorithms = df["algorithm"].astype(str).tolist()
    scores = df[SCORE_COL].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 4, figsize=(7.05, 4.45))
    axes_flat = axes.ravel()

    for rank, ax in enumerate(axes_flat):
        draw_grid(ax, angles, show_axis_labels=False)
        ax.set_aspect("equal")
        ax.set_xlim(-1.34, 1.34)
        ax.set_ylim(-1.30, 1.30)
        ax.axis("off")

        # 背景只提供整体形状感，不抢当前算法的视觉中心。
        for vals in values:
            x, y = polygon_points(vals, angles)
            ax.plot(x, y, color="#1f2933", lw=0.52, alpha=0.095, solid_joinstyle="round", solid_capstyle="round", zorder=1)
            ax.fill(x, y, color="#1f2933", alpha=0.020, zorder=1)

        # 当前算法：只让边框明显高亮，填充保持克制，避免遮住背景集合。
        x, y = polygon_points(values[rank], angles)
        color = HIGHLIGHT_COLORS[rank % len(HIGHLIGHT_COLORS)]
        ax.plot(x, y, color=color, lw=1.70, alpha=0.98, solid_joinstyle="round", solid_capstyle="round", zorder=4)
        ax.fill(x, y, color=color, alpha=0.055, zorder=3)

        title = f"{rank + 1}. {algorithms[rank]}  ({scores[rank]:.1f})"
        ax.text(
            0.0,
            1.12,
            title,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            clip_on=False,
            zorder=6,
        )

    draw_axis_key(fig, angles)
    fig.text(
        0.025,
        0.982,
        "Core-50 six-axis profiles",
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.957,
        "Ordered by HexaScore. Pale profiles show all algorithms; colored outline highlights the current algorithm.",
        ha="left",
        va="top",
        fontsize=5.9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.850, bottom=0.030, wspace=0.08, hspace=0.00)

    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_IMGS.mkdir(parents=True, exist_ok=True)

    out_png = ANALYSIS_OUT / out_name
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    paper_png = PAPER_IMGS / out_png.name
    paper_pdf = PAPER_IMGS / out_pdf.name
    shutil.copy2(out_png, paper_png)
    shutil.copy2(out_pdf, paper_pdf)

    manifest = {
        "source": str(SCORES.relative_to(ROOT)),
        "analysis_png": str(out_png.relative_to(ROOT)),
        "analysis_pdf": str(out_pdf.relative_to(ROOT)),
        "paper_png": str(paper_png.relative_to(ROOT)),
        "paper_pdf": str(paper_pdf.relative_to(ROOT)),
        "orientation": orientation,
        "order": [
            {"rank": int(i + 1), "algorithm": algorithms[i], "hexa_score": float(scores[i])}
            for i in range(len(algorithms))
        ],
        "axes": AXES,
        "background_alpha": 0.095,
        "highlight_linewidth": 1.70,
        "axis_labels": "shared_once",
    }
    return manifest


def main() -> None:
    set_neurips_style()
    if not SCORES.exists():
        raise FileNotFoundError(SCORES)

    df = pd.read_csv(SCORES)
    required = {"algorithm", SCORE_COL, *AXES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.sort_values(SCORE_COL, ascending=False).reset_index(drop=True)
    point_up_angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(AXES), endpoint=False)
    horizontal_angles = np.linspace(0.0, -2 * np.pi, len(AXES), endpoint=False)

    manifests = [
        render_small_multiples(
            df,
            point_up_angles,
            out_name="fig_hexagon_small_multiples_neurips.png",
            orientation="point_up",
        ),
        render_small_multiples(
            df,
            horizontal_angles,
            out_name="fig_hexagon_small_multiples_neurips_horizontal.png",
            orientation="horizontal",
        ),
    ]
    (ANALYSIS_OUT / "fig_hexagon_small_multiples_neurips_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
