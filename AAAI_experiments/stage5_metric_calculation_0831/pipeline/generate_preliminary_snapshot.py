"""生成 Stage5 当前六轴快照表和晚期爬升对比图。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.font_manager import FontProperties

from .trajectories import reconstruct_trajectory


HORIZON = 180
FOCUS_ALGORITHMS = ("QLattice", "dso", "gplearn", "symbolfit")
DISPLAY_NAMES = {
    "QLattice": "QLattice",
    "dso": "DSO",
    "gplearn": "gplearn",
    "symbolfit": "SymbolFit",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage5_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_columns(frame: pd.DataFrame) -> list[str]:
    expected = [f"q_{minute:04d}" for minute in range(1, HORIZON + 1)]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"EFF CSV 缺少分钟列: {missing[:5]}")
    return expected


def _normalized_progress(frame: pd.DataFrame) -> np.ndarray:
    values = frame[_quality_columns(frame)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    best = np.max(values, axis=1, keepdims=True)
    return np.divide(values, best, out=np.zeros_like(values), where=best > 0)


def _load_eff_curves(path: Path, algorithms: Iterable[str]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    curves: dict[str, np.ndarray] = {}
    for algorithm in algorithms:
        selected = frame[frame["algorithm"].astype(str).str.lower() == algorithm.lower()]
        if selected.empty:
            raise ValueError(f"{path} 中没有算法 {algorithm}")
        curves[algorithm] = _normalized_progress(selected)
    return curves


def _symbolfit_task_progress_dirs(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.glob("symbolfit/seed*/tasks/*/*/symbolfit/*/progress"):
        if "experiments" not in path.parts:
            result.append(path)
    return sorted(result)


def _load_symbolfit_current(root: Path, excluded_task_ids: set[str]) -> np.ndarray:
    def load_task(progress_dir: Path) -> tuple[str, list[float]]:
        task_id = progress_dir.parents[3].name
        snapshots: dict[int, dict[str, object]] = {}
        for minute in range(1, HORIZON + 1):
            path = progress_dir / f"minute_{minute:04d}.json"
            if not path.is_file():
                raise ValueError(f"SymbolFit 缺少快照: {path}")
            snapshots[minute] = json.loads(path.read_text(encoding="utf-8"))
        trajectory = reconstruct_trajectory(
            snapshots,
            horizon=HORIZON,
            algorithm="symbolfit",
        )
        return task_id, [point.quality for point in trajectory]

    progress_dirs: list[Path] = []
    seen: set[str] = set()
    for progress_dir in _symbolfit_task_progress_dirs(root):
        task_id = progress_dir.parents[3].name
        if task_id in excluded_task_ids:
            continue
        if task_id in seen:
            raise ValueError(f"SymbolFit 出现重复任务: {task_id}")
        seen.add(task_id)
        progress_dirs.append(progress_dir)
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = list(executor.map(load_task, progress_dirs))
    rows = [row for _, row in sorted(loaded)]
    if not rows:
        raise ValueError(f"没有找到 SymbolFit 轨迹: {root}")
    values = np.asarray(rows, dtype=float)
    best = np.max(values, axis=1, keepdims=True)
    return np.divide(values, best, out=np.zeros_like(values), where=best > 0)


def _curve_summary(algorithm: str, version: str, curves: np.ndarray) -> dict[str, object]:
    mean = curves.mean(axis=0)
    late_steps = np.diff(mean[149:])
    run_delta = curves[:, 179] - curves[:, 149]
    return {
        "algorithm": DISPLAY_NAMES[algorithm],
        "version": version,
        "run_count": int(curves.shape[0]),
        "eff_score": float(curves.mean() * 100),
        "mean_progress_minute_150": float(mean[149] * 100),
        "mean_progress_minute_179": float(mean[178] * 100),
        "mean_progress_minute_180": float(mean[179] * 100),
        "delta_150_to_180": float((mean[179] - mean[149]) * 100),
        "delta_179_to_180": float((mean[179] - mean[178]) * 100),
        "max_mean_one_minute_rise_150_to_180": float(max(0.0, late_steps.max()) * 100),
        "runs_rising_at_least_5pp_150_to_180": int(np.sum(run_delta >= 0.05)),
        "runs_rising_at_least_10pp_150_to_180": int(np.sum(run_delta >= 0.10)),
    }


def _font() -> FontProperties:
    preferred = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return FontProperties(fname=str(preferred)) if preferred.is_file() else FontProperties()


def _plot_late_rise(
    old_curves: dict[str, np.ndarray],
    current_curves: dict[str, np.ndarray],
    summaries: list[dict[str, object]],
    output: Path,
) -> None:
    font = _font()
    current_summary = {
        str(row["algorithm"]): row for row in summaries if row["version"] == "current_preliminary"
    }
    palette = {
        "QLattice": "#0072B2",
        "dso": "#D55E00",
        "gplearn": "#009E73",
        "symbolfit": "#CC79A7",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    minutes = np.arange(1, HORIZON + 1)
    for axis, algorithm in zip(axes.flat, FOCUS_ALGORITHMS, strict=True):
        old_mean = old_curves[algorithm].mean(axis=0) * 100
        current_mean = current_curves[algorithm].mean(axis=0) * 100
        axis.plot(minutes, old_mean, color="#777777", linewidth=1.8, linestyle="--", label="修复前完整快照")
        axis.plot(minutes, current_mean, color=palette[algorithm], linewidth=2.4, label="当前 preliminary")
        axis.axvspan(150, 180, color="#F0E442", alpha=0.12)
        axis.axvline(150, color="#555555", linewidth=0.8, alpha=0.6)
        axis.set_title(DISPLAY_NAMES[algorithm], fontproperties=font, fontsize=14, weight="bold")
        axis.set_xlim(1, 180)
        axis.set_ylim(0, 102)
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
        summary = current_summary[DISPLAY_NAMES[algorithm]]
        annotation = (
            f"150→180: {float(summary['delta_150_to_180']):+.2f} pp\n"
            f"179→180: {float(summary['delta_179_to_180']):+.2f} pp\n"
            f"n={int(summary['run_count'])}"
        )
        axis.text(
            0.025,
            0.96,
            annotation,
            transform=axis.transAxes,
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9, "pad": 5},
        )
    for axis in axes[-1, :]:
        axis.set_xlabel("搜索时间（分钟）", fontproperties=font, fontsize=11)
    for axis in axes[:, 0]:
        axis.set_ylabel("平均相对搜索进展（%）", fontproperties=font, fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        prop=font,
    )
    fig.suptitle(
        "后段突然爬升检查：修复前与当前轨迹对比",
        fontproperties=font,
        fontsize=18,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "黄色区域为第 150–180 分钟。SymbolFit 使用 149 条严格通过的全量重跑轨迹；g0039 等待合并。"
        "QLattice、DSO、gplearn 为当前内部目标重建配合冻结数值评分。",
        ha="center",
        fontproperties=font,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.03, 0.045, 0.98, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_metric_table(frame: pd.DataFrame, output: Path) -> None:
    font = _font()
    metrics = ["ID", "OOD", "SYM", "MIN", "EFF*", "STAB"]
    display = frame[["rank", "algorithm", *metrics]].copy()
    display[metrics] = display[metrics].map(lambda value: f"{float(value):.2f}")
    fig, axis = plt.subplots(figsize=(14, 10.5))
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=["排名", "算法", *metrics],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.06, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.45)
    metric_values = frame[metrics].astype(float)
    normalizer = colors.Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("RdYlGn")
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.get_text().set_fontproperties(font)
        if row == 0:
            cell.set_facecolor("#243447")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif column >= 2:
            value = metric_values.iloc[row - 1, column - 2]
            cell.set_facecolor(cmap(normalizer(value), alpha=0.42))
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7F9")
    axis.set_title(
        "六轴指标当前快照（非最终榜单）",
        fontproperties=font,
        fontsize=20,
        weight="bold",
        pad=18,
    )
    axis.text(
        0.5,
        0.925,
        "15 个算法 × 50 个任务 × 3 个随机种子",
        ha="center",
        transform=axis.transAxes,
        fontproperties=font,
        fontsize=11,
        color="#444444",
    )
    axis.text(
        0.02,
        0.015,
        "* EFF：SymbolFit 使用新重跑轨迹；QLattice、DSO、gplearn 使用当前内部目标重建和冻结数值评分；其余算法仍为旧完整快照。\n"
        "ID/OOD/SYM/MIN/STAB 尚未合并最新定向重跑，因此本表只能用于查看当前趋势，不能作为正式论文榜单。",
        transform=axis.transAxes,
        fontproperties=font,
        fontsize=9.5,
        color="#444444",
        va="bottom",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    stage5 = _stage5_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-metrics",
        type=Path,
        default=stage5 / "results/algorithm_six_axis.csv",
    )
    parser.add_argument(
        "--legacy-eff",
        type=Path,
        default=stage5 / "results/clean_eff_run_metrics.csv",
    )
    parser.add_argument(
        "--current-eff",
        type=Path,
        default=stage5 / "reports/preliminary_20260905/eff_current_code.csv",
    )
    parser.add_argument(
        "--symbolfit-root",
        type=Path,
        default=stage5 / "reruns/symbolfit_internal_progress_v1/collected/full150",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=stage5 / "reports/preliminary_20260905",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_curves = _load_eff_curves(args.legacy_eff, FOCUS_ALGORITHMS)
    current_curves = _load_eff_curves(args.current_eff, FOCUS_ALGORITHMS[:-1])
    current_curves["symbolfit"] = _load_symbolfit_current(
        args.symbolfit_root,
        excluded_task_ids={"symbolfit_s521_clean_g0039"},
    )

    summaries: list[dict[str, object]] = []
    for algorithm in FOCUS_ALGORITHMS:
        summaries.append(_curve_summary(algorithm, "legacy_complete", old_curves[algorithm]))
        summaries.append(
            _curve_summary(algorithm, "current_preliminary", current_curves[algorithm])
        )

    output_dir = args.output_dir.resolve()
    late_csv = output_dir / "late_rise_comparison.csv"
    late_png = output_dir / "late_rise_comparison.png"
    _write_csv(late_csv, summaries)
    _plot_late_rise(old_curves, current_curves, summaries, late_png)

    metrics = pd.read_csv(args.legacy_metrics)
    metrics = metrics.rename(
        columns={
            "id_score": "ID",
            "ood_score": "OOD",
            "sym_score": "SYM",
            "min_score": "MIN",
            "eff_score": "EFF*",
            "stab_score": "STAB",
        }
    )
    current_eff = {
        str(row["algorithm"]): float(row["eff_score"])
        for row in summaries
        if row["version"] == "current_preliminary"
    }
    for index, row in metrics.iterrows():
        display = DISPLAY_NAMES.get(str(row["algorithm"]), str(row["algorithm"]))
        if display in current_eff:
            metrics.at[index, "EFF*"] = current_eff[display]
    metric_columns = ["ID", "OOD", "SYM", "MIN", "EFF*", "STAB"]
    metrics["mean_six"] = metrics[metric_columns].mean(axis=1)
    metrics = metrics.sort_values(["mean_six", "algorithm"], ascending=[False, True]).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    metrics["eff_basis"] = metrics["algorithm"].map(
        lambda value: (
            "new_rerun_native_numeric_preliminary"
            if str(value) == "symbolfit"
            else "current_internal_policy_frozen_numeric_preliminary"
            if str(value) in FOCUS_ALGORITHMS
            else "legacy_complete_pending_internal_audit"
        )
    )
    metric_csv = output_dir / "algorithm_metrics_preliminary.csv"
    metric_png = output_dir / "algorithm_metrics_preliminary.png"
    metrics.to_csv(metric_csv, index=False, float_format="%.12g")
    _plot_metric_table(metrics, metric_png)

    manifest = {
        "status": "preliminary_not_for_formal_reporting",
        "scope": "clean",
        "horizon_minutes": HORIZON,
        "excluded_current_symbolfit_tasks": ["symbolfit_s521_clean_g0039"],
        "inputs": {
            "legacy_metrics": {"path": str(args.legacy_metrics.resolve()), "sha256": _sha256(args.legacy_metrics)},
            "legacy_eff": {"path": str(args.legacy_eff.resolve()), "sha256": _sha256(args.legacy_eff)},
            "current_eff": {"path": str(args.current_eff.resolve()), "sha256": _sha256(args.current_eff)},
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (metric_csv, metric_png, late_csv, late_png)
        },
        "warnings": [
            "ID/OOD/SYM/MIN/STAB 仍来自旧完整聚合，尚未合并最新定向重跑。",
            "只有 QLattice、DSO、gplearn、SymbolFit 的 EFF 替换为当前内部搜索口径。",
            "QLattice、DSO、gplearn 的当前曲线使用冻结数值评分，尚未完成新版 canonical replay。",
            "SymbolFit 当前曲线使用 149 条严格通过的 full150 轨迹，g0039 暂不纳入。",
            "该快照不得作为正式论文榜单引用。",
        ],
    }
    manifest_path = output_dir / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
