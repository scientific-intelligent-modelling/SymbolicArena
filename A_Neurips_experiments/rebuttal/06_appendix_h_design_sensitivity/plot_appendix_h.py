#!/usr/bin/env python3
"""验证 Appendix H 完整结果并绘制 Figure H.1。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_CONFIGURATIONS = 5000
CORE50_SIZE = 50
EXPECTED_RESERVOIR_TASKS = 664
EXPECTED_MEMBERSHIP_ROWS = EXPECTED_CONFIGURATIONS * CORE50_SIZE
FIGURE_STEM = "figure_h1_design_sensitivity"


class PlotContractError(RuntimeError):
    """绘图输入不满足 Appendix H 的完整性合同。"""


class ValidatedInputs(NamedTuple):
    probe: pd.DataFrame
    core_config: pd.DataFrame
    memberships: pd.DataFrame
    task_frequency: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, label: str, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise PlotContractError(f"{label} 不存在或不是文件: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise PlotContractError(f"无法读取 {label}: {path}") from exc
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PlotContractError(f"{label} 缺少字段: {missing}")
    return frame


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if not np.isfinite(output[column].to_numpy(dtype=float)).all():
            raise PlotContractError(f"{label}.{column} 包含缺失或非有限数值")
    return output


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    normalized = series.astype("string").str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        raise PlotContractError(f"{label} 包含无法识别的布尔值")
    return normalized.astype(bool)


def _validate_probe(path: Path) -> pd.DataFrame:
    required = {
        "configuration_id",
        "selected_score",
        "selected_rank",
        "best_score",
        "relative_gap_pct",
        "selected_is_best",
    }
    probe = _read_csv(path, "Probe-4 扰动结果", required)
    if len(probe) != EXPECTED_CONFIGURATIONS:
        raise PlotContractError(
            f"Probe-4 必须恰有 5000 个配置，实际为 {len(probe)}"
        )
    config_ids = probe["configuration_id"].astype("string").str.strip()
    if config_ids.eq("").any() or config_ids.nunique() != EXPECTED_CONFIGURATIONS:
        raise PlotContractError("Probe-4 configuration_id 必须包含 5000 个唯一非空值")
    probe = _finite_numeric(
        probe,
        ("selected_score", "selected_rank", "best_score", "relative_gap_pct"),
        "Probe-4 扰动结果",
    )
    ranks = probe["selected_rank"].to_numpy(dtype=float)
    if np.any(ranks < 1) or not np.array_equal(ranks, np.floor(ranks)):
        raise PlotContractError("Probe-4 selected_rank 必须是正整数")
    if (probe["best_score"] <= 0).any():
        raise PlotContractError("Probe-4 best_score 必须为正数")
    if (probe["selected_score"] > probe["best_score"] + 1e-12).any():
        raise PlotContractError("Probe-4 selected_score 不能超过 best_score")
    expected_gap = 100.0 * (
        probe["best_score"] - probe["selected_score"]
    ) / probe["best_score"]
    if not np.allclose(expected_gap, probe["relative_gap_pct"], atol=1e-9, rtol=1e-9):
        raise PlotContractError("Probe-4 relative_gap_pct 无法由原始分数复算")
    is_best = _boolean_series(probe["selected_is_best"], "Probe-4 selected_is_best")
    if not np.array_equal(is_best.to_numpy(), (probe["selected_rank"] == 1).to_numpy()):
        raise PlotContractError("Probe-4 selected_is_best 与 selected_rank 不一致")
    score_is_best = np.isclose(
        probe["selected_score"].to_numpy(dtype=float),
        probe["best_score"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
    )
    if not np.array_equal(is_best.to_numpy(), score_is_best):
        raise PlotContractError("Probe-4 selected_is_best 与 selected/best score 不一致")
    probe["configuration_id"] = config_ids
    probe["selected_is_best"] = is_best
    return probe


def _validate_core_config(path: Path) -> pd.DataFrame:
    required = {
        "config_id",
        "status",
        "selected_count",
        "feasible",
        "overlap_with_frozen",
        "jaccard_with_frozen",
    }
    config = _read_csv(path, "Core50 配置结果", required)
    if len(config) != EXPECTED_CONFIGURATIONS:
        raise PlotContractError(f"Core50 必须恰有 5000 个配置，实际为 {len(config)}")
    config_ids = config["config_id"].astype("string").str.strip()
    if config_ids.eq("").any() or config_ids.nunique() != EXPECTED_CONFIGURATIONS:
        raise PlotContractError("Core50 config_id 必须包含 5000 个唯一非空值")
    if not config["status"].astype("string").str.strip().str.lower().eq("ok").all():
        raise PlotContractError("Core50 配置结果包含非 ok 状态")
    feasible = _boolean_series(config["feasible"], "Core50 feasible")
    if not feasible.all():
        raise PlotContractError("Core50 配置结果包含不可行子集")
    config = _finite_numeric(
        config,
        ("selected_count", "overlap_with_frozen", "jaccard_with_frozen"),
        "Core50 配置结果",
    )
    if not config["selected_count"].eq(CORE50_SIZE).all():
        raise PlotContractError("每个 Core50 配置必须报告 selected_count=50")
    overlaps = config["overlap_with_frozen"].to_numpy(dtype=float)
    if np.any(overlaps < 0) or np.any(overlaps > CORE50_SIZE) or not np.array_equal(overlaps, np.floor(overlaps)):
        raise PlotContractError("overlap_with_frozen 必须是 [0, 50] 内的整数")
    jaccards = config["jaccard_with_frozen"].to_numpy(dtype=float)
    if np.any(jaccards < 0) or np.any(jaccards > 1):
        raise PlotContractError("jaccard_with_frozen 必须位于 [0, 1]")
    expected_jaccard = overlaps / (2 * CORE50_SIZE - overlaps)
    if not np.allclose(jaccards, expected_jaccard, atol=1e-9, rtol=1e-9):
        raise PlotContractError("Core50 Jaccard 与 50 任务集合的 overlap 不一致")
    config["config_id"] = config_ids
    config["feasible"] = feasible
    return config


def _validate_memberships(path: Path, config_ids: set[str]) -> pd.DataFrame:
    memberships = _read_csv(path, "Core50 membership", {"config_id", "dataset_id"})
    if len(memberships) != EXPECTED_MEMBERSHIP_ROWS:
        raise PlotContractError(
            f"Core50 membership 必须恰有 250000 行，实际为 {len(memberships)}"
        )
    memberships = memberships.copy()
    for column in ("config_id", "dataset_id"):
        memberships[column] = memberships[column].astype("string").str.strip()
        if memberships[column].eq("").any() or memberships[column].isna().any():
            raise PlotContractError(f"Core50 membership.{column} 包含空值")
    membership_ids = set(memberships["config_id"].tolist())
    if membership_ids != config_ids:
        missing = len(config_ids - membership_ids)
        extra = len(membership_ids - config_ids)
        raise PlotContractError(
            f"Core50 membership 与配置 ID 不一致: missing={missing}, extra={extra}"
        )
    per_config = memberships.groupby("config_id", sort=False)["dataset_id"].agg(["size", "nunique"])
    invalid = per_config[(per_config["size"] != CORE50_SIZE) | (per_config["nunique"] != CORE50_SIZE)]
    if not invalid.empty:
        raise PlotContractError(
            f"每个 Core50 配置必须有 50 个唯一任务；异常配置数={len(invalid)}"
        )
    return memberships


def _validate_frequency(path: Path, memberships: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "selection_count", "selection_frequency"}
    frequency = _read_csv(path, "Core50 membership frequency", required).copy()
    if len(frequency) != EXPECTED_RESERVOIR_TASKS:
        raise PlotContractError(
            "Core50 membership frequency 必须覆盖全部 664 个候选任务，"
            f"实际为 {len(frequency)}"
        )
    frequency["dataset_id"] = frequency["dataset_id"].astype("string").str.strip()
    if frequency["dataset_id"].eq("").any() or frequency["dataset_id"].nunique() != EXPECTED_RESERVOIR_TASKS:
        raise PlotContractError("Core50 membership frequency 的 dataset_id 必须唯一且非空")
    frequency = _finite_numeric(
        frequency,
        ("selection_count", "selection_frequency"),
        "Core50 membership frequency",
    )
    counts = frequency["selection_count"].to_numpy(dtype=float)
    if np.any(counts < 0) or np.any(counts > EXPECTED_CONFIGURATIONS) or not np.array_equal(counts, np.floor(counts)):
        raise PlotContractError("Core50 selection_count 必须是 [0, 5000] 内的整数")
    expected_frequency = counts / EXPECTED_CONFIGURATIONS
    if not np.allclose(
        expected_frequency,
        frequency["selection_frequency"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise PlotContractError("Core50 selection_frequency 无法由 selection_count 复算")

    observed = memberships.groupby("dataset_id")["config_id"].nunique()
    recorded = frequency.set_index("dataset_id")["selection_count"]
    if not set(observed.index).issubset(set(recorded.index)):
        raise PlotContractError("Core50 membership 包含不在 664 任务清单中的 dataset_id")
    recomputed = recorded.index.to_series().map(observed).fillna(0).astype(int)
    if not np.array_equal(recorded.astype(int).to_numpy(), recomputed.to_numpy()):
        raise PlotContractError("Core50 membership frequency 与逐配置成员表不一致")
    return frequency.sort_values(
        ["selection_frequency", "dataset_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def load_and_validate(
    probe_results_path: Path,
    core_config_path: Path,
    core_membership_path: Path,
    core_frequency_path: Path,
) -> ValidatedInputs:
    probe = _validate_probe(Path(probe_results_path))
    core_config = _validate_core_config(Path(core_config_path))
    memberships = _validate_memberships(
        Path(core_membership_path), set(core_config["config_id"].tolist())
    )
    frequency = _validate_frequency(Path(core_frequency_path), memberships)
    return ValidatedInputs(probe, core_config, memberships, frequency)


def _summary(
    validated: ValidatedInputs,
    paths: dict[str, Path],
) -> dict[str, Any]:
    gaps = validated.probe["relative_gap_pct"]
    ranks = validated.probe["selected_rank"]
    frequencies = validated.task_frequency["selection_frequency"]
    overlaps = validated.core_config["overlap_with_frozen"]
    jaccards = validated.core_config["jaccard_with_frozen"]
    return {
        "status": "complete",
        "schema_version": "symbolicarena.appendix_h.figure_h1.v1",
        "contract": {
            "configurations_per_experiment": EXPECTED_CONFIGURATIONS,
            "core50_tasks_per_configuration": CORE50_SIZE,
            "core50_membership_rows": EXPECTED_MEMBERSHIP_ROWS,
        },
        "validation": {
            "probe_configurations": len(validated.probe),
            "core_configurations": len(validated.core_config),
            "core_membership_rows": len(validated.memberships),
            "tasks_observed_in_reselected_subsets": len(validated.task_frequency),
        },
        "probe4": {
            "within_1pct_fraction": float((gaps <= 1.0).mean()),
            "within_1_5pct_fraction": float((gaps <= 1.5).mean()),
            "mean_relative_gap_pct": float(gaps.mean()),
            "maximum_relative_gap_pct": float(gaps.max()),
            "mean_selected_rank": float(ranks.mean()),
            "selected_is_best_fraction": float(validated.probe["selected_is_best"].mean()),
        },
        "core50": {
            "tasks_selected_every_configuration": int((frequencies == 1.0).sum()),
            "tasks_selected_at_least_75pct": int((frequencies >= 0.75).sum()),
            "mean_jaccard_with_frozen": float(jaccards.mean()),
            "median_jaccard_with_frozen": float(jaccards.median()),
            "minimum_jaccard_with_frozen": float(jaccards.min()),
            "maximum_jaccard_with_frozen": float(jaccards.max()),
            "minimum_overlap_with_frozen": int(overlaps.min()),
            "maximum_overlap_with_frozen": int(overlaps.max()),
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def _render(validated: ValidatedInputs, output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)

    gaps = np.sort(validated.probe["relative_gap_pct"].to_numpy(dtype=float))
    ecdf = np.arange(1, len(gaps) + 1, dtype=float) / len(gaps)
    axes[0].step(gaps, ecdf, where="post", color="#2563A8", linewidth=2.0)
    axes[0].axvline(1.0, color="#C74440", linestyle="--", linewidth=1.2, label="1% gap")
    axes[0].axvline(1.5, color="#23856D", linestyle=":", linewidth=1.5, label="1.5% gap")
    axes[0].set_xlabel("Relative score gap (%)")
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].set_title("(a) Probe-4 robustness")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(axis="y", color="#D5D9DE", linewidth=0.7)
    axes[0].legend(frameon=False, loc="lower right")

    frequencies = validated.task_frequency["selection_frequency"].to_numpy(dtype=float)
    positions = np.arange(1, len(frequencies) + 1)
    colors = np.where(frequencies >= 0.75, "#23856D", "#8BA6B5")
    axes[1].bar(positions, frequencies, width=1.0, color=colors, linewidth=0)
    axes[1].axhline(0.75, color="#C74440", linestyle="--", linewidth=1.2, label="75% inclusion")
    axes[1].set_xlabel("Tasks sorted by inclusion frequency")
    axes[1].set_ylabel("Inclusion frequency")
    axes[1].set_title("(b) Core50 task-selection stability")
    axes[1].set_xlim(0.5, max(1.5, len(frequencies) + 0.5))
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(axis="y", color="#D5D9DE", linewidth=0.7)
    axes[1].legend(frameon=False, loc="upper right")

    png_path = output_dir / f"{FIGURE_STEM}.png"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "SymbolicArena Appendix H reproducible plot"},
    )
    plt.close(fig)
    return png_path, pdf_path


def build_figure(
    probe_results_path: Path,
    core_config_path: Path,
    core_membership_path: Path,
    core_frequency_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    probe_results_path = Path(probe_results_path)
    core_config_path = Path(core_config_path)
    core_membership_path = Path(core_membership_path)
    core_frequency_path = Path(core_frequency_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PlotContractError(f"输出目录已存在，拒绝覆盖: {output_dir}")

    # 所有规模和一致性检查都必须先通过，之后才允许创建图件目录。
    validated = load_and_validate(
        probe_results_path, core_config_path, core_membership_path, core_frequency_path
    )
    paths = {
        "probe4_perturbation_results": probe_results_path,
        "core50_configuration_results": core_config_path,
        "core50_memberships": core_membership_path,
        "core50_membership_frequency": core_frequency_path,
    }
    summary = _summary(validated, paths)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        png_path, pdf_path = _render(validated, stage)
        summary["artifacts"] = {
            png_path.name: {"sha256": sha256_file(png_path)},
            pdf_path.name: {"sha256": sha256_file(pdf_path)},
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output_dir)
    except BaseException:
        plt.close("all")
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe4-results", required=True, type=Path)
    parser.add_argument("--core50-config-results", required=True, type=Path)
    parser.add_argument("--core50-memberships", required=True, type=Path)
    parser.add_argument("--core50-frequency", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_figure(
        probe_results_path=args.probe4_results,
        core_config_path=args.core50_config_results,
        core_membership_path=args.core50_memberships,
        core_frequency_path=args.core50_frequency,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
