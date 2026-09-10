#!/usr/bin/env python3
"""计算 Core-50 与 full-664 在九算法面板上的分数和排名相关性。

主口径与论文 clean leaderboard 一致：

1. 对每个 algorithm x dataset 的每个 seed 使用 split-level penalized log NMSE；
2. 对 seed 取中位数；
3. 对数据集取均值；
4. OOD 分数越低，排名越高。

Probe-2 只有一个 seed，因此在输出中单独标记，不把它描述成同三种子验证。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL7_RUNS = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
    / "analysis"
    / "full664_7alg_run_level.csv"
)
DEFAULT_PROBE2_RUNS = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage1_664dats_2probes_1seed_1h"
    / "01_probe_run_results"
    / "one_seed_probe_task_results_1328.csv"
)
DEFAULT_CORE50_MANIFEST = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage4_core50_12algs_5seeds_4noise_1h"
    / "core50_manifest"
    / "core50_datasets.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "02_core50_full664_rank_correlation_9algs"
)

PROBE4_ALGORITHMS = ("dso", "imcts", "pyoperon", "udsr")
HELDOUT_NEW3_ALGORITHMS = ("fepysr", "jaxsr", "symbolfit")
PROBE2_ALGORITHMS = ("llmsr", "pysr")
ALL_ALGORITHMS = (
    *PROBE4_ALGORITHMS,
    *HELDOUT_NEW3_ALGORITHMS,
    *PROBE2_ALGORITHMS,
)
THREE_SEED_ALGORITHMS = (*PROBE4_ALGORITHMS, *HELDOUT_NEW3_ALGORITHMS)

ALGORITHM_DISPLAY = {
    "dso": "DSO",
    "fepysr": "FePySR",
    "imcts": "iMCTS",
    "jaxsr": "JAXSR",
    "llmsr": "LLM-SR",
    "pyoperon": "PyOperon",
    "pysr": "PySR",
    "symbolfit": "SymbolFit",
    "udsr": "uDSR",
}
ALGORITHM_GROUP = {
    **{algorithm: "probe4_direct_construction" for algorithm in PROBE4_ALGORITHMS},
    **{algorithm: "post_submission_heldout" for algorithm in HELDOUT_NEW3_ALGORITHMS},
    **{algorithm: "probe2_discovery_construction" for algorithm in PROBE2_ALGORITHMS},
}
PANEL_ALGORITHMS = {
    "all_9": ALL_ALGORITHMS,
    "same_three_seed_7": THREE_SEED_ALGORITHMS,
    "non_probe4_5": (*HELDOUT_NEW3_ALGORITHMS, *PROBE2_ALGORITHMS),
    "probe4_4": PROBE4_ALGORITHMS,
    "post_submission_heldout_3": HELDOUT_NEW3_ALGORITHMS,
}

LOG_FLOOR = 1e-12
LOG_MIN = -12.0
LOG_MAX = 12.0
MISSING_LOG_PENALTY = 12.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _int(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number


def _finite(value: Any, *, nonnegative: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _normalize_dataset_dir(value: Any) -> str:
    text = _text(value).replace("\\", "/").rstrip("/")
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    return text


def _clip_log_nmse(value: Any) -> float | None:
    number = _finite(value, nonnegative=True)
    if number is None:
        return None
    raw = math.log10(max(number, LOG_FLOOR))
    return min(LOG_MAX, max(LOG_MIN, raw))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"路径位于仓库根目录之外: {resolved}") from exc


def load_full7_runs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(path):
        algorithm = _text(source.get("algorithm")).lower()
        dataset_dir = _normalize_dataset_dir(source.get("dataset_rel"))
        seed = _int(source.get("seed"))
        id_log = _finite(source.get("id_log_nmse_penalized"))
        ood_log = _finite(source.get("ood_log_nmse_penalized"))
        if (
            algorithm not in THREE_SEED_ALGORITHMS
            or not dataset_dir
            or seed is None
            or id_log is None
            or ood_log is None
            or not (LOG_MIN <= id_log <= LOG_MAX)
            or not (LOG_MIN <= ood_log <= LOG_MAX)
        ):
            raise ValueError(
                "full7 run-level 行非法: "
                f"algorithm={algorithm!r}, dataset={dataset_dir!r}, seed={seed!r}"
            )
        rows.append(
            {
                "algorithm": algorithm,
                "dataset_dir": dataset_dir,
                "seed": seed,
                "id_log_nmse_penalized": id_log,
                "ood_log_nmse_penalized": ood_log,
                "source_group": _text(source.get("source_group")),
            }
        )
    return rows


def load_probe2_runs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(path):
        algorithm = _text(source.get("method")).lower()
        dataset_dir = _normalize_dataset_dir(source.get("dataset_dir"))
        seed = _int(source.get("seed"))
        if algorithm not in PROBE2_ALGORITHMS or not dataset_dir or seed is None:
            raise ValueError(
                "Probe-2 run-level 行非法: "
                f"algorithm={algorithm!r}, dataset={dataset_dir!r}, seed={seed!r}"
            )
        id_log = _clip_log_nmse(source.get("id_nmse"))
        ood_log = _clip_log_nmse(source.get("ood_nmse"))
        rows.append(
            {
                "algorithm": algorithm,
                "dataset_dir": dataset_dir,
                "seed": seed,
                # 数值榜与 symbolic valid 分开；仅 split 数值缺失时施加惩罚。
                "id_log_nmse_penalized": (
                    id_log if id_log is not None else MISSING_LOG_PENALTY
                ),
                "ood_log_nmse_penalized": (
                    ood_log if ood_log is not None else MISSING_LOG_PENALTY
                ),
                "source_group": "stage1_probe2",
            }
        )
    return rows


def _validate_run_grid(
    runs: Sequence[dict[str, Any]],
    *,
    expected_dataset_count: int,
    expected_core_count: int,
    core_dirs: set[str],
) -> dict[str, Any]:
    algorithms = {_text(row.get("algorithm")) for row in runs}
    if algorithms != set(ALL_ALGORITHMS):
        raise ValueError(
            f"九算法集合不一致: actual={sorted(algorithms)}, "
            f"expected={sorted(ALL_ALGORITHMS)}"
        )

    keys: set[tuple[str, str, int]] = set()
    datasets_by_algorithm: dict[str, set[str]] = defaultdict(set)
    seeds_by_algorithm: dict[str, set[int]] = defaultdict(set)
    for row in runs:
        algorithm = _text(row.get("algorithm"))
        dataset_dir = _normalize_dataset_dir(row.get("dataset_dir"))
        seed = _int(row.get("seed"))
        if seed is None:
            raise ValueError(f"非法 seed: {row.get('seed')!r}")
        key = (algorithm, dataset_dir, seed)
        if key in keys:
            raise ValueError(f"run-level 重复键: {key}")
        keys.add(key)
        datasets_by_algorithm[algorithm].add(dataset_dir)
        seeds_by_algorithm[algorithm].add(seed)

    reference_datasets = datasets_by_algorithm[ALL_ALGORITHMS[0]]
    if len(reference_datasets) != expected_dataset_count:
        raise ValueError(
            f"full 数据集数量错误: {len(reference_datasets)} != " f"{expected_dataset_count}"
        )
    for algorithm in ALL_ALGORITHMS:
        if datasets_by_algorithm[algorithm] != reference_datasets:
            missing = sorted(reference_datasets - datasets_by_algorithm[algorithm])[:5]
            extra = sorted(datasets_by_algorithm[algorithm] - reference_datasets)[:5]
            raise ValueError(
                f"{algorithm} full 数据集键空间不一致: " f"missing={missing}, extra={extra}"
            )
    if len(core_dirs) != expected_core_count or not core_dirs <= reference_datasets:
        missing = sorted(core_dirs - reference_datasets)[:5]
        raise ValueError(f"Core 数据集键空间非法: core={len(core_dirs)}, missing={missing}")

    expected_seed_counts = {
        algorithm: 3 if algorithm in THREE_SEED_ALGORITHMS else 1
        for algorithm in ALL_ALGORITHMS
    }
    for algorithm, expected in expected_seed_counts.items():
        if len(seeds_by_algorithm[algorithm]) != expected:
            raise ValueError(
                f"{algorithm} seed 数量错误: "
                f"{sorted(seeds_by_algorithm[algorithm])}, expected={expected}"
            )
        expected_rows = expected_dataset_count * expected
        actual_rows = sum(1 for row in runs if _text(row.get("algorithm")) == algorithm)
        if actual_rows != expected_rows:
            raise ValueError(f"{algorithm} run 数量错误: {actual_rows} != {expected_rows}")

    return {
        "run_rows": len(runs),
        "dataset_count": len(reference_datasets),
        "core_dataset_count": len(core_dirs),
        "seeds_by_algorithm": {
            algorithm: sorted(seeds_by_algorithm[algorithm])
            for algorithm in ALL_ALGORITHMS
        },
        "runs_by_algorithm": {
            algorithm: sum(
                1 for row in runs if _text(row.get("algorithm")) == algorithm
            )
            for algorithm in ALL_ALGORITHMS
        },
    }


def build_dataset_algorithm_scores(
    runs: Sequence[dict[str, Any]],
    core_dirs: set[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        key = (
            _text(row.get("algorithm")),
            _normalize_dataset_dir(row.get("dataset_dir")),
        )
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for (algorithm, dataset_dir), group in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        id_median = statistics.median(
            float(row["id_log_nmse_penalized"]) for row in group
        )
        ood_median = statistics.median(
            float(row["ood_log_nmse_penalized"]) for row in group
        )
        output.append(
            {
                "algorithm": algorithm,
                "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                "algorithm_group": ALGORITHM_GROUP[algorithm],
                "dataset_dir": dataset_dir,
                "is_core50": dataset_dir in core_dirs,
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "id_log_nmse_seed_median_penalized": id_median,
                "ood_log_nmse_seed_median_penalized": ood_median,
                "aggregate_id_ood_log_score": 0.5 * (id_median + ood_median),
            }
        )
    return output


def _rankdata(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1],
            ordered[index][1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def summarize_algorithms(
    dataset_scores: Sequence[dict[str, Any]],
    runs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_algorithm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_scores:
        by_algorithm[_text(row.get("algorithm"))].append(row)

    run_counts: dict[tuple[str, bool], int] = defaultdict(int)
    core_dirs = {
        _text(row.get("dataset_dir"))
        for row in dataset_scores
        if bool(row.get("is_core50"))
    }
    for row in runs:
        key = (
            _text(row.get("algorithm")),
            _normalize_dataset_dir(row.get("dataset_dir")) in core_dirs,
        )
        run_counts[key] += 1

    summaries: list[dict[str, Any]] = []
    for algorithm in ALL_ALGORITHMS:
        full_rows = by_algorithm[algorithm]
        core_rows = [row for row in full_rows if bool(row.get("is_core50"))]
        summaries.append(
            {
                "algorithm": algorithm,
                "algorithm_display": ALGORITHM_DISPLAY[algorithm],
                "algorithm_group": ALGORITHM_GROUP[algorithm],
                "is_post_submission_heldout": (algorithm in HELDOUT_NEW3_ALGORITHMS),
                "seeds": ",".join(
                    str(seed)
                    for seed in sorted(
                        {
                            int(row["seed"])
                            for row in runs
                            if _text(row.get("algorithm")) == algorithm
                        }
                    )
                ),
                "full664_dataset_count": len(full_rows),
                "full664_run_count": run_counts[(algorithm, False)]
                + run_counts[(algorithm, True)],
                "core50_dataset_count": len(core_rows),
                "core50_run_count": run_counts[(algorithm, True)],
                "full664_id_score": statistics.mean(
                    float(row["id_log_nmse_seed_median_penalized"])
                    for row in full_rows
                ),
                "core50_id_score": statistics.mean(
                    float(row["id_log_nmse_seed_median_penalized"])
                    for row in core_rows
                ),
                "full664_ood_score": statistics.mean(
                    float(row["ood_log_nmse_seed_median_penalized"])
                    for row in full_rows
                ),
                "core50_ood_score": statistics.mean(
                    float(row["ood_log_nmse_seed_median_penalized"])
                    for row in core_rows
                ),
                "full664_aggregate_score": statistics.mean(
                    float(row["aggregate_id_ood_log_score"]) for row in full_rows
                ),
                "core50_aggregate_score": statistics.mean(
                    float(row["aggregate_id_ood_log_score"]) for row in core_rows
                ),
            }
        )

    for prefix in ("ood", "aggregate"):
        full_ranks = _rankdata(
            {
                row["algorithm"]: float(row[f"full664_{prefix}_score"])
                for row in summaries
            }
        )
        core_ranks = _rankdata(
            {
                row["algorithm"]: float(row[f"core50_{prefix}_score"])
                for row in summaries
            }
        )
        for row in summaries:
            algorithm = row["algorithm"]
            row[f"full664_{prefix}_rank"] = full_ranks[algorithm]
            row[f"core50_{prefix}_rank"] = core_ranks[algorithm]
            row[f"{prefix}_rank_shift_core_minus_full"] = (
                core_ranks[algorithm] - full_ranks[algorithm]
            )

    summaries.sort(
        key=lambda row: (
            float(row["full664_ood_rank"]),
            _text(row["algorithm"]),
        )
    )
    return summaries


def build_compact_id_ood_rows(
    summaries: Sequence[dict[str, Any]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    if scope not in {"full664", "core50"}:
        raise ValueError(f"未知 ID/OOD 导出范围: {scope}")
    return [
        {
            "algorithm": row["algorithm_display"],
            "algorithm_key": row["algorithm"],
            "seeds": row["seeds"],
            "dataset_count": row[f"{scope}_dataset_count"],
            "run_count": row[f"{scope}_run_count"],
            "ID": row[f"{scope}_id_score"],
            "OOD": row[f"{scope}_ood_score"],
        }
        for row in summaries
    ]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denom <= 0:
        return math.nan
    return (
        sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y))
        / denom
    )


def _kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float:
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left, right in itertools.combinations(range(len(xs)), 2):
        delta_x = (xs[left] > xs[right]) - (xs[left] < xs[right])
        delta_y = (ys[left] > ys[right]) - (ys[left] < ys[right])
        if delta_x == 0 and delta_y == 0:
            continue
        if delta_x == 0:
            ties_x += 1
        elif delta_y == 0:
            ties_y += 1
        elif delta_x == delta_y:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    return (concordant - discordant) / denominator if denominator > 0 else math.nan


def _exact_permutation_pvalue(
    xs: Sequence[float],
    ys: Sequence[float],
) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3 or len(xs) > 9:
        return None
    observed = abs(_pearson(xs, ys))
    if not math.isfinite(observed):
        return None
    exceed = 0
    total = 0
    for permutation in itertools.permutations(ys):
        total += 1
        candidate = abs(_pearson(xs, permutation))
        if candidate + 1e-15 >= observed:
            exceed += 1
    return exceed / total


def _metric_correlations(
    summaries: Sequence[dict[str, Any]],
    *,
    panel: str,
    algorithms: Sequence[str],
    metric: str,
) -> dict[str, Any]:
    by_algorithm = {
        _text(row.get("algorithm")): row
        for row in summaries
        if _text(row.get("algorithm")) in algorithms
    }
    if set(by_algorithm) != set(algorithms):
        raise ValueError(f"{panel} 算法集合不完整")

    ordered_algorithms = sorted(algorithms)
    full_scores = [
        float(by_algorithm[algorithm][f"full664_{metric}_score"])
        for algorithm in ordered_algorithms
    ]
    core_scores = [
        float(by_algorithm[algorithm][f"core50_{metric}_score"])
        for algorithm in ordered_algorithms
    ]
    full_rank_map = _rankdata(dict(zip(ordered_algorithms, full_scores)))
    core_rank_map = _rankdata(dict(zip(ordered_algorithms, core_scores)))
    full_ranks = [full_rank_map[algorithm] for algorithm in ordered_algorithms]
    core_ranks = [core_rank_map[algorithm] for algorithm in ordered_algorithms]

    pairwise_total = 0
    pairwise_agree = 0
    for left, right in itertools.combinations(ordered_algorithms, 2):
        full_delta = (
            by_algorithm[left][f"full664_{metric}_score"]
            - by_algorithm[right][f"full664_{metric}_score"]
        )
        core_delta = (
            by_algorithm[left][f"core50_{metric}_score"]
            - by_algorithm[right][f"core50_{metric}_score"]
        )
        if full_delta == 0:
            continue
        pairwise_total += 1
        if full_delta * core_delta > 0:
            pairwise_agree += 1

    pearson_score = _pearson(full_scores, core_scores)
    spearman = _pearson(full_ranks, core_ranks)
    return {
        "panel": panel,
        "metric": metric,
        "algorithm_count": len(ordered_algorithms),
        "algorithms": ",".join(ordered_algorithms),
        "pearson_score": pearson_score,
        "pearson_score_exact_two_sided_p": _exact_permutation_pvalue(
            full_scores, core_scores
        ),
        # 对排名做 Pearson 等价于 Spearman，两个名称都保留以回应原问题。
        "pearson_rank": spearman,
        "spearman_rank": spearman,
        "spearman_exact_two_sided_p": _exact_permutation_pvalue(full_ranks, core_ranks),
        "kendall_tau_b": _kendall_tau_b(full_scores, core_scores),
        "pairwise_agreement": (
            pairwise_agree / pairwise_total if pairwise_total else math.nan
        ),
        "pairwise_agree_count": pairwise_agree,
        "pairwise_total_count": pairwise_total,
        "score_mae": statistics.mean(
            abs(full_value - core_value)
            for full_value, core_value in zip(full_scores, core_scores)
        ),
        "rank_mae": statistics.mean(
            abs(full_value - core_value)
            for full_value, core_value in zip(full_ranks, core_ranks)
        ),
        "exact_rank_match_count": sum(
            math.isclose(full_value, core_value)
            for full_value, core_value in zip(full_ranks, core_ranks)
        ),
        "max_absolute_rank_shift": max(
            abs(full_value - core_value)
            for full_value, core_value in zip(full_ranks, core_ranks)
        ),
    }


def build_correlation_rows(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel, algorithms in PANEL_ALGORITHMS.items():
        for metric in ("ood", "aggregate"):
            rows.append(
                _metric_correlations(
                    summaries,
                    panel=panel,
                    algorithms=algorithms,
                    metric=metric,
                )
            )
    return rows


def build_ood_comparison_rows(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda row: float(row["full664_ood_rank"]),
    ):
        rows.append(
            {
                "algorithm": summary["algorithm_display"],
                "full664_ood_score": summary["full664_ood_score"],
                "full664_ood_rank": int(float(summary["full664_ood_rank"])),
                "core50_ood_score": summary["core50_ood_score"],
                "core50_ood_rank": int(float(summary["core50_ood_rank"])),
                "rank_shift_core_minus_full": int(
                    float(summary["ood_rank_shift_core_minus_full"])
                ),
            }
        )
    return rows


def _ood_markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Algorithm | Full-664 OOD | Rank | Core-50 OOD | Rank | Rank shift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shift = int(row["rank_shift_core_minus_full"])
        shift_text = f"{shift:+d}" if shift else "0"
        lines.append(
            "| {algorithm} | {full:.3f} | {full_rank} | "
            "{core:.3f} | {core_rank} | {shift} |".format(
                algorithm=row["algorithm"],
                full=float(row["full664_ood_score"]),
                full_rank=int(row["full664_ood_rank"]),
                core=float(row["core50_ood_score"]),
                core_rank=int(row["core50_ood_rank"]),
                shift=shift_text,
            )
        )
    return "\n".join(lines)


def _write_ood_plot(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    pearson: float,
    spearman: float,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width = 1440
    height = 1240
    left = 175
    right = 90
    top = 145
    bottom = 180
    plot_size = min(width - left - right, height - top - bottom)
    plot_right = left + plot_size
    plot_bottom = top + plot_size
    full_scores = [float(row["full664_ood_score"]) for row in rows]
    core_scores = [float(row["core50_ood_score"]) for row in rows]
    low = min(*full_scores, *core_scores) - 0.45
    high = max(*full_scores, *core_scores) + 0.45

    def load_font(filename: str, size: int) -> Any:
        candidates = (
            Path("/usr/share/fonts/truetype/dejavu") / filename,
            Path(filename),
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
        return ImageFont.load_default()

    fonts = {
        "title": load_font("DejaVuSans-Bold.ttf", 38),
        "axis": load_font("DejaVuSans.ttf", 26),
        "tick": load_font("DejaVuSans.ttf", 21),
        "label": load_font("DejaVuSans-Bold.ttf", 21),
        "note": load_font("DejaVuSans.ttf", 22),
    }

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def x_pixel(value: float) -> float:
        return left + (value - low) / (high - low) * plot_size

    def y_pixel(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * plot_size

    tick_start = math.ceil(low)
    tick_end = math.floor(high)
    for tick in range(tick_start, tick_end + 1):
        x_value = x_pixel(tick)
        y_value = y_pixel(tick)
        draw.line(
            [(x_value, top), (x_value, plot_bottom)],
            fill="#E5E7EB",
            width=2,
        )
        draw.line(
            [(left, y_value), (plot_right, y_value)],
            fill="#E5E7EB",
            width=2,
        )
        tick_text = str(tick)
        tick_box = draw.textbbox((0, 0), tick_text, font=fonts["tick"])
        tick_width = tick_box[2] - tick_box[0]
        tick_height = tick_box[3] - tick_box[1]
        draw.text(
            (x_value - tick_width / 2, plot_bottom + 14),
            tick_text,
            font=fonts["tick"],
            fill="#374151",
        )
        draw.text(
            (left - tick_width - 16, y_value - tick_height / 2),
            tick_text,
            font=fonts["tick"],
            fill="#374151",
        )

    draw.line(
        [(x_pixel(low), y_pixel(low)), (x_pixel(high), y_pixel(high))],
        fill="#9CA3AF",
        width=3,
    )
    draw.line([(left, top), (left, plot_bottom)], fill="#111827", width=3)
    draw.line(
        [(left, plot_bottom), (plot_right, plot_bottom)],
        fill="#111827",
        width=3,
    )

    offsets = {
        "uDSR": (16, -38),
        "iMCTS": (16, 12),
        "DSO": (18, -50),
        "FePySR": (-104, -42),
        "SymbolFit": (-132, 12),
        "JAXSR": (16, -42),
        "PySR": (16, 12),
        "LLM-SR": (-112, 12),
        "PyOperon": (16, -42),
    }
    for row, x_value, y_value in zip(rows, full_scores, core_scores):
        x_coordinate = x_pixel(x_value)
        y_coordinate = y_pixel(y_value)
        radius = 9
        draw.ellipse(
            [
                (x_coordinate - radius, y_coordinate - radius),
                (x_coordinate + radius, y_coordinate + radius),
            ],
            fill="#176B87",
            outline="white",
            width=2,
        )
        offset = offsets.get(str(row["algorithm"]), (16, 12))
        draw.text(
            (x_coordinate + offset[0], y_coordinate + offset[1]),
            str(row["algorithm"]),
            font=fonts["label"],
            fill="#111827",
        )

    title = "Core-50 versus Full-664 OOD performance"
    title_box = draw.textbbox((0, 0), title, font=fonts["title"])
    draw.text(
        ((width - (title_box[2] - title_box[0])) / 2, 46),
        title,
        font=fonts["title"],
        fill="#111827",
    )

    x_label = "Full-664 penalized OOD log NMSE (lower is better)"
    x_label_box = draw.textbbox((0, 0), x_label, font=fonts["axis"])
    draw.text(
        (
            left + (plot_size - (x_label_box[2] - x_label_box[0])) / 2,
            plot_bottom + 70,
        ),
        x_label,
        font=fonts["axis"],
        fill="#111827",
    )

    y_label = "Core-50 penalized OOD log NMSE (lower is better)"
    y_label_box = fonts["axis"].getbbox(y_label)
    y_label_image = Image.new(
        "RGBA",
        (
            y_label_box[2] - y_label_box[0] + 12,
            y_label_box[3] - y_label_box[1] + 12,
        ),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(y_label_image).text(
        (6, 6 - y_label_box[1]),
        y_label,
        font=fonts["axis"],
        fill="#111827",
    )
    y_label_image = y_label_image.rotate(90, expand=True)
    image.paste(
        y_label_image,
        (
            35,
            int(top + (plot_size - y_label_image.height) / 2),
        ),
        y_label_image,
    )

    note = f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}"
    note_box = draw.multiline_textbbox(
        (0, 0),
        note,
        font=fonts["note"],
        spacing=8,
    )
    note_width = note_box[2] - note_box[0]
    note_height = note_box[3] - note_box[1]
    note_left = left + 28
    note_top = top + 25
    draw.rectangle(
        [
            (note_left, note_top),
            (note_left + note_width + 34, note_top + note_height + 32),
        ],
        fill="white",
        outline="#D1D5DB",
        width=2,
    )
    draw.multiline_text(
        (note_left + 17, note_top + 14),
        note,
        font=fonts["note"],
        fill="#111827",
        spacing=8,
    )

    draw.line(
        [(plot_right - 195, plot_bottom - 38), (plot_right - 145, plot_bottom - 38)],
        fill="#9CA3AF",
        width=3,
    )
    draw.text(
        (plot_right - 133, plot_bottom - 52),
        "Equal score",
        font=fonts["tick"],
        fill="#4B5563",
    )
    image.save(path, format="PNG", optimize=True)


def _write_readme(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
    ood_rows: Sequence[dict[str, Any]],
) -> None:
    correlations = {(row["panel"], row["metric"]): row for row in correlation_rows}
    ood = correlations[("all_9", "ood")]
    aggregate = correlations[("all_9", "aggregate")]
    pearson_score = float(ood["pearson_score"])
    pearson_p = float(ood["pearson_score_exact_two_sided_p"])
    spearman_rank = float(ood["spearman_rank"])
    spearman_p = float(ood["spearman_exact_two_sided_p"])
    pairwise_agree = int(ood["pairwise_agree_count"])
    pairwise_total = int(ood["pairwise_total_count"])
    aggregate_pearson = float(aggregate["pearson_score"])
    aggregate_spearman = float(aggregate["spearman_rank"])

    order_full = ", ".join(
        row["algorithm_display"]
        for row in sorted(
            summaries,
            key=lambda row: float(row["full664_ood_rank"]),
        )
    )
    order_core = ", ".join(
        row["algorithm_display"]
        for row in sorted(
            summaries,
            key=lambda row: float(row["core50_ood_rank"]),
        )
    )
    ood_table = _ood_markdown_table(ood_rows)
    text = f"""# Core-50 vs Full-664 OOD comparison: 9 algorithms

## Direct answer

Yes: Core-50 closely tracks Full-664 for the leaderboard's primary numerical
metric. This is a matched-run comparison: for every method, the Core-50 score is
computed from the same completed Full-664 runs, restricted to the frozen 50
tasks. Thus, the two columns differ only in the task set being averaged.

{ood_table}

![Core-50 versus Full-664 OOD comparison](ood_score_comparison.png)

`OOD` means the penalized OOD `log10(NMSE)` used to rank the clean leaderboard;
lower is better. The top two methods stay first and second, PySR stays seventh,
and 32 of the 36 pairwise method orderings are preserved. The largest change is
DSO moving from rank 3 to rank 6; the methods originally ranked 3--6 remain the
same four-method block.

The matching compact ID/OOD tables are `664_id_ood_9alg.csv` and
`core50_id_ood_9alg.csv`. In both files, `ID` and `OOD` use the same penalized
log-NMSE scale and lower values are better.

## Two statistics to report

- **Pearson `r={pearson_score:.3f}`** measures whether the actual OOD score
  values move together (`p={pearson_p:.4g}`).
- **Spearman `rho={spearman_rank:.3f}`** measures whether the method ordering is
  preserved (`p={spearman_p:.4g}`).

These are complementary rather than duplicate coefficients: Pearson compares
the continuous OOD values, while Spearman compares their ranks. As a secondary
check on the reviewer's ID/OOD aggregate score, Pearson is
`{aggregate_pearson:.3f}` and Spearman is `{aggregate_spearman:.3f}`.

The resulting claim is deliberately limited: Core-50 preserves the broad
Full-664 numerical performance structure, not every adjacent rank exactly.

## Evidence boundary

- Probe-4 (`DSO`, `iMCTS`, `PyOperon`, `uDSR`) directly participated in the
  final construction panel.
- Probe-2 (`PySR`, `LLM-SR`) participated in earlier discovery and has only
  one seed (`1314`), while the other seven algorithms use seeds
  `520, 521, 522`.
- Only `FePySR`, `JAXSR`, and `SymbolFit` are post-submission held-out
  algorithms. Therefore the nine-algorithm result is an expanded
  representativeness check, not a fully independent nine-algorithm validation.
- The three held-out methods keep the same internal OOD order on both task sets,
  but `n=3` is too small to use that fact as a standalone significance claim.

## Paste-ready response

> We agree that MAE alone does not directly establish representativeness. We
> therefore compared Core-50 with Full-664 using the same completed runs and the
> leaderboard's primary metric, seed-median penalized OOD log-NMSE (lower is
> better). Across nine methods, the continuous OOD scores have Pearson
> `r={pearson_score:.3f}` (`p={pearson_p:.4g}`) and the method ranks have
> Spearman `rho={spearman_rank:.3f}` (`p={spearman_p:.4g}`);
> `{pairwise_agree}/{pairwise_total}` pairwise orderings are preserved.
> Concretely, the Full-664 order is {order_full}, while the Core-50 order is
> {order_core}. For the ID/OOD aggregate score mentioned in the review, Pearson
> is `{aggregate_pearson:.3f}` and Spearman is
> `{aggregate_spearman:.3f}`. These results support the narrower claim that
> Core-50 preserves broad Full-664 numerical conclusions, rather than every
> adjacent rank exactly.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def analyze(
    *,
    full7_runs_path: Path,
    probe2_runs_path: Path,
    core50_manifest_path: Path,
    output_dir: Path,
    expected_dataset_count: int = 664,
    expected_core_count: int = 50,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    core_rows = _read_csv(core50_manifest_path)
    core_dirs = {_normalize_dataset_dir(row.get("dataset_dir")) for row in core_rows}
    if "" in core_dirs:
        raise ValueError("Core-50 manifest 包含空 dataset_dir")

    runs = [
        *load_full7_runs(full7_runs_path),
        *load_probe2_runs(probe2_runs_path),
    ]
    coverage = _validate_run_grid(
        runs,
        expected_dataset_count=expected_dataset_count,
        expected_core_count=expected_core_count,
        core_dirs=core_dirs,
    )
    dataset_scores = build_dataset_algorithm_scores(runs, core_dirs)
    expected_dataset_scores = len(ALL_ALGORITHMS) * expected_dataset_count
    if len(dataset_scores) != expected_dataset_scores:
        raise ValueError(
            f"dataset-level 行数错误: {len(dataset_scores)} != "
            f"{expected_dataset_scores}"
        )

    summaries = summarize_algorithms(dataset_scores, runs)
    correlation_rows = build_correlation_rows(summaries)
    ood_rows = build_ood_comparison_rows(summaries)
    full664_id_ood_rows = build_compact_id_ood_rows(summaries, scope="full664")
    core50_id_ood_rows = build_compact_id_ood_rows(summaries, scope="core50")
    correlations = {(row["panel"], row["metric"]): row for row in correlation_rows}
    primary_ood = correlations[("all_9", "ood")]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dataset_algorithm_scores.csv", dataset_scores)
    _write_csv(output_dir / "algorithm_scores_and_ranks.csv", summaries)
    _write_csv(output_dir / "664_id_ood_9alg.csv", full664_id_ood_rows)
    _write_csv(output_dir / "core50_id_ood_9alg.csv", core50_id_ood_rows)
    _write_csv(output_dir / "correlation_metrics.csv", correlation_rows)
    _write_csv(output_dir / "ood_score_comparison.csv", ood_rows)
    (output_dir / "ood_score_comparison.md").write_text(
        _ood_markdown_table(ood_rows) + "\n",
        encoding="utf-8",
    )
    _write_ood_plot(
        output_dir / "ood_score_comparison.png",
        ood_rows,
        pearson=float(primary_ood["pearson_score"]),
        spearman=float(primary_ood["spearman_rank"]),
    )
    _write_readme(output_dir, summaries, correlation_rows, ood_rows)

    source_paths = {
        "full7_runs": full7_runs_path,
        "probe2_runs": probe2_runs_path,
        "core50_manifest": core50_manifest_path,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path_base": "repository_root",
        "protocol": {
            "split_metric": "log10(max(NMSE, 1e-12)) clipped to [-12, 12]",
            "missing_split_penalty": MISSING_LOG_PENALTY,
            "seed_aggregation": "median per algorithm x dataset",
            "dataset_aggregation": "mean",
            "primary_ranking_metric": "penalized OOD log NMSE",
            "aggregate_metric": (
                "0.5 * seed-median penalized ID log NMSE + "
                "0.5 * seed-median penalized OOD log NMSE"
            ),
            "direction": "lower_is_better",
        },
        "sources": {
            key: {
                "path": _repo_relative(path, repo_root),
                "sha256": _sha256(path),
            }
            for key, path in source_paths.items()
        },
        "coverage": coverage,
        "algorithm_scores": summaries,
        "correlations": correlation_rows,
        "outputs": {
            "readme": _repo_relative(output_dir / "README.md", repo_root),
            "dataset_algorithm_scores": _repo_relative(
                output_dir / "dataset_algorithm_scores.csv", repo_root
            ),
            "algorithm_scores_and_ranks": _repo_relative(
                output_dir / "algorithm_scores_and_ranks.csv", repo_root
            ),
            "full664_id_ood_9alg": _repo_relative(
                output_dir / "664_id_ood_9alg.csv", repo_root
            ),
            "core50_id_ood_9alg": _repo_relative(
                output_dir / "core50_id_ood_9alg.csv", repo_root
            ),
            "correlation_metrics": _repo_relative(
                output_dir / "correlation_metrics.csv", repo_root
            ),
            "ood_score_comparison_csv": _repo_relative(
                output_dir / "ood_score_comparison.csv", repo_root
            ),
            "ood_score_comparison_markdown": _repo_relative(
                output_dir / "ood_score_comparison.md", repo_root
            ),
            "ood_score_comparison_plot": _repo_relative(
                output_dir / "ood_score_comparison.png", repo_root
            ),
            "summary": _repo_relative(
                output_dir / "correlation_summary.json", repo_root
            ),
        },
        "limitations": [
            (
                "PySR and LLM-SR use one full-reservoir seed (1314); "
                "the other seven algorithms use seeds 520, 521, and 522."
            ),
            (
                "Probe-4 directly participated in Core-50 construction and "
                "Probe-2 participated in discovery; only the new three "
                "algorithms are post-submission held-out methods."
            ),
            (
                "The held-out-only panel contains three algorithms, so its "
                "standalone correlation estimate has low statistical power."
            ),
        ],
    }
    (output_dir / "correlation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full7-runs", type=Path, default=DEFAULT_FULL7_RUNS)
    parser.add_argument("--probe2-runs", type=Path, default=DEFAULT_PROBE2_RUNS)
    parser.add_argument(
        "--core50-manifest",
        type=Path,
        default=DEFAULT_CORE50_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-datasets", type=int, default=664)
    parser.add_argument("--expected-core", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(
        full7_runs_path=args.full7_runs,
        probe2_runs_path=args.probe2_runs,
        core50_manifest_path=args.core50_manifest,
        output_dir=args.output_dir,
        expected_dataset_count=args.expected_datasets,
        expected_core_count=args.expected_core,
    )
    primary = next(
        row
        for row in summary["correlations"]
        if row["panel"] == "all_9" and row["metric"] == "ood"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "pearson_score": primary["pearson_score"],
                "spearman_rank": primary["spearman_rank"],
                "pairwise_agreement": primary["pairwise_agreement"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
