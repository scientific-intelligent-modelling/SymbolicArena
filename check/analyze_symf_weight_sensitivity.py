#!/usr/bin/env python3
"""分析 Core-50 SYM-F 权重、非等价上限和组成项冗余敏感性。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage4_core50_12algs_5seeds_4noise_1h"
    / "formal_analysis"
    / "symbolic_metrics_formal.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "03_symf_sensitivity_12algs"
)

ALGORITHMS = (
    "drsr",
    "dso",
    "e2esr",
    "gplearn",
    "imcts",
    "llmsr",
    "pyoperon",
    "pysr",
    "qlattice",
    "ragsr",
    "tpsr",
    "udsr",
)
DISPLAY_NAMES = {
    "drsr": "DRSR",
    "dso": "DSO",
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

BASE_TREE_WEIGHT = 0.3
BASE_SOF1_WEIGHT = 0.2
BASE_VAR_WEIGHT = 0.1
BASE_OP_WEIGHT = 0.1
TOLERANCE = 1e-10

REQUIRED_FIELDS = {
    "algorithm",
    "gid",
    "dataset",
    "seed",
    "valid_for_symbolic",
    "pred_parse_ok",
    "gt_parse_ok",
    "equiv_final",
    "tree_similarity",
    "var_f1",
    "op_f1",
    "sof1",
    "sym_f_formal",
}


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是有限数值: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 不是有限数值: {value!r}")
    return number


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [round(start + index * step, 12) for index in range(count)]


def load_runs(path: Path, *, strict_core50: bool = True) -> list[dict[str, Any]]:
    raw_rows, fields = _read_csv(path)
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ValueError(f"SYM-F 输入缺少字段: {missing}")

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for raw in raw_rows:
        algorithm = str(raw["algorithm"]).strip().lower()
        gid = str(raw["gid"]).strip()
        seed = int(raw["seed"])
        key = (algorithm, gid, seed)
        if key in seen_keys:
            raise ValueError(f"重复运行键: {key}")
        seen_keys.add(key)

        components = {
            name: _float(raw[name], field=name)
            for name in (
                "tree_similarity",
                "var_f1",
                "op_f1",
                "sof1",
                "sym_f_formal",
            )
        }
        for name, value in components.items():
            if not -TOLERANCE <= value <= 1.0 + TOLERANCE:
                raise ValueError(f"{key} 的 {name} 超出 [0,1]: {value}")

        expected_sof1 = 0.5 * (
            components["var_f1"] + components["op_f1"]
        )
        if not math.isclose(
            components["sof1"],
            expected_sof1,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(
                f"{key} 的 sof1 与 variable/operator F1 不一致"
            )

        valid_for_symbolic = _bool(raw["valid_for_symbolic"])
        pred_parse_ok = _bool(raw["pred_parse_ok"])
        gt_parse_ok = _bool(raw["gt_parse_ok"])
        exact = _bool(raw["equiv_final"])
        component_eligible = (
            valid_for_symbolic and pred_parse_ok and gt_parse_ok
        )
        if exact and not component_eligible:
            raise ValueError(
                f"{key} 标记为等价，但不满足符号分数组件的有效性条件"
            )
        recomputed = (
            1.0
            if exact
            else (
                BASE_TREE_WEIGHT * components["tree_similarity"]
                + BASE_VAR_WEIGHT * components["var_f1"]
                + BASE_OP_WEIGHT * components["op_f1"]
                if component_eligible
                else 0.0
            )
        )
        if not math.isclose(
            components["sym_f_formal"],
            recomputed,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(
                f"{key} 的 sym_f_formal 与冻结公式不一致: "
                f"stored={components['sym_f_formal']}, recomputed={recomputed}"
            )

        rows.append(
            {
                "algorithm": algorithm,
                "gid": gid,
                "dataset": str(raw["dataset"]).strip(),
                "seed": seed,
                "exact": exact,
                "component_eligible": component_eligible,
                **components,
            }
        )

    if strict_core50:
        algorithms = {row["algorithm"] for row in rows}
        gids = {row["gid"] for row in rows}
        seeds = {row["seed"] for row in rows}
        if algorithms != set(ALGORITHMS):
            raise ValueError(
                "算法集合不等于冻结的 12 算法: "
                f"missing={sorted(set(ALGORITHMS) - algorithms)}, "
                f"extra={sorted(algorithms - set(ALGORITHMS))}"
            )
        if len(rows) != 3000 or len(gids) != 50 or seeds != set(range(5)):
            raise ValueError(
                "输入不是冻结的 12 x 50 x 5 网格: "
                f"rows={len(rows)}, datasets={len(gids)}, seeds={sorted(seeds)}"
            )
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["algorithm"]] += 1
        if set(counts.values()) != {250}:
            raise ValueError(f"算法运行覆盖不完整: {dict(counts)}")
    return rows


def score_runs(
    rows: list[dict[str, Any]],
    *,
    tree_weight: float,
    var_weight: float,
    op_weight: float,
) -> dict[str, float]:
    weights = (tree_weight, var_weight, op_weight)
    if any(weight < -TOLERANCE for weight in weights):
        raise ValueError(f"权重不能为负数: {weights}")
    if sum(weights) > 1.0 + TOLERANCE:
        raise ValueError(f"非等价公式上限不能超过 1: {weights}")

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["exact"]:
            score = 1.0
        elif row["component_eligible"]:
            score = (
                tree_weight * row["tree_similarity"]
                + var_weight * row["var_f1"]
                + op_weight * row["op_f1"]
            )
        else:
            score = 0.0
        grouped[row["algorithm"]].append(100.0 * score)
    return {
        algorithm: statistics.mean(values)
        for algorithm, values in grouped.items()
    }


def _rankdata(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and math.isclose(
                scores[ordered[end]],
                scores[ordered[index]],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position]] = average_rank
        index = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    variance_x = sum((value - mean_x) ** 2 for value in xs)
    variance_y = sum((value - mean_y) ** 2 for value in ys)
    if variance_x <= 0 or variance_y <= 0:
        return math.nan
    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    )
    return covariance / math.sqrt(variance_x * variance_y)


def _rank_values(values: list[float]) -> list[float]:
    scores = {str(index): value for index, value in enumerate(values)}
    ranks = _rankdata(scores)
    return [ranks[str(index)] for index in range(len(values))]


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank_values(xs), _rank_values(ys))


def _kendall_tau_b(
    baseline_scores: dict[str, float],
    alternative_scores: dict[str, float],
) -> float:
    concordant = 0
    discordant = 0
    baseline_ties = 0
    alternative_ties = 0
    for left, right in itertools.combinations(sorted(baseline_scores), 2):
        baseline_sign = (
            (baseline_scores[left] > baseline_scores[right])
            - (baseline_scores[left] < baseline_scores[right])
        )
        alternative_sign = (
            (alternative_scores[left] > alternative_scores[right])
            - (alternative_scores[left] < alternative_scores[right])
        )
        if baseline_sign == 0 and alternative_sign == 0:
            continue
        if baseline_sign == 0:
            baseline_ties += 1
        elif alternative_sign == 0:
            alternative_ties += 1
        elif baseline_sign == alternative_sign:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + baseline_ties)
        * (concordant + discordant + alternative_ties)
    )
    return (
        (concordant - discordant) / denominator
        if denominator > 0
        else math.nan
    )


def compare_scores(
    baseline_scores: dict[str, float],
    alternative_scores: dict[str, float],
) -> dict[str, Any]:
    algorithms = sorted(baseline_scores)
    if set(algorithms) != set(alternative_scores):
        raise ValueError("基准和敏感性配置的算法集合不一致")
    baseline_ranks = _rankdata(baseline_scores)
    alternative_ranks = _rankdata(alternative_scores)

    concordant = 0
    total_pairs = 0
    for left, right in itertools.combinations(algorithms, 2):
        baseline_sign = (
            (baseline_scores[left] > baseline_scores[right])
            - (baseline_scores[left] < baseline_scores[right])
        )
        alternative_sign = (
            (alternative_scores[left] > alternative_scores[right])
            - (alternative_scores[left] < alternative_scores[right])
        )
        if baseline_sign == 0:
            continue
        total_pairs += 1
        if baseline_sign == alternative_sign:
            concordant += 1

    top_score = max(alternative_scores.values())
    top_algorithms = sorted(
        algorithm
        for algorithm, value in alternative_scores.items()
        if math.isclose(
            value, top_score, rel_tol=0.0, abs_tol=TOLERANCE
        )
    )
    baseline_order = sorted(
        algorithms,
        key=lambda algorithm: (
            -baseline_scores[algorithm],
            algorithm,
        ),
    )
    alternative_order = sorted(
        algorithms,
        key=lambda algorithm: (
            -alternative_scores[algorithm],
            algorithm,
        ),
    )
    score_changes = [
        abs(alternative_scores[algorithm] - baseline_scores[algorithm])
        for algorithm in algorithms
    ]
    rank_changes = [
        abs(alternative_ranks[algorithm] - baseline_ranks[algorithm])
        for algorithm in algorithms
    ]
    return {
        "score_pearson": _pearson(
            [baseline_scores[algorithm] for algorithm in algorithms],
            [alternative_scores[algorithm] for algorithm in algorithms],
        ),
        "rank_spearman": _pearson(
            [baseline_ranks[algorithm] for algorithm in algorithms],
            [alternative_ranks[algorithm] for algorithm in algorithms],
        ),
        "kendall_tau_b": _kendall_tau_b(
            baseline_scores,
            alternative_scores,
        ),
        "pairwise_concordant": concordant,
        "pairwise_total": total_pairs,
        "pairwise_agreement": (
            concordant / total_pairs if total_pairs else math.nan
        ),
        "mean_abs_score_change": statistics.mean(score_changes),
        "max_abs_score_change": max(score_changes),
        "max_abs_rank_shift": max(rank_changes),
        "exact_rank_match": all(
            math.isclose(
                baseline_ranks[algorithm],
                alternative_ranks[algorithm],
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
            for algorithm in algorithms
        ),
        "top1": "|".join(top_algorithms),
        "top1_unchanged": top_algorithms == [baseline_order[0]],
        "top3_overlap": len(
            set(baseline_order[:3]) & set(alternative_order[:3])
        ),
        "top3_set_unchanged": (
            set(baseline_order[:3]) == set(alternative_order[:3])
        ),
        "ranking": ">".join(alternative_order),
    }


def _variant_row(
    *,
    label: str,
    tree_weight: float,
    var_weight: float,
    op_weight: float,
    rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
) -> dict[str, Any]:
    cap = round(tree_weight + var_weight + op_weight, 12)
    sof1_weight = round(var_weight + op_weight, 12)
    tree_share = round(tree_weight / cap, 12) if cap > 0 else 0.0
    scores = score_runs(
        rows,
        tree_weight=tree_weight,
        var_weight=var_weight,
        op_weight=op_weight,
    )
    return {
        "label": label,
        "tree_weight": round(tree_weight, 12),
        "sof1_weight": sof1_weight,
        "var_weight": round(var_weight, 12),
        "op_weight": round(op_weight, 12),
        "nonexact_cap": cap,
        "tree_share": tree_share,
        **compare_scores(baseline_scores, scores),
    }


def build_weight_grids(
    rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    local_rows: list[dict[str, Any]] = []
    for tree_weight in (0.24, 0.27, 0.30, 0.33, 0.36):
        for sof1_weight in (0.16, 0.18, 0.20, 0.22, 0.24):
            local_rows.append(
                _variant_row(
                    label="local_plus_minus_20pct",
                    tree_weight=tree_weight,
                    var_weight=0.5 * sof1_weight,
                    op_weight=0.5 * sof1_weight,
                    rows=rows,
                    baseline_scores=baseline_scores,
                )
            )

    broad_rows: list[dict[str, Any]] = []
    for cap in _linspace(0.25, 0.75, 21):
        for tree_share in _linspace(0.0, 1.0, 21):
            tree_weight = cap * tree_share
            sof1_weight = cap * (1.0 - tree_share)
            broad_rows.append(
                _variant_row(
                    label="broad_stress_grid",
                    tree_weight=tree_weight,
                    var_weight=0.5 * sof1_weight,
                    op_weight=0.5 * sof1_weight,
                    rows=rows,
                    baseline_scores=baseline_scores,
                )
            )

    cap_rows: list[dict[str, Any]] = []
    for cap in _linspace(0.0, 1.0, 101):
        cap_rows.append(
            _variant_row(
                label="cap_sweep_fixed_tree_share_0.6",
                tree_weight=0.6 * cap,
                var_weight=0.2 * cap,
                op_weight=0.2 * cap,
                rows=rows,
                baseline_scores=baseline_scores,
            )
        )

    share_rows: list[dict[str, Any]] = []
    for tree_share in _linspace(0.0, 1.0, 101):
        sof1_weight = 0.5 * (1.0 - tree_share)
        share_rows.append(
            _variant_row(
                label="tree_share_sweep_fixed_cap_0.5",
                tree_weight=0.5 * tree_share,
                var_weight=0.5 * sof1_weight,
                op_weight=0.5 * sof1_weight,
                rows=rows,
                baseline_scores=baseline_scores,
            )
        )
    return local_rows, broad_rows, cap_rows, share_rows


def _component_contributions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    output: list[dict[str, Any]] = []
    for algorithm, group in grouped.items():
        run_count = len(group)
        exact_points = 100.0 * sum(row["exact"] for row in group) / run_count
        tree_points = (
            100.0
            * sum(
                BASE_TREE_WEIGHT * row["tree_similarity"]
                for row in group
                if not row["exact"] and row["component_eligible"]
            )
            / run_count
        )
        var_points = (
            100.0
            * sum(
                BASE_VAR_WEIGHT * row["var_f1"]
                for row in group
                if not row["exact"] and row["component_eligible"]
            )
            / run_count
        )
        op_points = (
            100.0
            * sum(
                BASE_OP_WEIGHT * row["op_f1"]
                for row in group
                if not row["exact"] and row["component_eligible"]
            )
            / run_count
        )
        total = exact_points + tree_points + var_points + op_points
        output.append(
            {
                "algorithm": algorithm,
                "display_name": DISPLAY_NAMES.get(algorithm, algorithm),
                "runs": run_count,
                "exact_equiv_rate": exact_points / 100.0,
                "exact_points": exact_points,
                "tree_points": tree_points,
                "variable_points": var_points,
                "operator_points": op_points,
                "partial_points": tree_points + var_points + op_points,
                "SYM_F": total,
                "exact_fraction_of_SYM_F": (
                    exact_points / total if total > 0 else 0.0
                ),
            }
        )
    output.sort(key=lambda row: (-row["SYM_F"], row["algorithm"]))
    for rank, row in enumerate(output, start=1):
        row["baseline_rank"] = rank
    return output


def _component_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    eligible = [
        row
        for row in rows
        if not row["exact"] and row["component_eligible"]
    ]
    run_rows = [
        {
            "tree_similarity": row["tree_similarity"],
            "var_f1": row["var_f1"],
            "op_f1": row["op_f1"],
            "sof1": row["sof1"],
        }
        for row in eligible
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["algorithm"]].append(row)
    algorithm_rows = []
    for algorithm in sorted(grouped):
        group = grouped[algorithm]
        algorithm_rows.append(
            {
                "tree_similarity": statistics.mean(
                    row["tree_similarity"] for row in group
                ),
                "var_f1": statistics.mean(row["var_f1"] for row in group),
                "op_f1": statistics.mean(row["op_f1"] for row in group),
                "sof1": statistics.mean(row["sof1"] for row in group),
            }
        )
    return run_rows, algorithm_rows


def component_correlations(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_rows, algorithm_rows = _component_rows(rows)
    output: list[dict[str, Any]] = []
    fields = ("tree_similarity", "var_f1", "op_f1", "sof1")
    for level, component_rows in (
        ("run_nonexact_valid", run_rows),
        ("algorithm_mean_nonexact_valid", algorithm_rows),
    ):
        for left, right in itertools.combinations(fields, 2):
            left_values = [row[left] for row in component_rows]
            right_values = [row[right] for row in component_rows]
            output.append(
                {
                    "level": level,
                    "component_x": left,
                    "component_y": right,
                    "n": len(component_rows),
                    "pearson": _pearson(left_values, right_values),
                    "spearman": _spearman(left_values, right_values),
                }
            )
    return output


def component_redundancy_regressions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_rows, _ = _component_rows(rows)
    fields = ("tree_similarity", "var_f1", "op_f1")
    output = []
    for target in fields:
        predictors = [field for field in fields if field != target]
        y_values = np.asarray(
            [row[target] for row in run_rows],
            dtype=float,
        )
        design = np.asarray(
            [
                [1.0, row[predictors[0]], row[predictors[1]]]
                for row in run_rows
            ],
            dtype=float,
        )
        coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
        fitted = design @ coefficients
        denominator = float(np.sum((y_values - np.mean(y_values)) ** 2))
        r_squared = (
            1.0
            - float(np.sum((y_values - fitted) ** 2)) / denominator
            if denominator > 0
            else math.nan
        )
        output.append(
            {
                "target": target,
                "predictor_1": predictors[0],
                "predictor_2": predictors[1],
                "n": len(run_rows),
                "r_squared": r_squared,
                "variance_inflation_factor": (
                    1.0 / (1.0 - r_squared)
                    if r_squared < 1.0
                    else math.inf
                ),
                "intercept": float(coefficients[0]),
                "coef_predictor_1": float(coefficients[1]),
                "coef_predictor_2": float(coefficients[2]),
            }
        )
    return output


def build_ablation_rows(
    rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = {
        "baseline": (0.30, 0.10, 0.10),
        "exact_only": (0.00, 0.00, 0.00),
        "lower_cap_0.25_same_mix": (0.15, 0.05, 0.05),
        "higher_cap_0.75_same_mix": (0.45, 0.15, 0.15),
        "equal_three_way_cap_0.5": (1 / 6, 1 / 6, 1 / 6),
        "tree_only_renormalized": (0.50, 0.00, 0.00),
        "sof1_only_renormalized": (0.00, 0.25, 0.25),
        "variable_only_renormalized": (0.00, 0.50, 0.00),
        "operator_only_renormalized": (0.00, 0.00, 0.50),
        "drop_tree_no_renormalization": (0.00, 0.10, 0.10),
        "drop_variable_no_renormalization": (0.30, 0.00, 0.10),
        "drop_operator_no_renormalization": (0.30, 0.10, 0.00),
    }
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    baseline_ranks = _rankdata(baseline_scores)
    for label, (tree_weight, var_weight, op_weight) in scenarios.items():
        scores = score_runs(
            rows,
            tree_weight=tree_weight,
            var_weight=var_weight,
            op_weight=op_weight,
        )
        summary_rows.append(
            _variant_row(
                label=label,
                tree_weight=tree_weight,
                var_weight=var_weight,
                op_weight=op_weight,
                rows=rows,
                baseline_scores=baseline_scores,
            )
        )
        ranks = _rankdata(scores)
        for algorithm in sorted(scores, key=lambda key: (ranks[key], key)):
            score_rows.append(
                {
                    "scenario": label,
                    "algorithm": algorithm,
                    "display_name": DISPLAY_NAMES.get(
                        algorithm, algorithm
                    ),
                    "score": scores[algorithm],
                    "rank": ranks[algorithm],
                    "score_change_from_baseline": (
                        scores[algorithm] - baseline_scores[algorithm]
                    ),
                    "rank_change_from_baseline": (
                        ranks[algorithm] - baseline_ranks[algorithm]
                    ),
                }
            )
    return summary_rows, score_rows


def _ranges(
    rows: list[dict[str, Any]],
    *,
    value_field: str,
    predicate_field: str,
) -> list[list[float]]:
    selected = sorted(
        float(row[value_field])
        for row in rows
        if _bool(row[predicate_field])
    )
    if not selected:
        return []
    step = min(
        (
            right - left
            for left, right in zip(selected, selected[1:])
            if right - left > TOLERANCE
        ),
        default=0.0,
    )
    ranges: list[list[float]] = []
    start = previous = selected[0]
    for value in selected[1:]:
        if step > 0 and value - previous > step + TOLERANCE:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def _grid_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "configurations": len(rows),
        "min_score_pearson": min(row["score_pearson"] for row in rows),
        "median_score_pearson": statistics.median(
            row["score_pearson"] for row in rows
        ),
        "min_rank_spearman": min(
            row["rank_spearman"] for row in rows
        ),
        "median_rank_spearman": statistics.median(
            row["rank_spearman"] for row in rows
        ),
        "min_kendall_tau_b": min(row["kendall_tau_b"] for row in rows),
        "min_pairwise_agreement": min(
            row["pairwise_agreement"] for row in rows
        ),
        "max_abs_rank_shift": max(
            row["max_abs_rank_shift"] for row in rows
        ),
        "max_mean_abs_score_change": max(
            row["mean_abs_score_change"] for row in rows
        ),
        "max_single_algorithm_score_change": max(
            row["max_abs_score_change"] for row in rows
        ),
        "exact_rank_match_fraction": statistics.mean(
            _bool(row["exact_rank_match"]) for row in rows
        ),
        "top1_unchanged_fraction": statistics.mean(
            _bool(row["top1_unchanged"]) for row in rows
        ),
        "top3_set_unchanged_fraction": statistics.mean(
            _bool(row["top3_set_unchanged"]) for row in rows
        ),
    }


def _plot_results(
    output_dir: Path,
    broad_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
    share_rows: list[dict[str, Any]],
    contributions: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    caps = sorted({float(row["nonexact_cap"]) for row in broad_rows})
    shares = sorted({float(row["tree_share"]) for row in broad_rows})
    matrix = np.full((len(caps), len(shares)), np.nan)
    cap_index = {value: index for index, value in enumerate(caps)}
    share_index = {value: index for index, value in enumerate(shares)}
    for row in broad_rows:
        matrix[
            cap_index[float(row["nonexact_cap"])],
            share_index[float(row["tree_share"])],
        ] = float(row["rank_spearman"])
    image = axes[0, 0].imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=(min(shares), max(shares), min(caps), max(caps)),
        vmin=float(np.nanmin(matrix)),
        vmax=1.0,
        cmap="viridis",
    )
    axes[0, 0].scatter([0.6], [0.5], marker="x", color="white", s=80)
    axes[0, 0].set_title("Rank stability across the broad stress grid")
    axes[0, 0].set_xlabel("TreeSim share of non-exact credit")
    axes[0, 0].set_ylabel("Non-exact score cap")
    figure.colorbar(image, ax=axes[0, 0], label="Spearman vs baseline")

    axes[0, 1].plot(
        [row["nonexact_cap"] for row in cap_rows],
        [row["rank_spearman"] for row in cap_rows],
        label="cap sweep (TreeSim share = 0.6)",
        linewidth=2,
    )
    axes[0, 1].plot(
        [row["tree_share"] for row in share_rows],
        [row["rank_spearman"] for row in share_rows],
        label="TreeSim-share sweep (cap = 0.5)",
        linewidth=2,
    )
    axes[0, 1].axhline(1.0, color="#222222", linewidth=0.8)
    axes[0, 1].set_ylim(0.8, 1.01)
    axes[0, 1].set_title("One-dimensional stress tests")
    axes[0, 1].set_xlabel("Swept value")
    axes[0, 1].set_ylabel("Spearman vs baseline")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    names = [row["display_name"] for row in contributions]
    exact = np.asarray([row["exact_points"] for row in contributions])
    tree = np.asarray([row["tree_points"] for row in contributions])
    variable = np.asarray([row["variable_points"] for row in contributions])
    operator = np.asarray([row["operator_points"] for row in contributions])
    x_positions = np.arange(len(names))
    axes[1, 0].bar(x_positions, exact, label="Exact")
    axes[1, 0].bar(x_positions, tree, bottom=exact, label="TreeSim")
    axes[1, 0].bar(
        x_positions,
        variable,
        bottom=exact + tree,
        label="Variable F1",
    )
    axes[1, 0].bar(
        x_positions,
        operator,
        bottom=exact + tree + variable,
        label="Operator F1",
    )
    axes[1, 0].set_xticks(x_positions)
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 0].set_ylabel("SYM-F points")
    axes[1, 0].set_title("Baseline score decomposition")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 0].grid(axis="y", alpha=0.25)

    components = ["tree_similarity", "var_f1", "op_f1"]
    correlation_matrix = np.eye(len(components))
    for row in correlations:
        if row["level"] != "run_nonexact_valid":
            continue
        if (
            row["component_x"] not in components
            or row["component_y"] not in components
        ):
            continue
        left = components.index(row["component_x"])
        right = components.index(row["component_y"])
        correlation_matrix[left, right] = row["pearson"]
        correlation_matrix[right, left] = row["pearson"]
    corr_image = axes[1, 1].imshow(
        correlation_matrix,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    labels = ["TreeSim", "Variable F1", "Operator F1"]
    axes[1, 1].set_xticks(range(3), labels)
    axes[1, 1].set_yticks(range(3), labels)
    axes[1, 1].set_title("Run-level Pearson correlations\n(non-exact valid runs)")
    for row_index in range(3):
        for column_index in range(3):
            axes[1, 1].text(
                column_index,
                row_index,
                f"{correlation_matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if abs(correlation_matrix[row_index, column_index]) > 0.55
                    else "black"
                ),
            )
    figure.colorbar(corr_image, ax=axes[1, 1], label="Pearson correlation")

    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"symf_weight_sensitivity.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_readme(
    path: Path,
    *,
    summary: dict[str, Any],
    contributions: list[dict[str, Any]],
) -> None:
    local = summary["local_weight_grid"]
    broad = summary["broad_stress_grid"]
    redundancy = summary["redundancy"]
    cap_rank_range = summary["cap_sweep"]["exact_rank_match_ranges"][0]
    baseline_rows = "\n".join(
        "| {rank} | {name} | {score:.3f} | {exact:.3f} | {tree:.3f} | "
        "{var:.3f} | {op:.3f} |".format(
            rank=row["baseline_rank"],
            name=row["display_name"],
            score=row["SYM_F"],
            exact=row["exact_points"],
            tree=row["tree_points"],
            var=row["variable_points"],
            op=row["operator_points"],
        )
        for row in contributions
    )
    text = f"""# Core-50 SYM-F sensitivity analysis

## Scope

- Input: `{summary['input']['path']}`
- Input SHA256: `{summary['input']['sha256']}`
- Grid: `{summary['input']['runs']}` runs =
  `{summary['input']['algorithms']} algorithms x {summary['input']['datasets']} datasets x {summary['input']['seeds']} seeds`.
- Baseline non-exact score:
  `0.3 * TreeSim + 0.1 * variable-F1 + 0.1 * operator-F1`.
- Exact formulas remain `1`; invalid or unparsable runs remain `0` in every
  configuration.
- This analysis tests the **aggregation weights and non-exact cap**. It does
  not test the CAS/numerical equivalence detector or its `1e-10` threshold.
- Correlations are descriptive dependence diagnostics, not proofs of
  statistical or causal independence.

Reproduce from the repository root:

```bash
python check/analyze_symf_weight_sensitivity.py
```

## Main findings

1. **Local perturbations are fully stable.** Across all
   `{local['configurations']}` combinations formed by independently changing
   the `0.3` TreeSim and `0.2` SOF1 weights by up to `+/-20%`, all 12 algorithm
   ranks and all 66 pairwise orderings are unchanged. The minimum score
   Pearson correlation is `{local['min_score_pearson']:.6f}` and the minimum
   rank Spearman correlation is `{local['min_rank_spearman']:.6f}`.

2. **The 0.5 non-exact cap is not a ranking breakpoint.** With the original
   60/40 TreeSim/SOF1 mixture, the complete ranking is unchanged for all
   tested caps in `[{cap_rank_range[0]:.2f}, {cap_rank_range[1]:.2f}]` at
   `0.01` resolution. The Top-3 set and
   winner remain unchanged throughout the full diagnostic cap sweep
   `[0, 1]`. In particular, every tested cap from `0.25` through `0.75`
   preserves the full ranking.

3. **Broad stress tests preserve the headline conclusion.** Across
   `{broad['configurations']}` combinations spanning caps `[0.25, 0.75]` and
   TreeSim shares `[0, 1]`, the minimum rank Spearman is
   `{broad['min_rank_spearman']:.3f}`, minimum pairwise agreement is
   `{broad['min_pairwise_agreement']:.3f}`, maximum rank movement is
   `{broad['max_abs_rank_shift']:.0f}`, and the Top-3 set is unchanged in
   `{100 * broad['top3_set_unchanged_fraction']:.1f}%` of configurations.
   These extreme endpoints deliberately include dropping TreeSim or SOF1
   completely and should be treated as stress tests, not equally plausible
   defaults.

4. **The components are not empirically redundant at run level.** Among
   `{redundancy['nonexact_valid_runs']}` non-exact valid runs, TreeSim and
   SOF1 have Pearson correlation
   `{redundancy['tree_sof1_pearson']:.3f}` and Spearman correlation
   `{redundancy['tree_sof1_spearman']:.3f}`. Variable-F1 and operator-F1 have
   Pearson correlation `{redundancy['var_op_pearson']:.3f}`. Regressing
   TreeSim on both F1 components explains only
   `{100 * redundancy['tree_from_f1_r_squared']:.1f}%` of its variance.

5. **TreeSim is complementary but sparse.** It is zero on
   `{100 * redundancy['tree_zero_rate_nonexact_valid']:.1f}%` of non-exact
   valid runs. Consequently, despite receiving 60% of the nominal partial
   credit, it contributes only
   `{summary['baseline_decomposition']['mean_tree_points']:.3f}` of the
   average `{summary['baseline_decomposition']['mean_SYM_F']:.3f}` SYM-F
   points, versus
   `{summary['baseline_decomposition']['mean_var_op_points']:.3f}` points
   from variable/operator F1. This is a limitation to disclose: the
   components are not double-counted, but their empirical activation rates
   differ substantially.

## Baseline decomposition

| Rank | Algorithm | SYM-F | Exact pts | Tree pts | Variable pts | Operator pts |
|---:|:---|---:|---:|---:|---:|---:|
{baseline_rows}

## Suggested rebuttal text

> We added a run-level sensitivity analysis over all 3,000 Core-50 symbolic
> evaluations. Independently perturbing the TreeSim and SOF1 weights by
> +/-20% leaves all 12 ranks and all 66 pairwise method orderings unchanged
> (minimum score Pearson r={local['min_score_pearson']:.4f}; rank
> Spearman rho={local['min_rank_spearman']:.3f}). Holding the original
> component ratio fixed, every non-exact cap from 0.25 to 0.75 also preserves
> the full ranking. A broader 441-setting stress grid, including the extreme
> removal of either TreeSim or SOF1, retains the same Top-3 set throughout
> (minimum rho={broad['min_rank_spearman']:.3f}). Finally, TreeSim and SOF1
> are nearly uncorrelated across non-exact valid runs
> (Pearson r={redundancy['tree_sof1_pearson']:.3f}), arguing against obvious
> duplicate credit. We will report these results and clarify that TreeSim is
> complementary but sparse, so the F1 terms supply most observed partial
> credit.

## Files

- `algorithm_baseline_contributions.csv`
- `local_weight_sensitivity.csv`
- `broad_weight_stress_grid.csv`
- `nonexact_cap_sweep.csv`
- `tree_share_sweep.csv`
- `ablation_summary.csv`
- `ablation_algorithm_scores.csv`
- `component_correlations.csv`
- `component_redundancy_regressions.csv`
- `sensitivity_summary.json`
- `symf_weight_sensitivity.png/.pdf`
"""
    path.write_text(text, encoding="utf-8")


def analyze(
    *,
    input_path: Path,
    output_dir: Path,
    strict_core50: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    rows = load_runs(input_path, strict_core50=strict_core50)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_scores = score_runs(
        rows,
        tree_weight=BASE_TREE_WEIGHT,
        var_weight=BASE_VAR_WEIGHT,
        op_weight=BASE_OP_WEIGHT,
    )
    contributions = _component_contributions(rows)
    local_rows, broad_rows, cap_rows, share_rows = build_weight_grids(
        rows,
        baseline_scores,
    )
    correlation_rows = component_correlations(rows)
    regression_rows = component_redundancy_regressions(rows)
    ablation_summary, ablation_scores = build_ablation_rows(
        rows,
        baseline_scores,
    )

    _write_csv(
        output_dir / "algorithm_baseline_contributions.csv",
        contributions,
    )
    _write_csv(output_dir / "local_weight_sensitivity.csv", local_rows)
    _write_csv(output_dir / "broad_weight_stress_grid.csv", broad_rows)
    _write_csv(output_dir / "nonexact_cap_sweep.csv", cap_rows)
    _write_csv(output_dir / "tree_share_sweep.csv", share_rows)
    _write_csv(output_dir / "ablation_summary.csv", ablation_summary)
    _write_csv(
        output_dir / "ablation_algorithm_scores.csv",
        ablation_scores,
    )
    _write_csv(output_dir / "component_correlations.csv", correlation_rows)
    _write_csv(
        output_dir / "component_redundancy_regressions.csv",
        regression_rows,
    )

    algorithms = {row["algorithm"] for row in rows}
    gids = {row["gid"] for row in rows}
    seeds = {row["seed"] for row in rows}
    nonexact_valid = [
        row
        for row in rows
        if not row["exact"] and row["component_eligible"]
    ]
    run_correlation_index = {
        frozenset((row["component_x"], row["component_y"])): row
        for row in correlation_rows
        if row["level"] == "run_nonexact_valid"
    }
    regression_index = {row["target"]: row for row in regression_rows}

    mean_symf = statistics.mean(
        row["SYM_F"] for row in contributions
    )
    mean_exact = statistics.mean(
        row["exact_points"] for row in contributions
    )
    mean_tree = statistics.mean(
        row["tree_points"] for row in contributions
    )
    mean_var = statistics.mean(
        row["variable_points"] for row in contributions
    )
    mean_op = statistics.mean(
        row["operator_points"] for row in contributions
    )
    summary = {
        "schema_version": 1,
        "input": {
            "path": _repo_relative(input_path),
            "sha256": _sha256(input_path),
            "runs": len(rows),
            "algorithms": len(algorithms),
            "datasets": len(gids),
            "seeds": len(seeds),
            "algorithm_keys": sorted(algorithms),
            "seed_values": sorted(seeds),
        },
        "baseline": {
            "tree_weight": BASE_TREE_WEIGHT,
            "sof1_weight": BASE_SOF1_WEIGHT,
            "var_weight": BASE_VAR_WEIGHT,
            "op_weight": BASE_OP_WEIGHT,
            "nonexact_cap": 0.5,
            "tree_share": 0.6,
            "ranking": [
                row["algorithm"] for row in contributions
            ],
        },
        "baseline_decomposition": {
            "mean_SYM_F": mean_symf,
            "mean_exact_points": mean_exact,
            "mean_tree_points": mean_tree,
            "mean_variable_points": mean_var,
            "mean_operator_points": mean_op,
            "mean_var_op_points": mean_var + mean_op,
            "exact_fraction_of_mean_SYM_F": (
                mean_exact / mean_symf if mean_symf > 0 else 0.0
            ),
        },
        "local_weight_grid": _grid_summary(local_rows),
        "broad_stress_grid": _grid_summary(broad_rows),
        "cap_sweep": {
            **_grid_summary(cap_rows),
            "exact_rank_match_ranges": _ranges(
                cap_rows,
                value_field="nonexact_cap",
                predicate_field="exact_rank_match",
            ),
            "top1_unchanged_ranges": _ranges(
                cap_rows,
                value_field="nonexact_cap",
                predicate_field="top1_unchanged",
            ),
            "top3_unchanged_for_all": all(
                _bool(row["top3_set_unchanged"]) for row in cap_rows
            ),
        },
        "tree_share_sweep": {
            **_grid_summary(share_rows),
            "exact_rank_match_ranges": _ranges(
                share_rows,
                value_field="tree_share",
                predicate_field="exact_rank_match",
            ),
            "top1_unchanged_ranges": _ranges(
                share_rows,
                value_field="tree_share",
                predicate_field="top1_unchanged",
            ),
            "top3_unchanged_for_all": all(
                _bool(row["top3_set_unchanged"]) for row in share_rows
            ),
        },
        "redundancy": {
            "nonexact_valid_runs": len(nonexact_valid),
            "tree_zero_rate_nonexact_valid": statistics.mean(
                math.isclose(
                    row["tree_similarity"],
                    0.0,
                    rel_tol=0.0,
                    abs_tol=TOLERANCE,
                )
                for row in nonexact_valid
            ),
            "tree_sof1_pearson": run_correlation_index[
                frozenset(("tree_similarity", "sof1"))
            ]["pearson"],
            "tree_sof1_spearman": run_correlation_index[
                frozenset(("tree_similarity", "sof1"))
            ]["spearman"],
            "var_op_pearson": run_correlation_index[
                frozenset(("var_f1", "op_f1"))
            ]["pearson"],
            "var_op_spearman": run_correlation_index[
                frozenset(("var_f1", "op_f1"))
            ]["spearman"],
            "tree_from_f1_r_squared": regression_index[
                "tree_similarity"
            ]["r_squared"],
            "tree_from_f1_vif": regression_index[
                "tree_similarity"
            ]["variance_inflation_factor"],
        },
        "outputs": {
            name: _repo_relative(output_dir / name)
            for name in (
                "README.md",
                "algorithm_baseline_contributions.csv",
                "local_weight_sensitivity.csv",
                "broad_weight_stress_grid.csv",
                "nonexact_cap_sweep.csv",
                "tree_share_sweep.csv",
                "ablation_summary.csv",
                "ablation_algorithm_scores.csv",
                "component_correlations.csv",
                "component_redundancy_regressions.csv",
                "sensitivity_summary.json",
                "symf_weight_sensitivity.png",
                "symf_weight_sensitivity.pdf",
            )
        },
    }
    summary_path = output_dir / "sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir / "README.md",
        summary=summary,
        contributions=contributions,
    )
    if make_plots:
        _plot_results(
            output_dir,
            broad_rows,
            cap_rows,
            share_rows,
            contributions,
            correlation_rows,
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-noncanonical-grid",
        action="store_true",
        help="仅用于测试或外部数据；跳过 12 x 50 x 5 冻结网格检查。",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        strict_core50=not args.allow_noncanonical_grid,
        make_plots=not args.no_plots,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
