#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/"
    "e1_12_dataset_algorithm_nmse_table.csv"
)
DEFAULT_OUTPUT = Path(
    "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/"
    "probe4_selection_nmse_only"
)
DEFAULT_CANDIDATE = Path("exp-planning/02.E1选择验证/generated/candidate200_unified.csv")
EXCLUDED_ALGORITHMS = {"drsr", "llmsr"}
LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12.0
LOG_CLIP_MAX = 12.0
EXPLOSION_THRESHOLD = 100.0

TAXONOMY = {
    "dso": "rl_policy",
    "e2esr": "pretrained_neural",
    "gplearn": "classic_gp",
    "imcts": "mcts",
    "pyoperon": "evolutionary_gp",
    "pysr": "evolutionary_gp",
    "qlattice": "graph_hybrid",
    "ragsr": "rag_hybrid",
    "tpsr": "pretrained_neural",
    "udsr": "rl_hybrid",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _global_index(dataset_id: str) -> int | None:
    text = str(dataset_id or "").strip()
    if text.startswith("g") and text[1:].isdigit():
        return int(text[1:])
    return None


def _enrich_with_candidate_metadata(rows: list[dict[str, str]], candidate_path: Path) -> list[dict[str, str]]:
    if not candidate_path.exists():
        return rows
    candidates = _read_csv(candidate_path)
    by_index = {int(row["global_index"]): row for row in candidates if str(row.get("global_index", "")).isdigit()}
    enriched: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        idx = _global_index(out.get("dataset_id", ""))
        cand = by_index.get(idx) if idx is not None else None
        if cand:
            for key in ("family", "subgroup", "basename", "selection_mode", "candidate_advantage_side"):
                if not out.get(key):
                    out[key] = cand.get(key, "")
        enriched.append(out)
    return enriched


def _float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _clipped_log_nmse(value: Any) -> float | None:
    num = _float(value)
    if num is None or num < 0:
        return None
    out = math.log10(max(num, LOG_FLOOR))
    return min(LOG_CLIP_MAX, max(LOG_CLIP_MIN, out))


def _combined_log_score(row: dict[str, str]) -> float | None:
    id_log = _clipped_log_nmse(row.get("id_nmse"))
    ood_log = _clipped_log_nmse(row.get("ood_nmse"))
    if id_log is None or ood_log is None:
        return None
    return 0.5 * id_log + 0.5 * ood_log


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def _iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    ordered = sorted(values)
    q1 = statistics.quantiles(ordered, n=4, method="inclusive")[0]
    q3 = statistics.quantiles(ordered, n=4, method="inclusive")[2]
    return q3 - q1


def _rankdata(items: dict[str, float]) -> dict[str, float]:
    ordered = sorted(items.items(), key=lambda kv: (kv[1], kv[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[i][1], ordered[j][1], abs_tol=1e-12):
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(a: dict[str, float], b: dict[str, float]) -> tuple[float | None, int]:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None, len(keys)
    ar = _rankdata({k: a[k] for k in keys})
    br = _rankdata({k: b[k] for k in keys})
    return _pearson([ar[k] for k in keys], [br[k] for k in keys]), len(keys)


def _normalize_by_max(rows: list[dict[str, Any]], field: str, out_field: str) -> None:
    values = [_float(row.get(field)) for row in rows]
    max_value = max((v for v in values if v is not None), default=0.0)
    for row, value in zip(rows, values):
        row[out_field] = (value / max_value) if value is not None and max_value > 0 else 0.0


def _algorithm_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    by_alg: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        alg = row["algorithm"]
        if alg not in EXCLUDED_ALGORITHMS:
            by_alg[alg].append(row)

    out: list[dict[str, Any]] = []
    score_vectors: dict[str, dict[str, float]] = {}
    for alg, group in sorted(by_alg.items()):
        total = len(group)
        finite = {
            split: sum(_float(row.get(f"{split}_nmse")) is not None for row in group)
            for split in ("train", "valid", "id", "ood")
        }
        train_id_ood = sum(
            all(_float(row.get(f"{split}_nmse")) is not None for split in ("train", "id", "ood"))
            for row in group
        )
        id_ood = sum(
            _float(row.get("id_nmse")) is not None and _float(row.get("ood_nmse")) is not None
            for row in group
        )
        any_id_ood_explosion = sum(
            any(
                (value is not None and value > EXPLOSION_THRESHOLD)
                for value in (_float(row.get("id_nmse")), _float(row.get("ood_nmse")))
            )
            for row in group
        )
        any_train_id_ood_explosion = sum(
            any(
                (value is not None and value > EXPLOSION_THRESHOLD)
                for value in (
                    _float(row.get("train_nmse")),
                    _float(row.get("id_nmse")),
                    _float(row.get("ood_nmse")),
                )
            )
            for row in group
        )
        scores: list[float] = []
        vector: dict[str, float] = {}
        for row in group:
            score = _combined_log_score(row)
            if score is None:
                continue
            dataset_id = row["dataset_id"]
            scores.append(score)
            vector[dataset_id] = score
        score_vectors[alg] = vector

        family_stats = _coverage_stats(group, "family")
        subgroup_stats = _coverage_stats(group, "subgroup")
        family_mean_values = list(family_stats["mean_scores"].values())
        subgroup_mean_values = list(subgroup_stats["mean_scores"].values())
        out.append(
            {
                "algorithm": alg,
                "taxonomy": TAXONOMY.get(alg, "unknown"),
                "rows": total,
                "finite_train_rate": _rate(finite["train"], total),
                "finite_valid_rate": _rate(finite["valid"], total),
                "finite_id_rate": _rate(finite["id"], total),
                "finite_ood_rate": _rate(finite["ood"], total),
                "finite_id_ood_rate": _rate(id_ood, total),
                "train_id_ood_present_rate": _rate(train_id_ood, total),
                "id_ood_explosion_rate_gt_100": _rate(any_id_ood_explosion, total),
                "train_id_ood_explosion_rate_gt_100": _rate(any_train_id_ood_explosion, total),
                "median_combined_log_id_ood_nmse": _median(scores),
                "mean_combined_log_id_ood_nmse": _mean(scores),
                "std_combined_log_id_ood_nmse": _stdev(scores),
                "iqr_combined_log_id_ood_nmse": _iqr(scores),
                "family_mean_score_std": _stdev(family_mean_values),
                "subgroup_mean_score_std": _stdev(subgroup_mean_values),
                "families_with_finite_id_ood_rate_ge_0_9": family_stats["groups_ge_0_9"],
                "subgroups_with_finite_id_ood_rate_ge_0_9": subgroup_stats["groups_ge_0_9"],
                "family_coverage_rate_ge_0_9": family_stats["coverage_rate_ge_0_9"],
                "subgroup_coverage_rate_ge_0_9": subgroup_stats["coverage_rate_ge_0_9"],
            }
        )

    _normalize_by_max(out, "std_combined_log_id_ood_nmse", "dataset_discrimination_std_norm")
    _normalize_by_max(out, "iqr_combined_log_id_ood_nmse", "dataset_discrimination_iqr_norm")
    _normalize_by_max(out, "family_mean_score_std", "family_discrimination_norm")
    _normalize_by_max(out, "subgroup_mean_score_std", "subgroup_discrimination_norm")
    for row in out:
        row["operational_stability_score"] = (
            0.60 * row["finite_id_ood_rate"] + 0.40 * row["train_id_ood_present_rate"]
        )
        row["discrimination_score"] = (
            0.45 * row["dataset_discrimination_std_norm"]
            + 0.25 * row["dataset_discrimination_iqr_norm"]
            + 0.15 * row["family_discrimination_norm"]
            + 0.15 * row["subgroup_discrimination_norm"]
        )
        row["coverage_score"] = 0.60 * row["family_coverage_rate_ge_0_9"] + 0.40 * row[
            "subgroup_coverage_rate_ge_0_9"
        ]
        median_score = _float(row["median_combined_log_id_ood_nmse"])
        row["baseline_quality_score"] = (
            max(0.0, min(1.0, (LOG_CLIP_MAX - median_score) / (LOG_CLIP_MAX - LOG_CLIP_MIN)))
            if median_score is not None
            else 0.0
        )
        row["available_metric_score"] = (
            0.15 * row["operational_stability_score"]
            + 0.30 * row["discrimination_score"]
            + 0.15 * row["coverage_score"]
            + 0.05 * row["baseline_quality_score"]
            - 0.10 * (1.0 - row["finite_id_ood_rate"])
        )
    out.sort(key=lambda row: row["available_metric_score"], reverse=True)
    return out, score_vectors


def _coverage_stats(group: list[dict[str, str]], field: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in group:
        by_group[row.get(field, "")].append(row)
    mean_scores: dict[str, float] = {}
    groups_ge_0_9 = 0
    for name, rows in by_group.items():
        finite = 0
        scores: list[float] = []
        for row in rows:
            score = _combined_log_score(row)
            if score is not None:
                finite += 1
                scores.append(score)
        if _rate(finite, len(rows)) >= 0.9:
            groups_ge_0_9 += 1
        if scores:
            mean_scores[name] = statistics.mean(scores)
    return {
        "groups_total": len(by_group),
        "groups_ge_0_9": groups_ge_0_9,
        "coverage_rate_ge_0_9": _rate(groups_ge_0_9, len(by_group)),
        "mean_scores": mean_scores,
    }


def _pairwise_rows(score_vectors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(sorted(score_vectors), 2):
        rho, overlap = _spearman(score_vectors[a], score_vectors[b])
        complementarity = None if rho is None else 1.0 - abs(rho)
        rows.append(
            {
                "algorithm_a": a,
                "algorithm_b": b,
                "spearman_combined_log_score": rho,
                "overlap_dataset_count": overlap,
                "complementarity_score": complementarity,
            }
        )
    rows.sort(key=lambda row: (-1 if row["complementarity_score"] is None else row["complementarity_score"]), reverse=True)
    return rows


def _combo_rows(algorithm_rows: list[dict[str, Any]], pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_by_alg = {row["algorithm"]: row for row in algorithm_rows}
    pair_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pairwise_rows:
        key = tuple(sorted([row["algorithm_a"], row["algorithm_b"]]))
        pair_key[key] = row

    out: list[dict[str, Any]] = []
    for combo in itertools.combinations(sorted(row_by_alg), 4):
        alg_rows = [row_by_alg[alg] for alg in combo]
        pair_values = [
            _float(pair_key[tuple(sorted(pair))].get("complementarity_score"))
            for pair in itertools.combinations(combo, 2)
        ]
        pair_values = [v for v in pair_values if v is not None]
        taxonomies = [row_by_alg[alg]["taxonomy"] for alg in combo]
        taxonomy_counts = Counter(taxonomies)
        taxonomy_diversity_score = len(taxonomy_counts) / 4.0
        taxonomy_redundancy_penalty = max(0, max(taxonomy_counts.values()) - 2) * 0.10
        stability = statistics.mean(row["operational_stability_score"] for row in alg_rows)
        discrimination = statistics.mean(row["discrimination_score"] for row in alg_rows)
        coverage = statistics.mean(row["coverage_score"] for row in alg_rows)
        quality = statistics.mean(row["baseline_quality_score"] for row in alg_rows)
        finite_rate = statistics.mean(row["finite_id_ood_rate"] for row in alg_rows)
        complementarity = statistics.mean(pair_values) if pair_values else 0.0
        practical_cost_score = 0.5
        combo_score = (
            0.15 * stability
            + 0.30 * discrimination
            + 0.25 * complementarity
            + 0.15 * coverage
            + 0.10 * practical_cost_score
            + 0.05 * quality
            + 0.05 * taxonomy_diversity_score
            - 0.10 * (1.0 - finite_rate)
            - taxonomy_redundancy_penalty
        )
        out.append(
            {
                "combo": ";".join(combo),
                "combo_score_nmse_only": combo_score,
                "mean_operational_stability": stability,
                "mean_discrimination": discrimination,
                "mean_pairwise_complementarity": complementarity,
                "mean_family_subgroup_coverage": coverage,
                "mean_baseline_quality": quality,
                "mean_finite_id_ood_rate": finite_rate,
                "taxonomy_diversity_score": taxonomy_diversity_score,
                "taxonomy_redundancy_penalty": taxonomy_redundancy_penalty,
                "practical_cost_score_neutral_due_to_missing_runtime": practical_cost_score,
                "taxonomies": ";".join(taxonomies),
            }
        )
    out.sort(key=lambda row: row["combo_score_nmse_only"], reverse=True)
    return out


def _family_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selected = [row for row in rows if row["algorithm"] not in EXCLUDED_ALGORITHMS]
    by_alg_family: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_alg_family[(row["algorithm"], row["family"])].append(row)
    for (alg, family), group in sorted(by_alg_family.items()):
        total = len(group)
        scores = [_combined_log_score(row) for row in group]
        scores = [score for score in scores if score is not None]
        id_ood = sum(
            _float(row.get("id_nmse")) is not None and _float(row.get("ood_nmse")) is not None
            for row in group
        )
        explosion = sum(
            any(
                (value is not None and value > EXPLOSION_THRESHOLD)
                for value in (_float(row.get("id_nmse")), _float(row.get("ood_nmse")))
            )
            for row in group
        )
        out.append(
            {
                "algorithm": alg,
                "family": family,
                "rows": total,
                "finite_id_ood_rate": _rate(id_ood, total),
                "id_ood_explosion_rate_gt_100": _rate(explosion, total),
                "median_combined_log_id_ood_nmse": _median(scores),
                "iqr_combined_log_id_ood_nmse": _iqr(scores),
            }
        )
    return out


def _write_report(
    output_dir: Path,
    input_path: Path,
    candidate_path: Path,
    algorithm_rows: list[dict[str, Any]],
    combo_rows: list[dict[str, Any]],
) -> None:
    top_alg = algorithm_rows[:10]
    top_combo = combo_rows[:10]
    lines = [
        "# Probe-4 Selection Metrics",
        "",
        "## Scope",
        "",
        f"- Input: `{input_path}`",
        f"- Candidate metadata: `{candidate_path}`",
        "- Excluded algorithms: `drsr`, `llmsr`",
        "- Current report is `NMSE-only`: latest 20260429 digest has no runtime/status/failure fields.",
        "- Practical cost is therefore set to a neutral constant in combo scoring.",
        "",
        "## Output Files",
        "",
        "- `probe4_algorithm_scores_nmse_only.csv`",
        "- `probe4_pairwise_complementarity_nmse_only.csv`",
        "- `probe4_combo_scores_nmse_only.csv`",
        "- `probe4_family_coverage_nmse_only.csv`",
        "",
        "## Top Algorithms By Available Metric Score",
        "",
        "| rank | algorithm | taxonomy | score | finite_id_ood | explosion_gt_100 | discrimination | coverage |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(top_alg, 1):
        lines.append(
            "| {rank} | {algorithm} | {taxonomy} | {score:.4f} | {finite:.3f} | {explosion:.3f} | {disc:.3f} | {coverage:.3f} |".format(
                rank=i,
                algorithm=row["algorithm"],
                taxonomy=row["taxonomy"],
                score=row["available_metric_score"],
                finite=row["finite_id_ood_rate"],
                explosion=row["id_ood_explosion_rate_gt_100"],
                disc=row["discrimination_score"],
                coverage=row["coverage_score"],
            )
        )
    lines.extend(
        [
            "",
            "## Top 4-Algorithm Combos",
            "",
            "| rank | combo | score | stability | discrimination | complementarity | coverage | finite_id_ood |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, row in enumerate(top_combo, 1):
        lines.append(
            "| {rank} | `{combo}` | {score:.4f} | {stability:.3f} | {disc:.3f} | {comp:.3f} | {coverage:.3f} | {finite:.3f} |".format(
                rank=i,
                combo=row["combo"],
                score=row["combo_score_nmse_only"],
                stability=row["mean_operational_stability"],
                disc=row["mean_discrimination"],
                comp=row["mean_pairwise_complementarity"],
                coverage=row["mean_family_subgroup_coverage"],
                finite=row["mean_finite_id_ood_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This should be used as the first pass for Probe-4 selection, not as the final decision.",
            "- Runtime/status based operational stability should be added once raw result tables are available.",
            "- `valid_extreme_error` is treated as an informative finite result; missing/nonfinite metrics are penalized.",
        ]
    )
    (output_dir / "probe4_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()

    rows = _enrich_with_candidate_metadata(_read_csv(args.input), args.candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_rows, score_vectors = _algorithm_rows(rows)
    pairwise = _pairwise_rows(score_vectors)
    combos = _combo_rows(algorithm_rows, pairwise)
    family = _family_rows(rows)

    _write_csv(args.output_dir / "probe4_algorithm_scores_nmse_only.csv", algorithm_rows)
    _write_csv(args.output_dir / "probe4_pairwise_complementarity_nmse_only.csv", pairwise)
    _write_csv(args.output_dir / "probe4_combo_scores_nmse_only.csv", combos)
    _write_csv(args.output_dir / "probe4_family_coverage_nmse_only.csv", family)
    _write_report(args.output_dir, args.input, args.candidate, algorithm_rows, combos)
    print(
        {
            "output_dir": str(args.output_dir),
            "algorithms": len(algorithm_rows),
            "combos": len(combos),
        }
    )


if __name__ == "__main__":
    main()
