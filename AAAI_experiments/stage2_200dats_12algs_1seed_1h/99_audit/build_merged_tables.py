#!/usr/bin/env python3
"""从整理包里的长表生成便于人工查看的算法横向大表。"""

from __future__ import annotations

import csv
import math
from collections import OrderedDict, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "06_merged_tables"

ALG12 = [
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
]
PROBE4 = ["dso", "imcts", "pyoperon", "udsr"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def pivot_long_to_wide(
    rows: list[dict[str, str]],
    *,
    key_cols: list[str],
    pivot_col: str,
    pivot_values: list[str],
    value_cols: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    grouped: OrderedDict[tuple[str, ...], dict[str, str]] = OrderedDict()

    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        if key not in grouped:
            grouped[key] = {col: row.get(col, "") for col in key_cols}

        pivot_value = row.get(pivot_col, "")
        if pivot_value not in pivot_values:
            continue

        out_row = grouped[key]
        for col in value_cols:
            out_row[f"{pivot_value}__{col}"] = row.get(col, "")

    fieldnames = list(key_cols)
    for pivot_value in pivot_values:
        fieldnames.extend(f"{pivot_value}__{col}" for col in value_cols)

    return fieldnames, list(grouped.values())


def candidate200_manifest_by_id() -> OrderedDict[str, dict[str, str]]:
    manifest_rows = read_csv(BASE / "02_candidate200_inputs/candidate200_unified_with_paths.csv")
    manifest: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in manifest_rows:
        try:
            dataset_id = f"g{int(row.get('global_index', '')):04d}"
        except ValueError:
            continue
        manifest[dataset_id] = {
            "dataset_id": dataset_id,
            "global_index": row.get("global_index", ""),
            "dataset_name": row.get("dataset_name", ""),
            "dataset_rel": row.get("dataset_rel", ""),
            "family": row.get("family", ""),
            "subgroup": row.get("subgroup", ""),
            "basename": row.get("basename", ""),
            "pool": row.get("pool", ""),
            "selection_mode": row.get("selection_mode", ""),
            "candidate_advantage_side": row.get("candidate_advantage_side", ""),
        }
    return manifest


def pivot_candidate200_rows(
    rows: list[dict[str, str]],
    *,
    pivot_values: list[str],
    value_cols: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    manifest = candidate200_manifest_by_id()
    meta_cols = [
        "dataset_id",
        "global_index",
        "dataset_name",
        "dataset_rel",
        "family",
        "subgroup",
        "basename",
        "pool",
        "selection_mode",
        "candidate_advantage_side",
    ]
    observed_cols = [
        "dataset_name",
        "family",
        "subgroup",
        "srsd_variant",
        "basename",
        "selection_mode",
        "candidate_advantage_side",
    ]
    observed_suffix_cols = [f"observed_{col}_values" for col in observed_cols]

    grouped: OrderedDict[str, dict[str, str]] = OrderedDict()
    observed_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        dataset_id = row.get("dataset_id", "")
        if not dataset_id:
            continue
        if dataset_id not in grouped:
            grouped[dataset_id] = dict(manifest.get(dataset_id, {"dataset_id": dataset_id}))

        for col in observed_cols:
            value = row.get(col, "")
            if value:
                observed_values[dataset_id][col].add(value)

        pivot_value = row.get("algorithm", "")
        if pivot_value not in pivot_values:
            continue
        for col in value_cols:
            grouped[dataset_id][f"{pivot_value}__{col}"] = row.get(col, "")

    for dataset_id, out_row in grouped.items():
        for col in observed_cols:
            out_row[f"observed_{col}_values"] = ";".join(sorted(observed_values[dataset_id].get(col, set())))

    fieldnames = meta_cols + observed_suffix_cols
    for pivot_value in pivot_values:
        fieldnames.extend(f"{pivot_value}__{col}" for col in value_cols)

    # 以 Candidate-200 manifest 的顺序输出，避免算法表内部顺序抖动。
    ordered_rows = [grouped[dataset_id] for dataset_id in manifest if dataset_id in grouped]
    ordered_rows.extend(row for dataset_id, row in grouped.items() if dataset_id not in manifest)
    return fieldnames, ordered_rows


def to_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def combined_score(row: dict[str, str]) -> float | None:
    values = [to_float(row.get("id_log_nmse_used", "")), to_float(row.get("ood_log_nmse_used", ""))]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def build_candidate200_v02() -> None:
    rows = read_csv(BASE / "01_v02_selection_digest/probe4_run_level_big_table_2400.csv")
    value_cols = [
        "taxonomy",
        "prompt_semantics_mode",
        "llm_model_assignment",
        "train_nmse",
        "id_nmse",
        "ood_nmse",
        "log_train_nmse_clipped",
        "log_id_nmse_clipped",
        "log_ood_nmse_clipped",
        "combined_log_id_ood_nmse",
        "gap_log_ood_minus_id",
        "probe4_success",
        "normalized_status_for_probe4",
        "budget_exhausted",
        "wall_time_seconds",
        "expression_artifact_available",
        "equation_present",
        "artifact_valid",
        "sympy_parse_ok",
        "expression_health_label",
        "used_variable_count",
        "variable_coverage_ratio",
        "operator_set",
        "normalized_expression_preview",
        "source_result_path",
    ]
    fieldnames, wide_rows = pivot_candidate200_rows(
        rows,
        pivot_values=ALG12,
        value_cols=value_cols,
    )
    write_csv(OUT / "candidate200_v02_by_dataset_wide_12alg.csv", fieldnames, wide_rows)


def build_e1_12alg_nmse() -> None:
    rows = read_csv(BASE / "03_e1_12alg_calibration/e1_12_dataset_algorithm_nmse_table_2400.csv")
    value_cols = [
        "train_nmse",
        "valid_nmse",
        "id_nmse",
        "ood_nmse",
        "finite_train",
        "finite_id",
        "finite_ood",
        "finite_id_ood",
        "finite_train_id_ood",
        "log_train_nmse_clipped",
        "log_id_nmse_clipped",
        "log_ood_nmse_clipped",
        "combined_log_id_ood_nmse",
        "gap_log_ood_minus_id",
        "delta_id_minus_train",
        "delta_ood_minus_id",
        "id_ood_explosion_gt_100",
        "train_id_ood_explosion_gt_100",
    ]
    fieldnames, wide_rows = pivot_candidate200_rows(
        rows,
        pivot_values=ALG12,
        value_cols=value_cols,
    )
    write_csv(OUT / "candidate200_e1_nmse_by_dataset_wide_12alg.csv", fieldnames, wide_rows)


def build_full664_by_dataset_seed() -> None:
    rows = read_csv(BASE / "05_downstream_freeze_probe4_full664/postprocess_run_level_7968.csv")
    key_cols = [
        "dataset_id",
        "global_index",
        "dataset_name",
        "family",
        "subgroup",
        "metadata_class",
        "dataset_rel",
        "basename",
        "formula_identity",
        "feature_count",
        "target_name",
        "train_samples",
        "valid_samples",
        "id_test_samples",
        "ood_test_samples",
        "dataset_key",
        "seed_norm",
    ]
    value_cols = [
        "is_finished_run",
        "is_evaluable_run",
        "not_finished_flag",
        "train_log_nmse_used",
        "valid_log_nmse_used",
        "id_log_nmse_used",
        "ood_log_nmse_used",
        "extreme_error_flag",
        "run_outcome_class",
        "failure_reason_normalized",
        "result_status_raw",
        "state_raw",
        "timeout_type_raw",
        "result_seconds",
        "id_r2",
        "ood_r2",
        "result_equation",
        "result_complexity",
        "result_tree_depth",
        "synthetic_missing_row",
    ]
    fieldnames, wide_rows = pivot_long_to_wide(
        rows,
        key_cols=key_cols,
        pivot_col="method_norm",
        pivot_values=PROBE4,
        value_cols=value_cols,
    )
    write_csv(OUT / "full664_probe4_by_dataset_seed_wide_4alg.csv", fieldnames, wide_rows)


def build_full664_mean_over_seed() -> None:
    rows = read_csv(BASE / "05_downstream_freeze_probe4_full664/postprocess_run_level_7968.csv")
    key_cols = [
        "dataset_id",
        "global_index",
        "dataset_name",
        "family",
        "subgroup",
        "metadata_class",
        "dataset_rel",
        "basename",
        "formula_identity",
        "feature_count",
        "target_name",
        "train_samples",
        "valid_samples",
        "id_test_samples",
        "ood_test_samples",
        "dataset_key",
    ]
    numeric_cols = [
        "train_log_nmse_used",
        "valid_log_nmse_used",
        "id_log_nmse_used",
        "ood_log_nmse_used",
        "id_r2",
        "ood_r2",
        "result_seconds",
        "result_complexity",
        "result_tree_depth",
    ]

    grouped: OrderedDict[tuple[str, ...], dict[str, str]] = OrderedDict()
    per_method: dict[tuple[tuple[str, ...], str], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = tuple(row.get(col, "") for col in key_cols)
        if key not in grouped:
            grouped[key] = {col: row.get(col, "") for col in key_cols}
        method = row.get("method_norm", "")
        if method in PROBE4:
            per_method[(key, method)].append(row)

    fieldnames = list(key_cols)
    method_fields = []
    for method in PROBE4:
        method_fields.extend(
            [
                f"{method}__observed_runs",
                f"{method}__finished_runs",
                f"{method}__evaluable_runs",
                f"{method}__valid_finite_result_runs",
                f"{method}__partial_output_runs",
                f"{method}__timeout_no_output_runs",
                f"{method}__valid_extreme_error_runs",
                f"{method}__run_outcome_classes",
            ]
        )
        method_fields.extend(f"{method}__mean_{col}" for col in numeric_cols)
        method_fields.extend(
            [
                f"{method}__best_seed",
                f"{method}__best_combined_log_id_ood",
                f"{method}__best_id_log_nmse_used",
                f"{method}__best_ood_log_nmse_used",
                f"{method}__best_run_outcome_class",
                f"{method}__best_equation",
            ]
        )
    fieldnames.extend(method_fields)

    out_rows: list[dict[str, str]] = []
    for key, out_row in grouped.items():
        row_out = dict(out_row)
        for method in PROBE4:
            method_rows = per_method.get((key, method), [])
            row_out[f"{method}__observed_runs"] = str(len(method_rows))
            row_out[f"{method}__finished_runs"] = str(sum(truthy(row.get("is_finished_run", "")) for row in method_rows))
            row_out[f"{method}__evaluable_runs"] = str(sum(truthy(row.get("is_evaluable_run", "")) for row in method_rows))
            row_out[f"{method}__valid_finite_result_runs"] = str(sum(row.get("run_outcome_class") == "valid_finite_result" for row in method_rows))
            row_out[f"{method}__partial_output_runs"] = str(sum(row.get("run_outcome_class") == "partial_output" for row in method_rows))
            row_out[f"{method}__timeout_no_output_runs"] = str(sum(row.get("run_outcome_class") == "timeout_no_output" for row in method_rows))
            row_out[f"{method}__valid_extreme_error_runs"] = str(sum(row.get("run_outcome_class") == "valid_extreme_error" for row in method_rows))
            row_out[f"{method}__run_outcome_classes"] = ";".join(sorted({row.get("run_outcome_class", "") for row in method_rows if row.get("run_outcome_class", "")}))

            for col in numeric_cols:
                values = [to_float(row.get(col, "")) for row in method_rows]
                values = [value for value in values if value is not None]
                row_out[f"{method}__mean_{col}"] = "" if not values else f"{sum(values) / len(values):.12g}"

            scored_rows = [(combined_score(row), row) for row in method_rows]
            scored_rows = [(score, row) for score, row in scored_rows if score is not None]
            if scored_rows:
                best_score, best_row = min(scored_rows, key=lambda item: item[0])
                row_out[f"{method}__best_seed"] = best_row.get("seed_norm", "")
                row_out[f"{method}__best_combined_log_id_ood"] = f"{best_score:.12g}"
                row_out[f"{method}__best_id_log_nmse_used"] = best_row.get("id_log_nmse_used", "")
                row_out[f"{method}__best_ood_log_nmse_used"] = best_row.get("ood_log_nmse_used", "")
                row_out[f"{method}__best_run_outcome_class"] = best_row.get("run_outcome_class", "")
                row_out[f"{method}__best_equation"] = best_row.get("result_equation", "")
        out_rows.append(row_out)

    write_csv(OUT / "full664_probe4_by_dataset_mean_over_seed_wide_4alg.csv", fieldnames, out_rows)


def main() -> None:
    build_candidate200_v02()
    build_e1_12alg_nmse()


if __name__ == "__main__":
    main()
