#!/usr/bin/env python3
"""Reproduce and analyze SymbolicArena OOD-G weight sensitivity.

The analysis reconstructs the submitted Figure-3 clean input as:

    original 3000 runs
    - original DRSR/LLM-SR 500 runs
    + model-split rerun DRSR/LLM-SR 500 runs

It never launches an SR algorithm. All computations are deterministic local
post-processing except for the task bootstrap, whose random seed is fixed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))

STAGE4_ROOT = REPO_ROOT / "A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h"
FORMAL_METRIC_IMPLEMENTATION = REPO_ROOT / "check/plot_core50_hexagon_metrics.py"
DEFAULT_ORIGINAL_RUNS = (
    STAGE4_ROOT / "collected_results/core50_12alg_run_level_log_nmse_20260503.csv"
)
DEFAULT_REPLACEMENT_RUNS = (
    REPO_ROOT
    / "exp-planning/04.Core50正式全量评测/analysis/"
    "hexagon_v1_with_artifacts_20260504/clean_modelsplit_artifact_runs.csv"
)
DEFAULT_EXPECTED_MERGED_RUNS = (
    REPO_ROOT
    / "exp-planning/04.Core50正式全量评测/analysis/"
    "hexagon_v1_with_artifacts_20260504/clean_final_runs_updated.csv"
)
DEFAULT_CORE50_MANIFEST = STAGE4_ROOT / "core50_manifest/core50_datasets.csv"
DEFAULT_ARCHIVE_RESULTS = STAGE4_ROOT / "run_archive/results"
DEFAULT_FORMAL_COMPONENTS = STAGE4_ROOT / "hexagon/dataset_axis_components_formal.csv"
DEFAULT_FORMAL_SCORES = STAGE4_ROOT / "hexagon/hexagon_scores_formal.csv"

REPLACED_ALGORITHMS = ("drsr", "llmsr")
EXPECTED_ALGORITHMS = (
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
KEY_COLUMNS = ["algorithm", "gid", "dataset", "seed"]
RUN_COLUMNS = [
    "algorithm",
    "gid",
    "dataset",
    "seed",
    "status",
    "valid_output",
    "seconds",
    "train_nmse",
    "valid_nmse",
    "id_test_nmse",
    "ood_test_nmse",
    "metric_complete",
]
FULL_SCAN_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
CANDIDATE_WEIGHTS = np.array([0.70, 0.80, 0.85, 0.875, 0.90, 0.95, 1.00])
ALL_WEIGHTS = np.array(sorted(set(FULL_SCAN_WEIGHTS.tolist() + CANDIDATE_WEIGHTS.tolist())))
REVERSAL_FACTORS = (10, 100, 1000, 10000)
EPSILON = 1e-12
FLOAT_TOL = 1e-12
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPS = 1000


def load_formal_metric_functions() -> tuple[Any, Any]:
    """直接从正式脚本提取两个纯函数，避免导入其重型符号分析依赖。"""
    tree = ast.parse(
        FORMAL_METRIC_IMPLEMENTATION.read_text(encoding="utf-8"),
        filename=str(FORMAL_METRIC_IMPLEMENTATION),
    )
    required = {"phi_from_nmse", "safe_nmse"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in required
    ]
    found = {node.name for node in definitions}
    if found != required:
        raise RuntimeError(
            f"Could not extract formal metric functions; found={sorted(found)}"
        )
    module = ast.Module(body=definitions, type_ignores=[])
    namespace: dict[str, Any] = {
        "Any": Any,
        "math": math,
        "np": np,
        "pd": pd,
    }
    exec(
        compile(module, str(FORMAL_METRIC_IMPLEMENTATION), "exec"),
        namespace,
        namespace,
    )
    return namespace["phi_from_nmse"], namespace["safe_nmse"]


phi_from_nmse, safe_nmse = load_formal_metric_functions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline OOD-G weight sensitivity analysis. The quality weight w is "
            "scanned from 0 to 1; no SR algorithms are launched."
        )
    )
    parser.add_argument("--original-runs", type=Path, default=DEFAULT_ORIGINAL_RUNS)
    parser.add_argument("--replacement-runs", type=Path, default=DEFAULT_REPLACEMENT_RUNS)
    parser.add_argument("--expected-merged-runs", type=Path, default=DEFAULT_EXPECTED_MERGED_RUNS)
    parser.add_argument("--core50-manifest", type=Path, default=DEFAULT_CORE50_MANIFEST)
    parser.add_argument("--archive-results", type=Path, default=DEFAULT_ARCHIVE_RESULTS)
    parser.add_argument("--formal-components", type=Path, default=DEFAULT_FORMAL_COMPONENTS)
    parser.add_argument("--formal-scores", type=Path, default=DEFAULT_FORMAL_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def normalize_algorithm(value: Any) -> str:
    text = str(value).strip().lower()
    return {"qlattice": "qlattice", "imcts": "imcts"}.get(text, text)


def normalize_gid(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("g") and text[1:].isdigit():
        return f"g{int(text[1:]):04d}"
    if text.isdigit():
        return f"g{int(text):04d}"
    raise ValueError(f"Invalid gid: {value!r}")


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return values.astype("string").str.strip().str.lower().isin(truthy)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out >= 0 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "matplotlib"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def normalize_run_frame(frame: pd.DataFrame, source_version: str) -> pd.DataFrame:
    missing = sorted(set(RUN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Run table is missing required columns: {missing}")
    out = frame.copy()
    out["algorithm"] = out["algorithm"].map(normalize_algorithm)
    out["gid"] = out["gid"].map(normalize_gid)
    out["dataset"] = out["dataset"].astype(str)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["valid_output"] = bool_series(out["valid_output"])
    out["metric_complete"] = bool_series(out["metric_complete"])
    for column in ["seconds", "train_nmse", "valid_nmse", "id_test_nmse", "ood_test_nmse"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["status"] = out["status"].astype("string").fillna("unknown")
    out["source_version"] = source_version
    return out


def load_archive_failure_details(root: Path) -> pd.DataFrame:
    """读取原始 JSON 的失败语义，避免仅凭聚合 CSV 猜测失败原因。"""
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tool = payload.get("tool")
        seed = payload.get("seed")
        index = payload.get("task_global_index")
        if tool is None or seed is None or index is None:
            continue
        try:
            gid = f"g{int(index):04d}"
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "algorithm": normalize_algorithm(tool),
                "gid": gid,
                "seed": int(seed),
                "raw_status": payload.get("status"),
                "error": payload.get("error"),
                "no_valid_output_reason": payload.get("no_valid_output_reason"),
                "termination_reason": payload.get("termination_reason"),
                "timeout_type": payload.get("timeout_type"),
                "raw_result_json": str(path.resolve()),
            }
        )
    details = pd.DataFrame(rows)
    if details.empty:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "gid",
                "seed",
                "raw_status",
                "error",
                "no_valid_output_reason",
                "termination_reason",
                "timeout_type",
                "raw_result_json",
            ]
        )
    if details.duplicated(["algorithm", "gid", "seed"]).any():
        raise ValueError("Archive failure lookup contains duplicate algorithm/gid/seed keys")
    return details


def derive_failure_reason(row: pd.Series) -> str:
    status = str(row.get("status", "unknown"))
    valid_output = bool(row.get("valid_output", False))
    metric_complete = bool(row.get("metric_complete", False))
    id_nmse = finite_float(row.get("id_test_nmse"))
    ood_nmse = finite_float(row.get("ood_test_nmse"))
    is_failure = status != "ok" or not valid_output or not metric_complete
    if not is_failure and id_nmse is not None and ood_nmse is not None:
        return ""

    reasons: list[str] = []
    for column in ["error", "no_valid_output_reason"]:
        value = row.get(column)
        if value is not None and str(value).strip() not in {"", "nan", "None", "<NA>"}:
            reasons.append(f"{column}={str(value).strip()}")
    if status != "ok":
        reasons.append(f"status={status}")
    if not valid_output:
        reasons.append("valid_output=false")
    if not metric_complete:
        reasons.append("metric_complete=false")
    if id_nmse is None:
        reasons.append("id_test_nmse=missing_or_nonfinite")
    if ood_nmse is None:
        reasons.append("ood_test_nmse=missing_or_nonfinite")
    for column in ["termination_reason", "timeout_type"]:
        value = row.get(column)
        if value is not None and str(value).strip() not in {"", "nan", "None", "<NA>"}:
            reasons.append(f"{column}={str(value).strip()}")
    return "; ".join(dict.fromkeys(reasons))


def validate_unique_grid(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate run keys: {sample}")


def load_and_merge_inputs(
    original_path: Path,
    replacement_path: Path,
    expected_merged_path: Path | None,
    archive_results: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original = normalize_run_frame(pd.read_csv(original_path), "original_archive_20260502")
    replacement_raw = pd.read_csv(replacement_path)
    replacement = normalize_run_frame(
        replacement_raw,
        "drsr_llmsr_modelsplit_rerun_20260503",
    )
    validate_unique_grid(original, "original")
    validate_unique_grid(replacement, "replacement")

    replacement_algorithms = tuple(sorted(replacement["algorithm"].unique()))
    if replacement_algorithms != tuple(sorted(REPLACED_ALGORITHMS)):
        raise ValueError(
            f"Replacement contains {replacement_algorithms}, expected {REPLACED_ALGORITHMS}"
        )
    if len(original) != 3000 or len(replacement) != 500:
        raise ValueError(
            f"Unexpected input sizes: original={len(original)}, replacement={len(replacement)}"
        )

    old_replaced = original[original["algorithm"].isin(REPLACED_ALGORITHMS)].copy()
    old_keys = set(map(tuple, old_replaced[KEY_COLUMNS].itertuples(index=False, name=None)))
    new_keys = set(map(tuple, replacement[KEY_COLUMNS].itertuples(index=False, name=None)))
    if old_keys != new_keys:
        missing = sorted(old_keys - new_keys)[:10]
        extra = sorted(new_keys - old_keys)[:10]
        raise ValueError(f"Replacement key mismatch; missing={missing}, extra={extra}")

    untouched = original[~original["algorithm"].isin(REPLACED_ALGORITHMS)].copy()
    merged = pd.concat([untouched, replacement], ignore_index=True, sort=False)
    validate_unique_grid(merged, "merged")
    merged = merged.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    raw_details = (
        load_archive_failure_details(archive_results)
        if archive_results is not None and archive_results.exists()
        else pd.DataFrame()
    )
    if not raw_details.empty:
        merged = merged.merge(
            raw_details,
            on=["algorithm", "gid", "seed"],
            how="left",
            validate="one_to_one",
        )
        replacement_mask = merged["source_version"].eq(
            "drsr_llmsr_modelsplit_rerun_20260503"
        )
        for column in [
            "raw_status",
            "error",
            "no_valid_output_reason",
            "termination_reason",
            "timeout_type",
            "raw_result_json",
        ]:
            merged.loc[replacement_mask, column] = pd.NA
    else:
        for column in [
            "raw_status",
            "error",
            "no_valid_output_reason",
            "termination_reason",
            "timeout_type",
            "raw_result_json",
        ]:
            merged[column] = pd.NA

    replacement_metadata = replacement_raw.copy()
    replacement_metadata["algorithm"] = replacement_metadata["algorithm"].map(normalize_algorithm)
    replacement_metadata["gid"] = replacement_metadata["gid"].map(normalize_gid)
    replacement_metadata["seed"] = pd.to_numeric(
        replacement_metadata["seed"], errors="raise"
    ).astype(int)
    metadata_columns = [
        column
        for column in ["algorithm", "gid", "seed", "source_batch", "model_split", "result_path"]
        if column in replacement_metadata.columns
    ]
    if len(metadata_columns) > 3:
        replacement_metadata = replacement_metadata[metadata_columns].rename(
            columns={"result_path": "replacement_result_path"}
        )
        merged = merged.merge(
            replacement_metadata,
            on=["algorithm", "gid", "seed"],
            how="left",
            validate="one_to_one",
        )
        replacement_path_available = merged["replacement_result_path"].notna()
        merged.loc[replacement_path_available, "raw_result_json"] = merged.loc[
            replacement_path_available, "replacement_result_path"
        ]

    merged["failure_reason"] = merged.apply(derive_failure_reason, axis=1)
    merged["analysis_penalty_applied"] = (
        ~merged["valid_output"]
        | ~merged["metric_complete"]
        | ~np.isfinite(merged["id_test_nmse"])
        | ~np.isfinite(merged["ood_test_nmse"])
        | (merged["id_test_nmse"] < 0)
        | (merged["ood_test_nmse"] < 0)
    )

    comparison: dict[str, Any] = {
        "expected_merged_available": bool(
            expected_merged_path is not None and expected_merged_path.exists()
        ),
        "max_abs_metric_difference_vs_expected": None,
        "max_relative_metric_difference_vs_expected": None,
        "metrics_match_expected_with_float_tolerance": None,
    }
    if expected_merged_path is not None and expected_merged_path.exists():
        expected = normalize_run_frame(
            pd.read_csv(expected_merged_path),
            "expected_clean_final_runs_updated",
        )
        validate_unique_grid(expected, "expected merged")
        compare = merged.merge(
            expected[KEY_COLUMNS + ["id_test_nmse", "ood_test_nmse"]],
            on=KEY_COLUMNS,
            how="outer",
            suffixes=("_actual", "_expected"),
            indicator=True,
            validate="one_to_one",
        )
        if not compare["_merge"].eq("both").all():
            raise ValueError("Reconstructed merged grid differs from expected merged grid")
        absolute_diffs: list[float] = []
        relative_diffs: list[float] = []
        for column in ["id_test_nmse", "ood_test_nmse"]:
            actual = compare[f"{column}_actual"].to_numpy(float)
            expected_values = compare[f"{column}_expected"].to_numpy(float)
            both_nan = np.isnan(actual) & np.isnan(expected_values)
            unequal_nan = np.isnan(actual) ^ np.isnan(expected_values)
            if unequal_nan.any():
                raise ValueError(f"NaN placement differs for {column}")
            finite = ~(both_nan | unequal_nan)
            if finite.any():
                difference = np.abs(actual[finite] - expected_values[finite])
                absolute_diffs.append(float(np.max(difference)))
                relative_diffs.append(
                    float(
                        np.max(
                            difference
                            / np.maximum(np.abs(expected_values[finite]), 1.0)
                        )
                    )
                )
        comparison["max_abs_metric_difference_vs_expected"] = max(
            absolute_diffs, default=0.0
        )
        comparison["max_relative_metric_difference_vs_expected"] = max(
            relative_diffs, default=0.0
        )
        comparison["metrics_match_expected_with_float_tolerance"] = bool(
            comparison["max_relative_metric_difference_vs_expected"] <= 1e-14
        )
        if not comparison["metrics_match_expected_with_float_tolerance"]:
            raise ValueError(
                "Reconstructed replacement metrics differ materially from the "
                f"expected formal input: {comparison}"
            )

    audit = {
        "original_rows": int(len(original)),
        "replacement_rows": int(len(replacement)),
        "untouched_rows": int(len(untouched)),
        "merged_rows": int(len(merged)),
        "replaced_algorithms": list(REPLACED_ALGORITHMS),
        "replacement_key_match": True,
        **comparison,
    }
    return merged, audit


def compute_task_components(
    runs: pd.DataFrame,
    expected_component_count: int | None = 600,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = runs.copy()
    frame["id_nmse_used"] = frame.apply(lambda row: safe_nmse(row, "id_test_nmse"), axis=1)
    frame["ood_nmse_used"] = frame.apply(
        lambda row: safe_nmse(row, "ood_test_nmse"), axis=1
    )
    rows: list[dict[str, Any]] = []
    for (algorithm, gid, dataset), group in frame.groupby(
        ["algorithm", "gid", "dataset"], sort=True, dropna=False
    ):
        median_id = float(np.median(group["id_nmse_used"]))
        median_ood = float(np.median(group["ood_nmse_used"]))
        q_id = phi_from_nmse(median_id)
        q_ood = phi_from_nmse(median_ood)
        gap = max(
            math.log10((median_ood + EPSILON) / (median_id + EPSILON)),
            0.0,
        )
        retention = 1.0 - float(np.clip(gap / 4.0, 0.0, 1.0))
        rows.append(
            {
                "algorithm": algorithm,
                "gid": gid,
                "dataset": dataset,
                "n_runs": int(len(group)),
                "n_penalized_runs": int(group["analysis_penalty_applied"].sum()),
                "valid_rate": float(group["valid_output"].mean()),
                "metric_complete_rate": float(group["metric_complete"].mean()),
                "median_id_nmse": median_id,
                "median_ood_nmse": median_ood,
                "q_id": q_id,
                "q_ood": q_ood,
                "ood_retention": retention,
                "q_ood_g_w070": 0.7 * q_ood + 0.3 * retention,
            }
        )
    components = pd.DataFrame(rows)
    if expected_component_count is not None and len(components) != expected_component_count:
        raise ValueError(
            f"Expected {expected_component_count} algorithm-task components, "
            f"found {len(components)}"
        )
    if set(components["n_runs"]) != {5}:
        raise ValueError(
            f"Every algorithm-task must have five seeds: {components['n_runs'].value_counts().to_dict()}"
        )
    return frame, components


def data_audit(
    merged: pd.DataFrame,
    components: pd.DataFrame,
    core50_manifest: Path,
    merge_audit: dict[str, Any],
) -> dict[str, Any]:
    manifest = pd.read_csv(core50_manifest)
    if "gid" in manifest.columns:
        manifest_gids = {normalize_gid(value) for value in manifest["gid"]}
    elif "core50_index" in manifest.columns:
        manifest_gids = {
            f"g{int(value):04d}" for value in manifest["core50_index"]
        }
    else:
        raise ValueError("Core-50 manifest has neither gid nor core50_index")
    run_gids = set(merged["gid"])
    algorithm_counts = merged.groupby("algorithm").size().sort_index().to_dict()
    seed_counts = (
        merged.groupby(["algorithm", "gid"])["seed"].nunique().value_counts().sort_index().to_dict()
    )
    id_values = pd.to_numeric(merged["id_test_nmse"], errors="coerce").to_numpy(float)
    ood_values = pd.to_numeric(merged["ood_test_nmse"], errors="coerce").to_numpy(float)
    expected_keys = {
        (algorithm, gid, seed)
        for algorithm in EXPECTED_ALGORITHMS
        for gid in sorted(manifest_gids)
        for seed in range(5)
    }
    actual_keys = set(
        map(
            tuple,
            merged[["algorithm", "gid", "seed"]].itertuples(index=False, name=None),
        )
    )
    status_counts = merged["status"].value_counts(dropna=False).sort_index().to_dict()
    return {
        **merge_audit,
        "algorithms": int(merged["algorithm"].nunique()),
        "algorithm_names": sorted(merged["algorithm"].unique()),
        "tasks": int(merged["gid"].nunique()),
        "manifest_tasks": int(len(manifest_gids)),
        "manifest_run_gid_match": manifest_gids == run_gids,
        "runs": int(len(merged)),
        "algorithm_run_counts": {key: int(value) for key, value in algorithm_counts.items()},
        "algorithm_task_seed_count_distribution": {
            str(key): int(value) for key, value in seed_counts.items()
        },
        "missing_run_keys": int(len(expected_keys - actual_keys)),
        "unexpected_run_keys": int(len(actual_keys - expected_keys)),
        "duplicate_run_keys": int(merged.duplicated(KEY_COLUMNS).sum()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "invalid_status_count": int(merged["status"].eq("invalid").sum()),
        "valid_output_false_count": int((~merged["valid_output"]).sum()),
        "timeout_count": int(merged["status"].eq("timed_out").sum()),
        "no_valid_output_count": int(merged["status"].eq("no_valid_output").sum()),
        "id_nan_count": int(np.isnan(id_values).sum()),
        "id_inf_count": int(np.isinf(id_values).sum()),
        "ood_nan_count": int(np.isnan(ood_values).sum()),
        "ood_inf_count": int(np.isinf(ood_values).sum()),
        "metric_incomplete_count": int((~merged["metric_complete"]).sum()),
        "id_metric_incomplete_count": int(
            ((~merged["metric_complete"]) | ~np.isfinite(id_values)).sum()
        ),
        "ood_metric_incomplete_count": int(
            ((~merged["metric_complete"]) | ~np.isfinite(ood_values)).sum()
        ),
        "analysis_penalty_run_count": int(merged["analysis_penalty_applied"].sum()),
        "algorithm_task_units": int(len(components)),
        "algorithm_task_units_with_five_seeds": int(components["n_runs"].eq(5).sum()),
    }


def verify_formal_reproduction(
    components: pd.DataFrame,
    formal_components_path: Path,
    formal_scores_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    formal_components = pd.read_csv(formal_components_path)
    formal_components["algorithm"] = formal_components["algorithm"].map(normalize_algorithm)
    formal_components["gid"] = formal_components["gid"].map(normalize_gid)
    comparison = components.merge(
        formal_components[
            [
                "algorithm",
                "gid",
                "dataset",
                "median_id_nmse",
                "median_ood_nmse",
                "ID_Q_component",
                "q_ood",
                "ood_retention",
                "OOD_G_component",
            ]
        ],
        on=["algorithm", "gid", "dataset"],
        how="outer",
        suffixes=("_computed", "_formal"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Computed component keys do not match formal Figure-3 component keys")

    field_pairs = {
        "median_id_nmse": ("median_id_nmse_computed", "median_id_nmse_formal"),
        "median_ood_nmse": ("median_ood_nmse_computed", "median_ood_nmse_formal"),
        "q_id": ("q_id", "ID_Q_component"),
        "q_ood": ("q_ood_computed", "q_ood_formal"),
        "ood_retention": ("ood_retention_computed", "ood_retention_formal"),
        "ood_g_w070": ("q_ood_g_w070", "OOD_G_component"),
    }
    errors: dict[str, float] = {}
    for name, (computed_column, formal_column) in field_pairs.items():
        errors[f"max_abs_error_{name}"] = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(comparison[computed_column], errors="coerce")
                    - pd.to_numeric(comparison[formal_column], errors="coerce")
                )
            )
        )

    algorithm_scores = (
        components.groupby("algorithm", as_index=False)
        .agg(Q=("q_ood", "mean"), R=("ood_retention", "mean"))
        .sort_values("algorithm")
    )
    algorithm_scores["computed_OOD_G_w070"] = 100.0 * (
        0.7 * algorithm_scores["Q"] + 0.3 * algorithm_scores["R"]
    )
    formal_scores = pd.read_csv(formal_scores_path)[["algorithm", "OOD_G"]]
    formal_scores["algorithm"] = formal_scores["algorithm"].map(normalize_algorithm)
    algorithm_scores = algorithm_scores.merge(
        formal_scores.rename(columns={"OOD_G": "formal_OOD_G_w070"}),
        on="algorithm",
        validate="one_to_one",
    )
    algorithm_scores["abs_error"] = np.abs(
        algorithm_scores["computed_OOD_G_w070"]
        - algorithm_scores["formal_OOD_G_w070"]
    )
    errors["max_abs_error_algorithm_OOD_G_w070"] = float(
        algorithm_scores["abs_error"].max()
    )
    if errors["max_abs_error_algorithm_OOD_G_w070"] >= 1e-6:
        raise ValueError(f"w=0.7 reproduction failed: {errors}")
    return algorithm_scores, errors


def average_descending_ranks(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(
            float(values[order[position]]),
            float(values[order[end]]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOL,
        ):
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def spearman_rank_correlation(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau_b(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    """Kendall tau-b for the small algorithm ranking vectors used here."""
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    concordant = discordant = ties_a_only = ties_b_only = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        delta_a = a[i] - a[j]
        delta_b = b[i] - b[j]
        tied_a = abs(delta_a) <= FLOAT_TOL
        tied_b = abs(delta_b) <= FLOAT_TOL
        if tied_a and tied_b:
            continue
        if tied_a:
            ties_a_only += 1
        elif tied_b:
            ties_b_only += 1
        elif delta_a * delta_b > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_a_only)
        * (concordant + discordant + ties_b_only)
    )
    if denominator == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return (concordant - discordant) / denominator


def stable_topk(algorithms: Sequence[str], scores: Sequence[float], k: int) -> list[str]:
    ordered = sorted(
        zip(algorithms, scores, strict=True),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    return [algorithm for algorithm, _ in ordered[:k]]


def compute_weight_scan(
    components: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    algorithm_components = (
        components.groupby("algorithm", as_index=False)
        .agg(Q=("q_ood", "mean"), R=("ood_retention", "mean"))
        .sort_values("algorithm")
        .reset_index(drop=True)
    )
    algorithm_components["slope"] = 100.0 * (
        algorithm_components["Q"] - algorithm_components["R"]
    )
    for weight in CANDIDATE_WEIGHTS:
        label = f"score_w{str(weight).replace('.', '').ljust(4, '0')}"
        algorithm_components[label] = 100.0 * (
            weight * algorithm_components["Q"]
            + (1.0 - weight) * algorithm_components["R"]
        )

    algorithms = algorithm_components["algorithm"].tolist()
    q_values = algorithm_components["Q"].to_numpy(float)
    r_values = algorithm_components["R"].to_numpy(float)
    baseline_scores = 100.0 * (0.7 * q_values + 0.3 * r_values)
    baseline_ranks = average_descending_ranks(baseline_scores)
    baseline_top = {
        k: set(stable_topk(algorithms, baseline_scores, k)) for k in (1, 3, 5)
    }

    score_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for weight in ALL_WEIGHTS:
        scores = 100.0 * (weight * q_values + (1.0 - weight) * r_values)
        ranks = average_descending_ranks(scores)
        for index, algorithm in enumerate(algorithms):
            score_rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "score": float(scores[index]),
                    "score_relative_to_w070": float(scores[index] - baseline_scores[index]),
                }
            )
            rank_rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "rank": float(ranks[index]),
                    "rank_relative_to_w070": float(ranks[index] - baseline_ranks[index]),
                }
            )

        spearman = spearman_rank_correlation(baseline_ranks, ranks)
        kendall = kendall_tau_b(baseline_ranks, ranks)
        top_sets = {k: set(stable_topk(algorithms, scores, k)) for k in (1, 3, 5)}
        flips = 0
        for i, j in itertools.combinations(range(len(algorithms)), 2):
            baseline_diff = baseline_scores[i] - baseline_scores[j]
            current_diff = scores[i] - scores[j]
            if baseline_diff * current_diff < -(FLOAT_TOL**2):
                flips += 1
        changed = [
            algorithm
            for algorithm, baseline_rank, current_rank in zip(
                algorithms, baseline_ranks, ranks, strict=True
            )
            if not math.isclose(
                float(baseline_rank), float(current_rank), abs_tol=FLOAT_TOL
            )
        ]
        stability_rows.append(
            {
                "weight": float(weight),
                "spearman_vs_w070": spearman,
                "kendall_vs_w070": kendall,
                "top1_overlap": len(top_sets[1] & baseline_top[1]),
                "top3_overlap": len(top_sets[3] & baseline_top[3]) / 3.0,
                "top5_overlap": len(top_sets[5] & baseline_top[5]) / 5.0,
                "pairwise_order_flip_count": flips,
                "changed_algorithms": ";".join(changed),
            }
        )

    scores_by_weight = pd.DataFrame(score_rows)
    ranks_by_weight = pd.DataFrame(rank_rows)
    rank_stability = pd.DataFrame(stability_rows)
    max_rank_change = (
        ranks_by_weight.assign(
            abs_rank_change=lambda frame: frame["rank_relative_to_w070"].abs()
        )
        .groupby("algorithm", as_index=False)
        .agg(
            max_rank_change=("abs_rank_change", "max"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
    )
    focus_ranks = ranks_by_weight[
        ranks_by_weight["weight"].between(0.70 - FLOAT_TOL, 0.95 + FLOAT_TOL)
    ].copy()
    focus_rank_change = (
        focus_ranks.assign(
            abs_rank_change=lambda frame: frame["rank_relative_to_w070"].abs()
        )
        .groupby("algorithm", as_index=False)
        .agg(
            max_rank_change_070_095=("abs_rank_change", "max"),
            min_rank_070_095=("rank", "min"),
            max_rank_070_095=("rank", "max"),
            unique_ranks_070_095=("rank", "nunique"),
        )
    )
    max_rank_change = max_rank_change.merge(
        focus_rank_change,
        on="algorithm",
        validate="one_to_one",
    )

    for _, row in algorithm_components.iterrows():
        expected = 100.0 * (
            row["R"] + ALL_WEIGHTS * (row["Q"] - row["R"])
        )
        observed = scores_by_weight.loc[
            scores_by_weight["algorithm"].eq(row["algorithm"]), "score"
        ].to_numpy(float)
        if not np.allclose(expected, observed, atol=1e-12, rtol=0):
            raise AssertionError(f"Nonlinear score curve detected for {row['algorithm']}")
    return (
        algorithm_components,
        scores_by_weight,
        ranks_by_weight,
        rank_stability,
        max_rank_change,
    )


def interval_rank_summary(rank_stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for low, high in [(0.70, 0.80), (0.80, 0.90), (0.85, 0.95), (0.70, 0.95)]:
        subset = rank_stability[
            rank_stability["weight"].between(low - FLOAT_TOL, high + FLOAT_TOL)
        ]
        changed: set[str] = set()
        for value in subset["changed_algorithms"]:
            changed.update(filter(None, str(value).split(";")))
        rows.append(
            {
                "interval": f"[{low:.2f},{high:.2f}]",
                "weights_evaluated": int(len(subset)),
                "min_spearman_vs_w070": float(subset["spearman_vs_w070"].min()),
                "min_kendall_vs_w070": float(subset["kendall_vs_w070"].min()),
                "min_top1_overlap": float(subset["top1_overlap"].min()),
                "min_top3_overlap": float(subset["top3_overlap"].min()),
                "min_top5_overlap": float(subset["top5_overlap"].min()),
                "max_pairwise_order_flip_count": int(
                    subset["pairwise_order_flip_count"].max()
                ),
                "changed_algorithms": ";".join(sorted(changed)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_crossings(algorithm_components: pd.DataFrame) -> pd.DataFrame:
    values = algorithm_components.set_index("algorithm")[["Q", "R"]].to_dict("index")
    algorithms = sorted(values)
    baseline_scores = {
        algorithm: 100.0 * (0.7 * values[algorithm]["Q"] + 0.3 * values[algorithm]["R"])
        for algorithm in algorithms
    }
    baseline_top5 = set(
        sorted(algorithms, key=lambda algorithm: (-baseline_scores[algorithm], algorithm))[:5]
    )
    rows: list[dict[str, Any]] = []
    for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
        qa, ra = values[algorithm_a]["Q"], values[algorithm_a]["R"]
        qb, rb = values[algorithm_b]["Q"], values[algorithm_b]["R"]
        denominator = (qa - ra) - (qb - rb)
        numerator = rb - ra
        parallel = abs(denominator) <= FLOAT_TOL
        coincident = parallel and abs(numerator) <= FLOAT_TOL
        if parallel:
            rows.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "crossing_weight": np.nan,
                    "rank_order_below_crossing": "coincident" if coincident else "parallel",
                    "rank_order_above_crossing": "coincident" if coincident else "parallel",
                    "is_crossing_in_070_095": False,
                    "is_crossing_in_080_090": False,
                    "involves_baseline_top5": bool(
                        algorithm_a in baseline_top5 or algorithm_b in baseline_top5
                    ),
                    "is_parallel": True,
                    "is_coincident": coincident,
                    "distance_to_w070": np.nan,
                }
            )
            continue
        crossing = numerator / denominator
        if crossing < -FLOAT_TOL or crossing > 1.0 + FLOAT_TOL:
            continue
        crossing = float(np.clip(crossing, 0.0, 1.0))

        def score(algorithm: str, weight: float) -> float:
            component = values[algorithm]
            return component["R"] + weight * (component["Q"] - component["R"])

        delta = 1e-9
        below_diff = score(algorithm_a, crossing - delta) - score(
            algorithm_b, crossing - delta
        )
        above_diff = score(algorithm_a, crossing + delta) - score(
            algorithm_b, crossing + delta
        )

        def order_label(diff: float) -> str:
            if diff > 0:
                return f"{algorithm_a}>{algorithm_b}"
            if diff < 0:
                return f"{algorithm_b}>{algorithm_a}"
            return "tie"

        rows.append(
            {
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "crossing_weight": crossing,
                "rank_order_below_crossing": order_label(below_diff),
                "rank_order_above_crossing": order_label(above_diff),
                "is_crossing_in_070_095": 0.70 - FLOAT_TOL
                <= crossing
                <= 0.95 + FLOAT_TOL,
                "is_crossing_in_080_090": 0.80 - FLOAT_TOL
                <= crossing
                <= 0.90 + FLOAT_TOL,
                "involves_baseline_top5": bool(
                    algorithm_a in baseline_top5 or algorithm_b in baseline_top5
                ),
                "is_parallel": False,
                "is_coincident": False,
                "distance_to_w070": abs(crossing - 0.7),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_parallel", "crossing_weight", "algorithm_a", "algorithm_b"],
        na_position="last",
    )


def bootstrap_analysis(
    components: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[float, dict[str, np.ndarray]]]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    q_pivot = components.pivot(index="gid", columns="algorithm", values="q_ood").sort_index()
    r_pivot = components.pivot(
        index="gid", columns="algorithm", values="ood_retention"
    ).sort_index()
    if not q_pivot.index.equals(r_pivot.index) or not q_pivot.columns.equals(r_pivot.columns):
        raise ValueError("Q/R bootstrap matrices do not align")
    algorithms = q_pivot.columns.tolist()
    n_tasks = len(q_pivot)
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, n_tasks, size=(repetitions, n_tasks))
    q_bootstrap = q_pivot.to_numpy(float)[sampled_indices].mean(axis=1)
    r_bootstrap = r_pivot.to_numpy(float)[sampled_indices].mean(axis=1)

    score_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    raw: dict[float, dict[str, np.ndarray]] = {}
    baseline_point_scores = 100.0 * (
        0.7 * q_pivot.to_numpy(float).mean(axis=0)
        + 0.3 * r_pivot.to_numpy(float).mean(axis=0)
    )
    baseline_order = stable_topk(algorithms, baseline_point_scores, len(algorithms))
    baseline_adjacent = {
        frozenset(pair) for pair in zip(baseline_order[:-1], baseline_order[1:], strict=True)
    }
    baseline_top5 = set(baseline_order[:5])

    for weight in CANDIDATE_WEIGHTS:
        scores = 100.0 * (weight * q_bootstrap + (1.0 - weight) * r_bootstrap)
        ranks = np.vstack([average_descending_ranks(row) for row in scores])
        stable_indices = np.argsort(-scores, axis=1, kind="stable")
        raw[float(weight)] = {"scores": scores, "ranks": ranks}
        for index, algorithm in enumerate(algorithms):
            values = scores[:, index]
            rank_values = ranks[:, index]
            top1 = np.mean(stable_indices[:, 0] == index)
            top3 = np.mean(np.any(stable_indices[:, :3] == index, axis=1))
            top5 = np.mean(np.any(stable_indices[:, :5] == index, axis=1))
            score_rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "mean_score": float(np.mean(values)),
                    "median_score": float(np.median(values)),
                    "score_ci_2_5": float(np.percentile(values, 2.5)),
                    "score_ci_97_5": float(np.percentile(values, 97.5)),
                }
            )
            rank_rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "mean_rank": float(np.mean(rank_values)),
                    "median_rank": float(np.median(rank_values)),
                    "top1_probability": float(top1),
                    "top3_probability": float(top3),
                    "top5_probability": float(top5),
                }
            )
        for i, j in itertools.combinations(range(len(algorithms)), 2):
            algorithm_a, algorithm_b = algorithms[i], algorithms[j]
            differences = scores[:, i] - scores[:, j]
            probability_rows.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "weight": float(weight),
                    "p_a_greater_b": float(np.mean(differences > FLOAT_TOL)),
                    "p_tie": float(np.mean(np.abs(differences) <= FLOAT_TOL)),
                    "p_a_less_b": float(np.mean(differences < -FLOAT_TOL)),
                    "is_baseline_adjacent_pair": frozenset(
                        (algorithm_a, algorithm_b)
                    )
                    in baseline_adjacent,
                    "is_baseline_top5_pair": bool(
                        algorithm_a in baseline_top5 and algorithm_b in baseline_top5
                    ),
                }
            )
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(rank_rows),
        pd.DataFrame(probability_rows),
        raw,
    )


def bootstrap_pairwise_stability_summary(
    pairwise_probabilities: pd.DataFrame,
    confidence_threshold: float = 0.95,
) -> pd.DataFrame:
    """汇总算法对胜率，不把 bootstrap 概率误写成经典检验 p 值。"""
    frame = pairwise_probabilities.copy()
    frame["directional_probability"] = frame[
        ["p_a_greater_b", "p_a_less_b"]
    ].max(axis=1)
    frame["directionally_stable"] = (
        frame["directional_probability"] >= confidence_threshold
    )
    rows: list[dict[str, Any]] = []
    for weight, group in frame.groupby("weight", sort=True):
        adjacent = group[group["is_baseline_adjacent_pair"]]
        top5 = group[group["is_baseline_top5_pair"]]
        rows.append(
            {
                "weight": float(weight),
                "confidence_threshold": float(confidence_threshold),
                "all_pair_count": int(len(group)),
                "all_pair_stable_count": int(group["directionally_stable"].sum()),
                "baseline_adjacent_pair_count": int(len(adjacent)),
                "baseline_adjacent_pair_stable_count": int(
                    adjacent["directionally_stable"].sum()
                ),
                "baseline_top5_pair_count": int(len(top5)),
                "baseline_top5_pair_stable_count": int(
                    top5["directionally_stable"].sum()
                ),
                "minimum_adjacent_directional_probability": float(
                    adjacent["directional_probability"].min()
                ),
                "minimum_top5_directional_probability": float(
                    top5["directional_probability"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def gated_retention_analysis(
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gated = components.copy()
    gated["r_gated"] = np.where(
        (gated["q_id"] > 0.0) & (gated["q_ood"] > 0.0),
        gated["ood_retention"],
        0.0,
    )
    aggregated = (
        gated.groupby("algorithm", as_index=False)
        .agg(Q=("q_ood", "mean"), R=("ood_retention", "mean"), R_gated=("r_gated", "mean"))
        .sort_values("algorithm")
    )
    algorithms = aggregated["algorithm"].tolist()
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for weight in CANDIDATE_WEIGHTS:
        original_scores = 100.0 * (
            weight * aggregated["Q"].to_numpy(float)
            + (1.0 - weight) * aggregated["R"].to_numpy(float)
        )
        gated_scores = 100.0 * (
            weight * aggregated["Q"].to_numpy(float)
            + (1.0 - weight) * aggregated["R_gated"].to_numpy(float)
        )
        original_ranks = average_descending_ranks(original_scores)
        gated_ranks = average_descending_ranks(gated_scores)
        original_top3 = set(stable_topk(algorithms, original_scores, 3))
        gated_top3 = set(stable_topk(algorithms, gated_scores, 3))
        spearman = spearman_rank_correlation(original_ranks, gated_ranks)
        kendall = kendall_tau_b(original_ranks, gated_ranks)
        for index, algorithm in enumerate(algorithms):
            rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "original_score": float(original_scores[index]),
                    "gated_score": float(gated_scores[index]),
                    "score_benefit_from_original_retention": float(
                        original_scores[index] - gated_scores[index]
                    ),
                    "original_rank": float(original_ranks[index]),
                    "gated_rank": float(gated_ranks[index]),
                    "rank_change_gated_minus_original": float(
                        gated_ranks[index] - original_ranks[index]
                    ),
                    "spearman_original_vs_gated": spearman,
                    "kendall_original_vs_gated": kendall,
                    "top3_overlap_original_vs_gated": len(original_top3 & gated_top3)
                    / 3.0,
                }
            )
        summary_rows.append(
            {
                "weight": float(weight),
                "spearman_original_vs_gated": spearman,
                "kendall_original_vs_gated": kendall,
                "top3_overlap_original_vs_gated": len(original_top3 & gated_top3)
                / 3.0,
                "original_top3": ";".join(sorted(original_top3)),
                "gated_top3": ";".join(sorted(gated_top3)),
                "max_algorithm_score_benefit": float(
                    np.max(original_scores - gated_scores)
                ),
                "most_benefited_algorithm": algorithms[
                    int(np.argmax(original_scores - gated_scores))
                ],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def failure_retention_diagnostics(
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = components[
        np.isclose(components["q_ood"], 0.0, atol=FLOAT_TOL)
        & np.isclose(components["ood_retention"], 1.0, atol=FLOAT_TOL)
    ].copy()
    cases["failure_retention_case"] = True
    rows: list[dict[str, Any]] = []
    for algorithm in sorted(components["algorithm"].unique()):
        algorithm_cases = cases[cases["algorithm"].eq(algorithm)]
        for weight in CANDIDATE_WEIGHTS:
            contribution = (
                100.0
                / 50.0
                * (1.0 - weight)
                * algorithm_cases["ood_retention"].sum()
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "weight": float(weight),
                    "case_count": int(len(algorithm_cases)),
                    "total_OOD_G_point_contribution": float(contribution),
                }
            )
    return cases, pd.DataFrame(rows)


def reversal_diagnostics(
    components: pd.DataFrame,
    scores_by_weight: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    algorithms = sorted(components["algorithm"].unique())
    score_lookup = {
        (row.algorithm, float(row.weight)): float(row.score)
        for row in scores_by_weight.itertuples(index=False)
    }
    task_groups = {
        gid: group.set_index("algorithm")
        for gid, group in components.groupby("gid", sort=True)
    }
    aggregate_rows: list[dict[str, Any]] = []
    candidate_case_rows: list[dict[str, Any]] = []
    for weight in ALL_WEIGHTS:
        current_scores = [score_lookup[(algorithm, float(weight))] for algorithm in algorithms]
        top5 = set(stable_topk(algorithms, current_scores, 5))
        for factor in REVERSAL_FACTORS:
            count = 0
            involving_top5 = 0
            both_top5 = 0
            aggregate_order_reversed = 0
            involved: set[str] = set()
            max_ratio = 0.0
            for gid, task in task_groups.items():
                for algorithm_x, algorithm_y in itertools.combinations(algorithms, 2):
                    ex = float(task.loc[algorithm_x, "median_ood_nmse"])
                    ey = float(task.loc[algorithm_y, "median_ood_nmse"])
                    if ex <= ey:
                        better, worse, e_better, e_worse = (
                            algorithm_x,
                            algorithm_y,
                            ex,
                            ey,
                        )
                    else:
                        better, worse, e_better, e_worse = (
                            algorithm_y,
                            algorithm_x,
                            ey,
                            ex,
                        )
                    ratio = (e_worse + EPSILON) / (e_better + EPSILON)
                    if ratio + FLOAT_TOL < factor:
                        continue
                    better_task = float(
                        weight * task.loc[better, "q_ood"]
                        + (1.0 - weight) * task.loc[better, "ood_retention"]
                    )
                    worse_task = float(
                        weight * task.loc[worse, "q_ood"]
                        + (1.0 - weight) * task.loc[worse, "ood_retention"]
                    )
                    if better_task >= worse_task - FLOAT_TOL:
                        continue
                    count += 1
                    max_ratio = max(max_ratio, ratio)
                    involved.update((better, worse))
                    involving_top5 += int(better in top5 or worse in top5)
                    both_top5 += int(better in top5 and worse in top5)
                    aggregate_reversal = (
                        score_lookup[(better, float(weight))]
                        < score_lookup[(worse, float(weight))] - FLOAT_TOL
                    )
                    aggregate_order_reversed += int(aggregate_reversal)
                    if any(
                        math.isclose(float(weight), float(candidate), abs_tol=FLOAT_TOL)
                        for candidate in CANDIDATE_WEIGHTS
                    ):
                        candidate_case_rows.append(
                            {
                                "weight": float(weight),
                                "ood_nmse_factor_threshold": factor,
                                "gid": gid,
                                "dataset": task.iloc[0]["dataset"],
                                "ood_better_algorithm": better,
                                "retention_favored_algorithm": worse,
                                "better_ood_nmse": e_better,
                                "worse_ood_nmse": e_worse,
                                "ood_nmse_ratio": ratio,
                                "better_task_OOD_G_component": better_task,
                                "worse_task_OOD_G_component": worse_task,
                                "changes_algorithm_level_pair_order": aggregate_reversal,
                                "both_algorithms_in_current_top5": bool(
                                    better in top5 and worse in top5
                                ),
                            }
                        )
            aggregate_rows.append(
                {
                    "weight": float(weight),
                    "ood_nmse_factor_threshold": factor,
                    "reversal_count": count,
                    "involved_algorithms": ";".join(sorted(involved)),
                    "max_ood_nmse_ratio": max_ratio,
                    "aggregate_pair_order_reversal_count": aggregate_order_reversed,
                    "changes_any_algorithm_level_pair_order": aggregate_order_reversed > 0,
                    "reversals_involving_current_top5": involving_top5,
                    "reversals_with_both_algorithms_in_current_top5": both_top5,
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(candidate_case_rows)


def candidate_summary(
    rank_stability: pd.DataFrame,
    ranks_by_weight: pd.DataFrame,
    crossings: pd.DataFrame,
    reversals: pd.DataFrame,
    bootstrap_ranks: pd.DataFrame,
    gated_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid_crossings = crossings[~crossings["is_parallel"]].copy()
    rank_pivot = ranks_by_weight.pivot(
        index="algorithm", columns="weight", values="rank"
    ).sort_index()
    algorithms = rank_pivot.index.tolist()
    for weight in CANDIDATE_WEIGHTS:
        stability = rank_stability[
            np.isclose(rank_stability["weight"], weight, atol=FLOAT_TOL)
        ].iloc[0]
        gated = gated_summary[
            np.isclose(gated_summary["weight"], weight, atol=FLOAT_TOL)
        ].iloc[0]
        bootstrap = bootstrap_ranks[
            np.isclose(bootstrap_ranks["weight"], weight, atol=FLOAT_TOL)
        ]
        top1_row = bootstrap.sort_values(
            ["top1_probability", "algorithm"], ascending=[False, True]
        ).iloc[0]
        reversal = reversals[
            np.isclose(reversals["weight"], weight, atol=FLOAT_TOL)
            & reversals["ood_nmse_factor_threshold"].eq(10)
        ].iloc[0]
        local_crossings = valid_crossings[
            valid_crossings["crossing_weight"].between(
                weight - 0.05 - FLOAT_TOL,
                weight + 0.05 + FLOAT_TOL,
            )
        ]
        local_weights = [
            float(value)
            for value in rank_pivot.columns
            if max(0.0, weight - 0.05) - FLOAT_TOL
            <= float(value)
            <= min(1.0, weight + 0.05) + FLOAT_TOL
        ]
        candidate_ranks = rank_pivot[weight].to_numpy(float)
        candidate_top3 = set(
            algorithm
            for algorithm, _ in sorted(
                zip(algorithms, candidate_ranks, strict=True),
                key=lambda item: (item[1], item[0]),
            )[:3]
        )
        local_kendalls: list[float] = []
        local_top3_overlaps: list[float] = []
        for local_weight in local_weights:
            local_ranks = rank_pivot[local_weight].to_numpy(float)
            local_kendalls.append(kendall_tau_b(candidate_ranks, local_ranks))
            local_top3 = set(
                algorithm
                for algorithm, _ in sorted(
                    zip(algorithms, local_ranks, strict=True),
                    key=lambda item: (item[1], item[0]),
                )[:3]
            )
            local_top3_overlaps.append(len(candidate_top3 & local_top3) / 3.0)
        compensation = 0.0 if weight == 1.0 else 14.0 * (1.0 - weight) / weight
        rows.append(
            {
                "weight": float(weight),
                "quality_weight": float(weight),
                "retention_weight": float(1.0 - weight),
                "equivalent_log_order_compensation": compensation,
                "spearman_vs_w070": float(stability["spearman_vs_w070"]),
                "kendall_vs_w070": float(stability["kendall_vs_w070"]),
                "top3_overlap": float(stability["top3_overlap"]),
                "number_of_pairwise_flips": int(
                    stability["pairwise_order_flip_count"]
                ),
                "number_of_crossings_within_plus_minus_0_05": int(
                    len(local_crossings)
                ),
                "number_of_quality_reversals": int(reversal["reversal_count"]),
                "bootstrap_top1_algorithm": str(top1_row["algorithm"]),
                "bootstrap_top1_probability": float(top1_row["top1_probability"]),
                "original_vs_gated_rank_correlation": float(
                    gated["spearman_original_vs_gated"]
                ),
                "local_min_kendall_plus_minus_0_05": float(min(local_kendalls)),
                "local_min_top3_overlap_plus_minus_0_05": float(
                    min(local_top3_overlaps)
                ),
            }
        )
    summary = pd.DataFrame(rows)

    # 预定义判据只依赖稳定性、异常行为和数量级解释，不依赖算法身份。
    summary["stability_pass"] = (
        summary["local_min_kendall_plus_minus_0_05"].ge(0.90)
        & summary["local_min_top3_overlap_plus_minus_0_05"].eq(1.0)
    )
    summary["gated_diagnostic_pass"] = (
        summary["original_vs_gated_rank_correlation"].ge(0.90)
    )
    summary["compensation_interpretable"] = (
        summary["equivalent_log_order_compensation"].le(2.5)
        & summary["retention_weight"].gt(0.0)
    )
    summary["recommendation"] = "diagnostic candidate"
    return summary


def choose_recommendation(summary: pd.DataFrame) -> tuple[str, str]:
    row070 = summary[np.isclose(summary["weight"], 0.70)].iloc[0]
    row085 = summary[np.isclose(summary["weight"], 0.85)].iloc[0]
    row090 = summary[np.isclose(summary["weight"], 0.90)].iloc[0]

    if (
        row070["stability_pass"]
        and row070["gated_diagnostic_pass"]
        and row070["compensation_interpretable"]
    ):
        return (
            "retain_0.70",
            "Current 0.70 passes all pre-defined local stability, gated-retention, "
            "and log-order compensation checks.",
        )
    if (
        row085["stability_pass"]
        and row085["gated_diagnostic_pass"]
        and row085["compensation_interpretable"]
    ):
        return (
            "recommend_0.85",
            "0.85 is the closest pre-specified candidate that limits full "
            "retention compensation to at most 2.5 OOD-NMSE orders while "
            "passing local rank and gated-retention stability checks.",
        )
    if (
        row090["stability_pass"]
        and row090["gated_diagnostic_pass"]
        and row090["compensation_interpretable"]
    ):
        return (
            "recommend_0.90",
            "0.90 passes local rank and gated-retention stability checks and "
            "limits full retention compensation to 1.56 OOD-NMSE orders.",
        )
    return (
        "no_stable_single_weight",
        "No permitted single weight passes all pre-defined local stability, "
        "gated-retention, and interpretability checks; report Q and R separately.",
    )


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(
    output_dir: Path,
    algorithm_components: pd.DataFrame,
    scores: pd.DataFrame,
    ranks: pd.DataFrame,
    rank_stability: pd.DataFrame,
    bootstrap_ranks: pd.DataFrame,
    original_vs_gated: pd.DataFrame,
    reversals: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_scores = scores[np.isclose(scores["weight"], 0.70)]
    algorithm_order = (
        baseline_scores.sort_values(["score", "algorithm"], ascending=[False, True])[
            "algorithm"
        ].tolist()
    )
    palette = dict(
        zip(
            algorithm_order,
            plt.get_cmap("tab20")(np.linspace(0.0, 0.95, len(algorithm_order))),
            strict=True,
        )
    )

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for algorithm in algorithm_order:
        subset = scores[scores["algorithm"].eq(algorithm)].sort_values("weight")
        ax.plot(
            subset["weight"],
            subset["score"],
            label=DISPLAY_NAMES.get(algorithm, algorithm),
            color=palette[algorithm],
            linewidth=1.8,
        )
    for weight, linestyle in [(0.70, "--"), (0.85, ":"), (0.90, "-.")]:
        ax.axvline(weight, color="black", linestyle=linestyle, linewidth=1.0)
    ax.set(xlabel="OOD absolute-quality weight w", ylabel="OOD-G score", xlim=(0, 1), ylim=(0, 100))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir, "score_weight_curves")

    rank_pivot = (
        ranks.pivot(index="algorithm", columns="weight", values="rank")
        .reindex(algorithm_order)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(15.5, 6.0))
    image = ax.imshow(rank_pivot.to_numpy(), aspect="auto", cmap="viridis_r", vmin=1, vmax=12)
    columns = rank_pivot.columns.to_numpy(float)
    tick_indices = [
        index
        for index, value in enumerate(columns)
        if math.isclose(value % 0.1, 0.0, abs_tol=1e-9)
        or math.isclose(value, 1.0, abs_tol=1e-9)
    ]
    ax.set_xticks(tick_indices, [f"{columns[index]:.1f}" for index in tick_indices])
    ax.set_yticks(
        range(len(algorithm_order)),
        [DISPLAY_NAMES.get(algorithm, algorithm) for algorithm in algorithm_order],
    )
    ax.set(xlabel="Weight w", ylabel="Algorithm")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rank (1 is best)")
    fig.tight_layout()
    save_figure(fig, output_dir, "rank_weight_heatmap")

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ordered_stability = rank_stability.sort_values("weight")
    ax.plot(
        ordered_stability["weight"],
        ordered_stability["spearman_vs_w070"],
        label="Spearman",
        linewidth=2,
    )
    ax.plot(
        ordered_stability["weight"],
        ordered_stability["kendall_vs_w070"],
        label="Kendall tau",
        linewidth=2,
    )
    ax.axvline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Weight w", ylabel="Rank correlation vs w=0.7", xlim=(0, 1), ylim=(-1, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "rank_correlation_vs_weight")

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for column, label in [
        ("top1_overlap", "Top-1"),
        ("top3_overlap", "Top-3"),
        ("top5_overlap", "Top-5"),
    ]:
        ax.plot(
            ordered_stability["weight"],
            ordered_stability[column],
            label=label,
            linewidth=2,
        )
    ax.axvline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Weight w", ylabel="Overlap with w=0.7", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "topk_stability_vs_weight")

    decomposition = algorithm_components.set_index("algorithm").reindex(algorithm_order)
    x = np.arange(len(decomposition))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.bar(x - width / 2, 100 * decomposition["Q"], width, label="Q: OOD absolute quality")
    ax.bar(x + width / 2, 100 * decomposition["R"], width, label="R: ID-to-OOD retention")
    ax.set_xticks(
        x,
        [DISPLAY_NAMES.get(algorithm, algorithm) for algorithm in decomposition.index],
        rotation=35,
        ha="right",
    )
    ax.set(ylabel="Component score", ylim=(0, 100))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "component_decomposition")

    probability_pivot = (
        bootstrap_ranks.pivot(index="algorithm", columns="weight", values="top3_probability")
        .reindex(algorithm_order)
        .reindex(columns=CANDIDATE_WEIGHTS)
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    image = ax.imshow(
        probability_pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1
    )
    ax.set_xticks(
        range(len(CANDIDATE_WEIGHTS)),
        [f"{weight:g}" for weight in CANDIDATE_WEIGHTS],
    )
    ax.set_yticks(
        range(len(algorithm_order)),
        [DISPLAY_NAMES.get(algorithm, algorithm) for algorithm in algorithm_order],
    )
    ax.set(xlabel="Weight w", ylabel="Algorithm")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Bootstrap Top-3 probability")
    fig.tight_layout()
    save_figure(fig, output_dir, "bootstrap_rank_probability")

    compare_weights = [0.70, 0.85, 0.90]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharex=True, sharey=True)
    for ax, weight in zip(axes, compare_weights, strict=True):
        subset = original_vs_gated[np.isclose(original_vs_gated["weight"], weight)]
        for row in subset.itertuples(index=False):
            ax.scatter(
                row.original_score,
                row.gated_score,
                color=palette[row.algorithm],
                s=35,
            )
        ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
        ax.set(title=f"w={weight:g}", xlim=(0, 100), ylim=(0, 100))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Gated OOD-G")
    for ax in axes:
        ax.set_xlabel("Original OOD-G")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=palette[algorithm],
            label=DISPLAY_NAMES.get(algorithm, algorithm),
        )
        for algorithm in algorithm_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=6,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "original_vs_gated")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for factor in REVERSAL_FACTORS:
        subset = reversals[
            reversals["ood_nmse_factor_threshold"].eq(factor)
        ].sort_values("weight")
        ax.plot(
            subset["weight"],
            subset["reversal_count"],
            label=f"{factor}x OOD-NMSE gap",
            linewidth=2,
        )
    ax.axvline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Weight w", ylabel="Task-level quality reversal count", xlim=(0, 1), ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "reversal_count_vs_weight")


def markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    decimals: int = 4,
) -> str:
    table = frame.copy()
    if columns is not None:
        table = table[list(columns)]
    if max_rows is not None:
        table = table.head(max_rows)
    rendered: list[list[str]] = []
    for row in table.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if math.isnan(float(value)) else f"{float(value):.{decimals}f}")
            else:
                cells.append(str(value))
        rendered.append(cells)
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    audit: dict[str, Any],
    reproduction: pd.DataFrame,
    reproduction_errors: dict[str, float],
    algorithm_components: pd.DataFrame,
    rank_stability: pd.DataFrame,
    interval_summary: pd.DataFrame,
    crossings: pd.DataFrame,
    bootstrap_scores: pd.DataFrame,
    bootstrap_ranks: pd.DataFrame,
    bootstrap_pair_summary: pd.DataFrame,
    algorithm_rank_sensitivity: pd.DataFrame,
    failure_cases: pd.DataFrame,
    failure_summary: pd.DataFrame,
    gated_summary: pd.DataFrame,
    reversals: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    recommendation: tuple[str, str],
    reproducibility: dict[str, Any],
) -> None:
    finite_crossings = crossings[~crossings["is_parallel"]].copy()
    nearest = finite_crossings.sort_values("distance_to_w070").head(10)
    in_070_095 = finite_crossings[finite_crossings["is_crossing_in_070_095"]]
    in_080_090 = finite_crossings[finite_crossings["is_crossing_in_080_090"]]
    top5_crossings = finite_crossings[finite_crossings["involves_baseline_top5"]]
    candidate_display = candidate_weights.copy()
    candidate_display["recommendation"] = "not selected"
    selected_map = {
        "retain_0.70": 0.70,
        "recommend_0.85": 0.85,
        "recommend_0.90": 0.90,
    }
    if recommendation[0] in selected_map:
        candidate_display.loc[
            np.isclose(candidate_display["weight"], selected_map[recommendation[0]]),
            "recommendation",
        ] = recommendation[0]
    elif recommendation[0] == "no_stable_single_weight":
        candidate_display["recommendation"] = "report Q and R separately"

    failure_at_070 = failure_summary[np.isclose(failure_summary["weight"], 0.70)]
    reversal_candidates = reversals[
        reversals["weight"].isin(CANDIDATE_WEIGHTS)
        & reversals["ood_nmse_factor_threshold"].eq(10)
    ]
    top_bootstrap = (
        bootstrap_ranks.sort_values(
            ["weight", "top1_probability", "algorithm"],
            ascending=[True, False, True],
        )
        .groupby("weight", as_index=False)
        .first()
    )
    stable_focus_algorithms = algorithm_rank_sensitivity.loc[
        algorithm_rank_sensitivity["max_rank_change_070_095"].eq(0),
        "algorithm",
    ].map(lambda algorithm: DISPLAY_NAMES.get(algorithm, algorithm)).tolist()
    focused_max_change = algorithm_rank_sensitivity[
        "max_rank_change_070_095"
    ].max()
    sensitive_focus_algorithms = algorithm_rank_sensitivity.loc[
        algorithm_rank_sensitivity["max_rank_change_070_095"].eq(
            focused_max_change
        ),
        "algorithm",
    ].map(lambda algorithm: DISPLAY_NAMES.get(algorithm, algorithm)).tolist()
    candidate_top1_probability_min = float(top_bootstrap["top1_probability"].min())
    candidate_top1_probability_max = float(top_bootstrap["top1_probability"].max())
    baseline_scores = algorithm_components.assign(
        score_w070=lambda frame: 100.0 * (0.7 * frame["Q"] + 0.3 * frame["R"])
    )
    baseline_top3 = set(
        baseline_scores.sort_values(
            ["score_w070", "algorithm"],
            ascending=[False, True],
        )
        .head(3)["algorithm"]
    )
    baseline_top3_probability_min = float(
        bootstrap_ranks[
            bootstrap_ranks["algorithm"].isin(baseline_top3)
        ]["top3_probability"].min()
    )
    report = f"""# OOD-G Weight Sensitivity Report

This report is a local post-processing analysis. No symbolic-regression
algorithm was restarted. The analysis input replaces only the original DRSR
and LLM-SR rows with their existing model-split reruns.

## 1. Data audit

- Original rows: `{audit['original_rows']}`
- Replacement rows: `{audit['replacement_rows']}`
- Final rows: `{audit['merged_rows']}`
- Algorithms: `{audit['algorithms']}`
- Core-50 tasks: `{audit['tasks']}`
- Missing run keys: `{audit['missing_run_keys']}`
- Duplicate run keys: `{audit['duplicate_run_keys']}`
- Every algorithm-task has five seeds: `{audit['algorithm_task_units_with_five_seeds'] == 600}`
- Status counts: `{json.dumps(audit['status_counts'], sort_keys=True)}`
- `valid_output=False`: `{audit['valid_output_false_count']}`
- `timed_out`: `{audit['timeout_count']}`
- `no_valid_output`: `{audit['no_valid_output_count']}`
- ID NaN / Inf: `{audit['id_nan_count']} / {audit['id_inf_count']}`
- OOD NaN / Inf: `{audit['ood_nan_count']} / {audit['ood_inf_count']}`
- `metric_complete=False`: `{audit['metric_incomplete_count']}`
- Runs receiving the `e=1e2` metric penalty: `{audit['analysis_penalty_run_count']}`

The final grid is constructed as `2500 original + 500 DRSR/LLM-SR rerun`
records. Replacement keys match the original `(algorithm, gid, dataset, seed)`
keys exactly.

## 2. Exact reproduction of w=0.7

The implementation extracts the exact `phi_from_nmse` and `safe_nmse`
definitions from `check/plot_core50_hexagon_metrics.py` without importing its
unrelated symbolic-analysis dependencies. It first applies run-level failure
semantics, then takes the five-seed ID/OOD NMSE medians, and finally computes
task components.

- Maximum algorithm OOD-G error: `{reproduction_errors['max_abs_error_algorithm_OOD_G_w070']:.3e}`
- Maximum task OOD-G-component error: `{reproduction_errors['max_abs_error_ood_g_w070']:.3e}`
- Required tolerance: `<1e-6`

{markdown_table(reproduction, ['algorithm', 'computed_OOD_G_w070', 'formal_OOD_G_w070', 'abs_error'], decimals=8)}

## 3. Mathematical decomposition

For each algorithm, `Q` and `R` are fixed after the task aggregation:

`OOD_G(a,w) = 100 * [R(a) + w * (Q(a)-R(a))]`.

The verified slope is `100*(Q-R)`, so every score-weight curve is exactly
linear.

{markdown_table(algorithm_components, ['algorithm', 'Q', 'R', 'slope', 'score_w0700', 'score_w0850', 'score_w0900', 'score_w1000'])}

## 4. Score sensitivity

The complete scan covers `w=0.00,0.01,...,1.00`, plus the pre-specified
`w=0.875`. Scores and ranks are in `scores_by_weight.csv` and
`ranks_by_weight.csv`. Full-span plots use the untruncated 0--100 score axis.

## 5. Ranking sensitivity

{markdown_table(interval_summary)}

Top-1, Top-3, Top-5 overlap, pairwise flips, and changed algorithms for every
weight are recorded in `rank_stability_by_weight.csv`.

Across `[0.70,0.95]`, Top-1, Top-3, and Top-5 memberships never change.
Algorithms with unchanged exact ranks are
`{';'.join(stable_focus_algorithms)}`. The largest focused-range rank changes
belong to `{';'.join(sensitive_focus_algorithms)}`; the complete per-algorithm
summary is in `algorithm_rank_sensitivity.csv`.

## 6. Pairwise crossing points

- Crossings in `[0.70,0.95]`: `{len(in_070_095)}`
- Crossings in `[0.80,0.90]`: `{len(in_080_090)}`
- Crossings involving a baseline Top-5 algorithm: `{len(top5_crossings)}`

Nearest ten crossings to `w=0.7`:

{markdown_table(nearest, ['algorithm_a', 'algorithm_b', 'crossing_weight', 'rank_order_below_crossing', 'rank_order_above_crossing', 'involves_baseline_top5'], max_rows=10, decimals=6)}

All crossings involving a baseline Top-5 algorithm:

{markdown_table(top5_crossings, ['algorithm_a', 'algorithm_b', 'crossing_weight', 'rank_order_below_crossing', 'rank_order_above_crossing'], decimals=6)}

## 7. Bootstrap uncertainty

The analysis uses `{reproducibility['bootstrap_repetitions']}` synchronized
task-bootstrap replicates with seed `{reproducibility['bootstrap_seed']}`.
Each sampled task retains all five seeds. Percentile intervals are task-sampling
uncertainty intervals, not seed-resampling intervals.

{markdown_table(top_bootstrap, ['weight', 'algorithm', 'top1_probability', 'top3_probability', 'mean_rank', 'median_rank'])}

Although iMCTS has the highest point estimate at every candidate weight, its
Top-1 probability ranges only from `{candidate_top1_probability_min:.3f}` to
`{candidate_top1_probability_max:.3f}`. Therefore the analysis does not support
describing the first-place algorithm as statistically decisive. By contrast,
the minimum Top-3 probability among the baseline Top-3 algorithms is
`{baseline_top3_probability_min:.3f}`, so Top-3 membership is much more robust
than internal Top-3 ordering.

Score intervals, rank probabilities, and all pairwise winning probabilities
are saved separately. The table below counts an algorithm pair as
directionally stable when the larger of `P(A>B)` and `P(A<B)` is at least
`0.95`. This is a pre-defined bootstrap evidence threshold, not a classical
hypothesis-test significance level.

{markdown_table(bootstrap_pair_summary, ['weight', 'all_pair_stable_count', 'all_pair_count', 'baseline_adjacent_pair_stable_count', 'baseline_adjacent_pair_count', 'baseline_top5_pair_stable_count', 'baseline_top5_pair_count', 'minimum_adjacent_directional_probability'])}

## 8. Invalid-output and gated-retention diagnostics

There are `{len(failure_cases)}` algorithm-task pairs with `q_ood=0` and
`r_ood=1`. Their original OOD-G contribution at `w=0.7` is:

{markdown_table(failure_at_070, ['algorithm', 'case_count', 'total_OOD_G_point_contribution'])}

The diagnostic gate is exactly:

`r_gated = r_ood if q_id>0 and q_ood>0 else 0`.

{markdown_table(gated_summary, ['weight', 'spearman_original_vs_gated', 'kendall_original_vs_gated', 'top3_overlap_original_vs_gated', 'most_benefited_algorithm', 'max_algorithm_score_benefit'])}

The ungated definition remains the paper's primary result.

## 9. Quality-reversal diagnostics

A reversal occurs when algorithm A has at least the stated OOD-NMSE advantage
over B on a task, but A receives a lower mixed task OOD-G component because of
retention. The summary below uses the 10x threshold; the CSV also reports
100x, 1000x, and 10000x thresholds.

{markdown_table(reversal_candidates, ['weight', 'reversal_count', 'max_ood_nmse_ratio', 'aggregate_pair_order_reversal_count', 'reversals_with_both_algorithms_in_current_top5'])}

## 10. Candidate-weight comparison

The pre-defined decision checks are:

1. minimum Kendall tau within `w +/- 0.05` is at least `0.90`;
2. Top-3 overlap throughout that neighborhood is `1.0`;
3. original-vs-gated Spearman correlation is at least `0.90`;
4. an interpretable quality-dominant candidate limits full retention
   compensation to at most 2.5 OOD-NMSE orders while retaining a non-zero
   retention term.

The 2.5-order bound is an explicit operational rule for this analysis, not an
empirically optimized constant. The selection never uses the identity of the
winning algorithm.

## 11. Recommended weight

**Decision:** `{recommendation[0]}`

{recommendation[1]}

## 12. Limitations

1. The bootstrap quantifies sensitivity to the 50-task sample, not uncertainty
   over independently rerun seeds or hardware.
2. Only DRSR and LLM-SR use the existing model-split rerun snapshot; the other
   ten algorithms use the original clean archive.
3. The gated score is an anomaly diagnostic, not a retroactive replacement for
   the submitted definition.
4. Pairwise bootstrap probabilities are uncertainty summaries and should not
   be described as classical hypothesis-test p-values.

## Candidate-weight summary

{markdown_table(candidate_display, ['weight', 'quality_weight', 'retention_weight', 'equivalent_log_order_compensation', 'spearman_vs_w070', 'kendall_vs_w070', 'top3_overlap', 'number_of_pairwise_flips', 'number_of_crossings_within_plus_minus_0_05', 'number_of_quality_reversals', 'bootstrap_top1_algorithm', 'bootstrap_top1_probability', 'original_vs_gated_rank_correlation', 'recommendation'])}

## Reproducibility

- Git commit before analysis: `{reproducibility['git_commit']}`
- Python: `{reproducibility['python']}`
- Platform: `{reproducibility['platform']}`
- Package versions: `{json.dumps(reproducibility['packages'], sort_keys=True)}`
- Input files and SHA-256: see `reproducibility.json`

Commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
python analysis/ood_weight_sensitivity/analyze_ood_weight_sensitivity.py \\
  --bootstrap-reps {reproducibility['bootstrap_repetitions']} \\
  --bootstrap-seed {reproducibility['bootstrap_seed']}

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
timeout 60 python -m pytest -vv -s \\
  analysis/ood_weight_sensitivity/test_analyze_ood_weight_sensitivity.py
```
"""
    (output_dir / "OOD_G_weight_sensitivity_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "original_runs": args.original_runs.resolve(),
        "replacement_runs": args.replacement_runs.resolve(),
        "expected_merged_runs": args.expected_merged_runs.resolve(),
        "core50_manifest": args.core50_manifest.resolve(),
        "formal_components": args.formal_components.resolve(),
        "formal_scores": args.formal_scores.resolve(),
    }
    for label, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    merged, merge_audit = load_and_merge_inputs(
        input_paths["original_runs"],
        input_paths["replacement_runs"],
        input_paths["expected_merged_runs"],
        args.archive_results.resolve(),
    )
    run_metrics, components = compute_task_components(merged)
    audit = data_audit(merged, components, input_paths["core50_manifest"], merge_audit)
    reproduction, reproduction_errors = verify_formal_reproduction(
        components,
        input_paths["formal_components"],
        input_paths["formal_scores"],
    )
    (
        algorithm_components,
        scores,
        ranks,
        rank_stability,
        max_rank_change,
    ) = compute_weight_scan(components)
    interval_summary = interval_rank_summary(rank_stability)
    crossings = pairwise_crossings(algorithm_components)
    bootstrap_scores, bootstrap_ranks, bootstrap_pairs, _ = bootstrap_analysis(
        components,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    bootstrap_pair_summary = bootstrap_pairwise_stability_summary(bootstrap_pairs)
    original_vs_gated, gated_summary = gated_retention_analysis(components)
    failure_cases, failure_summary = failure_retention_diagnostics(components)
    reversals, reversal_cases = reversal_diagnostics(components, scores)
    candidate_weights = candidate_summary(
        rank_stability,
        ranks,
        crossings,
        reversals,
        bootstrap_ranks,
        gated_summary,
    )
    recommendation = choose_recommendation(candidate_weights)

    run_export_columns = [
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "source_version",
        "status",
        "valid_output",
        "metric_complete",
        "id_test_nmse",
        "ood_test_nmse",
        "id_nmse_used",
        "ood_nmse_used",
        "analysis_penalty_applied",
        "failure_reason",
        "termination_reason",
        "timeout_type",
        "raw_result_json",
    ]
    save_csv(run_metrics[run_export_columns], output_dir / "merged_run_level_input.csv")
    save_csv(components, output_dir / "dataset_components.csv")
    save_csv(algorithm_components, output_dir / "algorithm_components.csv")
    save_csv(scores, output_dir / "scores_by_weight.csv")
    save_csv(ranks, output_dir / "ranks_by_weight.csv")
    save_csv(rank_stability, output_dir / "rank_stability_by_weight.csv")
    save_csv(max_rank_change, output_dir / "algorithm_rank_sensitivity.csv")
    save_csv(interval_summary, output_dir / "rank_stability_intervals.csv")
    save_csv(crossings, output_dir / "pairwise_crossings.csv")
    save_csv(bootstrap_scores, output_dir / "bootstrap_score_summary.csv")
    save_csv(bootstrap_ranks, output_dir / "bootstrap_rank_summary.csv")
    save_csv(bootstrap_pairs, output_dir / "bootstrap_pairwise_probability.csv")
    save_csv(
        bootstrap_pair_summary,
        output_dir / "bootstrap_pairwise_stability_summary.csv",
    )
    save_csv(original_vs_gated, output_dir / "original_vs_gated.csv")
    save_csv(gated_summary, output_dir / "gated_rank_summary.csv")
    save_csv(failure_cases, output_dir / "failure_retention_cases.csv")
    save_csv(failure_summary, output_dir / "failure_retention_diagnostics.csv")
    save_csv(reversals, output_dir / "quality_reversal_diagnostics.csv")
    save_csv(reversal_cases, output_dir / "quality_reversal_cases_candidate_weights.csv")
    save_csv(candidate_weights, output_dir / "candidate_weight_summary.csv")
    save_csv(reproduction, output_dir / "w070_reproduction.csv")

    audit_payload = {
        **audit,
        "reproduction_errors": reproduction_errors,
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reproducibility = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "command": " ".join(sys.argv),
        "execution_environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "git_commit": git_commit(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": package_versions(),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "weights": [float(value) for value in ALL_WEIGHTS],
        "candidate_weights": [float(value) for value in CANDIDATE_WEIGHTS],
        "epsilon": EPSILON,
        "input_files": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in input_paths.items()
        },
        "recommendation": {
            "decision": recommendation[0],
            "reason": recommendation[1],
        },
    }
    (output_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_all(
        output_dir,
        algorithm_components,
        scores,
        ranks,
        rank_stability,
        bootstrap_ranks,
        original_vs_gated,
        reversals,
    )
    write_report(
        output_dir,
        audit,
        reproduction,
        reproduction_errors,
        algorithm_components,
        rank_stability,
        interval_summary,
        crossings,
        bootstrap_scores,
        bootstrap_ranks,
        bootstrap_pair_summary,
        max_rank_change,
        failure_cases,
        failure_summary,
        gated_summary,
        reversals,
        candidate_weights,
        recommendation,
        reproducibility,
    )
    print(json.dumps(audit_payload, indent=2, sort_keys=True))
    print(f"recommendation={recommendation[0]}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
