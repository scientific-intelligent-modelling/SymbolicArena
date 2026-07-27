#!/usr/bin/env python3
"""Build the 15-algorithm six-axis uncertainty figure and inference tables."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
NEURIPS_ROOT = (
    REPO_ROOT
    / "A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h"
)
NEW3_ROOT = (
    REPO_ROOT
    / "A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h"
)
AAAI_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage4_ssr50_15algs_3seeds_3noise_3h"
)
FORMAL3H_ROOT = (
    REPO_ROOT
    / "benchmark-runs/formal3h/"
    "formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658"
)

CHECKPOINT_CSV = OUT_DIR / "aaai_new3_minute60_run_level.csv"
SYMF_PARAMS_CSV = OUT_DIR / "symf_params_ssr50.csv"
SYMF_CLEAN_RUNS_CSV = OUT_DIR / "symf_clean_minute60_run_level.csv"
SYMF_OUT_DIR = OUT_DIR / "symf_1h"
NEURIPS_COMPONENTS_CSV = (
    NEURIPS_ROOT / "hexagon/dataset_axis_components_formal.csv"
)
CORE50_CSV = NEURIPS_ROOT / "core50_manifest/core50_datasets.csv"
AAAI_DATASETS_CSV = FORMAL3H_ROOT / "manifest/datasets.csv"

NEW_ALGORITHMS = ("fepysr", "jaxsr", "symbolfit")
AXES = ("ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB")
COMPONENT_COLUMNS = {
    "ID_Q": "ID_Q_component",
    "OOD_G": "OOD_G_component",
    "SYM_F": "SYM_F_component",
    "EFF": "EFF_component",
    "ROB": "ROB_component",
    "STAB": "STAB_component",
}
DISPLAY_NAMES = {
    "dso": "DSO",
    "drsr": "DRSR",
    "e2esr": "E2ESR",
    "fepysr": "FePySR",
    "gplearn": "gplearn",
    "imcts": "iMCTS",
    "jaxsr": "JAXSR",
    "llmsr": "LLM-SR",
    "pyoperon": "PyOperon",
    "pysr": "PySR",
    "qlattice": "QLattice",
    "ragsr": "RAG-SR",
    "symbolfit": "SymbolFit",
    "tpsr": "TPSR",
    "udsr": "uDSR",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def phi_from_nmse(value: Any) -> float:
    number = finite_nonnegative(value)
    if number is None:
        number = 1e2
    log_value = np.clip(np.log10(max(number, 1e-12)), -12, 2)
    return float(1.0 - (log_value + 12.0) / 14.0)


def clipped_log_nmse(value: Any) -> float:
    number = finite_nonnegative(value)
    if number is None:
        number = 1e2
    return float(np.clip(np.log10(max(number, 1e-12)), -12, 2))


def safe_nmse(row: pd.Series, column: str) -> float:
    if not bool_value(row.get("valid_output")):
        return 1e2
    if not bool_value(row.get("metric_complete")):
        return 1e2
    value = finite_nonnegative(row.get(column))
    return value if value is not None else 1e2


def json_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, list) else set()


def set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def expression_skeleton(expression: Any) -> str | None:
    text = str(expression or "").strip()
    if not text:
        return None
    text = text.replace("np.", "").replace("numpy.", "").replace("math.", "")
    text = re.sub(r"\bx_(\d+)\b", r"x\1", text)
    text = text.replace("^", "**")
    try:
        root = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def rec(node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            return "C"
        if isinstance(node, ast.Name):
            return "X"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id.lower()
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr.lower()
            else:
                name = "call"
            return f"{name}({','.join(rec(arg) for arg in node.args)})"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return f"mul(C,{rec(node.operand)})"
            return f"{node.op.__class__.__name__.lower()}({rec(node.operand)})"
        if isinstance(node, ast.BinOp):
            names = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.Pow: "pow",
                ast.Mod: "mod",
            }
            name = names.get(type(node.op), type(node.op).__name__.lower())
            children = [rec(node.left), rec(node.right)]
            if name in {"add", "mul"}:
                children.sort()
            return f"{name}({','.join(children)})"
        return node.__class__.__name__.lower()

    return rec(root)


def structural_consistency(signatures: pd.Series) -> float:
    values = list(signatures)
    if len(values) < 2:
        return 0.0
    pairs = list(itertools.combinations(values, 2))
    matches = sum(
        isinstance(left, str)
        and bool(left)
        and isinstance(right, str)
        and bool(right)
        and left == right
        for left, right in pairs
    )
    return matches / len(pairs)


def locate_full664_params() -> Path:
    candidates = sorted(
        (
            NEW3_ROOT / "symf/source_snapshots"
        ).glob("*/params.csv")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"full-664 params snapshots={len(candidates)}, expected=1"
        )
    return candidates[0]


def prepare_symf_inputs() -> None:
    checkpoints = pd.read_csv(CHECKPOINT_CSV)
    checkpoints["algorithm"] = checkpoints["algorithm"].str.lower()
    clean = checkpoints[checkpoints["noise_tag"] == "clean"].copy()
    if len(clean) != 450:
        raise ValueError(f"clean checkpoint rows={len(clean)}, expected=450")

    full_params = pd.read_csv(locate_full664_params())
    core50 = pd.read_csv(CORE50_CSV)
    aaai_datasets = pd.read_csv(AAAI_DATASETS_CSV)
    if len(core50) != 50 or len(aaai_datasets) != 50:
        raise ValueError("Core-50 manifests must each contain 50 datasets")

    core_by_dir = {
        str(row["dataset_dir"]): row
        for _, row in core50.iterrows()
    }
    params_by_dir = {
        str(row["dataset_dir"]): row
        for _, row in full_params.iterrows()
    }
    params_rows: list[pd.Series] = []
    dataset_order: list[str] = []
    prefix = "sim-datasets-data/ssr50/datasets/"
    for index, row in aaai_datasets.reset_index(drop=True).iterrows():
        aaai_dir = str(row["dataset_dir"])
        if not aaai_dir.startswith(prefix):
            raise ValueError(f"unexpected SSR-50 path: {aaai_dir}")
        original_dir = "sim-datasets-data/" + aaai_dir[len(prefix) :]
        if original_dir not in core_by_dir:
            raise ValueError(f"AAAI dataset is not in NeurIPS Core-50: {aaai_dir}")
        if original_dir not in params_by_dir:
            raise ValueError(f"missing formal params: {original_dir}")
        params = params_by_dir[original_dir].copy()
        gid = f"g{index + 1:04d}"
        params["gid"] = gid
        params["core50_index"] = index + 1
        params["global_index"] = index + 1
        params["dataset"] = row["dataset_id"]
        params_rows.append(params)
        dataset_order.append(str(row["dataset_id"]))

    if dataset_order != aaai_datasets["dataset_id"].astype(str).tolist():
        raise AssertionError("dataset order changed unexpectedly")
    pd.DataFrame(params_rows).to_csv(SYMF_PARAMS_CSV, index=False)

    required = [
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "status",
        "valid_output",
        "metric_complete",
        "snapshot_path",
        "expression_canonical",
    ]
    clean[required].rename(
        columns={"snapshot_path": "result_path"}
    ).to_csv(SYMF_CLEAN_RUNS_CSV, index=False)

    summary = {
        "params_rows": len(params_rows),
        "clean_run_rows": len(clean),
        "algorithms": sorted(clean["algorithm"].unique().tolist()),
        "datasets": int(clean["dataset"].nunique()),
        "seeds": sorted(clean["seed"].unique().astype(int).tolist()),
        "params_csv": str(SYMF_PARAMS_CSV),
        "clean_runs_csv": str(SYMF_CLEAN_RUNS_CSV),
    }
    print(json.dumps(summary, indent=2))


def build_clean_components(
    checkpoints: pd.DataFrame,
    formal_metrics: pd.DataFrame,
) -> pd.DataFrame:
    clean = checkpoints[checkpoints["noise_tag"] == "clean"].copy()
    clean["id_nmse_used"] = clean.apply(
        lambda row: safe_nmse(row, "id_test_nmse"), axis=1
    )
    clean["ood_nmse_used"] = clean.apply(
        lambda row: safe_nmse(row, "ood_test_nmse"), axis=1
    )
    clean["id_log_used"] = clean["id_nmse_used"].map(clipped_log_nmse)
    clean["ood_log_used"] = clean["ood_nmse_used"].map(clipped_log_nmse)
    clean["valid_for_metric"] = (
        clean["valid_output"].map(bool_value)
        & clean["metric_complete"].map(bool_value)
    )

    numeric_rows: list[dict[str, Any]] = []
    for (algorithm, gid, dataset), group in clean.groupby(
        ["algorithm", "gid", "dataset"], sort=False
    ):
        med_id = float(np.median(group["id_nmse_used"]))
        med_ood = float(np.median(group["ood_nmse_used"]))
        q_id = phi_from_nmse(med_id)
        q_ood = phi_from_nmse(med_ood)
        delta = max(
            0.0,
            math.log10((med_ood + 1e-12) / (med_id + 1e-12)),
        )
        ood_retention = 1.0 - float(np.clip(delta / 4.0, 0.0, 1.0))
        q_ood_g = 0.7 * q_ood + 0.3 * ood_retention
        final_quality = 0.5 * q_id + 0.5 * q_ood
        seconds = pd.to_numeric(group["seconds"], errors="coerce")
        median_seconds = (
            float(seconds.median()) if seconds.notna().any() else 3600.0
        )
        time_bonus = 1.0 - float(
            np.clip(median_seconds / 3600.0, 0.0, 1.0)
        )
        eff = final_quality * (0.35 + 0.65 * time_bonus)
        id_iqr = float(
            np.percentile(group["id_log_used"], 75)
            - np.percentile(group["id_log_used"], 25)
        )
        ood_iqr = float(
            np.percentile(group["ood_log_used"], 75)
            - np.percentile(group["ood_log_used"], 25)
        )
        numeric_stability = 0.5 * (
            1.0 - float(np.clip(id_iqr / 3.0, 0.0, 1.0))
        ) + 0.5 * (
            1.0 - float(np.clip(ood_iqr / 3.0, 0.0, 1.0))
        )
        numeric_rows.append(
            {
                "algorithm": algorithm,
                "gid": gid,
                "dataset": dataset,
                "n_runs": len(group),
                "valid_rate": float(group["valid_for_metric"].mean()),
                "median_id_nmse": med_id,
                "median_ood_nmse": med_ood,
                "ID_Q_component": q_id,
                "q_ood": q_ood,
                "ood_retention": ood_retention,
                "OOD_G_component": q_ood_g,
                "final_quality": final_quality,
                "median_seconds": median_seconds,
                "EFF_component": eff,
                "id_log_iqr": id_iqr,
                "ood_log_iqr": ood_iqr,
                "numeric_stability": numeric_stability,
            }
        )
    numeric = pd.DataFrame(numeric_rows)

    formal = formal_metrics.copy()
    formal["algorithm"] = formal["algorithm"].str.lower()
    keys = ["algorithm", "gid", "dataset", "seed"]
    merged = clean.merge(
        formal[
            keys
            + [
                "pred_parse_ok",
                "pred_expression_cleaned",
                "pred_variables",
                "gt_variables",
                "pred_operators",
                "gt_operators",
                "var_f1",
                "op_f1",
                "sym_f_formal",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged["sym_f_formal"].isna().any():
        raise ValueError("formal SYM-F rows do not cover all clean checkpoints")

    proxy_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        parse_ok = bool_value(row["pred_parse_ok"])
        pred_vars = json_set(row["pred_variables"])
        gt_vars = json_set(row["gt_variables"])
        pred_ops = json_set(row["pred_operators"])
        gt_ops = json_set(row["gt_operators"])
        numeric_equiv_proxy = bool(
            parse_ok
            and row["id_nmse_used"] <= 1e-10
            and row["ood_nmse_used"] <= 1e-10
        )
        var_f1 = set_f1(pred_vars, gt_vars) if parse_ok else 0.0
        op_f1 = set_f1(pred_ops, gt_ops) if parse_ok else 0.0
        sof1 = 0.5 * var_f1 + 0.5 * op_f1
        tree_proxy = (
            jaccard(pred_vars | pred_ops, gt_vars | gt_ops)
            if parse_ok
            else 0.0
        )
        q_sym_proxy = (
            1.0
            if numeric_equiv_proxy
            else 0.3 * tree_proxy + 0.2 * sof1
        )
        proxy_rows.append(
            {
                "algorithm": row["algorithm"],
                "gid": row["gid"],
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "q_sym_proxy_run": float(np.clip(q_sym_proxy, 0.0, 1.0)),
                "pred_parse_ok": parse_ok,
                "numeric_equiv_proxy": numeric_equiv_proxy,
                "pred_skeleton": expression_skeleton(
                    row["pred_expression_cleaned"]
                ),
                "sym_f_formal": float(row["sym_f_formal"]),
            }
        )
    proxy = pd.DataFrame(proxy_rows)

    symbolic = (
        proxy.groupby(["algorithm", "gid", "dataset"], sort=False)
        .agg(
            SYM_F_proxy_component=("q_sym_proxy_run", "mean"),
            parse_rate=("pred_parse_ok", "mean"),
            numeric_equiv_proxy_rate=("numeric_equiv_proxy", "mean"),
            SYM_F_component=("sym_f_formal", "mean"),
            structural_consistency_proxy=(
                "pred_skeleton",
                structural_consistency,
            ),
        )
        .reset_index()
    )
    components = numeric.merge(
        symbolic,
        on=["algorithm", "gid", "dataset"],
        how="left",
        validate="one_to_one",
    )
    components["q_perf_proxy"] = (
        0.4 * components["ID_Q_component"]
        + 0.3 * components["OOD_G_component"]
        + 0.3 * components["SYM_F_proxy_component"]
    )
    components["pure_stab_proxy"] = (
        0.4 * components["numeric_stability"]
        + 0.3 * components["valid_rate"]
        + 0.3 * components["structural_consistency_proxy"]
    )
    components["STAB_component"] = (
        components["pure_stab_proxy"]
        * np.sqrt(components["q_perf_proxy"].clip(lower=0.0))
    )
    return components


def add_robustness(
    checkpoints: pd.DataFrame,
    clean_components: pd.DataFrame,
) -> pd.DataFrame:
    noise = checkpoints[checkpoints["noise_tag"] != "clean"].copy()
    noise["id_nmse_used"] = noise.apply(
        lambda row: safe_nmse(row, "id_test_nmse"), axis=1
    )
    noise["ood_nmse_used"] = noise.apply(
        lambda row: safe_nmse(row, "ood_test_nmse"), axis=1
    )
    quality = clean_components[
        ["algorithm", "gid", "dataset", "final_quality"]
    ].rename(columns={"final_quality": "q_clean"})
    noise_rows: list[dict[str, Any]] = []
    for (algorithm, gid, dataset, sigma), group in noise.groupby(
        ["algorithm", "gid", "dataset", "noise_sigma"],
        sort=False,
    ):
        med_id = float(np.median(group["id_nmse_used"]))
        med_ood = float(np.median(group["ood_nmse_used"]))
        q_noise = 0.5 * phi_from_nmse(med_id) + 0.5 * phi_from_nmse(
            med_ood
        )
        noise_rows.append(
            {
                "algorithm": algorithm,
                "gid": gid,
                "dataset": dataset,
                "noise_sigma": float(sigma),
                "q_noise": q_noise,
            }
        )
    noise_components = pd.DataFrame(noise_rows).merge(
        quality,
        on=["algorithm", "gid", "dataset"],
        how="left",
        validate="many_to_one",
    )
    noise_components["retention"] = (
        noise_components["q_noise"]
        / noise_components["q_clean"].clip(lower=0.25)
    ).clip(lower=0.0, upper=1.0)
    noise_components["rob_component"] = (
        0.7 * noise_components["q_noise"]
        + 0.3 * noise_components["retention"]
    )
    rob = (
        noise_components.groupby(
            ["algorithm", "gid", "dataset"], sort=False
        )
        .agg(ROB_component=("rob_component", "mean"))
        .reset_index()
    )
    output = clean_components.merge(
        rob,
        on=["algorithm", "gid", "dataset"],
        how="left",
        validate="one_to_one",
    )
    if output["ROB_component"].isna().any():
        raise ValueError("ROBU does not cover every new-algorithm dataset")
    return output


def load_combined_components() -> pd.DataFrame:
    neurips = pd.read_csv(NEURIPS_COMPONENTS_CSV)
    neurips["algorithm"] = neurips["algorithm"].str.lower()
    neurips["cohort"] = "NeurIPS original"
    neurips["clean_seeds"] = 5
    neurips["noise_levels"] = "0.01|0.05|0.10"

    checkpoints = pd.read_csv(CHECKPOINT_CSV)
    checkpoints["algorithm"] = checkpoints["algorithm"].str.lower()
    formal = pd.read_csv(SYMF_OUT_DIR / "symbolic_metrics_formal.csv")
    new = add_robustness(
        checkpoints,
        build_clean_components(checkpoints, formal),
    )
    new["cohort"] = "Post-submission"
    new["clean_seeds"] = 3
    new["noise_levels"] = "0.01|0.05"

    keep = [
        "algorithm",
        "gid",
        "dataset",
        "cohort",
        "clean_seeds",
        "noise_levels",
        *COMPONENT_COLUMNS.values(),
    ]
    combined = pd.concat([neurips[keep], new[keep]], ignore_index=True)
    for column in COMPONENT_COLUMNS.values():
        combined[column] = pd.to_numeric(
            combined[column], errors="raise"
        ).clip(0.0, 1.0)
    counts = combined.groupby("algorithm")["dataset"].nunique()
    if len(counts) != 15 or not (counts == 50).all():
        raise ValueError(f"invalid 15x50 grid: {counts.to_dict()}")
    if combined.duplicated(["algorithm", "dataset"]).any():
        raise ValueError("duplicate algorithm/dataset components")
    return combined


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_and_pairwise(
    components: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    algorithms = sorted(components["algorithm"].unique())
    datasets = sorted(components["dataset"].unique())
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0,
        len(datasets),
        size=(n_bootstrap, len(datasets)),
    )
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_permutations, len(datasets)),
    )
    means_rows: dict[str, dict[str, Any]] = {
        algorithm: {
            "algorithm": algorithm,
            "display_name": DISPLAY_NAMES[algorithm],
            "cohort": (
                "Post-submission"
                if algorithm in NEW_ALGORITHMS
                else "NeurIPS original"
            ),
        }
        for algorithm in algorithms
    }
    pairwise_rows: list[dict[str, Any]] = []

    for axis in AXES:
        column = COMPONENT_COLUMNS[axis]
        matrix = (
            components.pivot(
                index="algorithm",
                columns="dataset",
                values=column,
            )
            .reindex(index=algorithms, columns=datasets)
            .to_numpy(dtype=float)
            * 100.0
        )
        if np.isnan(matrix).any():
            raise ValueError(f"{axis}: incomplete algorithm/dataset matrix")
        bootstrap_means = np.stack(
            [
                values[bootstrap_indices].mean(axis=1)
                for values in matrix
            ]
        )
        for index, algorithm in enumerate(algorithms):
            means_rows[algorithm][axis] = float(matrix[index].mean())
            means_rows[algorithm][f"{axis}_ci_low"] = float(
                np.percentile(bootstrap_means[index], 2.5)
            )
            means_rows[algorithm][f"{axis}_ci_high"] = float(
                np.percentile(bootstrap_means[index], 97.5)
            )

        axis_rows: list[dict[str, Any]] = []
        p_values: list[float] = []
        for left_index, right_index in itertools.combinations(
            range(len(algorithms)), 2
        ):
            difference = matrix[left_index] - matrix[right_index]
            observed = float(difference.mean())
            bootstrap_difference = (
                bootstrap_means[left_index]
                - bootstrap_means[right_index]
            )
            ci_low = float(np.percentile(bootstrap_difference, 2.5))
            ci_high = float(np.percentile(bootstrap_difference, 97.5))
            null_distribution = signs @ difference / len(datasets)
            p_value = float(
                (
                    np.count_nonzero(
                        np.abs(null_distribution) >= abs(observed)
                    )
                    + 1
                )
                / (n_permutations + 1)
            )
            row = {
                "axis": axis,
                "algorithm_a": algorithms[left_index],
                "algorithm_b": algorithms[right_index],
                "difference_a_minus_b": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_probability_a_gt_b": float(
                    np.mean(bootstrap_difference > 0)
                ),
                "paired_sign_flip_p": p_value,
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "tasks": len(datasets),
            }
            axis_rows.append(row)
            p_values.append(p_value)
        adjusted = holm_adjust(p_values)
        for row, p_adjusted in zip(axis_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = p_adjusted
            row["significant_holm_0_05"] = bool(p_adjusted < 0.05)
        pairwise_rows.extend(axis_rows)

    means = pd.DataFrame(means_rows.values())
    means["six_axis_mean"] = means[list(AXES)].mean(axis=1)
    pairwise = pd.DataFrame(pairwise_rows)
    return means, pairwise


def invert_shared_y_axis(axes: Any) -> None:
    """共享 y 轴只反转一次，避免偶数次调用把排序翻回去。"""
    axes.ravel()[0].invert_yaxis()


def plot_uncertainty(means: pd.DataFrame, output: Path) -> None:
    order = means.sort_values(
        ["six_axis_mean", "algorithm"],
        ascending=[False, True],
    )["algorithm"].tolist()
    indexed = means.set_index("algorithm").loc[order]
    y_positions = np.arange(len(order))
    original_color = "#236782"
    added_color = "#D26932"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#4A4A4A",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 10.2),
        sharex=True,
        sharey=True,
    )
    for axis_name, axis_plot in zip(AXES, axes.ravel(), strict=True):
        for position, algorithm in enumerate(order):
            row = indexed.loc[algorithm]
            added = algorithm in NEW_ALGORITHMS
            color = added_color if added else original_color
            marker = "D" if added else "o"
            value = float(row[axis_name])
            low = float(row[f"{axis_name}_ci_low"])
            high = float(row[f"{axis_name}_ci_high"])
            axis_plot.errorbar(
                value,
                position,
                xerr=[[value - low], [high - value]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=2.2,
                capthick=1.0,
                markersize=4.8 if added else 4.4,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
            )
        axis_plot.set_title(
            {
                "ID_Q": "ID-Q",
                "OOD_G": "OOD-G",
                "SYM_F": "SYM-F",
                "EFF": "EFF",
                "ROB": "ROBU",
                "STAB": "STAB",
            }[axis_name]
        )
        axis_plot.set_xlim(0, 100)
        axis_plot.set_xticks([0, 20, 40, 60, 80, 100])
        axis_plot.set_yticks(
            y_positions,
            [DISPLAY_NAMES[algorithm] for algorithm in order],
        )
        axis_plot.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        axis_plot.set_axisbelow(True)
        axis_plot.set_xlabel("Score")
    invert_shared_y_axis(axes)

    original_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=original_color,
        markeredgecolor="white",
        markersize=7,
        label="NeurIPS original 12",
    )
    added_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor=added_color,
        markeredgecolor="white",
        markersize=7,
        label="Post-submission 3",
    )
    fig.suptitle(
        "Six-axis uncertainty across 15 symbolic regression methods",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=[original_handle, added_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.018,
        "Points are task-level means; bars are 95% confidence intervals from "
        "20,000 synchronized Core-50 task bootstrap resamples.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.004,
        "Original methods: NeurIPS 1h/5 seeds and ROBU at 1%, 5%, 10%; "
        "added methods: AAAI 3h trajectories truncated at 1h/3 seeds and ROBU "
        "at 1%, 5%.",
        ha="center",
        fontsize=8.3,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.91,
        bottom=0.085,
        wspace=0.17,
        hspace=0.24,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output.with_suffix(".jpg"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)


def write_readme(
    means: pd.DataFrame,
    pairwise: pd.DataFrame,
    n_bootstrap: int,
    n_permutations: int,
) -> None:
    significant = (
        pairwise.groupby("axis")["significant_holm_0_05"]
        .agg(["sum", "count"])
        .reset_index()
    )
    score_table = means.sort_values(
        "six_axis_mean", ascending=False
    )[
        ["display_name", *AXES, "six_axis_mean"]
    ].to_markdown(index=False, floatfmt=".2f")
    significance_table = significant.to_markdown(index=False)
    text = f"""# Six-axis uncertainty for 15 algorithms

This directory adds FePySR, JAXSR, and SymbolFit to the frozen NeurIPS
12-algorithm Core-50 six-axis components.

## Protocol

- Original 12 methods: frozen NeurIPS task components, 1h, 5 clean seeds, and
  ROBU from noise sigma 0.01/0.05/0.10.
- Added 3 methods: AAAI three-hour trajectories read at `minute_0060`, 3 seeds,
  with ROBU from the available sigma 0.01/0.05 runs.
- SYM-F for the added methods is recomputed from the exact one-hour checkpoint
  expressions using the frozen formal judge.
- Marginal 95% intervals use {n_bootstrap} synchronized task bootstrap
  resamples over the 50 shared tasks.
- Pairwise inference uses paired bootstrap intervals for score differences and
  {n_permutations} task-level sign-flip permutations, with Holm correction
  within each axis.

Marginal interval overlap is not used as the significance decision. The
pairwise difference table is the authoritative source for close comparisons.
This is a rebuttal supplement rather than a retrospective protocol change:
the original task components and formal judge remain frozen. EFF and STAB
retain the archived proxy-component definitions, and fine-grained ROBU
comparisons across the two cohorts should account for the different noise
grids stated above.

## Frozen-judge audit note

The frozen formal judge re-applies `feature_to_x_map` to run-level canonical
expressions that already use anonymous `x0`, `x1`, ... variables. For example,
on `g0001` (`Keijzer-11`) a raw `x0*x1` term is cleaned as `x0*x0`. This
supplement deliberately preserves that frozen behavior so the original 12
scores do not drift. A corrected judge must be followed by a complete
15-method SYM-F/STAB recomputation; corrected and frozen scores must not be
mixed in one figure.

## Scores

{score_table}

## Holm-significant pair counts

{significance_table}

## Outputs

- `six_axis_uncertainty_15algs.png/.pdf/.jpg`
- `six_axis_task_components_15algs.csv`
- `six_axis_means_ci_15algs.csv`
- `six_axis_pairwise_inference.csv`
- `analysis_summary.json`
- `reviewer_p5tg_statistical_significance.md`
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    components = load_combined_components()
    means, pairwise = bootstrap_and_pairwise(
        components,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    components.to_csv(
        OUT_DIR / "six_axis_task_components_15algs.csv",
        index=False,
    )
    means.to_csv(OUT_DIR / "six_axis_means_ci_15algs.csv", index=False)
    pairwise.to_csv(
        OUT_DIR / "six_axis_pairwise_inference.csv",
        index=False,
    )
    figure = OUT_DIR / "six_axis_uncertainty_15algs.png"
    plot_uncertainty(means, figure)
    write_readme(
        means,
        pairwise,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
    )

    summary = {
        "algorithms": int(components["algorithm"].nunique()),
        "datasets": int(components["dataset"].nunique()),
        "task_component_rows": len(components),
        "new_algorithms": list(NEW_ALGORITHMS),
        "bootstrap_resamples": args.bootstrap,
        "permutation_resamples": args.permutations,
        "random_seed": args.seed,
        "significant_pairs_holm": {
            axis: int(
                pairwise.loc[
                    pairwise["axis"] == axis,
                    "significant_holm_0_05",
                ].sum()
            )
            for axis in AXES
        },
        "input_sha256": {
            "neurips_task_components": sha256_file(NEURIPS_COMPONENTS_CSV),
            "aaai_new3_minute60_runs": sha256_file(CHECKPOINT_CSV),
            "new3_formal_symf": sha256_file(
                SYMF_OUT_DIR / "symbolic_metrics_formal.csv"
            ),
        },
        "output_sha256": {
            "task_components": sha256_file(
                OUT_DIR / "six_axis_task_components_15algs.csv"
            ),
            "means_ci": sha256_file(
                OUT_DIR / "six_axis_means_ci_15algs.csv"
            ),
            "pairwise_inference": sha256_file(
                OUT_DIR / "six_axis_pairwise_inference.csv"
            ),
            "figure_png": sha256_file(figure),
        },
        "figure_png": str(figure),
        "warnings": [
            "The added algorithms use 3 clean seeds; the original methods use 5.",
            "Added-method ROBU averages sigma 0.01/0.05; original-method ROBU also includes 0.10.",
            "EFF and STAB follow the archived proxy-component definitions.",
            "The frozen formal judge re-applies feature_to_x_map to canonical x-indexed expressions; this supplement preserves that behavior for comparability.",
        ],
    }
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-symf")
    prepare.set_defaults(func=lambda _: prepare_symf_inputs())

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--bootstrap", type=int, default=20000)
    analysis.add_argument("--permutations", type=int, default=20000)
    analysis.add_argument("--seed", type=int, default=20260727)
    analysis.set_defaults(func=analyze)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
