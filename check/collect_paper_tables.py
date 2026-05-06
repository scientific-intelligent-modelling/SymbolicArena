#!/usr/bin/env python3
"""Collect paper-ready data tables for SR-Infra."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "Paper-SRInfra" / "tables"

PF = ROOT / "exp-planning/04.Core50正式全量评测/analysis/paper_figures_20260505"
HEX = ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v2_formal_20260505"
HEX_V1 = ROOT / "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504"
SYMF = ROOT / "exp-planning/04.Core50正式全量评测/analysis/symf_formal_metrics_20260504"
P4 = ROOT / "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508"
E1 = ROOT / "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest"
CAND = ROOT / "experiment-results/benchmark_selection_dossier_20260422/tables"

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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
    return value


def format_df_for_md(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            out[col] = out[col].map(clean_value)
    return out


def write_table(name: str, df: pd.DataFrame, caption: str, notes: str = "") -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / name
    df = df.copy()
    df.to_csv(base.with_suffix(".csv"), index=False)
    md_df = format_df_for_md(df)
    md_text = f"# {caption}\n\n"
    if notes:
        md_text += f"{notes}\n\n"
    md_text += md_df.to_markdown(index=False) + "\n"
    base.with_suffix(".md").write_text(md_text, encoding="utf-8")
    tex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.3f}",
        caption=caption,
        label=f"tab:{name}",
    )
    base.with_suffix(".tex").write_text(tex, encoding="utf-8")
    return {
        "name": name,
        "caption": caption,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "files": [base.with_suffix(ext).name for ext in [".csv", ".md", ".tex"]],
        "notes": notes,
    }


def pct(x):
    return float(x) * 100.0


def table_dataset_stats() -> pd.DataFrame:
    df = read_csv(PF / "table01_dataset_statistics.csv")
    rename = {
        "family": "Family",
        "datasets": "#Datasets",
        "subgroups": "#Subgroups",
        "median_variables": "Median variables",
        "median_complexity": "Median formula ops",
        "median_train_samples": "Median train samples",
        "ood_available": "#OOD available",
    }
    return df.rename(columns=rename)


def table_candidate200_family() -> pd.DataFrame:
    df = read_csv(CAND / "stage1_candidate200_flat.csv")
    family = (
        df.groupby("family", dropna=False)
        .agg(
            **{
                "#Datasets": ("dataset", "count"),
                "Median gap": ("overall_gap_score", "median"),
                "Mean gap": ("overall_gap_score", "mean"),
                "PySR-adv": ("advantage_side", lambda s: int((s == "pysr").sum())),
                "LLM-SR-adv": ("advantage_side", lambda s: int((s == "llmsr").sum())),
            }
        )
        .reset_index()
        .rename(columns={"family": "Family"})
        .sort_values("#Datasets", ascending=False)
    )
    return family


def table_candidate200_modes() -> pd.DataFrame:
    df = read_csv(CAND / "stage1_candidate200_flat.csv")
    mode = (
        df.groupby(["pool", "selection_mode"], dropna=False)
        .agg(
            **{
                "#Datasets": ("dataset", "count"),
                "Median gap": ("overall_gap_score", "median"),
                "Mean gap": ("overall_gap_score", "mean"),
            }
        )
        .reset_index()
        .rename(columns={"pool": "Pool", "selection_mode": "Selection mode"})
        .sort_values(["Pool", "#Datasets"], ascending=[True, False])
    )
    return mode


def table_probe4_algorithm_health() -> pd.DataFrame:
    df = read_csv(E1 / "probe4_selection_nmse_only/probe4_algorithm_scores_nmse_only.csv")
    cols = [
        "algorithm",
        "taxonomy",
        "rows",
        "finite_id_ood_rate",
        "id_ood_explosion_rate_gt_100",
        "operational_stability_score",
        "discrimination_score",
        "coverage_score",
        "available_metric_score",
    ]
    out = df[cols].copy()
    out = out.rename(
        columns={
            "algorithm": "Algorithm",
            "taxonomy": "Taxonomy",
            "rows": "Rows",
            "finite_id_ood_rate": "Finite ID/OOD rate",
            "id_ood_explosion_rate_gt_100": "Explosion rate",
            "operational_stability_score": "Operational stability",
            "discrimination_score": "Discrimination",
            "coverage_score": "Coverage",
            "available_metric_score": "Score",
        }
    )
    return out.sort_values("Score", ascending=False)


def table_probe4_top_combos() -> pd.DataFrame:
    rows = [
        (
            "Health-gated rank 1",
            "DSO; PyOperon; QLattice; uDSR",
            0.6570,
            "strict score optimum",
        ),
        (
            "Health-gated rank 2",
            "DSO; iMCTS; PyOperon; uDSR",
            0.6512,
            "selected: near tie; adds MCTS/tree search",
        ),
        (
            "Health-gated rank 3",
            "DSO; gplearn; iMCTS; uDSR",
            0.6482,
            "close, but weaker evolutionary-GP coverage",
        ),
        (
            "Health-gated rank 4",
            "DSO; gplearn; PySR; TPSR",
            0.6454,
            "high score, but no explicit MCTS probe",
        ),
        (
            "Health-gated rank 5",
            "DSO; gplearn; iMCTS; PySR",
            0.6407,
            "lower panel fidelity than selected panel",
        ),
        (
            "Finite-only rank 7",
            "DRSR; gplearn; PySR; TPSR",
            0.6374,
            "LLM-assisted panel enters only after Top-5",
        ),
        (
            "Finite-only rank 9",
            "gplearn; LLM-SR; PySR; TPSR",
            0.6191,
            "LLM-SR remains below health-gated alternatives",
        ),
        (
            "No health gate",
            "E2ESR; PySR; QLattice; RAG-SR",
            0.6817,
            "rejected: missing/explosion artifacts dominate",
        ),
    ]
    return pd.DataFrame(rows, columns=["Audit setting", "Probe panel", "Score", "Decision"])


def table_probe4_full_completion() -> pd.DataFrame:
    data = json.loads((P4 / "probe4_postprocess_summary.json").read_text(encoding="utf-8"))
    rows = []
    for method, m in data["method_completion"].items():
        rows.append(
            {
                "Method": method,
                "Observed runs": m["observed_runs"],
                "Finished runs": m["finished_runs"],
                "Valid runs": m["valid_runs"],
                "Completion rate": m["completion_rate"],
                "Valid rate": m["valid_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("Method")


def table_probe4_dataset_labels() -> pd.DataFrame:
    data = json.loads((P4 / "probe4_postprocess_summary.json").read_text(encoding="utf-8"))
    rows = []
    for label, counts in [
        ("Difficulty", data["difficulty_bin_counts"]),
        ("Eligibility", data["eligible_class_counts"]),
        ("Failure mode", data["failure_mode_counts"]),
        ("Run outcome", data["run_outcome_counts"]),
    ]:
        for k, v in counts.items():
            rows.append({"Category": label, "Label": k, "Count": v})
    return pd.DataFrame(rows)


def table_core50_baseline_quality() -> pd.DataFrame:
    df = read_csv(PF / "core50_baseline_quality_metrics.csv")
    cols = [
        "subset",
        "size",
        "coverage",
        "mean_info",
        "rank_fidelity",
        "pairwise_win_agreement",
        "aggregate_error",
        "stability",
        "non_redundancy",
        "difficulty_balance",
        "family_match",
        "failure_match",
        "selection_balance",
        "raw_selection_score",
        "hard_constraint_violations",
        "hard_constraint_excess",
        "feasible_selection_score",
    ]
    out = df[cols].copy()
    core50_manifest = ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
    if core50_manifest.exists():
        out.loc[out["subset"].eq("Core-50"), "size"] = len(read_csv(core50_manifest))
    return out.rename(
        columns={
            "subset": "Subset",
            "size": "Size",
            "coverage": "Coverage",
            "mean_info": "Mean info",
            "rank_fidelity": "Rank fidelity",
            "pairwise_win_agreement": "Pairwise win agreement",
            "aggregate_error": "Aggregate error",
            "stability": "Stability",
            "non_redundancy": "Non-redundancy",
            "difficulty_balance": "Difficulty balance",
            "family_match": "Family match",
            "failure_match": "Failure match",
            "selection_balance": "Selection balance",
            "raw_selection_score": "Raw selection score",
            "hard_constraint_violations": "Hard violations",
            "hard_constraint_excess": "Hard violation excess",
            "feasible_selection_score": "Feasible selection score",
        }
    )


def table_k_scaling() -> pd.DataFrame:
    df = read_csv(PF / "figure14_k_scaling_metrics.csv")
    cols = [
        "K",
        "selector",
        "coverage",
        "mean_info",
        "rank_fidelity",
        "pairwise_win_agreement",
        "aggregate_error",
        "stability",
        "non_redundancy",
        "difficulty_balance",
    ]
    out = df[cols].copy()
    out["selector"] = out["selector"].fillna("unknown")
    return out.rename(
        columns={
            "selector": "Selector",
            "coverage": "Coverage",
            "mean_info": "Mean info",
            "rank_fidelity": "Rank fidelity",
            "pairwise_win_agreement": "Pairwise win agreement",
            "aggregate_error": "Aggregate error",
            "stability": "Stability",
            "non_redundancy": "Non-redundancy",
            "difficulty_balance": "Difficulty balance",
        }
    )


def table_clean_leaderboard() -> pd.DataFrame:
    df = read_csv(HEX_V1 / "clean_final_runs_updated.csv")
    for col in ["id_test_nmse", "ood_test_nmse"]:
        df[f"log_{col}"] = np.log10(np.maximum(df[col].astype(float), 1e-12))
        df[f"log_{col}"] = df[f"log_{col}"].replace([np.inf, -np.inf], np.nan).clip(-12, 12)
    rows = []
    for alg, g in df.groupby("algorithm"):
        rows.append(
            {
                "Algorithm": alg,
                "Runs": len(g),
                "Valid rate": g["valid_output"].astype(bool).mean(),
                "Metric complete rate": g["metric_complete"].astype(bool).mean(),
                "Median seconds": g["seconds"].median(),
                "Mean ID log NMSE": g["log_id_test_nmse"].fillna(12).mean(),
                "Mean OOD log NMSE": g["log_ood_test_nmse"].fillna(12).mean(),
                "Median OOD log NMSE": g["log_ood_test_nmse"].median(),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("Mean OOD log NMSE")


def table_hexagon_scores() -> pd.DataFrame:
    df = read_csv(HEX / "hexagon_scores_formal_with_ci.csv")
    df = df.sort_values("HexaScore_formal_with_ROB", ascending=False).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Rank": np.arange(1, len(df) + 1),
            "Algorithm": df["algorithm"].map(lambda x: ALGORITHM_DISPLAY.get(str(x).lower(), str(x))),
            "ID-Q": df["ID_Q"].round(1),
            "OOD-G": df["OOD_G"].round(1),
            "SYM-F": df["SYM_F"].round(1),
            "EFF": df["EFF"].round(1),
            "ROBU": df["ROB"].round(1),
            "STAB": df["STAB"].round(1),
            "HexaScore [95% CI]": [
                f"{score:.1f} [{lo:.1f}, {hi:.1f}]"
                for score, lo, hi in zip(
                    df["HexaScore_formal_with_ROB"],
                    df["HexaScore_formal_with_ROB_ci_low"],
                    df["HexaScore_formal_with_ROB_ci_high"],
                    strict=True,
                )
            ],
        }
    )
    return out


def table_symbolic_summary() -> pd.DataFrame:
    df = read_csv(SYMF / "symbolic_metrics_formal_algorithm_summary.csv")
    out = df.rename(
        columns={
            "algorithm": "Algorithm",
            "datasets": "#Datasets",
            "exact_equiv_rate": "Exact equiv rate",
            "cas_equiv_rate": "CAS equiv rate",
            "numeric_equiv_rate": "Numeric equiv rate",
            "pred_parse_rate": "Parse rate",
            "mean_tree_similarity": "Tree similarity",
            "mean_var_f1": "Variable F1",
            "mean_op_f1": "Operator F1",
        }
    )
    return out.sort_values("SYM_F_formal", ascending=False)


def table_noise_summary() -> pd.DataFrame:
    scores = read_csv(HEX / "hexagon_scores_formal.csv")
    noise = read_csv(HEX_V1 / "noise_completion_summary.csv")
    agg = (
        noise.groupby("algorithm")
        .agg(
            **{
                "Noise observed runs": ("observed_runs", "sum"),
                "Noise valid runs": ("valid_runs", "sum"),
                "Mean noise valid rate": ("valid_rate", "mean"),
                "Median noisy ID NMSE": ("median_id_nmse", "median"),
                "Median noisy OOD NMSE": ("median_ood_nmse", "median"),
            }
        )
        .reset_index()
    )
    out = scores[["algorithm", "ROB", "noise_cell_coverage", "noise_valid_rate"]].merge(agg, on="algorithm", how="left")
    out = out.rename(
        columns={
            "algorithm": "Algorithm",
            "noise_cell_coverage": "Noise cell coverage",
            "noise_valid_rate": "Overall noise valid rate",
        }
    )
    out = out.rename(columns={"ROB": "ROBU"})
    return out.sort_values("ROBU", ascending=False)


def table_noise_by_sigma() -> pd.DataFrame:
    df = read_csv(HEX_V1 / "noise_completion_summary.csv")
    cols = [
        "algorithm",
        "noise_sigma",
        "observed_runs",
        "valid_runs",
        "valid_rate",
        "median_id_nmse",
        "median_ood_nmse",
    ]
    return df[cols].rename(
        columns={
            "algorithm": "Algorithm",
            "noise_sigma": "Noise sigma",
            "observed_runs": "Observed runs",
            "valid_runs": "Valid runs",
            "valid_rate": "Valid rate",
            "median_id_nmse": "Median ID NMSE",
            "median_ood_nmse": "Median OOD NMSE",
        }
    ).sort_values(["Algorithm", "Noise sigma"])


def table_low_nmse_non_equiv() -> pd.DataFrame:
    df = read_csv(PF / "figure19_low_nmse_non_equivalent_cases.csv")
    cols = [
        "algorithm",
        "dataset",
        "seed",
        "id_test_nmse",
        "ood_test_nmse",
        "gt_expression_x",
        "pred_expression_cleaned",
        "sym_f_formal",
    ]
    return df[cols].rename(
        columns={
            "algorithm": "Algorithm",
            "dataset": "Dataset",
            "seed": "Seed",
            "id_test_nmse": "ID NMSE",
            "ood_test_nmse": "OOD NMSE",
            "gt_expression_x": "Ground truth",
            "pred_expression_cleaned": "Predicted expression",
            "sym_f_formal": "SYM-F",
        }
    )


def table_artifact_checklist() -> pd.DataFrame:
    rows = [
        ("GT-Reservoir-664 metadata", "CSV + dataset metadata", "exp-planning/01.双探针实验/datasets_runnable.csv", "Reservoir definition"),
        ("Candidate-200 selection log", "CSV", "experiment-results/benchmark_selection_dossier_20260422/tables/stage1_candidate200_flat.csv", "Dual-probe candidate construction"),
        ("E1 12-algorithm calibration", "CSV", "exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv", "Probe-4 selection"),
        ("Probe4-Full run-level table", "CSV", "exp-planning/03.四探针全量664三种子验证/generated/postprocess_final_20260501-105508/probe4_postprocess_run_level.csv", "Core-50 distillation"),
        ("Core-50 dataset manifest", "CSV", "exp-planning/04.Core50正式全量评测/core50_datasets.csv", "Frozen task list"),
        ("Clean Core-50 final runs", "CSV", "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/clean_final_runs_updated.csv", "Leaderboard scoring"),
        ("Formal symbolic metrics", "CSV", "exp-planning/04.Core50正式全量评测/analysis/symf_formal_metrics_20260504/symbolic_metrics_formal.csv", "SYM-F axis"),
        ("Noise robustness runs", "CSV", "exp-planning/04.Core50正式全量评测/analysis/hexagon_v1_with_artifacts_20260504/noise_final_runs.csv", "ROBU axis"),
        ("Final six-axis scores", "CSV", "exp-planning/04.Core50正式全量评测/analysis/hexagon_v2_formal_20260505/hexagon_scores_formal_with_ci.csv", "Main leaderboard table"),
        ("Paper figures", "PNG/PDF", "paper/Paper-SRInfra/imgs", "Publication figures"),
    ]
    return pd.DataFrame(rows, columns=["Component", "Format", "Path", "Role"])


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = []
    registry.append(write_table("table01_dataset_reservoir_stats", table_dataset_stats(), "Dataset statistics for GT-Reservoir-664 by family."))
    registry.append(write_table("table02_candidate200_family_summary", table_candidate200_family(), "Candidate-200 family composition and dual-probe gap summary."))
    registry.append(write_table("table03_candidate200_selection_modes", table_candidate200_modes(), "Candidate-200 selection modes and gap statistics."))
    registry.append(write_table("table04_probe4_algorithm_health", table_probe4_algorithm_health(), "Candidate-200 algorithm health metrics for Probe-4 selection."))
    registry.append(
        write_table(
            "table05_probe4_top_combinations",
            table_probe4_top_combos(),
            "Probe-4 candidate-panel audit with health-gated alternatives and relaxed-policy stress checks.",
            "The final selected panel is DSO + iMCTS + PyOperon + uDSR. It is selected under the health-gated identity-agnostic protocol because it is within epsilon=0.01 of the strict top panel while adding explicit MCTS/tree-search coverage.",
        )
    )
    registry.append(write_table("table06_probe4_full_completion", table_probe4_full_completion(), "Probe4-Full completion and valid-output rates on GT-Reservoir-664."))
    registry.append(write_table("table07_probe4_dataset_labels", table_probe4_dataset_labels(), "Probe4-Full dataset labels, eligibility classes, and run outcomes."))
    registry.append(
        write_table(
            "table08_core50_baseline_quality",
            table_core50_baseline_quality(),
            "Core-50 representativeness compared with subset-selection baselines.",
            "Core-50 is matched by dataset_dir against the frozen core50_datasets.csv manifest. Feasible selection score applies hard-constraint gating before the weighted objective.",
        )
    )
    registry.append(write_table("table09_k_scaling_metrics", table_k_scaling(), "Subset-size scaling metrics for Core-K selection."))
    registry.append(write_table("table10_clean_core50_leaderboard", table_clean_leaderboard(), "Clean Core-50 leaderboard summary over 12 algorithms and five seeds."))
    registry.append(write_table("table11_hexagon_scores_formal", table_hexagon_scores(), "Formal Core-50 six-axis scores with dataset-bootstrap 95% confidence intervals."))
    registry.append(write_table("table12_symbolic_fidelity_summary", table_symbolic_summary(), "Formal symbolic-fidelity metrics by algorithm."))
    registry.append(write_table("table13_noise_robustness_summary", table_noise_summary(), "Noise robustness summary by algorithm."))
    registry.append(write_table("table14_noise_by_sigma", table_noise_by_sigma(), "Noise-track completion and median clean-test NMSE by noise level."))
    registry.append(write_table("table15_low_nmse_non_equiv_cases", table_low_nmse_non_equiv(), "Low-NMSE but non-equivalent symbolic-regression cases."))
    registry.append(write_table("table16_artifact_checklist", table_artifact_checklist(), "Paper artifact and reproducibility table."))

    registry_df = pd.DataFrame(registry)
    registry_df.to_csv(OUT_DIR / "table_registry.csv", index=False)

    readme = ["# Paper Table Bank", ""]
    readme.append("This directory contains paper-ready tables collected from the current SR-Infra experimental artifacts.")
    readme.append("Each table is exported as CSV, Markdown, and LaTeX.")
    readme.append("")
    readme.append("| table | rows | purpose |")
    readme.append("| --- | ---: | --- |")
    for item in registry:
        readme.append(f"| `{item['name']}` | {item['rows']} | {item['caption']} |")
    readme.append("")
    readme.append("Recommended main-paper tables: `table01`, `table05`, `table08`, `table10`, `table11`, `table12`, and `table16`.")
    readme.append("Recommended appendix tables: all remaining tables, especially detailed noise-by-sigma and low-NMSE non-equivalent cases.")
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(OUT_DIR), "tables": len(registry)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
