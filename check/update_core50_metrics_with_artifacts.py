#!/usr/bin/env python3
"""用本地拉回的 Core50 clean/noise artifacts 补齐六轴指标统计。

口径：
- clean 正式榜单继续使用 12 algorithms x Core50 x 5 seeds。
- `llmsr` / `drsr` 的 clean 5 seeds 用有物理语义的
  `core50_llmsr_drsr_modelsplit_rerun_20260503` 替换旧结果。
- DRSR seed5/6/7 clean 补跑作为 budget audit 单独输出，不混入正式 5-seed 榜单。
- ROB 使用 noisy-train / clean-test 的 3 个噪声水平结果计算。
- SYM-F / EFF / STAB 仍沿用 v0 proxy；EFF 不混用局部 minute snapshots。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_DIR = REPO_ROOT / "check"
sys.path.insert(0, str(CHECK_DIR))

import plot_core50_hexagon_metrics as base  # noqa: E402


CORE50_ROOT = REPO_ROOT / "exp-planning/04.Core50正式全量评测"
NOISE_ROOT = REPO_ROOT / "exp-planning/05.Core50噪声鲁棒性评测"
DEFAULT_RUNS_CSV = (
    CORE50_ROOT
    / "generated/core50_12alg_5seed_final_results/analysis/core50_12alg_run_level_log_nmse_20260503.csv"
)
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments/core50_12alg_5seed_all_20260502-065700"
DEFAULT_CORE50_CSV = CORE50_ROOT / "core50_datasets.csv"
DEFAULT_MANIFEST = NOISE_ROOT / "results/LOCAL_ARTIFACT_MANIFEST_20260504-1932.json"
DEFAULT_OUTDIR = CORE50_ROOT / "analysis/hexagon_v1_with_artifacts_20260504"
NOISE_LEVELS = [0.01, 0.05, 0.10]
LLM_METHODS = {"llmsr", "drsr"}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) and out >= 0 else None


def split_nmse(payload: dict[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    return finite_float(block.get("nmse"))


def metric_complete(payload: dict[str, Any]) -> bool:
    return split_nmse(payload, "id_test") is not None and split_nmse(payload, "ood_test") is not None


def valid_output(payload: dict[str, Any]) -> bool:
    artifact = payload.get("canonical_artifact")
    artifact_valid = None
    if isinstance(artifact, dict):
        artifact_valid = artifact.get("artifact_valid")
    return bool(payload.get("equation")) and artifact_valid is not False and metric_complete(payload)


def find_base_from_report(report_path: Path) -> Path | None:
    parts = list(report_path.parts)
    try:
        idx = parts.index("__launcher__")
    except ValueError:
        return None
    return Path(*parts[:idx])


def infer_model_split(path: Path) -> str:
    text = str(path)
    if "/turbo/" in text:
        return "turbo"
    if "/base/" in text:
        return "base"
    return ""


def infer_sigma(batch: str, payload: dict[str, Any]) -> float:
    noise = payload.get("train_label_noise")
    if isinstance(noise, dict):
        sigma = finite_float(noise.get("sigma"))
        if sigma is not None:
            return sigma
    if "sigma001" in batch:
        return 0.01
    if "sigma005" in batch:
        return 0.05
    if "sigma010" in batch:
        return 0.10
    return 0.0


def load_manifest_dirs(manifest_path: Path) -> dict[str, Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for item in data.get("used_artifact_dirs", []):
        path = REPO_ROOT / item["path"]
        if path.exists():
            out[item["name"]] = path
    return out


def result_path_for_report(report_path: Path, report: dict[str, Any]) -> Path | None:
    base_dir = find_base_from_report(report_path)
    if base_dir is None:
        return None
    idx = report.get("task_global_index")
    try:
        gid = int(idx)
    except Exception:
        return None
    candidates = [
        p
        for p in base_dir.glob(f"*/g{gid:04d}_*/result.json")
        if "/experiments/" not in str(p)
    ]
    if candidates:
        return sorted(candidates)[0]
    return None


def artifact_rows(root: Path, kind: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(root.glob("**/*.report.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result_path = result_path_for_report(report_path, report)
        payload: dict[str, Any] = {}
        if result_path and result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        tool = str(payload.get("tool") or "").lower()
        if not tool:
            # 兜底从路径推断，主要用于缺失 result.json 的污染排查。
            for part in report_path.parts:
                if part in {
                    "llmsr",
                    "drsr",
                    "tpsr",
                    "pysr",
                    "dso",
                    "gplearn",
                    "pyoperon",
                    "imcts",
                    "udsr",
                    "ragsr",
                    "e2esr",
                    "qlattice",
                }:
                    tool = part
                    break
        batch = next((part for part in report_path.parts if part.startswith("core50_")), "")
        gid_value = payload.get("task_global_index", report.get("task_global_index"))
        try:
            gid_int = int(gid_value)
            gid = f"g{gid_int:04d}"
        except Exception:
            gid_int = None
            gid = None
        seed_value = payload.get("seed")
        if seed_value is None:
            m = re.search(r"/seed(\d+)/", str(report_path))
            seed_value = int(m.group(1)) if m else None
        host = next((part for part in report_path.parts if part.startswith("iaaccn")), "")
        status = payload.get("status") or report.get("status")
        m_complete = metric_complete(payload)
        v_output = valid_output(payload)
        rows.append(
            {
                "algorithm": tool,
                "gid": gid,
                "gid_int": gid_int,
                "dataset": payload.get("dataset"),
                "seed": seed_value,
                "status": status,
                "valid_output": bool(v_output),
                "seconds": finite_float(payload.get("seconds")) or finite_float(report.get("seconds")) or np.nan,
                "train_nmse": split_nmse(payload, "train"),
                "valid_nmse": split_nmse(payload, "valid"),
                "id_test_nmse": split_nmse(payload, "id_test"),
                "ood_test_nmse": split_nmse(payload, "ood_test"),
                "metric_complete": bool(m_complete),
                "kind": kind,
                "noise_sigma": infer_sigma(batch, payload),
                "source_batch": batch,
                "source_host": host,
                "model_split": infer_model_split(report_path),
                "report_path": str(report_path),
                "result_path": str(result_path) if result_path else "",
                "payload": payload,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 一个 task 理论上只有一条 report；若重复拉取，保留最后一条非空 result。
    subset = ["algorithm", "gid", "dataset", "seed", "noise_sigma", "source_batch", "model_split"]
    df["_has_result"] = df["result_path"].astype(str).str.len() > 0
    df = df.sort_values(["_has_result", "report_path"]).drop_duplicates(subset=subset, keep="last")
    return df.drop(columns=["_has_result"])


def dedupe_artifact_runs(df: pd.DataFrame) -> pd.DataFrame:
    """按一次算法运行的身份去重，消除多目录重复拉取造成的重复统计。"""
    if df.empty:
        return df
    out = df.copy()
    gid = out["gid"].astype("string").fillna("")
    dataset = out["dataset"].astype("string").fillna("")
    out["_run_id"] = gid.where(gid.ne(""), dataset)
    subset = ["algorithm", "_run_id", "seed", "noise_sigma"]
    out["_has_result"] = out["result_path"].astype(str).str.len() > 0
    out["_metric_complete"] = out["metric_complete"].astype(bool)
    out["_valid_output"] = out["valid_output"].astype(bool)
    out["_seconds"] = pd.to_numeric(out["seconds"], errors="coerce").fillna(-1)
    out = out.sort_values(
        ["_has_result", "_metric_complete", "_valid_output", "_seconds", "report_path"],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=subset, keep="last")
    return out.drop(columns=["_run_id", "_has_result", "_metric_complete", "_valid_output", "_seconds"])


def concat_artifact_roots(roots: list[Path], kind: str) -> pd.DataFrame:
    frames = [artifact_rows(root, kind) for root in roots]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return dedupe_artifact_runs(pd.concat(frames, ignore_index=True))


def rows_for_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    out = df[cols].copy()
    for col in ["train_nmse", "valid_nmse", "id_test_nmse", "ood_test_nmse", "seconds"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["valid_output"] = out["valid_output"].astype(bool)
    out["metric_complete"] = out["metric_complete"].astype(bool)
    return out


def extract_return_from_function(source: str) -> str:
    text = str(source or "")
    m = re.search(r"return\s+(.+)", text)
    return m.group(1).strip() if m else text


def artifact_result_records(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        artifact = payload.get("canonical_artifact") if isinstance(payload, dict) else {}
        if not isinstance(artifact, dict):
            artifact = {}
        expr = (
            artifact.get("normalized_expression")
            or artifact.get("instantiated_expression")
            or artifact.get("return_expression_source")
            or extract_return_from_function(payload.get("equation", ""))
        )
        parsed = base.parse_expr(expr or "")
        rows.append(
            {
                "algorithm": row.get("algorithm"),
                "gid": row.get("gid"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "status_json": payload.get("status"),
                "result_path": row.get("result_path"),
                "experiment_dir": payload.get("experiment_dir"),
                "expression_raw": payload.get("equation"),
                "expression_canonical": expr,
                "expression_parse_ok": parsed is not None,
                "pred_variables": sorted(base.variable_set(parsed)),
                "pred_operators": sorted(base.operator_set(parsed)),
                "pred_skeleton": base.skeleton(parsed),
            }
        )
    return pd.DataFrame(rows)


def safe_nmse_for_row(row: pd.Series, col: str) -> float:
    if not bool(row.get("metric_complete")) or not bool(row.get("valid_output")):
        return 1e2
    value = finite_float(row.get(col))
    return value if value is not None else 1e2


def compute_robustness(noise: pd.DataFrame, clean_components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_quality = clean_components[["algorithm", "gid", "dataset", "final_quality"]].copy()
    clean_quality = clean_quality.rename(columns={"final_quality": "q_clean"})
    expected = clean_quality.assign(_key=1).merge(
        pd.DataFrame({"noise_sigma": NOISE_LEVELS, "_key": 1}), on="_key"
    ).drop(columns=["_key"])

    rows: list[dict[str, Any]] = []
    if not noise.empty:
        noise = noise.copy()
        noise["id_nmse_used"] = noise.apply(lambda r: safe_nmse_for_row(r, "id_test_nmse"), axis=1)
        noise["ood_nmse_used"] = noise.apply(lambda r: safe_nmse_for_row(r, "ood_test_nmse"), axis=1)
        for (alg, gid, dataset, sigma), g in noise.groupby(["algorithm", "gid", "dataset", "noise_sigma"], dropna=False):
            med_id = float(np.median(g["id_nmse_used"]))
            med_ood = float(np.median(g["ood_nmse_used"]))
            q_noise = 0.5 * base.phi_from_nmse(med_id) + 0.5 * base.phi_from_nmse(med_ood)
            rows.append(
                {
                    "algorithm": alg,
                    "gid": gid,
                    "dataset": dataset,
                    "noise_sigma": float(sigma),
                    "observed_runs": int(len(g)),
                    "valid_runs": int((g["valid_output"].astype(bool) & g["metric_complete"].astype(bool)).sum()),
                    "median_noise_id_nmse": med_id,
                    "median_noise_ood_nmse": med_ood,
                    "q_noise": q_noise,
                }
            )
    observed = pd.DataFrame(rows)
    if observed.empty:
        observed = pd.DataFrame(columns=["algorithm", "gid", "dataset", "noise_sigma"])
    full = expected.merge(observed, on=["algorithm", "gid", "dataset", "noise_sigma"], how="left")
    full["observed_runs"] = full["observed_runs"].fillna(0).astype(int)
    full["valid_runs"] = full["valid_runs"].fillna(0).astype(int)
    full["q_noise"] = full["q_noise"].fillna(0.0)
    full["retention"] = (full["q_noise"] / full["q_clean"].clip(lower=0.25)).clip(lower=0.0, upper=1.0)
    full["rob_component"] = 0.7 * full["q_noise"] + 0.3 * full["retention"]
    alg = (
        full.groupby("algorithm", dropna=False)
        .agg(
            ROB=("rob_component", lambda x: 100 * float(np.mean(x))),
            noise_expected_cells=("rob_component", "size"),
            noise_observed_runs=("observed_runs", "sum"),
            noise_valid_runs=("valid_runs", "sum"),
            noise_cell_coverage=("observed_runs", lambda x: float((x > 0).mean())),
            noise_valid_rate=("valid_runs", lambda x: float(x.sum()) / max(1, int(full.loc[x.index, "observed_runs"].sum()))),
        )
        .reset_index()
    )
    return full, alg


def plot_hexagon_heatmap(scores: pd.DataFrame, out: Path) -> None:
    cols = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
    data = scores.set_index("algorithm")[cols].rename(columns={"ID_Q": "ID-Q", "OOD_G": "OOD-G", "SYM_F": "SYM-F"})
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.42 * len(data))))
    mat = data.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#111")
    ax.set_title("Core-50 Six-Axis Scores v1 (ROB from noisy track; SYM/EFF/STAB proxies)")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_report(
    outdir: Path,
    scores: pd.DataFrame,
    completion: pd.DataFrame,
    drsr_seed567_summary: pd.DataFrame,
    clean_counts: dict[str, Any],
) -> None:
    lines = [
        "# Core-50 hexagon v1 with artifacts",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Clean `llmsr` / `drsr` rows are replaced by the semantic model-split rerun.",
        "- ROB is computed from noisy-train / clean-test artifacts.",
        "- SYM-F / EFF / STAB remain proxy axes until formal symbolic judge and full clean minute AUC are wired.",
        "",
        "## Clean Replacement Counts",
        "",
    ]
    for key, value in clean_counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Hexagon Scores",
            "",
            scores[
                [
                    "algorithm",
                    "datasets",
                    "ID_Q",
                    "OOD_G",
                    "SYM_F",
                    "EFF",
                    "ROB",
                    "STAB",
                    "HexaScore_with_ROB",
                    "HexaScore_clean_proxy",
                ]
            ].to_markdown(index=False, floatfmt=".2f"),
            "",
            "## Noise Completion By Algorithm And Sigma",
            "",
            completion.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## DRSR Seed567 Clean Audit",
            "",
            drsr_seed567_summary.to_markdown(index=False, floatfmt=".3f") if not drsr_seed567_summary.empty else "- no seed567 rows found",
            "",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Core50 metrics using pulled clean/noise artifacts")
    parser.add_argument("--runs-csv", default=str(DEFAULT_RUNS_CSV))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--core50-csv", default=str(DEFAULT_CORE50_CSV))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact_dirs = load_manifest_dirs(Path(args.manifest))

    model_clean_roots = [p for name, p in artifact_dirs.items() if "clean_modelsplit" in name]
    seed567_roots = [p for name, p in artifact_dirs.items() if name in {"remote_artifacts_20260504-183740", "remote_artifacts_clean_retry_slim_20260504-193115"}]
    noise_roots = [p for name, p in artifact_dirs.items() if "noise_" in name and "_slim" in name]

    model_clean = concat_artifact_roots(model_clean_roots, "clean_modelsplit")
    seed567_clean = concat_artifact_roots(seed567_roots, "clean_seed567")
    noise = concat_artifact_roots(noise_roots, "noise")

    old_clean = pd.read_csv(args.runs_csv)
    replacement = rows_for_metrics(model_clean[model_clean["algorithm"].isin(LLM_METHODS)])
    clean_updated = pd.concat([old_clean[~old_clean["algorithm"].isin(LLM_METHODS)], replacement], ignore_index=True)
    clean_updated["algorithm"] = clean_updated["algorithm"].astype(str).str.lower()
    clean_updated = clean_updated.sort_values(["algorithm", "seed", "gid"]).reset_index(drop=True)

    run_df, dataset_df = base.compute_run_and_dataset_metrics(clean_updated)
    old_results = base.extract_result_records(Path(args.experiment_root))
    replacement_results = artifact_result_records(model_clean[model_clean["algorithm"].isin(LLM_METHODS)])
    result_df = pd.concat(
        [old_results[~old_results["algorithm"].isin(LLM_METHODS)], replacement_results],
        ignore_index=True,
    )
    core_meta = base.load_core50_metadata(Path(args.core50_csv))
    symbolic_df = base.compute_symbolic_proxy(run_df, result_df, core_meta)
    dataset_components, scores = base.compute_algorithm_scores(dataset_df, symbolic_df)
    rob_components, rob_alg = compute_robustness(rows_for_metrics(noise).assign(noise_sigma=noise["noise_sigma"].to_numpy()), dataset_components)
    scores = scores.drop(columns=["ROB"], errors="ignore").merge(rob_alg, on="algorithm", how="left")
    scores["ROB"] = scores["ROB"].fillna(0.0).clip(lower=0, upper=100)
    axes = ["ID_Q", "OOD_G", "SYM_F", "EFF", "ROB", "STAB"]
    scores["HexaScore_with_ROB"] = 100 * np.exp(
        np.mean(np.log(np.maximum(scores[axes].to_numpy(dtype=float) / 100.0, 1e-6)), axis=1)
    )
    scores = scores.sort_values("HexaScore_with_ROB", ascending=False)
    ci = base.bootstrap_ci(dataset_components, n=args.bootstrap)
    scores_ci = scores.merge(ci, on="algorithm", how="left")

    noise_completion = (
        rows_for_metrics(noise)
        .assign(noise_sigma=noise["noise_sigma"].to_numpy())
        .groupby(["algorithm", "noise_sigma"], dropna=False)
        .agg(
            observed_runs=("dataset", "size"),
            datasets=("dataset", "nunique"),
            valid_runs=("valid_output", "sum"),
            metric_complete=("metric_complete", "sum"),
            median_id_nmse=("id_test_nmse", "median"),
            median_ood_nmse=("ood_test_nmse", "median"),
        )
        .reset_index()
    )
    noise_completion["expected_runs"] = 250
    noise_completion["observed_rate"] = noise_completion["observed_runs"] / noise_completion["expected_runs"]
    noise_completion["valid_rate"] = noise_completion["valid_runs"] / noise_completion["observed_runs"].clip(lower=1)

    seed567_metrics = rows_for_metrics(seed567_clean)
    drsr_seed567_summary = pd.DataFrame()
    if not seed567_metrics.empty:
        drsr_seed567_summary = (
            seed567_metrics.groupby("seed", dropna=False)
            .agg(
                observed_runs=("dataset", "size"),
                datasets=("dataset", "nunique"),
                valid_runs=("valid_output", "sum"),
                median_id_nmse=("id_test_nmse", "median"),
                median_ood_nmse=("ood_test_nmse", "median"),
                median_seconds=("seconds", "median"),
            )
            .reset_index()
        )

    clean_counts = {
        "clean_updated_rows": len(clean_updated),
        "clean_updated_algorithms": clean_updated["algorithm"].nunique(),
        "clean_updated_algorithm_counts": dict(sorted(Counter(clean_updated["algorithm"]).items())),
        "modelsplit_llmsr_drsr_rows": len(replacement),
        "seed567_drsr_audit_rows": len(seed567_metrics),
        "noise_rows": len(noise),
    }

    clean_updated.to_csv(outdir / "clean_final_runs_updated.csv", index=False)
    rows_for_metrics(noise).assign(noise_sigma=noise["noise_sigma"].to_numpy()).to_csv(outdir / "noise_final_runs.csv", index=False)
    model_clean.drop(columns=["payload"], errors="ignore").to_csv(outdir / "clean_modelsplit_artifact_runs.csv", index=False)
    seed567_clean.drop(columns=["payload"], errors="ignore").to_csv(outdir / "clean_drsr_seed567_artifact_runs.csv", index=False)
    run_df.to_csv(outdir / "clean_final_runs_scored.csv", index=False)
    result_df.to_csv(outdir / "core50_result_expressions_updated.csv", index=False)
    symbolic_df.to_csv(outdir / "symbolic_metrics_proxy.csv", index=False)
    dataset_components.to_csv(outdir / "dataset_axis_components.csv", index=False)
    rob_components.to_csv(outdir / "robustness_components.csv", index=False)
    noise_completion.to_csv(outdir / "noise_completion_summary.csv", index=False)
    drsr_seed567_summary.to_csv(outdir / "drsr_seed567_clean_audit_summary.csv", index=False)
    scores.to_csv(outdir / "hexagon_scores.csv", index=False)
    scores_ci.to_csv(outdir / "hexagon_scores_with_ci.csv", index=False)
    (outdir / "metric_update_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "clean_counts": clean_counts,
                "noise_completion_rows": len(noise_completion),
                "output_dir": str(outdir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_hexagon_heatmap(scores, outdir / "fig_hexagon_heatmap_v1.png")
    base.save_metric_bar(scores, "HexaScore_with_ROB", outdir / "fig_hexascore_with_rob_bar.png", "HexaScore with ROB")
    base.save_scatter(scores, "OOD_G", "ROB", outdir / "fig_tradeoff_oodg_rob.png", "OOD-G vs ROB")
    write_report(outdir, scores, noise_completion, drsr_seed567_summary, clean_counts)
    print(json.dumps({"outdir": str(outdir), "scores": len(scores), "clean_rows": len(clean_updated), "noise_rows": len(noise)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
