#!/usr/bin/env python3
"""生成 uDSR/iMCTS/PySR 的 10--60 分钟五种子均值表。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = (
    REPO_ROOT
    / "A_Neurips_experiments/rebuttal/05_stage4_anytime_12algs_5seeds_clean_1h"
)
MINUTES = (10, 20, 30, 40, 50, 60)
ALGORITHMS = ("udsr", "imcts", "pysr")
METRICS = ("ID-Q", "OOD-G", "SYM-F", "EFF", "STAB")
TAG = "top3_10_20_30_40_50_60"


def load_base_module():
    path = REPO_ROOT / "check/analyze_neurips_stage4_anytime.py"
    spec = importlib.util.spec_from_file_location("stage4_anytime_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.TARGET_MINUTES = MINUTES
    module.ALGORITHM_ORDER = ALGORITHMS
    return module


def proxy_input(rows: pd.DataFrame, minutes: tuple[int, ...]) -> pd.DataFrame:
    selected = rows[
        rows["algorithm"].isin(ALGORITHMS) & rows["minute"].isin(minutes)
    ].copy()
    selected["status"] = selected["status"].fillna("no_valid_output")
    selected["result_path"] = ""
    columns = [
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "minute",
        "status",
        "valid_output",
        "metric_complete",
        "result_path",
        "expression_canonical",
        "id_test_nmse",
        "ood_test_nmse",
    ]
    return selected[columns]


def compact_table(scores: pd.DataFrame, base) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for algorithm in ALGORITHMS:
        subset = scores[scores["algorithm"] == algorithm].set_index("minute")
        for metric in METRICS:
            display = f"{metric} (proxy)" if metric in {"SYM-F", "STAB"} else metric
            row = {"算法": base.DISPLAY_NAMES[algorithm], "指标": display}
            for minute in MINUTES:
                row[f"{minute}min"] = f"{float(subset.loc[minute, metric]):.2f}"
            rows.append(row)
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Algorithm & Metric & 10min & 20min & 30min & 40min & 50min & 60min \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        values = " & ".join(str(row[f"{minute}min"]) for minute in MINUTES)
        lines.append(f"{row['算法']} & {row['指标']} & {values} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = load_base_module()
    run_path = OUTDIR / "clean_minute_run_level.csv.gz"
    runs = pd.read_csv(run_path)
    runs["algorithm"] = runs["algorithm"].map(base.normalize_algorithm)
    runs["minute"] = pd.to_numeric(runs["minute"], errors="raise").astype(int)
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)

    expected_rows = len(ALGORITHMS) * 50 * 5 * 60
    top3_runs = runs[runs["algorithm"].isin(ALGORITHMS)].copy()
    if len(top3_runs) != expected_rows:
        raise ValueError(f"top3 run-minute 应为 {expected_rows}，实际 {len(top3_runs)}")

    existing = pd.read_csv(OUTDIR / "checkpoint_symbolic_metrics_proxy.csv")
    existing = existing[
        existing["algorithm"].isin(ALGORITHMS) & existing["minute"].isin((10, 30, 60))
    ].copy()

    new_input = proxy_input(top3_runs, (20, 40, 50))
    input_path = OUTDIR / f"checkpoint_runs_for_symf_proxy_{TAG}.csv"
    new_input.to_csv(input_path, index=False)
    new_proxy_path = OUTDIR / f"checkpoint_symbolic_metrics_proxy_new_{TAG}.csv"
    base.compute_symf_proxy(input_path, base.DEFAULT_PARAMS, new_proxy_path)
    proxy = pd.concat([existing, pd.read_csv(new_proxy_path)], ignore_index=True)
    proxy_path = OUTDIR / f"checkpoint_symbolic_metrics_proxy_{TAG}.csv"
    proxy.to_csv(proxy_path, index=False)

    protocol, seed_mean, dataset = base.compute_tables(top3_runs, proxy, OUTDIR)
    score_path = OUTDIR / f"checkpoint_scores_seed_mean_{TAG}.csv"
    seed_mean.to_csv(score_path, index=False)
    protocol.to_csv(OUTDIR / f"checkpoint_scores_protocol_{TAG}.csv", index=False)
    dataset.to_csv(OUTDIR / f"dataset_axis_components_{TAG}.csv", index=False)

    table = compact_table(seed_mean, base)
    table.to_csv(OUTDIR / f"table_seed_mean_{TAG}.csv", index=False)
    (OUTDIR / f"table_seed_mean_{TAG}.md").write_text(
        table.to_markdown(index=False, disable_numparse=True) + "\n",
        encoding="utf-8",
    )
    write_latex(table, OUTDIR / f"table_seed_mean_{TAG}.tex")

    summary = {
        "algorithms": list(ALGORITHMS),
        "minutes": list(MINUTES),
        "datasets": 50,
        "seeds": [0, 1, 2, 3, 4],
        "table_rows": int(len(table)),
        "seed_aggregation": "mean run-level scores, then mean across datasets",
        "symf": "lightweight proxy",
        "stab": "lightweight structural-consistency proxy",
    }
    (OUTDIR / f"analysis_summary_{TAG}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
