#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BATCH = "semantic200_llm_physics_v2_4host_seed1314_20260429-205250"
DEFAULT_REMOTE_ROOT = (
    "/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/"
    f"{DEFAULT_BATCH}"
)
DEFAULT_HOSTS = ("iaaccn23", "iaaccn24", "iaaccn25", "iaaccn26")
DEFAULT_TOOLS = ("llmsr", "drsr")
DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "exp-planning"
    / "02.E1选择验证"
    / "e1_final_results_current_20260429"
    / "digest"
    / "e1_12_dataset_algorithm_nmse_table.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "exp-planning"
    / "02.E1选择验证"
    / "semantic200_llm_physics_v2_comparison_20260430"
)


REMOTE_SCAN_SCRIPT = r"""
from __future__ import annotations

import json
import pathlib
import sys


def metric_nmse(result: dict, split: str):
    value = result.get(split)
    if isinstance(value, dict):
        return value.get("nmse")
    return None


def metric_r2(result: dict, split: str):
    value = result.get(split)
    if isinstance(value, dict):
        return value.get("r2")
    return None


def first_scalar(params: dict, key: str):
    value = params.get(key)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


root = pathlib.Path(sys.argv[1])
host = sys.argv[2]
tools = sys.argv[3].split(",")

for tool in tools:
    host_result_dir = root / tool / host / tool
    for path in sorted(host_result_dir.glob("*/result.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "tool": tool,
                        "host": host,
                        "result_path": str(path),
                        "parse_error": repr(exc),
                    },
                    ensure_ascii=False,
                )
            )
            continue

        task_label = str(result.get("task_label") or path.parent.name)
        dataset_id = task_label.split("_", 1)[0] if task_label.startswith("g") else ""
        artifact = result.get("canonical_artifact") or {}
        params = result.get("params") or {}
        row = {
            "dataset_id": dataset_id,
            "dataset_name": result.get("dataset") or path.parent.name,
            "algorithm": result.get("tool") or tool,
            "seed": result.get("seed"),
            "host": host,
            "task_label": task_label,
            "status": result.get("status"),
            "termination_reason": result.get("termination_reason"),
            "timeout_type": result.get("timeout_type"),
            "budget_exhausted": result.get("budget_exhausted"),
            "recovered_from_timeout": result.get("recovered_from_timeout"),
            "seconds": result.get("seconds"),
            "train_nmse": metric_nmse(result, "train"),
            "valid_nmse": metric_nmse(result, "valid"),
            "id_nmse": metric_nmse(result, "id_test"),
            "ood_nmse": metric_nmse(result, "ood_test"),
            "id_r2": metric_r2(result, "id_test"),
            "ood_r2": metric_r2(result, "ood_test"),
            "has_expression": bool(result.get("equation")),
            "artifact_valid": artifact.get("artifact_valid"),
            "normalized_expression": artifact.get("normalized_expression"),
            "instantiated_expression": artifact.get("instantiated_expression"),
            "variables": ";".join(str(v) for v in artifact.get("variables") or []),
            "operator_set": ";".join(str(v) for v in artifact.get("operator_set") or []),
            "ast_node_count": artifact.get("ast_node_count"),
            "tree_depth": artifact.get("tree_depth"),
            "n_features": params.get("n_features"),
            "feature_names": ";".join(str(v) for v in result.get("feature_names") or []),
            "target_name": result.get("target_name"),
            "inject_prompt_semantics": params.get("inject_prompt_semantics"),
            "llm_model_assignment": first_scalar(params, "llm_model_assignment"),
            "llm_config_path": first_scalar(params, "llm_config_path"),
            "background_preview": " ".join(str(params.get("background") or "").split())[:220],
            "result_path": str(path),
        }
        print(json.dumps(row, ensure_ascii=False))
"""


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _log10_clipped(value: Any, lower: float = 1e-12, upper: float = 1e12) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    number = min(max(number, lower), upper)
    return math.log10(number)


def _classify_delta(delta: float | None, threshold: float = 0.30103) -> str:
    if delta is None or not math.isfinite(delta):
        return "unavailable"
    if delta <= -threshold:
        return "semantic_better_ge_2x"
    if delta >= threshold:
        return "semantic_worse_ge_2x"
    return "similar_within_2x"


def _run_remote_scan(host: str, remote_root: str, tools: tuple[str, ...], timeout: int) -> list[dict[str, Any]]:
    ssh_target = host
    jump_args: list[str] = []
    if host.startswith("iaaccn"):
        suffix = host.removeprefix("iaaccn")
        if suffix.isdigit() and int(suffix) >= 25:
            # 本地直连 25/26 偶发 banner timeout，稳定路径是先到 iaaccn23 再走内网。
            jump_args = ["-J", "iaaccn23"]
            ssh_target = f"zhangziwen@10.10.100.{int(suffix)}"

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        *jump_args,
        ssh_target,
        "python3",
        "-",
        remote_root,
        host,
        ",".join(tools),
    ]
    proc = subprocess.run(cmd, input=REMOTE_SCAN_SCRIPT, capture_output=True, text=True, timeout=timeout + 240)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"remote scan failed on {host}: returncode={proc.returncode}")

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def collect_semantic_rows(
    hosts: tuple[str, ...],
    remote_root: str,
    tools: tuple[str, ...],
    connect_timeout: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for host in hosts:
        host_rows = _run_remote_scan(host, remote_root, tools, connect_timeout)
        rows.extend(host_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 同一台机器只应有自己的 50 条；若路径残留造成重复，保留最新/最完整的一条。
    df["_metric_count"] = df[["train_nmse", "valid_nmse", "id_nmse", "ood_nmse"]].notna().sum(axis=1)
    df = (
        df.sort_values(["algorithm", "dataset_id", "_metric_count", "seconds"], na_position="first")
        .drop_duplicates(["algorithm", "dataset_id"], keep="last")
        .drop(columns=["_metric_count"])
        .reset_index(drop=True)
    )
    return df


def load_baseline(path: Path, tools: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(tools)].copy()
    keep = [
        "dataset_id",
        "dataset_name",
        "family",
        "srsd_variant",
        "algorithm",
        "train_nmse",
        "valid_nmse",
        "id_nmse",
        "ood_nmse",
    ]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"baseline missing columns: {missing}")
    return df[keep].copy()


def build_comparison(baseline: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    merged = baseline.merge(
        semantic,
        on=["dataset_id", "algorithm"],
        how="outer",
        suffixes=("_nonsemantic", "_semantic"),
    )
    if "dataset_name_nonsemantic" in merged.columns and "dataset_name_semantic" in merged.columns:
        merged["dataset_name"] = merged["dataset_name_nonsemantic"].fillna(merged["dataset_name_semantic"])

    for split in ("train", "valid", "id", "ood"):
        old_col = f"{split}_nmse_nonsemantic"
        new_col = f"{split}_nmse_semantic"
        merged[f"log10_{split}_nmse_nonsemantic_clipped"] = merged[old_col].map(_log10_clipped)
        merged[f"log10_{split}_nmse_semantic_clipped"] = merged[new_col].map(_log10_clipped)
        merged[f"delta_log10_{split}_nmse_semantic_minus_nonsemantic"] = (
            merged[f"log10_{split}_nmse_semantic_clipped"]
            - merged[f"log10_{split}_nmse_nonsemantic_clipped"]
        )
        merged[f"{split}_outcome_2x"] = merged[f"delta_log10_{split}_nmse_semantic_minus_nonsemantic"].map(
            _classify_delta
        )

    column_order = [
        "dataset_id",
        "dataset_name",
        "family",
        "srsd_variant",
        "algorithm",
        "status",
        "termination_reason",
        "timeout_type",
        "budget_exhausted",
        "recovered_from_timeout",
        "seconds",
        "train_nmse_nonsemantic",
        "train_nmse_semantic",
        "delta_log10_train_nmse_semantic_minus_nonsemantic",
        "train_outcome_2x",
        "valid_nmse_nonsemantic",
        "valid_nmse_semantic",
        "delta_log10_valid_nmse_semantic_minus_nonsemantic",
        "valid_outcome_2x",
        "id_nmse_nonsemantic",
        "id_nmse_semantic",
        "delta_log10_id_nmse_semantic_minus_nonsemantic",
        "id_outcome_2x",
        "ood_nmse_nonsemantic",
        "ood_nmse_semantic",
        "delta_log10_ood_nmse_semantic_minus_nonsemantic",
        "ood_outcome_2x",
        "normalized_expression",
        "instantiated_expression",
        "variables",
        "operator_set",
        "ast_node_count",
        "tree_depth",
        "n_features",
        "feature_names",
        "target_name",
        "inject_prompt_semantics",
        "llm_model_assignment",
        "background_preview",
        "result_path",
    ]
    columns = [col for col in column_order if col in merged.columns]
    extras = [col for col in merged.columns if col not in columns and not col.endswith("_nonsemantic")]
    return merged[columns + extras].sort_values(["algorithm", "dataset_id"]).reset_index(drop=True)


def _summary_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(part)
        row["semantic_rows_with_metrics"] = int(part["ood_nmse_semantic"].notna().sum())
        row["nonsemantic_rows_with_metrics"] = int(part["ood_nmse_nonsemantic"].notna().sum())
        for split in ("train", "valid", "id", "ood"):
            delta_col = f"delta_log10_{split}_nmse_semantic_minus_nonsemantic"
            row[f"median_log10_{split}_nonsemantic"] = part[
                f"log10_{split}_nmse_nonsemantic_clipped"
            ].median()
            row[f"median_log10_{split}_semantic"] = part[f"log10_{split}_nmse_semantic_clipped"].median()
            row[f"median_delta_log10_{split}"] = part[delta_col].median()
            row[f"mean_delta_log10_{split}"] = part[delta_col].mean()
            row[f"{split}_better_ge_2x"] = int((part[f"{split}_outcome_2x"] == "semantic_better_ge_2x").sum())
            row[f"{split}_similar_within_2x"] = int((part[f"{split}_outcome_2x"] == "similar_within_2x").sum())
            row[f"{split}_worse_ge_2x"] = int((part[f"{split}_outcome_2x"] == "semantic_worse_ge_2x").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_top_changes(df: pd.DataFrame, metric: str = "ood", n: int = 20) -> pd.DataFrame:
    delta_col = f"delta_log10_{metric}_nmse_semantic_minus_nonsemantic"
    cols = [
        "dataset_id",
        "dataset_name",
        "family",
        "srsd_variant",
        "algorithm",
        f"{metric}_nmse_nonsemantic",
        f"{metric}_nmse_semantic",
        delta_col,
        f"{metric}_outcome_2x",
        "normalized_expression",
        "llm_model_assignment",
    ]
    cols = [col for col in cols if col in df.columns]
    valid = df[df[delta_col].notna()].copy()
    improved = valid.nsmallest(n, delta_col)[cols].copy()
    worsened = valid.nlargest(n, delta_col)[cols].copy()
    improved.insert(0, "change_type", f"best_{metric}_semantic_improvements")
    worsened.insert(0, "change_type", f"worst_{metric}_semantic_regressions")
    return pd.concat([improved, worsened], ignore_index=True)


def write_readme(output_dir: Path, comparison: pd.DataFrame, method_summary: pd.DataFrame) -> None:
    lines = [
        "# Semantic-200 LLM Comparison",
        "",
        f"- 新语义批次：`{DEFAULT_BATCH}`",
        "- 旧基线：`e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv`",
        "- 比较对象：`llmsr`、`drsr`，均为 Candidate-200、seed=1314。",
        "- 主要判据：`delta_log10_*_nmse_semantic_minus_nonsemantic`，负数表示加入语义后 NMSE 变小。",
        "- delta 按 `NMSE` 裁剪到 `[1e-12, 1e12]` 后计算，避免少数爆炸值主导均值。",
        "",
        "## Coverage",
        "",
    ]
    coverage = comparison.groupby("algorithm").agg(
        rows=("dataset_id", "count"),
        semantic_metrics=("ood_nmse_semantic", lambda s: int(s.notna().sum())),
        nonsemantic_metrics=("ood_nmse_nonsemantic", lambda s: int(s.notna().sum())),
    )
    lines.extend(_markdown_table(coverage.reset_index()))
    lines.extend(["", "## Method Summary", ""])
    display_cols = [
        "algorithm",
        "n",
        "median_log10_ood_nonsemantic",
        "median_log10_ood_semantic",
        "median_delta_log10_ood",
        "ood_better_ge_2x",
        "ood_similar_within_2x",
        "ood_worse_ge_2x",
        "median_log10_valid_nonsemantic",
        "median_log10_valid_semantic",
        "median_delta_log10_valid",
        "valid_better_ge_2x",
        "valid_similar_within_2x",
        "valid_worse_ge_2x",
    ]
    lines.extend(_markdown_table(method_summary[display_cols]))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `semantic_results_raw.csv`：远端新语义批次 `result.json` 摘要。",
            "- `semantic_vs_nonsemantic_run_table.csv`：逐数据集逐算法对比主表。",
            "- `semantic_vs_nonsemantic_method_summary.csv`：按算法汇总。",
            "- `semantic_vs_nonsemantic_family_summary.csv`：按算法和 family 汇总。",
            "- `semantic_vs_nonsemantic_top_changes.csv`：OOD 指标改善/退化最大的样本。",
        ]
    )
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in columns) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较 Candidate-200 的语义/非语义 llmsr 与 drsr 结果")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="旧非语义 E1 指标 CSV")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="远端语义批次根目录")
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS), help="逗号分隔的远端主机")
    parser.add_argument("--tools", default=",".join(DEFAULT_TOOLS), help="逗号分隔的算法名")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--connect-timeout", type=int, default=10, help="SSH 连接超时秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hosts = tuple(host.strip() for host in args.hosts.split(",") if host.strip())
    tools = tuple(tool.strip() for tool in args.tools.split(",") if tool.strip())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic = collect_semantic_rows(hosts, args.remote_root, tools, args.connect_timeout)
    baseline = load_baseline(Path(args.baseline), tools)
    comparison = build_comparison(baseline, semantic)
    method_summary = _summary_group(comparison, ["algorithm"])
    family_summary = _summary_group(comparison, ["algorithm", "family"])
    top_changes = build_top_changes(comparison, metric="ood", n=20)

    semantic.to_csv(output_dir / "semantic_results_raw.csv", index=False)
    comparison.to_csv(output_dir / "semantic_vs_nonsemantic_run_table.csv", index=False)
    method_summary.to_csv(output_dir / "semantic_vs_nonsemantic_method_summary.csv", index=False)
    family_summary.to_csv(output_dir / "semantic_vs_nonsemantic_family_summary.csv", index=False)
    top_changes.to_csv(output_dir / "semantic_vs_nonsemantic_top_changes.csv", index=False)
    write_readme(output_dir, comparison, method_summary)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "semantic_rows": len(semantic),
                "comparison_rows": len(comparison),
                "hosts": hosts,
                "tools": tools,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
