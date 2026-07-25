#!/usr/bin/env python3
"""严格校验并合并正式 SYM-F 算法分片。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

CHECK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHECK_DIR))

import generate_symf_formal_metrics as generator  # noqa: E402


REQUIRED_METRIC_COLUMNS = {
    "algorithm",
    "gid",
    "dataset",
    "seed",
    "valid_for_symbolic",
    "pred_parse_ok",
    "cas_equiv",
    "numeric_equiv",
    "numeric_equiv_reason",
    "equiv_final",
    "tree_similarity",
    "var_f1",
    "op_f1",
    "sym_f_formal",
}


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["algorithm"] = normalized["algorithm"].map(
        generator.normalize_algorithm
    )
    normalized["gid"] = normalized["gid"].astype(str).str.strip()
    normalized["seed"] = pd.to_numeric(
        normalized["seed"],
        errors="raise",
    ).astype(int)
    return normalized


def validate_metrics_grid(
    metrics: pd.DataFrame,
    params: pd.DataFrame,
    *,
    expected_algorithm_keys: Sequence[str],
    expected_runs: int,
    expected_datasets: int,
    expected_seeds: Sequence[int],
    expected_runs_per_algorithm: int | None,
) -> dict[str, Any]:
    """校验 algorithm × gid × seed 的完整笛卡尔积。"""
    missing_columns = REQUIRED_METRIC_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"SYM-F 指标分片缺少字段: {sorted(missing_columns)}"
        )
    if "gid" not in params.columns:
        raise ValueError("参数 CSV 缺少 gid 字段")

    normalized = _normalize_keys(metrics)
    if normalized[["algorithm", "gid"]].eq("").any(axis=None):
        raise ValueError("SYM-F 指标存在空 algorithm 或 gid")

    key_columns = ["algorithm", "gid", "seed"]
    duplicate_rows = int(
        normalized.duplicated(key_columns, keep=False).sum()
    )
    if duplicate_rows:
        raise ValueError(
            "SYM-F 指标存在重复 algorithm/gid/seed: "
            f"{duplicate_rows}"
        )

    params_gids = params["gid"].astype(str).str.strip()
    if params_gids.eq("").any():
        raise ValueError("参数 CSV 存在空 gid")
    duplicate_param_gids = int(params_gids.duplicated(keep=False).sum())
    if duplicate_param_gids:
        raise ValueError(
            f"参数 CSV 存在重复 gid: {duplicate_param_gids}"
        )

    expected_algorithms = sorted(
        {
            generator.normalize_algorithm(value)
            for value in expected_algorithm_keys
            if str(value).strip()
        }
    )
    expected_gid_values = sorted(params_gids.tolist())
    expected_seed_values = sorted({int(value) for value in expected_seeds})

    if len(expected_gid_values) != expected_datasets:
        raise ValueError(
            "参数数据集数错误: "
            f"actual={len(expected_gid_values)}, "
            f"expected={expected_datasets}"
        )
    if len(normalized) != expected_runs:
        raise ValueError(
            f"SYM-F 行数错误: actual={len(normalized)}, "
            f"expected={expected_runs}"
        )

    actual_algorithms = sorted(normalized["algorithm"].unique().tolist())
    if actual_algorithms != expected_algorithms:
        raise ValueError(
            "算法集合错误: "
            f"actual={actual_algorithms}, expected={expected_algorithms}"
        )

    actual_gids = sorted(normalized["gid"].unique().tolist())
    actual_seeds = sorted(normalized["seed"].unique().tolist())
    expected_index = pd.MultiIndex.from_product(
        [
            expected_algorithms,
            expected_gid_values,
            expected_seed_values,
        ],
        names=key_columns,
    )
    actual_index = pd.MultiIndex.from_frame(normalized[key_columns])
    missing_keys = expected_index.difference(actual_index)
    extra_keys = actual_index.difference(expected_index)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            "SYM-F 运行网格不完整: "
            f"missing={len(missing_keys)}, extra={len(extra_keys)}, "
            f"actual_gids={len(actual_gids)}, "
            f"actual_seeds={actual_seeds}"
        )

    runs_per_algorithm = (
        normalized.groupby("algorithm", dropna=False)
        .size()
        .astype(int)
        .sort_index()
        .to_dict()
    )
    if expected_runs_per_algorithm is not None:
        invalid_counts = {
            algorithm: count
            for algorithm, count in runs_per_algorithm.items()
            if count != expected_runs_per_algorithm
        }
        if invalid_counts:
            raise ValueError(
                "每算法运行数错误: "
                f"actual={invalid_counts}, "
                f"expected={expected_runs_per_algorithm}"
            )

    return {
        "runs": int(len(normalized)),
        "datasets": len(actual_gids),
        "algorithms": len(actual_algorithms),
        "algorithm_keys": actual_algorithms,
        "seeds": actual_seeds,
        "runs_per_algorithm": runs_per_algorithm,
    }


def _comma_separated_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge formal SYM-F algorithm shards"
    )
    parser.add_argument("--params-csv", required=True)
    parser.add_argument(
        "--shard-csv",
        action="append",
        required=True,
        help="可重复提供；每个路径指向一个算法分片 CSV",
    )
    parser.add_argument("--expected-algorithm-keys", required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-datasets", type=int, required=True)
    parser.add_argument("--expected-seeds", default="520,521,522")
    parser.add_argument("--expected-runs-per-algorithm", type=int)
    parser.add_argument("--outdir")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    params_path = Path(args.params_csv)
    shard_paths = [Path(value) for value in args.shard_csv]
    params = pd.read_csv(params_path)
    metrics = pd.concat(
        [pd.read_csv(path) for path in shard_paths],
        ignore_index=True,
    )
    metrics = _normalize_keys(metrics).sort_values(
        ["algorithm", "gid", "seed"],
        kind="stable",
    ).reset_index(drop=True)
    summary = validate_metrics_grid(
        metrics,
        params,
        expected_algorithm_keys=_comma_separated_strings(
            args.expected_algorithm_keys
        ),
        expected_runs=args.expected_runs,
        expected_datasets=args.expected_datasets,
        expected_seeds=_comma_separated_ints(args.expected_seeds),
        expected_runs_per_algorithm=args.expected_runs_per_algorithm,
    )
    summary["params_csv"] = str(params_path.resolve())
    summary["shard_csvs"] = [
        str(path.resolve()) for path in shard_paths
    ]

    if not args.validate_only:
        if not args.outdir:
            parser.error("--outdir is required unless --validate-only is set")
        outdir = Path(args.outdir)
        generator.write_outputs(outdir, metrics, params)
        merge_summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **summary,
        }
        (outdir / "shard_merge_summary.json").write_text(
            json.dumps(merge_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["outdir"] = str(outdir.resolve())

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
