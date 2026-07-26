#!/usr/bin/env python3
"""严格校验并合并正式 SYM-F 算法分片。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
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
BOOLEAN_COLUMNS = (
    "valid_for_symbolic",
    "pred_parse_ok",
    "cas_equiv",
    "numeric_equiv",
    "equiv_final",
)
SCORE_COLUMNS = (
    "tree_similarity",
    "var_f1",
    "op_f1",
    "sym_f_formal",
)
PROVENANCE_FILENAME = "symbolic_metrics_formal_provenance.json"
MERGED_OUTPUT_FILENAMES = (
    "README.md",
    "symbolic_metrics_formal.csv",
    "symbolic_metrics_formal_algorithm_summary.csv",
    "symbolic_metrics_formal_dataset_summary.csv",
    "symbolic_metrics_formal_summary.json",
    PROVENANCE_FILENAME,
)


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized[["algorithm", "gid", "seed"]].isna().any(axis=None):
        raise ValueError("SYM-F 指标存在空 algorithm、gid 或 seed")
    normalized["algorithm"] = normalized["algorithm"].map(
        generator.normalize_algorithm
    )
    normalized["gid"] = normalized["gid"].astype(str).str.strip()
    numeric_seed = pd.to_numeric(
        normalized["seed"],
        errors="raise",
    )
    seed_values = numeric_seed.to_numpy(dtype=float)
    if (
        not np.isfinite(seed_values).all()
        or not np.equal(seed_values, np.floor(seed_values)).all()
    ):
        raise ValueError("SYM-F seed 必须是有限整数")
    normalized["seed"] = numeric_seed.astype(int)
    return normalized


def _normalize_boolean(value: Any, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{column} 必须是布尔值，实际为 {value!r}")


def _validate_metric_semantics(
    metrics: pd.DataFrame,
    params: pd.DataFrame,
) -> pd.DataFrame:
    normalized = metrics.copy()
    for column in BOOLEAN_COLUMNS:
        normalized[column] = normalized[column].map(
            lambda value, name=column: _normalize_boolean(
                value,
                column=name,
            )
        )

    for column in SCORE_COLUMNS:
        numeric = pd.to_numeric(normalized[column], errors="raise")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} 必须全部为有限值")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} 超出 [0, 1] 范围")
        normalized[column] = numeric.astype(float)

    expected_equiv = normalized["valid_for_symbolic"] & (
        normalized["cas_equiv"] | normalized["numeric_equiv"]
    )
    inconsistent_equiv = normalized["equiv_final"] != expected_equiv
    if inconsistent_equiv.any():
        raise ValueError(
            "equiv_final 与 valid_for_symbolic、cas_equiv、"
            "numeric_equiv 不一致"
        )
    non_unit_exact = normalized["equiv_final"] & ~np.isclose(
        normalized["sym_f_formal"],
        1.0,
    )
    if non_unit_exact.any():
        raise ValueError("equiv_final=true 时 sym_f_formal 必须为 1")
    invalid_nonzero = (
        ~normalized["valid_for_symbolic"]
        | ~normalized["pred_parse_ok"]
    ) & ~np.isclose(normalized["sym_f_formal"], 0.0)
    if invalid_nonzero.any():
        raise ValueError(
            "无效或不可解析结果的 sym_f_formal 必须为 0"
        )
    expected_score = np.where(
        normalized["equiv_final"],
        1.0,
        np.where(
            normalized["valid_for_symbolic"]
            & normalized["pred_parse_ok"],
            0.3 * normalized["tree_similarity"]
            + 0.1 * normalized["var_f1"]
            + 0.1 * normalized["op_f1"],
            0.0,
        ),
    )
    if not np.isclose(
        normalized["sym_f_formal"].to_numpy(dtype=float),
        expected_score,
        rtol=1e-12,
        atol=1e-12,
    ).all():
        raise ValueError("sym_f_formal 不符合正式评分公式")

    if "dataset" not in params.columns:
        raise ValueError("参数 CSV 缺少 dataset 字段")
    if params["dataset"].isna().any() or normalized["dataset"].isna().any():
        raise ValueError("参数或 SYM-F 指标存在空 dataset")
    param_dataset = {
        str(row["gid"]).strip(): str(row["dataset"]).strip()
        for _, row in params.iterrows()
    }
    actual_dataset = normalized["dataset"].astype(str).str.strip()
    expected_dataset = normalized["gid"].map(param_dataset)
    mismatch = expected_dataset.isna() | (
        actual_dataset != expected_dataset
    )
    if mismatch.any():
        sample = normalized.loc[
            mismatch,
            ["algorithm", "gid", "dataset", "seed"],
        ].head(5)
        raise ValueError(
            "SYM-F gid 与 dataset 映射不一致: "
            f"{sample.to_dict(orient='records')}"
        )
    normalized["dataset"] = actual_dataset
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

    if params["gid"].isna().any():
        raise ValueError("参数 CSV 存在空 gid")
    params_gids = params["gid"].astype(str).str.strip()
    if params_gids.eq("").any():
        raise ValueError("参数 CSV 存在空 gid")
    duplicate_param_gids = int(params_gids.duplicated(keep=False).sum())
    if duplicate_param_gids:
        raise ValueError(
            f"参数 CSV 存在重复 gid: {duplicate_param_gids}"
        )
    normalized = _validate_metric_semantics(normalized, params)

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


def build_source_fingerprint(
    *,
    params_csv: Path,
    run_level_csv: Path,
    generator_script: Path,
    expression_source: str,
) -> dict[str, str]:
    return generator.build_source_fingerprint(
        params_csv=params_csv,
        run_level_csv=run_level_csv,
        generator_script=generator_script,
        expression_source=expression_source,
    )


def validate_shard_provenance(
    shard_paths: Sequence[Path],
    *,
    expected_source_fingerprint: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        provenance_path = shard_path.parent / PROVENANCE_FILENAME
        if not provenance_path.is_file():
            raise ValueError(
                f"shard provenance 缺失: {provenance_path}"
            )
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        if provenance.get("schema_version") != (
            generator.PROVENANCE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"shard provenance schema 错误: {provenance_path}"
            )
        if provenance.get("source_fingerprint") != (
            expected_source_fingerprint
        ):
            raise ValueError(
                "shard provenance 与当前输入不一致: "
                f"{provenance_path}"
            )
        metrics_sha256 = generator.file_sha256(shard_path)
        if provenance.get("metrics_sha256") != metrics_sha256:
            raise ValueError(
                f"shard provenance 指标哈希错误: {shard_path}"
            )
        records.append(provenance)
    return records


def _input_shard_records(
    shard_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    records = []
    for shard_path in shard_paths:
        provenance_path = shard_path.parent / PROVENANCE_FILENAME
        logical_key = shard_path.stem
        if provenance_path.is_file():
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            algorithm_keys = provenance.get("algorithm_keys")
            if isinstance(algorithm_keys, list) and algorithm_keys:
                logical_key = ",".join(
                    sorted(str(value) for value in algorithm_keys)
                )
        records.append(
            {
                "logical_key": logical_key,
                "filename": shard_path.name,
                "sha256": generator.file_sha256(shard_path),
                "provenance_filename": (
                    provenance_path.name
                    if provenance_path.is_file()
                    else None
                ),
                "provenance_sha256": (
                    generator.file_sha256(provenance_path)
                    if provenance_path.is_file()
                    else None
                ),
            }
        )
    return records


def _output_hashes(outdir: Path) -> dict[str, str]:
    output_hashes = {}
    for filename in MERGED_OUTPUT_FILENAMES:
        path = outdir / filename
        if not path.is_file():
            raise ValueError(f"合并输出缺失: {path}")
        output_hashes[filename] = generator.file_sha256(path)
    return output_hashes


def write_merge_manifest(
    outdir: Path,
    *,
    validation_summary: dict[str, Any],
    shard_paths: Sequence[Path],
    source_fingerprint: dict[str, str] | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merge_script_sha256": generator.file_sha256(
            Path(__file__).resolve()
        ),
        "source_fingerprint": source_fingerprint,
        "validation": validation_summary,
        "input_shards": _input_shard_records(shard_paths),
        "output_hashes": _output_hashes(outdir),
        **validation_summary,
    }
    (outdir / "shard_merge_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def verify_merge_output(
    outdir: Path,
    *,
    shard_paths: Sequence[Path],
    source_fingerprint: dict[str, str] | None,
) -> dict[str, Any]:
    manifest_path = outdir / "shard_merge_summary.json"
    if not manifest_path.is_file():
        raise ValueError(f"合并 manifest 缺失: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("合并 manifest schema 错误")
    current_script_sha256 = generator.file_sha256(
        Path(__file__).resolve()
    )
    if manifest.get("merge_script_sha256") != current_script_sha256:
        raise ValueError("合并 manifest 与当前校验器版本不一致")
    if manifest.get("source_fingerprint") != source_fingerprint:
        raise ValueError("合并 manifest 与当前 source provenance 不一致")
    if manifest.get("input_shards") != _input_shard_records(shard_paths):
        raise ValueError("合并 manifest 与当前分片哈希不一致")

    expected_output_hashes = manifest.get("output_hashes")
    actual_output_hashes = _output_hashes(outdir)
    if expected_output_hashes != actual_output_hashes:
        raise ValueError(
            "合并派生输出哈希不一致: "
            f"actual={actual_output_hashes}, "
            f"expected={expected_output_hashes}"
        )
    return manifest


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
    parser.add_argument("--source-run-level-csv")
    parser.add_argument(
        "--generator-script",
        default=str(Path(generator.__file__).resolve()),
    )
    parser.add_argument("--expected-expression-source")
    parser.add_argument("--outdir")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-output-dir")
    args = parser.parse_args()

    params_path = Path(args.params_csv)
    shard_paths = [Path(value) for value in args.shard_csv]
    params = pd.read_csv(params_path)
    shard_frames = [pd.read_csv(path) for path in shard_paths]
    metrics = pd.concat(shard_frames, ignore_index=True)
    metrics = _normalize_keys(metrics).sort_values(
        ["algorithm", "gid", "seed"],
        kind="stable",
    ).reset_index(drop=True)
    source_fingerprint = None
    if args.source_run_level_csv or args.expected_expression_source:
        if not (
            args.source_run_level_csv
            and args.expected_expression_source
        ):
            parser.error(
                "--source-run-level-csv and "
                "--expected-expression-source must be used together"
            )
        source_fingerprint = build_source_fingerprint(
            params_csv=params_path,
            run_level_csv=Path(args.source_run_level_csv),
            generator_script=Path(args.generator_script),
            expression_source=args.expected_expression_source,
        )
        provenance_records = validate_shard_provenance(
            shard_paths,
            expected_source_fingerprint=source_fingerprint,
        )
        for shard_path, shard_frame, provenance in zip(
            shard_paths,
            shard_frames,
            provenance_records,
            strict=True,
        ):
            shard_algorithms = sorted(
                shard_frame["algorithm"]
                .map(generator.normalize_algorithm)
                .unique()
                .tolist()
            )
            if provenance.get("algorithm_keys") != shard_algorithms:
                raise ValueError(
                    "shard provenance 算法集合错误: "
                    f"{shard_path}"
                )
            if provenance.get("runs") != len(shard_frame):
                raise ValueError(
                    f"shard provenance 行数错误: {shard_path}"
                )
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
    metrics = _validate_metric_semantics(metrics, params)
    input_shards = _input_shard_records(shard_paths)
    summary["params_csv"] = params_path.name
    summary["params_sha256"] = generator.file_sha256(params_path)
    summary["shard_csvs"] = [
        f"{record['logical_key']}/{record['filename']}"
        for record in input_shards
    ]

    if not args.validate_only:
        if not args.outdir:
            parser.error("--outdir is required unless --validate-only is set")
        outdir = Path(args.outdir)
        merge_provenance = {
            "schema_version": generator.PROVENANCE_SCHEMA_VERSION,
            "kind": "merged_formal_metrics",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "algorithm_keys": summary["algorithm_keys"],
            "runs": summary["runs"],
            "source_fingerprint": source_fingerprint,
            "input_shards": _input_shard_records(shard_paths),
        }
        generator.write_outputs(
            outdir,
            metrics,
            params,
            provenance=merge_provenance,
        )
        write_merge_manifest(
            outdir,
            validation_summary=summary,
            shard_paths=shard_paths,
            source_fingerprint=source_fingerprint,
        )
        summary["outdir"] = str(outdir.resolve())

    if args.verify_output_dir:
        verified = verify_merge_output(
            Path(args.verify_output_dir),
            shard_paths=shard_paths,
            source_fingerprint=source_fingerprint,
        )
        summary["verified_output_dir"] = str(
            Path(args.verify_output_dir).resolve()
        )
        summary["verified_output_hashes"] = verified["output_hashes"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
