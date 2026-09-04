#!/usr/bin/env python3
"""把 75 条 JAXSR/iMCTS clean 重跑 final 合并进 2250-run 冻结全集。

输出是 ``prepare_clean_numeric`` 可直接消费的精简 final bundle、对应的
clean source CSV 和 binding manifest。分钟快照只用于验证 replacement 证据，
不会复制到 final bundle；EFF 仍由独立 trajectory overlay 处理。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .build_clean_rerun_overlay import (
    CleanRerunOverlayError,
    _atomic_write_csv,
    _atomic_write_gzip_jsonl,
    _atomic_write_text,
    _load_clean_source_rows,
    _load_jax_imcts_records,
    _logical_key,
    _metric,
    _normal_algorithm,
    _replacement_row,
    _sha256_file,
    _source_dict,
)
from .trajectories import canonical_expression


CleanFinalReplacementError = CleanRerunOverlayError
SCHEMA_VERSION = "clean_final_replacement_merge.v1"
DEFAULT_EXPECTED_COUNTS = {"jaxsr": 35, "imcts": 40}
_TASK_ID = re.compile(
    r"(?P<algorithm>[A-Za-z0-9_-]+)_s(?P<seed>\d+)_clean_g(?P<index>\d{4})$"
)
_SEED_PAIRS = ((520, 521), (520, 522), (521, 522))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanFinalReplacementError(f"{context} 无法读取: {exc}") from exc
    if not isinstance(payload, dict):
        raise CleanFinalReplacementError(f"{context} 顶层不是 JSON object")
    return payload


def _iter_gzip_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise CleanFinalReplacementError(
                        f"{path}:{line_number} 顶层不是 JSON object"
                    )
                yield line_number, payload
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanFinalReplacementError(f"无法解析 base freeze {path}: {exc}") from exc


def _base_binding_hashes(path: Path, *, expected_rows: int) -> dict[str, str]:
    payload = _read_json_object(path, context="base freeze binding")
    if payload.get("contract_ok") is not True:
        raise CleanFinalReplacementError("base freeze binding.contract_ok 不是 true")
    counts = payload.get("freeze_counts")
    if not isinstance(counts, Mapping) or int(counts.get("tasks", -1)) != expected_rows:
        raise CleanFinalReplacementError("base freeze binding task 数与 clean 网格不一致")
    inputs = payload.get("input_files")
    records = inputs.get("freeze_records") if isinstance(inputs, Mapping) else None
    if not isinstance(records, list) or not records:
        raise CleanFinalReplacementError("base freeze binding 缺少 freeze_records")
    hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CleanFinalReplacementError("base freeze_records 条目不是 object")
        name = Path(str(record.get("path", ""))).name
        digest = str(record.get("sha256", ""))
        if not name or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CleanFinalReplacementError("base freeze_records path/SHA256 非法")
        if name in hashes:
            raise CleanFinalReplacementError(f"base freeze bundle 文件名重复: {name}")
        hashes[name] = digest
    return hashes


def _validate_base_source(
    source: Mapping[str, Any], row: Mapping[str, str], *, context: str
) -> str:
    key = _logical_key(source.get("algorithm"), source.get("dataset_id"), source.get("seed"))
    if source.get("noise_tag") != "clean" or key != row["logical_key"]:
        raise CleanFinalReplacementError(f"{context} source logical identity 漂移")
    for field in ("batch", "algorithm", "dataset_id", "seed", "noise_tag", "task_id", "host", "path"):
        if str(source.get(field)) != str(row[field]):
            raise CleanFinalReplacementError(f"{context} source.{field} 与 source CSV 不一致")
    return key


def _validated_result(
    record: Mapping[str, Any], row: Mapping[str, str], *, context: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    result = record.get("result")
    if not isinstance(result, Mapping) or result.get("status") != "ok":
        raise CleanFinalReplacementError(f"{context} result 不可用")
    raw_text = result.get("raw_text")
    digest = result.get("sha256")
    if not isinstance(raw_text, str) or _sha256_text(raw_text) != digest:
        raise CleanFinalReplacementError(f"{context} result raw SHA256 校验失败")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CleanFinalReplacementError(f"{context} result raw_text 不是合法 JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "ok":
        raise CleanFinalReplacementError(f"{context} result payload status 不是 ok")
    for split, field in (("id_test", "id_nmse"), ("ood_test", "ood_nmse")):
        actual = _metric(payload, split, "nmse")
        expected = float(row[field])
        if abs(actual - expected) > 1.0e-12 + 1.0e-9 * abs(expected):
            raise CleanFinalReplacementError(f"{context} {split}.nmse 与 source CSV 不一致")
    return dict(result), payload


def _load_base_records(
    *,
    base_freeze_paths: Sequence[Path],
    expected_hashes: Mapping[str, str],
    base_by_key: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths_by_name = {path.name: path for path in base_freeze_paths}
    if len(paths_by_name) != len(base_freeze_paths):
        raise CleanFinalReplacementError("base freeze paths 文件名重复")
    if set(paths_by_name) != set(expected_hashes):
        raise CleanFinalReplacementError("base freeze paths 与 binding 文件集合不一致")
    records: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for name in sorted(paths_by_name):
        path = paths_by_name[name]
        actual_sha = _sha256_file(path)
        if actual_sha != expected_hashes[name]:
            raise CleanFinalReplacementError(f"base freeze SHA256 漂移: {name}")
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
            }
        )
        for line_number, record in _iter_gzip_jsonl(path):
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise CleanFinalReplacementError(f"{path}:{line_number} 缺少 source")
            key = _logical_key(source.get("algorithm"), source.get("dataset_id"), source.get("seed"))
            row = base_by_key.get(key)
            if row is None:
                raise CleanFinalReplacementError(f"base freeze 出现未声明 key: {key}")
            if key in records:
                raise CleanFinalReplacementError(f"base freeze key 重复: {key}")
            _validate_base_source(source, row, context=key)
            result, _ = _validated_result(record, row, context=key)
            records[key] = {"source": dict(source), "result": result}
    missing = sorted(set(base_by_key).difference(records))
    if missing:
        raise CleanFinalReplacementError(f"base freeze 缺少 {len(missing)} 个 clean key")
    return records, inputs


def _symbolic_refresh_contract(
    replacements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pred_ids: set[str] = set()
    equivalence_ids: set[str] = set()
    structure_ids: set[str] = set()
    for item in replacements:
        task_id = str(item["task_id"])
        match = _TASK_ID.fullmatch(task_id)
        if match is None:
            raise CleanFinalReplacementError(f"replacement task_id 非法: {task_id}")
        slug = _normal_algorithm(item["algorithm"])
        seed = int(item["seed"])
        index = f"g{match.group('index')}"
        pred_ids.add(f"pred_simplify::{slug}::{index}::s{seed}::clean")
        equivalence_ids.add(f"equivalence::{slug}::{index}::s{seed}::clean")
        for left, right in _SEED_PAIRS:
            if seed in {left, right}:
                structure_ids.add(f"stab_structure::{slug}::{index}::s{left}-s{right}")
    return {
        "stale_symbolic_run_count": len(pred_ids),
        "required_pred_simplify_logical_ids": sorted(pred_ids),
        "required_equivalence_logical_ids": sorted(equivalence_ids),
        "required_structure_logical_ids": sorted(structure_ids),
        "required_structure_count": len(structure_ids),
    }


def build_clean_final_replacement_merge(
    *,
    base_source_runs_csv: Path,
    base_freeze_binding_json: Path,
    base_freeze_paths: Sequence[Path],
    replacement_freeze_dir: Path,
    output_bundle: Path,
    output_composite_csv: Path,
    output_binding_manifest: Path,
    horizon: int = 180,
    expected_clean_rows: int = 2250,
    expected_replacement_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """构建 final replacement 合并层，并显式标记 symbolic refresh 阻塞。"""

    counts_expected = {
        _normal_algorithm(key): int(value)
        for key, value in (expected_replacement_counts or DEFAULT_EXPECTED_COUNTS).items()
    }
    fieldnames, clean_rows, base_by_key = _load_clean_source_rows(
        base_source_runs_csv, expected_clean_rows=expected_clean_rows
    )
    expected_hashes = _base_binding_hashes(
        base_freeze_binding_json, expected_rows=expected_clean_rows
    )
    merged_by_key, base_bundle_inputs = _load_base_records(
        base_freeze_paths=base_freeze_paths,
        expected_hashes=expected_hashes,
        base_by_key=base_by_key,
    )
    replacement_inputs: list[dict[str, Any]] = []
    loaded = _load_jax_imcts_records(
        replacement_freeze_dir,
        horizon=horizon,
        input_files=replacement_inputs,
    )
    base_by_normal_key = {
        (_normal_algorithm(row["algorithm"]), row["dataset_id"], int(row["seed"])): row
        for row in base_by_key.values()
    }
    composite_by_key = {key: dict(row) for key, row in base_by_key.items()}
    details: list[dict[str, Any]] = []
    actual_counts: Counter[str] = Counter()
    seen_replacements: set[str] = set()
    for record, payload, origin in loaded:
        source = record["source"]
        normal_key = (
            _normal_algorithm(source["algorithm"]),
            str(source["dataset_id"]),
            int(source["seed"]),
        )
        base_row = base_by_normal_key.get(normal_key)
        if base_row is None:
            raise CleanFinalReplacementError(f"replacement 不属于 base clean 网格: {normal_key}")
        key = base_row["logical_key"]
        if key in seen_replacements:
            raise CleanFinalReplacementError(f"replacement logical_key 重复: {key}")
        if str(source["task_id"]) != base_row["task_id"]:
            raise CleanFinalReplacementError(f"{key} replacement task_id 与 base 不一致")
        normalized_source = _source_dict(
            batch=str(source["batch"]),
            algorithm=base_row["algorithm"],
            dataset=base_row["dataset_id"],
            host=str(source["host"]),
            seed=int(base_row["seed"]),
            task_id=base_row["task_id"],
            result_path=str(source["path"]),
        )
        old_result = merged_by_key[key]["result"]
        old_result_sha = str(old_result["sha256"])
        old_payload = json.loads(str(old_result["raw_text"]))
        old_expression = canonical_expression(old_payload)
        replacement_expression = canonical_expression(payload)
        new_record = {
            "source": normalized_source,
            "result": dict(record["result"]),
            "replacement": {
                "scope": "final_and_eff",
                "origin": origin,
                "old_result_sha256": old_result_sha,
            },
        }
        new_row = _replacement_row(base_row, new_record, payload)
        merged_by_key[key] = new_record
        composite_by_key[key] = new_row
        seen_replacements.add(key)
        actual_counts[normal_key[0]] += 1
        details.append(
            {
                "logical_key": key,
                "algorithm": base_row["algorithm"],
                "dataset_id": base_row["dataset_id"],
                "seed": int(base_row["seed"]),
                "task_id": base_row["task_id"],
                "replacement_scope": "final_and_eff",
                "old_result_sha256": old_result_sha,
                "replacement_result_sha256": record["result"]["sha256"],
                "result_binding_changed": old_result_sha != record["result"]["sha256"],
                "old_expression_sha256": _sha256_text(old_expression or ""),
                "replacement_expression_sha256": _sha256_text(replacement_expression or ""),
                "expression_text_changed": old_expression != replacement_expression,
                "checkpoint_count": horizon,
                "checkpoint_identity_verified": True,
            }
        )
    if dict(sorted(actual_counts.items())) != dict(sorted(counts_expected.items())):
        raise CleanFinalReplacementError(
            f"replacement 算法计数不符: actual={dict(actual_counts)}, expected={counts_expected}"
        )
    if len(seen_replacements) != sum(counts_expected.values()):
        raise CleanFinalReplacementError("replacement 唯一 key 数与期望不一致")
    if len(merged_by_key) != expected_clean_rows or set(merged_by_key) != set(base_by_key):
        raise CleanFinalReplacementError("合并后 final bundle 未闭合 2250 clean 网格")

    merged_rows = [merged_by_key[key] for key in sorted(merged_by_key)]
    composite_rows = [composite_by_key[key] for key in sorted(composite_by_key)]
    _atomic_write_gzip_jsonl(output_bundle, merged_rows)
    _atomic_write_csv(output_composite_csv, fieldnames, composite_rows)

    symbolic = _symbolic_refresh_contract(details)
    result_binding_changed_count = sum(
        bool(item["result_binding_changed"]) for item in details
    )
    expression_text_changed_count = sum(
        bool(item["expression_text_changed"]) for item in details
    )
    output_bundle_info = {
        "path": str(output_bundle.resolve()),
        "sha256": _sha256_file(output_bundle),
        "size_bytes": output_bundle.stat().st_size,
        "rows": len(merged_rows),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "contract_ok": True,
        "scope": "clean_final_and_numeric_replay_input",
        "freeze_counts": {"tasks": len(merged_rows)},
        "replacement_contract": {
            "replacement_count": len(details),
            "replacement_keys": sorted(seen_replacements),
            "algorithm_counts": dict(sorted(actual_counts.items())),
            "expected_algorithm_counts": dict(sorted(counts_expected.items())),
            "result_binding_changed_count": result_binding_changed_count,
            "expression_text_changed_count": expression_text_changed_count,
            "details": sorted(details, key=lambda item: str(item["logical_key"])),
        },
        "evaluation_policy": {
            "formal_numeric_path": "canonical_replay.v1",
            "native_metrics_in_composite_source_csv": "binding_check_only",
            "native_metrics_are_not_formal_scores": True,
            "eff_input": "separate_clean_trajectory_overlay",
        },
        "aggregation_entry": {
            "numeric_prepare_module": "AAAI_experiments.stage5_metric_calculation_0831.pipeline.prepare_clean_numeric",
            "numeric_prepare_inputs": {
                "source_runs_csv": str(output_composite_csv.resolve()),
                "freeze_bundle": str(output_bundle.resolve()),
                "freeze_binding_json": str(output_binding_manifest.resolve()),
            },
            "aggregate_clean_metrics_blocked_until_symbolic_refresh": True,
            "required_order": [
                "prepare_clean_numeric_from_merged_bundle",
                "refresh_75_pred_simplify_and_equivalence_results",
                "refresh_affected_structure_pairs",
                "aggregate_clean_metrics",
            ],
        },
        "aggregation_readiness": {
            "numeric_canonical_replay_ready": True,
            "formal_six_axis_ready": False,
            "blocker": "75 replacement finals invalidate the prior pred simplify, equivalence, and affected STAB structure bindings",
            **symbolic,
        },
        "inputs": {
            "base_source_runs_csv": {
                "path": str(base_source_runs_csv.resolve()),
                "sha256": _sha256_file(base_source_runs_csv),
                "clean_rows": len(clean_rows),
            },
            "base_freeze_binding": {
                "path": str(base_freeze_binding_json.resolve()),
                "sha256": _sha256_file(base_freeze_binding_json),
            },
            "base_freeze_records": base_bundle_inputs,
            "replacement_evidence": sorted(
                replacement_inputs, key=lambda item: (str(item["role"]), str(item["path"]))
            ),
        },
        "input_files": {
            "freeze_records": [output_bundle_info],
        },
        "outputs": {
            "final_bundle": output_bundle_info,
            "composite_source_runs_csv": {
                "path": str(output_composite_csv.resolve()),
                "sha256": _sha256_file(output_composite_csv),
                "size_bytes": output_composite_csv.stat().st_size,
                "rows": len(composite_rows),
                "replaced_rows": len(details),
            },
        },
    }
    _atomic_write_text(
        output_binding_manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    stage_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-source-runs-csv",
        type=Path,
        default=stage_root / "manifests/source_runs.csv",
    )
    parser.add_argument(
        "--base-freeze-binding-json",
        type=Path,
        default=stage_root / "reports/freeze_binding.json",
    )
    parser.add_argument(
        "--base-freeze-glob",
        default=str(stage_root / "source_snapshot/trajectory_freeze/clean_freeze_*.jsonl.gz"),
    )
    parser.add_argument(
        "--replacement-freeze-dir",
        type=Path,
        default=stage_root / "work/clean_rerun_trajectory_freeze_v1/collected",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1",
    )
    args = parser.parse_args(argv)
    import glob

    output_root = args.output_root.resolve()
    manifest = build_clean_final_replacement_merge(
        base_source_runs_csv=args.base_source_runs_csv.resolve(),
        base_freeze_binding_json=args.base_freeze_binding_json.resolve(),
        base_freeze_paths=[Path(path).resolve() for path in sorted(glob.glob(args.base_freeze_glob))],
        replacement_freeze_dir=args.replacement_freeze_dir.resolve(),
        output_bundle=output_root / "clean_final_merged.jsonl.gz",
        output_composite_csv=output_root / "source_runs_clean_composite.csv",
        output_binding_manifest=output_root / "binding_manifest.json",
    )
    print(
        json.dumps(
            {
                "clean_keys": manifest["freeze_counts"]["tasks"],
                "replacements": manifest["replacement_contract"]["replacement_count"],
                "formal_six_axis_ready": manifest["aggregation_readiness"]["formal_six_axis_ready"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
