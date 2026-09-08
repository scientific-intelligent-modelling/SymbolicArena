"""把 Stage5 noise base v6 与 targeted rerun 合并为当前 canonical 数值表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import export_result_summary as result_summary
from . import prepare_noise_numeric as noise_numeric
from .metrics import phi_nmse
from .performance_replay import (
    EVALUATION_PATH,
    PerformanceReplayCache,
    PerformanceReplayError,
    replay_payload_performance,
)


class CurrentNoiseNumericExportError(ValueError):
    """当前 noise 数值表无法满足严格来源与完整网格契约。"""


def _raise(message: str) -> None:
    raise CurrentNoiseNumericExportError(message)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(f"{context} 必须是非空字符串")
    return value.strip()


def _finite(value: object, *, context: str, lower: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        _raise(f"{context} 缺少数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CurrentNoiseNumericExportError(f"{context} 不是合法数值: {value!r}") from exc
    if not math.isfinite(number) or number < lower:
        _raise(f"{context} 必须是大于等于 {lower} 的有限数值")
    return number


def _reject_forbidden_source(source: Mapping[str, Any], *, context: str) -> None:
    """对 source 整体扫描，避免路径字段换名后绕过禁止来源检查。"""

    for field, value in source.items():
        if isinstance(value, str):
            try:
                result_summary._check_source_text(value, context=f"{context}.{field}")
            except result_summary.ResultSummaryError as exc:
                raise CurrentNoiseNumericExportError(str(exc)) from exc


def _read_base_rows(path: Path, *, condition: str) -> dict[str, dict[str, str]]:
    try:
        rows = result_summary._read_csv(path)
    except result_summary.ResultSummaryError as exc:
        raise CurrentNoiseNumericExportError(str(exc)) from exc
    required = set(noise_numeric.NUMERIC_FIELDS)
    missing = sorted(required - set(rows[0]))
    if missing:
        _raise(f"base numeric CSV 缺少字段: {missing}")

    indexed: dict[str, dict[str, str]] = {}
    for line_number, raw in enumerate(rows, start=2):
        _reject_forbidden_source(raw, context=f"base numeric CSV:{line_number}")
        algorithm = _nonempty(raw.get("algorithm"), context=f"base:{line_number}.algorithm")
        dataset_id = _nonempty(raw.get("dataset_id"), context=f"base:{line_number}.dataset_id")
        try:
            seed = int(raw.get("seed", ""))
        except ValueError as exc:
            raise CurrentNoiseNumericExportError(
                f"base:{line_number}.seed 不是合法整数"
            ) from exc
        if seed not in noise_numeric.SEEDS:
            _raise(f"base:{line_number}.seed 不在 {noise_numeric.SEEDS} 中")
        if raw.get("noise_tag") != condition:
            _raise(f"base:{line_number}.noise_tag 不是 {condition}")
        key = f"{algorithm}::{dataset_id}::s{seed}::{condition}"
        if raw.get("logical_key") != key:
            _raise(f"base:{line_number}.logical_key 不符合 canonical 格式")
        if key in indexed:
            _raise(f"base numeric CSV 出现重复 logical_key: {key}")
        indexed[key] = {field: str(raw.get(field, "")) for field in noise_numeric.NUMERIC_FIELDS}
    return indexed


def _replay_overlay_record(
    record: Mapping[str, Any],
    *,
    condition: str,
    repo_root: Path,
    cache: PerformanceReplayCache,
) -> dict[str, str]:
    source = record.get("source")
    if not isinstance(source, Mapping):
        _raise("targeted overlay 记录缺少 source")
    _reject_forbidden_source(source, context="targeted overlay.source")
    algorithm = _nonempty(source.get("algorithm"), context="overlay.source.algorithm")
    dataset_id = _nonempty(source.get("dataset_id"), context="overlay.source.dataset_id")
    task_id = _nonempty(source.get("task_id"), context="overlay.source.task_id")
    host = _nonempty(source.get("host"), context="overlay.source.host")
    try:
        seed = int(source.get("seed"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CurrentNoiseNumericExportError("overlay.source.seed 不是合法整数") from exc
    if seed not in noise_numeric.SEEDS:
        _raise(f"overlay.source.seed 不在 {noise_numeric.SEEDS} 中")
    if source.get("noise_tag") != condition:
        _raise(f"overlay.source.noise_tag 不是 {condition}")
    key = f"{algorithm}::{dataset_id}::s{seed}::{condition}"

    result = record.get("result")
    if not isinstance(result, Mapping):
        _raise(f"{key} targeted overlay 缺少 result")
    raw_text = result.get("raw_text")
    result_sha = str(result.get("sha256", "")).lower()
    if not isinstance(raw_text, str) or not _is_sha256(result_sha):
        _raise(f"{key} targeted overlay 缺少合法 result raw_text/SHA-256")
    if _sha256_text(raw_text) != result_sha:
        _raise(f"{key} targeted overlay result raw_text SHA-256 校验失败")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CurrentNoiseNumericExportError(f"{key} result.raw_text 不是合法 JSON") from exc
    if not isinstance(payload, Mapping):
        _raise(f"{key} result.raw_text 顶层必须是 object")
    try:
        native_id, native_ood, formula = noise_numeric._validate_noise_payload(
            payload, condition=condition, key=key
        )
    except noise_numeric.NoiseNumericPreparationError as exc:
        raise CurrentNoiseNumericExportError(str(exc)) from exc
    if not formula:
        _raise(f"{key} targeted overlay 缺少可重放公式")

    try:
        replay = replay_payload_performance(
            payload,
            algorithm=algorithm,
            repo_root=repo_root,
            cache=cache,
            task_id=task_id,
            condition=condition,
            result_sha256=result_sha,
        )
    except PerformanceReplayError as exc:
        raise CurrentNoiseNumericExportError(f"{key} canonical replay 失败: {exc}") from exc
    if replay.get("error"):
        _raise(f"{key} canonical replay 返回错误: {replay['error']}")
    if replay.get("evaluation_path") != EVALUATION_PATH:
        _raise(f"{key} canonical replay evaluation_path 不匹配")
    artifact_sha = str(replay.get("canonical_artifact_sha256", "")).lower()
    if not _is_sha256(artifact_sha):
        _raise(f"{key} canonical replay 缺少合法 artifact SHA-256")
    rebuilt = replay.get("artifact_rebuilt")
    valid = replay.get("valid_output")
    if not isinstance(rebuilt, bool) or not isinstance(valid, bool):
        _raise(f"{key} canonical replay 的 artifact_rebuilt/valid_output 不是 bool")

    invalid_reason = str(replay.get("invalid_reason") or "").strip()
    replay_id: float | None = None
    replay_ood: float | None = None
    if valid:
        id_block = replay.get("id_test")
        ood_block = replay.get("ood_test")
        if not isinstance(id_block, Mapping) or not isinstance(ood_block, Mapping):
            _raise(f"{key} canonical replay 缺少 ID/OOD 指标")
        replay_id = _finite(id_block.get("nmse"), context=f"{key}.id_nmse")
        replay_ood = _finite(ood_block.get("nmse"), context=f"{key}.ood_nmse")
        if invalid_reason:
            _raise(f"{key} valid replay 不应含 invalid_reason")
    elif not invalid_reason:
        _raise(f"{key} invalid replay 缺少 invalid_reason")

    id_quality = _finite(replay.get("id_quality"), context=f"{key}.id_quality")
    ood_quality = _finite(replay.get("ood_quality"), context=f"{key}.ood_quality")
    if id_quality > 1.0 or ood_quality > 1.0:
        _raise(f"{key} canonical quality 超出 [0, 1]")
    if not valid and (id_quality != 0.0 or ood_quality != 0.0):
        _raise(f"{key} invalid replay 的 quality 必须为 0")

    return {
        "logical_key": key,
        "algorithm": algorithm,
        "dataset_id": dataset_id,
        "seed": str(seed),
        "noise_tag": condition,
        "task_id": task_id,
        "host": host,
        "result_sha256": result_sha,
        "evaluation_status": "valid" if valid else "invalid_output",
        "valid_output": str(valid).lower(),
        "invalid_reason": invalid_reason,
        "formula_source": (
            "canonical_artifact"
            if isinstance(payload.get("canonical_artifact"), Mapping)
            else "equation"
        ),
        "evaluation_path": EVALUATION_PATH,
        "canonical_artifact_sha256": artifact_sha,
        "artifact_rebuilt": str(rebuilt).lower(),
        "native_id_nmse": noise_numeric._format(native_id),
        "native_ood_nmse": noise_numeric._format(native_ood),
        "id_nmse": noise_numeric._format(replay_id),
        "ood_nmse": noise_numeric._format(replay_ood),
        "id_quality": noise_numeric._format(id_quality),
        "ood_quality": noise_numeric._format(ood_quality),
        "id_nmse_delta_from_native": (
            noise_numeric._format(replay_id - native_id) if replay_id is not None else ""
        ),
        "ood_nmse_delta_from_native": (
            noise_numeric._format(replay_ood - native_ood) if replay_ood is not None else ""
        ),
        "id_quality_delta_from_native": noise_numeric._format(
            id_quality - phi_nmse(native_id)
        ),
        "ood_quality_delta_from_native": noise_numeric._format(
            ood_quality - phi_nmse(native_ood)
        ),
        "replay_error": "",
    }


def _validate_complete_grid(
    rows: Mapping[str, Mapping[str, str]],
    *,
    condition: str,
    expected_runs: int,
    expected_algorithms: int,
    expected_datasets: int,
) -> None:
    if len(rows) != expected_runs:
        _raise(f"{condition} 当前 numeric 应为 {expected_runs} 行，实际为 {len(rows)}")
    algorithms = {row["algorithm"] for row in rows.values()}
    datasets = {row["dataset_id"] for row in rows.values()}
    if len(algorithms) != expected_algorithms:
        _raise(f"{condition} 算法数应为 {expected_algorithms}，实际为 {len(algorithms)}")
    if len(datasets) != expected_datasets:
        _raise(f"{condition} 数据集数应为 {expected_datasets}，实际为 {len(datasets)}")
    expected_keys = {
        f"{algorithm}::{dataset}::s{seed}::{condition}"
        for algorithm in algorithms
        for dataset in datasets
        for seed in noise_numeric.SEEDS
    }
    if set(rows) != expected_keys:
        _raise(
            f"{condition} algorithm/dataset/seed 网格不完整: "
            f"missing={len(expected_keys - set(rows))}, unexpected={len(set(rows) - expected_keys)}"
        )
    for key, row in rows.items():
        if row.get("evaluation_path") != EVALUATION_PATH:
            _raise(f"{key} evaluation_path 非 {EVALUATION_PATH}")
        if row.get("evaluation_status") not in {"valid", "invalid_output"}:
            _raise(f"{key} canonical replay 尚未闭合: {row.get('evaluation_status')!r}")
        valid_text = row.get("valid_output")
        if valid_text not in {"true", "false"}:
            _raise(f"{key} valid_output 不是明确布尔值")
        _nonempty(row.get("task_id"), context=f"{key}.task_id")
        if not _is_sha256(row.get("result_sha256")):
            _raise(f"{key}.result_sha256 不是 SHA-256")
        id_quality = _finite(row.get("id_quality"), context=f"{key}.id_quality")
        ood_quality = _finite(row.get("ood_quality"), context=f"{key}.ood_quality")
        if id_quality > 1.0 or ood_quality > 1.0:
            _raise(f"{key} quality 超出 [0, 1]")
        if valid_text == "true":
            _finite(row.get("id_nmse"), context=f"{key}.id_nmse")
            _finite(row.get("ood_nmse"), context=f"{key}.ood_nmse")
        elif id_quality != 0.0 or ood_quality != 0.0:
            _raise(f"{key} invalid_output 的 quality 必须为 0")


def export_current_noise_numeric_runs(
    *,
    base_numeric_csv: Path,
    overlay_jsonl_paths: Sequence[Path],
    condition: str,
    output_csv: Path,
    repo_root: Path,
    expected_runs: int = noise_numeric.DEFAULT_EXPECTED_RUNS,
    expected_algorithms: int = noise_numeric.DEFAULT_EXPECTED_ALGORITHMS,
    expected_datasets: int = noise_numeric.DEFAULT_EXPECTED_DATASETS,
    expected_overlay_runs: int | None = None,
) -> dict[str, Any]:
    """导出一个 noise condition 的完整当前数值表。"""

    if condition not in noise_numeric.NOISE_CONDITIONS:
        _raise(f"未知 noise condition: {condition!r}")
    if not overlay_jsonl_paths:
        _raise("至少需要一个 targeted overlay JSONL")
    base = _read_base_rows(base_numeric_csv.resolve(), condition=condition)
    try:
        overlay_all = result_summary._load_record_map(path.resolve() for path in overlay_jsonl_paths)
    except result_summary.ResultSummaryError as exc:
        raise CurrentNoiseNumericExportError(str(exc)) from exc
    overlay = {
        key: record for key, record in overlay_all.items() if key.endswith(f"::{condition}")
    }
    if expected_overlay_runs is not None and len(overlay) != expected_overlay_runs:
        _raise(
            f"{condition} targeted overlay 应为 {expected_overlay_runs} 条，实际为 {len(overlay)}"
        )
    unexpected = set(overlay) - set(base)
    if unexpected:
        _raise(f"{condition} targeted overlay 含 base 外 logical_key: {sorted(unexpected)[:5]}")

    merged = dict(base)
    cache = PerformanceReplayCache()
    for key in sorted(overlay):
        row = _replay_overlay_record(
            overlay[key], condition=condition, repo_root=repo_root.resolve(), cache=cache
        )
        if row["logical_key"] != key:
            _raise(f"{key} targeted overlay replay 后 logical_key 漂移")
        merged[key] = row
    _validate_complete_grid(
        merged,
        condition=condition,
        expected_runs=expected_runs,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
    )
    ordered = sorted(
        merged.values(),
        key=lambda row: (row["algorithm"], row["dataset_id"], int(row["seed"])),
    )
    noise_numeric._atomic_write_csv(output_csv.resolve(), ordered)
    return {
        "schema_version": "stage5.current_noise_numeric_runs.v1",
        "status": "ok",
        "condition": condition,
        "row_count": len(ordered),
        "overlay_run_count": len(overlay),
        "base_run_count": len(ordered) - len(overlay),
        "replay_unavailable_count": 0,
        "evaluation_path": EVALUATION_PATH,
        "forbidden_source_tokens": list(result_summary.FORBIDDEN_SOURCE_TOKENS),
        "output_csv": str(output_csv.resolve()),
        "output_sha256": result_summary._sha256_file(output_csv.resolve()),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 noise base v6 与 targeted rerun，导出完整 canonical numeric run CSV"
    )
    parser.add_argument("--condition", choices=noise_numeric.NOISE_CONDITIONS, required=True)
    parser.add_argument("--base-numeric-csv", type=Path, required=True)
    parser.add_argument("--overlay-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--expected-runs", type=int, default=noise_numeric.DEFAULT_EXPECTED_RUNS)
    parser.add_argument(
        "--expected-algorithms", type=int, default=noise_numeric.DEFAULT_EXPECTED_ALGORITHMS
    )
    parser.add_argument(
        "--expected-datasets", type=int, default=noise_numeric.DEFAULT_EXPECTED_DATASETS
    )
    parser.add_argument("--expected-overlay-runs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = export_current_noise_numeric_runs(
            base_numeric_csv=args.base_numeric_csv,
            overlay_jsonl_paths=args.overlay_jsonl,
            condition=args.condition,
            output_csv=args.output_csv,
            repo_root=args.repo_root,
            expected_runs=args.expected_runs,
            expected_algorithms=args.expected_algorithms,
            expected_datasets=args.expected_datasets,
            expected_overlay_runs=args.expected_overlay_runs,
        )
    except CurrentNoiseNumericExportError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
