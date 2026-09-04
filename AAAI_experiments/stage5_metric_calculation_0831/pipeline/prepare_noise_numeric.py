"""准备 Stage5 noise001/noise005 的最终性能 canonical replay 结果。

该模块只消费 ``source_runs.csv`` 与 result-freeze bundle，不重新运行算法，也不
修改既有 clean/eff/trajectory 管线。每条冻结结果先与 source manifest 严格绑定，
再通过 ``performance_replay.replay_payload_performance`` 在统一数据尺度上重放。
输出的数值 CSV 保留 ``aggregate_noise_supplement`` 所需的最小字段，并把 native
与 replay 指标、canonical artifact 指纹及错误信息一并落盘。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import phi_nmse
from .performance_replay import (
    EVALUATION_PATH,
    FormulaRecoveryManifest,
    PerformanceReplayCache,
    PerformanceReplayError,
    load_formula_recovery_manifest,
    replay_payload_performance,
)


NOISE_CONDITIONS = ("noise001", "noise005")
SEEDS = (520, 521, 522)
CONDITION_SIGMA = {"noise001": 0.01, "noise005": 0.05}
DEFAULT_EXPECTED_RUNS = 2250
DEFAULT_EXPECTED_ALGORITHMS = 15
DEFAULT_EXPECTED_DATASETS = 50

# 前十列与 aggregate_noise_supplement._load_numeric_rows 的输入契约一致；其余列
# 是 replay 审计信息，聚合器会忽略但下游报告可以直接追溯。
NUMERIC_FIELDS = (
    "logical_key",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "task_id",
    "host",
    "result_sha256",
    "evaluation_status",
    "valid_output",
    "invalid_reason",
    "formula_source",
    "evaluation_path",
    "canonical_artifact_sha256",
    "artifact_rebuilt",
    "native_id_nmse",
    "native_ood_nmse",
    "id_nmse",
    "ood_nmse",
    "id_quality",
    "ood_quality",
    "id_nmse_delta_from_native",
    "ood_nmse_delta_from_native",
    "id_quality_delta_from_native",
    "ood_quality_delta_from_native",
    "replay_error",
)

EVALUATION_VALID = "valid"
EVALUATION_INVALID_OUTPUT = "invalid_output"
EVALUATION_REPLAY_UNAVAILABLE = "replay_unavailable"

SOURCE_FIELDS = {
    "batch",
    "algorithm",
    "dataset_id",
    "seed",
    "noise_tag",
    "task_id",
    "host",
    "status",
    "id_nmse",
    "ood_nmse",
    "path",
    "logical_key",
}
BUNDLE_SOURCE_FIELDS = {
    "algorithm",
    "batch",
    "dataset_id",
    "host",
    "noise_tag",
    "path",
    "seed",
    "source_row_sha256",
    "task_id",
}


class NoiseNumericPreparationError(ValueError):
    """noise 冻结来源或 canonical replay 结果不满足闭环契约。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _raise(message: str) -> None:
    raise NoiseNumericPreparationError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise NoiseNumericPreparationError(f"无法读取文件: {path}: {exc}") from exc
    return digest.hexdigest()


def _nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(f"{context} 必须是非空字符串")
    return value.strip()


def _finite_nonnegative(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        _raise(f"{context} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoiseNumericPreparationError(f"{context} 不是合法数值: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        _raise(f"{context} 必须是非负有限数值: {value!r}")
    return number


def _approximately_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-12)


def _parse_seed(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        _raise(f"{context} 不是合法 seed")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoiseNumericPreparationError(f"{context} 不是合法 seed: {value!r}") from exc
    if seed not in SEEDS:
        _raise(f"{context} 必须在 {SEEDS} 中，实际为 {seed}")
    return seed


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        _raise(f"{label} 不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NoiseNumericPreparationError(f"无法读取 {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _raise(f"{label} 顶层必须是 JSON object")
    return payload


def _read_source_manifest(
    path: Path,
    *,
    expected_runs: int,
    expected_algorithms: int,
    expected_datasets: int,
) -> tuple[dict[str, dict[tuple[str, str, int], dict[str, str]]], set[str], set[str]]:
    """读取并校验两个 noise condition 的完整 15 x 50 x 3 网格。"""

    if not path.is_file():
        _raise(f"source_runs.csv 不存在: {path}")
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise NoiseNumericPreparationError(f"无法打开 source_runs.csv: {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(SOURCE_FIELDS - fields)
        if missing:
            _raise(f"source_runs.csv 缺少字段: {missing}")
        by_condition: dict[str, dict[tuple[str, str, int], dict[str, str]]] = {
            condition: {} for condition in NOISE_CONDITIONS
        }
        for line_number, raw in enumerate(reader, start=2):
            condition = raw.get("noise_tag", "")
            if condition not in NOISE_CONDITIONS:
                continue
            context = f"source_runs.csv:{line_number}"
            algorithm = _nonempty(raw.get("algorithm"), context=f"{context}.algorithm")
            dataset_id = _nonempty(raw.get("dataset_id"), context=f"{context}.dataset_id")
            seed = _parse_seed(raw.get("seed"), context=f"{context}.seed")
            _nonempty(raw.get("batch"), context=f"{context}.batch")
            _nonempty(raw.get("task_id"), context=f"{context}.task_id")
            _nonempty(raw.get("host"), context=f"{context}.host")
            _nonempty(raw.get("path"), context=f"{context}.path")
            if raw.get("status") != "ok":
                _raise(f"{context} status 必须为 ok")
            _finite_nonnegative(raw.get("id_nmse"), context=f"{context}.id_nmse")
            _finite_nonnegative(raw.get("ood_nmse"), context=f"{context}.ood_nmse")
            expected_key = f"{algorithm}::{dataset_id}::s{seed}::{condition}"
            if raw.get("logical_key") != expected_key:
                _raise(f"{context}.logical_key 不符合 canonical 格式: {raw.get('logical_key')!r}")
            key = (algorithm, dataset_id, seed)
            if key in by_condition[condition]:
                _raise(f"source_runs.csv 出现重复逻辑键: {expected_key}")
            # DictReader 返回的字段均为字符串。复制一份而不是保留 reader 的可变视图，
            # 以便后续严格重算 source_row_sha256。
            by_condition[condition][key] = dict(raw)

    for condition in NOISE_CONDITIONS:
        rows = by_condition[condition]
        if len(rows) != expected_runs:
            _raise(f"{condition} source_runs 应为 {expected_runs} 条，实际为 {len(rows)}")
    algorithms = set(key[0] for key in by_condition[NOISE_CONDITIONS[0]])
    datasets = set(key[1] for key in by_condition[NOISE_CONDITIONS[0]])
    if len(algorithms) != expected_algorithms:
        _raise(f"noise source_runs 算法数应为 {expected_algorithms}，实际为 {len(algorithms)}")
    if len(datasets) != expected_datasets:
        _raise(f"noise source_runs 数据集数应为 {expected_datasets}，实际为 {len(datasets)}")
    expected_keys = {
        (algorithm, dataset, seed)
        for algorithm in algorithms
        for dataset in datasets
        for seed in SEEDS
    }
    for condition in NOISE_CONDITIONS:
        keys = set(by_condition[condition])
        if keys != expected_keys:
            _raise(
                f"{condition} source_runs algorithm/dataset/seed 网格不完整: "
                f"missing={len(expected_keys - keys)}, unexpected={len(keys - expected_keys)}"
            )
        condition_algorithms = {key[0] for key in keys}
        condition_datasets = {key[1] for key in keys}
        if condition_algorithms != algorithms or condition_datasets != datasets:
            _raise(f"{condition} source_runs 的算法/数据集集合与 noise001 不一致")
    return by_condition, algorithms, datasets


def _resolve_declared_path(value: object, *, report_path: Path, label: str) -> Path:
    declared = _nonempty(value, context=label)
    path = Path(declared)
    if path.is_absolute():
        return path.resolve()
    candidates = ((report_path.parent / path).resolve(), (_repo_root() / path).resolve())
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _validate_bundle_report(
    report_path: Path,
    *,
    condition: str,
    bundle_path: Path,
    source_runs_path: Path,
    expected_runs: int,
) -> dict[str, Any]:
    """绑定既有 result_freeze_adapter report（若存在），防止 bundle 替换。"""

    report = _read_json_object(report_path, label=f"{condition} bundle report")
    required_fields = {
        "status",
        "condition",
        "row_count",
        "output_jsonl_gz",
        "output_sha256",
        "source_runs_csv",
        "source_runs_sha256",
    }
    missing_fields = sorted(required_fields - set(report))
    if missing_fields:
        _raise(f"{condition} bundle report 缺少必填字段: {missing_fields}")
    if report.get("status") != "ok":
        _raise(f"{condition} bundle report.status 必须为 ok")
    if report.get("condition") != condition:
        _raise(f"{condition} bundle report.condition 不一致")
    try:
        row_count = int(report["row_count"])
    except (TypeError, ValueError) as exc:
        raise NoiseNumericPreparationError(f"{condition} bundle report.row_count 非法") from exc
    if row_count != expected_runs:
        _raise(f"{condition} bundle report.row_count 应为 {expected_runs}，实际为 {row_count}")

    actual_bundle_sha = _sha256_file(bundle_path)
    declared_bundle = _resolve_declared_path(
        report["output_jsonl_gz"], report_path=report_path, label=f"{condition}.output_jsonl_gz"
    )
    if declared_bundle != bundle_path.resolve():
        _raise(f"{condition} bundle report.output_jsonl_gz 与输入 bundle 不一致")
    if report["output_sha256"] != actual_bundle_sha:
        _raise(
            f"{condition} bundle SHA-256 校验失败: {actual_bundle_sha} != {report['output_sha256']}"
        )

    source_sha = _sha256_file(source_runs_path)
    declared_source = _resolve_declared_path(
        report["source_runs_csv"], report_path=report_path, label=f"{condition}.source_runs_csv"
    )
    if declared_source != source_runs_path.resolve():
        _raise(f"{condition} bundle report.source_runs_csv 与输入 manifest 不一致")
    if report["source_runs_sha256"] != source_sha:
        _raise(f"{condition} bundle report.source_runs_sha256 与 source_runs 不一致")
    return {
        "path": str(report_path.resolve()),
        "sha256": _sha256_file(report_path),
        "bundle_sha256": actual_bundle_sha,
    }


def _iter_bundle(path: Path, *, condition: str) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        _raise(f"{condition} result bundle 不存在: {path}")
    try:
        handle = gzip.open(path, "rt", encoding="utf-8")
    except OSError as exc:
        raise NoiseNumericPreparationError(f"无法打开 {condition} result bundle: {path}: {exc}") from exc
    try:
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    _raise(f"{condition} result bundle:{line_number} 不得是空行")
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise NoiseNumericPreparationError(
                        f"{condition} result bundle:{line_number} 不是合法 JSON"
                    ) from exc
                if not isinstance(record, dict):
                    _raise(f"{condition} result bundle:{line_number} 顶层必须是 object")
                yield line_number, record
    except OSError as exc:
        raise NoiseNumericPreparationError(
            f"读取 {condition} result bundle 失败: {path}: {exc}"
        ) from exc


def _validate_bundle_source(
    source: Mapping[str, Any],
    *,
    condition: str,
    manifest_by_key: Mapping[tuple[str, str, int], Mapping[str, str]],
    context: str,
) -> tuple[tuple[str, str, int], Mapping[str, str]]:
    missing = sorted(BUNDLE_SOURCE_FIELDS - set(source))
    if missing:
        _raise(f"{context}.source 缺少字段: {missing}")
    algorithm = _nonempty(source.get("algorithm"), context=f"{context}.source.algorithm")
    dataset_id = _nonempty(source.get("dataset_id"), context=f"{context}.source.dataset_id")
    seed = _parse_seed(source.get("seed"), context=f"{context}.source.seed")
    source_noise = _nonempty(source.get("noise_tag"), context=f"{context}.source.noise_tag")
    if source_noise != condition:
        _raise(f"{context}.source.noise_tag 必须为 {condition}")
    key = (algorithm, dataset_id, seed)
    manifest = manifest_by_key.get(key)
    if manifest is None:
        _raise(f"{context} 出现未声明逻辑键: {algorithm}::{dataset_id}::s{seed}::{condition}")

    for field in ("batch", "algorithm", "dataset_id", "noise_tag", "task_id", "host", "path"):
        if str(source.get(field)) != str(manifest.get(field)):
            _raise(f"{context}.source.{field} 与 source_runs 不一致")
    if _parse_seed(source.get("seed"), context=f"{context}.source.seed") != int(manifest["seed"]):
        _raise(f"{context}.source.seed 与 source_runs 不一致")
    source_row_hash = _nonempty(
        source.get("source_row_sha256"), context=f"{context}.source.source_row_sha256"
    )
    expected_hash = _sha256_text(_canonical_json(dict(manifest)))
    if source_row_hash.lower() != expected_hash:
        _raise(f"{context}.source.source_row_sha256 与 source_runs 不一致")
    return key, manifest


def _payload_split_nmse(payload: Mapping[str, Any], split: str, *, context: str) -> float:
    block = payload.get(split)
    if not isinstance(block, Mapping):
        _raise(f"{context}.{split} 缺失")
    return _finite_nonnegative(block.get("nmse"), context=f"{context}.{split}.nmse")


def _validate_noise_payload(
    payload: Mapping[str, Any], *, condition: str, key: str
) -> tuple[float, float, str]:
    if payload.get("status") != "ok":
        _raise(f"{key} 的冻结 result.status 必须为 ok")
    noise = payload.get("train_label_noise")
    if not isinstance(noise, Mapping):
        _raise(f"{key} 缺少 train_label_noise")
    if noise.get("enabled") is not True or noise.get("requested") is not True:
        _raise(f"{key} train_label_noise 标志与 {condition} 不一致")
    sigma = _finite_nonnegative(noise.get("sigma"), context=f"{key}.train_label_noise.sigma")
    if not math.isclose(sigma, CONDITION_SIGMA[condition], rel_tol=0.0, abs_tol=1.0e-12):
        _raise(f"{key} train_label_noise.sigma 与 {condition} 不一致")
    id_nmse = _payload_split_nmse(payload, "id_test", context=key)
    ood_nmse = _payload_split_nmse(payload, "ood_test", context=key)
    formula = ""
    artifact = payload.get("canonical_artifact")
    if isinstance(artifact, Mapping):
        for field in (
            "instantiated_expression",
            "normalized_expression",
            "return_expression_source",
            "python_function_source",
        ):
            value = artifact.get(field)
            if isinstance(value, str) and value.strip():
                formula = value.strip()
                break
    if not formula and isinstance(payload.get("equation"), str):
        formula = payload["equation"].strip()
    return id_nmse, ood_nmse, formula


def _resolve_replay(
    payload: Mapping[str, Any],
    *,
    algorithm: str,
    repo_root: Path,
    cache: PerformanceReplayCache,
    key: str,
    recovery_manifest: FormulaRecoveryManifest,
    task_id: str,
    condition: str,
    result_sha256: str,
) -> dict[str, Any]:
    """执行 replay 并规范化字段；失败由调用方转为该行 replay_error。"""

    replay = replay_payload_performance(
        payload,
        algorithm=algorithm,
        repo_root=repo_root,
        cache=cache,
        recovery_manifest=recovery_manifest,
        task_id=task_id,
        condition=condition,
        result_sha256=result_sha256,
    )
    if not isinstance(replay, Mapping):
        _raise(f"{key} canonical replay 返回值不是 object")
    if replay.get("evaluation_path") != EVALUATION_PATH:
        _raise(f"{key} canonical replay evaluation_path 非 {EVALUATION_PATH}")
    artifact = replay.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        _raise(f"{key} canonical replay 缺少 canonical_artifact")
    artifact_sha = _nonempty(
        replay.get("canonical_artifact_sha256"), context=f"{key}.canonical_artifact_sha256"
    )
    if len(artifact_sha) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_sha.lower()):
        _raise(f"{key}.canonical_artifact_sha256 必须为 SHA256")
    rebuilt = replay.get("artifact_rebuilt")
    if not isinstance(rebuilt, bool):
        _raise(f"{key}.artifact_rebuilt 必须为 bool")
    valid = replay.get("valid_output")
    if not isinstance(valid, bool):
        _raise(f"{key}.valid_output 必须为 bool")
    id_block = replay.get("id_test")
    ood_block = replay.get("ood_test")
    invalid_reason = str(replay.get("invalid_reason") or "").strip()
    if valid:
        if not isinstance(id_block, Mapping) or not isinstance(ood_block, Mapping):
            _raise(f"{key} canonical replay 缺少 id_test/ood_test")
        id_nmse = _finite_nonnegative(
            id_block.get("nmse"), context=f"{key}.replay.id_test.nmse"
        )
        ood_nmse = _finite_nonnegative(
            ood_block.get("nmse"), context=f"{key}.replay.ood_test.nmse"
        )
    else:
        if not invalid_reason:
            _raise(f"{key} canonical replay 的无效输出缺少 invalid_reason")
        id_nmse = None
        ood_nmse = None
    return {
        "artifact_sha256": artifact_sha,
        "artifact_rebuilt": rebuilt,
        "valid_output": valid,
        "invalid_reason": invalid_reason,
        "id_nmse": id_nmse,
        "ood_nmse": ood_nmse,
        "error": replay.get("error"),
    }


def _format(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return f"{number:.17g}"


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NUMERIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_bundle_report(source_runs_csv: Path, condition: str) -> Path | None:
    candidate = source_runs_csv.parent.parent / "reports" / f"{condition}_result_freeze_adapter.json"
    return candidate if candidate.is_file() else None


def _prepare_condition(
    *,
    condition: str,
    bundle_path: Path,
    bundle_report_path: Path | None,
    manifest_by_key: Mapping[tuple[str, str, int], Mapping[str, str]],
    source_runs_csv: Path,
    run_csv: Path,
    expected_runs: int,
    repo_root: Path,
    replay_cache: PerformanceReplayCache,
    recovery_manifest: FormulaRecoveryManifest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in NOISE_CONDITIONS:
        _raise(f"未知 noise condition: {condition!r}")
    bundle_path = bundle_path.resolve()
    report_info = None
    if bundle_report_path is not None:
        report_info = _validate_bundle_report(
            bundle_report_path.resolve(),
            condition=condition,
            bundle_path=bundle_path,
            source_runs_path=source_runs_csv,
            expected_runs=expected_runs,
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for line_number, record in _iter_bundle(bundle_path, condition=condition):
        context = f"{condition} result bundle:{line_number}"
        source = record.get("source")
        if not isinstance(source, Mapping):
            _raise(f"{context} 缺少 source")
        key_tuple, manifest = _validate_bundle_source(
            source,
            condition=condition,
            manifest_by_key=manifest_by_key,
            context=context,
        )
        if key_tuple in seen:
            _raise(f"{condition} result bundle 出现重复逻辑键: {key_tuple}")
        result = record.get("result")
        if not isinstance(result, Mapping):
            _raise(f"{context} 缺少 result")
        raw_text = result.get("raw_text")
        result_sha = _nonempty(result.get("sha256"), context=f"{context}.result.sha256")
        if len(result_sha) != 64 or any(ch not in "0123456789abcdef" for ch in result_sha.lower()):
            _raise(f"{context}.result.sha256 必须为 SHA256")
        if not isinstance(raw_text, str) or _sha256_text(raw_text) != result_sha:
            _raise(f"{context}.result.raw_text SHA-256 校验失败")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise NoiseNumericPreparationError(f"{context}.result.raw_text 不是合法 JSON") from exc
        if not isinstance(payload, Mapping):
            _raise(f"{context}.result.raw_text 顶层必须是 object")

        algorithm, dataset_id, seed = key_tuple
        key = f"{algorithm}::{dataset_id}::s{seed}::{condition}"
        native_id, native_ood, formula = _validate_noise_payload(
            payload, condition=condition, key=key
        )
        source_id = _finite_nonnegative(manifest["id_nmse"], context=f"{key}.source.id_nmse")
        source_ood = _finite_nonnegative(manifest["ood_nmse"], context=f"{key}.source.ood_nmse")
        if not _approximately_equal(native_id, source_id):
            _raise(f"{key} 的 ID NMSE 与 source_runs 不一致")
        if not _approximately_equal(native_ood, source_ood):
            _raise(f"{key} 的 OOD NMSE 与 source_runs 不一致")

        replay_error = ""
        invalid_reason = ""
        replay_id: float | None = None
        replay_ood: float | None = None
        artifact_sha = ""
        artifact_rebuilt = False
        evaluation_status = EVALUATION_REPLAY_UNAVAILABLE
        valid_output: bool | None = None
        if formula:
            try:
                replay = _resolve_replay(
                    payload,
                    algorithm=algorithm,
                    repo_root=repo_root,
                    cache=replay_cache,
                    key=key,
                    recovery_manifest=recovery_manifest,
                    task_id=str(manifest["task_id"]),
                    condition=condition,
                    result_sha256=result_sha,
                )
            except (NoiseNumericPreparationError, PerformanceReplayError) as exc:
                replay_error = str(exc)
            else:
                artifact_sha = str(replay["artifact_sha256"])
                artifact_rebuilt = bool(replay["artifact_rebuilt"])
                valid_output = bool(replay["valid_output"])
                invalid_reason = str(replay.get("invalid_reason") or "")
                if valid_output:
                    evaluation_status = EVALUATION_VALID
                    replay_id = float(replay["id_nmse"])
                    replay_ood = float(replay["ood_nmse"])
                else:
                    evaluation_status = EVALUATION_INVALID_OUTPUT
                if replay.get("error"):
                    replay_error = str(replay["error"])
                    evaluation_status = EVALUATION_REPLAY_UNAVAILABLE
                    valid_output = None
                    invalid_reason = ""
                    replay_id = None
                    replay_ood = None
        else:
            replay_error = "冻结 result 缺少 equation/canonical_artifact expression"

        native_id_quality = phi_nmse(native_id)
        native_ood_quality = phi_nmse(native_ood)
        if evaluation_status == EVALUATION_VALID:
            id_quality: float | None = phi_nmse(replay_id)
            ood_quality: float | None = phi_nmse(replay_ood)
        elif evaluation_status == EVALUATION_INVALID_OUTPUT:
            id_quality = 0.0
            ood_quality = 0.0
        else:
            valid_output = None
            invalid_reason = ""
            id_quality = None
            ood_quality = None
        rows.append(
            {
                "logical_key": key,
                "algorithm": algorithm,
                "dataset_id": dataset_id,
                "seed": seed,
                "noise_tag": condition,
                "task_id": manifest["task_id"],
                "host": manifest["host"],
                "result_sha256": result_sha,
                "evaluation_status": evaluation_status,
                "valid_output": "" if valid_output is None else str(valid_output).lower(),
                "invalid_reason": invalid_reason,
                "formula_source": (
                    "canonical_artifact"
                    if isinstance(payload.get("canonical_artifact"), Mapping)
                    else "equation"
                ),
                "evaluation_path": EVALUATION_PATH,
                "canonical_artifact_sha256": artifact_sha,
                "artifact_rebuilt": str(artifact_rebuilt).lower(),
                "native_id_nmse": _format(native_id),
                "native_ood_nmse": _format(native_ood),
                "id_nmse": _format(replay_id),
                "ood_nmse": _format(replay_ood),
                "id_quality": _format(id_quality),
                "ood_quality": _format(ood_quality),
                "id_nmse_delta_from_native": (
                    _format(replay_id - native_id) if replay_id is not None else ""
                ),
                "ood_nmse_delta_from_native": (
                    _format(replay_ood - native_ood) if replay_ood is not None else ""
                ),
                "id_quality_delta_from_native": (
                    _format(id_quality - native_id_quality) if id_quality is not None else ""
                ),
                "ood_quality_delta_from_native": (
                    _format(ood_quality - native_ood_quality) if ood_quality is not None else ""
                ),
                "replay_error": replay_error,
            }
        )
        seen.add(key_tuple)

    if len(rows) != expected_runs:
        _raise(f"{condition} result bundle 应为 {expected_runs} 条，实际为 {len(rows)}")
    missing = set(manifest_by_key) - seen
    unexpected = seen - set(manifest_by_key)
    if missing or unexpected:
        _raise(f"{condition} result bundle 网格不完整: missing={len(missing)}, unexpected={len(unexpected)}")
    rows.sort(key=lambda row: (str(row["algorithm"]), str(row["dataset_id"]), int(row["seed"])))
    _atomic_write_csv(run_csv.resolve(), rows)
    status_counts = Counter(row["evaluation_status"] for row in rows)
    condition_report: dict[str, Any] = {
        "path": str(run_csv.resolve()),
        "sha256": _sha256_file(run_csv.resolve()),
        "row_count": len(rows),
    }
    stats = {
        "runs": len(rows),
        "valid_outputs": status_counts[EVALUATION_VALID],
        "canonical_invalid_outputs": status_counts[EVALUATION_INVALID_OUTPUT],
        "replay_unavailable": status_counts[EVALUATION_REPLAY_UNAVAILABLE],
        "artifact_rebuilt": sum(row["artifact_rebuilt"] == "true" for row in rows),
        "replay_errors": sum(bool(row["replay_error"]) for row in rows),
    }
    input_info: dict[str, Any] = {
        "path": str(bundle_path),
        "sha256": _sha256_file(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "row_count": len(rows),
    }
    if report_info is not None:
        input_info["bundle_report"] = report_info
    return {"output": condition_report, "stats": stats}, input_info


def prepare_noise_numeric(
    *,
    source_runs_csv: Path,
    noise001_bundle: Path,
    noise005_bundle: Path,
    noise001_run_csv: Path,
    noise005_run_csv: Path,
    report_json: Path,
    noise001_bundle_report_json: Path | None = None,
    noise005_bundle_report_json: Path | None = None,
    noise001_formula_recovery_json: Path | None = None,
    noise005_formula_recovery_json: Path | None = None,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_algorithms: int = DEFAULT_EXPECTED_ALGORITHMS,
    expected_datasets: int = DEFAULT_EXPECTED_DATASETS,
    repo_root: Path | None = None,
    require_bundle_reports: bool = False,
) -> dict[str, Any]:
    """为 noise001/noise005 生成可直接供噪声聚合器读取的 run CSV。"""

    if expected_runs != expected_algorithms * expected_datasets * len(SEEDS):
        _raise("expected_runs 必须等于 expected_algorithms * expected_datasets * 3")
    source_runs_csv = source_runs_csv.resolve()
    manifest, _algorithms, _datasets = _read_source_manifest(
        source_runs_csv,
        expected_runs=expected_runs,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
    )
    replay_repo_root = repo_root.resolve() if repo_root is not None else _repo_root()
    replay_cache = PerformanceReplayCache()
    recovery_paths = {
        "noise001": (
            noise001_formula_recovery_json.resolve()
            if noise001_formula_recovery_json is not None
            else _repo_root()
            / "AAAI_experiments/stage5_metric_calculation_0831/manifests/noise001_formula_recovery.v1.json"
        ),
        "noise005": (
            noise005_formula_recovery_json.resolve()
            if noise005_formula_recovery_json is not None
            else _repo_root()
            / "AAAI_experiments/stage5_metric_calculation_0831/manifests/noise005_formula_recovery.v1.json"
        ),
    }
    recovery_manifests: dict[str, FormulaRecoveryManifest] = {}
    for condition, path in recovery_paths.items():
        try:
            recovery_manifests[condition] = load_formula_recovery_manifest(
                path, expected_condition=condition
            )
        except PerformanceReplayError as exc:
            raise NoiseNumericPreparationError(str(exc)) from exc
    bundle_reports = {
        "noise001": noise001_bundle_report_json,
        "noise005": noise005_bundle_report_json,
    }
    # 若调用方没有显式传 report，则仅在当前 Stage5 report 存在时自动绑定；测试夹具
    # 或独立导出可以不提供 report，但 bundle 自身仍始终进行 SHA 计算并写入总报告。
    for condition in NOISE_CONDITIONS:
        if bundle_reports[condition] is None:
            bundle_reports[condition] = _default_bundle_report(source_runs_csv, condition)
        if require_bundle_reports and (
            bundle_reports[condition] is None
            or not Path(bundle_reports[condition]).resolve().is_file()
        ):
            _raise(f"{condition} 必须提供冻结 report")

    outputs: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "source_runs_csv": {
            "path": str(source_runs_csv),
            "sha256": _sha256_file(source_runs_csv),
        },
        "bundles": {},
        "formula_recovery_manifests": {
            condition: {
                "path": str(recovery_manifests[condition].path),
                "sha256": recovery_manifests[condition].sha256,
                "condition": recovery_manifests[condition].condition,
                "entry_count": len(recovery_manifests[condition].entries),
            }
            for condition in NOISE_CONDITIONS
        },
    }
    counts: dict[str, Any] = {}
    condition_paths = {
        "noise001": (noise001_bundle, noise001_run_csv),
        "noise005": (noise005_bundle, noise005_run_csv),
    }
    for condition in NOISE_CONDITIONS:
        condition_report, bundle_info = _prepare_condition(
            condition=condition,
            bundle_path=condition_paths[condition][0],
            bundle_report_path=bundle_reports[condition],
            manifest_by_key=manifest[condition],
            source_runs_csv=source_runs_csv,
            run_csv=condition_paths[condition][1],
            expected_runs=expected_runs,
            repo_root=replay_repo_root,
            replay_cache=replay_cache,
            recovery_manifest=recovery_manifests[condition],
        )
        outputs[condition] = condition_report["output"]
        counts[condition] = condition_report["stats"]
        inputs["bundles"][condition] = bundle_info

    replay_error_count = sum(int(condition["replay_errors"]) for condition in counts.values())
    replay_unavailable_count = sum(
        int(condition["replay_unavailable"]) for condition in counts.values()
    )
    report: dict[str, Any] = {
        "status": "ok" if replay_error_count == 0 and replay_unavailable_count == 0 else "error",
        "contract_ok": replay_error_count == 0 and replay_unavailable_count == 0,
        "evaluation_path": EVALUATION_PATH,
        "metric_definition": "canonical-replay-phi-per-task-seed",
        "counts": counts,
        "inputs": inputs,
        "outputs": outputs,
        "conditions": list(NOISE_CONDITIONS),
        "bundle_reports_required": require_bundle_reports,
    }
    _atomic_write_json(report_json.resolve(), report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    stage5_root = _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="准备 Stage5 noise canonical replay 数值结果")
    parser.add_argument("--source-runs-csv", type=Path, default=stage5_root / "manifests/source_runs.csv")
    parser.add_argument(
        "--noise001-bundle",
        type=Path,
        default=stage5_root / "source_snapshot/result_freeze/noise001_results.jsonl.gz",
    )
    parser.add_argument(
        "--noise005-bundle",
        type=Path,
        default=stage5_root / "source_snapshot/result_freeze/noise005_results.jsonl.gz",
    )
    parser.add_argument(
        "--noise001-run-csv",
        type=Path,
        default=stage5_root / "results/noise001_numeric_run_metrics.csv",
    )
    parser.add_argument(
        "--noise005-run-csv",
        type=Path,
        default=stage5_root / "results/noise005_numeric_run_metrics.csv",
    )
    parser.add_argument(
        "--noise001-bundle-report-json", type=Path, default=None
    )
    parser.add_argument(
        "--noise005-bundle-report-json", type=Path, default=None
    )
    parser.add_argument(
        "--noise001-formula-recovery-json",
        type=Path,
        default=stage5_root / "manifests/noise001_formula_recovery.v1.json",
    )
    parser.add_argument(
        "--noise005-formula-recovery-json",
        type=Path,
        default=stage5_root / "manifests/noise005_formula_recovery.v1.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=stage5_root / "reports/noise_numeric_preparation.json",
    )
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--expected-algorithms", type=int, default=DEFAULT_EXPECTED_ALGORITHMS)
    parser.add_argument("--expected-datasets", type=int, default=DEFAULT_EXPECTED_DATASETS)
    parser.add_argument("--repo-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    report = prepare_noise_numeric(
        source_runs_csv=args.source_runs_csv,
        noise001_bundle=args.noise001_bundle,
        noise005_bundle=args.noise005_bundle,
        noise001_run_csv=args.noise001_run_csv,
        noise005_run_csv=args.noise005_run_csv,
        report_json=args.report_json,
        noise001_bundle_report_json=args.noise001_bundle_report_json,
        noise005_bundle_report_json=args.noise005_bundle_report_json,
        noise001_formula_recovery_json=args.noise001_formula_recovery_json,
        noise005_formula_recovery_json=args.noise005_formula_recovery_json,
        expected_runs=args.expected_runs,
        expected_algorithms=args.expected_algorithms,
        expected_datasets=args.expected_datasets,
        repo_root=args.repo_root,
        require_bundle_reports=True,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0 if report.get("contract_ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
