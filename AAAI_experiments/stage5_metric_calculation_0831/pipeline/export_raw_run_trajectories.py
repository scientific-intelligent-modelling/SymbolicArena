"""导出 Stage5 三种条件下可复现的逐 run 180 分钟轨迹。

本模块只读取既有冻结输入并创建新的导出目录，
不修改实验源数据或聚合结果。
clean 严格复用 ``export_result_summary`` 的 current 优先、legacy 显式回退
口径；noise 严格复用其 native snapshot 数值 best-so-far 后处理口径。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .metrics import phi_nmse


HORIZON = 180
CONDITIONS = ("clean", "noise001", "noise005")
FORBIDDEN_SOURCE_TOKENS = ("all_15alg_fullcpu_v1",)
CLEAN_BASIS = "internal_search_canonical_replay_with_explicit_legacy_fallback.v1"
NOISE_BASIS = "observed_numeric_best_so_far_native_snapshot.v1"


class RawTrajectoryExportError(ValueError):
    """输入不满足逐 run 轨迹导出契约。"""


@dataclass(frozen=True)
class GridContract:
    expected_algorithm_count: int = 15
    expected_dataset_count: int = 50
    expected_seeds: tuple[int, ...] = (520, 521, 522)
    expected_runs_per_condition: int = 2250
    expected_runs_per_algorithm: int = 150


def _check_forbidden(value: object, *, context: str) -> None:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value)
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise RawTrajectoryExportError(f"{context} 命中禁止来源 {token!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    _check_forbidden(resolved, context="输入路径")
    if not resolved.is_file():
        raise RawTrajectoryExportError(f"输入文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "record_count": 0,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RawTrajectoryExportError(f"CSV 为空: {path}")
    for line_number, row in enumerate(rows, start=2):
        _check_forbidden(row, context=f"{path}:{line_number}")
    return rows


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".gz":
        handle_context = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle_context = path.open("r", encoding="utf-8")
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RawTrajectoryExportError(f"{path}:{line_number} JSON 无效") from exc
            if not isinstance(record, dict):
                raise RawTrajectoryExportError(f"{path}:{line_number} 不是 JSON object")
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise RawTrajectoryExportError(f"{path}:{line_number} 缺少 source object")
            _check_forbidden(source, context=f"{path}:{line_number}.source")
            yield record


def _logical_key(source: Mapping[str, Any]) -> str:
    try:
        return (
            f"{source['algorithm']}::{source['dataset_id']}::"
            f"s{int(source['seed'])}::{source['noise_tag']}"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RawTrajectoryExportError("source 缺少合法的轨迹身份字段") from exc


def _qualities_from_clean_row(row: Mapping[str, str]) -> list[float]:
    values: list[float] = []
    for minute in range(1, HORIZON + 1):
        field = f"q_{minute:04d}"
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RawTrajectoryExportError(
                f"{row.get('logical_key')} 缺少有效 {field}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RawTrajectoryExportError(
                f"{row.get('logical_key')}.{field} 不在 [0,1]"
            )
        values.append(value)
    return values


def _snapshot_nmse(snapshot: Mapping[str, Any], field: str) -> float | None:
    value = snapshot.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _postprocessed_best_so_far(
    record: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    """与 ``export_result_summary.postprocessed_best_so_far`` 保持同一口径。"""

    raw_snapshots = record.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise RawTrajectoryExportError("轨迹记录缺少 snapshots")
    snapshots = {
        int(snapshot["minute"]): snapshot
        for snapshot in raw_snapshots
        if isinstance(snapshot, Mapping) and snapshot.get("minute") is not None
    }
    best = (0.0, 0.0, 0.0)
    ids: list[float] = []
    oods: list[float] = []
    qualities: list[float] = []
    for minute in range(1, HORIZON + 1):
        snapshot = snapshots.get(minute)
        if snapshot is not None:
            id_nmse = _snapshot_nmse(snapshot, "id_nmse")
            ood_nmse = _snapshot_nmse(snapshot, "ood_nmse")
            if id_nmse is not None and ood_nmse is not None:
                id_quality = phi_nmse(id_nmse)
                ood_quality = phi_nmse(ood_nmse)
                candidate = (
                    id_quality,
                    ood_quality,
                    (id_quality + ood_quality) / 2.0,
                )
                if candidate[2] > best[2]:
                    best = candidate
        ids.append(best[0])
        oods.append(best[1])
        qualities.append(best[2])
    return ids, oods, qualities


def _eff_summary(qualities: Sequence[float]) -> tuple[float, float]:
    if len(qualities) != HORIZON:
        raise RawTrajectoryExportError(f"轨迹点数不是 {HORIZON}: {len(qualities)}")
    q_star = max(qualities, default=0.0)
    m_eff = (
        sum(value / q_star for value in qualities) / HORIZON if q_star > 0.0 else 0.0
    )
    return q_star, m_eff


def _base_output_row(
    *,
    source: Mapping[str, Any],
    source_tier: str,
    trajectory_basis: str,
    source_input: Path,
    source_input_sha256: str,
    q_star: float,
    m_eff: float,
) -> dict[str, Any]:
    condition = str(source["noise_tag"])
    key = _logical_key(source)
    return {
        "logical_key": key,
        "algorithm": str(source["algorithm"]),
        "dataset_id": str(source["dataset_id"]),
        "seed": int(source["seed"]),
        "task_id": str(source.get("task_id") or ""),
        "condition": condition,
        "host": str(source.get("host") or ""),
        "source_tier": source_tier,
        "trajectory_basis": trajectory_basis,
        "source_input": str(source_input.resolve()),
        "source_input_sha256": source_input_sha256,
        "source_record_path": str(source.get("path") or ""),
        "q_star": f"{q_star:.17g}",
        "m_eff": f"{m_eff:.17g}",
    }


def _clean_output_row(
    row: Mapping[str, str],
    *,
    source_tier: str,
    source_input: Path,
    source_input_sha256: str,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "algorithm": row.get("algorithm"),
        "dataset_id": row.get("dataset_id"),
        "seed": row.get("seed"),
        "noise_tag": row.get("noise_tag"),
        "task_id": row.get("task_id"),
        "host": row.get("host"),
    }
    key = _logical_key(source)
    if str(row.get("logical_key")) != key:
        raise RawTrajectoryExportError(
            f"clean logical_key 与字段不一致: {row.get('logical_key')}"
        )
    if source["noise_tag"] != "clean":
        raise RawTrajectoryExportError(f"clean 输入含非 clean 记录: {key}")
    qualities = _qualities_from_clean_row(row)
    q_star, m_eff = _eff_summary(qualities)
    try:
        recorded_m_eff = float(row["m_eff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RawTrajectoryExportError(f"{key} 缺少有效 m_eff") from exc
    if not math.isclose(m_eff, recorded_m_eff, rel_tol=1e-12, abs_tol=1e-12):
        raise RawTrajectoryExportError(
            f"{key} m_eff 与 180 分钟轨迹不一致: {recorded_m_eff} != {m_eff}"
        )
    output = _base_output_row(
        source=source,
        source_tier=source_tier,
        trajectory_basis=CLEAN_BASIS,
        source_input=source_input,
        source_input_sha256=source_input_sha256,
        q_star=q_star,
        m_eff=m_eff,
    )
    output["id_ood_quality_available"] = "false"
    for minute, quality in enumerate(qualities, start=1):
        output[f"q_{minute:04d}"] = f"{quality:.17g}"
    return output


def _noise_output_row(
    record: Mapping[str, Any],
    *,
    source_tier: str,
    source_input: Path,
    source_input_sha256: str,
) -> dict[str, Any]:
    source = record["source"]
    condition = str(source.get("noise_tag"))
    if condition not in ("noise001", "noise005"):
        raise RawTrajectoryExportError(f"noise 输入含非法 condition: {condition!r}")
    try:
        id_qualities, ood_qualities, qualities = _postprocessed_best_so_far(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise RawTrajectoryExportError(
            f"{_logical_key(source)} 的 snapshot 无法重建 180 分钟轨迹"
        ) from exc
    q_star, m_eff = _eff_summary(qualities)
    output = _base_output_row(
        source=source,
        source_tier=source_tier,
        trajectory_basis=NOISE_BASIS,
        source_input=source_input,
        source_input_sha256=source_input_sha256,
        q_star=q_star,
        m_eff=m_eff,
    )
    output["id_ood_quality_available"] = "true"
    for minute, value in enumerate(id_qualities, start=1):
        output[f"id_q_{minute:04d}"] = f"{value:.17g}"
    for minute, value in enumerate(ood_qualities, start=1):
        output[f"ood_q_{minute:04d}"] = f"{value:.17g}"
    for minute, value in enumerate(qualities, start=1):
        output[f"q_{minute:04d}"] = f"{value:.17g}"
    return output


def _coverage(
    rows: Sequence[Mapping[str, Any]], condition: str, contract: GridContract
) -> dict[str, Any]:
    keys: set[str] = set()
    grid: set[tuple[str, str, int]] = set()
    algorithms: set[str] = set()
    datasets: set[str] = set()
    seeds: set[int] = set()
    task_ids: set[str] = set()
    per_algorithm: Counter[str] = Counter()
    source_tiers: Counter[str] = Counter()
    for row in rows:
        if str(row["condition"]) != condition:
            raise RawTrajectoryExportError(f"{condition} 输出混入其它 condition")
        key = str(row["logical_key"])
        identity = (str(row["algorithm"]), str(row["dataset_id"]), int(row["seed"]))
        if key in keys or identity in grid:
            raise RawTrajectoryExportError(f"{condition} 出现重复 run: {key}")
        keys.add(key)
        grid.add(identity)
        algorithms.add(identity[0])
        datasets.add(identity[1])
        seeds.add(identity[2])
        task_id = str(row["task_id"])
        if not task_id:
            raise RawTrajectoryExportError(f"{key} 缺少 task_id")
        task_ids.add(task_id)
        per_algorithm[identity[0]] += 1
        source_tiers[str(row["source_tier"])] += 1

    expected_seed_set = set(contract.expected_seeds)
    errors: list[str] = []
    if len(rows) != contract.expected_runs_per_condition:
        errors.append(f"run={len(rows)}，期望 {contract.expected_runs_per_condition}")
    if len(algorithms) != contract.expected_algorithm_count:
        errors.append(f"算法={len(algorithms)}，期望 {contract.expected_algorithm_count}")
    if len(datasets) != contract.expected_dataset_count:
        errors.append(f"任务={len(datasets)}，期望 {contract.expected_dataset_count}")
    if len(task_ids) != len(rows):
        errors.append(f"task_id 唯一数={len(task_ids)}，run 数={len(rows)}")
    if seeds != expected_seed_set:
        errors.append(f"seeds={sorted(seeds)}，期望 {sorted(expected_seed_set)}")
    invalid_algorithm_counts = {
        algorithm: count
        for algorithm, count in per_algorithm.items()
        if count != contract.expected_runs_per_algorithm
    }
    if invalid_algorithm_counts:
        errors.append(f"算法 run 数错误: {invalid_algorithm_counts}")
    expected_grid = {
        (algorithm, dataset, seed)
        for algorithm in algorithms
        for dataset in datasets
        for seed in contract.expected_seeds
    }
    if grid != expected_grid:
        missing = sorted(expected_grid - grid)[:5]
        extra = sorted(grid - expected_grid)[:5]
        errors.append(f"笛卡尔网格不闭合，missing={missing}, extra={extra}")
    if errors:
        raise RawTrajectoryExportError(f"{condition} 覆盖契约失败: {'; '.join(errors)}")
    return {
        "record_count": len(rows),
        "algorithm_count": len(algorithms),
        "algorithms": sorted(algorithms),
        "dataset_count": len(datasets),
        "datasets": sorted(datasets),
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "task_id_count": len(task_ids),
        "per_algorithm_run_counts": dict(sorted(per_algorithm.items())),
        "source_tier_counts": dict(sorted(source_tiers.items())),
        "cartesian_grid_complete": True,
    }


def reverse_aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, condition: str
) -> list[dict[str, Any]]:
    """从逐 run 导出表反向生成现有 ``eff_180min.csv`` 的数值字段。"""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) != condition:
            raise RawTrajectoryExportError(f"反向聚合 {condition} 时混入其它 condition")
        grouped[str(row["algorithm"])].append(row)
    output: list[dict[str, Any]] = []
    for algorithm in sorted(grouped):
        runs = grouped[algorithm]
        relative_runs: list[list[float]] = []
        for row in runs:
            q_star = float(row["q_star"])
            qualities = [float(row[f"q_{minute:04d}"]) for minute in range(1, HORIZON + 1)]
            relative_runs.append(
                [quality / q_star if q_star > 0.0 else 0.0 for quality in qualities]
            )
        cumulative_relative = 0.0
        for minute in range(1, HORIZON + 1):
            mean_quality = sum(float(row[f"q_{minute:04d}"]) for row in runs) / len(runs)
            mean_relative = sum(run[minute - 1] for run in relative_runs) / len(runs)
            cumulative_relative += mean_relative
            result: dict[str, Any] = {
                "condition": condition,
                "algorithm": algorithm,
                "minute": minute,
                "mean_quality": mean_quality,
                "mean_relative_progress": mean_relative,
                "cumulative_eff_score": 100.0 * cumulative_relative / minute,
            }
            if condition == "clean":
                result["current_path_run_count"] = sum(
                    str(row["source_tier"]) == "current_canonical" for row in runs
                )
                result["fallback_run_count"] = sum(
                    str(row["source_tier"]) == "legacy_fallback" for row in runs
                )
            else:
                result["mean_id_quality"] = sum(
                    float(row[f"id_q_{minute:04d}"]) for row in runs
                ) / len(runs)
                result["mean_ood_quality"] = sum(
                    float(row[f"ood_q_{minute:04d}"]) for row in runs
                ) / len(runs)
                result["base_run_count"] = sum(
                    str(row["source_tier"]) == "noise_base" for row in runs
                )
                result["targeted_overlay_run_count"] = sum(
                    str(row["source_tier"]) == "targeted_overlay" for row in runs
                )
            output.append(result)
    return output


def _validate_reverse_aggregation(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_paths: Mapping[str, Path],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    numeric_fields = {
        "clean": ("mean_quality", "mean_relative_progress", "cumulative_eff_score"),
        "noise001": (
            "mean_id_quality",
            "mean_ood_quality",
            "mean_quality",
            "mean_relative_progress",
            "cumulative_eff_score",
        ),
        "noise005": (
            "mean_id_quality",
            "mean_ood_quality",
            "mean_quality",
            "mean_relative_progress",
            "cumulative_eff_score",
        ),
    }
    integer_fields = {
        "clean": ("current_path_run_count", "fallback_run_count"),
        "noise001": ("base_run_count", "targeted_overlay_run_count"),
        "noise005": ("base_run_count", "targeted_overlay_run_count"),
    }
    for condition in CONDITIONS:
        path = reference_paths[condition].resolve()
        reference = _read_csv(path)
        actual = reverse_aggregate_rows(rows_by_condition[condition], condition=condition)
        reference_by_key = {
            (str(row["algorithm"]), int(row["minute"])): row for row in reference
        }
        actual_by_key = {
            (str(row["algorithm"]), int(row["minute"])): row for row in actual
        }
        if set(reference_by_key) != set(actual_by_key):
            raise RawTrajectoryExportError(
                f"{condition} 反向聚合行键无法复现参考 eff_180min.csv"
            )
        maximum_delta = {field: 0.0 for field in numeric_fields[condition]}
        for key, actual_row in actual_by_key.items():
            reference_row = reference_by_key[key]
            for field in numeric_fields[condition]:
                delta = abs(float(actual_row[field]) - float(reference_row[field]))
                maximum_delta[field] = max(maximum_delta[field], delta)
                if delta > 1.0e-12:
                    raise RawTrajectoryExportError(
                        f"{condition}/{key}/{field} 无法反向复现，差值 {delta}"
                    )
            for field in integer_fields[condition]:
                if int(actual_row[field]) != int(reference_row[field]):
                    raise RawTrajectoryExportError(
                        f"{condition}/{key}/{field} 无法反向复现"
                    )
        reports[condition] = {
            "status": "exact_within_tolerance",
            "absolute_tolerance": 1.0e-12,
            "reference_path": str(path),
            "reference_sha256": _sha256_file(path),
            "reference_row_count": len(reference),
            "maximum_absolute_delta": maximum_delta,
        }
    return reports


def _write_csv_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RawTrajectoryExportError(f"拒绝写出空表: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=list(rows[0]),
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _sorted_paths(paths: Iterable[Path]) -> list[Path]:
    resolved = sorted((Path(path).resolve() for path in paths), key=str)
    if not resolved:
        raise RawTrajectoryExportError("轨迹 bundle 路径列表为空")
    if len(set(resolved)) != len(resolved):
        raise RawTrajectoryExportError("轨迹 bundle 路径列表存在重复")
    return resolved


def export_raw_run_trajectories(
    *,
    current_clean_path: Path,
    legacy_clean_path: Path,
    noise_base_paths: Iterable[Path],
    noise_overlay_paths: Iterable[Path],
    output_dir: Path,
    contract: GridContract = GridContract(),
    reference_eff_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """合并冻结输入并导出三张逐 run 宽表与可审计 manifest。"""

    current_clean_path = current_clean_path.resolve()
    legacy_clean_path = legacy_clean_path.resolve()
    base_paths = _sorted_paths(noise_base_paths)
    overlay_paths = _sorted_paths(noise_overlay_paths)
    overlap = set(base_paths) & set(overlay_paths)
    if overlap:
        raise RawTrajectoryExportError(f"noise base/overlay 路径重叠: {sorted(overlap)}")
    input_artifacts: dict[Path, dict[str, Any]] = {
        path: _input_artifact(path)
        for path in [current_clean_path, legacy_clean_path, *base_paths, *overlay_paths]
    }

    current_rows = _read_csv(current_clean_path)
    legacy_rows = _read_csv(legacy_clean_path)
    input_artifacts[current_clean_path]["record_count"] = len(current_rows)
    input_artifacts[legacy_clean_path]["record_count"] = len(legacy_rows)
    current = {str(row.get("logical_key")): row for row in current_rows}
    legacy = {str(row.get("logical_key")): row for row in legacy_rows}
    if len(current) != len(current_rows) or len(legacy) != len(legacy_rows):
        raise RawTrajectoryExportError("clean current/legacy 存在重复 logical_key")
    if len(legacy) != contract.expected_runs_per_condition:
        raise RawTrajectoryExportError(
            f"clean legacy 应有 {contract.expected_runs_per_condition} 行，实际 {len(legacy)}"
        )
    extra_current = set(current) - set(legacy)
    if extra_current:
        raise RawTrajectoryExportError(
            f"clean current 含 legacy 网格外记录: {sorted(extra_current)[:5]}"
        )

    clean_rows: list[dict[str, Any]] = []
    for key in sorted(legacy):
        use_current = key in current
        source_path = current_clean_path if use_current else legacy_clean_path
        clean_rows.append(
            _clean_output_row(
                current[key] if use_current else legacy[key],
                source_tier="current_canonical" if use_current else "legacy_fallback",
                source_input=source_path,
                source_input_sha256=str(input_artifacts[source_path]["sha256"]),
            )
        )

    overlay: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in overlay_paths:
        for record in _iter_jsonl(path):
            input_artifacts[path]["record_count"] += 1
            key = _logical_key(record["source"])
            if key in overlay:
                raise RawTrajectoryExportError(f"noise overlay 重复 logical_key: {key}")
            overlay[key] = (record, path)

    base_keys: set[str] = set()
    noise_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in base_paths:
        for record in _iter_jsonl(path):
            input_artifacts[path]["record_count"] += 1
            key = _logical_key(record["source"])
            if key in base_keys:
                raise RawTrajectoryExportError(f"noise base 重复 logical_key: {key}")
            base_keys.add(key)
            if key in overlay:
                continue
            condition = str(record["source"].get("noise_tag"))
            noise_rows[condition].append(
                _noise_output_row(
                    record,
                    source_tier="noise_base",
                    source_input=path,
                    source_input_sha256=str(input_artifacts[path]["sha256"]),
                )
            )
    overlay_outside_base = set(overlay) - base_keys
    if overlay_outside_base:
        raise RawTrajectoryExportError(
            f"noise targeted overlay 含 base 网格外记录: {sorted(overlay_outside_base)[:5]}"
        )
    for key in sorted(overlay):
        record, path = overlay[key]
        condition = str(record["source"].get("noise_tag"))
        noise_rows[condition].append(
            _noise_output_row(
                record,
                source_tier="targeted_overlay",
                source_input=path,
                source_input_sha256=str(input_artifacts[path]["sha256"]),
            )
        )

    rows_by_condition = {
        "clean": clean_rows,
        "noise001": noise_rows.get("noise001", []),
        "noise005": noise_rows.get("noise005", []),
    }
    unexpected_conditions = set(noise_rows) - {"noise001", "noise005"}
    if unexpected_conditions:
        raise RawTrajectoryExportError(
            f"noise bundle 含非预期 condition: {sorted(unexpected_conditions)}"
        )

    coverage = {
        condition: _coverage(rows, condition, contract)
        for condition, rows in rows_by_condition.items()
    }
    if reference_eff_paths is None:
        reverse_validation: dict[str, Any] = {"status": "not_requested"}
    else:
        if set(reference_eff_paths) != set(CONDITIONS):
            raise RawTrajectoryExportError(
                "reference_eff_paths 必须同时提供 clean/noise001/noise005"
            )
        for path in reference_eff_paths.values():
            _check_forbidden(path.resolve(), context="参考聚合表路径")
        reverse_validation = {
            "status": "passed",
            "conditions": _validate_reverse_aggregation(
                rows_by_condition, reference_eff_paths
            ),
        }
    outputs: dict[str, Any] = {}
    output_dir = output_dir.resolve()
    for condition in CONDITIONS:
        rows = sorted(
            rows_by_condition[condition],
            key=lambda row: (
                str(row["algorithm"]),
                str(row["dataset_id"]),
                int(row["seed"]),
                str(row["logical_key"]),
            ),
        )
        path = output_dir / f"{condition}_run_trajectories_180min.csv.gz"
        _write_csv_gz(path, rows)
        outputs[condition] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "record_count": len(rows),
            "column_count": len(rows[0]),
        }

    manifest = {
        "schema_version": "stage5.raw_run_trajectories.v1",
        "status": "ok",
        "horizon_minutes": HORIZON,
        "forbidden_source_check": {
            "tokens": list(FORBIDDEN_SOURCE_TOKENS),
            "checked_input_paths": True,
            "checked_clean_rows": True,
            "checked_noise_source_objects": True,
            "passed": True,
        },
        "contracts": {
            "clean_trajectory_basis": CLEAN_BASIS,
            "noise_trajectory_basis": NOISE_BASIS,
            "noise_id_ood_quality_exported": True,
            "clean_merge_precedence": "current_canonical_then_explicit_legacy_fallback",
            "noise_merge_precedence": "targeted_overlay_then_noise_base",
        },
        "conditions": coverage,
        "reverse_aggregation_validation": reverse_validation,
        "inputs": [input_artifacts[path] for path in sorted(input_artifacts, key=str)],
        "outputs": outputs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    stage_root = repo_root / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="导出 Stage5 逐 run 的 180 分钟原始轨迹")
    parser.add_argument("--stage-root", type=Path, default=stage_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=stage_root / "exports/raw_run_trajectories_180min",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    stage_root = args.stage_root.resolve()
    try:
        manifest = export_raw_run_trajectories(
            current_clean_path=(
                stage_root / "work/result_summary_20260905/clean_eff_run_metrics.csv"
            ),
            legacy_clean_path=stage_root / "results/clean_eff_run_metrics.csv",
            noise_base_paths=sorted(
                (stage_root / "work/noise_trajectory_freeze_v1/collected").glob("*.jsonl.gz")
            ),
            noise_overlay_paths=sorted(
                (stage_root / "work/remote_collect/all_conditions_cpu_v2/collected").glob(
                    "*.jsonl.gz"
                )
            ),
            output_dir=args.output_dir,
            reference_eff_paths={
                condition: stage_root
                / f"reports/result_summary_20260905/{condition}_eff_180min.csv"
                for condition in CONDITIONS
            },
        )
    except RawTrajectoryExportError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
