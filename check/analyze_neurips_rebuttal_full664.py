#!/usr/bin/env python3
"""汇总 NeurIPS rebuttal Full664 新三算法，并与 Stage3 四算法合并。

正式榜单沿用 Core50 口径：NMSE 先取 log10、截断到 [-12, 12]，
缺失值在算法均值中按 +12 计。脚本始终保留 manifest 的完整期望网格，
避免仅统计成功任务造成幸存者偏差。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "rebuttal"
    / "01_new3algs_full664_3seeds_clean_1h"
)
DEFAULT_STAGE3_RUN_LEVEL = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage3_664dats_4probes_3seeds_1h"
    / "probe4_postprocess_run_level.csv"
)
DEFAULT_STAGE3_RAW_DIGEST = (
    REPO_ROOT
    / "A_Neurips_experiments"
    / "stage3_664dats_4probes_3seeds_1h"
    / "probe4_current_run_level_raw_digest_7968.csv"
)

NEW3_ALGORITHMS = ("fepysr", "jaxsr", "symbolfit")
STAGE3_ALGORITHMS = ("dso", "imcts", "pyoperon", "udsr")
ALGORITHM_DISPLAY = {
    "dso": "DSO",
    "fepysr": "FePySR",
    "imcts": "iMCTS",
    "jaxsr": "JAXSR",
    "pyoperon": "PyOperon",
    "symbolfit": "SymbolFit",
    "udsr": "uDSR",
}
LOG_FLOOR = 1e-12
LOG_MIN = -12.0
LOG_MAX = 12.0
MISSING_LOG_PENALTY = 12.0

RUN_LEVEL_FIELDS = [
    "task_id",
    "algorithm",
    "gid",
    "dataset_id",
    "global_index",
    "dataset",
    "dataset_name",
    "family",
    "subgroup",
    "dataset_rel",
    "seed",
    "noise_tag",
    "status",
    "result_present",
    "finished",
    "identity_match",
    "valid_output",
    "metric_complete",
    "metric_complete_raw",
    "has_expression",
    "id_test_nmse",
    "ood_test_nmse",
    "id_log_nmse",
    "ood_log_nmse",
    "id_log_nmse_penalized",
    "ood_log_nmse_penalized",
    "seconds",
    "complexity",
    "tree_depth",
    "equation",
    "expression_canonical",
    "result_path",
    "experiment_dir",
    "source_group",
    "run_outcome_class",
    "failure_reason",
]


def _repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"分析审计路径位于仓库根目录之外: {resolved}"
        ) from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "match"}


def _float(value: Any, *, nonnegative: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _clip_log(value: float) -> float:
    return min(LOG_MAX, max(LOG_MIN, value))


def _nmse_log(value: float | None) -> float | None:
    if value is None:
        return None
    return _clip_log(math.log10(max(value, LOG_FLOOR)))


def _split_nmse(payload: dict[str, Any], split: str) -> float | None:
    block = payload.get(split)
    if not isinstance(block, dict):
        return None
    return _float(block.get("nmse"), nonnegative=True)


def _metric_complete(payload: dict[str, Any]) -> bool:
    return (
        _split_nmse(payload, "id_test") is not None
        and _split_nmse(payload, "ood_test") is not None
    )


def _has_expression(payload: dict[str, Any]) -> bool:
    return bool(_text(payload.get("equation")))


def _valid_output(payload: dict[str, Any]) -> bool:
    artifact = payload.get("canonical_artifact")
    artifact_valid = None
    if isinstance(artifact, dict):
        artifact_valid = artifact.get("artifact_valid")
    return _has_expression(payload) and artifact_valid is not False and _metric_complete(payload)


def _canonical_expression(payload: dict[str, Any]) -> str:
    artifact = payload.get("canonical_artifact")
    if isinstance(artifact, dict):
        for key in (
            "instantiated_expression",
            "normalized_expression",
            "sympy_expression",
            "return_expression_source",
        ):
            value = _text(artifact.get(key))
            if value:
                return value
    return _text(payload.get("equation"))


def _artifact_number(payload: dict[str, Any], key: str) -> int | float | None:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, dict):
        return None
    number = _float(artifact.get(key), nonnegative=True)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


def _normalize_dataset_rel(value: Any) -> str:
    text = _text(value).replace("\\", "/").rstrip("/")
    marker = "sim-datasets-data/"
    position = text.find(marker)
    return text[position:] if position >= 0 else text


def _identity_match(task: dict[str, str], payload: dict[str, Any]) -> bool:
    if _text(payload.get("tool")).lower() != _text(task.get("algorithm")).lower():
        return False
    if _int(payload.get("seed")) != _int(task.get("seed")):
        return False
    if _int(payload.get("task_global_index")) != _int(task.get("global_index")):
        return False
    identity = payload.get("dataset_identity_check")
    if isinstance(identity, dict) and "match" in identity:
        return _bool(identity.get("match"))
    actual_rel = _normalize_dataset_rel(payload.get("dataset_dir"))
    expected_rel = _normalize_dataset_rel(task.get("dataset_rel"))
    return bool(actual_rel and expected_rel and actual_rel == expected_rel)


def _new3_outcome(
    *,
    present: bool,
    identity_match: bool | None,
    status: str,
    has_expression: bool,
    valid_output: bool,
    metric_complete: bool,
) -> tuple[str, str]:
    if not present:
        return "not_finished", "missing_result"
    if identity_match is False:
        return "wrong_dataset", "dataset_identity_mismatch"
    if valid_output and metric_complete:
        return "valid_finite_result", "none"
    if has_expression and not metric_complete:
        return "partial_output", "metric_incomplete"
    if "timeout" in status:
        return "timeout_no_output", "timeout"
    if not has_expression:
        return "no_output", "no_expression"
    return "invalid_output", "invalid_expression_or_metrics"


def _validate_new3_manifest(tasks: list[dict[str, str]]) -> None:
    if not tasks:
        raise ValueError("新三算法 manifest 为空")
    required = {
        "task_id",
        "algorithm",
        "dataset_id",
        "global_index",
        "dataset_name",
        "dataset_rel",
        "seed",
        "noise_tag",
    }
    missing_columns = required - set(tasks[0])
    if missing_columns:
        raise ValueError(f"manifest 缺少字段: {sorted(missing_columns)}")

    task_ids = [_text(row.get("task_id")) for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest 存在重复 task_id")
    keys = [
        (
            _text(row.get("algorithm")).lower(),
            _text(row.get("dataset_id")),
            _int(row.get("seed")),
        )
        for row in tasks
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest 存在重复 algorithm/dataset_id/seed")
    algorithms = {key[0] for key in keys}
    if algorithms != set(NEW3_ALGORITHMS):
        raise ValueError(
            f"manifest 算法集合错误: actual={sorted(algorithms)}, "
            f"expected={list(NEW3_ALGORITHMS)}"
        )
    if {_text(row.get("noise_tag")).lower() for row in tasks} != {"clean"}:
        raise ValueError("rebuttal manifest 必须只包含 clean 任务")

    grids: dict[str, set[tuple[str, int | None]]] = defaultdict(set)
    for algorithm, dataset_id, seed in keys:
        grids[algorithm].add((dataset_id, seed))
    reference = grids[NEW3_ALGORITHMS[0]]
    for algorithm in NEW3_ALGORITHMS[1:]:
        if grids[algorithm] != reference:
            raise ValueError(f"新三算法键空间不一致: {algorithm}")


def build_new3_run_rows(
    tasks_path: Path,
    runs_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 manifest 完整网格读取结果；缺失结果仍输出一行。"""
    tasks = _read_csv(tasks_path)
    _validate_new3_manifest(tasks)
    datasets_path = tasks_path.parent / "datasets.csv"
    dataset_metadata: dict[str, dict[str, str]] = {}
    if datasets_path.is_file():
        for dataset in _read_csv(datasets_path):
            dataset_id = _text(dataset.get("dataset_id"))
            if not dataset_id or dataset_id in dataset_metadata:
                raise ValueError(
                    f"datasets.csv 存在空或重复 dataset_id: {dataset_id!r}"
                )
            dataset_metadata[dataset_id] = dataset
    rows: list[dict[str, Any]] = []
    unreadable_results = 0
    identity_mismatches = 0

    for task in tasks:
        algorithm = _text(task.get("algorithm")).lower()
        dataset_id = _text(task.get("dataset_id"))
        metadata = dataset_metadata.get(dataset_id, {})
        seed = _int(task.get("seed"))
        result_path = (
            runs_dir
            / algorithm
            / f"seed{seed}"
            / "clean"
            / dataset_id
            / "result.json"
        )
        payload: dict[str, Any] = {}
        present = result_path.is_file()
        readable = False
        if present:
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
                    readable = True
                else:
                    unreadable_results += 1
            except (OSError, json.JSONDecodeError):
                unreadable_results += 1

        identity: bool | None = _identity_match(task, payload) if readable else None
        if identity is False:
            identity_mismatches += 1
        metric_complete_raw = _metric_complete(payload) if readable else False
        valid_output_raw = _valid_output(payload) if readable else False
        metric_complete = bool(metric_complete_raw and identity is True)
        valid_output = bool(valid_output_raw and identity is True)
        has_expression = _has_expression(payload) if readable else False
        id_nmse = _split_nmse(payload, "id_test") if readable and identity is True else None
        ood_nmse = _split_nmse(payload, "ood_test") if readable and identity is True else None
        id_log = _nmse_log(id_nmse)
        ood_log = _nmse_log(ood_nmse)
        status = _text(payload.get("status")).lower() if readable else "missing"
        outcome, failure = _new3_outcome(
            present=readable,
            identity_match=identity,
            status=status,
            has_expression=has_expression,
            valid_output=valid_output,
            metric_complete=metric_complete,
        )
        seconds = None
        if readable:
            seconds = _float(payload.get("runtime_seconds"), nonnegative=True)
            if seconds is None:
                seconds = _float(payload.get("seconds"), nonnegative=True)

        rows.append(
            {
                "task_id": _text(task.get("task_id")),
                "algorithm": algorithm,
                "gid": dataset_id,
                "dataset_id": dataset_id,
                "global_index": _int(task.get("global_index")),
                "dataset": _text(task.get("dataset_name")),
                "dataset_name": _text(task.get("dataset_name")),
                "family": _text(task.get("family") or metadata.get("family")),
                "subgroup": _text(
                    task.get("subgroup") or metadata.get("subgroup")
                ),
                "dataset_rel": _text(task.get("dataset_rel")),
                "seed": seed,
                "noise_tag": "clean",
                "status": status,
                "result_present": readable,
                "finished": readable,
                "identity_match": identity,
                "valid_output": valid_output,
                "metric_complete": metric_complete,
                "metric_complete_raw": metric_complete_raw,
                "has_expression": has_expression,
                "id_test_nmse": id_nmse,
                "ood_test_nmse": ood_nmse,
                "id_log_nmse": id_log,
                "ood_log_nmse": ood_log,
                "id_log_nmse_penalized": (
                    id_log if id_log is not None else MISSING_LOG_PENALTY
                ),
                "ood_log_nmse_penalized": (
                    ood_log if ood_log is not None else MISSING_LOG_PENALTY
                ),
                "seconds": seconds,
                "complexity": _artifact_number(payload, "ast_node_count"),
                "tree_depth": _artifact_number(payload, "tree_depth"),
                "equation": _text(payload.get("equation")),
                "expression_canonical": _canonical_expression(payload),
                "result_path": str(result_path.resolve()) if readable else "",
                "experiment_dir": _text(payload.get("experiment_dir")),
                "source_group": "rebuttal_new3",
                "run_outcome_class": outcome,
                "failure_reason": failure,
            }
        )

    diagnostics = {
        "expected_tasks": len(tasks),
        "present_results": sum(bool(row["result_present"]) for row in rows),
        "missing_results": sum(not bool(row["result_present"]) for row in rows),
        "unreadable_results": unreadable_results,
        "identity_mismatches": identity_mismatches,
        "valid_outputs": sum(bool(row["valid_output"]) for row in rows),
        "metric_complete": sum(bool(row["metric_complete"]) for row in rows),
    }
    return rows, diagnostics


def _stage3_log(row: dict[str, str], key: str) -> float | None:
    value = _float(row.get(key))
    return _clip_log(value) if value is not None else None


def _load_stage3_rows(
    path: Path,
    expected_dataset_seeds: set[tuple[str, int]],
    dataset_metadata: dict[str, dict[str, Any]],
    raw_digest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = _read_csv(path)
    actual_keys: list[tuple[str, str, int | None]] = []
    for row in source_rows:
        actual_keys.append(
            (
                _text(row.get("method_norm")).lower(),
                _text(row.get("dataset_id")),
                _int(row.get("seed_norm")),
            )
        )
    duplicate_count = len(actual_keys) - len(set(actual_keys))
    expected_keys = {
        (algorithm, dataset_id, seed)
        for algorithm in STAGE3_ALGORITHMS
        for dataset_id, seed in expected_dataset_seeds
    }
    actual_key_set = set(actual_keys)
    if duplicate_count or actual_key_set != expected_keys:
        missing = sorted(expected_keys - actual_key_set)[:5]
        extra = sorted(actual_key_set - expected_keys)[:5]
        raise ValueError(
            "Stage3 键空间与 rebuttal manifest 不一致: "
            f"rows={len(source_rows)}, unique={len(actual_key_set)}, "
            f"expected={len(expected_keys)}, duplicates={duplicate_count}, "
            f"missing_sample={missing}, extra_sample={extra}"
        )

    raw_by_key: dict[tuple[str, str, int | None], dict[str, str]] = {}
    if raw_digest_path is not None:
        raw_rows = _read_csv(raw_digest_path)
        for raw in raw_rows:
            key = (
                _text(raw.get("method") or raw.get("algorithm")).lower(),
                _text(raw.get("dataset_id")),
                _int(raw.get("seed")),
            )
            if key in raw_by_key:
                raise ValueError(f"Stage3 raw digest 存在重复键: {key}")
            raw_by_key[key] = raw
        raw_key_set = set(raw_by_key)
        if raw_key_set != expected_keys:
            missing = sorted(expected_keys - raw_key_set)[:5]
            extra = sorted(raw_key_set - expected_keys)[:5]
            raise ValueError(
                "Stage3 raw digest 键空间不一致: "
                f"rows={len(raw_rows)}, expected={len(expected_keys)}, "
                f"missing_sample={missing}, extra_sample={extra}"
            )

    out: list[dict[str, Any]] = []
    for row in source_rows:
        algorithm = _text(row.get("method_norm")).lower()
        dataset_id = _text(row.get("dataset_id"))
        seed = _int(row.get("seed_norm"))
        raw = raw_by_key.get((algorithm, dataset_id, seed), {})
        metadata = dataset_metadata[dataset_id]
        identity = not _bool(row.get("wrong_dataset_flag"))
        present = not _bool(row.get("synthetic_missing_row"))
        finished = _bool(row.get("is_finished_run"))
        id_log = _stage3_log(row, "id_log_nmse_used") if identity else None
        ood_log = _stage3_log(row, "ood_log_nmse_used") if identity else None
        raw_metric_text = _text(row.get("result_metric_complete_raw"))
        metric_complete_raw = (
            _bool(raw_metric_text)
            if raw_metric_text
            else id_log is not None and ood_log is not None
        )
        raw_valid_text = _text(row.get("result_valid_output_raw"))
        valid_output_raw = (
            _bool(raw_valid_text)
            if raw_valid_text
            else _text(row.get("run_outcome_class"))
            in {"valid_finite_result", "valid_extreme_error"}
        )
        metric_complete = bool(metric_complete_raw and identity)
        valid_output = bool(valid_output_raw and identity)
        equation = _text(
            raw.get("result_equation") or row.get("result_equation")
        )
        expression_canonical = _text(
            raw.get("result_instantiated_expression")
            or raw.get("result_normalized_expression")
            or equation
        )
        out.append(
            {
                "task_id": f"{algorithm}__seed{seed}__clean__{dataset_id}",
                "algorithm": algorithm,
                "gid": dataset_id,
                "dataset_id": dataset_id,
                "global_index": _int(metadata.get("global_index")),
                "dataset": _text(metadata.get("dataset_name")),
                "dataset_name": _text(metadata.get("dataset_name")),
                "family": _text(metadata.get("family")),
                "subgroup": _text(metadata.get("subgroup")),
                "dataset_rel": _text(metadata.get("dataset_rel")),
                "seed": seed,
                "noise_tag": "clean",
                "status": _text(row.get("result_status_raw"))
                or _text(row.get("run_outcome_class")),
                "result_present": present,
                "finished": finished,
                "identity_match": identity,
                "valid_output": valid_output,
                "metric_complete": metric_complete,
                "metric_complete_raw": metric_complete_raw,
                "has_expression": _bool(row.get("result_has_expression_raw"))
                if _text(row.get("result_has_expression_raw"))
                else bool(expression_canonical),
                "id_test_nmse": None,
                "ood_test_nmse": None,
                "id_log_nmse": id_log,
                "ood_log_nmse": ood_log,
                "id_log_nmse_penalized": (
                    id_log if id_log is not None else MISSING_LOG_PENALTY
                ),
                "ood_log_nmse_penalized": (
                    ood_log if ood_log is not None else MISSING_LOG_PENALTY
                ),
                "seconds": _float(row.get("result_seconds"), nonnegative=True),
                "complexity": _float(
                    row.get("result_complexity"),
                    nonnegative=True,
                ),
                "tree_depth": _float(
                    row.get("result_tree_depth"),
                    nonnegative=True,
                ),
                "equation": equation,
                "expression_canonical": expression_canonical,
                "result_path": _text(raw.get("result_result_path")),
                "experiment_dir": "",
                "source_group": "stage3_probe4",
                "run_outcome_class": _text(row.get("run_outcome_class")),
                "failure_reason": _text(row.get("failure_reason_normalized")),
            }
        )

    diagnostics = {
        "path": _repo_relative(path, repo_root),
        "rows": len(out),
        "algorithms": len(STAGE3_ALGORITHMS),
        "expected_rows": len(expected_keys),
        "keyspace_valid": True,
        "raw_digest_path": (
            _repo_relative(raw_digest_path, repo_root)
            if raw_digest_path is not None
            else None
        ),
        "raw_digest_rows": len(raw_by_key),
    }
    return out, diagnostics


def _median(values: list[Any]) -> float | None:
    numbers = [
        number
        for value in values
        if (number := _float(value)) is not None
    ]
    return statistics.median(numbers) if numbers else None


def _mean(values: list[Any]) -> float:
    numbers = [float(value) for value in values]
    return statistics.mean(numbers) if numbers else math.nan


def summarize_algorithms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_text(row.get("algorithm")).lower()].append(row)

    summaries: list[dict[str, Any]] = []
    for algorithm, group in groups.items():
        expected = len(group)
        summaries.append(
            {
                "Algorithm key": algorithm,
                "Algorithm": ALGORITHM_DISPLAY.get(algorithm, algorithm),
                "Runs expected": expected,
                "Runs present": sum(bool(row["result_present"]) for row in group),
                "Runs finished": sum(bool(row["finished"]) for row in group),
                "Valid rate": sum(bool(row["valid_output"]) for row in group)
                / expected,
                "Metric complete rate": sum(
                    bool(row["metric_complete"]) for row in group
                )
                / expected,
                "Mean ID log NMSE": _mean(
                    [row["id_log_nmse_penalized"] for row in group]
                ),
                "Mean OOD log NMSE": _mean(
                    [row["ood_log_nmse_penalized"] for row in group]
                ),
                "Median ID log NMSE": _median(
                    [row["id_log_nmse"] for row in group]
                ),
                "Median OOD log NMSE": _median(
                    [row["ood_log_nmse"] for row in group]
                ),
                "Median seconds": _median([row["seconds"] for row in group]),
                "Median complexity": _median(
                    [row["complexity"] for row in group]
                ),
                "Median tree depth": _median(
                    [row["tree_depth"] for row in group]
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            float(row["Mean OOD log NMSE"]),
            _text(row["Algorithm key"]),
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["Rank"] = rank
    fields = ["Rank"] + [key for key in summaries[0] if key != "Rank"] if summaries else []
    return [{key: row.get(key) for key in fields} for row in summaries]


def _load_audit_gate(
    batch_dir: Path,
    expected_tasks: int,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    path = batch_dir / "audit" / "audit_gate_summary.json"
    logical_path = _repo_relative(path, repo_root)
    if not path.is_file():
        return {
            "path": logical_path,
            "valid": False,
            "reason": "missing_audit_gate",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "path": logical_path,
            "valid": False,
            "reason": "unreadable_audit_gate",
        }
    valid = (
        payload.get("audit_passed") is True
        and _int(payload.get("expected_total_tasks")) == expected_tasks
        and _int(payload.get("task_audit_rows")) == expected_tasks
        and _int(payload.get("failure_rows")) == 0
    )
    return {
        "path": logical_path,
        "valid": valid,
        "reason": "ok" if valid else "audit_gate_failed_or_count_mismatch",
        "payload": payload,
    }


def _write_analysis_readme(
    output_dir: Path,
    *,
    final_ready: bool,
    stage3_enabled: bool,
) -> None:
    status = "FINAL_READY" if final_ready else "INCOMPLETE_PREVIEW"
    comparison = (
        "`full664_7alg_run_level.csv` 与 `full664_7alg_leaderboard.csv`"
        if stage3_enabled
        else "本次未启用 Stage3 四算法合并"
    )
    text = f"""# NeurIPS rebuttal Full664 分析

- 状态：`{status}`
- `Valid rate`：有表达式、canonical artifact 未明确无效，且 ID/OOD NMSE 完整。
- `Metric complete rate`：ID/OOD NMSE 均为有限非负数。
- `Mean ID/OOD log NMSE`：`log10(max(NMSE, 1e-12))` 截断到 `[-12, 12]`；
  对各自缺失值按 `+12` 计入均值。
- 排名：按 penalized mean OOD log NMSE 升序。
- Stage3 对比：{comparison}。

`INCOMPLETE_PREVIEW` 只用于运行中巡检，不得作为论文最终结果引用。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def analyze_batch(
    *,
    batch_dir: Path,
    stage3_run_level: Path | None,
    output_dir: Path,
    allow_incomplete: bool,
    stage3_raw_digest: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """生成新三算法汇总，并可选严格合并 Stage3 四算法。"""
    repository_root = (repo_root or REPO_ROOT).resolve()
    tasks_path = batch_dir / "manifest" / "tasks.csv"
    new3_rows, new3_diagnostics = build_new3_run_rows(
        tasks_path,
        batch_dir / "runs",
    )
    task_rows = _read_csv(tasks_path)
    dataset_metadata: dict[str, dict[str, Any]] = {
        _text(row.get("dataset_id")): {
            "global_index": _int(row.get("global_index")),
            "dataset_name": _text(row.get("dataset_name")),
            "family": _text(row.get("family")),
            "subgroup": _text(row.get("subgroup")),
            "dataset_rel": _text(row.get("dataset_rel")),
        }
        for row in new3_rows
        if _text(row.get("algorithm")) == NEW3_ALGORITHMS[0]
    }
    expected_dataset_seeds: set[tuple[str, int]] = set()
    for task in task_rows:
        dataset_id = _text(task.get("dataset_id"))
        if _text(task.get("algorithm")).lower() == NEW3_ALGORITHMS[0]:
            seed = _int(task.get("seed"))
            if seed is None:
                raise ValueError(f"非法 seed: {task.get('seed')!r}")
            expected_dataset_seeds.add((dataset_id, seed))

    stage3_rows: list[dict[str, Any]] = []
    stage3_diagnostics: dict[str, Any] | None = None
    if stage3_run_level is not None:
        stage3_rows, stage3_diagnostics = _load_stage3_rows(
            stage3_run_level,
            expected_dataset_seeds,
            dataset_metadata,
            stage3_raw_digest,
            repository_root,
        )

    audit_gate = _load_audit_gate(
        batch_dir,
        new3_diagnostics["expected_tasks"],
        repository_root,
    )
    final_ready = (
        new3_diagnostics["missing_results"] == 0
        and new3_diagnostics["unreadable_results"] == 0
        and new3_diagnostics["identity_mismatches"] == 0
        and audit_gate["valid"]
    )
    if not allow_incomplete and not final_ready:
        raise RuntimeError(
            "rebuttal 批次不完整，拒绝生成最终榜单: "
            f"missing={new3_diagnostics['missing_results']}, "
            f"unreadable={new3_diagnostics['unreadable_results']}, "
            f"identity_mismatch={new3_diagnostics['identity_mismatches']}, "
            f"audit={audit_gate['reason']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "new3_run_level.csv",
        new3_rows,
        RUN_LEVEL_FIELDS,
    )
    new3_summary = summarize_algorithms(new3_rows)
    _write_csv(
        output_dir / "new3_algorithm_summary.csv",
        new3_summary,
    )

    if stage3_rows:
        combined = sorted(
            [*stage3_rows, *new3_rows],
            key=lambda row: (
                _text(row.get("algorithm")),
                _text(row.get("dataset_id")),
                _int(row.get("seed")) or -1,
            ),
        )
        leaderboard = summarize_algorithms(combined)
        _write_csv(
            output_dir / "full664_7alg_run_level.csv",
            combined,
            RUN_LEVEL_FIELDS,
        )
        _write_csv(
            output_dir / "full664_7alg_leaderboard.csv",
            leaderboard,
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "path_base": "repository_root",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_dir": _repo_relative(batch_dir, repository_root),
        "allow_incomplete": allow_incomplete,
        "final_ready": final_ready,
        "new3": new3_diagnostics,
        "audit_gate": audit_gate,
        "stage3": stage3_diagnostics,
        "outputs": {
            "new3_run_level": _repo_relative(
                output_dir / "new3_run_level.csv",
                repository_root,
            ),
            "new3_algorithm_summary": _repo_relative(
                output_dir / "new3_algorithm_summary.csv",
                repository_root,
            ),
            "full664_7alg_run_level": (
                _repo_relative(
                    output_dir / "full664_7alg_run_level.csv",
                    repository_root,
                )
                if stage3_rows
                else None
            ),
            "full664_7alg_leaderboard": (
                _repo_relative(
                    output_dir / "full664_7alg_leaderboard.csv",
                    repository_root,
                )
                if stage3_rows
                else None
            ),
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_analysis_readme(
        output_dir,
        final_ready=final_ready,
        stage3_enabled=bool(stage3_rows),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument(
        "--stage3-run-level",
        type=Path,
        default=DEFAULT_STAGE3_RUN_LEVEL,
    )
    parser.add_argument(
        "--stage3-raw-digest",
        type=Path,
        default=DEFAULT_STAGE3_RAW_DIGEST,
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="只输出新三算法汇总，不合并 Stage3 四算法",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许运行中生成预览；输出不得作为最终论文结果",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else batch_dir / "analysis"
    )
    stage3 = None if args.skip_stage3 else args.stage3_run_level.resolve()
    stage3_raw = (
        None if args.skip_stage3 else args.stage3_raw_digest.resolve()
    )
    summary = analyze_batch(
        batch_dir=batch_dir,
        stage3_run_level=stage3,
        output_dir=output_dir,
        allow_incomplete=args.allow_incomplete,
        stage3_raw_digest=stage3_raw,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
