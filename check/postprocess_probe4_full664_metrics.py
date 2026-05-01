#!/usr/bin/env python3
"""把 Probe4 Full664 run-level 结果转换成 Core-50 选择所需指标。

本脚本只做后处理与诊断，不选择 Core-50。核心约束是：
unfinished run 只能记为 not_finished，不能记为 invalid。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE4_ROOT = REPO_ROOT / "exp-planning" / "02.E1选择验证" / "generated" / "probe4_full664_v1"
DEFAULT_DATASET_CATALOG = PROBE4_ROOT / "full664_unified.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exp-planning" / "03.四探针全量664三种子验证" / "generated"
DEFAULT_METHODS = ("dso", "imcts", "pyoperon", "udsr")
DEFAULT_SEEDS = (520, 521, 522)

LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12.0
LOG_CLIP_MAX = 12.0
SOFTMAX_TEMPERATURE = 1.0

FINISHED_STATUS = {
    "done",
    "completed",
    "complete",
    "success",
    "succeeded",
    "ok",
    "failed",
    "failure",
    "error",
    "errored",
    "timeout",
    "timed_out",
    "cancelled",
    "canceled",
    "no_valid_output",
}
UNFINISHED_STATUS = {
    "",
    "nan",
    "none",
    "pending",
    "running",
    "queued",
    "not_started",
    "submitted",
    "empty",
}
METHOD_ALIASES = {
    "dso": "dso",
    "imcts": "imcts",
    "imcts_wrapper": "imcts",
    "iMCTS": "imcts",
    "pyoperon": "pyoperon",
    "PyOperon": "pyoperon",
    "udsr": "udsr",
    "uDSR": "udsr",
}


def _default_input() -> Path:
    latest = PROBE4_ROOT / "current_digest_latest.txt"
    if latest.exists():
        text = latest.read_text(encoding="utf-8").strip()
        if text:
            path = Path(text)
            if (path / "probe4_current_run_level.csv").exists():
                return path / "probe4_current_run_level.csv"
    return PROBE4_ROOT / "current_digest_20260501-021406" / "probe4_current_run_level.csv"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    return text in {"1", "true", "yes", "y", "match"}


def _false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    text = _clean_text(value).lower()
    return text in {"0", "false", "no", "n", "mismatch", "wrong_dataset_collision"}


def _float(value: Any, *, finite_only: bool = True) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    if finite_only and not math.isfinite(out):
        return None
    return out


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _method_norm(value: Any) -> str:
    text = _clean_text(value)
    return METHOD_ALIASES.get(text, METHOD_ALIASES.get(text.lower(), text.lower()))


def _seed_norm(value: Any) -> int | None:
    text = _clean_text(value).replace("seed", "")
    return _int(text)


def _dataset_id_from_index(value: Any) -> str:
    idx = _int(value)
    return f"g{idx:04d}" if idx is not None else ""


def _dataset_key(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("dataset_id", "dataset_key", "dataset_name", "result_dataset"):
        value = _clean_text(row.get(key))
        if value:
            return value
    global_id = _dataset_id_from_index(row.get("global_index"))
    if global_id:
        return global_id
    return f"row_{fallback_index:06d}"


def _result_status(row: dict[str, Any]) -> str:
    for key in ("state", "status", "result_status", "task_status"):
        value = _clean_text(row.get(key)).lower()
        if value:
            return value
    return ""


def _is_finished(row: dict[str, Any]) -> bool:
    values = [
        _clean_text(row.get("state")).lower(),
        _clean_text(row.get("status")).lower(),
        _clean_text(row.get("result_status")).lower(),
        _clean_text(row.get("task_status")).lower(),
    ]
    if any(value in FINISHED_STATUS for value in values if value):
        return True
    if any(value in UNFINISHED_STATUS for value in values):
        return False
    # 有 result path 或完整指标时，保守认为已经进入终态。
    return bool(_clean_text(row.get("result_result_path")) or _bool(row.get("result_metric_complete")))


def _wrong_dataset(row: dict[str, Any]) -> bool:
    if "result_dataset_identity_match" in row and _false(row.get("result_dataset_identity_match")):
        return True
    if "dataset_identity_trusted" in row and _false(row.get("dataset_identity_trusted")):
        return True
    status = _clean_text(row.get("result_dataset_identity_status")).lower()
    if status and status not in {"match", "exact_match", "temp_copy_equivalent", "ok"}:
        return True
    return False


def _raw_nmse(row: dict[str, Any], split: str) -> float | None:
    keys = {
        "train": ("result_train_nmse", "train_nmse"),
        "valid": ("result_valid_nmse", "valid_nmse"),
        "id": ("result_id_test_nmse", "id_nmse", "result_id_nmse"),
        "ood": ("result_ood_test_nmse", "ood_nmse"),
    }[split]
    for key in keys:
        value = _float(row.get(key), finite_only=False)
        if value is not None:
            return value
    return None


def _log_nmse(row: dict[str, Any], split: str) -> float | None:
    preferred = {
        "train": ("train_log10_nmse_clip12", "train_log_nmse_used"),
        "valid": ("valid_log10_nmse_clip12", "valid_log_nmse_used"),
        "id": ("id_test_log10_nmse_clip12", "id_log_nmse_used"),
        "ood": ("ood_test_log10_nmse_clip12", "ood_log_nmse_used"),
    }[split]
    for key in preferred:
        value = _float(row.get(key))
        if value is not None:
            return _clip(value, LOG_CLIP_MIN, LOG_CLIP_MAX)

    raw = _raw_nmse(row, split)
    if raw is None or raw < 0 or not math.isfinite(raw):
        return None
    return _clip(math.log10(max(raw, LOG_FLOOR)), LOG_CLIP_MIN, LOG_CLIP_MAX)


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _metric_complete(row: dict[str, Any]) -> bool:
    if "result_metric_complete" in row:
        return _bool(row.get("result_metric_complete"))
    return all(_log_nmse(row, split) is not None for split in ("id", "ood"))


def _has_expression(row: dict[str, Any]) -> bool:
    if "result_has_expression" in row:
        return _bool(row.get("result_has_expression"))
    return bool(_clean_text(row.get("result_equation")) or _clean_text(row.get("expression")))


def _valid_output(row: dict[str, Any]) -> bool:
    if "result_valid_output" in row:
        return _bool(row.get("result_valid_output"))
    if "valid_output" in row:
        return _bool(row.get("valid_output"))
    return _has_expression(row) and _metric_complete(row)


def _runtime(row: dict[str, Any]) -> float | None:
    for key in ("result_seconds", "runtime", "seconds", "wall_time"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _r2(row: dict[str, Any], split: str) -> float | None:
    keys = {
        "id": ("result_id_test_r2", "id_r2"),
        "ood": ("result_ood_test_r2", "ood_r2"),
    }[split]
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _complexity(row: dict[str, Any]) -> float | None:
    for key in ("result_complexity", "complexity"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _tree_depth(row: dict[str, Any]) -> float | None:
    for key in ("result_tree_depth", "tree_depth"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _extreme_error(row: dict[str, Any], *, is_finished: bool) -> bool:
    if not is_finished or not _metric_complete(row) or not _valid_output(row):
        return False
    logs = [_log_nmse(row, "id"), _log_nmse(row, "ood")]
    if any(value is not None and math.isclose(value, LOG_CLIP_MAX) for value in logs):
        return True
    raws = [_raw_nmse(row, "id"), _raw_nmse(row, "ood")]
    return any(value is not None and math.isfinite(value) and value >= 10**LOG_CLIP_MAX for value in raws)


def _run_outcome(row: dict[str, Any], *, is_finished: bool, wrong_dataset: bool, extreme: bool) -> tuple[str, str]:
    if not is_finished:
        return "not_finished", "not_finished"
    if wrong_dataset:
        return "wrong_dataset", "dataset_identity_mismatch"

    status = _result_status(row)
    timeout_type = _clean_text(row.get("result_timeout_type") or row.get("timeout_type")).lower()
    error_text = _clean_text(row.get("result_error") or row.get("error") or row.get("state_error")).lower()
    no_valid_reason = _clean_text(row.get("result_no_valid_output_reason")).lower()
    has_expr = _has_expression(row)
    metrics = _metric_complete(row)
    valid = _valid_output(row)

    if valid and metrics:
        return ("valid_extreme_error" if extreme else "valid_finite_result"), "none"

    if "budget_exhausted_with_output" in timeout_type or timeout_type == "partial_output":
        return "partial_output", "budget_exhausted_with_output"
    if has_expr and not metrics:
        return "partial_output", "metric_incomplete"

    if "timeout" in status or "timed_out" in status or "timeout" in timeout_type:
        return "timeout_no_output", "timeout"
    if "budget_exhausted" in timeout_type:
        return "timeout_no_output", "budget_exhausted"
    if "not trained" in error_text or "not fitted" in error_text or "please call fit" in error_text or "模型尚未训练" in error_text:
        return "system_error", "not_trained"
    if "subprocess" in error_text or "dependency" in error_text or "modulenotfound" in error_text:
        return "system_error", "subprocess_error"
    if "nan" in error_text or "nan" in no_valid_reason:
        return "invalid_output", "nan_prediction"
    if "inf" in error_text or "overflow" in error_text or "inf" in no_valid_reason:
        return "invalid_output", "inf_prediction"
    if "invalid" in error_text or "parse" in error_text or "canonical" in error_text:
        return "invalid_output", "invalid_expression"
    if not has_expr:
        return "no_output", "no_expression"
    if not metrics:
        return "invalid_output", "metric_incomplete"
    if status in {"error", "failed", "failure"}:
        return "system_error", "unknown_error"
    return "unknown_failure", "unknown_error"


def _median(values: list[float | None]) -> float | None:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(nums) if nums else None


def _mean(values: list[float | None]) -> float | None:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(nums) if nums else None


def _pvariance(values: list[float | None]) -> float:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.pvariance(nums) if len(nums) >= 2 else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    nums = sorted(value for value in values if math.isfinite(value))
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    pos = (len(nums) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return nums[low]
    return nums[low] * (high - pos) + nums[high] * (pos - low)


def _iqr(values: list[float | None]) -> float | None:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    if not nums:
        return None
    q1 = _percentile(nums, 0.25)
    q3 = _percentile(nums, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def _mode(values: list[str]) -> str:
    if not values:
        return ""
    counter = Counter(values)
    max_count = max(counter.values())
    return sorted(key for key, count in counter.items() if count == max_count)[0]


def _entropy_binary(p: float) -> float:
    p = _clip(p, 0.0, 1.0)
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _softmax_entropy(scores: list[float], temperature: float = SOFTMAX_TEMPERATURE) -> float | None:
    nums = [value for value in scores if math.isfinite(value)]
    if not nums:
        return None
    logits = [-value / max(temperature, 1e-12) for value in nums]
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    probs = [value / total for value in exps if total > 0]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _robust_norm(values: list[float | None]) -> list[float]:
    nums = [value for value in values if value is not None and math.isfinite(value)]
    if not nums:
        return [0.0 for _ in values]
    lo = _percentile(nums, 0.05)
    hi = _percentile(nums, 0.95)
    if lo is None or hi is None or math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [
        _clip(((value - lo) / (hi - lo)) if value is not None and math.isfinite(value) else 0.0, 0.0, 1.0)
        for value in values
    ]


def _load_dataset_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    _, rows = _read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = _dataset_id_from_index(row.get("global_index")) or _clean_text(row.get("dataset_id"))
        if not dataset_id:
            continue
        item = dict(row)
        item["dataset_id"] = dataset_id
        item.setdefault("dataset_name", row.get("dataset_name", ""))
        item.setdefault("dataset_rel", row.get("dataset_rel") or row.get("dataset_dir") or "")
        out[dataset_id] = item
    return out


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset_id = _clean_text(row.get("dataset_id")) or _dataset_id_from_index(row.get("global_index"))
    return {
        "dataset_id": dataset_id,
        "global_index": _int(row.get("global_index")),
        "dataset_name": _clean_text(row.get("dataset_name") or row.get("result_dataset")),
        "family": _clean_text(row.get("family")),
        "subgroup": _clean_text(row.get("subgroup")),
        "metadata_class": _clean_text(row.get("metadata_class")),
        "dataset_rel": _clean_text(row.get("dataset_rel") or row.get("dataset_dir")),
        "basename": _clean_text(row.get("basename")),
        "formula_identity": _clean_text(row.get("formula_identity")),
        "feature_count": _int(row.get("metadata_feature_count") or row.get("feature_count")),
        "target_name": _clean_text(row.get("metadata_target_name") or row.get("target_name")),
        "train_samples": _int(row.get("metadata_train_samples") or row.get("train_samples")),
        "valid_samples": _int(row.get("metadata_valid_samples") or row.get("valid_samples")),
        "id_test_samples": _int(row.get("metadata_id_test_samples") or row.get("id_test_samples")),
        "ood_test_samples": _int(row.get("metadata_ood_test_samples") or row.get("ood_test_samples")),
        "formula_char_count": _int(row.get("formula_char_count")),
        "formula_operator_count": _int(row.get("formula_operator_count")),
        "formula_return_expr": _clean_text(row.get("formula_return_expr")),
    }


def _merge_metadata(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if out.get(key) in (None, "") and value not in (None, ""):
            out[key] = value
    return out


def _catalog_metadata(row: dict[str, Any]) -> dict[str, Any]:
    out = _metadata_from_row(row)
    if not out.get("dataset_rel"):
        out["dataset_rel"] = _clean_text(row.get("dataset_dir"))
    if not out.get("formula_identity"):
        family = _clean_text(row.get("family"))
        subgroup = _clean_text(row.get("subgroup"))
        basename = _clean_text(row.get("basename") or row.get("dataset_name"))
        out["formula_identity"] = "/".join(x for x in (family, subgroup, basename) if x)
    return out


def _complete_expected_grid(
    rows: list[dict[str, Any]],
    dataset_catalog: dict[str, dict[str, Any]],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_dataset_meta: dict[str, dict[str, Any]] = {}
    for dataset_id, item in dataset_catalog.items():
        by_dataset_meta[dataset_id] = _catalog_metadata(item)
    for row in rows:
        dataset_id = _clean_text(row.get("dataset_id")) or _dataset_id_from_index(row.get("global_index"))
        if dataset_id:
            by_dataset_meta[dataset_id] = _merge_metadata(by_dataset_meta.get(dataset_id, {}), _metadata_from_row(row))

    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicates: Counter[tuple[str, str, int]] = Counter()
    for idx, row in enumerate(rows):
        method = _method_norm(row.get("method") or row.get("algorithm") or row.get("tool"))
        seed = _seed_norm(row.get("seed"))
        dataset_id = _clean_text(row.get("dataset_id")) or _dataset_id_from_index(row.get("global_index"))
        if not dataset_id:
            dataset_id = _dataset_key(row, idx)
        if seed is None:
            continue
        key = (dataset_id, method, seed)
        if key in existing:
            duplicates[key] += 1
            existing[key] = _pick_better_row(existing[key], row)
        else:
            existing[key] = row

    completed: list[dict[str, Any]] = []
    for dataset_id in sorted(by_dataset_meta, key=lambda value: (_int(value[1:]) if value.startswith("g") else 10**9, value)):
        meta = by_dataset_meta[dataset_id]
        for method in methods:
            for seed in seeds:
                row = existing.get((dataset_id, method, seed))
                if row is None:
                    row = {
                        **meta,
                        "method": method,
                        "seed": seed,
                        "state": "not_started",
                        "result_status": "",
                        "_synthetic_missing_row": "1",
                    }
                else:
                    row = {**meta, **row, "_synthetic_missing_row": "0"}
                completed.append(row)
    return completed


def _pick_better_row(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, int, float]:
        finished = 1 if _is_finished(row) else 0
        valid = 1 if _valid_output(row) else 0
        seconds = _runtime(row) or 0.0
        return finished, valid, seconds

    return b if score(b) > score(a) else a


def _derive_run_rows(rows: list[dict[str, Any]], methods: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        method = _method_norm(row.get("method") or row.get("algorithm") or row.get("tool"))
        seed = _seed_norm(row.get("seed"))
        dataset_key = _dataset_key(row, idx)
        is_finished = _is_finished(row)
        wrong = _wrong_dataset(row)
        evaluable = is_finished and not wrong
        extreme = _extreme_error(row, is_finished=is_finished)
        outcome, failure = _run_outcome(row, is_finished=is_finished, wrong_dataset=wrong, extreme=extreme)

        train_log = _log_nmse(row, "train")
        valid_log = _log_nmse(row, "valid")
        id_log = _log_nmse(row, "id")
        ood_log = _log_nmse(row, "ood")
        meta = _metadata_from_row(row)
        dataset_id = meta["dataset_id"] or dataset_key
        out.append(
            {
                **meta,
                "dataset_key": dataset_key,
                "method_norm": method,
                "seed_norm": seed,
                "is_expected_method": method in methods,
                "is_expected_seed": seed in seeds,
                "is_finished_run": is_finished,
                "wrong_dataset_flag": wrong,
                "is_evaluable_run": evaluable,
                "not_finished_flag": not is_finished,
                "train_log_nmse_used": train_log,
                "valid_log_nmse_used": valid_log,
                "id_log_nmse_used": id_log,
                "ood_log_nmse_used": ood_log,
                "extreme_error_flag": extreme,
                "run_outcome_class": outcome,
                "failure_reason_normalized": failure,
                "result_status_raw": _clean_text(row.get("result_status")),
                "state_raw": _clean_text(row.get("state")),
                "timeout_type_raw": _clean_text(row.get("result_timeout_type") or row.get("timeout_type")),
                "result_error_raw": _clean_text(row.get("result_error") or row.get("error") or row.get("state_error")),
                "result_valid_output_raw": _clean_text(row.get("result_valid_output") or row.get("valid_output")),
                "result_metric_complete_raw": _clean_text(row.get("result_metric_complete") or row.get("has_full_metrics")),
                "result_has_expression_raw": _clean_text(row.get("result_has_expression") or row.get("has_expression")),
                "result_seconds": _runtime(row),
                "id_r2": _r2(row, "id"),
                "ood_r2": _r2(row, "ood"),
                "result_equation": _clean_text(row.get("result_equation") or row.get("expression")),
                "result_complexity": _complexity(row),
                "result_tree_depth": _tree_depth(row),
                "synthetic_missing_row": _bool(row.get("_synthetic_missing_row")),
            }
        )
    return out


def _valid_performance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"}]


def _derive_dataset_algorithm_rows(
    run_rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["dataset_id"], row["method_norm"])].append(row)

    out: list[dict[str, Any]] = []
    expected_seeds = len(seeds)
    for (dataset_id, method), group in sorted(groups.items()):
        first = group[0]
        perf = _valid_performance_rows(group)
        evaluable = [row for row in group if row["is_evaluable_run"]]
        n_finished = sum(row["is_finished_run"] for row in group)
        n_evaluable = len(evaluable)
        n_valid_finite = sum(row["run_outcome_class"] == "valid_finite_result" for row in group)
        n_valid_extreme = sum(row["run_outcome_class"] == "valid_extreme_error" for row in group)
        n_invalid = sum(row["run_outcome_class"] in {"invalid_output", "no_output", "system_error", "unknown_failure"} for row in group)
        n_timeout = sum(row["run_outcome_class"] == "timeout_no_output" for row in group)
        n_not_finished = sum(row["run_outcome_class"] == "not_finished" for row in group)
        n_wrong = sum(row["run_outcome_class"] == "wrong_dataset" for row in group)
        row = {
            "dataset_id": dataset_id,
            "dataset_name": first.get("dataset_name"),
            "family": first.get("family"),
            "subgroup": first.get("subgroup"),
            "metadata_class": first.get("metadata_class"),
            "dataset_rel": first.get("dataset_rel"),
            "basename": first.get("basename"),
            "formula_identity": first.get("formula_identity"),
            "method_norm": method,
            "n_expected_seeds": expected_seeds,
            "n_observed_rows": len([r for r in group if not r.get("synthetic_missing_row")]),
            "n_finished_seeds": n_finished,
            "n_evaluable_seeds": n_evaluable,
            "n_valid_finite_seeds": n_valid_finite,
            "n_valid_extreme_seeds": n_valid_extreme,
            "n_valid_total_seeds": n_valid_finite + n_valid_extreme,
            "n_invalid_seeds": n_invalid,
            "n_timeout_seeds": n_timeout,
            "n_not_finished_seeds": n_not_finished,
            "n_wrong_dataset_seeds": n_wrong,
            "completion_rate": _rate(n_finished, expected_seeds),
            "evaluable_rate": _rate(n_evaluable, expected_seeds),
            "valid_rate": _rate(n_valid_finite + n_valid_extreme, max(n_evaluable, 1)),
            "valid_rate_over_expected": _rate(n_valid_finite + n_valid_extreme, expected_seeds),
            "invalid_rate": _rate(n_invalid, max(n_evaluable, 1)),
            "timeout_rate": _rate(n_timeout, max(n_evaluable, 1)),
            "extreme_error_rate": _rate(n_valid_extreme, max(n_evaluable, 1)),
            "median_log_train_nmse": _median([r["train_log_nmse_used"] for r in perf]),
            "median_log_valid_nmse": _median([r["valid_log_nmse_used"] for r in perf]),
            "median_log_id_nmse": _median([r["id_log_nmse_used"] for r in perf]),
            "median_log_ood_nmse": _median([r["ood_log_nmse_used"] for r in perf]),
            "iqr_log_id_nmse": _iqr([r["id_log_nmse_used"] for r in perf]),
            "iqr_log_ood_nmse": _iqr([r["ood_log_nmse_used"] for r in perf]),
            "median_id_r2": _median([r["id_r2"] for r in perf]),
            "median_ood_r2": _median([r["ood_r2"] for r in perf]),
            "median_runtime": _median([r["result_seconds"] for r in group]),
            "p95_runtime": _percentile([r["result_seconds"] for r in group if r["result_seconds"] is not None], 0.95),
            "median_complexity": _median([r["result_complexity"] for r in perf]),
            "iqr_complexity": _iqr([r["result_complexity"] for r in perf]),
            "median_tree_depth": _median([r["result_tree_depth"] for r in perf]),
            "dominant_run_outcome_class": _mode([r["run_outcome_class"] for r in group]),
            "dominant_failure_reason": _mode([r["failure_reason_normalized"] for r in group]),
        }
        out.append(row)
    return out


def _pairwise_gaps(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    gaps = [abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]]
    return statistics.mean(gaps), max(gaps)


def _semantic_duplicate_group(row: dict[str, Any]) -> str:
    formula_identity = _clean_text(row.get("formula_identity"))
    if formula_identity:
        return formula_identity
    parts = [_clean_text(row.get("family")), _clean_text(row.get("subgroup")), _clean_text(row.get("basename"))]
    joined = "/".join(part for part in parts if part)
    return joined or _clean_text(row.get("dataset_id"))


def _formula_hash(row: dict[str, Any]) -> str:
    text = _clean_text(row.get("formula_identity") or row.get("formula_return_expr") or row.get("dataset_id"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _operator_group(text: str) -> str:
    expr = text.lower()
    if not expr:
        return "unknown"
    has_trig = any(token in expr for token in ("sin", "cos", "tan"))
    has_exp_log = "exp" in expr or "log" in expr
    has_div = "/" in expr or "div" in expr
    has_pow = "**" in expr or "pow" in expr or "sqrt" in expr
    if has_trig or has_exp_log:
        if has_trig and not has_exp_log:
            return "trigonometric"
        if has_exp_log and not has_trig:
            return "exponential_log"
        return "mixed_elementary"
    if has_div:
        return "rational"
    if has_pow:
        return "polynomial"
    if any(op in expr for op in ("+", "-", "*")):
        return "polynomial"
    return "unknown"


def _assign_quantile_bins(rows: list[dict[str, Any]], field: str, out_field: str, labels: tuple[str, str, str]) -> None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float)) and math.isfinite(row[field])]
    q1 = _percentile(values, 1 / 3)
    q2 = _percentile(values, 2 / 3)
    for row in rows:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or q1 is None or q2 is None:
            row[out_field] = "unknown"
        elif value <= q1:
            row[out_field] = labels[0]
        elif value <= q2:
            row[out_field] = labels[1]
        else:
            row[out_field] = labels[2]


def _derive_dataset_rows(
    run_rows: list[dict[str, Any]],
    dataset_algorithm_rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    run_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alg_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        run_by_dataset[row["dataset_id"]].append(row)
    for row in dataset_algorithm_rows:
        alg_by_dataset[row["dataset_id"]].append(row)

    out: list[dict[str, Any]] = []
    expected_methods = len(methods)
    expected_total_runs = len(methods) * len(seeds)
    for dataset_id in sorted(run_by_dataset, key=lambda value: (_int(value[1:]) if value.startswith("g") else 10**9, value)):
        runs = run_by_dataset[dataset_id]
        algs = alg_by_dataset.get(dataset_id, [])
        first = runs[0]
        valid_algs = [row for row in algs if (row.get("n_valid_total_seeds") or 0) > 0]
        finished_runs = sum(row["is_finished_run"] for row in runs)
        valid_runs = sum(row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"} for row in runs)
        invalid_runs = sum(row["run_outcome_class"] in {"invalid_output", "no_output", "system_error", "unknown_failure"} for row in runs)
        timeout_runs = sum(row["run_outcome_class"] == "timeout_no_output" for row in runs)
        extreme_runs = sum(row["run_outcome_class"] == "valid_extreme_error" for row in runs)
        wrong_runs = sum(row["run_outcome_class"] == "wrong_dataset" for row in runs)
        not_finished_runs = sum(row["run_outcome_class"] == "not_finished" for row in runs)
        completed_methods = sum(row.get("completion_rate") == 1 for row in algs)
        valid_methods = len(valid_algs)

        id_values = [row.get("median_log_id_nmse") for row in valid_algs]
        ood_values = [row.get("median_log_ood_nmse") for row in valid_algs]
        ood_gap_values = [
            row["median_log_ood_nmse"] - row["median_log_id_nmse"]
            for row in valid_algs
            if row.get("median_log_ood_nmse") is not None and row.get("median_log_id_nmse") is not None
        ]
        pair_mean, pair_max = _pairwise_gaps([value for value in ood_values if value is not None])
        valid_pattern_entropy = _entropy_binary(_rate(valid_methods, expected_methods))

        ranked = [
            row
            for row in valid_algs
            if row.get("median_log_ood_nmse") is not None
        ]
        ranked.sort(
            key=lambda row: (
                row["median_log_ood_nmse"],
                row.get("median_log_id_nmse") if row.get("median_log_id_nmse") is not None else math.inf,
                -(row.get("valid_rate") or 0),
                row.get("median_runtime") if row.get("median_runtime") is not None else math.inf,
                row["method_norm"],
            )
        )
        winner = ranked[0]["method_norm"] if ranked else ""
        worst = max(ranked, key=lambda row: row["median_log_ood_nmse"])["method_norm"] if ranked else ""
        winner_margin = None
        if len(ranked) >= 2:
            winner_margin = ranked[1]["median_log_ood_nmse"] - ranked[0]["median_log_ood_nmse"]

        row = {
            "dataset_id": dataset_id,
            "global_index": first.get("global_index"),
            "dataset_name": first.get("dataset_name"),
            "family": first.get("family"),
            "subgroup": first.get("subgroup"),
            "metadata_class": first.get("metadata_class"),
            "dataset_rel": first.get("dataset_rel"),
            "basename": first.get("basename"),
            "formula_identity": first.get("formula_identity"),
            "feature_count": first.get("feature_count"),
            "target_name": first.get("target_name"),
            "train_samples": first.get("train_samples"),
            "valid_samples": first.get("valid_samples"),
            "id_test_samples": first.get("id_test_samples"),
            "ood_test_samples": first.get("ood_test_samples"),
            "formula_char_count": first.get("formula_char_count"),
            "formula_operator_count": first.get("formula_operator_count"),
            "formula_return_expr": first.get("formula_return_expr"),
            "expected_probe_methods": expected_methods,
            "expected_total_runs": expected_total_runs,
            "observed_methods": len({row["method_norm"] for row in runs if not row.get("synthetic_missing_row")}),
            "completed_methods": completed_methods,
            "valid_methods": valid_methods,
            "finished_runs": finished_runs,
            "valid_runs": valid_runs,
            "invalid_runs": invalid_runs,
            "timeout_runs": timeout_runs,
            "valid_extreme_error_runs": extreme_runs,
            "wrong_dataset_runs": wrong_runs,
            "not_finished_runs": not_finished_runs,
            "completion_rate": _rate(finished_runs, expected_total_runs),
            "dataset_valid_rate": _rate(valid_runs, max(finished_runs, 1)),
            "dataset_invalid_rate": _rate(invalid_runs, max(finished_runs, 1)),
            "dataset_timeout_rate": _rate(timeout_runs, max(finished_runs, 1)),
            "dataset_extreme_error_rate": _rate(extreme_runs, max(finished_runs, 1)),
            "mean_median_log_id_nmse": _mean(id_values),
            "mean_median_log_ood_nmse": _mean(ood_values),
            "median_log_id_nmse_across_probes": _median(id_values),
            "median_log_ood_nmse_across_probes": _median(ood_values),
            "mean_iqr_log_id_nmse": _mean([row.get("iqr_log_id_nmse") for row in valid_algs]),
            "mean_iqr_log_ood_nmse": _mean([row.get("iqr_log_ood_nmse") for row in valid_algs]),
            "max_iqr_log_ood_nmse": max(
                [row.get("iqr_log_ood_nmse") for row in valid_algs if row.get("iqr_log_ood_nmse") is not None],
                default=0.0,
            ),
            "id_probe_variance": _pvariance(id_values),
            "ood_probe_variance": _pvariance(ood_values),
            "ood_id_gap_variance": _pvariance(ood_gap_values),
            "pairwise_gap_mean": pair_mean,
            "pairwise_gap_max": pair_max,
            "valid_pattern_entropy": valid_pattern_entropy,
            "rank_entropy": _softmax_entropy([value for value in ood_values if value is not None]),
            "winner_probe": winner,
            "winner_margin": winner_margin,
            "worst_probe": worst,
            "semantic_duplicate_group": _semantic_duplicate_group(first),
            "formula_hash": _formula_hash(first),
            "operator_group": _operator_group(_clean_text(first.get("formula_return_expr"))),
        }
        row["difficulty_score"] = _mean(
            [
                0.5 * alg["median_log_id_nmse"] + 0.5 * alg["median_log_ood_nmse"]
                for alg in valid_algs
                if alg.get("median_log_id_nmse") is not None and alg.get("median_log_ood_nmse") is not None
            ]
        )
        row["instability_raw"] = (
            0.40 * (row["mean_iqr_log_ood_nmse"] or 0.0)
            + 0.20 * (row["mean_iqr_log_id_nmse"] or 0.0)
            + 0.20 * (row["max_iqr_log_ood_nmse"] or 0.0)
            + 0.10 * row["dataset_invalid_rate"]
            + 0.10 * row["dataset_timeout_rate"]
        )
        out.append(row)

    _apply_dataset_scores(out)
    return out


def _apply_dataset_scores(rows: list[dict[str, Any]]) -> None:
    fields = [
        "id_probe_variance",
        "ood_probe_variance",
        "ood_id_gap_variance",
        "pairwise_gap_mean",
        "valid_pattern_entropy",
        "instability_raw",
    ]
    norms = {field: _robust_norm([row.get(field) for row in rows]) for field in fields}
    for idx, row in enumerate(rows):
        row["norm_id_probe_variance"] = norms["id_probe_variance"][idx]
        row["norm_ood_probe_variance"] = norms["ood_probe_variance"][idx]
        row["norm_ood_id_gap_variance"] = norms["ood_id_gap_variance"][idx]
        row["norm_pairwise_gap_mean"] = norms["pairwise_gap_mean"][idx]
        row["norm_valid_pattern_entropy"] = norms["valid_pattern_entropy"][idx]
        row["discrimination_score"] = (
            0.35 * row["norm_id_probe_variance"]
            + 0.35 * row["norm_ood_probe_variance"]
            + 0.15 * row["norm_ood_id_gap_variance"]
            + 0.10 * row["norm_pairwise_gap_mean"]
            + 0.05 * row["norm_valid_pattern_entropy"]
        )
        row["stability_score"] = 1.0 - norms["instability_raw"][idx]
        row["info_score"] = row["discrimination_score"] * row["stability_score"]

    difficulty_values = [row["difficulty_score"] for row in rows if row.get("difficulty_score") is not None]
    q20 = _percentile(difficulty_values, 0.20)
    q60 = _percentile(difficulty_values, 0.60)
    q90 = _percentile(difficulty_values, 0.90)
    max_iqr_values = [row["max_iqr_log_ood_nmse"] for row in rows if row.get("max_iqr_log_ood_nmse") is not None]
    unstable_q90 = _percentile(max_iqr_values, 0.90) or 0.0
    ood_gap_values = [
        (row.get("mean_median_log_ood_nmse") or 0.0) - (row.get("mean_median_log_id_nmse") or 0.0)
        for row in rows
        if row.get("mean_median_log_ood_nmse") is not None and row.get("mean_median_log_id_nmse") is not None
    ]
    ood_gap_q75 = _percentile(ood_gap_values, 0.75) or 0.0
    winner_margins = [row["winner_margin"] for row in rows if row.get("winner_margin") is not None]
    winner_margin_q75 = _percentile(winner_margins, 0.75) or 0.0

    for row in rows:
        score = row.get("difficulty_score")
        if score is None or q20 is None or q60 is None or q90 is None:
            row["difficulty_bin"] = "unknown"
        elif row["dataset_invalid_rate"] >= 0.5 or row["dataset_timeout_rate"] >= 0.5:
            row["difficulty_bin"] = "extreme"
        elif score <= q20:
            row["difficulty_bin"] = "easy"
        elif score <= q60:
            row["difficulty_bin"] = "medium"
        elif score <= q90:
            row["difficulty_bin"] = "hard"
        else:
            row["difficulty_bin"] = "extreme"

        ood_gap = None
        if row.get("mean_median_log_ood_nmse") is not None and row.get("mean_median_log_id_nmse") is not None:
            ood_gap = row["mean_median_log_ood_nmse"] - row["mean_median_log_id_nmse"]
        row["mean_ood_minus_id_log_nmse"] = ood_gap
        row["failure_mode"] = _failure_mode(row, unstable_q90, ood_gap_q75, winner_margin_q75)
        row["eligible_class"] = _eligible_class(row)

    _assign_quantile_bins(rows, "feature_count", "feature_count_bin", ("low", "medium", "high"))
    for row in rows:
        sample_count = sum(
            value or 0
            for value in (
                row.get("train_samples"),
                row.get("valid_samples"),
                row.get("id_test_samples"),
                row.get("ood_test_samples"),
            )
        )
        row["total_samples"] = sample_count
    _assign_quantile_bins(rows, "total_samples", "sample_count_bin", ("small", "medium", "large"))
    complexity_field = "formula_operator_count"
    _assign_quantile_bins(rows, complexity_field, "complexity_bin", ("simple", "moderate", "complex"))


def _failure_mode(row: dict[str, Any], unstable_q90: float, ood_gap_q75: float, winner_margin_q75: float) -> str:
    if row["completion_rate"] < 1.0:
        return "incomplete"
    if row["dataset_invalid_rate"] >= 0.5 or row["dataset_timeout_rate"] >= 0.5:
        return "invalid_prone"
    if row["valid_methods"] <= 2:
        return "one_sided"
    if (row.get("max_iqr_log_ood_nmse") or 0.0) >= unstable_q90 and unstable_q90 > 0:
        return "unstable"
    if row.get("mean_ood_minus_id_log_nmse") is not None and row["mean_ood_minus_id_log_nmse"] >= ood_gap_q75 and ood_gap_q75 > 0:
        return "ood_failure"
    if row.get("difficulty_bin") == "extreme":
        return "all_struggle"
    if row.get("winner_margin") is not None and row["winner_margin"] >= winner_margin_q75 and winner_margin_q75 > 0:
        return "one_method_wins"
    return "all_good"


def _eligible_class(row: dict[str, Any]) -> str:
    if row["completion_rate"] < 1.0:
        return "incomplete"
    if row["wrong_dataset_runs"] > 0 or row["valid_methods"] == 0 or row["dataset_valid_rate"] == 0:
        return "excluded"
    if row["failure_mode"] in {"invalid_prone", "one_sided", "unstable", "all_struggle"} or row["difficulty_bin"] == "extreme":
        return "limited_quota"
    return "eligible"


def _summary(
    run_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    input_fields: list[str],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    expected_total = len({row["dataset_id"] for row in dataset_rows}) * len(methods) * len(seeds)
    observed_runs = sum(1 for row in run_rows if not row.get("synthetic_missing_row"))
    finished_runs = sum(row["is_finished_run"] for row in run_rows)
    valid_runs = sum(row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"} for row in run_rows)
    not_finished_runs = sum(row["run_outcome_class"] == "not_finished" for row in run_rows)
    method_completion: dict[str, dict[str, Any]] = {}
    for method in methods:
        group = [row for row in run_rows if row["method_norm"] == method]
        method_completion[method] = {
            "observed_runs": sum(1 for row in group if not row.get("synthetic_missing_row")),
            "finished_runs": sum(row["is_finished_run"] for row in group),
            "valid_runs": sum(row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"} for row in group),
            "completion_rate": _rate(sum(row["is_finished_run"] for row in group), len(group)),
            "valid_rate": _rate(
                sum(row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"} for row in group),
                max(sum(row["is_finished_run"] for row in group), 1),
            ),
        }
    seed_completion: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        group = [row for row in run_rows if row["seed_norm"] == seed]
        seed_completion[str(seed)] = {
            "observed_runs": sum(1 for row in group if not row.get("synthetic_missing_row")),
            "finished_runs": sum(row["is_finished_run"] for row in group),
            "valid_runs": sum(row["run_outcome_class"] in {"valid_finite_result", "valid_extreme_error"} for row in group),
            "completion_rate": _rate(sum(row["is_finished_run"] for row in group), len(group)),
        }

    missing_required: list[str] = []
    if not any(field in input_fields for field in ("method", "algorithm", "tool")):
        missing_required.append("method|algorithm|tool")
    if "seed" not in input_fields:
        missing_required.append("seed")
    if not any(field in input_fields for field in ("dataset_id", "global_index", "dataset_name")):
        missing_required.append("dataset_id|global_index|dataset_name")
    optional = [
        "result_train_nmse",
        "result_valid_nmse",
        "result_id_test_nmse",
        "result_ood_test_nmse",
        "result_valid_output",
        "result_metric_complete",
        "result_dataset_identity_match",
    ]
    missing_optional = [field for field in optional if field not in input_fields]

    return {
        "expected_total_runs": expected_total,
        "observed_runs": observed_runs,
        "finished_runs": finished_runs,
        "valid_runs": valid_runs,
        "not_finished_runs": not_finished_runs,
        "ready_for_core50_selection": bool(
            expected_total > 0
            and finished_runs == expected_total
            and not_finished_runs == 0
            and not missing_required
            and all(row["eligible_class"] != "incomplete" for row in dataset_rows)
        ),
        "method_completion": method_completion,
        "seed_completion": seed_completion,
        "run_outcome_counts": dict(sorted(Counter(row["run_outcome_class"] for row in run_rows).items())),
        "eligible_class_counts": dict(sorted(Counter(row["eligible_class"] for row in dataset_rows).items())),
        "failure_mode_counts": dict(sorted(Counter(row["failure_mode"] for row in dataset_rows).items())),
        "difficulty_bin_counts": dict(sorted(Counter(row["difficulty_bin"] for row in dataset_rows).items())),
        "missing_required_columns": missing_required,
        "missing_optional_columns": missing_optional,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Probe4 Full664 后处理完成度与选择约束报告",
        "",
        "本报告只判断当前后处理指标是否可用于后续 Core-50 selection，不执行 Core-50 选择。",
        "",
        "## 总体完成度",
        "",
        f"- expected_total_runs: `{summary['expected_total_runs']}`",
        f"- observed_runs: `{summary['observed_runs']}`",
        f"- finished_runs: `{summary['finished_runs']}`",
        f"- valid_runs: `{summary['valid_runs']}`",
        f"- not_finished_runs: `{summary['not_finished_runs']}`",
        f"- ready_for_core50_selection: `{str(summary['ready_for_core50_selection']).lower()}`",
        "",
        "## Method Completion",
        "",
        "| method | observed_runs | finished_runs | valid_runs | completion_rate | valid_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in summary["method_completion"].items():
        lines.append(
            f"| {method} | {row['observed_runs']} | {row['finished_runs']} | {row['valid_runs']} | "
            f"{row['completion_rate']:.4f} | {row['valid_rate']:.4f} |"
        )
    lines.extend(["", "## Seed Completion", "", "| seed | observed_runs | finished_runs | valid_runs | completion_rate |", "| --- | ---: | ---: | ---: | ---: |"])
    for seed, row in summary["seed_completion"].items():
        lines.append(
            f"| {seed} | {row['observed_runs']} | {row['finished_runs']} | {row['valid_runs']} | "
            f"{row['completion_rate']:.4f} |"
        )
    for title, key in (
        ("Run Outcome Distribution", "run_outcome_counts"),
        ("Eligible Class Distribution", "eligible_class_counts"),
        ("Difficulty Distribution", "difficulty_bin_counts"),
        ("Failure Mode Distribution", "failure_mode_counts"),
    ):
        lines.extend(["", f"## {title}", ""])
        for name, count in summary[key].items():
            lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## 字段缺失",
            "",
            f"- missing_required_columns: `{summary['missing_required_columns']}`",
            f"- missing_optional_columns: `{summary['missing_optional_columns']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(output_dir: Path, input_path: Path) -> None:
    lines = [
        "# Probe4 Full664 Postprocess Outputs",
        "",
        f"- 输入: `{input_path}`",
        "- 目标: 生成 Core-50 selection 前置诊断指标，不选择 Core-50。",
        "",
        "## 文件",
        "",
        "- `probe4_postprocess_run_level.csv`: 每条 run 的完成/有效/失败分类与 log NMSE。",
        "- `probe4_postprocess_dataset_algorithm.csv`: 每个 dataset × algorithm 的 3-seed 聚合。",
        "- `probe4_postprocess_dataset_level.csv`: 每个 dataset 的区分度、稳定性、难度和候选资格指标。",
        "- `probe4_postprocess_summary.json`: 机器可读 summary。",
        "- `selection_constraints_report.md`: 人读完成度与选择约束报告。",
        "",
        "## 重要口径",
        "",
        "- `not_finished` 不计入 invalid、timeout 或算法失败。",
        "- `valid_extreme_error` 仍是有效输出，会参与 clipped log NMSE 统计。",
        "- `eligible_class` 是诊断标签，不等于最终 Core-50 选择结果。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).resolve()
    dataset_catalog = _load_dataset_catalog(Path(args.dataset_catalog).resolve())
    input_fields, input_rows = _read_csv(input_path)
    methods = tuple(_method_norm(method) for method in args.methods.split(",") if method.strip())
    seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed.strip())
    completed_rows = _complete_expected_grid(input_rows, dataset_catalog, methods, seeds)
    run_rows = _derive_run_rows(completed_rows, methods, seeds)
    dataset_algorithm_rows = _derive_dataset_algorithm_rows(run_rows, methods, seeds)
    dataset_rows = _derive_dataset_rows(run_rows, dataset_algorithm_rows, methods, seeds)
    summary = _summary(run_rows, dataset_rows, input_fields, methods, seeds)

    output_dir = Path(args.output).resolve() if args.output else (
        DEFAULT_OUTPUT_ROOT / f"postprocess_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "probe4_postprocess_run_level.csv", run_rows)
    _write_csv(output_dir / "probe4_postprocess_dataset_algorithm.csv", dataset_algorithm_rows)
    _write_csv(output_dir / "probe4_postprocess_dataset_level.csv", dataset_rows)
    (output_dir / "probe4_postprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "selection_constraints_report.md", summary)
    _write_readme(output_dir, input_path)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_default_input()), help="Probe4 run-level CSV。")
    parser.add_argument("--dataset-catalog", default=str(DEFAULT_DATASET_CATALOG), help="Full664 数据集目录 CSV。")
    parser.add_argument("--output", default="", help="输出目录；默认写入 03/generated/postprocess_<timestamp>。")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="逗号分隔的预期 probe methods。")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="逗号分隔的预期 seeds。")
    return parser.parse_args()


def main() -> None:
    output_dir = build(parse_args())
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
