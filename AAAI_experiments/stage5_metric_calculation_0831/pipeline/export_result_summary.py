"""导出 Stage5 clean 六轴与三条件 180 分钟汇总表。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import aggregate_clean_metrics as clean_aggregate
from .metrics import (
    RunQuality,
    minimality_score,
    phi_nmse,
    stability_score,
    symbolic_fidelity_score,
)
from .performance_replay import (
    EVALUATION_PATH,
    PerformanceReplayCache,
    PerformanceReplayError,
    replay_payload_performance,
)
from .symbolic_evidence import (
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)


HORIZON = 180
SEEDS = (520, 521, 522)
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))
FORBIDDEN_SOURCE_TOKENS = ("all_15alg_fullcpu_v1",)
CONSISTENT_STRUCTURE_DECISIONS = {
    "mathematically_equivalent",
    "same_canonical_structure",
}


class ResultSummaryError(ValueError):
    """汇总输入不满足可审计契约。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source_text(value: object, *, context: str) -> None:
    text = str(value)
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise ResultSummaryError(f"{context} 命中禁止来源 {token!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _check_source_text(path.resolve(), context="输入路径")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ResultSummaryError(f"CSV 为空: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _check_source_text(path.resolve(), context="输入路径")
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ResultSummaryError(f"{path}:{line_number} 不是 JSON object")
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ResultSummaryError(f"拒绝写出空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _logical_key(source: Mapping[str, Any]) -> str:
    algorithm = str(source["algorithm"])
    dataset_id = str(source["dataset_id"])
    seed = int(source["seed"])
    condition = str(source["noise_tag"])
    return f"{algorithm}::{dataset_id}::s{seed}::{condition}"


def _load_record_map(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in _read_jsonl(path):
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise ResultSummaryError(f"{path} 中记录缺少 source")
            for field in ("path", "batch", "task_id"):
                _check_source_text(source.get(field, ""), context=f"source.{field}")
            key = _logical_key(source)
            if key in records:
                raise ResultSummaryError(f"重复轨迹 logical_key: {key}")
            records[key] = record
    return records


def merge_clean_eff_rows(
    current_rows: Sequence[Mapping[str, str]],
    legacy_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """仅保留当前严格 EFF；历史表只用于识别缺口，不得作为正式回退。"""

    legacy = {str(row["logical_key"]): dict(row) for row in legacy_rows}
    current = {str(row["logical_key"]): dict(row) for row in current_rows}
    if len(legacy) != len(legacy_rows) or len(current) != len(current_rows):
        raise ResultSummaryError("clean EFF 输入存在重复 logical_key")
    if len(legacy) != 2250:
        raise ResultSummaryError(f"历史 clean EFF 应有 2250 行，实际 {len(legacy)}")
    extra = set(current) - set(legacy)
    if extra:
        raise ResultSummaryError(f"当前 clean EFF 含基底外 logical_key: {sorted(extra)[:5]}")
    unavailable = set(legacy) - set(current)
    return current, unavailable


def _trajectory_from_eff_row(row: Mapping[str, str]) -> list[float]:
    values: list[float] = []
    for minute in range(1, HORIZON + 1):
        field = f"q_{minute:04d}"
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultSummaryError(f"{row.get('logical_key')} 缺少有效 {field}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ResultSummaryError(f"{row.get('logical_key')}.{field} 越界")
        values.append(value)
    return values


def aggregate_eff_rows(
    rows_by_key: Mapping[str, Mapping[str, str]],
    *,
    condition: str,
    fallback_keys: set[str] | None = None,
    trajectory_basis: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    fallback_keys = fallback_keys or set()
    grouped: dict[str, list[tuple[Mapping[str, str], list[float], list[float]]]] = defaultdict(list)
    for key, row in rows_by_key.items():
        if str(row.get("noise_tag")) != condition:
            raise ResultSummaryError(f"{key} condition 不是 {condition}")
        qualities = _trajectory_from_eff_row(row)
        q_star = max(qualities, default=0.0)
        relative = [value / q_star if q_star > 0.0 else 0.0 for value in qualities]
        grouped[str(row["algorithm"])].append((row, qualities, relative))

    output: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for algorithm in sorted(grouped):
        runs = grouped[algorithm]
        if len(runs) != 150:
            raise ResultSummaryError(f"{condition}/{algorithm} 应有 150 条轨迹，实际 {len(runs)}")
        algorithm_fallbacks = sum(
            1 for row, _, _ in runs if str(row["logical_key"]) in fallback_keys
        )
        cumulative_relative = 0.0
        for minute in range(1, HORIZON + 1):
            mean_quality = sum(item[1][minute - 1] for item in runs) / len(runs)
            mean_relative = sum(item[2][minute - 1] for item in runs) / len(runs)
            cumulative_relative += mean_relative
            output.append(
                {
                    "condition": condition,
                    "algorithm": algorithm,
                    "minute": minute,
                    "expected_run_count": 150,
                    "available_run_count": len(runs),
                    "coverage_rate": "1",
                    "mean_quality": f"{mean_quality:.17g}",
                    "mean_relative_progress": f"{mean_relative:.17g}",
                    "cumulative_eff_score": f"{100.0 * cumulative_relative / minute:.17g}",
                    "current_path_run_count": len(runs) - algorithm_fallbacks,
                    "fallback_run_count": algorithm_fallbacks,
                    "trajectory_basis": trajectory_basis,
                    "eff_complete_for_algorithm": str(algorithm_fallbacks == 0).lower(),
                    "formal_ready": "false",
                }
            )
        scores[algorithm] = 100.0 * cumulative_relative / HORIZON
    if len(grouped) != 15 or len(output) != 15 * HORIZON:
        raise ResultSummaryError("clean 180 分钟输出未闭合到 15 算法 x 180 分钟")
    return output, scores


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ResultSummaryError(f"无法解析布尔值: {value!r}")


def _load_latest_clean_symbolic(
    stage_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str, tuple[int, int]], dict[str, Any]]]:
    work = stage_root / "work/clean_final_replacement_v1"
    downstream = work / "downstream_refresh_v2"
    pred_plan = work / "clean_pred_hybrid_active_v7_identity.jsonl"
    pred_identity, _ = clean_aggregate._load_pred_simplify_identity_map(
        pred_plan, expected_runs=2250
    )
    pred_index, _ = clean_aggregate._load_frozen_index_rows(
        index_path=downstream / "clean_pred_frozen_active_v7.jsonl",
        summary_path=downstream / "clean_pred_frozen_active_v7_summary.json",
        plan_path=pred_plan,
        expected_task_type="pred_simplify",
        label="latest_pred",
    )
    pred_rows = clean_aggregate._build_pred_index(
        pred_index, pred_identity_map=pred_identity, expected_runs=2250
    )
    eq_plan = downstream / "clean_equivalence_hybrid_active_v3.jsonl"
    eq_index, _ = clean_aggregate._load_frozen_index_rows(
        index_path=downstream / "clean_equivalence_frozen_active_v3.jsonl",
        summary_path=downstream / "clean_equivalence_frozen_active_v3_summary.json",
        plan_path=eq_plan,
        expected_task_type="equivalence",
        label="latest_equivalence",
    )
    eq_rows = clean_aggregate._build_equivalence_index(
        eq_index, pred_identity_map=pred_identity, expected_runs=2250
    )
    structure_plan = downstream / "clean_structure_hybrid_active_v2.jsonl"
    structure_index, _ = clean_aggregate._load_frozen_index_rows(
        index_path=downstream / "clean_structure_frozen_active_v2.jsonl",
        summary_path=downstream / "clean_structure_frozen_active_v2_summary.json",
        plan_path=structure_plan,
        expected_task_type="stab_structure",
        label="latest_structure",
    )
    structure_rows = clean_aggregate._build_structure_index(
        structure_index,
        pred_identity_map=pred_identity,
        expected_rows=15 * 50 * len(SEED_PAIRS),
    )
    return pred_rows, eq_rows, structure_rows


def _load_gt_audit_override(stage_root: Path) -> tuple[str, dict[str, object]] | None:
    manifest = stage_root / "audits/formula_quality_1000_0903_v2/corrections_v2/manifest/corrections_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    corrections_path = Path(payload["outputs"]["corrections"]["path"])
    if not corrections_path.is_absolute():
        corrections_path = stage_root.parents[1] / corrections_path
    overrides = []
    for row in _read_jsonl(corrections_path):
        if row.get("target_kind") == "gt" and row.get("action") == "replace_expression":
            logical_id = str(row["target_logical_id"])
            dataset_id = logical_id.split("::")[1]
            expression = str(row["after"]["effective_expression"])
            overrides.append((dataset_id, build_symbolic_artifact(expression)))
    if len(overrides) > 1:
        raise ResultSummaryError("当前导出器只允许一个已审计 GT override")
    return overrides[0] if overrides else None


def build_clean_six_axis(
    stage_root: Path,
    *,
    eff_scores: Mapping[str, float],
    eff_fallback_keys: set[str],
) -> list[dict[str, Any]]:
    downstream = stage_root / "work/clean_final_replacement_v1/downstream_refresh_v2"
    numeric_rows = _read_csv(downstream / "clean_numeric_run_metrics.csv")
    if len(numeric_rows) != 2250:
        raise ResultSummaryError("最新 clean numeric 不是 2250 行")
    numeric = {row["logical_key"]: row for row in numeric_rows}
    old_run_rows = _read_csv(stage_root / "results/clean_run_metrics.csv")
    old_runs = {row["logical_key"]: dict(row) for row in old_run_rows}
    old_task_rows = _read_csv(stage_root / "results/task_stability.csv")
    old_tasks = {(row["algorithm"], row["dataset_id"]): row for row in old_task_rows}
    binding = json.loads(
        (stage_root / "work/clean_final_replacement_v1/binding_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    replacement_keys = set(binding["replacement_contract"]["replacement_keys"])
    if len(replacement_keys) != 75:
        raise ResultSummaryError("clean final replacement keys 不是 75 条")

    pred_rows, eq_rows, structure_rows = _load_latest_clean_symbolic(stage_root)
    evidence_rows = {
        row["logical_key"]: row
        for row in _read_jsonl(downstream / "clean_pred_vs_gt_evidence_active_v3.jsonl")
    }
    gt_override = _load_gt_audit_override(stage_root)

    symbolic_rows = {key: dict(row) for key, row in old_runs.items()}
    for key in replacement_keys:
        pred = pred_rows[key]
        equivalence = eq_rows[key]
        old = symbolic_rows[key]
        symbolic_valid = bool(pred["symbolic_valid"])
        reference_complexity = int(old["reference_complexity"])
        predicted_complexity = 0
        tree_value = variable_value = operator_value = 0.0
        decision = "non_applicable"
        if symbolic_valid:
            evidence = evidence_rows[key]
            predicted_complexity = int(pred["artifact"]["node_count"])
            decision = str(equivalence["decision"])
            if gt_override is not None and old["dataset_id"] == gt_override[0]:
                gt_artifact = gt_override[1]
                reference_complexity = int(gt_artifact["node_count"])
                tree_value = tree_similarity(gt_artifact, pred["artifact"])
                variable_value = variable_f1(gt_artifact, pred["artifact"])
                operator_value = operator_f1(gt_artifact, pred["artifact"])
            else:
                tree_value = float(evidence["tree"]["tree_similarity"])
                variable_value = float(evidence["variable"]["f1"])
                operator_value = float(evidence["operator"]["f1"])
        equivalent = decision == "equivalent"
        old.update(
            {
                "pred_state": str(pred["state"]),
                "equivalence_state": str(equivalence["state"]),
                "equivalence_decision": decision,
                "equivalent": str(equivalent).lower(),
                "tree_similarity": f"{tree_value:.17g}",
                "variable_f1": f"{variable_value:.17g}",
                "operator_f1": f"{operator_value:.17g}",
                "m_sym": f"{symbolic_fidelity_score(equivalent=equivalent, tree_similarity=tree_value, variable_f1=variable_value, operator_f1=operator_value, valid=symbolic_valid):.17g}",
                "reference_complexity": str(reference_complexity),
                "predicted_complexity": str(predicted_complexity),
                "m_min": f"{minimality_score(reference_complexity, predicted_complexity if predicted_complexity > 0 else 1, valid=symbolic_valid):.17g}",
            }
        )

    grouped_numeric: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in numeric_rows:
        grouped_numeric[(row["algorithm"], row["dataset_id"])].append(row)
    stab_by_algorithm: dict[str, list[float]] = defaultdict(list)
    for (algorithm, dataset_id), group in grouped_numeric.items():
        group.sort(key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in group) != SEEDS:
            raise ResultSummaryError(f"{algorithm}/{dataset_id} seed 不完整")
        affected = any(row["logical_key"] in replacement_keys for row in group)
        old_task = old_tasks[(algorithm, dataset_id)]
        structural: list[bool] = []
        for pair in SEED_PAIRS:
            if affected:
                decision = str(structure_rows[(algorithm, dataset_id, pair)]["decision"])
            else:
                decision = str(old_task[f"pair_{pair[0]}_{pair[1]}"])
            structural.append(decision in CONSISTENT_STRUCTURE_DECISIONS)
        qualities = [
            RunQuality(
                id_quality=float(row["id_quality"]),
                ood_quality=float(row["ood_quality"]),
                valid=_parse_bool(row["valid_output"]),
            )
            for row in group
        ]
        stab_by_algorithm[algorithm].append(
            stability_score(qualities, structural_pair_results=structural).score
        )

    rows_by_algorithm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for key, row in symbolic_rows.items():
        rows_by_algorithm[numeric[key]["algorithm"]].append(row)
    output: list[dict[str, Any]] = []
    for algorithm in sorted(rows_by_algorithm):
        runs = rows_by_algorithm[algorithm]
        numeric_group = [row for row in numeric_rows if row["algorithm"] == algorithm]
        fallbacks = sum(1 for key in eff_fallback_keys if key.startswith(f"{algorithm}::"))
        values = {
            "ID": 100.0 * sum(float(row["id_quality"]) for row in numeric_group) / 150,
            "OOD": 100.0 * sum(float(row["ood_quality"]) for row in numeric_group) / 150,
            "SYM": 100.0 * sum(float(row["m_sym"]) for row in runs) / 150,
            "MIN": 100.0 * sum(float(row["m_min"]) for row in runs) / 150,
            "EFF": float(eff_scores[algorithm]),
            "STAB": 100.0 * sum(stab_by_algorithm[algorithm]) / 50,
        }
        output.append(
            {
                "rank": 0,
                "algorithm": algorithm,
                "run_count": 150,
                "task_count": 50,
                **{name: f"{value:.12g}" for name, value in values.items()},
                "mean_six": f"{sum(values.values()) / 6.0:.12g}",
                "numeric_evaluation_path": EVALUATION_PATH,
                "final_replacement_run_count": sum(
                    1 for key in replacement_keys if key.startswith(f"{algorithm}::")
                ),
                "eff_current_run_count": 150 - fallbacks,
                "eff_legacy_fallback_run_count": fallbacks,
                "eff_complete_for_algorithm": str(fallbacks == 0).lower(),
                "formal_ready": "false",
                "result_status": (
                    "provisional_latest_targeted_overlay"
                    if fallbacks == 0
                    else "provisional_latest_targeted_overlay_with_explicit_eff_fallback"
                ),
            }
        )
    output.sort(key=lambda row: float(row["mean_six"]), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    if len(output) != 15:
        raise ResultSummaryError("clean 六轴输出不是 15 行")
    return output


def _snapshot_nmse(snapshot: Mapping[str, Any], field: str) -> float | None:
    value = snapshot.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def postprocessed_best_so_far(record: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    """按每分钟 ID/OOD 数值质量做确定性的 observed best-so-far 后处理。"""

    raw_snapshots = record.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise ResultSummaryError("轨迹记录缺少 snapshots")
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
                candidate = (phi_nmse(id_nmse), phi_nmse(ood_nmse), 0.0)
                candidate = (candidate[0], candidate[1], (candidate[0] + candidate[1]) / 2.0)
                if candidate[2] > best[2]:
                    best = candidate
        ids.append(best[0])
        oods.append(best[1])
        qualities.append(best[2])
    return ids, oods, qualities


def aggregate_noise_trajectory(
    records: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
    overlay_keys: set[str],
) -> list[dict[str, Any]]:
    selected = {
        key: record for key, record in records.items() if key.endswith(f"::{condition}")
    }
    if len(selected) != 2250:
        raise ResultSummaryError(f"{condition} 轨迹应有 2250 条，实际 {len(selected)}")
    grouped: dict[str, list[tuple[list[float], list[float], list[float]]]] = defaultdict(list)
    overlay_by_algorithm: dict[str, int] = defaultdict(int)
    for key, record in selected.items():
        source = record["source"]
        algorithm = str(source["algorithm"])
        grouped[algorithm].append(postprocessed_best_so_far(record))
        if key in overlay_keys:
            overlay_by_algorithm[algorithm] += 1
    output: list[dict[str, Any]] = []
    for algorithm in sorted(grouped):
        runs = grouped[algorithm]
        if len(runs) != 150:
            raise ResultSummaryError(f"{condition}/{algorithm} 轨迹不是 150 条")
        relative_runs: list[list[float]] = []
        for _, _, qualities in runs:
            q_star = max(qualities, default=0.0)
            relative_runs.append(
                [quality / q_star if q_star > 0.0 else 0.0 for quality in qualities]
            )
        cumulative_relative = 0.0
        for minute in range(1, HORIZON + 1):
            mean_id = sum(run[0][minute - 1] for run in runs) / 150
            mean_ood = sum(run[1][minute - 1] for run in runs) / 150
            mean_quality = sum(run[2][minute - 1] for run in runs) / 150
            mean_relative = sum(run[minute - 1] for run in relative_runs) / 150
            cumulative_relative += mean_relative
            output.append(
                {
                    "condition": condition,
                    "algorithm": algorithm,
                    "minute": minute,
                    "expected_run_count": 150,
                    "available_run_count": 150,
                    "coverage_rate": "1",
                    "mean_id_quality": f"{mean_id:.17g}",
                    "mean_ood_quality": f"{mean_ood:.17g}",
                    "mean_quality": f"{mean_quality:.17g}",
                    "mean_relative_progress": f"{mean_relative:.17g}",
                    "cumulative_eff_score": f"{100.0 * cumulative_relative / minute:.17g}",
                    "base_run_count": 150 - overlay_by_algorithm[algorithm],
                    "targeted_overlay_run_count": overlay_by_algorithm[algorithm],
                    "trajectory_basis": "observed_numeric_best_so_far_native_snapshot.v1",
                    "formal_ready": "false",
                }
            )
    if len(grouped) != 15 or len(output) != 15 * HORIZON:
        raise ResultSummaryError(f"{condition} 180 分钟输出未闭合")
    return output


def _replay_overlay_final(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    cache: PerformanceReplayCache,
) -> dict[str, str]:
    source = record["source"]
    result = record.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("raw_text"), str):
        raise ResultSummaryError(f"{_logical_key(source)} 缺少冻结 result raw_text")
    payload = json.loads(str(result["raw_text"]))
    key = _logical_key(source)
    try:
        replay = replay_payload_performance(
            payload,
            algorithm=str(source["algorithm"]),
            repo_root=repo_root,
            cache=cache,
            task_id=str(source["task_id"]),
            condition=str(source["noise_tag"]),
            result_sha256=str(result.get("sha256") or ""),
        )
    except PerformanceReplayError as exc:
        return {
            "logical_key": key,
            "algorithm": str(source["algorithm"]),
            "dataset_id": str(source["dataset_id"]),
            "seed": str(source["seed"]),
            "noise_tag": str(source["noise_tag"]),
            "task_id": str(source["task_id"]),
            "evaluation_status": "replay_unavailable",
            "valid_output": "false",
            "id_quality": "0",
            "ood_quality": "0",
            "replay_error": str(exc),
        }
    return {
        "logical_key": key,
        "algorithm": str(source["algorithm"]),
        "dataset_id": str(source["dataset_id"]),
        "seed": str(source["seed"]),
        "noise_tag": str(source["noise_tag"]),
        "task_id": str(source["task_id"]),
        "evaluation_status": "valid" if replay["valid_output"] else "invalid_output",
        "valid_output": str(bool(replay["valid_output"])).lower(),
        "id_quality": f"{float(replay['id_quality']):.17g}",
        "ood_quality": f"{float(replay['ood_quality']):.17g}",
        "replay_error": "",
    }


def build_noise_metrics(
    stage_root: Path,
    *,
    condition: str,
    overlay_records: Mapping[str, Mapping[str, Any]],
    clean_scores: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    base_path = stage_root / f"work/current_replay_audit_v6/noise/{condition}_numeric_run_metrics.csv"
    base_rows = _read_csv(base_path)
    merged = {row["logical_key"]: dict(row) for row in base_rows}
    cache = PerformanceReplayCache()
    selected_overlay = {
        key: record for key, record in overlay_records.items() if key.endswith(f"::{condition}")
    }
    for key, record in selected_overlay.items():
        if key not in merged:
            raise ResultSummaryError(f"targeted noise key 不在基底: {key}")
        merged[key].update(_replay_overlay_final(record, repo_root=repo_root, cache=cache))
    if len(merged) != 2250:
        raise ResultSummaryError(f"{condition} numeric merge 不是 2250 行")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in merged.values():
        grouped[row["algorithm"]].append(row)
    output: list[dict[str, Any]] = []
    for algorithm in sorted(grouped):
        runs = grouped[algorithm]
        if len(runs) != 150:
            raise ResultSummaryError(f"{condition}/{algorithm} numeric 不是 150 行")
        valid_count = sum(_parse_bool(row["valid_output"]) for row in runs)
        unavailable = sum(row["evaluation_status"] == "replay_unavailable" for row in runs)
        id_mean = sum(float(row["id_quality"]) for row in runs) / 150
        ood_mean = sum(float(row["ood_quality"]) for row in runs) / 150
        clean_id = float(clean_scores[algorithm]["ID"]) / 100.0
        clean_ood = float(clean_scores[algorithm]["OOD"]) / 100.0
        id_retention = id_mean / clean_id if clean_id > 0.0 else 0.0
        ood_retention = ood_mean / clean_ood if clean_ood > 0.0 else 0.0
        overlay_count = sum(1 for key in selected_overlay if key.startswith(f"{algorithm}::"))
        output.append(
            {
                "algorithm": algorithm,
                "condition": condition,
                "run_count": 150,
                "valid_output_count": valid_count,
                "valid_output_rate": f"{valid_count / 150.0:.17g}",
                "ID": f"{100.0 * id_mean:.12g}",
                "OOD": f"{100.0 * ood_mean:.12g}",
                "clean_ID": f"{100.0 * clean_id:.12g}",
                "clean_OOD": f"{100.0 * clean_ood:.12g}",
                "id_quality_retention": f"{id_retention:.17g}",
                "id_quality_drop": f"{1.0 - id_retention:.17g}",
                "ood_quality_retention": f"{ood_retention:.17g}",
                "ood_quality_drop": f"{1.0 - ood_retention:.17g}",
                "targeted_overlay_run_count": overlay_count,
                "canonical_replay_unavailable_count": unavailable,
                "numeric_evaluation_path": EVALUATION_PATH,
                "formal_six_axis_condition": "false",
                "result_status": "supplementary_noise_numeric",
            }
        )
    if len(output) != 15:
        raise ResultSummaryError(f"{condition} 指标输出不是 15 行")
    return output


def _artifact_info(path: Path, row_count: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }
    if row_count is not None:
        info["row_count"] = row_count
    return info


def export_summary(stage_root: Path, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    current_eff_path = stage_root / "work/result_summary_20260905/clean_eff_run_metrics.csv"
    current_eff = _read_csv(current_eff_path)
    legacy_eff = _read_csv(stage_root / "results/clean_eff_run_metrics.csv")
    merged_eff, unavailable_keys = merge_clean_eff_rows(current_eff, legacy_eff)
    if unavailable_keys:
        raise ResultSummaryError(
            "严格原生 clean EFF 尚缺 "
            f"{len(unavailable_keys)} 条运行；禁止用 legacy proxy 回填"
        )
    clean_trajectory, eff_scores = aggregate_eff_rows(
        merged_eff,
        condition="clean",
        fallback_keys=set(),
        trajectory_basis="algorithm_native_internal_best_so_far.v1",
    )
    clean_six_axis = build_clean_six_axis(
        stage_root, eff_scores=eff_scores, eff_fallback_keys=set()
    )
    clean_scores = {row["algorithm"]: row for row in clean_six_axis}

    base_noise_paths = sorted(
        (stage_root / "work/noise_trajectory_freeze_v1/collected").glob("*.jsonl.gz")
    )
    overlay_noise_paths = sorted(
        (stage_root / "work/remote_collect/all_conditions_cpu_v2/collected").glob(
            "*.jsonl.gz"
        )
    )
    base_noise = _load_record_map(base_noise_paths)
    overlay_noise = _load_record_map(overlay_noise_paths)
    if len(base_noise) != 4500 or len(overlay_noise) != 143:
        raise ResultSummaryError(
            f"noise 轨迹基底/overlay 数量错误: {len(base_noise)}/{len(overlay_noise)}"
        )
    if not set(overlay_noise).issubset(base_noise):
        raise ResultSummaryError("noise targeted overlay 含基底外 logical_key")
    merged_noise = dict(base_noise)
    merged_noise.update(overlay_noise)

    outputs: dict[str, list[dict[str, Any]]] = {
        "clean_six_axis.csv": clean_six_axis,
        "clean_eff_180min.csv": clean_trajectory,
    }
    for condition in ("noise001", "noise005"):
        outputs[f"{condition}_metrics.csv"] = build_noise_metrics(
            stage_root,
            condition=condition,
            overlay_records=overlay_noise,
            clean_scores=clean_scores,
            repo_root=repo_root,
        )
        outputs[f"{condition}_eff_180min.csv"] = aggregate_noise_trajectory(
            merged_noise,
            condition=condition,
            overlay_keys=set(overlay_noise),
        )

    output_info: dict[str, Any] = {}
    for name, rows in outputs.items():
        path = output_dir / name
        _write_csv(path, rows)
        output_info[name] = _artifact_info(path, len(rows))
    manifest = {
        "schema_version": "stage5.result_summary_delivery.v1",
        "status": "ok",
        "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        "clean": {
            "formal_ready": False,
            "current_eff_run_count": len(current_eff),
            "legacy_eff_fallback_run_count": 0,
            "final_replacement_run_count": 75,
            "formal_blockers": [
                "最新 targeted symbolic 采用增量重聚合，尚未通过正式 aggregate_clean_metrics 总契约",
            ],
        },
        "noise": {
            "formal_six_axis_condition": False,
            "base_run_count": len(base_noise),
            "targeted_overlay_run_count": len(overlay_noise),
            "trajectory_basis": "observed_numeric_best_so_far_native_snapshot.v1",
        },
        "inputs": {
            "current_clean_eff": _artifact_info(current_eff_path, len(current_eff)),
            "legacy_clean_eff_audit_only": _artifact_info(
                stage_root / "results/clean_eff_run_metrics.csv", len(legacy_eff)
            ),
            "noise_base_bundles": [_artifact_info(path) for path in base_noise_paths],
            "noise_overlay_bundles": [_artifact_info(path) for path in overlay_noise_paths],
        },
        "outputs": output_info,
    }
    manifest_path = output_dir / "delivery_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    stage_root = repo_root / "AAAI_experiments/stage5_metric_calculation_0831"
    parser = argparse.ArgumentParser(description="导出 Stage5 六个结果汇总 CSV")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--stage-root", type=Path, default=stage_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=stage_root / "reports/result_summary_20260905",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        manifest = export_summary(
            args.stage_root.resolve(), args.repo_root.resolve(), args.output_dir.resolve()
        )
    except (ResultSummaryError, clean_aggregate.AggregateCleanMetricsError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
