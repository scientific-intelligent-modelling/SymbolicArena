"""聚合单个噪声条件的补充六轴指标与逐公式审计表。

噪声 EFF 仍来自 native minute snapshot 的 observed best-so-far 后处理，因而本模块
始终把整套输出标为 supplementary / formal_ready=false。符号与数值部分则严格绑定
当前 result SHA、canonical replay、冻结计划及 LLM 裁决依赖。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics import RunQuality, minimality_score, stability_score, symbolic_fidelity_score
from .symbolic_evidence import (
    SymbolicEvidenceError,
    build_symbolic_artifact,
    operator_f1,
    tree_similarity,
    variable_f1,
)


SEEDS = (520, 521, 522)
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))
EFF_HORIZON = 180
TRAJECTORY_BASIS = "observed_numeric_best_so_far_native_snapshot.v1"
RUN_EFF_BASIS = "algorithm_mean_from_supplementary_trajectory.v1"
CANONICAL_REPLAY_PATH = "canonical_replay.v1"
FORBIDDEN_SOURCE_TOKEN = "all_15alg_fullcpu_v1"
ALLOWED_CONDITIONS = {"noise001", "noise005"}
CONSISTENT_STRUCTURE_DECISIONS = {
    "mathematically_equivalent",
    "same_canonical_structure",
}
STRUCTURE_DECISIONS = CONSISTENT_STRUCTURE_DECISIONS | {
    "different_structure",
    "undetermined",
}
EQUIVALENCE_DECISIONS = {"equivalent", "not_equivalent", "undetermined"}


class AggregateNoiseSixAxisError(ValueError):
    """噪声六轴输入未满足闭环契约。"""


def _raise(message: str) -> None:
    raise AggregateNoiseSixAxisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _check_forbidden(value: object, *, context: str) -> None:
    if FORBIDDEN_SOURCE_TOKEN in str(value):
        _raise(f"{context} 命中禁止来源 {FORBIDDEN_SOURCE_TOKEN}")


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(f"{context} 必须是非空字符串")
    return value.strip()


def _bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    _raise(f"{context} 必须是 true/false")
    raise AssertionError("unreachable")


def _unit(value: object, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AggregateNoiseSixAxisError(f"{context} 不是合法数值") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        _raise(f"{context} 必须是 [0, 1] 内有限数值")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    _check_forbidden(path, context="输入路径")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        _raise(f"CSV 为空: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _check_forbidden(path, context="输入路径")
    rows: list[dict[str, Any]] = []
    handle_context = (
        gzip.open(path, mode="rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(mode="r", encoding="utf-8")
    )
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AggregateNoiseSixAxisError(
                    f"JSONL 第 {line_number} 行非法: {path}"
                ) from exc
            if not isinstance(row, dict):
                _raise(f"JSONL 第 {line_number} 行不是 object: {path}")
            _check_forbidden(_json_text(row), context=f"{path}:{line_number}")
            rows.append(row)
    if not rows:
        _raise(f"JSONL 为空: {path}")
    return rows


def _write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _algorithm_key(value: object) -> str:
    return _string(value, context="algorithm").casefold()


def _plan_identity(row: Mapping[str, Any], *, condition: str) -> tuple[str, str, int]:
    request = row.get("request")
    if not isinstance(request, Mapping):
        _raise(f"{row.get('logical_id')} 缺少 request")
    algorithm = request.get("algorithm_slug") or request.get("algorithm")
    dataset = request.get("dataset_id")
    seed = request.get("seed")
    if seed is None:
        task_id = str(request.get("task_id") or "")
        marker = "_s"
        try:
            seed = int(task_id.split(marker, 1)[1].split("_", 1)[0])
        except (IndexError, ValueError):
            _raise(f"{row.get('logical_id')} 无法解析 seed")
    if str(request.get("noise_tag") or row.get("condition")) != condition:
        _raise(f"{row.get('logical_id')} condition 漂移")
    return _algorithm_key(algorithm), _string(dataset, context="dataset_id"), int(seed)


def _load_plan_index_bundle(
    plan_path: Path,
    index_path: Path,
    *,
    task_type: str,
    condition: str,
    expected_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    plans = _read_jsonl(plan_path)
    indexes = _read_jsonl(index_path)
    if len(plans) != expected_rows or len(indexes) != expected_rows:
        _raise(
            f"{task_type} plan/index 行数应均为 {expected_rows}，实际 {len(plans)}/{len(indexes)}"
        )
    plan_map: dict[str, dict[str, Any]] = {}
    for row in plans:
        logical_id = _string(row.get("logical_id"), context=f"{task_type}.logical_id")
        if row.get("task_type") != task_type:
            _raise(f"{logical_id}.task_type 非法")
        row_condition = str(row.get("condition"))
        if task_type == "gt_simplify" and row_condition != "clean":
            _raise(f"{logical_id}.condition 应为 clean")
        if task_type != "gt_simplify" and row_condition != condition:
            _raise(f"{logical_id}.condition 应为 {condition}")
        if logical_id in plan_map:
            _raise(f"{task_type} plan logical_id 重复: {logical_id}")
        plan_map[logical_id] = row
    plan_sha = _sha256(plan_path)
    index_map: dict[str, dict[str, Any]] = {}
    for row in indexes:
        logical_id = _string(row.get("logical_id"), context=f"{task_type}.logical_id")
        plan = plan_map.get(logical_id)
        if plan is None:
            _raise(f"{task_type} index 找不到对应 plan: {logical_id}")
        if row.get("task_type") != task_type:
            _raise(f"{logical_id}.task_type 非法")
        if row.get("evaluation_key") != plan.get("evaluation_key"):
            _raise(f"{logical_id} evaluation_key 与 plan 不一致")
        bound_plan_sha = row.get("plan_sha256")
        if bound_plan_sha is not None and bound_plan_sha != plan_sha:
            _raise(f"{logical_id} plan_sha256 与计划文件不一致")
        if row.get("state") not in {"frozen", "non_applicable"}:
            _raise(f"{logical_id} 尚未形成可聚合终态")
        if logical_id in index_map:
            _raise(f"{task_type} index logical_id 重复: {logical_id}")
        index_map[logical_id] = row
    return plan_map, index_map


def _load_raw_results(
    path: Path, *, condition: str, expected_runs: int
) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        source = row.get("source")
        result = row.get("result")
        if not isinstance(source, Mapping) or not isinstance(result, Mapping):
            _raise("merged final raw 行缺少 source/result")
        if source.get("noise_tag") != condition:
            _raise("merged final raw condition 漂移")
        key = (
            _algorithm_key(source.get("algorithm")),
            _string(source.get("dataset_id"), context="raw.dataset_id"),
            int(source.get("seed")),
        )
        result_sha = _string(result.get("sha256"), context="raw.result.sha256")
        raw_text = _string(result.get("raw_text"), context="raw.result.raw_text")
        if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != result_sha:
            _raise(f"{key} raw_text SHA 与 result.sha256 不一致")
        if key in output:
            _raise(f"merged final raw key 重复: {key}")
        output[key] = dict(row)
    if len(output) != expected_runs:
        _raise(f"merged final raw 应为 {expected_runs} 行，实际 {len(output)}")
    return output


def _load_numeric(
    path: Path, *, condition: str, expected_runs: int
) -> tuple[dict[tuple[str, str, int], dict[str, str]], set[str], set[str]]:
    output: dict[tuple[str, str, int], dict[str, str]] = {}
    algorithms: set[str] = set()
    datasets: set[str] = set()
    for row in _read_csv(path):
        if row.get("noise_tag") != condition:
            _raise(f"numeric condition 应为 {condition}")
        key = (
            _algorithm_key(row.get("algorithm")),
            _string(row.get("dataset_id"), context="numeric.dataset_id"),
            int(row.get("seed") or -1),
        )
        expected_logical_key = (
            f"{row.get('algorithm')}::{key[1]}::s{key[2]}::{condition}"
        )
        if row.get("logical_key") != expected_logical_key:
            _raise(f"numeric logical_key 漂移: {row.get('logical_key')}")
        if row.get("evaluation_path") != CANONICAL_REPLAY_PATH:
            _raise(f"{expected_logical_key} 未走 {CANONICAL_REPLAY_PATH}")
        if row.get("evaluation_status") == "replay_unavailable" or row.get("replay_error"):
            _raise(f"{expected_logical_key} canonical replay unavailable，禁止聚合")
        valid = _bool(row.get("valid_output"), context=f"{expected_logical_key}.valid_output")
        id_quality = _unit(row.get("id_quality"), context=f"{expected_logical_key}.id_quality")
        ood_quality = _unit(row.get("ood_quality"), context=f"{expected_logical_key}.ood_quality")
        if not valid and (id_quality != 0.0 or ood_quality != 0.0):
            _raise(f"{expected_logical_key} 无效输出的数值质量必须为 0")
        if key in output:
            _raise(f"numeric key 重复: {key}")
        output[key] = row
        algorithms.add(key[0])
        datasets.add(key[1])
    if len(output) != expected_runs:
        _raise(f"numeric 应为 {expected_runs} 行，实际 {len(output)}")
    return output, algorithms, datasets


def _bundle_by_identity(
    plans: Mapping[str, dict[str, Any]],
    indexes: Mapping[str, dict[str, Any]],
    *,
    condition: str,
) -> dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]]:
    output = {}
    for logical_id, plan in plans.items():
        key = _plan_identity(plan, condition=condition)
        if key in output:
            _raise(f"计划运行身份重复: {key}")
        output[key] = (plan, indexes[logical_id])
    return output


def _gt_by_dataset(
    plans: Mapping[str, dict[str, Any]], indexes: Mapping[str, dict[str, Any]]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    output = {}
    for logical_id, plan in plans.items():
        request = plan.get("request")
        if not isinstance(request, Mapping):
            _raise(f"{logical_id} 缺少 request")
        dataset = _string(request.get("dataset_id"), context=f"{logical_id}.dataset_id")
        index = indexes[logical_id]
        if index.get("state") != "frozen":
            _raise(f"GT {logical_id} 必须为 frozen")
        if dataset in output:
            _raise(f"GT dataset 重复: {dataset}")
        output[dataset] = (plan, index)
    return output


def _effective_expression(
    plan: Mapping[str, Any], index: Mapping[str, Any], *, context: str
) -> tuple[str, str]:
    request = plan["request"]
    original = request.get("original_expression") or request.get("expression")
    original_text = _string(original, context=f"{context}.original_expression")
    effective = index.get("effective_expression")
    if not isinstance(effective, str) or not effective.strip():
        structured = index.get("structured_output")
        if isinstance(structured, Mapping):
            effective = structured.get("simplified_expression")
    return original_text, _string(effective, context=f"{context}.effective_expression")


def _pred_source_sha(request: Mapping[str, Any]) -> object:
    """兼容当前计划把 result SHA 放在 AST 来源证据中的真实结构。"""

    direct = request.get("result_raw_sha256")
    if direct is not None:
        return direct
    evidence = request.get("ast_source_evidence")
    return evidence.get("result_raw_sha256") if isinstance(evidence, Mapping) else None


def _load_eff(
    path: Path,
    *,
    condition: str,
    algorithms: set[str],
    expected_runs_per_algorithm: int,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    rows = _read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    normalized: list[dict[str, str]] = []
    for row in rows:
        if row.get("condition") != condition:
            _raise(f"EFF trajectory condition 应为 {condition}")
        algorithm_key = _algorithm_key(row.get("algorithm"))
        if algorithm_key not in algorithms:
            _raise(f"EFF trajectory 出现未知算法: {row.get('algorithm')}")
        if row.get("trajectory_basis") != TRAJECTORY_BASIS:
            _raise("噪声 EFF trajectory_basis 不是 observed numeric best-so-far")
        if _bool(row.get("formal_ready"), context="EFF.formal_ready"):
            _raise("噪声 native snapshot EFF 不得标记 formal_ready=true")
        if int(row.get("expected_run_count") or -1) != expected_runs_per_algorithm:
            _raise(f"{row.get('algorithm')} EFF expected_run_count 错误")
        if int(row.get("available_run_count") or -1) != expected_runs_per_algorithm:
            _raise(f"{row.get('algorithm')} EFF 运行覆盖不完整")
        if not math.isclose(float(row.get("coverage_rate") or -1), 1.0):
            _raise(f"{row.get('algorithm')} EFF coverage_rate 不是 1")
        copied = dict(row)
        copied["trajectory_basis"] = TRAJECTORY_BASIS
        copied["formal_ready"] = "false"
        copied["result_status"] = "supplementary_noise_eff"
        grouped[algorithm_key].append(copied)
        normalized.append(copied)
    expected_rows = len(algorithms) * EFF_HORIZON
    if len(normalized) != expected_rows:
        _raise(f"EFF 180min 应为 {expected_rows} 行，实际 {len(normalized)}")
    final_scores: dict[str, float] = {}
    for algorithm, group in grouped.items():
        group.sort(key=lambda item: int(item["minute"]))
        if [int(item["minute"]) for item in group] != list(range(1, EFF_HORIZON + 1)):
            _raise(f"{algorithm} EFF 分钟网格不完整")
        score = float(group[-1]["cumulative_eff_score"])
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            _raise(f"{algorithm} EFF 最终分数不在 [0, 100]")
        final_scores[algorithm] = score / 100.0
    normalized.sort(key=lambda item: (_algorithm_key(item["algorithm"]), int(item["minute"])))
    return normalized, final_scores


def _structure_identity(
    plan: Mapping[str, Any], *, condition: str
) -> tuple[str, str, tuple[int, int]]:
    request = plan.get("request")
    if not isinstance(request, Mapping):
        _raise(f"{plan.get('logical_id')} 缺少 request")
    left = request.get("seed_a", request.get("seed_left"))
    right = request.get("seed_b", request.get("seed_right"))
    pair = tuple(sorted((int(left), int(right))))
    if pair not in SEED_PAIRS:
        _raise(f"{plan.get('logical_id')} seed pair 非法")
    if str(request.get("noise_tag") or plan.get("condition")) != condition:
        _raise(f"{plan.get('logical_id')} condition 漂移")
    return (
        _algorithm_key(request.get("algorithm_slug") or request.get("algorithm")),
        _string(request.get("dataset_id"), context="structure.dataset_id"),
        pair,
    )


def aggregate_noise_six_axis(
    *,
    condition: str,
    numeric_csv: Path,
    raw_results_jsonl: Path,
    pred_plan_jsonl: Path,
    pred_index_jsonl: Path,
    gt_plan_jsonl: Path,
    gt_index_jsonl: Path,
    equivalence_plan_jsonl: Path,
    equivalence_index_jsonl: Path,
    structure_plan_jsonl: Path,
    structure_index_jsonl: Path,
    eff_180min_csv: Path,
    six_axis_csv: Path,
    run_formulas_csv: Path,
    task_stability_csv: Path,
    output_eff_180min_csv: Path,
    report_json: Path,
    expected_algorithms: int = 15,
    expected_datasets: int = 50,
) -> dict[str, Any]:
    """校验并聚合一个噪声条件；不会调用 LLM，也不会补造缺失裁决。"""

    if condition not in ALLOWED_CONDITIONS:
        _raise(f"condition 仅允许 {sorted(ALLOWED_CONDITIONS)}")
    expected_runs = expected_algorithms * expected_datasets * len(SEEDS)
    expected_tasks = expected_algorithms * expected_datasets
    expected_structure = expected_tasks * len(SEED_PAIRS)
    numeric, algorithms, datasets = _load_numeric(
        numeric_csv, condition=condition, expected_runs=expected_runs
    )
    if len(algorithms) != expected_algorithms or len(datasets) != expected_datasets:
        _raise("numeric 的算法数或数据集数不符合期望")
    expected_grid = {
        (algorithm, dataset, seed)
        for algorithm in algorithms
        for dataset in datasets
        for seed in SEEDS
    }
    if set(numeric) != expected_grid:
        _raise("numeric 不是 algorithm x dataset x 3 seeds 的完整笛卡尔网格")
    raw = _load_raw_results(raw_results_jsonl, condition=condition, expected_runs=expected_runs)
    if set(raw) != expected_grid:
        _raise("merged final raw 与 numeric 网格不一致")

    pred_plans, pred_indexes = _load_plan_index_bundle(
        pred_plan_jsonl,
        pred_index_jsonl,
        task_type="pred_simplify",
        condition=condition,
        expected_rows=expected_runs,
    )
    eq_plans, eq_indexes = _load_plan_index_bundle(
        equivalence_plan_jsonl,
        equivalence_index_jsonl,
        task_type="equivalence",
        condition=condition,
        expected_rows=expected_runs,
    )
    gt_plans, gt_indexes = _load_plan_index_bundle(
        gt_plan_jsonl,
        gt_index_jsonl,
        task_type="gt_simplify",
        condition=condition,
        expected_rows=expected_datasets,
    )
    structure_plans, structure_indexes = _load_plan_index_bundle(
        structure_plan_jsonl,
        structure_index_jsonl,
        task_type="stab_structure",
        condition=condition,
        expected_rows=expected_structure,
    )
    pred = _bundle_by_identity(pred_plans, pred_indexes, condition=condition)
    equivalence = _bundle_by_identity(eq_plans, eq_indexes, condition=condition)
    gt = _gt_by_dataset(gt_plans, gt_indexes)
    if set(pred) != expected_grid or set(equivalence) != expected_grid or set(gt) != datasets:
        _raise("GT/pred/equivalence 网格与 numeric 不一致")

    structure: dict[tuple[str, str, tuple[int, int]], tuple[dict[str, Any], dict[str, Any]]] = {}
    for logical_id, plan in structure_plans.items():
        key = _structure_identity(plan, condition=condition)
        if key in structure:
            _raise(f"structure pair 重复: {key}")
        structure[key] = (plan, structure_indexes[logical_id])
    expected_structure_grid = {
        (algorithm, dataset, pair)
        for algorithm in algorithms
        for dataset in datasets
        for pair in SEED_PAIRS
    }
    if set(structure) != expected_structure_grid:
        _raise("structure 不是 algorithm x dataset x 3 seed pairs 的完整网格")

    eff_rows, eff_by_algorithm = _load_eff(
        eff_180min_csv,
        condition=condition,
        algorithms=algorithms,
        expected_runs_per_algorithm=expected_datasets * len(SEEDS),
    )

    gt_artifacts: dict[str, dict[str, object]] = {}
    gt_expressions: dict[str, tuple[str, str]] = {}
    for dataset, (plan, index) in gt.items():
        original, effective = _effective_expression(plan, index, context=f"GT/{dataset}")
        try:
            gt_artifacts[dataset] = build_symbolic_artifact(effective)
        except SymbolicEvidenceError as exc:
            raise AggregateNoiseSixAxisError(f"GT/{dataset} 无法规范化: {exc}") from exc
        gt_expressions[dataset] = (original, effective)

    run_rows: list[dict[str, Any]] = []
    grouped_tasks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    display_names = {key: row["algorithm"] for key, row in numeric.items()}
    for key in sorted(expected_grid):
        algorithm, dataset, seed = key
        numeric_row = numeric[key]
        raw_row = raw[key]
        pred_plan, pred_index = pred[key]
        eq_plan, eq_index = equivalence[key]
        raw_source = raw_row["source"]
        raw_result = raw_row["result"]
        result_sha = _string(raw_result.get("sha256"), context=f"{key}.result_sha256")
        if numeric_row.get("result_sha256") != result_sha:
            _raise(f"{key} numeric 与 raw result SHA 不一致")
        if str(raw_source.get("task_id")) != numeric_row.get("task_id"):
            _raise(f"{key} task_id 在 numeric/raw 间漂移")
        request = pred_plan["request"]
        if _pred_source_sha(request) != result_sha:
            _raise(f"{key} pred plan 未绑定当前 result SHA")
        if str(request.get("task_id")) != numeric_row.get("task_id"):
            _raise(f"{key} pred plan task_id 漂移")

        gt_plan, gt_index = gt[dataset]
        gt_eval = gt_plan.get("evaluation_key")
        pred_eval = pred_plan.get("evaluation_key")
        dependencies = set(eq_plan.get("dependencies") or [])
        if dependencies != {gt_eval, pred_eval}:
            _raise(f"{key} equivalence dependencies 未精确绑定 GT/pred")
        eq_request = eq_plan.get("request")
        if not isinstance(eq_request, Mapping):
            _raise(f"{key} equivalence request 缺失")
        if eq_request.get("prediction_result_sha256") != result_sha:
            _raise(f"{key} equivalence 未绑定当前 result SHA")
        if eq_request.get("prediction_frozen_evaluation_key") != pred_eval:
            _raise(f"{key} equivalence pred evaluation key 漂移")
        if eq_request.get("ground_truth_frozen_evaluation_key") != gt_eval:
            _raise(f"{key} equivalence GT evaluation key 漂移")

        valid_output = _bool(numeric_row["valid_output"], context=f"{key}.valid_output")
        symbolic_valid = pred_index.get("state") == "frozen"
        original_pred = request.get("original_expression") or request.get("expression") or ""
        effective_pred = ""
        c_pred = 0
        tree = variables = operators = 0.0
        if symbolic_valid:
            original_pred, effective_pred = _effective_expression(
                pred_plan, pred_index, context=f"pred/{key}"
            )
            try:
                pred_artifact = build_symbolic_artifact(effective_pred)
            except SymbolicEvidenceError as exc:
                raise AggregateNoiseSixAxisError(f"pred/{key} 无法规范化: {exc}") from exc
            gt_artifact = gt_artifacts[dataset]
            c_pred = int(pred_artifact["node_count"])
            tree = tree_similarity(gt_artifact, pred_artifact)
            variables = variable_f1(gt_artifact, pred_artifact)
            operators = operator_f1(gt_artifact, pred_artifact)
            if eq_index.get("state") != "frozen":
                _raise(f"{key} 有符号表达式但 equivalence 未 frozen")
            structured = eq_index.get("structured_output")
            if not isinstance(structured, Mapping):
                _raise(f"{key} equivalence structured_output 缺失")
            decision = str(structured.get("decision"))
            if decision not in EQUIVALENCE_DECISIONS:
                _raise(f"{key} equivalence decision 非法")
        else:
            if eq_index.get("state") != "non_applicable":
                _raise(f"{key} pred non_applicable 时 equivalence 也必须 non_applicable")
            decision = "non_applicable"
        equivalent = decision == "equivalent"
        c_ref = int(gt_artifacts[dataset]["node_count"])
        m_sym = symbolic_fidelity_score(
            equivalent=equivalent,
            tree_similarity=tree,
            variable_f1=variables,
            operator_f1=operators,
            valid=symbolic_valid,
        )
        m_min = minimality_score(c_ref, c_pred if c_pred else 1, valid=symbolic_valid)
        row = {
            "logical_key": numeric_row["logical_key"],
            "algorithm": numeric_row["algorithm"],
            "dataset_id": dataset,
            "seed": seed,
            "condition": condition,
            "task_id": numeric_row["task_id"],
            "result_sha256": result_sha,
            "valid_output": str(valid_output).lower(),
            "original_prediction_expression": str(original_pred),
            "effective_prediction_expression": effective_pred,
            "original_ground_truth_expression": gt_expressions[dataset][0],
            "effective_ground_truth_expression": gt_expressions[dataset][1],
            "equivalence_decision": decision,
            "equivalent": str(equivalent).lower(),
            "tree_similarity": f"{tree:.17g}",
            "variable_f1": f"{variables:.17g}",
            "operator_f1": f"{operators:.17g}",
            "C_ref": c_ref,
            "C_pred": c_pred,
            "m_sym": f"{m_sym:.17g}",
            "m_min": f"{m_min:.17g}",
            "id_quality": f"{_unit(numeric_row['id_quality'], context='id_quality'):.17g}",
            "ood_quality": f"{_unit(numeric_row['ood_quality'], context='ood_quality'):.17g}",
            "m_eff": f"{eff_by_algorithm[algorithm]:.17g}",
            "m_eff_basis": RUN_EFF_BASIS,
            "numeric_evaluation_path": CANONICAL_REPLAY_PATH,
            "trajectory_basis": TRAJECTORY_BASIS,
            "formal_ready": "false",
            "gt_logical_id": gt_plan["logical_id"],
            "pred_logical_id": pred_plan["logical_id"],
            "equivalence_logical_id": eq_plan["logical_id"],
        }
        run_rows.append(row)
        grouped_tasks[(algorithm, dataset)].append(row)

    task_rows: list[dict[str, Any]] = []
    for (algorithm, dataset), rows in sorted(grouped_tasks.items()):
        rows.sort(key=lambda item: int(item["seed"]))
        if [int(item["seed"]) for item in rows] != list(SEEDS):
            _raise(f"{algorithm}/{dataset} 的 3 seeds 不完整")
        valid_by_seed = {int(item["seed"]): _bool(item["valid_output"], context="valid") for item in rows}
        symbolic_by_seed = {int(item["seed"]): int(item["C_pred"]) > 0 for item in rows}
        pair_results: list[bool] = []
        pair_decisions: dict[tuple[int, int], str] = {}
        for pair in SEED_PAIRS:
            plan, index = structure[(algorithm, dataset, pair)]
            structure_request = plan.get("request")
            if not isinstance(structure_request, Mapping):
                _raise(f"{algorithm}/{dataset}/{pair} structure request 缺失")
            expected_dependencies = {
                pred[(algorithm, dataset, pair[0])][0]["evaluation_key"],
                pred[(algorithm, dataset, pair[1])][0]["evaluation_key"],
            }
            if set(plan.get("dependencies") or []) != expected_dependencies:
                _raise(f"{algorithm}/{dataset}/{pair} structure dependencies 漂移")
            expected_result_shas = {
                str(raw[(algorithm, dataset, pair[0])]["result"]["sha256"]),
                str(raw[(algorithm, dataset, pair[1])]["result"]["sha256"]),
            }
            bound_result_shas = {
                str(structure_request.get("prediction_a_result_sha256") or ""),
                str(structure_request.get("prediction_b_result_sha256") or ""),
            }
            if bound_result_shas != expected_result_shas:
                _raise(f"{algorithm}/{dataset}/{pair} structure 未绑定两侧当前 result SHA")
            if index.get("state") == "frozen":
                if not all(valid_by_seed[seed] and symbolic_by_seed[seed] for seed in pair):
                    _raise(f"{algorithm}/{dataset}/{pair} frozen structure 两侧必须有效")
                structured = index.get("structured_output")
                if not isinstance(structured, Mapping):
                    _raise(f"{algorithm}/{dataset}/{pair} structure output 缺失")
                decision = str(structured.get("decision"))
                if decision not in STRUCTURE_DECISIONS:
                    _raise(f"{algorithm}/{dataset}/{pair} structure decision 非法")
            else:
                if all(valid_by_seed[seed] and symbolic_by_seed[seed] for seed in pair):
                    _raise(f"{algorithm}/{dataset}/{pair} 两侧有效却缺少结构裁决")
                decision = "non_applicable"
            pair_decisions[pair] = decision
            pair_results.append(decision in CONSISTENT_STRUCTURE_DECISIONS)
        quality_rows = [
            RunQuality(
                id_quality=float(item["id_quality"]),
                ood_quality=float(item["ood_quality"]),
                valid=_bool(item["valid_output"], context="valid_output"),
            )
            for item in rows
        ]
        result = stability_score(quality_rows, structural_pair_results=pair_results)
        task_rows.append(
            {
                "algorithm": rows[0]["algorithm"],
                "dataset_id": dataset,
                "condition": condition,
                "seed_520_valid_output": str(valid_by_seed[520]).lower(),
                "seed_521_valid_output": str(valid_by_seed[521]).lower(),
                "seed_522_valid_output": str(valid_by_seed[522]).lower(),
                "pair_520_521": pair_decisions[(520, 521)],
                "pair_520_522": pair_decisions[(520, 522)],
                "pair_521_522": pair_decisions[(521, 522)],
                "numerical_consistency": f"{result.numerical_consistency:.17g}",
                "validity": f"{result.validity:.17g}",
                "structural_consistency": f"{result.structural_consistency:.17g}",
                "m_stab": f"{result.score:.17g}",
                "formal_ready": "false",
            }
        )
    if len(task_rows) != expected_tasks:
        _raise(f"task stability 应为 {expected_tasks} 行，实际 {len(task_rows)}")

    algorithm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    algorithm_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        algorithm_runs[_algorithm_key(row["algorithm"])].append(row)
    for row in task_rows:
        algorithm_tasks[_algorithm_key(row["algorithm"])].append(row)
    six_rows: list[dict[str, Any]] = []
    for algorithm in sorted(algorithms):
        runs = algorithm_runs[algorithm]
        tasks = algorithm_tasks[algorithm]
        if len(runs) != expected_datasets * len(SEEDS) or len(tasks) != expected_datasets:
            _raise(f"{algorithm} 的 run/task 数量不闭合")
        mean = lambda field, source: 100.0 * sum(float(row[field]) for row in source) / len(source)
        six_rows.append(
            {
                "algorithm": display_names[next(key for key in numeric if key[0] == algorithm)],
                "condition": condition,
                "run_count": len(runs),
                "task_count": len(tasks),
                "ID": f"{mean('id_quality', runs):.17g}",
                "OOD": f"{mean('ood_quality', runs):.17g}",
                "SYM": f"{mean('m_sym', runs):.17g}",
                "MIN": f"{mean('m_min', runs):.17g}",
                "EFF": f"{100.0 * eff_by_algorithm[algorithm]:.17g}",
                "STAB": f"{mean('m_stab', tasks):.17g}",
                "trajectory_basis": TRAJECTORY_BASIS,
                "result_status": "supplementary_noise_six_axis",
                "formal_ready": "false",
            }
        )

    run_fields = list(run_rows[0])
    task_fields = list(task_rows[0])
    six_fields = list(six_rows[0])
    eff_fields = list(eff_rows[0])
    _write_csv_atomic(run_formulas_csv, run_fields, run_rows)
    _write_csv_atomic(task_stability_csv, task_fields, task_rows)
    _write_csv_atomic(six_axis_csv, six_fields, six_rows)
    _write_csv_atomic(output_eff_180min_csv, eff_fields, eff_rows)
    outputs = {
        "six_axis_csv": {"path": str(six_axis_csv.resolve()), "sha256": _sha256(six_axis_csv), "row_count": len(six_rows)},
        "run_formulas_csv": {"path": str(run_formulas_csv.resolve()), "sha256": _sha256(run_formulas_csv), "row_count": len(run_rows)},
        "task_stability_csv": {"path": str(task_stability_csv.resolve()), "sha256": _sha256(task_stability_csv), "row_count": len(task_rows)},
        "eff_180min_csv": {"path": str(output_eff_180min_csv.resolve()), "sha256": _sha256(output_eff_180min_csv), "row_count": len(eff_rows)},
    }
    report = {
        "status": "ok",
        "condition": condition,
        "formal_ready": False,
        "result_status": "supplementary_noise_six_axis",
        "trajectory_basis": TRAJECTORY_BASIS,
        "run_eff_basis": RUN_EFF_BASIS,
        "summary": {
            "algorithm_count": len(six_rows),
            "dataset_count": len(datasets),
            "run_count": len(run_rows),
            "task_count": len(task_rows),
            "seed_pair_count": len(structure),
            "eff_row_count": len(eff_rows),
            "canonical_replay_unavailable_count": 0,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "numeric_csv": numeric_csv,
                "raw_results_jsonl": raw_results_jsonl,
                "pred_plan_jsonl": pred_plan_jsonl,
                "pred_index_jsonl": pred_index_jsonl,
                "gt_plan_jsonl": gt_plan_jsonl,
                "gt_index_jsonl": gt_index_jsonl,
                "equivalence_plan_jsonl": equivalence_plan_jsonl,
                "equivalence_index_jsonl": equivalence_index_jsonl,
                "structure_plan_jsonl": structure_plan_jsonl,
                "structure_index_jsonl": structure_index_jsonl,
                "eff_180min_csv": eff_180min_csv,
            }.items()
        },
        "outputs": outputs,
    }
    _write_json_atomic(report_json, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="聚合单个 noise 条件的补充六轴指标")
    parser.add_argument("--condition", required=True, choices=sorted(ALLOWED_CONDITIONS))
    for name in (
        "numeric_csv",
        "raw_results_jsonl",
        "pred_plan_jsonl",
        "pred_index_jsonl",
        "gt_plan_jsonl",
        "gt_index_jsonl",
        "equivalence_plan_jsonl",
        "equivalence_index_jsonl",
        "structure_plan_jsonl",
        "structure_index_jsonl",
        "eff_180min_csv",
        "six_axis_csv",
        "run_formulas_csv",
        "task_stability_csv",
        "output_eff_180min_csv",
        "report_json",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--expected-algorithms", type=int, default=15)
    parser.add_argument("--expected-datasets", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = vars(parser.parse_args(argv))
    try:
        aggregate_noise_six_axis(**args)
    except (AggregateNoiseSixAxisError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
