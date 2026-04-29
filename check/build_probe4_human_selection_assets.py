#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("exp-planning/02.E1选择验证")
INPUT_TABLE = ROOT / "e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv"
CANDIDATE_TABLE = ROOT / "generated/candidate200_unified.csv"
RAW_RESULTS = ROOT / "e1_final_results_20260424-041046_clean/all_results.jsonl"
SEMANTIC_LLM_RESULTS = ROOT / (
    "semantic200_llm_physics_v2_comparison_20260430/semantic_results_raw.csv"
)
OUTPUT_DIR = ROOT / "probe4_v02_readable_selection_20260429"

LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12.0
LOG_CLIP_MAX = 12.0
EXPLOSION_THRESHOLD = 100.0

TAXONOMY = {
    "dso": "rl_policy",
    "drsr": "llm_assisted_rl",
    "e2esr": "pretrained_neural",
    "gplearn": "classic_gp",
    "imcts": "mcts",
    "llmsr": "llm_symbolic",
    "pyoperon": "evolutionary_gp",
    "pysr": "evolutionary_gp",
    "qlattice": "graph_hybrid",
    "ragsr": "rag_hybrid",
    "tpsr": "pretrained_neural",
    "udsr": "rl_hybrid",
}

ARITHMETIC_LINEAR = {"add", "sub"}
ARITHMETIC_PRODUCT = {"mul", "div", "pow"}
TRIG = {"sin", "cos", "tan", "asin", "acos", "atan"}
EXP_LOG = {"exp", "log"}
ROOT_ABS_INV = {"sqrt", "abs", "inv"}
CONSTANT_SYMBOLS = {"pi", "e"}
COMPLEX_SYMBOLS = {"imaginaryunit", "I"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out >= 0 else None


def _log_nmse(value: Any) -> float | None:
    num = _float(value)
    if num is None:
        return None
    out = math.log10(max(num, LOG_FLOOR))
    return min(LOG_CLIP_MAX, max(LOG_CLIP_MIN, out))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _global_index(dataset_id: str) -> int | None:
    text = str(dataset_id or "").strip()
    if text.startswith("g") and text[1:].isdigit():
        return int(text[1:])
    return None


def _dataset_id(index: Any) -> str:
    try:
        return f"g{int(index):04d}"
    except (TypeError, ValueError):
        return ""


def _load_candidates() -> dict[int, dict[str, str]]:
    _, rows = _read_csv(CANDIDATE_TABLE)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        idx = _global_index(_dataset_id(row.get("global_index")))
        if idx is not None:
            out[idx] = row
    return out


def _load_expression_artifacts() -> dict[tuple[str, str], dict[str, Any]]:
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    if not RAW_RESULTS.exists():
        return artifacts
    with RAW_RESULTS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset_id = _dataset_id(row.get("global_index"))
            algorithm = str(row.get("tool") or "")
            result = row.get("result") or {}
            artifact = result.get("canonical_artifact") or {}
            artifacts[(dataset_id, algorithm)] = {
                "result_status": result.get("status") or row.get("status") or "",
                "outer_status": row.get("status") or "",
                "wall_time_seconds": result.get("seconds") or row.get("seconds") or "",
                "equation": result.get("equation") or "",
                "equation_count": result.get("equation_count") or "",
                "canonical_artifact": artifact,
                "canonical_artifact_error": result.get("canonical_artifact_error") or "",
                "source_result_path": row.get("source_result_path") or "",
                "host": row.get("host") or "",
                "wave": row.get("wave") or "",
            }
    return artifacts


def _operator_categories(operators: set[str]) -> dict[str, str]:
    return {
        "uses_linear_ops": _flag(bool(operators & ARITHMETIC_LINEAR)),
        "uses_product_power_ops": _flag(bool(operators & ARITHMETIC_PRODUCT)),
        "uses_trig_ops": _flag(bool(operators & TRIG)),
        "uses_exp_log_ops": _flag(bool(operators & EXP_LOG)),
        "uses_root_abs_inv_ops": _flag(bool(operators & ROOT_ABS_INV)),
        "uses_constant_symbols": _flag(bool(operators & CONSTANT_SYMBOLS)),
        "uses_complex_symbols": _flag(bool(operators & COMPLEX_SYMBOLS)),
    }


def _complexity_bucket(ast_node_count: int | None) -> str:
    if ast_node_count is None:
        return "unknown_no_artifact"
    if ast_node_count <= 1:
        return "constant_or_trivial"
    if ast_node_count <= 20:
        return "simple"
    if ast_node_count <= 80:
        return "medium"
    if ast_node_count <= 300:
        return "complex"
    return "very_complex"


def _int_from_numeric(value: Any) -> int | None:
    number = _float(value)
    if number is None:
        return None
    return int(number)


def _preview(text: Any, max_len: int = 180) -> str:
    if text is None:
        return ""
    value = str(text).replace("\n", "\\n")
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[-()+\\*/^,]", text))


def _health_label(finite_id_ood: bool, explosion: bool, artifact_available: bool, artifact_valid: bool) -> str:
    if not finite_id_ood:
        return "metric_missing"
    if explosion:
        return "finite_but_exploded"
    if not artifact_available:
        return "metric_ok_no_expression_artifact"
    if not artifact_valid:
        return "metric_ok_artifact_invalid"
    return "metric_ok_artifact_ok"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalized_status(raw_status: str, metric_success: bool, budget_exhausted: bool = False) -> str:
    if metric_success and (raw_status == "timed_out" or budget_exhausted):
        return "success_budget_exhausted"
    if metric_success and raw_status == "ok":
        return "success_completed"
    if metric_success and not raw_status:
        return "success_metrics_available_raw_status_unknown"
    if metric_success:
        return f"success_raw_{raw_status}"
    if raw_status == "timed_out":
        return "metric_missing_after_budget_exhausted"
    if raw_status:
        return f"metric_missing_raw_{raw_status}"
    return "metric_missing_raw_status_unknown"


def _load_semantic_llm_overrides() -> dict[tuple[str, str], dict[str, Any]]:
    """加载带物理语义的 llmsr/drsr 结果，用来替换旧的无语义 E1 结果。"""
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    if not SEMANTIC_LLM_RESULTS.exists():
        return overrides
    _, rows = _read_csv(SEMANTIC_LLM_RESULTS)
    for row in rows:
        algorithm = str(row.get("algorithm") or "")
        if algorithm not in {"llmsr", "drsr"}:
            continue
        dataset_id = str(row.get("dataset_id") or "")
        operators = [item for item in str(row.get("operator_set") or "").split(";") if item]
        variables = [item for item in str(row.get("variables") or "").split(";") if item]
        artifact_valid = _truthy(row.get("artifact_valid"))
        artifact: dict[str, Any] = {
            "artifact_valid": artifact_valid,
            # semantic_results_raw 是从 canonical artifact 摘要表生成的；没有单独存 sympy 标记时，
            # 用 artifact_valid 作为表达式可解析性的保守替代。
            "sympy_parse_ok": artifact_valid,
            "normalized_expression": row.get("normalized_expression") or "",
            "raw_equation": "",
            "variables": variables,
            "operator_set": operators,
            "ast_node_count": row.get("ast_node_count") or "",
            "tree_depth": row.get("tree_depth") or "",
            "expected_n_features": row.get("n_features") or "",
            "parameter_symbols": [],
        }
        overrides[(dataset_id, algorithm)] = {
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "train_nmse": row.get("train_nmse") or "",
            "valid_nmse": row.get("valid_nmse") or "",
            "id_nmse": row.get("id_nmse") or "",
            "ood_nmse": row.get("ood_nmse") or "",
            "result_status": row.get("status") or "",
            "outer_status": row.get("status") or "",
            "budget_exhausted": _truthy(row.get("budget_exhausted")),
            "wall_time_seconds": row.get("seconds") or "",
            "equation": row.get("normalized_expression") or "",
            "equation_count": "1" if row.get("normalized_expression") else "",
            "canonical_artifact": artifact,
            "canonical_artifact_error": "",
            "source_result_path": row.get("result_path") or "",
            "host": row.get("host") or "",
            "wave": "semantic200",
            "prompt_semantics_mode": "physics_semantic_hidden_mapping",
            "llm_model_assignment": row.get("llm_model_assignment") or "",
            "semantic_background_preview": row.get("background_preview") or "",
        }
    return overrides


def _enrich_row(
    row: dict[str, str],
    candidates: dict[int, dict[str, str]],
    artifacts: dict[tuple[str, str], dict[str, Any]],
    semantic_overrides: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    dataset_id = row["dataset_id"]
    algorithm = row["algorithm"]
    idx = _global_index(dataset_id)
    candidate = candidates.get(idx or -1, {})
    semantic_override = semantic_overrides.get((dataset_id, algorithm), {})
    artifact_info = semantic_override or artifacts.get((dataset_id, algorithm), {})
    artifact = artifact_info.get("canonical_artifact") or {}

    train_value = semantic_override.get("train_nmse", row.get("train_nmse"))
    valid_value = semantic_override.get("valid_nmse", row.get("valid_nmse"))
    id_value = semantic_override.get("id_nmse", row.get("id_nmse"))
    ood_value = semantic_override.get("ood_nmse", row.get("ood_nmse"))

    train = _float(train_value)
    id_nmse = _float(id_value)
    ood = _float(ood_value)
    log_train = _log_nmse(train_value)
    log_id = _log_nmse(id_value)
    log_ood = _log_nmse(ood_value)
    finite_train = train is not None
    finite_id = id_nmse is not None
    finite_ood = ood is not None
    finite_id_ood = finite_id and finite_ood
    finite_train_id_ood = finite_train and finite_id_ood
    id_ood_explosion = any(value is not None and value > EXPLOSION_THRESHOLD for value in (id_nmse, ood))
    train_id_ood_explosion = any(value is not None and value > EXPLOSION_THRESHOLD for value in (train, id_nmse, ood))

    variables = artifact.get("variables")
    if not isinstance(variables, list):
        variables = []
    operators = artifact.get("operator_set")
    if not isinstance(operators, list):
        operators = []
    operators_set = {str(op).lower() for op in operators}
    params = artifact.get("parameter_symbols")
    if not isinstance(params, list):
        params = []
    expected_n_features = artifact.get("expected_n_features")
    expected_n_features_int = _int_from_numeric(expected_n_features)

    variable_count = len(set(map(str, variables)))
    variable_coverage = (
        variable_count / expected_n_features_int
        if expected_n_features_int and expected_n_features_int > 0
        else None
    )
    artifact_available = bool(artifact)
    artifact_valid = bool(artifact.get("artifact_valid")) if artifact_available else False
    sympy_parse_ok = bool(artifact.get("sympy_parse_ok")) if artifact_available else False
    raw_equation = artifact_info.get("equation") or artifact.get("raw_equation") or ""
    normalized = artifact.get("normalized_expression") or ""
    ast_nodes = _int_from_numeric(artifact.get("ast_node_count")) if artifact_available else None
    tree_depth = _int_from_numeric(artifact.get("tree_depth")) if artifact_available else None

    category_flags = _operator_categories(operators_set)
    operator_category_count = sum(int(value) for key, value in category_flags.items() if key != "uses_complex_symbols")
    health = _health_label(finite_id_ood, id_ood_explosion, artifact_available, artifact_valid)
    raw_status = str(artifact_info.get("result_status", "") or "")
    budget_exhausted = bool(artifact_info.get("budget_exhausted")) or raw_status == "timed_out"
    probe4_success = finite_train_id_ood

    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "global_index": idx if idx is not None else "",
        "dataset_name": row.get("dataset_name", ""),
        "family": row.get("family") or candidate.get("family", ""),
        "subgroup": candidate.get("subgroup", ""),
        "srsd_variant": row.get("srsd_variant", ""),
        "basename": candidate.get("basename", ""),
        "selection_mode": candidate.get("selection_mode", ""),
        "candidate_advantage_side": candidate.get("candidate_advantage_side", ""),
        "algorithm": algorithm,
        "taxonomy": TAXONOMY.get(algorithm, "unknown"),
        "prompt_semantics_mode": artifact_info.get("prompt_semantics_mode", "none_or_original_e1"),
        "llm_model_assignment": artifact_info.get("llm_model_assignment", ""),
        "semantic_background_preview": artifact_info.get("semantic_background_preview", ""),
        "train_nmse": train_value or "",
        "valid_nmse_raw_observed_not_used_for_probe4": valid_value or "",
        "id_nmse": id_value or "",
        "ood_nmse": ood_value or "",
        "finite_train": _flag(finite_train),
        "finite_id": _flag(finite_id),
        "finite_ood": _flag(finite_ood),
        "finite_id_ood": _flag(finite_id_ood),
        "finite_train_id_ood": _flag(finite_train_id_ood),
        "log_train_nmse_clipped": _fmt(log_train),
        "log_id_nmse_clipped": _fmt(log_id),
        "log_ood_nmse_clipped": _fmt(log_ood),
        "combined_log_id_ood_nmse": _fmt(0.5 * log_id + 0.5 * log_ood if log_id is not None and log_ood is not None else None),
        "gap_log_ood_minus_id": _fmt(log_ood - log_id if log_id is not None and log_ood is not None else None),
        "delta_id_minus_train": _fmt(log_id - log_train if log_train is not None and log_id is not None else None),
        "delta_ood_minus_id": _fmt(log_ood - log_id if log_id is not None and log_ood is not None else None),
        "id_ood_explosion_gt_100": _flag(id_ood_explosion),
        "train_id_ood_explosion_gt_100": _flag(train_id_ood_explosion),
        "raw_result_status": raw_status,
        "outer_status": artifact_info.get("outer_status", ""),
        "budget_exhausted": _flag(budget_exhausted),
        "probe4_success": _flag(probe4_success),
        "normalized_status_for_probe4": _normalized_status(raw_status, probe4_success, budget_exhausted),
        "wall_time_seconds": artifact_info.get("wall_time_seconds", ""),
        "expression_artifact_available": _flag(artifact_available),
        "equation_present": _flag(bool(raw_equation)),
        "artifact_valid": _flag(artifact_valid),
        "sympy_parse_ok": _flag(sympy_parse_ok),
        "canonical_artifact_error": artifact_info.get("canonical_artifact_error", ""),
        "raw_equation_char_count": len(str(raw_equation)) if raw_equation else "",
        "normalized_expression_char_count": len(str(normalized)) if normalized else "",
        "raw_equation_token_count": _token_count(str(raw_equation)) if raw_equation else "",
        "normalized_expression_token_count": _token_count(str(normalized)) if normalized else "",
        "ast_node_count": ast_nodes if ast_nodes is not None else "",
        "tree_depth": tree_depth if tree_depth is not None else "",
        "expression_complexity_bucket": _complexity_bucket(ast_nodes),
        "expected_n_features": expected_n_features_int if expected_n_features_int is not None else "",
        "used_variable_count": variable_count if artifact_available else "",
        "variable_coverage_ratio": _fmt(variable_coverage),
        "uses_all_features": _flag(variable_coverage is not None and variable_coverage >= 0.999),
        "uses_no_features_constant": _flag(artifact_available and variable_count == 0),
        "operator_count": len(operators_set) if artifact_available else "",
        "operator_set": ";".join(sorted(operators_set)),
        "operator_category_count": operator_category_count if artifact_available else "",
        "parameter_count": len(params) if artifact_available else "",
        "expression_health_label": health,
        "raw_equation_preview": _preview(raw_equation),
        "normalized_expression_preview": _preview(normalized),
        "source_result_path": artifact_info.get("source_result_path", ""),
    }
    out.update(category_flags)
    return out


def _median(values: list[float]) -> str:
    return _fmt(statistics.median(values)) if values else ""


def _mean(values: list[float]) -> str:
    return _fmt(statistics.mean(values)) if values else ""


def _rate(num: int, den: int) -> str:
    return _fmt(num / den) if den else ""


def _algorithm_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_alg[str(row["algorithm"])].append(row)
    out: list[dict[str, Any]] = []
    for alg, group in sorted(by_alg.items()):
        total = len(group)
        combined = [_float(row.get("combined_log_id_ood_nmse")) for row in group]
        combined = [v for v in combined if v is not None]
        ast = [_float(row.get("ast_node_count")) for row in group]
        ast = [v for v in ast if v is not None]
        variable_coverage = [_float(row.get("variable_coverage_ratio")) for row in group]
        variable_coverage = [v for v in variable_coverage if v is not None]
        health_counts = Counter(str(row.get("expression_health_label", "")) for row in group)
        raw_status_counts = Counter(str(row.get("raw_result_status", "")) for row in group if row.get("raw_result_status"))
        normalized_status_counts = Counter(str(row.get("normalized_status_for_probe4", "")) for row in group)
        out.append(
            {
                "algorithm": alg,
                "taxonomy": TAXONOMY.get(alg, "unknown"),
                "rows": total,
                "finite_train_id_ood_rate": _rate(sum(row["finite_train_id_ood"] == "1" for row in group), total),
                "probe4_success_rate": _rate(sum(row["probe4_success"] == "1" for row in group), total),
                "budget_exhausted_success_rate": _rate(
                    sum(row["probe4_success"] == "1" and row["budget_exhausted"] == "1" for row in group),
                    total,
                ),
                "finite_id_ood_rate": _rate(sum(row["finite_id_ood"] == "1" for row in group), total),
                "id_ood_explosion_gt_100_rate": _rate(sum(row["id_ood_explosion_gt_100"] == "1" for row in group), total),
                "expression_artifact_available_rate": _rate(sum(row["expression_artifact_available"] == "1" for row in group), total),
                "artifact_valid_rate_among_all": _rate(sum(row["artifact_valid"] == "1" for row in group), total),
                "sympy_parse_ok_rate_among_all": _rate(sum(row["sympy_parse_ok"] == "1" for row in group), total),
                "median_combined_log_id_ood_nmse": _median(combined),
                "median_ast_node_count": _median(ast),
                "mean_variable_coverage_ratio": _mean(variable_coverage),
                "uses_trig_rate": _rate(sum(row["uses_trig_ops"] == "1" for row in group), total),
                "uses_exp_log_rate": _rate(sum(row["uses_exp_log_ops"] == "1" for row in group), total),
                "uses_product_power_rate": _rate(sum(row["uses_product_power_ops"] == "1" for row in group), total),
                "uses_complex_symbol_rate": _rate(sum(row["uses_complex_symbols"] == "1" for row in group), total),
                "health_counts": json.dumps(health_counts, ensure_ascii=False, sort_keys=True),
                "raw_status_counts": json.dumps(raw_status_counts, ensure_ascii=False, sort_keys=True),
                "normalized_status_counts": json.dumps(normalized_status_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return out


def _dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_id"])].append(row)
    out: list[dict[str, Any]] = []
    for dataset_id, group in sorted(by_dataset.items()):
        combined = [_float(row.get("combined_log_id_ood_nmse")) for row in group]
        combined = [v for v in combined if v is not None]
        out.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": group[0].get("dataset_name", ""),
                "family": group[0].get("family", ""),
                "subgroup": group[0].get("subgroup", ""),
                "srsd_variant": group[0].get("srsd_variant", ""),
                "algorithm_count": len(group),
                "finite_id_ood_count": sum(row["finite_id_ood"] == "1" for row in group),
                "explosion_gt_100_count": sum(row["id_ood_explosion_gt_100"] == "1" for row in group),
                "expression_artifact_available_count": sum(row["expression_artifact_available"] == "1" for row in group),
                "median_combined_log_id_ood_nmse": _median(combined),
                "spread_combined_log_id_ood_nmse": _fmt(max(combined) - min(combined) if combined else None),
                "high_information_hint": _flag(bool(combined) and (max(combined) - min(combined) >= 6.0)),
            }
        )
    return out


def _write_human_docs(output_dir: Path, run_rows: list[dict[str, Any]], alg_rows: list[dict[str, Any]]) -> None:
    artifact_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    for row in run_rows:
        alg = str(row["algorithm"])
        total_counts[alg] += 1
        if row["expression_artifact_available"] == "1":
            artifact_counts[alg] += 1
    full_artifact_algorithms = sorted(alg for alg, total in total_counts.items() if artifact_counts[alg] == total)
    partial_artifact_algorithms = sorted(
        alg for alg, total in total_counts.items() if 0 < artifact_counts[alg] < total
    )
    no_artifact_algorithms = sorted(alg for alg, total in total_counts.items() if artifact_counts[alg] == 0)
    top_explosion = sorted(
        alg_rows,
        key=lambda row: float(row["id_ood_explosion_gt_100_rate"] or 0),
        reverse=True,
    )[:5]
    lines = [
        "# Probe-4 v0.2 人话版选择包",
        "",
        "这个文件夹只放 Probe-4 v0.2 的可读结论和可计算大表，不再混放旧版 `nmse_only` 输出。",
        "",
        "## 我们现在到底要选什么",
        "",
        "我们已经有 `200 个数据集 x 12 个算法 = 2400 条实验结果`。下一步不是找 4 个平均分最高的算法，而是找 4 个最适合当“探针”的算法。",
        "",
        "探针的作用是：后面用这 4 个算法去跑更大的 664 数据集池，尽量低成本复现完整算法面板会看到的数据集差异。",
        "",
        "换句话说，Probe-4 要回答的是：",
        "",
        "```text",
        "如果只允许用 4 个算法观察数据集，哪 4 个算法最不容易看偏？",
        "```",
        "",
        "## 每个评分项说人话",
        "",
        "- `stability`：这个算法是不是经常能给出 train/id/ood 三类有效数值。经常缺结果的算法不能当主探针。",
        "- `discrimination_fidelity`：这 4 个算法认为“哪些数据集最有区分度”，是否接近完整算法面板的判断。这是最核心的项。",
        "- `complementarity`：4 个算法是不是看问题的角度不同。如果两个算法在所有数据集上同涨同跌，它们同时入选的价值就低。",
        "- `selected_dataset_coverage`：这 4 个算法挑出来的高信息量数据集，不能全堆在某个 family 或 subgroup。",
        "- `gradient_diversity`：不同算法的退化路径是否不同。这里现在只看 `train -> id` 和 `id -> ood`，不再用 valid。",
        "- `baseline_quality`：探针不能全是烂算法。它不要求最强，但至少要有基本可用的拟合能力。",
        "- `taxonomy_penalty`：避免 4 个算法都来自同一类方法，例如全是 GP。",
        "- `missing_penalty`：缺 train/id/ood 指标越多，扣分越多。",
        "- `explosion_penalty`：只要 ID/OOD NMSE 超过 100，就认为这个 run 对实际评价有爆炸风险。爆炸多的算法不能因为方差大而被奖励。",
        "",
        "## timeout 的语义",
        "",
        "`timed_out` 不等于失败。这里的 timeout 只是说明算法跑满了一小时预算。如果它在预算结束时已经落盘了 train/ID/OOD 指标，我们在 Probe-4 选择里把它记为 `success_budget_exhausted`。",
        "",
        "这正是本轮实验要看的问题：给算法一小时，它在这个预算内能交出多好的结果。",
        "",
        "## 这次大表额外加了什么",
        "",
        "除了原始 NMSE，大表还加入了三类信息：",
        "",
        "- 数值健康：finite 标记、log NMSE、train 到 ID 的退化、ID 到 OOD 的退化、NMSE > 100 爆炸标记。",
        "- 表达式结构：表达式长度、token 数、AST 节点数、树深度、用了几个变量、变量覆盖率、用了哪些算子类别。",
        "- 工程健康：是否有表达式 artifact、artifact 是否有效、是否能被 sympy 解析、原始状态、是否预算用尽、Probe-4 口径下是否成功。",
        "",
        "## LLM 算法结果口径",
        "",
        "`llmsr` 和 `drsr` 现在使用带物理语义背景的新批次结果，替换掉旧 E1 里不带语义的结果。",
        "这个替换只发生在这两个算法上，其它 10 个算法仍使用原 Candidate-200 E1 结果。",
        "",
        "语义批次的 prompt 会告诉模型目标物理量、候选变量语义角色集合和 dummy 变量数量，但不会告诉它具体哪个 `x_i` 对应哪个物理角色。",
        "大表里的 `prompt_semantics_mode = physics_semantic_hidden_mapping` 就表示该行来自这个新口径。",
        "",
        "## 表达式 artifact 覆盖情况",
        "",
        f"- 表达式 artifact 完整覆盖的算法：`{', '.join(full_artifact_algorithms)}`。",
        f"- 表达式 artifact 部分覆盖的算法：`{', '.join(partial_artifact_algorithms) or '无'}`。",
        f"- 当前 2400 行 digest 里存在、但 clean artifact 归档完全没有覆盖的算法：`{', '.join(no_artifact_algorithms)}`。",
        "- 对 artifact 缺失的算法，大表不会伪造表达式复杂度，而是把 `expression_artifact_available` 标成 `0`。",
        "",
        "## 当前最需要警惕的信号",
        "",
        "按 `ID/OOD NMSE > 100` 的爆炸率看，风险最高的算法是：",
        "",
        "| algorithm | explosion_gt_100_rate | artifact_available_rate | median_ast_nodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in top_explosion:
        lines.append(
            f"| `{row['algorithm']}` | {float(row['id_ood_explosion_gt_100_rate'] or 0):.3f} | "
            f"{float(row['expression_artifact_available_rate'] or 0):.3f} | {row['median_ast_node_count']} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `probe4_run_level_big_table.csv`：2400 行大表，每行是一个数据集和一个算法的结果，包含 NMSE、派生数值指标、表达式结构指标和工程健康指标。",
            "- `probe4_algorithm_level_summary.csv`：按算法汇总后的表，用来快速看每个算法是否稳定、是否容易爆炸、表达式复杂度是否过高。",
            "- `probe4_dataset_level_summary.csv`：按数据集汇总后的表，用来快速看哪些数据集在 12 个算法之间拉开了差异。",
            "- `probe4_metric_dictionary_human.md`：每个字段的人话解释。",
            "",
            "## 后续怎么用",
            "",
            "先用 `probe4_algorithm_level_summary.csv` 排除明显不适合作主探针的算法，再用 `probe4_run_level_big_table.csv` 计算组合级分数。最终 shortlist 不能只看一个总分，还要检查 PySR/no-PySR、RAGSR/no-RAGSR、with/without DRSR 的敏感性。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    process_lines = [
        "# 如何从 2400 条实验结果选出 4 个探针",
        "",
        "这份文件只讲流程，不堆公式。",
        "",
        "## 第一步：把每条实验结果变成可比较记录",
        "",
        "每一行代表一个算法在一个数据集上的表现。原始表里有 train、valid、ID、OOD 的 NMSE。Probe-4 当前只用 train、ID、OOD 三类；valid 只是保留原始记录，不进入派生指标。",
        "",
        "`llmsr` 和 `drsr` 这两类 LLM 算法已经切换成带物理语义背景的新批次结果；其它算法仍使用原 E1 结果。这样 Probe-4 评估的是“后续正式会使用的语义 LLM 口径”，而不是旧的无语义 LLM 口径。",
        "",
        "我们先判断这条记录是不是能用：train、ID、OOD 是否都是正常数字；ID 或 OOD 有没有超过 100；如果超过 100，就认为这条结果存在爆炸风险。",
        "",
        "这里要特别注意：`timed_out` 不自动算失败。很多算法本来就是按一小时预算跑，跑满预算后把当前最好表达式和指标落盘，这在本实验里就是成功。我们会把这种情况标成 `success_budget_exhausted`。",
        "",
        "## 第二步：把巨大 NMSE 压到可比较尺度",
        "",
        "有些 NMSE 会大到 1e100。如果直接比较，所有排序都会被这些极端值支配。所以我们把 NMSE 转成 log 尺度，并限制在一个固定范围里。这样做不是为了美化结果，而是为了让“爆炸”被记录为风险，同时不让它吞掉所有其它信息。",
        "",
        "## 第三步：看算法是怎么退化的",
        "",
        "我们看两段退化：",
        "",
        "- train 到 ID：如果这里变差很多，说明算法在同分布测试上就不稳。",
        "- ID 到 OOD：如果这里变差很多，说明算法外推能力差。",
        "",
        "两个算法即使最终 NMSE 差不多，只要退化路径不同，也可能是互补的探针。",
        "",
        "## 第四步：把表达式结构也加进来",
        "",
        "只看 NMSE 不够。符号回归还要看表达式是不是可解释、是不是过长、是不是只用了一个变量、是不是用了三角函数、log、sqrt、幂等结构。",
        "",
        "所以我们从 canonical artifact 里抽：表达式长度、token 数、AST 节点数、树深度、变量覆盖率、算子类别、是否能被 sympy 解析。",
        "",
        "如果某个算法当前没有 artifact 归档，就明确标记缺失，不补假数据。",
        "",
        "## 第五步：先评价单个算法能不能当探针",
        "",
        "一个算法适不适合当探针，不是看它是不是最强，而是看：",
        "",
        "- 它是不是大多数任务都能给出 train/ID/OOD 指标。",
        "- 它是不是经常爆炸。",
        "- 它的表达式是不是极端复杂或不可解析。",
        "- 它是不是只会输出常数或只用很少变量。",
        "- 它是不是只在某一类数据集上有用。",
        "",
        "这一层会淘汰明显不适合做主探针的算法，或者把它们标记为高风险。",
        "",
        "## 第六步：再评价 4 个算法组合",
        "",
        "4 个探针不是 4 个单算法分数相加。我们要看这个组合整体是不是能代表 12 算法面板。",
        "",
        "具体看：",
        "",
        "- 它们挑出来的高信息量数据集，和 12 个算法整体认为高信息量的数据集是否接近。",
        "- 它们的失败模式是否互补，而不是四个算法都在同一批题上同涨同跌。",
        "- 它们认为重要的数据集是否覆盖不同 family/subgroup。",
        "- 它们的方法类型是否多样，不要全是同一种 GP 或同一种 LLM-hybrid。",
        "- 它们是否因为缺失和爆炸太多而产生假区分度。",
        "",
        "## 第七步：做敏感性检查",
        "",
        "最终不能只看一个最高分组合。至少要检查：",
        "",
        "- 去掉 PySR 后，最优组合是否变化很大。",
        "- 去掉 RAGSR 后，最优组合是否变化很大。",
        "- 允许 DRSR 进入候选时，结果是否变化很大。",
        "- 用 all-12、no-LLM、no-stage1 三种 teacher panel 时，组合排名是否稳定。",
        "",
        "如果某个组合只在一个特殊设置下好看，就不能直接冻结。",
        "",
        "## 第八步：冻结 Probe-4",
        "",
        "只有当一个组合同时满足稳定、互补、覆盖、不严重爆炸、敏感性检查不过分依赖某个算法时，才适合进入下一轮 `4 x 664 x 3 seeds`。",
        "",
        "当前这个文件夹先提供基础大表和算法级摘要，下一步可以在这套表上实现组合级打分和敏感性检查。",
        "",
    ]
    (output_dir / "probe4_selection_process_human.md").write_text(
        "\n".join(process_lines), encoding="utf-8"
    )

    metric_lines = [
        "# Probe-4 大表字段解释",
        "",
        "这份字段解释故意不用公式堆砌，只说明每个指标在实际决策里有什么用。",
        "",
        "## 数值健康",
        "",
        "- `finite_train`：训练集 NMSE 是不是一个非负有限数。它不是“训练成功”的严格证明，只说明 train_nmse 能参与计算。",
        "- `finite_id`：ID test NMSE 是否可计算。",
        "- `finite_ood`：OOD test NMSE 是否可计算。",
        "- `finite_id_ood`：ID 和 OOD 两个最终评价 split 是否都可计算。",
        "- `finite_train_id_ood`：train、ID、OOD 三类是否都可计算。Probe-4 现在不要求 valid。",
        "- `prompt_semantics_mode`：该行是否使用物理语义 prompt。`physics_semantic_hidden_mapping` 表示 `llmsr/drsr` 新语义批次；`none_or_original_e1` 表示原 E1 口径。",
        "- `llm_model_assignment`：语义批次中实际使用的 LLM 模型分配。只用于审计，不作为 Probe-4 评分项。",
        "- `semantic_background_preview`：语义 prompt 背景摘要预览。只用于审计，不作为 Probe-4 评分项。",
        "- `combined_log_id_ood_nmse`：把 ID 和 OOD 的误差压到 log 尺度后取平均。它用于避免极端大数直接支配表格。",
        "- `gap_log_ood_minus_id`：OOD 比 ID 坏多少。越大说明外推退化越明显。",
        "- `delta_id_minus_train`：ID 比 train 坏多少。它反映从训练拟合到同分布测试是否稳定。",
        "- `delta_ood_minus_id`：OOD 比 ID 坏多少。它反映分布外泛化是否崩。",
        "- `id_ood_explosion_gt_100`：ID 或 OOD NMSE 是否超过 100。超过就算实际评价风险很高。",
        "- `train_id_ood_explosion_gt_100`：train、ID、OOD 任一超过 100。",
        "",
        "## 表达式结构",
        "",
        "- `expression_artifact_available`：是否有可解析的表达式归档。没有归档就不能计算表达式复杂度。",
        "- `artifact_valid`：工具集 canonicalizer 是否认为表达式 artifact 合法。",
        "- `sympy_parse_ok`：表达式是否能被 sympy 解析。",
        "- `raw_equation_char_count`：原始表达式字符长度。过长通常解释性差。",
        "- `normalized_expression_char_count`：归一化表达式字符长度。",
        "- `raw_equation_token_count`：原始表达式大概有多少 token。",
        "- `normalized_expression_token_count`：归一化后表达式大概有多少 token。",
        "- `ast_node_count`：表达式树节点数。比字符长度更接近结构复杂度。",
        "- `tree_depth`：表达式树有多深。很深的表达式通常更难解释，也更容易数值不稳定。",
        "- `expression_complexity_bucket`：把表达式粗分成 trivial/simple/medium/complex/very_complex，方便快速筛查。",
        "- `used_variable_count`：表达式用了多少个输入变量。",
        "- `variable_coverage_ratio`：表达式用到的变量数占数据集特征数的比例。",
        "- `uses_all_features`：是否使用了全部输入变量。",
        "- `uses_no_features_constant`：是否基本是常数表达式。",
        "- `operator_set`：表达式里出现过哪些算子。",
        "- `operator_category_count`：算子类别覆盖数。类别多不一定好，但能表示表达式结构更丰富。",
        "- `uses_trig_ops`：是否用 sin/cos/tan 等周期函数。",
        "- `uses_exp_log_ops`：是否用 exp/log。",
        "- `uses_product_power_ops`：是否用乘除幂。",
        "- `uses_root_abs_inv_ops`：是否用 sqrt/abs/inv。",
        "- `uses_complex_symbols`：是否出现复数相关符号。这通常需要额外审计。",
        "",
        "## 工程健康",
        "",
        "- `raw_result_status`：原始 result 的状态，例如 ok 或 timed_out。它只记录运行器看到的原始状态，不直接等于 Probe-4 成功/失败。",
        "- `budget_exhausted`：是否跑满预算。`1` 通常对应原始 `timed_out`。",
        "- `probe4_success`：Probe-4 选择口径下是否成功。只要 train、ID、OOD 指标都能落盘并可计算，就算成功；即使原始状态是 timed_out 也算成功。",
        "- `normalized_status_for_probe4`：把原始状态翻译成适合本实验的状态。例如 `success_budget_exhausted` 表示跑满预算但结果可用。",
        "- `wall_time_seconds`：这条 run 大概用了多久。",
        "- `expression_health_label`：把数值健康和 artifact 健康合成的人话标签，例如 `metric_ok_artifact_ok` 或 `finite_but_exploded`。",
        "",
        "## 为什么这些对选 Probe-4 有用",
        "",
        "Probe 不是只要误差低。一个好 probe 应该稳定、有不同失败模式、能产出可解释表达式，并且不会靠大量爆炸值制造虚假的区分度。这个大表就是为了把这些因素都显式摆出来。",
        "",
    ]
    (output_dir / "probe4_metric_dictionary_human.md").write_text("\n".join(metric_lines), encoding="utf-8")


def main() -> None:
    _, nmse_rows = _read_csv(INPUT_TABLE)
    candidates = _load_candidates()
    artifacts = _load_expression_artifacts()
    semantic_overrides = _load_semantic_llm_overrides()
    run_rows = [_enrich_row(row, candidates, artifacts, semantic_overrides) for row in nmse_rows]
    alg_rows = _algorithm_summary(run_rows)
    dataset_rows = _dataset_summary(run_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "probe4_run_level_big_table.csv", run_rows)
    _write_csv(OUTPUT_DIR / "probe4_algorithm_level_summary.csv", alg_rows)
    _write_csv(OUTPUT_DIR / "probe4_dataset_level_summary.csv", dataset_rows)
    _write_human_docs(OUTPUT_DIR, run_rows, alg_rows)

    report = {
        "output_dir": str(OUTPUT_DIR),
        "run_rows": len(run_rows),
        "algorithms": len({row["algorithm"] for row in run_rows}),
        "datasets": len({row["dataset_id"] for row in run_rows}),
        "expression_artifact_rows": sum(row["expression_artifact_available"] == "1" for row in run_rows),
        "semantic_llm_override_rows": sum(
            row["prompt_semantics_mode"] == "physics_semantic_hidden_mapping" for row in run_rows
        ),
        "semantic_llm_override_source": str(SEMANTIC_LLM_RESULTS),
    }
    (OUTPUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
