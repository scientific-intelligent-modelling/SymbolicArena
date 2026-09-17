"""以当前 gplearn 原生终点和 Opus4.8 冻结裁决聚合六轴。

不读取旧 gplearn 符号分数。Eq 的 undetermined 按未证实等价计算部分 SYM；
结构 undetermined 计零但保留原标签。缺 frozen 不是算法失败，仍为 unresolved。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .claude_contract import canonical_json
from .gplearn_native_prefix_evidence import (
    PrefixEvidenceError,
    build_inventory_prefix_evidence,
    build_prefix_evidence,
)
from .gplearn_current_metrics import GplearnMetricError, score_symbolic_run, structure_is_positive
from .metrics import (
    RunQuality,
    minimality_score,
    numerical_consistency,
    stability_score,
)
from .symbolic_evidence import build_symbolic_artifact

SEEDS = (520, 521, 522)
PAIRS = ((520, 521), (520, 522), (521, 522))
EQ_DECISIONS = {"equivalent", "not_equivalent", "undetermined"}
STRUCT_DECISIONS = {
    "mathematically_equivalent", "same_canonical_structure",
    "different_structure", "undetermined",
}


class GplearnAggregateError(ValueError):
    """来源、依赖或裁决不满足当前终点聚合合同。"""


def _sha_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["condition"]), str(row["dataset_id"]), int(row["seed"])


def _index(rows: list[dict[str, Any]], key_fn, label: str) -> dict[Any, dict[str, Any]]:
    indexed = {}
    for row in rows:
        key = key_fn(row)
        if key in indexed:
            raise GplearnAggregateError(f"{label} 重复身份 {key}")
        indexed[key] = row
    return indexed


def _finite_unit(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GplearnAggregateError(f"{label} 缺少数值") from exc
    if not math.isfinite(result) or result < 0 or result > 1:
        raise GplearnAggregateError(f"{label} 不在 [0,1]")
    return result


def _frozen_decision(
    plan: Mapping[str, Any], frozen: Mapping[str, Any] | None, *, task_type: str,
) -> tuple[str | None, str | None]:
    if frozen is None:
        return None, "frozen_missing"
    if frozen.get("status") != "frozen" or frozen.get("model") != "claude-opus-4-8":
        return None, "frozen_status_or_model_invalid"
    if frozen.get("source_evaluation_key") != plan.get("evaluation_key") or frozen.get("source_input_hash") != plan.get("input_hash"):
        return None, "frozen_source_plan_mismatch"
    if frozen.get("prompt_sha256") != plan.get("prompt_sha256") or frozen.get("schema_sha256") != plan.get("schema_sha256"):
        return None, "frozen_prompt_schema_mismatch"
    if frozen.get("request") is not None and frozen.get("request") != plan.get("request"):
        return None, "frozen_request_mismatch"
    if frozen.get("plan_dependencies") != plan.get("dependencies") or frozen.get("task_type") != task_type:
        return None, "frozen_dependencies_mismatch"
    evidence = plan["request"].get("deterministic_evidence")
    if not isinstance(evidence, dict) or evidence.get("evidence_sha256") != _sha_json({
        field: value for field, value in evidence.items() if field != "evidence_sha256"
    }) or plan["request"].get("evidence_hash") != evidence["evidence_sha256"]:
        return None, "plan_evidence_hash_mismatch"
    semantic = frozen.get("semantic_validation") or {}
    structured = frozen.get("structured_output") or {}
    decision = structured.get("decision")
    allowed = EQ_DECISIONS if task_type == "equivalence" else STRUCT_DECISIONS
    if semantic.get("status") != "promotable" or semantic.get("decision") != decision or decision not in allowed:
        return None, "decision_not_promotable"
    if semantic.get("evidence_sha256") not in (None, evidence["evidence_sha256"]):
        return None, "decision_evidence_hash_mismatch"
    return str(decision), None


def aggregate_gplearn(
    *,
    inventory_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    strict_numeric_rows: list[dict[str, Any]],
    prediction_frozen_by_model: Mapping[str, dict[str, Any]],
    equivalence_plan_rows: list[dict[str, Any]],
    equivalence_frozen_by_source: Mapping[str, dict[str, Any]],
    structure_plan_rows: list[dict[str, Any]],
    structure_frozen_by_source: Mapping[str, dict[str, Any]],
    prior_algorithm_rows: list[dict[str, Any]],
    expected_runs: int = 450,
    expected_runs_per_condition: int = 150,
    expected_tasks_per_condition: int = 50,
) -> dict[str, Any]:
    inventory = _index(inventory_rows, _key, "inventory")
    gt = _index(gt_rows, lambda row: row["dataset_id"], "GT")
    numeric = _index(
        [row for row in strict_numeric_rows if str(row.get("algorithm", "")).casefold() == "gplearn"],
        _key, "strict numeric",
    )
    eq_plans = _index(equivalence_plan_rows, lambda row: _key(row["request"]), "Eq plan")
    pair_plans = _index(
        structure_plan_rows,
        lambda row: (str(row["request"]["condition"]), str(row["request"]["dataset_id"]),
                     int(row["request"]["seed_left"]), int(row["request"]["seed_right"])),
        "structure plan",
    )
    if set(inventory) != set(numeric) or set(inventory) != set(eq_plans):
        raise GplearnAggregateError("inventory、strict 数值和 Eq 计划运行键不一致")
    if len(pair_plans) != len(inventory):
        raise GplearnAggregateError("结构计划数量应等于运行数量")
    gt_artifacts = {}
    for dataset, row in gt.items():
        gt_artifacts[dataset] = build_symbolic_artifact(row["fixed_reference_expression"])

    run_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    runs_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run_key, row in sorted(inventory.items()):
        condition, dataset, seed = run_key
        issues: list[str] = []
        if condition not in {"clean", "noise001", "noise005"} or seed not in SEEDS:
            raise GplearnAggregateError(f"非法 gplearn 运行键 {run_key}")
        native = build_inventory_prefix_evidence(row)
        number = numeric[run_key]
        gt_row = gt.get(dataset)
        if gt_row is None or gt_row.get("gt_frozen_evaluation_key") != row.get("gt_frozen_evaluation_key") or gt_row.get("gt_source_evidence_sha256") != row.get("gt_reference_source_sha256") or gt_row.get("fixed_reference_expression") != row.get("gt_reference_expression"):
            raise GplearnAggregateError(f"{run_key} 固定 GT 不匹配")
        for source_field, numeric_field in (
            ("terminal_expression_sha256", "terminal_expression_sha256"),
            ("terminal_snapshot_sha256", "terminal_source_sha256"),
            ("selected_result_sha256", "selected_result_sha256"),
        ):
            if row[source_field] != number.get(numeric_field):
                raise GplearnAggregateError(f"{run_key} strict 数值终点 {source_field} 不匹配")
        if number.get("valid_output") != "True" or number.get("numeric_ready") != "True":
            raise GplearnAggregateError(f"{run_key} strict 数值未就绪")
        id_q = _finite_unit(number.get("id_quality"), f"{run_key}.ID")
        ood_q = _finite_unit(number.get("ood_quality"), f"{run_key}.OOD")
        m_eff = _finite_unit(number.get("m_eff"), f"{run_key}.EFF")
        eq_plan = eq_plans[run_key]
        eq_request = eq_plan["request"]
        eq_evidence = eq_request["deterministic_evidence"]
        lhs, rhs = eq_evidence["lhs_binding"], eq_evidence["rhs_binding"]
        pred_key = rhs.get("frozen_evaluation_key")
        pred_frozen = prediction_frozen_by_model.get(pred_key)
        if pred_frozen is None or pred_frozen.get("status") != "frozen" or pred_frozen.get("model") != "claude-opus-4-8":
            issues.append("prediction_frozen_missing_or_invalid")
        else:
            if pred_frozen.get("source_evaluation_key") != rhs.get("prediction_source_evaluation_key") or pred_frozen.get("semantic_validation", {}).get("effective_expression") != rhs.get("frozen_effective_expression"):
                raise GplearnAggregateError(f"{run_key} 预测 source key/有效式漂移")
            response_sha = pred_frozen.get("_source_file_sha256")
            if response_sha is not None and response_sha != rhs.get("prediction_response_sha256"):
                raise GplearnAggregateError(f"{run_key} 预测 response SHA 漂移")
        if eq_plan.get("dependencies") != [gt_row["gt_frozen_evaluation_key"], pred_key] or lhs.get("frozen_effective_expression") != gt_row["fixed_reference_expression"] or lhs.get("gt_result_sha256") != gt_row.get("gt_result_sha256") or lhs.get("gt_source_evidence_sha256") != gt_row.get("gt_source_evidence_sha256") or rhs.get("native_prefix_sha256") != row["native_prefix_sha256"] or rhs.get("terminal_snapshot_sha256") != row["terminal_snapshot_sha256"] or rhs.get("terminal_expression_sha256") != row["terminal_expression_sha256"] or rhs.get("source_result_sha256") != row["selected_result_sha256"] or rhs.get("variable_mapping") != row["variable_mapping"]:
            raise GplearnAggregateError(f"{run_key} Eq 计划依赖当前终点失败")
        if pred_frozen is not None:
            selected_version = next((
                version for version in rhs.get("prediction_plan_versions", [])
                if version.get("model_bound_evaluation_key") == pred_key
            ), None)
            if selected_version is not None and (
                selected_version.get("source_evaluation_key") != pred_frozen.get("source_evaluation_key")
                or selected_version.get("prompt_sha256") != pred_frozen.get("prompt_sha256")
                or selected_version.get("input_hash") != pred_frozen.get("source_input_hash")
            ):
                raise GplearnAggregateError(f"{run_key} 预测 base/retry 版本绑定漂移")
        effective = rhs["frozen_effective_expression"]
        typed = build_prefix_evidence(effective, row["feature_names"])
        reference = gt_artifacts[dataset]
        eq_frozen = equivalence_frozen_by_source.get(eq_plan["evaluation_key"])
        eq_decision, eq_issue = _frozen_decision(eq_plan, eq_frozen, task_type="equivalence")
        if eq_issue:
            issues.append("equivalence_" + eq_issue)
        try:
            symbolic = score_symbolic_run(typed, reference, eq_decision or "not_established")
        except GplearnMetricError:
            issues.append("typed_symbolic_component_unavailable")
            symbolic = {
                "tree_similarity": None, "variable_f1": None, "operator_f1": None,
                "c_pred": typed["node_count"], "c_ref": reference["node_count"],
                "m_sym": None,
                "m_min": minimality_score(reference["node_count"], typed["node_count"]),
            }
        m_sym = symbolic["m_sym"] if eq_decision is not None else None
        result = {
            "condition": condition, "algorithm": "gplearn", "dataset_id": dataset, "seed": seed,
            "logical_key": row["logical_key"],
            "native_prefix_sha256": row["native_prefix_sha256"],
            "terminal_expression_sha256": row["terminal_expression_sha256"],
            "terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
            "selected_result_sha256": row["selected_result_sha256"],
            "prediction_source_key": rhs.get("prediction_source_evaluation_key"),
            "prediction_model_key": pred_key,
            "prediction_response_sha256": rhs.get("prediction_response_sha256"),
            "prediction_selection_source": rhs.get("prediction_selection_source"),
            "effective_typed_expression": effective,
            "typed_exact_fingerprint": typed["exact_fingerprint"],
            "typed_structure_fingerprint": typed["constants_abstracted_structure_fingerprint"],
            "gt_key": gt_row["gt_frozen_evaluation_key"],
            "eq_source_key": eq_plan["evaluation_key"],
            "eq_model_key": eq_frozen.get("evaluation_key") if eq_frozen else None,
            "eq_response_sha256": eq_frozen.get("_source_file_sha256") if eq_frozen else None,
            "eq_plan_evidence_sha256": eq_evidence.get("evidence_sha256"),
            "equivalence_decision": eq_decision,
            "id_quality": id_q, "ood_quality": ood_q, "m_eff": m_eff,
            "tree_similarity": symbolic["tree_similarity"],
            "variable_f1": symbolic["variable_f1"], "operator_f1": symbolic["operator_f1"],
            "c_pred": symbolic["c_pred"], "c_ref": symbolic["c_ref"],
            "m_sym": m_sym, "m_min": symbolic["m_min"],
            "unresolved": ";".join(issues), "formal_ready": not issues,
        }
        for issue in issues:
            unresolved.append({"key": row["logical_key"], "reason": issue})
        run_rows.append(result)
        runs_by_task[(condition, dataset)].append(result)

    task_rows: list[dict[str, Any]] = []
    for task_key, runs in sorted(runs_by_task.items()):
        condition, dataset = task_key
        runs = sorted(runs, key=lambda row: row["seed"])
        if [row["seed"] for row in runs] != list(SEEDS):
            raise GplearnAggregateError(f"{task_key} 三种子不完整")
        run_by_seed = {row["seed"]: row for row in runs}
        pairs: list[bool] = []
        labels: list[str | None] = []
        pair_source_keys: list[str] = []
        pair_model_keys: list[str | None] = []
        pair_response_hashes: list[str | None] = []
        pair_evidence_hashes: list[str] = []
        issues = []
        for seed_a, seed_b in PAIRS:
            pair_key = (condition, dataset, seed_a, seed_b)
            plan = pair_plans.get(pair_key)
            if plan is None:
                raise GplearnAggregateError(f"结构计划缺少 {pair_key}")
            request = plan["request"]
            evidence = request["deterministic_evidence"]
            a, b = run_by_seed[seed_a], run_by_seed[seed_b]
            if plan.get("dependencies") != [a["prediction_model_key"], b["prediction_model_key"]] or evidence["lhs_binding"].get("native_prefix_sha256") != a["native_prefix_sha256"] or evidence["rhs_binding"].get("native_prefix_sha256") != b["native_prefix_sha256"] or evidence["lhs_binding"].get("terminal_snapshot_sha256") != a["terminal_snapshot_sha256"] or evidence["rhs_binding"].get("terminal_snapshot_sha256") != b["terminal_snapshot_sha256"] or evidence["lhs_binding"].get("source_result_sha256") != a["selected_result_sha256"] or evidence["rhs_binding"].get("source_result_sha256") != b["selected_result_sha256"] or evidence["lhs_binding"].get("frozen_effective_expression") != a["effective_typed_expression"] or evidence["rhs_binding"].get("frozen_effective_expression") != b["effective_typed_expression"]:
                raise GplearnAggregateError(f"{pair_key} 结构计划依赖当前终点失败")
            frozen = structure_frozen_by_source.get(plan["evaluation_key"])
            decision, problem = _frozen_decision(plan, frozen, task_type="stab_structure")
            labels.append(decision)
            pair_source_keys.append(plan["evaluation_key"])
            pair_model_keys.append(frozen.get("evaluation_key") if frozen else None)
            pair_response_hashes.append(frozen.get("_source_file_sha256") if frozen else None)
            pair_evidence_hashes.append(evidence.get("evidence_sha256"))
            if problem:
                issues.append(f"pair_{seed_a}_{seed_b}_{problem}")
            else:
                pairs.append(structure_is_positive(decision))
        qualities = [RunQuality(r["id_quality"], r["ood_quality"], True) for r in runs]
        n = numerical_consistency(qualities)
        c = sum(pairs) / 3 if len(pairs) == 3 else None
        stab = stability_score(qualities, structural_pair_results=pairs) if c is not None else None
        result = {
            "condition": condition, "algorithm": "gplearn", "dataset_id": dataset,
            "seed_520_terminal_sha256": run_by_seed[520]["terminal_snapshot_sha256"],
            "seed_521_terminal_sha256": run_by_seed[521]["terminal_snapshot_sha256"],
            "seed_522_terminal_sha256": run_by_seed[522]["terminal_snapshot_sha256"],
            "pair_520_521_label": labels[0], "pair_520_522_label": labels[1],
            "pair_521_522_label": labels[2],
            "pair_source_keys": pair_source_keys, "pair_model_keys": pair_model_keys,
            "pair_response_sha256": pair_response_hashes,
            "pair_evidence_sha256": pair_evidence_hashes,
            "N": n, "V": 1.0, "C": c, "m_stab": stab.score if stab else None,
            "unresolved": ";".join(issues), "formal_ready": not issues,
        }
        for issue in issues:
            unresolved.append({"key": f"{condition}::gplearn::{dataset}", "reason": issue})
        task_rows.append(result)

    gplearn_alg: list[dict[str, Any]] = []
    for condition in ("clean", "noise001", "noise005"):
        runs = [row for row in run_rows if row["condition"] == condition]
        tasks = [row for row in task_rows if row["condition"] == condition]
        if len(runs) != expected_runs_per_condition or len(tasks) != expected_tasks_per_condition:
            raise GplearnAggregateError(f"{condition} 运行/任务数与预期不符")
        sym_ready = all(row["m_sym"] is not None and row["formal_ready"] for row in runs)
        stab_ready = all(row["m_stab"] is not None and row["formal_ready"] for row in tasks)
        gplearn_alg.append({
            "condition": condition, "algorithm": "gplearn", "run_count": len(runs), "task_count": len(tasks),
            "ID": 100 * sum(row["id_quality"] for row in runs) / len(runs),
            "OOD": 100 * sum(row["ood_quality"] for row in runs) / len(runs),
            "EFF": 100 * sum(row["m_eff"] for row in runs) / len(runs),
            "SYM": 100 * sum(row["m_sym"] for row in runs) / len(runs) if sym_ready else None,
            "MIN": 100 * sum(row["m_min"] for row in runs) / len(runs),
            "STAB": 100 * sum(row["m_stab"] for row in tasks) / len(tasks) if stab_ready else None,
            "formal_ready": sym_ready and stab_ready,
        })

    prior = [row for row in prior_algorithm_rows if str(row.get("algorithm", "")).casefold() != "gplearn"]
    if len(prior) != 42 or any(row.get("formal_ready") not in ("True", True) for row in prior):
        raise GplearnAggregateError("当前14算法表不是42条 formal-ready 数据")
    combined = prior + gplearn_alg
    if len({(row["condition"], row["algorithm"].casefold()) for row in combined}) != 45:
        raise GplearnAggregateError("15算法合并出现重复条件-算法键")
    manifest = {
        "schema_version": "gplearn_current_terminal_six_axis.v1",
        "run_count": len(run_rows), "task_count": len(task_rows),
        "gplearn_algorithm_count": len(gplearn_alg), "combined_algorithm_count": len(combined),
        "unresolved_count": len(unresolved),
        "formal_ready": len(run_rows) == expected_runs and not unresolved
                        and all(row["formal_ready"] for row in gplearn_alg),
        "gplearn_symbolic_basis": "protected_typed_prefix_vs_fixed_gt_canonical.v1",
        "gplearn_tree_basis_note": "NED rule is identical; typed protected tree vocabulary differs from ordinary SymPy canonical tree",
    }
    return {
        "gplearn_run_metrics": run_rows, "gplearn_task_stability": task_rows,
        "gplearn_algorithm_six_axis": gplearn_alg,
        "algorithm_six_axis_15alg": combined,
        "unresolved": unresolved, "manifest": manifest,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_frozen(paths: list[Path], *, by_source: bool) -> dict[str, dict[str, Any]]:
    result = {}
    field = "source_evaluation_key" if by_source else "evaluation_key"
    for directory in paths:
        for path in sorted(directory.glob("*.json")):
            raw = path.read_bytes()
            row = json.loads(raw)
            key = row.get(field)
            if not isinstance(key, str) or key in result:
                raise GplearnAggregateError(f"frozen {field} 重复或缺失: {path}")
            row["_source_file_sha256"] = hashlib.sha256(raw).hexdigest()
            result[key] = row
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--gt-binding", type=Path, required=True)
    parser.add_argument("--strict-run-metrics", type=Path, required=True)
    parser.add_argument("--strict-algorithm-metrics", type=Path, required=True)
    parser.add_argument("--prediction-frozen-dir", type=Path, action="append", required=True)
    parser.add_argument("--equivalence-plan", type=Path, required=True)
    parser.add_argument("--equivalence-frozen-dir", type=Path, action="append", required=True)
    parser.add_argument("--structure-plan", type=Path, required=True)
    parser.add_argument("--structure-frozen-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_gplearn(
        inventory_rows=_read_jsonl(args.inventory), gt_rows=_read_jsonl(args.gt_binding),
        strict_numeric_rows=_read_csv(args.strict_run_metrics),
        prediction_frozen_by_model=_load_frozen(args.prediction_frozen_dir, by_source=False),
        equivalence_plan_rows=_read_jsonl(args.equivalence_plan),
        equivalence_frozen_by_source=_load_frozen(args.equivalence_frozen_dir, by_source=True),
        structure_plan_rows=_read_jsonl(args.structure_plan),
        structure_frozen_by_source=_load_frozen(args.structure_frozen_dir, by_source=True),
        prior_algorithm_rows=_read_csv(args.strict_algorithm_metrics),
    )
    paths = {
        "inventory": args.inventory, "gt_binding": args.gt_binding,
        "strict_run_metrics": args.strict_run_metrics,
        "strict_algorithm_metrics": args.strict_algorithm_metrics,
        "equivalence_plan": args.equivalence_plan, "structure_plan": args.structure_plan,
    }
    result["manifest"]["input_sha256"] = {name: _sha_file(path) for name, path in paths.items()}
    result["manifest"]["frozen_directory_sha256"] = {
        str(directory): _sha_json([(path.name, _sha_file(path)) for path in sorted(directory.glob("*.json"))])
        for directory in (*args.prediction_frozen_dir, *args.equivalence_frozen_dir,
                          *args.structure_frozen_dir)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gplearn_run_metrics", "gplearn_task_stability", "gplearn_algorithm_six_axis",
                 "algorithm_six_axis_15alg", "unresolved"):
        _write_csv(args.output_dir / f"{name}.csv", result[name])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["manifest"], ensure_ascii=False))


if __name__ == "__main__":
    main()
