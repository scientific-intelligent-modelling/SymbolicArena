"""为当前 gplearn 原生终点构建受保护 Eq/STAB 计划，不发起 API。

输入必须是冻结的 450 条 terminal inventory、对应 v4 预测化简计划和 frozen
响应。预测缺失时默认 fail-closed；`--allow-partial` 仅用于先行 smoke。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .claude_contract import canonical_json
from .gplearn_native_prefix_evidence import (
    PrefixEvidenceError,
    build_inventory_prefix_evidence,
    build_prefix_evidence,
    evaluate_prefix_evidence,
    native_equivalence_probe,
)
from .symbolic_evidence import build_symbolic_artifact
from .symbolic_task_builder import PromptSchemaBundle, _task_from_request


class GplearnDownstreamError(ValueError):
    """当前 gplearn 下游计划缺少可验证的输入绑定。"""


SEEDS = (520, 521, 522)
PAIRS = ((520, 521), (520, 522), (521, 522))
STAGE_ROOT = Path(__file__).resolve().parents[1]

def _sha_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _index(rows: list[dict[str, Any]], field: str, kind: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in output:
            raise GplearnDownstreamError(f"{kind} 缺少或重复 {field}: {key!r}")
        output[key] = row
    return output


def _prompt_contract(kind: str) -> PromptSchemaBundle:
    filename = f"gplearn_protected_{kind}.v1.txt"
    prompt_path = STAGE_ROOT / "config" / "prompts" / filename
    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    schema_path = STAGE_ROOT / "config" / "schemas" / f"{kind}.v1.json"
    schema_bytes = schema_path.read_bytes()
    return PromptSchemaBundle(
        prompt_path=str(prompt_path.resolve()), prompt_version=prompt_path.stem,
        prompt_template=prompt, prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_path=str(schema_path.resolve()), schema_version=schema_path.stem,
        schema=json.loads(schema_bytes), schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
    )


def _points(probe: Mapping[str, Any], feature_names: list[str]) -> np.ndarray:
    if probe.get("variables") != feature_names:
        raise GplearnDownstreamError("dataset probe 变量顺序与 terminal 不一致")
    sample = {
        "schema_version": probe.get("schema_version"),
        "dataset_name": probe.get("dataset_name"),
        "variables": probe.get("variables"),
        "points": probe.get("points"),
    }
    if probe.get("sample_sha256") != _sha_json(sample):
        raise GplearnDownstreamError("dataset probe sample SHA 不匹配")
    if probe.get("evidence_sha256") != _sha_json({
        field: value for field, value in probe.items() if field != "evidence_sha256"
    }):
        raise GplearnDownstreamError("dataset probe evidence SHA 不匹配")
    try:
        points = np.asarray([
            [float(item["values"][name]) for name in feature_names]
            for item in probe["points"]
        ], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise GplearnDownstreamError("dataset probe 数值行非法") from exc
    if points.ndim != 2 or not points.size or not np.isfinite(points).all():
        raise GplearnDownstreamError("dataset probe 为空或含非有限输入")
    return points


def _gt_numeric_evidence(
    prediction: Mapping[str, Any], gt_expression: str,
    feature_names: list[str], points: np.ndarray,
) -> dict[str, Any]:
    """GT 仅按普通数学语义解析；预测始终用 protected typed evaluator。"""
    try:
        import sympy as sp

        gt = build_symbolic_artifact(gt_expression, allowed_variables=feature_names)
        fn = sp.lambdify([sp.Symbol(name) for name in feature_names], gt["sympy_expression"], modules="numpy")
        with np.errstate(all="ignore"):
            raw = fn(*[points[:, col] for col in range(points.shape[1])])
            gt_values = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),))
        pred_values = evaluate_prefix_evidence(dict(prediction), points)
    except Exception as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "sample_count": len(points)}
    finite = np.isfinite(gt_values) & np.isfinite(pred_values)
    mismatch = finite & ~np.isclose(gt_values, pred_values, rtol=1e-9, atol=1e-9)
    first = np.flatnonzero(mismatch)
    return {
        "status": "numeric_counterexample" if first.size else "numeric_support_only" if finite.any() else "unavailable_no_common_finite_points",
        "sample_count": int(len(points)), "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()), "mismatch_count": int(mismatch.sum()),
        "first_counterexample": None if not first.size else {
            "point_index": int(first[0]), "values": points[first[0]].tolist(),
            "ground_truth_value": float(gt_values[first[0]]),
            "prediction_value": float(pred_values[first[0]]),
        },
    }


def _frozen_prediction(
    row: Mapping[str, Any], plan: Mapping[str, Any], frozen: Mapping[str, Any],
    points: np.ndarray, native_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    key = plan["evaluation_key"]
    bound = frozen.get("terminal_binding_evidence") or {}
    planned_bound = (plan.get("request") or {}).get("terminal_binding_evidence") or {}
    expected = {
        "logical_key": row["logical_key"],
        "native_prefix_sha256": row["native_prefix_sha256"],
        "terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
        "selected_result_sha256": row["selected_result_sha256"],
    }
    if any(planned_bound.get(name) != value or bound.get(name) != value for name, value in expected.items()):
        raise GplearnDownstreamError(f"{row['logical_key']} 预测计划/响应与当前原生终点不绑定")
    if frozen.get("source_evaluation_key") != key or any(
        frozen.get(field) != plan.get(field) for field in ("prompt_sha256", "schema_sha256")
    ) or frozen.get("source_input_hash") != plan.get("input_hash"):
        raise GplearnDownstreamError(f"{row['logical_key']} 预测响应与 v4 计划键不一致")
    model_bound_key = frozen.get("evaluation_key")
    if not isinstance(model_bound_key, str) or not model_bound_key:
        raise GplearnDownstreamError(f"{row['logical_key']} 缺少模型绑定的冻结 evaluation_key")
    semantic = frozen.get("semantic_validation") or {}
    effective = semantic.get("effective_expression")
    if frozen.get("status") != "frozen" or semantic.get("status") != "promotable" or not isinstance(effective, str) or not effective:
        raise GplearnDownstreamError(f"{row['logical_key']} 预测响应未冻结为可用公式")
    if frozen.get("model") != "claude-opus-4-8":
        raise GplearnDownstreamError(f"{row['logical_key']} 预测响应模型不是 Opus4.8")
    if semantic.get("native_prefix_sha256") not in (None, row["native_prefix_sha256"]):
        raise GplearnDownstreamError(f"{row['logical_key']} 语义验证的原生前缀哈希漂移")
    if semantic.get("source_typed_fingerprint") not in (None, native_evidence["exact_fingerprint"]):
        raise GplearnDownstreamError(f"{row['logical_key']} 语义验证的 typed 指纹漂移")
    if re.search(r"\b(?:div|log|sqrt|inv)\s*\(", effective, flags=re.I):
        raise GplearnDownstreamError(f"{row['logical_key']} 化简式未显式保留 protected 算子")
    candidate = build_prefix_evidence(effective, row["feature_names"])
    native_check = native_equivalence_probe(
        row["native_prefix"], effective, row["feature_names"], points,
    )
    if native_check["status"] in {"native_representation_mismatch", "numeric_counterexample"}:
        raise GplearnDownstreamError(f"{row['logical_key']} 预测化简与原生 prefix 存在数值反例")
    return {
        "evaluation_key": model_bound_key, "source_evaluation_key": key,
        "plan_input_hash": plan["input_hash"],
        "plan_original_expression": plan["request"]["original_expression"],
        "response_sha256": frozen.get("_source_file_sha256") or _sha_json(frozen),
        "model": frozen["model"], "effective_expression": effective,
        "typed_evidence": candidate, "native_check": native_check,
    }


def _binding(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    typed = prediction["typed_evidence"]
    return {
        "condition": row["condition"], "algorithm": "gplearn", "dataset_id": row["dataset_id"],
        "seed": row["seed"], "logical_key": row["logical_key"],
        "native_prefix_sha256": row["native_prefix_sha256"],
        "terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
        "terminal_expression_sha256": row["terminal_expression_sha256"],
        "selected_result_sha256": row["selected_result_sha256"],
        "prediction_evaluation_key": prediction["evaluation_key"],
        "prediction_source_evaluation_key": prediction["source_evaluation_key"],
        "frozen_evaluation_key": prediction["evaluation_key"],
        "plan_original_expression": prediction["plan_original_expression"],
        "frozen_effective_expression": prediction["effective_expression"],
        "source_result_sha256": row["selected_result_sha256"],
        "prediction_response_sha256": prediction["response_sha256"],
        "prediction_selection_source": prediction["selection_source"],
        "prediction_available_versions": prediction["available_versions"],
        "prediction_plan_versions": prediction["plan_versions"],
        "effective_typed_expression": prediction["effective_expression"],
        "typed_exact_fingerprint": typed["exact_fingerprint"],
        "typed_structure_fingerprint": typed["constants_abstracted_structure_fingerprint"],
        "typed_node_count": typed["node_count"], "typed_tree_depth": typed["tree_depth"],
        "typed_operator_set": typed["operator_set"],
        "variable_mapping": row["variable_mapping"],
    }


def build_plans(
    inventory_rows: list[dict[str, Any]],
    prediction_plan_rows: list[dict[str, Any]],
    frozen_predictions: Mapping[str, dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """先严格核对绑定，再构造当前公式 Eq 与三种子结构计划。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    eq_contract = _prompt_contract("equivalence")
    structure_contract = _prompt_contract("structure")
    inventory = _index(inventory_rows, "logical_key", "inventory")
    gt = _index(gt_rows, "dataset_id", "GT")
    probes = _index(probe_rows, "dataset_name", "dataset probe")
    plan_by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_plan_keys: set[str] = set()
    for plan in prediction_plan_rows:
        key = (plan.get("request") or {}).get("terminal_binding_evidence", {}).get("logical_key")
        source_key = plan.get("evaluation_key")
        if not isinstance(key, str) or not isinstance(source_key, str) or source_key in seen_plan_keys:
            raise GplearnDownstreamError(f"pred 计划终点身份缺失或 source key 重复: {key}")
        seen_plan_keys.add(source_key)
        plan_by_logical[key].append(plan)
    if set(inventory) != set(plan_by_logical):
        raise GplearnDownstreamError("inventory 与 pred v4 计划运行键不一致")
    usable: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, str]] = []
    for logical_key, row in sorted(inventory.items()):
        evidence = build_inventory_prefix_evidence(row)
        gt_row = gt.get(row["dataset_id"])
        probe = probes.get(row["dataset_id"])
        if gt_row is None or probe is None:
            raise GplearnDownstreamError(f"{logical_key} 缺少固定 GT 或 dataset probes")
        if gt_row.get("gt_frozen_evaluation_key") != row.get("gt_frozen_evaluation_key") or gt_row.get("fixed_reference_expression") != row.get("gt_reference_expression") or gt_row.get("gt_source_evidence_sha256") != row.get("gt_reference_source_sha256"):
            raise GplearnDownstreamError(f"{logical_key} GT 绑定漂移")
        points = _points(probe, row["feature_names"])
        available = [
            (plan, frozen_predictions[plan["evaluation_key"]])
            for plan in plan_by_logical[logical_key]
            if plan["evaluation_key"] in frozen_predictions
        ]
        if not available:
            unresolved.append({"logical_key": logical_key, "reason": "prediction_frozen_missing"})
            if allow_partial:
                continue
            raise GplearnDownstreamError(f"{logical_key} 尚无冻结预测化简")
        plan, frozen = available[-1]
        prediction = _frozen_prediction(row, plan, frozen, points, evidence)
        prediction["selection_source"] = "retry" if len(plan_by_logical[logical_key]) > 1 and plan is not plan_by_logical[logical_key][0] else "base"
        prediction["available_versions"] = [
            {
                "source_evaluation_key": prior_plan["evaluation_key"],
                "model_bound_evaluation_key": prior_frozen.get("evaluation_key"),
                "prompt_sha256": prior_plan.get("prompt_sha256"),
                "response_sha256": prior_frozen.get("_source_file_sha256") or _sha_json(prior_frozen),
            }
            for prior_plan, prior_frozen in available
        ]
        prediction["plan_versions"] = [
            {
                "source_evaluation_key": candidate_plan["evaluation_key"],
                "input_hash": candidate_plan.get("input_hash"),
                "prompt_sha256": candidate_plan.get("prompt_sha256"),
                "model_bound_evaluation_key": (
                    frozen_predictions[candidate_plan["evaluation_key"]].get("evaluation_key")
                    if candidate_plan["evaluation_key"] in frozen_predictions else None
                ),
                "response_sha256": (
                    frozen_predictions[candidate_plan["evaluation_key"]].get("_source_file_sha256")
                    if candidate_plan["evaluation_key"] in frozen_predictions else None
                ),
            }
            for candidate_plan in plan_by_logical[logical_key]
        ]
        usable[logical_key] = {
            "inventory": row, "native_evidence": evidence,
            "prediction": prediction, "gt": gt_row, "probe": probe, "points": points,
        }

    equivalence_plan: list[dict[str, Any]] = []
    structure_plan: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for logical_key, item in sorted(usable.items()):
        row, pred, gt_row, probe, points = (
            item["inventory"], item["prediction"], item["gt"], item["probe"], item["points"]
        )
        condition, dataset, seed = row["condition"], row["dataset_id"], int(row["seed"])
        if condition not in {"clean", "noise001", "noise005"} or seed not in SEEDS:
            raise GplearnDownstreamError(f"非法条件或种子: {logical_key}")
        if seed in grouped[(condition, dataset)]:
            raise GplearnDownstreamError(f"重复 condition/dataset/seed: {logical_key}")
        grouped[(condition, dataset)][seed] = item
        lhs = {
            "dataset_id": dataset, "fixed_reference_expression": gt_row["fixed_reference_expression"],
            "frozen_evaluation_key": gt_row["gt_frozen_evaluation_key"],
            "plan_original_expression": gt_row["input_expression"],
            "frozen_effective_expression": gt_row["fixed_reference_expression"],
            "gt_frozen_evaluation_key": gt_row["gt_frozen_evaluation_key"],
            "gt_result_sha256": gt_row["gt_result_sha256"],
            "gt_source_evidence_sha256": gt_row["gt_source_evidence_sha256"],
        }
        rhs = _binding(row, pred)
        deterministic = {
            "schema_version": "gplearn_protected_pair_evidence.v1",
            "lhs_binding": lhs, "rhs_binding": rhs,
            "native_protected_structure": {
                "typed_expression": item["native_evidence"]["typed_expression"],
                "node_count": item["native_evidence"]["node_count"],
                "tree_depth": item["native_evidence"]["tree_depth"],
                "operator_set": item["native_evidence"]["operator_set"],
                "structure_fingerprint": item["native_evidence"]["constants_abstracted_structure_fingerprint"],
                "semantics": item["native_evidence"]["protected_semantics"],
            },
            "native_vs_simplified": pred["native_check"],
            "prediction_vs_gt": _gt_numeric_evidence(pred["typed_evidence"], gt_row["fixed_reference_expression"], row["feature_names"], points),
            "dataset_probe_sha256": probe["sample_sha256"],
        }
        deterministic["evidence_sha256"] = _sha_json(deterministic)
        request = {
            "algorithm": "gplearn", "algorithm_slug": "gplearn", "condition": condition,
            "noise_tag": condition, "dataset_id": dataset, "seed": seed,
            "feature_names": row["feature_names"], "variable_mapping": row["variable_mapping"],
            "effective_ground_truth_expression": gt_row["fixed_reference_expression"],
            "effective_prediction_expression": pred["effective_expression"],
            "prediction_result_sha256": row["selected_result_sha256"],
            "prediction_terminal_snapshot_sha256": row["terminal_snapshot_sha256"],
            "prediction_native_prefix_sha256": row["native_prefix_sha256"],
            "deterministic_evidence": deterministic,
            "evidence_hash": deterministic["evidence_sha256"],
            "model_binding": "claude-opus-4-8",
        }
        task = _task_from_request(
            logical_id=f"equivalence::gplearn::{dataset}::s{seed}::{condition}::v1",
            task_type="equivalence", priority=30, request=request,
            evidence_hash=deterministic["evidence_sha256"], contract=eq_contract,
            dependencies=(gt_row["gt_frozen_evaluation_key"], pred["evaluation_key"]),
            condition=condition,
        )
        equivalence_plan.append(task.to_json_record())

    for (condition, dataset), seeds in sorted(grouped.items()):
        for left_seed, right_seed in PAIRS:
            if left_seed not in seeds or right_seed not in seeds:
                continue
            left = seeds[left_seed]
            right = seeds[right_seed]
            a = left["prediction"]["typed_evidence"]
            b = right["prediction"]["typed_evidence"]
            probe = left["probe"]
            points = left["points"]
            numeric = native_equivalence_probe(
                left["inventory"]["native_prefix"],
                right["prediction"]["effective_expression"],
                left["inventory"]["feature_names"], points,
            )
            deterministic = {
                "schema_version": "gplearn_protected_pair_evidence.v1",
                "lhs_binding": _binding(left["inventory"], left["prediction"]),
                "rhs_binding": _binding(right["inventory"], right["prediction"]),
                "typed_structure_consistency": a["constants_abstracted_structure_fingerprint"] == b["constants_abstracted_structure_fingerprint"],
                "lhs_structure_fingerprint": a["constants_abstracted_structure_fingerprint"],
                "rhs_structure_fingerprint": b["constants_abstracted_structure_fingerprint"],
                "native_protected_numeric_comparison": numeric,
                "dataset_probe_sha256": probe["sample_sha256"],
            }
            deterministic["evidence_sha256"] = _sha_json(deterministic)
            request = {
                "algorithm": "gplearn", "algorithm_slug": "gplearn", "condition": condition,
                "noise_tag": condition, "dataset_id": dataset,
                "seed_left": left_seed, "seed_right": right_seed,
                "feature_names": left["inventory"]["feature_names"],
                "variable_mapping": left["inventory"]["variable_mapping"],
                "left_effective_expression": left["prediction"]["effective_expression"],
                "right_effective_expression": right["prediction"]["effective_expression"],
                "effective_prediction_a_expression": left["prediction"]["effective_expression"],
                "effective_prediction_b_expression": right["prediction"]["effective_expression"],
                "prediction_a_result_sha256": left["inventory"]["selected_result_sha256"],
                "prediction_b_result_sha256": right["inventory"]["selected_result_sha256"],
                "prediction_a_valid_output": True,
                "prediction_b_valid_output": True,
                "deterministic_evidence": deterministic,
                "deterministic_pair_evidence": deterministic,
                "evidence_hash": deterministic["evidence_sha256"],
                "model_binding": "claude-opus-4-8",
            }
            task = _task_from_request(
                logical_id=f"stab_structure::gplearn::{dataset}::s{left_seed}-s{right_seed}::{condition}::v1",
                task_type="stab_structure", priority=40, request=request,
                evidence_hash=deterministic["evidence_sha256"], contract=structure_contract,
                dependencies=(left["prediction"]["evaluation_key"], right["prediction"]["evaluation_key"]),
                condition=condition,
            )
            structure_plan.append(task.to_json_record())
    summary = {
        "inventory_rows": len(inventory), "frozen_usable": len(usable),
        "equivalence_rows": len(equivalence_plan), "structure_rows": len(structure_plan),
        "missing_frozen": len(unresolved), "allow_partial": allow_partial,
        "plan_ready": len(inventory) == 450 and len(equivalence_plan) == 450
                      and len(structure_plan) == 450 and not unresolved,
        "formal_ready": False,
        "protected_semantics": "gplearn_native_protected_prefix.v1",
        "model": "claude-opus-4-8", "requires_api_judgments": True,
    }
    return {"equivalence_plan": equivalence_plan, "structure_plan": structure_plan,
            "unresolved": unresolved, "summary": summary}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_frozen(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.glob("*.json")):
        raw = candidate.read_bytes()
        row = json.loads(raw)
        key = row.get("source_evaluation_key")
        if not isinstance(key, str) or key in records:
            raise GplearnDownstreamError(f"冻结响应重复或缺 evaluation_key: {candidate}")
        row["_source_file_sha256"] = hashlib.sha256(raw).hexdigest()
        records[key] = row
    return records


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prediction-plan", type=Path, action="append", required=True)
    parser.add_argument("--prediction-frozen-dir", type=Path, action="append", required=True)
    parser.add_argument("--gt-binding", type=Path, required=True)
    parser.add_argument("--dataset-probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    plan_rows = [row for path in args.prediction_plan for row in _read_jsonl(path)]
    frozen_rows: dict[str, dict[str, Any]] = {}
    for path in args.prediction_frozen_dir:
        for key, row in _load_frozen(path).items():
            if key in frozen_rows:
                raise GplearnDownstreamError(f"不同目录重复冻结 source key: {key}")
            frozen_rows[key] = row
    result = build_plans(
        _read_jsonl(args.inventory), plan_rows,
        frozen_rows, _read_jsonl(args.gt_binding),
        _read_jsonl(args.dataset_probes), output_dir=args.output_dir,
        allow_partial=args.allow_partial,
    )
    for name in ("equivalence_plan", "structure_plan", "unresolved"):
        _write_jsonl_atomic(args.output_dir / f"{name}.jsonl", result[name])
    (args.output_dir / "summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
