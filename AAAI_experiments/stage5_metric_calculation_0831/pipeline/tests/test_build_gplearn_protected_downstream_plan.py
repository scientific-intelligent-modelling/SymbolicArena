"""gplearn 当前终点 Eq/结构任务绑定测试。"""

from __future__ import annotations

import copy
import hashlib

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import canonical_json
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_gplearn_protected_downstream_plan import (
    GplearnDownstreamError,
    _write_jsonl_atomic,
    build_plans,
)
from check.run_opus48_terminal_plan import load_plan


def _rows():
    inventory, plans, frozen = [], [], {}
    raw = "div(X0, X1)"
    raw_sha = hashlib.sha256(raw.encode()).hexdigest()
    for seed in (520, 521, 522):
        key = f"gplearn::task-a::s{seed}::clean"
        result_sha = hashlib.sha256(f"result-{seed}".encode()).hexdigest()
        inventory.append({
            "algorithm": "gplearn", "condition": "clean", "dataset_id": "task-a", "seed": seed,
            "logical_key": key, "feature_names": ["x0", "x1"],
            "variable_mapping": {"X0": "x0", "X1": "x1"},
            "native_prefix": raw, "native_prefix_sha256": raw_sha,
            "terminal_expression": "x0/x1", "terminal_expression_sha256": hashlib.sha256(b"x0/x1").hexdigest(),
            "terminal_snapshot_sha256": f"snapshot-{seed}",
            "selected_result_sha256": result_sha,
            "gt_frozen_evaluation_key": "gt-key", "gt_reference_expression": "x0/x1",
            "gt_reference_source_sha256": "gt-source",
            "task_id": f"gplearn_s{seed}_clean_g0001",
        })
        plan_key = f"pred-{seed}"
        plans.append({
            "evaluation_key": plan_key, "input_hash": f"input-{seed}",
            "prompt_sha256": "prompt-sha", "schema_sha256": "schema-sha",
            "logical_id": f"pred_simplify::gplearn::g0001::task-a::s{seed}::clean::v4",
            "request": {"native_prefix_sha256": raw_sha,
                        "original_expression": "pdiv(x0,x1)",
                        "terminal_binding_evidence": {
                            "logical_key": key, "native_prefix_sha256": raw_sha,
                            "terminal_snapshot_sha256": f"snapshot-{seed}",
                            "selected_result_sha256": result_sha,
                        }},
        })
        frozen[plan_key] = {
            "evaluation_key": f"model-{plan_key}", "source_evaluation_key": plan_key,
            "source_input_hash": f"input-{seed}",
            "prompt_sha256": "prompt-sha", "schema_sha256": "schema-sha",
            "status": "frozen", "model": "claude-opus-4-8",
            "structured_output": {"outcome": "unchanged", "simplified_expression": "pdiv(x0,x1)"},
            "semantic_validation": {"status": "promotable", "effective_expression": "pdiv(x0,x1)"},
            "terminal_binding_evidence": {
                "logical_key": key, "native_prefix_sha256": raw_sha,
                "terminal_snapshot_sha256": f"snapshot-{seed}",
                "selected_result_sha256": result_sha,
            },
        }
    gt = [{"dataset_id": "task-a", "fixed_reference_expression": "x0/x1",
           "input_expression": "x0/x1",
           "gt_frozen_evaluation_key": "gt-key", "gt_result_sha256": "gt-result",
           "gt_source_evidence_sha256": "gt-source"}]
    probes = [{"dataset_name": "task-a", "variables": ["x0", "x1"],
               "schema_version": "dataset_probes_v1", "points": [
                   {"values": {"x0": 2.0, "x1": 0.5}, "split": "id_test", "row_index": 0},
                   {"values": {"x0": 2.0, "x1": 0.001}, "split": "ood_test", "row_index": 1},
               ]}]
    probes[0]["sample_sha256"] = hashlib.sha256(canonical_json(probes[0]).encode()).hexdigest()
    probes[0]["evidence_sha256"] = hashlib.sha256(canonical_json(probes[0]).encode()).hexdigest()
    return inventory, plans, frozen, gt, probes


def test_three_seeds_generate_three_eq_and_three_pairs(tmp_path):
    inventory, plans, frozen, gt, probes = _rows()
    result = build_plans(inventory, plans, frozen, gt, probes, output_dir=tmp_path)
    assert len(result["equivalence_plan"]) == 3
    assert len(result["structure_plan"]) == 3
    eq = result["equivalence_plan"][0]
    assert eq["dependencies"] == ["gt-key", "model-pred-520"]
    assert eq["request"]["deterministic_evidence"]["prediction_vs_gt"]["status"] == "numeric_counterexample"
    assert eq["request"]["prediction_terminal_snapshot_sha256"] == "snapshot-520"
    pair = result["structure_plan"][0]
    assert pair["dependencies"] == ["model-pred-520", "model-pred-521"]
    assert pair["request"]["deterministic_evidence"]["typed_structure_consistency"] is True
    assert result["summary"]["formal_ready"] is False  # synthetic partial cohort
    _write_jsonl_atomic(tmp_path / "equivalence.jsonl", result["equivalence_plan"])
    _write_jsonl_atomic(tmp_path / "structure.jsonl", result["structure_plan"])
    assert len(load_plan(tmp_path / "equivalence.jsonl")[0]) == 3
    assert len(load_plan(tmp_path / "structure.jsonl")[0]) == 3


def test_missing_or_stale_prediction_freeze_fails_closed(tmp_path):
    inventory, plans, frozen, gt, probes = _rows()
    frozen.pop("pred-522")
    with pytest.raises(GplearnDownstreamError):
        build_plans(inventory, plans, frozen, gt, probes, output_dir=tmp_path)
    partial = build_plans(inventory, plans, frozen, gt, probes, output_dir=tmp_path, allow_partial=True)
    assert len(partial["equivalence_plan"]) == 2
    assert len(partial["structure_plan"]) == 1
    assert partial["summary"]["missing_frozen"] == 1
    frozen["pred-522"] = _rows()[2]["pred-522"]
    frozen["pred-522"]["terminal_binding_evidence"]["terminal_snapshot_sha256"] = "old"
    with pytest.raises(GplearnDownstreamError):
        build_plans(inventory, plans, frozen, gt, probes, output_dir=tmp_path)


def test_frozen_retry_supersedes_base_and_keeps_both_versions(tmp_path):
    inventory, plans, frozen, gt, probes = _rows()
    retry = copy.deepcopy(plans[0])
    retry["evaluation_key"] = "retry-source-520"
    retry["input_hash"] = "retry-input-520"
    retry["prompt_sha256"] = "retry-prompt-sha"
    retry["logical_id"] += "::retry2"
    retried = copy.deepcopy(frozen["pred-520"])
    retried.update(
        evaluation_key="model-retry-520", source_evaluation_key="retry-source-520",
        source_input_hash="retry-input-520", prompt_sha256="retry-prompt-sha",
    )
    frozen["retry-source-520"] = retried
    result = build_plans(inventory, plans + [retry], frozen, gt, probes, output_dir=tmp_path)
    eq = next(row for row in result["equivalence_plan"] if row["request"]["seed"] == 520)
    rhs = eq["request"]["deterministic_evidence"]["rhs_binding"]
    assert eq["dependencies"] == ["gt-key", "model-retry-520"]
    assert rhs["prediction_selection_source"] == "retry"
    assert [version["model_bound_evaluation_key"] for version in rhs["prediction_available_versions"]] == [
        "model-pred-520", "model-retry-520",
    ]
    assert [version["source_evaluation_key"] for version in rhs["prediction_plan_versions"]] == [
        "pred-520", "retry-source-520",
    ]
