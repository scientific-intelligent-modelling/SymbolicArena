"""gplearn 当前终点六轴的裁决绑定和 undetermined 计分。"""

from __future__ import annotations

import copy
import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_gplearn_current_terminal import (
    GplearnAggregateError,
    aggregate_gplearn,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.build_gplearn_protected_downstream_plan import build_plans
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.tests.test_build_gplearn_protected_downstream_plan import _rows


def _fixture(tmp_path):
    base_inventory, base_plans, base_frozen, gt, probes = _rows()
    inventory, plans, frozen = [], [], {}
    for condition in ("clean", "noise001", "noise005"):
        for original, plan in zip(base_inventory, base_plans):
            seed = original["seed"]
            row = copy.deepcopy(original)
            row["condition"] = condition
            row["logical_key"] = f"gplearn::task-a::s{seed}::{condition}"
            inventory.append(row)
            cloned_plan = copy.deepcopy(plan)
            source_key = f"pred-{seed}-{condition}"
            cloned_plan["evaluation_key"] = source_key
            cloned_plan["request"]["terminal_binding_evidence"]["logical_key"] = row["logical_key"]
            plans.append(cloned_plan)
            response = copy.deepcopy(base_frozen[f"pred-{seed}"])
            response["evaluation_key"] = f"model-{source_key}"
            response["source_evaluation_key"] = source_key
            response["terminal_binding_evidence"]["logical_key"] = row["logical_key"]
            frozen[source_key] = response
    downstream = build_plans(inventory, plans, frozen, gt, probes, output_dir=tmp_path)
    pred_by_model = {row["evaluation_key"]: row for row in frozen.values()}
    eq_frozen = {}
    for plan in downstream["equivalence_plan"]:
        seed = plan["request"]["seed"]
        decision = {520: "equivalent", 521: "not_equivalent", 522: "undetermined"}[seed]
        eq_frozen[plan["evaluation_key"]] = {
            "status": "frozen", "model": "claude-opus-4-8", "task_type": "equivalence",
            "source_evaluation_key": plan["evaluation_key"], "evaluation_key": "model-" + plan["evaluation_key"],
            "source_input_hash": plan["input_hash"], "prompt_sha256": plan["prompt_sha256"],
            "schema_sha256": plan["schema_sha256"], "plan_dependencies": plan["dependencies"],
            "structured_output": {"decision": decision},
            "semantic_validation": {"status": "promotable", "decision": decision,
                                    "evidence_sha256": plan["request"]["evidence_hash"]},
        }
    structure_frozen = {}
    for plan in downstream["structure_plan"]:
        pair = (plan["request"]["seed_left"], plan["request"]["seed_right"])
        decision = {(520, 521): "mathematically_equivalent",
                    (520, 522): "same_canonical_structure",
                    (521, 522): "undetermined"}[pair]
        structure_frozen[plan["evaluation_key"]] = {
            "status": "frozen", "model": "claude-opus-4-8", "task_type": "stab_structure",
            "source_evaluation_key": plan["evaluation_key"], "evaluation_key": "model-" + plan["evaluation_key"],
            "source_input_hash": plan["input_hash"], "prompt_sha256": plan["prompt_sha256"],
            "schema_sha256": plan["schema_sha256"], "plan_dependencies": plan["dependencies"],
            "structured_output": {"decision": decision},
            "semantic_validation": {"status": "promotable", "decision": decision,
                                    "evidence_sha256": plan["request"]["evidence_hash"]},
        }
    numeric = [{
        "condition": row["condition"], "algorithm": "gplearn", "dataset_id": row["dataset_id"],
        "seed": row["seed"], "terminal_expression_sha256": row["terminal_expression_sha256"],
        "terminal_source_sha256": row["terminal_snapshot_sha256"],
        "selected_result_sha256": row["selected_result_sha256"],
        "valid_output": "True", "numeric_ready": "True",
        "id_quality": "0.8", "ood_quality": "0.6", "m_eff": "0.5",
    } for row in inventory]
    prior = [{"condition": condition, "algorithm": f"other-{i}", "formal_ready": "True"}
             for condition in ("clean", "noise001", "noise005") for i in range(14)]
    return dict(
        inventory_rows=inventory, gt_rows=gt, strict_numeric_rows=numeric,
        prediction_frozen_by_model=pred_by_model,
        equivalence_plan_rows=downstream["equivalence_plan"],
        equivalence_frozen_by_source=eq_frozen,
        structure_plan_rows=downstream["structure_plan"],
        structure_frozen_by_source=structure_frozen,
        prior_algorithm_rows=prior,
        expected_runs=9, expected_runs_per_condition=3, expected_tasks_per_condition=1,
    )


def test_full_current_binding_and_undetermined_policy(tmp_path):
    result = aggregate_gplearn(**_fixture(tmp_path))
    assert len(result["gplearn_run_metrics"]) == 9
    assert len(result["gplearn_task_stability"]) == 3
    assert len(result["algorithm_six_axis_15alg"]) == 45
    assert result["manifest"]["formal_ready"]
    clean = [row for row in result["gplearn_run_metrics"] if row["condition"] == "clean"]
    assert clean[0]["m_sym"] == 1
    assert clean[2]["equivalence_decision"] == "undetermined" and clean[2]["m_sym"] < 0.5
    task = next(row for row in result["gplearn_task_stability"] if row["condition"] == "clean")
    assert task["pair_521_522_label"] == "undetermined"
    assert task["C"] == pytest.approx(2 / 3)


def test_missing_judgment_is_not_counted_as_negative(tmp_path):
    inputs = _fixture(tmp_path)
    plan = inputs["equivalence_plan_rows"][0]
    inputs["equivalence_frozen_by_source"].pop(plan["evaluation_key"])
    result = aggregate_gplearn(**inputs)
    assert not result["manifest"]["formal_ready"]
    missing = next(row for row in result["gplearn_run_metrics"] if row["condition"] == "clean" and row["seed"] == 520)
    assert missing["m_sym"] is None and "equivalence_frozen_missing" in missing["unresolved"]


def test_missing_structure_frozen_does_not_become_zero_pair(tmp_path):
    inputs = _fixture(tmp_path)
    plan = inputs["structure_plan_rows"][0]
    inputs["structure_frozen_by_source"].pop(plan["evaluation_key"])
    result = aggregate_gplearn(**inputs)
    task = next(row for row in result["gplearn_task_stability"] if row["condition"] == "clean")
    assert task["N"] == 1 and task["V"] == 1
    assert task["C"] is None and task["m_stab"] is None
    assert not result["manifest"]["formal_ready"]


def test_stale_snapshot_cannot_bind_numeric(tmp_path):
    inputs = _fixture(tmp_path)
    inputs["strict_numeric_rows"][0]["terminal_source_sha256"] = "old-snapshot"
    with pytest.raises(GplearnAggregateError):
        aggregate_gplearn(**inputs)
