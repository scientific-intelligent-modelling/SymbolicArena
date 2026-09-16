"""Prediction planning must bind the actual terminal to a stable evaluation key."""

from __future__ import annotations

import json

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import clean_task_builder as builder
from AAAI_experiments.stage5_metric_calculation_0831.pipeline import prepare_current_terminal_opus_plan as plan


def _fixtures():
    key = ("clean", "drsr", "BPG3", 520)
    stable_id = "drsr_s520_clean_g0005"
    selected_id = "drsr_s520_clean_g0999"
    snapshot = {
        "dataset": "BPG3", "seed": 520, "status": "ok",
        "feature_names": ["t", "P"], "target_name": "dP_dt",
        "equation": "x0+x1",
        "canonical_artifact": {"instantiated_expression": "x0+x1", "variables": ["x0", "x1"]},
    }
    result = {
        "dataset": "BPG3", "seed": 520, "status": "ok",
        "feature_names": ["t", "P"], "target_name": "dP_dt",
        "equation": "x0+x1",
    }
    snapshot_raw = json.dumps(snapshot)
    result_raw = json.dumps(result)
    old_raw = json.dumps({"status": "ok"})
    source_path = "/remote/iaaccn22/progress/minute_0180.json"
    result_path = "/remote/iaaccn22/result.json"
    member = "runs/clean/drsr/BPG3/seed_520/task/progress/minute_0180.json"
    terminal = {
        "condition": "clean", "algorithm": "drsr", "algorithm_slug": "drsr",
        "dataset_id": "BPG3", "seed": 520, "task_id": selected_id,
        "old_formal_task_id": stable_id, "logical_key": "drsr::BPG3::s520::clean",
        "minute180_valid_output": True, "terminal_expression": "x0 + x1",
        "terminal_expression_sha256": plan.sha(b"x0 + x1"),
        "terminal_source_path": source_path,
        "terminal_source_sha256": plan.sha(snapshot_raw.encode()),
        "terminal_archive_member": member,
        "selected_result_source_path": result_path,
        "selected_result_sha256": plan.sha(result_raw.encode()),
        "old_formal_result_sha256": plan.sha(old_raw.encode()),
        "feature_names": ["t", "P"], "target_name": "dP_dt",
        "trajectory_bundle_sha256": "bundle-sha",
    }
    preflight = {
        "logical_key": terminal["logical_key"], "task_id": selected_id,
        "terminal_archive_member": member,
        "terminal_expression_sha256": terminal["terminal_expression_sha256"],
    }
    old = {
        "source": {
            "algorithm": "drsr", "dataset_id": "BPG3", "seed": "520",
            "noise_tag": "clean", "task_id": stable_id,
        },
        "result": {"raw_text": old_raw, "sha256": plan.sha(old_raw.encode())},
    }
    snapshot_record = {
        "archive_member": member, "selected_task_id": selected_id,
        "terminal_source_sha256": terminal["terminal_source_sha256"],
        "raw_text": snapshot_raw,
    }
    result_record = {
        "selected_task_id": selected_id, "result_source_path": result_path,
        "result_sha256": terminal["selected_result_sha256"],
        "raw_text": result_raw,
    }
    archive_record = {"sha256": terminal["terminal_source_sha256"], "source_path": source_path}
    gt_variables, gt_targets, probes, _ = plan.load_ground_truth_and_probes()
    contract = builder._load_prompt_schema(plan.ROOT)
    return {
        "key": key, "terminal": terminal, "preflight": preflight, "old": old,
        "snapshot_record": snapshot_record, "result_record": result_record,
        "archive_record": archive_record, "evaluation_path": "canonical_replay.v1", "contract": contract,
        "gt_variables": gt_variables, "gt_targets": gt_targets, "probes": probes,
    }


def test_changed_selected_task_id_uses_old_stable_evaluation_id() -> None:
    record = plan.build_one(**_fixtures())
    assert record["logical_id"] == "pred_simplify::drsr::g0005::s520::clean"
    assert record["source_binding"]["actual_selected_task_id"] == "drsr_s520_clean_g0999"
    assert record["request"]["expression"] == "t + P"
    assert record["terminal_snapshot"]["canonical_artifact"]["instantiated_expression"] == "x0+x1"


def test_selected_result_variable_mapping_mismatch_fails_closed() -> None:
    values = _fixtures()
    result = json.loads(values["result_record"]["raw_text"])
    result["feature_names"] = ["P", "t"]
    values["result_record"]["raw_text"] = json.dumps(result)
    values["result_record"]["result_sha256"] = plan.sha(values["result_record"]["raw_text"].encode())
    values["terminal"]["selected_result_sha256"] = values["result_record"]["result_sha256"]
    with pytest.raises(ValueError, match="feature mapping mismatch"):
        plan.build_one(**values)


def test_snapshot_sha_mismatch_fails_closed() -> None:
    values = _fixtures()
    values["archive_record"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="archive physical index"):
        plan.build_one(**values)


def test_qlattice_stale_canonical_is_rebuilt_from_raw_equation() -> None:
    values = _fixtures()
    values["terminal"]["algorithm"] = "QLattice"
    values["terminal"]["algorithm_slug"] = "qlattice"
    values["terminal"]["terminal_expression"] = "x1"
    values["terminal"]["terminal_expression_sha256"] = plan.sha(b"x1")
    snapshot = json.loads(values["snapshot_record"]["raw_text"])
    snapshot["equation"] = "x1"
    snapshot["canonical_artifact"]["instantiated_expression"] = "x0"
    snapshot["canonical_artifact"]["normalized_expression"] = "x0"
    values["snapshot_record"]["raw_text"] = json.dumps(snapshot)
    values["snapshot_record"]["terminal_source_sha256"] = plan.sha(values["snapshot_record"]["raw_text"].encode())
    values["terminal"]["terminal_source_sha256"] = values["snapshot_record"]["terminal_source_sha256"]
    values["archive_record"]["sha256"] = values["terminal"]["terminal_source_sha256"]
    audited = plan.audit_formula_source(snapshot, values["terminal"], evaluation_path="canonical_replay.v1")
    assert audited["discrepancy"] is True
    assert audited["resolution"] == "canonical_replay_forced_raw_rebuild"
    assert audited["corrected_artifact"]["instantiated_expression"] == "x1"


def test_relative_and_absolute_archive_source_paths_are_same() -> None:
    relative = "AAAI_experiments/stage5_metric_calculation_0831/source.json"
    assert plan._same_source_path(relative, str(plan.ROOT / relative))


def test_llmsr_recovery_function_and_params_prove_terminal_formula() -> None:
    snapshot = {
        "function": "def equation(x0, x1, params):\n    if x1 == 0:\n        raise ValueError()\n    return params[0] + params[1] * 9.81e9 * x0 / x1\n",
        "params": [3.0, 0.5],
    }
    terminal = {
        "terminal_expression": "3.0 + 9810000000.0*0.5*x0/x1",
        "feature_names": ["q", "epsilon"],
        "algorithm_slug": "llmsr",
    }
    audited = plan.audit_formula_source(snapshot, terminal, evaluation_path="canonical_replay.v1")
    assert audited["resolution"] == "recovery_function_params_symbolic_equality"
    assert audited["status"] == "verified"
