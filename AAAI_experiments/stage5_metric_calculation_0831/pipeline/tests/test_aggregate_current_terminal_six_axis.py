"""当前终点六轴聚合器的来源绑定和缺证据门禁。"""

from __future__ import annotations

import copy
import hashlib

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.aggregate_current_terminal_six_axis import (
    aggregate,
)


def _fixtures():
    terminals = []
    predictions = []
    equivalences = []
    structures = []
    grids = {}
    for seed in (520, 521, 522):
        expression = "x0"
        digest = hashlib.sha256(expression.encode()).hexdigest()
        key = ("clean", "toy", "task-a", seed)
        source = f"source-{seed}"
        terminal = {
            "condition": "clean", "algorithm": "toy", "dataset_id": "task-a", "seed": seed,
            "logical_key": f"toy::task-a::s{seed}::clean",
            "selection_status": "formal_raw_retained", "superseded": False,
            "feature_names": ["x0"], "terminal_expression": expression,
            "terminal_expression_sha256": digest, "terminal_source_sha256": source,
            "selected_result_sha256": f"result-{seed}", "minute180_valid_output": True,
            "minute180_id_quality": 0.8, "minute180_ood_quality": 0.6,
        }
        terminals.append(terminal)
        prediction = {field: terminal[field] for field in (
            "condition", "algorithm", "dataset_id", "seed", "terminal_expression",
            "terminal_expression_sha256", "terminal_source_sha256", "selected_result_sha256",
        )}
        prediction.update(processing_status="ready", prediction_effective_expression="x0",
                          prediction_frozen_evaluation_key=f"pred-{seed}",
                          prediction_response_sha256=f"response-{seed}")
        predictions.append(prediction)
        equivalences.append({
            "condition": "clean", "algorithm": "toy", "dataset_id": "task-a", "seed": seed,
            "state": "frozen", "evaluation_key": f"eq-{seed}", "response_sha256": f"eq-response-{seed}",
            "dependencies": ["gt", f"pred-{seed}"], "effective_decision": "equivalent",
            "terminal_expression_sha256": digest, "terminal_source_sha256": source,
        })
        grids[key] = {
            "q": {minute: 0.7 for minute in range(1, 181)},
            "endpoint": {"expression": expression, "source_sha256": source,
                         "valid_output": "true", "id_quality": "0.8", "ood_quality": "0.6"},
        }
    for left, right in ((520, 521), (520, 522), (521, 522)):
        structures.append({
            "condition": "clean", "algorithm": "toy", "dataset_id": "task-a",
            "seed_left": left, "seed_right": right, "state": "frozen",
            "evaluation_key": f"pair-{left}-{right}", "response_sha256": f"pair-response-{left}-{right}",
            "dependencies": [f"pred-{left}", f"pred-{right}"],
            "effective_decision": "same_canonical_structure",
            "left_terminal_expression_sha256": terminals[0]["terminal_expression_sha256"],
            "right_terminal_expression_sha256": terminals[0]["terminal_expression_sha256"],
            "left_terminal_source_sha256": f"source-{left}",
            "right_terminal_source_sha256": f"source-{right}",
        })
    return dict(terminal_rows=terminals, prediction_rows=predictions,
                gt_rows=[{"dataset_id": "task-a", "fixed_reference_expression": "x0",
                          "gt_frozen_evaluation_key": "gt", "gt_result_sha256": "gt-result",
                          "gt_source_evidence_sha256": "gt-source"}],
                equivalence_rows=equivalences, structure_rows=structures,
                grids=grids, expected_runs_per_condition=1)


def test_complete_evidence_computes_task_scores():
    result = aggregate(**_fixtures())
    assert len(result["run_metrics"]) == 3
    assert all(row["m_sym"] == 1 and row["m_min"] == 1 and row["m_eff"] == 1
               for row in result["run_metrics"])
    task = result["task_stability"][0]
    assert (task["n"], task["v"], task["c"], task["m_stab"]) == (1, 1, 1, 1)
    assert not result["manifest"]["formal_ready"]  # 全量仍需要三个条件 15 算法。


def test_missing_equivalence_is_unresolved_not_zero():
    inputs = _fixtures()
    inputs["equivalence_rows"] = inputs["equivalence_rows"][1:]
    result = aggregate(**inputs)
    missing = next(row for row in result["run_metrics"] if row["seed"] == 520)
    assert missing["m_sym"] is None and missing["m_min"] == 1
    assert "equivalence_judgment_missing" in missing["unresolved"]
    assert result["task_stability"][0]["m_stab"] == 1


def test_old_terminal_judgment_cannot_be_reused():
    inputs = _fixtures()
    inputs["equivalence_rows"][0]["terminal_expression_sha256"] = "old-expression"
    inputs["structure_rows"][0]["left_terminal_source_sha256"] = "old-source"
    result = aggregate(**inputs)
    assert "equivalence_judgment_terminal_expression_mismatch" in result["run_metrics"][0]["unresolved"]
    assert "pair_520_521_judgment_left_terminal_source_sha256_mismatch" in result["task_stability"][0]["unresolved"]


def test_minute180_must_match_terminal_formula():
    inputs = _fixtures()
    inputs["grids"] = copy.deepcopy(inputs["grids"])
    inputs["grids"][("clean", "toy", "task-a", 520)]["endpoint"]["expression"] = "x0+1"
    result = aggregate(**inputs)
    row = result["run_metrics"][0]
    assert "minute180_expression_mismatch" in row["unresolved"]
    assert row["m_eff"] is None


def test_missing_pair_preserves_known_n_and_v_but_not_stab():
    inputs = _fixtures()
    inputs["structure_rows"] = inputs["structure_rows"][1:]
    result = aggregate(**inputs)
    task = result["task_stability"][0]
    assert (task["n"], task["v"]) == (1, 1)
    assert task["c"] is None and task["m_stab"] is None


def test_invalid_seed_pair_is_zero_without_model_call():
    inputs = _fixtures()
    terminal = inputs["terminal_rows"][2]
    terminal["minute180_valid_output"] = False
    inputs["prediction_rows"][2]["processing_status"] = "invalid_final_output"
    inputs["grids"][("clean", "toy", "task-a", 522)]["endpoint"]["valid_output"] = "false"
    inputs["structure_rows"] = inputs["structure_rows"][:1]
    result = aggregate(**inputs)
    assert result["run_metrics"][2]["m_sym"] == 0
    assert result["run_metrics"][2]["m_min"] == 0
    task = result["task_stability"][0]
    assert task["v"] == 2 / 3 and task["c"] == 1 / 3
    assert task["m_stab"] is not None


def test_new_run_supersession_is_selected_not_rejected():
    inputs = _fixtures()
    inputs["terminal_rows"][0]["selection_status"] = "eff_overlay_supersession"
    inputs["terminal_rows"][0]["superseded"] = True
    result = aggregate(**inputs)
    assert result["run_metrics"][0]["formal_ready"]


def test_missing_fixed_gt_evidence_is_unresolved():
    inputs = _fixtures()
    inputs["gt_rows"][0]["gt_source_evidence_sha256"] = None
    result = aggregate(**inputs)
    assert all("ground_truth_missing" in row["unresolved"] for row in result["run_metrics"])
    assert all(row["m_sym"] is None and row["m_min"] is None for row in result["run_metrics"])
