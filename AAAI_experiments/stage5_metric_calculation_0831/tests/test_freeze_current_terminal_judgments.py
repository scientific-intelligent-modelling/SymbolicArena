"""Focused binding tests; full inventory is checked by the exporter itself."""

from __future__ import annotations

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline import (
    freeze_current_terminal_judgments as frozen,
)


def _fixture():
    prefix = ("clean", "pysr", "BPG3")
    bindings = {}
    for seed in (520, 521):
        bindings[(*prefix, seed)] = {
            "processing_status": "ready", "valid_output": True,
            "terminal_expression": f"x0+{seed}",
            "terminal_expression_sha256": f"term-{seed}",
            "terminal_source_sha256": f"source-{seed}",
            "selected_result_sha256": f"result-{seed}",
            "prediction_effective_expression": f"x+{seed}",
            "prediction_frozen_evaluation_key": f"pred-{seed}",
            "feature_names": ["x"], "variable_mapping": {"x0": "x"},
        }
    gt = {"BPG3": {"fixed_reference_expression": "x+1",
                    "gt_frozen_evaluation_key": "gt-key"}}
    return prefix, bindings, gt


def test_equivalence_must_match_current_terminal_and_reference() -> None:
    prefix, bindings, gt = _fixture()
    request = {
        "noise_tag": "clean", "prediction_terminal_expression_sha256": "term-520",
        "prediction_terminal_source_sha256": "source-520",
        "prediction_current_selected_result_sha256": "result-520",
        "effective_prediction_expression": "x+520",
        "prediction_frozen_evaluation_key": "pred-520",
        "effective_ground_truth_expression": "x+1",
        "ground_truth_frozen_evaluation_key": "gt-key",
        "variables": ["x"], "prediction_variable_mapping": {"x0": "x"},
    }
    assert frozen._check_current("equivalence", (*prefix, 520), request, bindings, gt) == [
        "gt-key", "pred-520"]
    with pytest.raises(frozen.JudgmentBindingError, match="terminal_source_sha256 drift"):
        frozen._check_current("equivalence", (*prefix, 520),
                              {**request, "prediction_terminal_source_sha256": "old-source"},
                              bindings, gt)
    with pytest.raises(frozen.JudgmentBindingError, match="gt_expression drift"):
        frozen._check_current("equivalence", (*prefix, 520),
                              {**request, "effective_ground_truth_expression": "x+2"},
                              bindings, gt)


def test_structure_requires_both_seed_dependencies() -> None:
    prefix, bindings, gt = _fixture()
    request = {"noise_tag": "clean"}
    for side, seed in (("a", 520), ("b", 521)):
        request.update({
            f"prediction_{side}_terminal_expression_sha256": f"term-{seed}",
            f"prediction_{side}_terminal_source_sha256": f"source-{seed}",
            f"prediction_{side}_current_selected_result_sha256": f"result-{seed}",
            f"effective_prediction_{side}_expression": f"x+{seed}",
            f"prediction_{side}_frozen_evaluation_key": f"pred-{seed}",
            f"prediction_{side}_variable_mapping": {"x0": "x"},
        })
    assert frozen._check_current("structure", (*prefix, 520, 521), request, bindings, gt) == [
        "pred-520", "pred-521"]
    with pytest.raises(frozen.JudgmentBindingError, match="b_prediction_key drift"):
        frozen._check_current("structure", (*prefix, 520, 521),
                              {**request, "prediction_b_frozen_evaluation_key": "old-key"},
                              bindings, gt)


def test_undetermined_cannot_be_written_as_effective() -> None:
    with pytest.raises(frozen.JudgmentBindingError, match="undecided"):
        frozen._effective_row("equivalence", ("clean", "pysr", "BPG3", 520), {},
                              decision="undetermined", dependencies=["gt", "pred"],
                              evaluation_key="key", response_path="path", response_sha256="sha",
                              structured_output={}, model="opus", prompt_version="v1",
                              prompt_sha256="sha", schema_version="v1", schema_sha256="sha",
                              source="test")


def test_historical_logical_id_accepts_only_version_suffix() -> None:
    current = "equivalence::imcts::g0005::s520::clean"
    assert frozen._logical_id_matches(current, current, "clean")
    assert frozen._logical_id_matches(current, current + "::v2", "clean")
    assert not frozen._logical_id_matches(current, current + "::noise001", "clean")
    assert not frozen._logical_id_matches(current, current + "::vfoo", "clean")
    structure = "stab_structure::drsr::g0005::s521-s522"
    assert frozen._logical_id_matches(structure, structure + "::noise001", "noise001")
    assert not frozen._logical_id_matches(structure, structure + "::noise005", "noise001")
