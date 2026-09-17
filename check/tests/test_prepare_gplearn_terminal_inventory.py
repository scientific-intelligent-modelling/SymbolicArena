"""A constant gplearn program is a valid native prefix leaf."""

from __future__ import annotations

import json

from check.prepare_gplearn_terminal_inventory import bind_one, sha_text


def test_constant_native_program_is_bound() -> None:
    key = "gplearn::CRK0::s522::noise005"
    snapshot = {"equation": "0.000", "status": "ok", "feature_names": ["x0", "x1"],
                "source_loss": 0.1,
                "canonical_artifact": {"raw_equation": "0.000", "raw_equation_kind": "plain_expression",
                                       "normalized_expression": "0.0", "normalization_mode": "gplearn_prefix_to_infix"}}
    selected = {"equation": "1.000"}
    snapshot_text, selected_text = json.dumps(snapshot), json.dumps(selected)
    terminal = {"logical_key": key, "condition": "noise005", "algorithm_slug": "gplearn",
                "dataset_id": "CRK0", "seed": 522, "task_id": "task",
                "feature_names": ["x0", "x1"], "terminal_expression": "0.0",
                "terminal_expression_sha256": sha_text("0.0"),
                "terminal_source_sha256": sha_text(snapshot_text),
                "terminal_archive_member": "member", "selected_result_sha256": sha_text(selected_text),
                "minute180_valid_output": True}
    native = {"logical_key": key, "expression": ["0.0"] * 180,
              "source_sha256": [sha_text(snapshot_text)] * 180,
              "valid_output": [True] * 180, "id_quality": [0.5] * 180,
              "ood_quality": [0.5] * 180, "incumbent_source_minute": [1] * 180}
    gt = {"dataset_id": "CRK0", "gt_frozen_evaluation_key": "gt",
          "fixed_reference_expression": "0", "gt_source_evidence_sha256": "g" * 64}
    row = bind_one(terminal, {"logical_key": key, "raw_text": snapshot_text},
                   {"logical_key": key, "raw_text": selected_text}, native, gt)
    assert row["native_prefix"] == "0.000"
