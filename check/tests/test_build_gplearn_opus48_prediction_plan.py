"""Frozen probes may contain fewer than sixteen distinct task points."""

from __future__ import annotations

from check.build_gplearn_opus48_prediction_plan import request_for


def test_request_accepts_frozen_ten_point_probe() -> None:
    row = {"logical_key": "gplearn::Nguyen-12::s520::clean", "feature_names": ["x1", "x2"],
           "dataset_id": "Nguyen-12", "task_id": "gplearn_s520_clean_g0025",
           "condition": "clean", "seed": 520,
           "terminal_snapshot_sha256": "a" * 64, "terminal_expression_sha256": "b" * 64,
           "native_prefix_sha256": "c" * 64, "selected_result_sha256": "d" * 64,
           "native_prefix": "add(X0,X1)"}
    probe = {"basename": "Nguyen-12", "variables": ["x1", "x2"],
             "points": [{"values": {"x1": float(i), "x2": float(i + 1)}} for i in range(10)],
             "point_count": 10, "sample_sha256": "e" * 64, "schema_version": "dataset_probes_v1"}
    typed = {"typed_expression": "add(x0,x1)", "exact_fingerprint": "f" * 64,
             "constants_abstracted_structure_fingerprint": "g" * 64,
             "node_count": 3, "nodes": [{"kind": "operator", "operator": "add"}],
             "protected_semantics": {}}
    request = request_for(row, probe, typed)
    assert len(request["probe_points"]) == 10
