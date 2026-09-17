"""Frozen probes may contain fewer than sixteen distinct task points."""

from __future__ import annotations

from check.build_gplearn_opus48_prediction_plan import logical_id_for, request_for


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


def test_dataset_identity_prevents_reused_g_index_collision() -> None:
    first = {"dataset_id": "Nguyen-12", "seed": 521, "condition": "noise001"}
    second = {"dataset_id": "feynman-i.39.22", "seed": 521, "condition": "noise001"}
    assert logical_id_for(first, "g0021") != logical_id_for(second, "g0021")
