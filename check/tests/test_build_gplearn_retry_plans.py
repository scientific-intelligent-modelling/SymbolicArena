"""Versioned retries keep the exhausted source and dependency keys."""

from __future__ import annotations

import json

from check.build_gplearn_exhausted_retry_plan import build as build_simplify_retry


def test_simplify_retry_preserves_source_binding(tmp_path) -> None:
    source = tmp_path / "pred.jsonl"
    unresolved = tmp_path / "unresolved.json"
    prompt = tmp_path / "prompt.txt"
    schema = tmp_path / "schema.json"
    source.write_text(json.dumps({
        "evaluation_key": "source", "logical_id": "pred_simplify::gplearn::test",
        "task_type": "pred_simplify", "priority": 20, "condition": "clean",
        "request": {"evidence_hash": "a" * 64, "expression": "x0"},
    }) + "\n", encoding="utf-8")
    unresolved.write_text(json.dumps({"items": [{"source_evaluation_key": "source",
                                                "evaluation_key": "model", "logical_id": "pred_simplify::gplearn::test",
                                                "attempts": 2}]}), encoding="utf-8")
    prompt.write_text("REQUEST_JSON:\n{{REQUEST_JSON}}", encoding="utf-8")
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    output = tmp_path / "simplify_retry"
    report = build_simplify_retry(source, unresolved, prompt, schema, output)
    row = json.loads((output / "pred_simplify_retry_plan.jsonl").read_text())
    assert report["logical_tasks"] == 1
    assert row["request"]["prior_exhausted_source_evaluation_key"] == "source"
    assert row["request"]["prior_exhausted_model_evaluation_key"] == "model"
