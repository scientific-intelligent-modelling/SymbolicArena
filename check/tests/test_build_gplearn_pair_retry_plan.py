"""Schema retry retains the frozen GT and prediction dependencies."""

from __future__ import annotations

import json

from check.build_gplearn_schema_retry_plan import build as build_pair_retry


def test_pair_retry_preserves_dependencies(tmp_path) -> None:
    source = tmp_path / "eq.jsonl"
    unresolved = tmp_path / "unresolved.json"
    prompt = tmp_path / "prompt.txt"
    schema = tmp_path / "schema.json"
    source.write_text(json.dumps({
        "evaluation_key": "source", "logical_id": "equivalence::gplearn::test",
        "task_type": "equivalence", "priority": 30, "condition": "clean",
        "dependencies": ["gt", "pred"], "request": {"evidence_hash": "b" * 64},
    }) + "\n", encoding="utf-8")
    unresolved.write_text(json.dumps({"items": [{"source_evaluation_key": "source",
                                                "evaluation_key": "model", "logical_id": "equivalence::gplearn::test",
                                                "attempts": 2}]}), encoding="utf-8")
    prompt.write_text("REQUEST_JSON:\n{{REQUEST_JSON}}", encoding="utf-8")
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    output = tmp_path / "eq_retry"
    report = build_pair_retry(source_plan=source, unresolved=unresolved,
                              prompt=prompt, schema=schema, output=output)
    row = json.loads((output / "pair_retry_plan.jsonl").read_text())
    assert report["logical_tasks"] == 1
    assert row["dependencies"] == ["gt", "pred"]
