"""Replanned API calls must exclude identities already in the frozen base plan."""

from __future__ import annotations

import json

from check.select_unresolved_opus48_tasks import select


def test_selects_only_missing_identity(tmp_path) -> None:
    def row(seed: int, key: str) -> dict:
        return {"condition": "clean", "task_type": "equivalence", "evaluation_key": key,
                "request": {"algorithm_slug": "pysr", "dataset_id": "BPG3", "seed": seed}}

    base = tmp_path / "base.jsonl"
    replan = tmp_path / "replan.jsonl"
    output = tmp_path / "out" / "selected.jsonl"
    base.write_text(json.dumps(row(520, "old")) + "\n", encoding="utf-8")
    replan.write_text("\n".join(json.dumps(item) for item in (
        row(520, "duplicate"), row(521, "new"))) + "\n", encoding="utf-8")
    report = select(base, replan, output)
    assert report["selected_count"] == 1
    assert report["already_planned_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["evaluation_key"] == "new"
