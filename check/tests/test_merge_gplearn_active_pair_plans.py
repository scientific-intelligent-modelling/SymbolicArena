"""Only a frozen, same-input retry may supersede a failed pair judgment."""

from __future__ import annotations

import json

from check.merge_gplearn_active_pair_plans import select_active


def test_retry_replaces_unfrozen_base_without_changing_input(tmp_path) -> None:
    base_request = {"dataset_id": "BPG3", "seed": 520, "evidence_hash": "a" * 64}
    base = {"condition": "clean", "task_type": "equivalence", "evaluation_key": "source-base",
            "input_hash": "input-base", "dependencies": ["gt", "pred"], "request": base_request}
    retry_request = {**base_request, "prior_exhausted_source_evaluation_key": "source-base",
                     "prior_exhausted_model_evaluation_key": "model-base"}
    retry = {**base, "evaluation_key": "source-retry", "input_hash": "input-retry",
             "request": retry_request}
    base_plan, retry_plan = tmp_path / "base.jsonl", tmp_path / "retry.jsonl"
    base_plan.write_text(json.dumps(base) + "\n", encoding="utf-8")
    retry_plan.write_text(json.dumps(retry) + "\n", encoding="utf-8")
    base_frozen, retry_frozen = tmp_path / "base_frozen", tmp_path / "retry_frozen"
    base_frozen.mkdir()
    retry_frozen.mkdir()
    frozen = {"source_evaluation_key": "source-retry", "evaluation_key": "model-retry",
              "source_input_hash": "input-retry", "status": "frozen", "model": "claude-opus-4-8",
              "request": retry_request, "plan_dependencies": ["gt", "pred"],
              "semantic_validation": {"status": "promotable"}}
    (retry_frozen / "model-retry.json").write_text(json.dumps(frozen), encoding="utf-8")
    output = tmp_path / "out"
    report = select_active(base_plan=base_plan, retry_plan=retry_plan,
                           base_frozen_dir=base_frozen, retry_frozen_dir=retry_frozen,
                           output=output, expected_rows=1)
    assert report["active_rows"] == 1 and report["retry_rows"] == 1
    active = json.loads((output / "equivalence_active_plan.jsonl").read_text())
    assert active["evaluation_key"] == "source-retry"
