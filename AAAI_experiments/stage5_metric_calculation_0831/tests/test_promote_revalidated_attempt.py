from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
    build_claude_command,
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.promote_revalidated_attempt import (
    PromotionError,
    promote_revalidated_attempt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_failed_validation_can_be_promoted_without_new_attempt(tmp_path: Path) -> None:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "dataset_id": "demo",
        "expression": "x0 + 0",
        "original_expression": "x0 + 0",
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": "demo-evidence",
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    logical_id = "gt_simplify::demo"
    task_key = evaluation_key(
        task_type="gt_simplify",
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash="demo-evidence",
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="gt_simplify",
        condition="clean",
        priority=10,
        input_hash=input_hash,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    plan_row = {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "gt_simplify",
        "condition": "clean",
        "priority": 10,
        "input_hash": input_hash,
        "prompt_version": "simplify.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "simplify.v1",
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_template": prompt_template,
        "schema_content": schema,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
        "rendered_prompt": render_prompt(prompt_template, request, schema),
    }
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(plan_row) + "\n", encoding="utf-8")

    state_db = tmp_path / "control/state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    lease = store.reserve_attempt(task_key, lease_seconds=30.0)
    store.finish_failure(
        lease.attempt_id,
        error_class="validation_failed",
        retryable=True,
    )
    structured_output = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "Removed an additive zero.",
    }
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 1,
        "stop_reason": "end_turn",
        "permission_denials": [],
        "result": "```json\n" + json.dumps(structured_output) + "\n```",
        "usage": {
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}
        },
        "subagent_stats": {"spawned": 0},
        "modelUsage": {
            CONTRACT_MODEL: {"canonicalModel": "claude-opus-5"}
        },
    }
    rendered_prompt = render_prompt(prompt_template, request, schema)
    stdout = json.dumps(envelope)
    stderr = ""
    attempt_path = tmp_path / f"{lease.attempt_id}.json"
    attempt_path.write_text(
        json.dumps(
            {
                "attempt_id": lease.attempt_id,
                "evaluation_key": task_key,
                "request": request,
                "prompt": rendered_prompt,
                "command": build_claude_command(schema),
                "stdout": stdout,
                "stderr": stderr,
                "envelope": envelope,
                "validation": {
                    "ok": False,
                    "error_class": "validation_failed",
                    "structured_output": None,
                },
                "metadata": {
                    "attempt_id": lease.attempt_id,
                    "attempt_number": lease.attempt_number,
                    "evaluation_key": task_key,
                    "logical_id": logical_id,
                    "task_type": "gt_simplify",
                    "task_kind": "simplify",
                    "requested_model": CONTRACT_MODEL,
                    "requested_effort": CONTRACT_EFFORT,
                    "transport_version": CONTRACT_TRANSPORT_VERSION,
                    "prompt_path": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "rendered_prompt_sha256": _sha256_text(rendered_prompt),
                    "schema_path": str(schema_path),
                    "schema_sha256": schema_sha256,
                    "request_sha256": _sha256_text(canonical_json(request)),
                    "stdout_sha256": _sha256_text(stdout),
                    "stderr_sha256": _sha256_text(stderr),
                    "error_class": "validation_failed",
                    "retryable": True,
                    "total_cost_usd": 0.1,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = promote_revalidated_attempt(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempt_json=attempt_path,
        frozen_dir=tmp_path / "frozen",
        predecessor_attempt_manifest=None,
        audit_reason="unit_test_validator_fix",
    )

    restarted = TaskStateStore(state_db)
    assert restarted.attempts_reserved() == 1
    assert restarted.task_state(task_key) == "frozen"
    assert report["new_model_call"] is False
    assert report["attempt_count_delta"] == 0
    frozen = json.loads(Path(str(report["frozen_path"])).read_text(encoding="utf-8"))
    assert frozen["validation"]["ok"] is True
    assert frozen["validation"]["semantic_evidence"]["decision"] == "equivalent"
    assert report["strict_contract_revalidated"] is True


def test_stored_structured_output_cannot_bypass_invalid_envelope(
    tmp_path: Path,
) -> None:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "dataset_id": "demo",
        "expression": "x0 + 0",
        "original_expression": "x0 + 0",
        "variables": ["x0"],
        "allowed_functions": [],
        "evidence_hash": "demo-evidence",
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    logical_id = "gt_simplify::demo"
    task_key = evaluation_key(
        task_type="gt_simplify",
        logical_id=logical_id,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash="demo-evidence",
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="gt_simplify",
        condition="clean",
        priority=10,
        input_hash=input_hash,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    rendered_prompt = render_prompt(prompt_template, request, schema)
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(
        canonical_json(
            {
                "evaluation_key": task_key,
                "logical_id": logical_id,
                "task_type": "gt_simplify",
                "condition": "clean",
                "priority": 10,
                "input_hash": input_hash,
                "prompt_version": "simplify.v1",
                "prompt_sha256": prompt_sha256,
                "schema_version": "simplify.v1",
                "schema_sha256": schema_sha256,
                "dependencies": [],
                "prompt_path": str(prompt_path),
                "schema_path": str(schema_path),
                "prompt_template": prompt_template,
                "schema_content": schema,
                "normalized_input": normalized_input,
                "request": request,
                "task_spec": json.loads(spec.canonical_json()),
                "rendered_prompt": rendered_prompt,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_db = tmp_path / "control/state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    lease = store.reserve_attempt(task_key, lease_seconds=30.0)
    store.finish_failure(
        lease.attempt_id,
        error_class="validation_failed",
        retryable=True,
    )
    structured_output = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "Removed an additive zero.",
    }
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 2,
        "stop_reason": "end_turn",
        "permission_denials": [],
        "result": json.dumps(structured_output),
        "usage": {"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}},
        "subagent_stats": {"spawned": 0},
        "modelUsage": {CONTRACT_MODEL: {"canonicalModel": "claude-opus-5"}},
    }
    stdout = json.dumps(envelope)
    attempt_path = tmp_path / f"{lease.attempt_id}.json"
    attempt_path.write_text(
        json.dumps(
            {
                "attempt_id": lease.attempt_id,
                "evaluation_key": task_key,
                "request": request,
                "prompt": rendered_prompt,
                "command": build_claude_command(schema),
                "stdout": stdout,
                "stderr": "",
                "envelope": envelope,
                "validation": {
                    "ok": False,
                    "error_class": "validation_failed",
                    "structured_output": structured_output,
                },
                "metadata": {
                    "attempt_id": lease.attempt_id,
                    "attempt_number": lease.attempt_number,
                    "evaluation_key": task_key,
                    "logical_id": logical_id,
                    "task_type": "gt_simplify",
                    "task_kind": "simplify",
                    "requested_model": CONTRACT_MODEL,
                    "requested_effort": CONTRACT_EFFORT,
                    "transport_version": CONTRACT_TRANSPORT_VERSION,
                    "prompt_path": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "rendered_prompt_sha256": _sha256_text(rendered_prompt),
                    "schema_path": str(schema_path),
                    "schema_sha256": schema_sha256,
                    "request_sha256": _sha256_text(canonical_json(request)),
                    "stdout_sha256": _sha256_text(stdout),
                    "stderr_sha256": _sha256_text(""),
                    "error_class": "validation_failed",
                    "retryable": True,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromotionError, match="num_turns"):
        promote_revalidated_attempt(
            plan_jsonl=plan_path,
            state_db=state_db,
            attempt_json=attempt_path,
            frozen_dir=tmp_path / "frozen",
            predecessor_attempt_manifest=None,
            audit_reason="unit_test_reject_multiturn",
        )
    assert store.task_state(task_key) == "retry_wait"
    assert not (tmp_path / "frozen" / f"{task_key}.json").exists()
