from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_frozen_simplifications import (
    FrozenSimplificationAuditError,
    audit_frozen_simplifications,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    build_claude_command,
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_ROOT = REPO_ROOT / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_frozen_result_is_revalidated_without_model_call(tmp_path: Path) -> None:
    prompt_path = STAGE_ROOT / "config/prompts/simplify.v1.txt"
    schema_path = STAGE_ROOT / "config/schemas/simplify.v1.json"
    prompt_template = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prompt_sha256 = _sha256_file(prompt_path)
    schema_sha256 = _sha256_file(schema_path)
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
    logical_id = "pred_simplify::demo::g0001::s520::clean"
    task_key = evaluation_key(
        task_type="pred_simplify",
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
        task_type="pred_simplify",
        condition="clean",
        priority=20,
        input_hash=input_hash,
        prompt_version="simplify.v1",
        schema_version="simplify.v1",
        dependencies=(),
    )
    current_rendered_prompt = render_prompt(prompt_template, request, schema)
    rendered_prompt = current_rendered_prompt + "\nARCHIVED_PROMPT_PROJECTION_V0\n"
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(
        canonical_json(
            {
                "evaluation_key": task_key,
                "logical_id": logical_id,
                "task_type": "pred_simplify",
                "task_kind": "simplify",
                "condition": "clean",
                "priority": 20,
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

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    lease = store.reserve_attempt(task_key, lease_seconds=30.0)
    structured = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 0.95,
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
        "result": json.dumps(structured),
        "usage": {"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}},
        "subagent_stats": {"spawned": 0},
        "modelUsage": {CONTRACT_MODEL: {"canonicalModel": "claude-opus-5"}},
    }
    stdout = json.dumps(envelope)
    metadata = {
        "attempt_id": lease.attempt_id,
        "attempt_number": lease.attempt_number,
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "pred_simplify",
        "task_kind": "simplify",
        "requested_model": CONTRACT_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "rendered_prompt_sha256": _sha256_text(rendered_prompt),
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha256,
        "request_sha256": _sha256_text(canonical_json(request)),
        "stdout_sha256": _sha256_text(stdout),
        "stderr_sha256": _sha256_text(""),
        "error_class": None,
        "retryable": False,
        "returncode": 0,
        "timed_out": False,
    }
    validation = {
        "ok": True,
        "error_class": None,
        "error_message": None,
        "structured_output": structured,
        "semantic_evidence": {"decision": "legacy-placeholder"},
    }
    common = {
        "attempt_id": lease.attempt_id,
        "evaluation_key": task_key,
        "request": request,
        "prompt": rendered_prompt,
        "command": build_claude_command(schema),
        "stdout": stdout,
        "stderr": "",
        "envelope": envelope,
        "validation": validation,
        "metadata": metadata,
    }
    attempts_dir = tmp_path / "attempts"
    frozen_dir = tmp_path / "frozen"
    attempts_dir.mkdir()
    frozen_dir.mkdir()
    attempt_path = attempts_dir / f"{lease.attempt_id}.json"
    attempt_path.write_text(json.dumps(common), encoding="utf-8")
    frozen_path = frozen_dir / f"{task_key}.json"
    frozen_payload = {
        **common,
        "logical_id": logical_id,
        "task_type": "pred_simplify",
        "task_kind": "simplify",
        "structured_output": structured,
    }
    frozen_path.write_text(json.dumps(frozen_payload), encoding="utf-8")
    store.freeze_result(
        lease.attempt_id,
        result_path=str(frozen_path.resolve()),
        result_sha256=_sha256_file(frozen_path),
    )

    with pytest.raises(FrozenSimplificationAuditError, match="rendered_prompt"):
        audit_frozen_simplifications(
            plan_jsonl=plan_path,
            state_db=state_db,
            attempts_dir=attempts_dir,
            frozen_dir=frozen_dir,
            output_jsonl=tmp_path / "strict-audit.jsonl",
            report_json=tmp_path / "strict-report.json",
            expected_plan_count=1,
            semantic_timeout_seconds=20.0,
            workers=1,
        )

    archived_prompt_only = audit_frozen_simplifications(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempts_dir=attempts_dir,
        frozen_dir=frozen_dir,
        output_jsonl=tmp_path / "archived-prompt-only.jsonl",
        report_json=tmp_path / "archived-prompt-only-report.json",
        expected_plan_count=1,
        semantic_timeout_seconds=20.0,
        workers=1,
        allow_archived_rendered_prompt=True,
    )
    assert archived_prompt_only["status"] == "failed"
    assert archived_prompt_only["failure_class_counts"] == {"metadata_identity_error": 1}

    report = audit_frozen_simplifications(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempts_dir=attempts_dir,
        frozen_dir=frozen_dir,
        output_jsonl=tmp_path / "audit.jsonl",
        report_json=tmp_path / "report.json",
        expected_plan_count=1,
        semantic_timeout_seconds=20.0,
        workers=1,
        allow_archived_rendered_prompt=True,
        allow_archived_execution_metadata=True,
    )

    assert report["status"] == "ok"
    assert report["passed_count"] == 1
    assert report["failed_count"] == 0
    assert report["model_invoked"] is False
    assert report["state_db_mutated"] is False
    assert report["rendered_prompt_mode"] == "archived_plan_v1"
    assert report["archived_rendered_prompt_mismatch_count"] == 1
    assert report["execution_metadata_mode_counts"] == {
        "archived_missing_transport_version": 1
    }
    assert report["stored_semantic_evidence_mismatch_count"] == 1
    row = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "passed"
    assert row["semantic_decision"] == "equivalent"
    assert row["stored_semantic_evidence_matches_current"] is False
