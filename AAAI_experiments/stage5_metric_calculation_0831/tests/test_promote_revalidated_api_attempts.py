from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    STRICT_EVALUATOR_SYSTEM_PROMPT,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.promote_revalidated_api_attempts import (
    ApiAttemptPromotionError,
    promote_revalidated_api_attempts,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    TaskSpec,
    TaskStateStore,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _output(decision: str = "equivalent") -> dict[str, object]:
    return {
        "decision": decision,
        "evidence_basis": "symbolic_proof",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "The expressions agree under the declared domain.",
    }


def _fixture(
    tmp_path: Path,
    *,
    output_text: str,
    stored_output_text: str | None = None,
) -> dict[str, Any]:
    prompt_template = "Judge the expressions.\n{{REQUEST_JSON}}\n{{OUTPUT_SCHEMA_JSON}}\n"
    schema: dict[str, object] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["equivalent", "not_equivalent", "undetermined"],
            },
            "evidence_basis": {
                "type": "string",
                "enum": [
                    "symbolic_proof",
                    "numerical_support",
                    "structural_analysis",
                    "mixed",
                    "insufficient",
                ],
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "brief_reason": {"type": "string", "minLength": 1},
        },
        "required": [
            "decision",
            "evidence_basis",
            "assumptions",
            "confidence",
            "brief_reason",
        ],
    }
    prompt_path = tmp_path / "equivalence.v1.txt"
    schema_path = tmp_path / "equivalence.v1.json"
    prompt_path.write_text(prompt_template, encoding="utf-8")
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    schema_sha256 = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    request = {
        "lhs": "x0 + x1",
        "rhs": "x1 + x0",
        "evidence_hash": "fixture-evidence",
    }
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    logical_id = "equivalence::demo::g0001::s520::clean"
    task_key = evaluation_key(
        task_type="equivalence",
        logical_id=logical_id,
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash="fixture-evidence",
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="equivalence",
        condition="clean",
        priority=30,
        input_hash=_sha256_text(canonical_json(normalized_input)),
        prompt_version="equivalence.v1",
        schema_version="equivalence.v1",
        dependencies=(),
    )
    rendered_prompt = render_prompt(prompt_template, request, schema)
    plan_row = {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "equivalence",
        "condition": "clean",
        "priority": 30,
        "input_hash": spec.input_hash,
        "prompt_version": "equivalence.v1",
        "prompt_sha256": prompt_sha256,
        "schema_version": "equivalence.v1",
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
        "task_kind": "equivalence",
    }
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(plan_row) + "\n", encoding="utf-8")

    state_db = tmp_path / "control/state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    for error_class in ("api_timeout", "api_transport_error"):
        lease = store.reserve_attempt(task_key, lease_seconds=30)
        store.finish_failure(lease.attempt_id, error_class=error_class, retryable=True)
    lease = store.reserve_attempt(task_key, lease_seconds=30)
    store.finish_failure(
        lease.attempt_id,
        error_class="structured_output_invalid",
        retryable=True,
    )

    response_body = {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": CONTRACT_CANONICAL_MODEL,
        "content": [{"type": "text", "text": output_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 30},
    }
    api_request = {
        "model": CONTRACT_CANONICAL_MODEL,
        "stream": False,
        "max_tokens": 6144,
        "system": STRICT_EVALUATOR_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": rendered_prompt}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": CONTRACT_EFFORT},
    }
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()
    attempt_path = attempts_dir / f"{lease.attempt_id}.json"
    attempt_payload = {
        "attempt_id": lease.attempt_id,
        "evaluation_key": task_key,
        "request": request,
        "prompt": rendered_prompt,
        "api_request": api_request,
        "api_response": response_body,
        "output_text": output_text if stored_output_text is None else stored_output_text,
        "validation": {
            "ok": False,
            "error_class": "structured_output_invalid",
            "error_message": "old parser rejected decorated output",
            "structured_output": None,
        },
        "metadata": {
            "attempt_id": lease.attempt_id,
            "attempt_number": lease.attempt_number,
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "task_type": "equivalence",
            "task_kind": "equivalence",
            "requested_model": CONTRACT_CANONICAL_MODEL,
            "requested_effort": CONTRACT_EFFORT,
            "transport_version": API_TRANSPORT_VERSION,
            "stream": False,
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "rendered_prompt_sha256": _sha256_text(rendered_prompt),
            "system_prompt_sha256": _sha256_text(STRICT_EVALUATOR_SYSTEM_PROMPT),
            "schema_path": str(schema_path),
            "schema_sha256": schema_sha256,
            "request_sha256": _sha256_text(canonical_json(request)),
            "api_request_sha256": _sha256_text(canonical_json(api_request)),
            "persisted_api_response_sha256": _sha256_text(canonical_json(response_body)),
            "http_status": 200,
            "response_model": CONTRACT_CANONICAL_MODEL,
            "error_class": "structured_output_invalid",
            "retryable": True,
            "usage": response_body["usage"],
        },
    }
    attempt_path.write_text(
        json.dumps(attempt_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "task_key": task_key,
        "attempt_id": lease.attempt_id,
        "attempt_path": attempt_path,
        "plan_path": plan_path,
        "state_db": state_db,
        "attempts_dir": attempts_dir,
    }


def _run(tmp_path: Path, fixture: dict[str, Any], *, dry_run: bool) -> dict[str, object]:
    return promote_revalidated_api_attempts(
        plan_jsonl=fixture["plan_path"],
        state_db=fixture["state_db"],
        attempts_dir=fixture["attempts_dir"],
        frozen_dir=tmp_path / "frozen",
        task_type="equivalence",
        manifest_jsonl=tmp_path / "reports/promotion_manifest.jsonl",
        report_json=tmp_path / "reports/promotion_report.json",
        audit_reason="unit_test_parser_revalidation",
        dry_run=dry_run,
    )


def test_dry_run_finds_unique_valid_object_without_mutating_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, output_text="Corrected result:\n" + json.dumps(_output()))

    report = _run(tmp_path, fixture, dry_run=True)

    store = TaskStateStore(fixture["state_db"])
    assert store.task_state(fixture["task_key"]) == "exhausted"
    assert store.attempts_reserved() == 3
    assert report["promotable_count"] == 1
    assert report["promoted_count"] == 0
    assert report["dry_run"] is True
    assert not (tmp_path / "frozen").exists()
    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "reports/promotion_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest_rows[0]["status"] == "promotable"
    assert manifest_rows[0]["structured_output_recovery"] == (
        "single_valid_embedded_json_object"
    )


def test_apply_promotes_failed_attempt_without_new_model_attempt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, output_text="```json\n" + json.dumps(_output()) + "\n```")

    report = _run(tmp_path, fixture, dry_run=False)

    store = TaskStateStore(fixture["state_db"])
    assert store.task_state(fixture["task_key"]) == "frozen"
    assert store.attempts_reserved() == 3
    assert report["promoted_count"] == 1
    frozen_record = store.frozen_result(fixture["task_key"])
    assert frozen_record is not None
    frozen_path = Path(str(frozen_record["result_path"]))
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_record["result_sha256"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["attempt_id"] == fixture["attempt_id"]
    assert frozen["structured_output"] == _output()
    assert frozen["validation"]["promotion"]["new_model_call"] is False
    assert frozen["metadata"]["promotion_audit_reason"] == (
        "unit_test_parser_revalidation"
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_empty_stored_output_text_recovers_from_hashed_api_response(
    tmp_path: Path,
    dry_run: bool,
) -> None:
    fixture = _fixture(
        tmp_path,
        output_text="Corrected result:\n" + json.dumps(_output()),
        stored_output_text="",
    )

    report = _run(tmp_path, fixture, dry_run=dry_run)

    assert report["promotable_count"] == 1
    assert report["promoted_count"] == (0 if dry_run else 1)
    expected_state = "exhausted" if dry_run else "frozen"
    assert TaskStateStore(fixture["state_db"]).task_state(fixture["task_key"]) == (
        expected_state
    )


def test_frozen_task_historical_failures_are_outside_matching_count(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, output_text=json.dumps(_output()))
    frozen_stub = tmp_path / "already_frozen.json"
    frozen_stub.write_text("{}\n", encoding="utf-8")
    with sqlite3.connect(fixture["state_db"]) as connection:
        accepted_attempt = connection.execute(
            "SELECT attempt_id FROM attempts WHERE evaluation_key=? ORDER BY attempt_number LIMIT 1",
            (fixture["task_key"],),
        ).fetchone()[0]
        connection.execute(
            "UPDATE attempts SET status='accepted' WHERE attempt_id=?",
            (accepted_attempt,),
        )
        connection.execute(
            "UPDATE tasks SET state='frozen' WHERE evaluation_key=?",
            (fixture["task_key"],),
        )
        connection.execute(
            """INSERT INTO frozen_results(
                   evaluation_key, attempt_id, result_path, result_sha256, frozen_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                fixture["task_key"],
                accepted_attempt,
                str(frozen_stub),
                hashlib.sha256(frozen_stub.read_bytes()).hexdigest(),
                1.0,
            ),
        )

    report = promote_revalidated_api_attempts(
        plan_jsonl=fixture["plan_path"],
        state_db=fixture["state_db"],
        attempts_dir=fixture["attempts_dir"],
        frozen_dir=tmp_path / "frozen",
        task_type="equivalence",
        manifest_jsonl=tmp_path / "reports/post_freeze_manifest.jsonl",
        report_json=tmp_path / "reports/post_freeze_report.json",
        audit_reason="unit_test_post_freeze_scope",
        dry_run=True,
    )

    assert report["matching_failed_attempt_count"] == 0
    assert report["promotable_count"] == 0


def test_two_distinct_valid_objects_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    output_text = json.dumps(_output("equivalent")) + "\n" + json.dumps(
        _output("not_equivalent")
    )
    fixture = _fixture(tmp_path, output_text=output_text)

    report = _run(tmp_path, fixture, dry_run=False)

    assert report["promotable_count"] == 0
    assert report["promoted_count"] == 0
    assert report["rejected_count"] == 1
    assert TaskStateStore(fixture["state_db"]).task_state(fixture["task_key"]) == (
        "exhausted"
    )


def test_explicit_task_type_must_match_entire_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, output_text=json.dumps(_output()))

    with pytest.raises(ApiAttemptPromotionError, match="task_type"):
        promote_revalidated_api_attempts(
            plan_jsonl=fixture["plan_path"],
            state_db=fixture["state_db"],
            attempts_dir=fixture["attempts_dir"],
            frozen_dir=tmp_path / "frozen",
            task_type="structure",
            manifest_jsonl=tmp_path / "manifest.jsonl",
            report_json=tmp_path / "report.json",
            audit_reason="unit_test",
            dry_run=True,
        )
