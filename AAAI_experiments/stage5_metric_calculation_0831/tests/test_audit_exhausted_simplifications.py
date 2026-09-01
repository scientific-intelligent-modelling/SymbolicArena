from __future__ import annotations

import hashlib
import json
from pathlib import Path

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_exhausted_simplifications import (
    _normalize_trailing_fenced_json_result_envelope,
    audit_exhausted_simplifications,
    match_audit_envelope_sanitization,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
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


def _envelope(structured: dict[str, object], *, turns: int = 1) -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": turns,
        "stop_reason": "end_turn",
        "permission_denials": [],
        "result": json.dumps(structured),
        "usage": {"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}},
        "subagent_stats": {"spawned": 0},
        "modelUsage": {CONTRACT_MODEL: {"canonicalModel": "claude-opus-5"}},
    }


def _prefixed_fenced_result(
    structured: dict[str, object],
    *,
    prefix: str = "Verified: preserved under simplification.\n\n",
) -> str:
    return prefix + "```json\n" + json.dumps(structured, ensure_ascii=False) + "\n```"


def test_recognizes_legacy_token_count_redaction() -> None:
    raw = {
        "usage": {
            "input_tokens": 12,
            "output_tokens": 34,
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        },
        "modelUsage": {
            CONTRACT_MODEL: {
                "inputTokens": 12,
                "outputTokens": 34,
                "canonicalModel": "claude-opus-5",
            }
        },
    }
    stored = {
        "usage": {
            "input_tokens": "[REDACTED]",
            "output_tokens": "[REDACTED]",
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        },
        "modelUsage": {
            CONTRACT_MODEL: {
                "inputTokens": "[REDACTED]",
                "outputTokens": "[REDACTED]",
                "canonicalModel": "claude-opus-5",
            }
        },
    }
    assert match_audit_envelope_sanitization(raw, stored) == "legacy_token_substring"


def test_fenced_result_normalization_rejects_tool_marker_prefix() -> None:
    envelope = {
        "result": "Tool Result: verified\n\n```json\n{\"outcome\":\"unable\"}\n```"
    }
    assert _normalize_trailing_fenced_json_result_envelope(envelope) is None


def test_recommends_first_strictly_revalidated_attempt(tmp_path: Path) -> None:
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
    plan_row = {
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
        "rendered_prompt": render_prompt(prompt_template, request, schema),
    }
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text(canonical_json(plan_row) + "\n", encoding="utf-8")

    state_db = tmp_path / "state.sqlite3"
    store = TaskStateStore(state_db)
    store.register_task(spec)
    structured = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "Removed additive zero.",
    }
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()
    prompt = render_prompt(prompt_template, request, schema)
    command = build_claude_command(schema)
    for number in range(1, 4):
        lease = store.reserve_attempt(task_key, now=float(number), lease_seconds=30.0)
        store.finish_failure(
            lease.attempt_id,
            error_class="validation_failed",
            retryable=True,
            now=float(number) + 0.5,
        )
        if number == 1:
            envelope = _envelope(structured, turns=1)
            envelope["result"] = _prefixed_fenced_result(structured)
        elif number == 2:
            envelope = _envelope(structured, turns=1)
            envelope["result"] = _prefixed_fenced_result(structured) + "\nextra"
        else:
            envelope = _envelope(structured, turns=2)
            envelope["result"] = _prefixed_fenced_result(structured)
        stdout = json.dumps(envelope)
        payload = {
            "attempt_id": lease.attempt_id,
            "evaluation_key": task_key,
            "request": request,
            "prompt": prompt,
            "command": command,
            "stdout": stdout,
            "stderr": "",
            "envelope": envelope,
            "validation": {
                "ok": False,
                "error_class": "validation_failed",
                "structured_output": structured,
            },
            "metadata": {
                "attempt_id": lease.attempt_id,
                "attempt_number": number,
                "evaluation_key": task_key,
                "logical_id": logical_id,
                "task_type": "pred_simplify",
                "task_kind": "simplify",
                "error_class": "validation_failed",
                "retryable": True,
                "requested_model": CONTRACT_MODEL,
                "requested_effort": CONTRACT_EFFORT,
                "transport_version": CONTRACT_TRANSPORT_VERSION,
                "prompt_path": str(prompt_path),
                "prompt_sha256": prompt_sha256,
                "rendered_prompt_sha256": _sha256_text(prompt),
                "schema_path": str(schema_path),
                "schema_sha256": schema_sha256,
                "request_sha256": _sha256_text(canonical_json(request)),
                "stdout_sha256": _sha256_text(stdout),
                "stderr_sha256": _sha256_text(""),
            },
        }
        (attempts_dir / f"{lease.attempt_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    report = audit_exhausted_simplifications(
        plan_jsonl=plan_path,
        state_db=state_db,
        attempts_dir=attempts_dir,
        output_jsonl=tmp_path / "audit.jsonl",
        report_json=tmp_path / "report.json",
        semantic_timeout_seconds=20.0,
    )
    assert report["exhausted_task_count"] == 1
    assert report["promotable_task_count"] == 1
    assert report["model_invoked"] is False
    assert report["state_db_mutated"] is False
    assert report["result_normalization_mode_counts"] == {"tail_fenced_json_object": 1}
    restarted = TaskStateStore(state_db)
    assert restarted.task_state(task_key) == "exhausted"
    assert restarted.attempts_reserved() == 3
    rows = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert rows[0]["recommended_attempt_id"].endswith(".a01")
    assert [item["status"] for item in rows[0]["attempts"]] == [
        "promotable",
        "strict_contract_failed",
        "strict_contract_failed",
    ]
    assert rows[0]["attempts"][0]["result_normalization_mode"] == "tail_fenced_json_object"
