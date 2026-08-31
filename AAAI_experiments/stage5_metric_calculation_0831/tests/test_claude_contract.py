from __future__ import annotations

import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_MODEL,
    ContractViolation,
    build_claude_command,
    evaluation_key,
    render_prompt,
    validate_claude_envelope,
    validate_structured_output,
)


STAGE = Path("AAAI_experiments/stage5_metric_calculation_0831")


def valid_envelope(structured: dict[str, object]) -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 2,
        "stop_reason": "tool_use",
        "permission_denials": [],
        "structured_output": structured,
        "usage": {
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}
        },
        "subagent_stats": {"spawned": 0},
        "modelUsage": {
            CONTRACT_MODEL: {
                "canonicalModel": "claude-opus-5",
                "provider": "firstParty",
            }
        },
    }


def test_evaluation_key_is_deterministic_and_contract_sensitive() -> None:
    kwargs = {
        "task_type": "pred_simplify",
        "logical_id": "pred_simplify::demo",
        "prompt_version": "simplify.v1",
        "schema_version": "simplify.v1",
        "prompt_sha256": "a" * 64,
        "schema_sha256": "b" * 64,
        "normalized_input": {"expression": "x0 + 0"},
        "evidence_hash": "abc",
    }
    first = evaluation_key(**kwargs)
    assert first == evaluation_key(**kwargs)
    assert first != evaluation_key(**{**kwargs, "evidence_hash": "def"})
    assert first != evaluation_key(**{**kwargs, "prompt_sha256": "c" * 64})
    assert first != evaluation_key(**{**kwargs, "schema_sha256": "d" * 64})


def test_build_command_fixes_single_turn_contract() -> None:
    schema = json.loads((STAGE / "config/schemas/simplify.v1.json").read_text())
    command = build_claude_command(schema)
    assert command[0] == "claude"
    assert command[command.index("--model") + 1] == "claude-opus-5[1m]"
    assert command[command.index("--effort") + 1] == "xhigh"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert "--no-session-persistence" in command
    assert command[command.index("--setting-sources") + 1] == "user"


def test_render_prompt_replaces_one_request_placeholder() -> None:
    template = (STAGE / "config/prompts/simplify.v1.txt").read_text()
    rendered = render_prompt(template, {"expression": "x0 + 0"})
    assert "{{REQUEST_JSON}}" not in rendered
    assert '"expression":"x0 + 0"' in rendered


def test_validate_simplify_output_is_strict() -> None:
    output = {
        "outcome": "simplified",
        "simplified_expression": "x0",
        "equivalence_assessment": "preserved",
        "assumptions": [],
        "confidence": 0.9,
        "brief_reason": "Zero was removed.",
    }
    assert validate_structured_output("simplify", output) == output
    with pytest.raises(ContractViolation):
        validate_structured_output("simplify", {**output, "extra": True})
    with pytest.raises(ContractViolation):
        validate_structured_output("simplify", {**output, "confidence": 2})
    unable = {
        **output,
        "outcome": "unable",
        "simplified_expression": None,
        "equivalence_assessment": "preserved",
    }
    with pytest.raises(ContractViolation, match="undetermined"):
        validate_structured_output("simplify", unable)


def test_validate_equivalence_and_structure_enums() -> None:
    equivalence = {
        "decision": "undetermined",
        "evidence_basis": "insufficient",
        "assumptions": [],
        "confidence": 0.3,
        "brief_reason": "Evidence is inconclusive.",
    }
    structure = {
        "decision": "same_canonical_structure",
        "confidence": 0.8,
        "brief_reason": "Trees match after constant abstraction.",
    }
    validate_structured_output("equivalence", equivalence)
    validate_structured_output("structure", structure)
    with pytest.raises(ContractViolation):
        validate_structured_output("structure", {**structure, "decision": "same"})
    with pytest.raises(ContractViolation, match="insufficient"):
        validate_structured_output(
            "equivalence",
            {**equivalence, "decision": "equivalent"},
        )


def test_validate_envelope_accepts_schema_tool_turn_but_no_real_tools() -> None:
    structured = {
        "decision": "equivalent",
        "evidence_basis": "symbolic_proof",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "Expressions are identical.",
    }
    envelope = valid_envelope(structured)
    assert validate_claude_envelope(envelope, task_kind="equivalence") == structured


def test_validate_envelope_rejects_model_or_tool_contract_drift() -> None:
    structured = {
        "decision": "different_structure",
        "confidence": 0.8,
        "brief_reason": "Trees differ.",
    }
    wrong_model = valid_envelope(structured)
    wrong_model["modelUsage"] = {"claude-sonnet": {"canonicalModel": "claude-sonnet"}}
    with pytest.raises(ContractViolation, match="模型"):
        validate_claude_envelope(wrong_model, task_kind="structure")

    web_used = valid_envelope(structured)
    web_used["usage"]["server_tool_use"]["web_search_requests"] = 1
    with pytest.raises(ContractViolation, match="工具"):
        validate_claude_envelope(web_used, task_kind="structure")

    missing_tool_usage = valid_envelope(structured)
    missing_tool_usage["usage"] = {}
    with pytest.raises(ContractViolation, match="server_tool_use"):
        validate_claude_envelope(missing_tool_usage, task_kind="structure")

    missing_subagent_stats = valid_envelope(structured)
    missing_subagent_stats.pop("subagent_stats")
    with pytest.raises(ContractViolation, match="subagent"):
        validate_claude_envelope(missing_subagent_stats, task_kind="structure")

    bad_turn_pair = valid_envelope(structured)
    bad_turn_pair["num_turns"] = 1
    with pytest.raises(ContractViolation, match="stop_reason"):
        validate_claude_envelope(bad_turn_pair, task_kind="structure")
