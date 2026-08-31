from __future__ import annotations

import json
from pathlib import Path

import pytest

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_MODEL,
    ContractViolation,
    build_claude_command,
    compact_request_for_prompt,
    evaluation_key,
    render_prompt,
    validate_claude_envelope,
    validate_structured_output,
)


STAGE = Path("AAAI_experiments/stage5_metric_calculation_0831")


def test_formal_schemas_use_claude_cli_compatible_draft_07() -> None:
    for path in sorted((STAGE / "config/schemas").glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["additionalProperties"] is False


def valid_envelope(structured: dict[str, object]) -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "num_turns": 1,
        "stop_reason": "end_turn",
        "permission_denials": [],
        "result": json.dumps(structured),
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
    assert "--json-schema" not in command


def test_render_prompt_replaces_one_request_placeholder() -> None:
    template = (STAGE / "config/prompts/simplify.v1.txt").read_text()
    schema = json.loads((STAGE / "config/schemas/simplify.v1.json").read_text())
    rendered = render_prompt(template, {"expression": "x0 + 0"}, schema)
    assert "{{REQUEST_JSON}}" not in rendered
    assert '"expression":"x0 + 0"' in rendered
    assert "OUTPUT_JSON_SCHEMA_DRAFT_07" in rendered
    assert '"$id":"symbolicarena.simplify.v1"' in rendered


def test_prompt_compaction_is_transport_only_and_preserves_hash_material() -> None:
    probe_points = [
        {
            "split": "id_test",
            "row_index": 0,
            "values": {"x0": 1.0, "x1": 2.0},
        }
    ]
    request = {
        "dataset_id": "DemoSet",
        "expression": "x0 + sin(x1)",
        "original_expression": "x0 + sin(x1)",
        "variables": ["x0", "x1"],
        "allowed_functions": ["sin"],
        "domain_assumptions": {
            "number_system": "real",
            "operator_semantics": ["ordinary real semantics"],
        },
        "probe_points": probe_points,
        "probe_source": "dataset_probes_v1",
        "probe_sample_sha256": "a" * 64,
        "dataset_probe_evidence": {
            "schema_version": "dataset_probes_v1",
            "dataset_name": "DemoSet",
            "target_name": "y",
            "variables": ["x0", "x1"],
            "point_count": 1,
            "points": probe_points,
            "source_sha256": {"metadata_yaml": "b" * 64},
            "sample_sha256": "a" * 64,
            "evidence_sha256": "c" * 64,
        },
        "deterministic_evidence": {
            "dataset_probe": {
                "schema_version": "dataset_probes_v1",
                "dataset_name": "DemoSet",
                "target_name": "y",
                "variables": ["x0", "x1"],
                "point_count": 1,
                "points": probe_points,
                "source_sha256": {"metadata_yaml": "b" * 64},
                "sample_sha256": "a" * 64,
                "evidence_sha256": "c" * 64,
            },
            "domain_assumptions": {
                "number_system": "real",
                "operator_semantics": ["ordinary real semantics"],
            },
            "symbolic_artifact": {
                "source_kind": "expression",
                "canonical_expression": "x0 + sin(x1)",
                "source_text": "x0 + sin(x1)",
                "canonical_tree": {"type": "add", "args": [{"type": "symbol"}]},
                "constants_abstracted_canonical_tree": {
                    "type": "add",
                    "args": [{"type": "const"}],
                },
                "constants_abstracted_tree_fingerprint": "d" * 64,
                "node_count": 4,
                "variables": ["x0", "x1"],
                "function_set": ["sin"],
                "operator_set": ["add", "sin"],
                "artifact_sha256": "e" * 64,
            },
        },
        "evidence_hash": "f" * 64,
        "ast_source_evidence": {
            "selected_expression_source": "canonical_artifact.instantiated_expression",
            "selected_expression_before_variable_mapping": "x0 + sin(x1)",
            "semantic_expression_after_variable_mapping": "x0 + sin(x1)",
            "extracted_expression_body": "x0 + sin(x1)",
            "feature_names": ["x0", "x1"],
            "variable_mapping": {"X0": "x0", "X1": "x1"},
            "formula_resolution": {
                "status": "mapped",
                "manifest_entry": {"resolution": "direct"},
            },
            "formula_candidates": {
                "instantiated_expression": "x0 + sin(x1)",
                "normalized_expression": "x0 + sin(x1)",
                "return_expression_source": "x0 + sin(x1)",
                "equation": "x0 + sin(x1)",
            },
            "canonical_artifact": {
                "instantiated_expression": "x0 + sin(x1)",
                "python_function_source": "def equation(x0, x1): return x0 + sin(x1)",
            },
            "result_status": "ok",
            "result_path": "/tmp/demo/result.json",
            "result_raw_sha256": "1" * 64,
            "source_row_sha256": "2" * 64,
        },
    }
    request_snapshot = json.loads(json.dumps(request, ensure_ascii=False, sort_keys=True))

    compacted = compact_request_for_prompt(request)
    assert request == request_snapshot
    assert request["dataset_probe_evidence"]["points"] == probe_points
    assert request["deterministic_evidence"]["dataset_probe"]["points"] == probe_points
    assert "canonical_tree" in request["deterministic_evidence"]["symbolic_artifact"]
    assert (
        "constants_abstracted_canonical_tree"
        in request["deterministic_evidence"]["symbolic_artifact"]
    )
    assert request["original_expression"] == request["expression"]
    assert "formula_candidates" in request["ast_source_evidence"]

    assert compacted["probe_points"] == probe_points
    assert "original_expression" not in compacted
    assert "points" not in compacted["dataset_probe_evidence"]
    assert "points" not in compacted["deterministic_evidence"]["dataset_probe"]
    assert "domain_assumptions" not in compacted["deterministic_evidence"]
    compact_artifact = compacted["deterministic_evidence"]["symbolic_artifact"]
    assert "source_text" not in compact_artifact
    assert "canonical_expression" not in compact_artifact
    assert "canonical_tree" not in compact_artifact
    assert "constants_abstracted_canonical_tree" not in compact_artifact
    assert compact_artifact["node_count"] == 4
    assert compact_artifact["function_set"] == ["sin"]
    assert compact_artifact["operator_set"] == ["add", "sin"]
    assert compact_artifact["artifact_sha256"] == "e" * 64
    assert compact_artifact["constants_abstracted_tree_fingerprint"] == "d" * 64
    assert compacted["dataset_probe_evidence"]["source_sha256"] == {"metadata_yaml": "b" * 64}
    assert compacted["dataset_probe_evidence"]["sample_sha256"] == "a" * 64
    assert compacted["dataset_probe_evidence"]["evidence_sha256"] == "c" * 64
    assert compacted["evidence_hash"] == "f" * 64
    compact_ast = compacted["ast_source_evidence"]
    assert compact_ast == {
        "selected_expression_source": "canonical_artifact.instantiated_expression",
        "feature_names": ["x0", "x1"],
        "variable_mapping": {"X0": "x0", "X1": "x1"},
        "formula_resolution": {
            "status": "mapped",
            "manifest_entry": {"resolution": "direct"},
        },
        "result_status": "ok",
        "result_path": "/tmp/demo/result.json",
        "result_raw_sha256": "1" * 64,
        "source_row_sha256": "2" * 64,
    }

    rendered = render_prompt("REQ={{REQUEST_JSON}}", request)
    assert '"probe_points":[{"row_index":0,"split":"id_test","values":{"x0":1.0,"x1":2.0}}]' in rendered
    assert '"artifact_sha256":"' + ("e" * 64) + '"' in rendered
    assert '"constants_abstracted_tree_fingerprint":"' + ("d" * 64) + '"' in rendered
    assert '"sample_sha256":"' + ("a" * 64) + '"' in rendered
    assert '"evidence_sha256":"' + ("c" * 64) + '"' in rendered
    assert '"canonical_tree"' not in rendered
    assert '"constants_abstracted_canonical_tree"' not in rendered
    assert '"original_expression"' not in rendered
    assert '"source_text"' not in rendered
    assert '"canonical_expression":"x0 + sin(x1)"' not in rendered
    assert '"selected_expression_before_variable_mapping"' not in rendered
    assert '"semantic_expression_after_variable_mapping"' not in rendered
    assert '"formula_candidates"' not in rendered
    assert '"canonical_artifact"' not in rendered
    assert request == request_snapshot


def test_prompt_compaction_keeps_ordinary_requests_compatible() -> None:
    request = {
        "expression": "x0 + 0",
        "metadata": {"notes": ["demo"], "points": [1, 2, 3]},
    }
    compacted = compact_request_for_prompt(request)
    assert compacted == request
    rendered = render_prompt("{{REQUEST_JSON}}", request)
    assert rendered == json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_prompt_compaction_preserves_nonidentical_expression_fields() -> None:
    request = {
        "expression": "x0 + 1",
        "original_expression": "x0 + 1.0",
        "domain_assumptions": {"number_system": "real"},
        "deterministic_evidence": {
            "domain_assumptions": {"number_system": "real"},
            "symbolic_artifact": {
                "source_text": "x0 + 1.0",
                "canonical_expression": "x0 + 1",
                "artifact_sha256": "a" * 64,
                "node_count": 3,
                "function_set": [],
                "operator_set": ["add"],
                "constants_abstracted_tree_fingerprint": "b" * 64,
            },
        },
        "ast_source_evidence": {
            "selected_expression_source": "equation",
            "selected_expression_before_variable_mapping": "x0 + 1.0",
            "semantic_expression_after_variable_mapping": "x0 + 1",
            "result_path": "/tmp/demo/result.json",
        },
    }
    compacted = compact_request_for_prompt(request)
    assert compacted["original_expression"] == "x0 + 1.0"
    assert "domain_assumptions" not in compacted["deterministic_evidence"]
    assert compacted["deterministic_evidence"]["symbolic_artifact"]["source_text"] == "x0 + 1.0"
    assert "canonical_expression" not in compacted["deterministic_evidence"]["symbolic_artifact"]
    assert compacted["ast_source_evidence"] == {
        "selected_expression_source": "equation",
        "result_path": "/tmp/demo/result.json",
    }


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


def test_validate_envelope_accepts_plain_json_single_turn_but_no_real_tools() -> None:
    structured = {
        "decision": "equivalent",
        "evidence_basis": "symbolic_proof",
        "assumptions": [],
        "confidence": 1.0,
        "brief_reason": "Expressions are identical.",
    }
    envelope = valid_envelope(structured)
    schema = json.loads((STAGE / "config/schemas/equivalence.v1.json").read_text())
    assert validate_claude_envelope(
        envelope,
        task_kind="equivalence",
        schema=schema,
    ) == structured


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
    bad_turn_pair["num_turns"] = 2
    with pytest.raises(ContractViolation, match="num_turns"):
        validate_claude_envelope(bad_turn_pair, task_kind="structure")


def test_validate_envelope_accepts_single_json_fence_but_rejects_extra_text() -> None:
    structured = {
        "decision": "different_structure",
        "confidence": 0.8,
        "brief_reason": "Trees differ.",
    }
    schema = json.loads((STAGE / "config/schemas/structure.v1.json").read_text())

    markdown = valid_envelope(structured)
    markdown["result"] = "```json\n" + json.dumps(structured) + "\n```"
    assert validate_claude_envelope(
        markdown,
        task_kind="structure",
        schema=schema,
    ) == structured

    extra_text = valid_envelope(structured)
    extra_text["result"] = "Result:\n```json\n" + json.dumps(structured) + "\n```"
    with pytest.raises(ContractViolation, match="合法 JSON"):
        validate_claude_envelope(extra_text, task_kind="structure", schema=schema)

    missing = valid_envelope(structured)
    missing["result"] = json.dumps({"decision": "different_structure"})
    with pytest.raises(ContractViolation, match="Draft-07"):
        validate_claude_envelope(missing, task_kind="structure", schema=schema)
