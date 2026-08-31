"""Claude Code 单轮调用契约、指纹和严格输出校验。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError, ValidationError


CONTRACT_MODEL = "claude-opus-5[1m]"
CONTRACT_CANONICAL_MODEL = "claude-opus-5"
CONTRACT_EFFORT = "xhigh"
CONTRACT_TRANSPORT_VERSION = "plain_json_prompt_schema.v1"
MAX_LOGICAL_TASKS = 15800
MAX_PHYSICAL_ATTEMPTS = 23700
MAX_ATTEMPTS_PER_TASK = 3


class ContractViolation(ValueError):
    """Claude 请求或响应违反冻结契约。"""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _compact_dataset_probe_for_prompt(
    probe: Mapping[str, Any],
    *,
    probe_points: object,
) -> dict[str, Any]:
    compact_probe = {key: _copy_json_value(value) for key, value in probe.items()}
    if compact_probe.get("points") == probe_points:
        compact_probe.pop("points", None)
    return compact_probe


def _compact_symbolic_artifact_for_prompt(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _copy_json_value(value)
        for key, value in artifact.items()
        if key not in {"canonical_tree", "constants_abstracted_canonical_tree"}
    }


def _top_level_expression_values(request: Mapping[str, object]) -> set[str]:
    expressions: set[str] = set()
    for key, value in request.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if key == "expression" or key.endswith("_expression"):
            expressions.add(value)
    return expressions


def _compact_symbolic_difference_for_prompt(
    difference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: _copy_json_value(value)
        for key, value in difference.items()
        if key not in {"numeric_probes", "skipped_probes"}
    }


def _compact_upstream_binding_for_prompt(
    binding: Mapping[str, Any],
    *,
    expression_values: set[str],
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in binding.items():
        if key == "frozen_simplified_expression" and value in expression_values:
            continue
        compacted[key] = _copy_json_value(value)
    return compacted


def _compact_pair_evidence_for_prompt(
    pair_evidence: Mapping[str, Any],
    *,
    expression_values: set[str],
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in pair_evidence.items():
        if key in {"numeric_probes", "skipped_probes"}:
            continue
        if key in {"lhs_artifact", "rhs_artifact"} and isinstance(value, Mapping):
            compact_artifact = _compact_symbolic_artifact_for_prompt(value)
            if compact_artifact.get("source_text") in expression_values:
                compact_artifact.pop("source_text", None)
            if compact_artifact.get("canonical_expression") in expression_values:
                compact_artifact.pop("canonical_expression", None)
            compacted[key] = compact_artifact
            continue
        if key == "symbolic_difference" and isinstance(value, Mapping):
            compacted[key] = _compact_symbolic_difference_for_prompt(value)
            continue
        compacted[key] = _copy_json_value(value)
    return compacted


def _probe_contract_from_dataset_probe(
    probe: Mapping[str, Any],
) -> tuple[object, str | None, str | None]:
    points = probe.get("points")
    source = probe.get("schema_version")
    sample_sha256 = probe.get("sample_sha256")
    if points is None:
        return None, None, None
    return _copy_json_value(points), str(source) if isinstance(source, str) else None, (
        str(sample_sha256) if isinstance(sample_sha256, str) else None
    )


def _probe_contract_from_request(
    request: Mapping[str, object],
) -> tuple[object, str | None, str | None]:
    if request.get("probe_points") is not None:
        return (
            _copy_json_value(request.get("probe_points")),
            str(request.get("probe_source")) if isinstance(request.get("probe_source"), str) else None,
            (
                str(request.get("probe_sample_sha256"))
                if isinstance(request.get("probe_sample_sha256"), str)
                else None
            ),
        )
    for key in ("deterministic_evidence", "deterministic_pair_evidence"):
        value = request.get(key)
        if not isinstance(value, Mapping):
            continue
        probe = value.get("dataset_probe")
        if isinstance(probe, Mapping) and probe.get("points") is not None:
            return _probe_contract_from_dataset_probe(probe)
    return None, None, None


def _compact_deterministic_evidence_for_prompt(
    evidence: Mapping[str, Any],
    *,
    probe_points: object,
    expression_values: set[str],
    domain_assumptions: object,
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for evidence_key, evidence_value in evidence.items():
        if evidence_key == "dataset_probe" and isinstance(evidence_value, Mapping):
            compacted[evidence_key] = _compact_dataset_probe_for_prompt(
                evidence_value,
                probe_points=probe_points,
            )
            continue
        if evidence_key == "domain_assumptions" and evidence_value == domain_assumptions:
            continue
        if evidence_key == "symbolic_artifact" and isinstance(evidence_value, Mapping):
            compact_artifact = _compact_symbolic_artifact_for_prompt(evidence_value)
            if compact_artifact.get("source_text") in expression_values:
                compact_artifact.pop("source_text", None)
            if compact_artifact.get("canonical_expression") in expression_values:
                compact_artifact.pop("canonical_expression", None)
            compacted[evidence_key] = compact_artifact
            continue
        if evidence_key == "pair_evidence" and isinstance(evidence_value, Mapping):
            compacted[evidence_key] = _compact_pair_evidence_for_prompt(
                evidence_value,
                expression_values=expression_values,
            )
            continue
        if evidence_key in {"lhs_binding", "rhs_binding"} and isinstance(evidence_value, Mapping):
            compacted[evidence_key] = _compact_upstream_binding_for_prompt(
                evidence_value,
                expression_values=expression_values,
            )
            continue
        compacted[evidence_key] = _copy_json_value(evidence_value)
    return compacted


def _compact_ast_source_evidence_for_prompt(
    evidence: Mapping[str, Any],
    *,
    expression_values: set[str],
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, str) and value in expression_values:
            continue
        if key in {
            "canonical_artifact",
            "formula_candidates",
            "extracted_expression_body",
            "selected_expression_before_variable_mapping",
            "semantic_expression_after_variable_mapping",
        }:
            continue
        compacted[key] = _copy_json_value(value)
    return compacted


def compact_request_for_prompt(request: Mapping[str, object]) -> dict[str, Any]:
    """生成仅用于 prompt 传输的稳定投影，不修改原始 request。"""

    expression_values = _top_level_expression_values(request)
    probe_points, probe_source, probe_sample_sha256 = _probe_contract_from_request(request)
    expression = request.get("expression")
    domain_assumptions = request.get("domain_assumptions")
    duplicate_pair_evidence = (
        isinstance(request.get("deterministic_evidence"), Mapping)
        and request.get("deterministic_evidence") == request.get("deterministic_pair_evidence")
    )
    projected: dict[str, Any] = {}
    for key, value in request.items():
        if key == "original_expression" and value == expression:
            continue
        if key == "deterministic_pair_evidence" and duplicate_pair_evidence:
            projected["deterministic_pair_evidence_alias"] = "deterministic_evidence"
            continue
        if key == "dataset_probe_evidence" and isinstance(value, Mapping):
            projected[key] = _compact_dataset_probe_for_prompt(
                value,
                probe_points=probe_points,
            )
            continue
        if key == "ast_source_evidence" and isinstance(value, Mapping):
            projected[key] = _compact_ast_source_evidence_for_prompt(
                value,
                expression_values=expression_values,
            )
            continue
        if key in {"deterministic_evidence", "deterministic_pair_evidence"} and isinstance(value, Mapping):
            projected[key] = _compact_deterministic_evidence_for_prompt(
                value,
                probe_points=probe_points,
                expression_values=expression_values,
                domain_assumptions=domain_assumptions,
            )
            continue
        projected[key] = _copy_json_value(value)
    if "probe_points" not in projected and probe_points is not None:
        projected["probe_points"] = probe_points
    if "probe_source" not in projected and probe_source is not None:
        projected["probe_source"] = probe_source
    if "probe_sample_sha256" not in projected and probe_sample_sha256 is not None:
        projected["probe_sample_sha256"] = probe_sample_sha256
    return projected


def evaluation_key(
    *,
    task_type: str,
    logical_id: str,
    prompt_version: str,
    schema_version: str,
    prompt_sha256: str,
    schema_sha256: str,
    normalized_input: Mapping[str, object],
    evidence_hash: str,
    model: str = CONTRACT_MODEL,
    effort: str = CONTRACT_EFFORT,
) -> str:
    """为逻辑任务生成包含全部评测契约的稳定指纹。"""

    for field_name, value in (
        ("prompt_sha256", prompt_sha256),
        ("schema_sha256", schema_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ContractViolation(f"{field_name} 必须是小写十六进制 SHA-256")

    payload = {
        "task_type": task_type,
        "logical_id": logical_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model": model,
        "effort": effort,
        "normalized_input": normalized_input,
        "deterministic_evidence_hash": evidence_hash,
    }
    return sha256_json(payload)


def render_prompt(
    template: str,
    request: Mapping[str, object],
    schema: Mapping[str, object] | None = None,
) -> str:
    """渲染请求，并把输出 schema 作为单轮 prompt 的显式组成部分。"""

    placeholder = "{{REQUEST_JSON}}"
    if template.count(placeholder) != 1:
        raise ContractViolation("prompt 模板必须恰好包含一个 {{REQUEST_JSON}} 占位符")
    rendered = template.replace(placeholder, canonical_json(compact_request_for_prompt(request)))
    if schema is None:
        return rendered
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise ContractViolation(f"输出 schema 不是合法 Draft-07: {exc.message}") from exc
    return (
        rendered.rstrip()
        + "\n\nOUTPUT_JSON_SCHEMA_DRAFT_07:\n"
        + canonical_json(schema)
        + "\n"
    )


def build_claude_command(schema: Mapping[str, object]) -> list[str]:
    """构造固定的一次性 Claude Code 命令；schema 由 stdin prompt 提供。"""

    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise ContractViolation(f"输出 schema 不是合法 Draft-07: {exc.message}") from exc

    return [
        "claude",
        "--print",
        "--safe-mode",
        "--setting-sources",
        "user",
        "--model",
        CONTRACT_MODEL,
        "--effort",
        CONTRACT_EFFORT,
        "--tools",
        "",
        "--max-turns",
        "1",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--output-format",
        "json",
    ]


def _require_exact_keys(output: Mapping[str, object], expected: set[str]) -> None:
    actual = set(output)
    if actual != expected:
        raise ContractViolation(
            f"结构化输出字段不匹配，缺失={sorted(expected-actual)}，额外={sorted(actual-expected)}"
        )


def _parse_single_json_result(result_text: str) -> Mapping[str, object]:
    """解析原始 JSON，或仅由单个 json 代码围栏包裹的 JSON。"""

    stripped = result_text.strip()
    fenced = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", stripped)
    candidate = fenced.group(1) if fenced is not None else stripped
    try:
        structured = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"Claude result 不是合法 JSON: {exc}") from exc
    if not isinstance(structured, Mapping):
        raise ContractViolation("Claude result 必须是 JSON object")
    return structured


def _require_confidence(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation("confidence 必须是数值")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ContractViolation("confidence 必须是 [0, 1] 内的有限数值")


def _require_reason(value: object) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ContractViolation("brief_reason 必须是 1--1000 字符的非空字符串")


def _require_assumptions(value: object) -> None:
    if not isinstance(value, list) or len(value) > 8:
        raise ContractViolation("assumptions 必须是最多 8 项的数组")
    if not all(isinstance(item, str) for item in value):
        raise ContractViolation("assumptions 的每一项都必须是字符串")


def validate_structured_output(
    task_kind: str,
    output: Mapping[str, object],
) -> dict[str, object]:
    """执行独立于 Claude CLI 的第二层严格校验。"""

    if not isinstance(output, Mapping):
        raise ContractViolation("structured_output 必须是 JSON object")
    normalized_kind = "simplify" if task_kind.endswith("simplify") else task_kind
    if normalized_kind == "simplify":
        expected = {
            "outcome",
            "simplified_expression",
            "equivalence_assessment",
            "assumptions",
            "confidence",
            "brief_reason",
        }
        _require_exact_keys(output, expected)
        outcome = output["outcome"]
        if outcome not in {"simplified", "unchanged", "unable"}:
            raise ContractViolation(f"未知 simplify outcome: {outcome!r}")
        expression = output["simplified_expression"]
        if outcome in {"simplified", "unchanged"}:
            if not isinstance(expression, str) or not expression.strip():
                raise ContractViolation("成功化简必须返回非空 simplified_expression")
        elif expression is not None:
            raise ContractViolation("outcome=unable 时 simplified_expression 必须为 null")
        assessment = output["equivalence_assessment"]
        if assessment not in {"preserved", "not_preserved", "undetermined"}:
            raise ContractViolation(f"未知 equivalence_assessment: {assessment!r}")
        if outcome in {"simplified", "unchanged"} and assessment != "preserved":
            raise ContractViolation("可接受的化简结果必须声明 equivalence_assessment=preserved")
        if outcome == "unable" and assessment != "undetermined":
            raise ContractViolation("outcome=unable 时 equivalence_assessment 必须为 undetermined")
        _require_assumptions(output["assumptions"])
    elif normalized_kind == "equivalence":
        expected = {
            "decision",
            "evidence_basis",
            "assumptions",
            "confidence",
            "brief_reason",
        }
        _require_exact_keys(output, expected)
        if output["decision"] not in {"equivalent", "not_equivalent", "undetermined"}:
            raise ContractViolation(f"未知 equivalence decision: {output['decision']!r}")
        evidence_basis = output["evidence_basis"]
        if evidence_basis not in {
            "symbolic_proof",
            "numerical_support",
            "structural_analysis",
            "mixed",
            "insufficient",
        }:
            raise ContractViolation(f"未知 evidence_basis: {output['evidence_basis']!r}")
        if output["decision"] in {"equivalent", "not_equivalent"} and evidence_basis == "insufficient":
            raise ContractViolation("确定的 equivalence decision 不能使用 insufficient 证据")
        _require_assumptions(output["assumptions"])
    elif normalized_kind == "structure":
        expected = {"decision", "confidence", "brief_reason"}
        _require_exact_keys(output, expected)
        if output["decision"] not in {
            "mathematically_equivalent",
            "same_canonical_structure",
            "different_structure",
            "undetermined",
        }:
            raise ContractViolation(f"未知 structure decision: {output['decision']!r}")
    else:
        raise ContractViolation(f"未知 task_kind: {task_kind!r}")
    _require_confidence(output["confidence"])
    _require_reason(output["brief_reason"])
    return dict(output)


def validate_claude_envelope(
    envelope: Mapping[str, object],
    *,
    task_kind: str,
    schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """校验单轮纯 JSON envelope、模型、零工具和本地输出 schema。"""

    if not isinstance(envelope, Mapping):
        raise ContractViolation("Claude 外层输出不是 JSON object")
    if envelope.get("type") != "result" or envelope.get("subtype") != "success":
        raise ContractViolation("Claude 外层输出不是成功 result")
    if envelope.get("is_error") is not False or envelope.get("terminal_reason") != "completed":
        raise ContractViolation("Claude 调用未正常完成")
    turns = envelope.get("num_turns")
    if turns != 1:
        raise ContractViolation(f"单轮调用出现异常 num_turns={turns!r}")
    stop_reason = envelope.get("stop_reason")
    if stop_reason != "end_turn":
        raise ContractViolation(
            f"单轮 stop_reason 不匹配: {stop_reason!r}，期望 'end_turn'"
        )

    model_usage = envelope.get("modelUsage")
    if not isinstance(model_usage, Mapping) or set(model_usage) != {CONTRACT_MODEL}:
        raise ContractViolation(f"Claude 模型契约不符: {list(model_usage or {})!r}")
    model_record = model_usage[CONTRACT_MODEL]
    if not isinstance(model_record, Mapping):
        raise ContractViolation("Claude modelUsage 记录无效")
    if model_record.get("canonicalModel") != CONTRACT_CANONICAL_MODEL:
        raise ContractViolation(
            f"Claude 实际模型不符: {model_record.get('canonicalModel')!r}"
        )

    permission_denials = envelope.get("permission_denials")
    if permission_denials != []:
        raise ContractViolation("Claude 尝试调用了被禁用的工具或权限")
    usage = envelope.get("usage")
    if not isinstance(usage, Mapping) or not isinstance(usage.get("server_tool_use"), Mapping):
        raise ContractViolation("Claude 响应缺少可审计的 server_tool_use")
    server_tools: Mapping[str, object] = usage["server_tool_use"]  # type: ignore[assignment]
    try:
        used_server_tool = any(int(value or 0) != 0 for value in server_tools.values())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractViolation("Claude server_tool_use 统计不是合法整数") from exc
    if used_server_tool:
        raise ContractViolation("Claude 使用了被禁止的服务器工具")
    subagents = envelope.get("subagent_stats")
    if not isinstance(subagents, Mapping) or "spawned" not in subagents:
        raise ContractViolation("Claude 响应缺少可审计的 subagent_stats")
    if int(subagents.get("spawned") or 0) != 0:
        raise ContractViolation("Claude 使用了被禁止的 subagent 工具")

    result_text = envelope.get("result")
    if not isinstance(result_text, str) or not result_text.strip():
        raise ContractViolation("Claude 输出缺失纯 JSON result 文本")
    structured = _parse_single_json_result(result_text)
    if schema is not None:
        try:
            Draft7Validator.check_schema(schema)
            Draft7Validator(schema).validate(structured)
        except SchemaError as exc:
            raise ContractViolation(f"输出 schema 不是合法 Draft-07: {exc.message}") from exc
        except ValidationError as exc:
            raise ContractViolation(f"Claude result 未通过 Draft-07 schema: {exc.message}") from exc
    return validate_structured_output(task_kind, structured)
