"""Stage5 simplify 语义验证 worker。

通过独立 Python 子进程执行 SymPy 验证，避免主 worker 被本地符号计算长期占用。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SimplificationContractError,
    SymbolicEvidenceError,
    validate_simplification,
)


JsonDict = dict[str, object]
WORKER_MODULE = (
    "AAAI_experiments.stage5_metric_calculation_0831.pipeline.semantic_validation_worker"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _semantic_seed(evaluation_key: str) -> int:
    digest = _sha256_text(evaluation_key)
    return int(digest[:16], 16)


def _validated_probe_contract(
    request: Mapping[str, object],
) -> tuple[Sequence[Mapping[str, object]] | None, str | None, str | None]:
    present = {
        key
        for key in (
            "probe_points",
            "probe_source",
            "probe_sample_sha256",
            "dataset_probe_evidence",
        )
        if request.get(key) is not None
    }
    if not present:
        return None, None, None
    required = {
        "probe_points",
        "probe_source",
        "probe_sample_sha256",
        "dataset_probe_evidence",
    }
    if present != required:
        raise SymbolicEvidenceError(
            f"dataset probe 请求字段不完整: 缺少 {sorted(required - present)}"
        )
    points = request.get("probe_points")
    source = request.get("probe_source")
    sample_sha256 = request.get("probe_sample_sha256")
    probe_raw = request.get("dataset_probe_evidence")
    if not isinstance(points, list) or not all(isinstance(item, Mapping) for item in points):
        raise SymbolicEvidenceError("probe_points 必须是对象数组")
    if not isinstance(source, str) or not source:
        raise SymbolicEvidenceError("probe_source 必须是非空字符串")
    if not isinstance(sample_sha256, str) or _SHA256_RE.fullmatch(sample_sha256) is None:
        raise SymbolicEvidenceError("probe_sample_sha256 非法")
    if not isinstance(probe_raw, Mapping):
        raise SymbolicEvidenceError("dataset_probe_evidence 必须是对象")
    probe = dict(probe_raw)
    if probe.get("schema_version") != source:
        raise SymbolicEvidenceError("probe_source 与 dataset_probe_evidence 不一致")
    if probe.get("points") != points:
        raise SymbolicEvidenceError("probe_points 与 dataset_probe_evidence 不一致")
    if probe.get("sample_sha256") != sample_sha256:
        raise SymbolicEvidenceError("probe_sample_sha256 与 dataset_probe_evidence 不一致")
    variables = request.get("variables")
    if probe.get("variables") != variables:
        raise SymbolicEvidenceError("probe variables 与 simplify request.variables 不一致")
    dataset_id = request.get("dataset_id")
    if probe.get("dataset_name") != dataset_id:
        raise SymbolicEvidenceError("probe dataset_name 与 simplify request.dataset_id 不一致")
    sample_payload = {
        "schema_version": probe.get("schema_version"),
        "dataset_name": probe.get("dataset_name"),
        "variables": probe.get("variables"),
        "points": probe.get("points"),
    }
    if _sha256_text(canonical_json(sample_payload)) != sample_sha256:
        raise SymbolicEvidenceError("dataset probe sample_sha256 校验失败")
    evidence_sha256 = probe.get("evidence_sha256")
    evidence_payload = {
        key: value for key, value in probe.items() if key != "evidence_sha256"
    }
    if not isinstance(evidence_sha256, str) or _sha256_text(
        canonical_json(evidence_payload)
    ) != evidence_sha256:
        raise SymbolicEvidenceError("dataset probe evidence_sha256 校验失败")
    return points, source, sample_sha256


def validate_simplify_semantics_payload(
    *,
    evaluation_key: str,
    request: Mapping[str, object],
    structured_output: Mapping[str, object],
) -> JsonDict:
    outcome = structured_output.get("outcome")
    if outcome == "unable":
        return {"decision": "not_applicable", "reason": "outcome_unable"}

    original = request.get("expression")
    simplified = structured_output.get("simplified_expression")
    variables = request.get("variables")
    functions = request.get("allowed_functions")
    if not isinstance(original, str) or not original.strip():
        raise SymbolicEvidenceError("simplify request.expression 缺失或无效")
    if not isinstance(simplified, str) or not simplified.strip():
        raise SymbolicEvidenceError("simplified_expression 缺失或无效")
    if not isinstance(variables, list) or not all(
        isinstance(item, str) and item for item in variables
    ):
        raise SymbolicEvidenceError("simplify request.variables 缺失或无效")
    if not isinstance(functions, list) or not all(
        isinstance(item, str) and item for item in functions
    ):
        raise SymbolicEvidenceError("simplify request.allowed_functions 缺失或无效")
    probe_points, probe_source, probe_sample_sha256 = _validated_probe_contract(request)
    evidence = validate_simplification(
        original=original,
        simplified=simplified,
        allowed_variables=variables,
        allowed_functions=functions,
        seed=_semantic_seed(evaluation_key),
        probe_points=probe_points,
        probe_source=probe_source,
        probe_sample_sha256=probe_sample_sha256,
    )
    deterministic = request.get("deterministic_evidence")
    if deterministic is not None:
        if not isinstance(deterministic, Mapping):
            raise SymbolicEvidenceError("deterministic_evidence 必须是对象")
        artifact = deterministic.get("symbolic_artifact")
        if not isinstance(artifact, Mapping):
            raise SymbolicEvidenceError("deterministic_evidence.symbolic_artifact 缺失")
        if artifact.get("artifact_sha256") != evidence.get("original_sha256"):
            raise SymbolicEvidenceError("请求中的 symbolic artifact 与原公式不一致")
        if list(artifact.get("variables", [])) != sorted(
            set(str(item) for item in artifact.get("variables", []))
        ):
            raise SymbolicEvidenceError("请求中的 symbolic artifact variables 未规范化")
    return evidence


def build_semantic_validation_payload(
    *,
    evaluation_key: str,
    request: Mapping[str, object],
    structured_output: Mapping[str, object],
) -> JsonDict:
    return {
        "evaluation_key": evaluation_key,
        "request": dict(request),
        "structured_output": dict(structured_output),
    }


def run_worker_request(payload: Mapping[str, object]) -> JsonDict:
    evaluation_key = payload.get("evaluation_key")
    request = payload.get("request")
    structured_output = payload.get("structured_output")
    if not isinstance(evaluation_key, str) or not evaluation_key:
        raise ValueError("evaluation_key 缺失或无效")
    if not isinstance(request, Mapping):
        raise ValueError("request 缺失或无效")
    if not isinstance(structured_output, Mapping):
        raise ValueError("structured_output 缺失或无效")
    try:
        semantic_evidence = validate_simplify_semantics_payload(
            evaluation_key=evaluation_key,
            request=request,
            structured_output=structured_output,
        )
    except SimplificationContractError as exc:
        return {
            "status": "simplification_contract_error",
            "error_message": str(exc),
            "semantic_evidence": dict(exc.evidence),
        }
    except SymbolicEvidenceError as exc:
        return {
            "status": "symbolic_evidence_error",
            "error_message": str(exc),
            "semantic_evidence": {
                "decision": "contract_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        }
    return {"status": "ok", "semantic_evidence": semantic_evidence}


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"semantic validation worker JSON decode failed: {exc}\n")
        return 2
    if not isinstance(payload, dict):
        sys.stderr.write("semantic validation worker payload 必须是 JSON object\n")
        return 2
    response = run_worker_request(payload)
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
