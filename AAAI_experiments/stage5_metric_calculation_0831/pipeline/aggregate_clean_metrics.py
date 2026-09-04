"""聚合 clean 条件下的离线六轴指标。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sympy

from .metrics import RunQuality, minimality_score, stability_score, symbolic_fidelity_score
from .metrics import efficiency_from_qualities
from .frozen_result_index import (
    LLM_SIMPLIFIED_EXPRESSION,
    ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE,
)
from .run_claude_plan import PlanContractError, load_plan_jsonl
from .symbolic_evidence import (
    SymbolicEvidenceError,
    build_symbolic_artifact,
    operator_f1 as operator_f1_metric,
    tree_similarity as tree_similarity_metric,
    variable_f1 as variable_f1_metric,
)


SEEDS = (520, 521, 522)
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))
NOISE_TAG = "clean"
EFF_HORIZON = 180
EFF_QUALITY_FIELDS = tuple(f"q_{index:04d}" for index in range(1, EFF_HORIZON + 1))
DEFAULT_EXPECTED_RUNS = 2250
DEFAULT_EXPECTED_ALGORITHMS = 15
DEFAULT_EXPECTED_DATASETS = 50
AUDIT_CORRECTIONS_SCHEMA_VERSION = "audit_corrections.v1"
AUDIT_FALLBACK_RESOLUTION = "original_identity_fallback_after_audit"
MINIMUM_FORMAL_AUDIT_ROWS = 1000
CANONICAL_REPLAY_EVALUATION_PATH = "canonical_replay.v1"
SIMPLIFY_OUTCOMES = {"simplified", "unchanged", "unable"}
SIMPLIFY_EQUIVALENCE_ASSESSMENTS = {"preserved", "not_preserved", "undetermined"}
GT_NO_CALL_REASONS = {"missing_ground_truth_expression"}
PRED_NO_CALL_REASONS = {"missing_final_expression", "unresolved_parameter_values"}
EQUIVALENCE_DECISIONS = {"equivalent", "not_equivalent", "undetermined"}
EQUIVALENCE_EVIDENCE_BASES = {
    "symbolic_proof",
    "numerical_support",
    "structural_analysis",
    "mixed",
    "insufficient",
}
EQUIVALENCE_NO_CALL_REASONS = {"upstream_gt_unavailable", "upstream_pred_unavailable"}
STRUCTURE_DECISIONS = {
    "mathematically_equivalent",
    "same_canonical_structure",
    "different_structure",
    "undetermined",
}
STRUCTURE_NO_CALL_REASONS = {"invalid_seed_or_expression"}
GT_LOGICAL_ID_RE = re.compile(r"^gt_simplify::([^:]+)(?:::(v2))?$")
RUN_LOGICAL_ID_RE_TEMPLATE = r"^{prefix}::([a-z0-9_]+)::(g\d{{4}})::s(520|521|522)::clean{suffix}$"
STRUCTURE_LOGICAL_ID_RE = re.compile(
    r"^stab_structure::([a-z0-9_]+)::(g\d{4})::s(520|521|522)-s?(520|521|522)"
    r"(?:::v[1-9]\d*)?$"
)
NUMERIC_LOGICAL_KEY_RE = re.compile(r"^([^:]+)::([^:]+)::s(520|521|522)::clean$")


class AggregateCleanMetricsError(ValueError):
    """clean 聚合输入或闭环契约不满足正式要求。"""


def _is_hex_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage5_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _raise(message: str) -> None:
    raise AggregateCleanMetricsError(message)


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AggregateCleanMetricsError(f"{context} 不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        _raise(f"{context} 顶层必须是 JSON object: {path}")
    return payload


def _parse_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    _raise(f"{context} 必须是 true/false，实际为 {value!r}")
    raise AssertionError("unreachable")


def _parse_int(value: object, *, context: str) -> int:
    if value is None or isinstance(value, bool):
        _raise(f"{context} 缺失或不是整数")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AggregateCleanMetricsError(f"{context} 不是合法整数: {value!r}") from exc


def _finite_unit_float(value: object, *, context: str) -> float:
    if value is None or isinstance(value, bool):
        _raise(f"{context} 缺失或不是数值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AggregateCleanMetricsError(f"{context} 不是合法数值: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        _raise(f"{context} 必须在 [0, 1] 内，实际为 {value!r}")
    return number


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        _raise(f"{context} 必须是非空字符串")
    return value


def _sha256_string(value: object, *, context: str) -> str:
    text = _string(value, context=context)
    if not _is_hex_sha256(text):
        _raise(f"{context} 必须是 64 位 SHA256 十六进制字符串")
    return text


def _optional_nonempty_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_report_path(raw_path: object, *, context: str) -> Path:
    text = _string(raw_path, context=context)
    path = Path(text)
    return path if path.is_absolute() else (_repo_root() / path).resolve()


def _parse_gt_logical_id(logical_id: str) -> str:
    match = GT_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        _raise(f"无法解析 GT logical_id: {logical_id!r}")
    return match.group(1)


def _parse_run_logical_id(logical_id: str, *, expected_prefix: str) -> tuple[str, str, int]:
    suffix = (
        r"(?:::v[1-9]\d*)?"
        if expected_prefix in {"pred_simplify", "equivalence"}
        else ""
    )
    pattern = re.compile(
        RUN_LOGICAL_ID_RE_TEMPLATE.format(
            prefix=re.escape(expected_prefix),
            suffix=suffix,
        )
    )
    match = pattern.fullmatch(logical_id)
    if match is None:
        _raise(f"无法解析 {expected_prefix} logical_id: {logical_id!r}")
    algorithm = match.group(1)
    dataset_id = match.group(2)
    seed = int(match.group(3))
    return algorithm, dataset_id, seed


def _parse_structure_logical_id(logical_id: str) -> tuple[str, str, tuple[int, int]]:
    match = STRUCTURE_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        _raise(f"无法解析 structure logical_id: {logical_id!r}")
    algorithm = match.group(1)
    dataset_id = match.group(2)
    left = int(match.group(3))
    right = int(match.group(4))
    pair = (min(left, right), max(left, right))
    if pair not in SEED_PAIRS:
        _raise(f"{logical_id!r} 的 seed pair 不在允许集合中")
    return algorithm, dataset_id, pair


def _parse_numeric_logical_key(logical_key: str) -> tuple[str, str, int]:
    match = NUMERIC_LOGICAL_KEY_RE.fullmatch(logical_key)
    if match is None:
        _raise(f"无法解析 numeric logical_key: {logical_key!r}")
    return match.group(1), match.group(2), int(match.group(3))


def _logical_key(algorithm: str, dataset_id: str, seed: int) -> str:
    return f"{algorithm}::{dataset_id}::s{seed}::{NOISE_TAG}"


def _validate_non_applicable(
    row: Mapping[str, Any],
    *,
    logical_id: str,
    allowed_reasons: set[str] | None = None,
) -> dict[str, Any]:
    structured_output = row.get("structured_output")
    if structured_output is not None:
        _raise(f"{logical_id} state=non_applicable 时 structured_output 必须为 null")
    payload = row.get("non_applicable")
    if not isinstance(payload, Mapping):
        _raise(f"{logical_id}.non_applicable 缺失")
    reason = _string(payload.get("reason"), context=f"{logical_id}.non_applicable.reason")
    if allowed_reasons is not None and reason not in allowed_reasons:
        _raise(f"{logical_id}.non_applicable.reason 非法: {reason!r}")
    evidence_sha256 = _sha256_string(
        payload.get("evidence_sha256"),
        context=f"{logical_id}.non_applicable.evidence_sha256",
    )
    evidence_path = _string(
        payload.get("evidence_path"),
        context=f"{logical_id}.non_applicable.evidence_path",
    )
    return {
        "reason": reason,
        "evidence_sha256": evidence_sha256,
        "evidence_path": evidence_path,
    }


def _validate_simplify_structured_output(
    row: Mapping[str, Any],
    *,
    logical_id: str,
    plan_request: Mapping[str, Any] | None,
    allow_missing: bool,
) -> tuple[str | None, str]:
    structured_output = row.get("structured_output")
    if not isinstance(structured_output, Mapping):
        _raise(f"{logical_id}.structured_output 缺失")
    outcome = _string(structured_output.get("outcome"), context=f"{logical_id}.outcome")
    if outcome not in SIMPLIFY_OUTCOMES:
        _raise(f"{logical_id}.outcome 非法: {outcome!r}")
    assessment = _string(
        structured_output.get("equivalence_assessment"),
        context=f"{logical_id}.equivalence_assessment",
    )
    if assessment not in SIMPLIFY_EQUIVALENCE_ASSESSMENTS:
        _raise(f"{logical_id}.equivalence_assessment 非法: {assessment!r}")
    simplified = _optional_nonempty_string(structured_output.get("simplified_expression"))
    if outcome in {"simplified", "unchanged"}:
        if assessment != "preserved":
            _raise(f"{logical_id} outcome={outcome} 时 equivalence_assessment 必须为 preserved")
        if simplified is None:
            _raise(f"{logical_id} 缺少可用 simplified_expression")
        effective_expression = _string(
            row.get("effective_expression"),
            context=f"{logical_id}.effective_expression",
        ).strip()
        expression_resolution = _string(
            row.get("expression_resolution"),
            context=f"{logical_id}.expression_resolution",
        )
        if expression_resolution != LLM_SIMPLIFIED_EXPRESSION:
            _raise(
                f"{logical_id}.expression_resolution 必须为 {LLM_SIMPLIFIED_EXPRESSION}"
            )
        if effective_expression != simplified:
            _raise(f"{logical_id}.effective_expression 必须与 simplified_expression 一致")
        return effective_expression, "frozen"
    if assessment != "undetermined":
        _raise(f"{logical_id} outcome=unable 时 equivalence_assessment 必须为 undetermined")
    if structured_output.get("simplified_expression") is not None:
        _raise(f"{logical_id} outcome=unable 时 simplified_expression 必须为 null")
    effective_expression = _string(
        row.get("effective_expression"),
        context=f"{logical_id}.effective_expression",
    )
    expression_resolution = _string(
        row.get("expression_resolution"),
        context=f"{logical_id}.expression_resolution",
    )
    if expression_resolution != ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE:
        _raise(
            f"{logical_id}.expression_resolution 必须为 "
            f"{ORIGINAL_IDENTITY_FALLBACK_AFTER_LLM_UNABLE}"
        )
    if plan_request is None:
        _raise(f"{logical_id} 缺少上游 plan request，无法校验 unable fallback 来源")
    original_expression = _string(
        plan_request.get("original_expression"),
        context=f"{logical_id}.plan_request.original_expression",
    )
    if effective_expression != original_expression:
        _raise(f"{logical_id}.effective_expression 必须回退到 plan request.original_expression")
    return effective_expression, "frozen"


def _resolve_pred_identity(
    identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
    *,
    algorithm_slug: str,
    dataset_index: str,
    seed: int,
    logical_id: str,
) -> dict[str, Any]:
    identity = identity_map.get((algorithm_slug, dataset_index, seed))
    if identity is None:
        _raise(f"{logical_id} 在 pred simplify plan 中缺少身份映射")
    return dict(identity)


def _read_csv_by_logical_key(
    path: Path,
    *,
    required_fields: set[str],
    label: str,
    expected_runs: int,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    rows: dict[str, dict[str, str]] = {}
    algorithms: set[str] = set()
    datasets: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_fields.difference(reader.fieldnames or ()))
        if missing:
            _raise(f"{label} 缺少字段: {missing}")
        for raw in reader:
            row = dict(raw)
            key = _string(row.get("logical_key"), context=f"{label}.logical_key")
            if key in rows:
                _raise(f"{label} 出现重复 logical_key: {key}")
            algorithm = _string(row.get("algorithm"), context=f"{key}.algorithm")
            dataset_id = _string(row.get("dataset_id"), context=f"{key}.dataset_id")
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError) as exc:
                raise AggregateCleanMetricsError(f"{key}.seed 非法") from exc
            expected_key = _logical_key(algorithm, dataset_id, seed)
            if key != expected_key:
                _raise(f"{label} logical_key 不一致: {key!r} != {expected_key!r}")
            if row.get("noise_tag") != NOISE_TAG:
                _raise(f"{key} 的 noise_tag 不是 clean")
            rows[key] = row
            algorithms.add(algorithm)
            datasets.add(dataset_id)
    if len(rows) != expected_runs:
        _raise(f"{label} 行数应为 {expected_runs}，实际为 {len(rows)}")
    return rows, {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "algorithm_count": len(algorithms),
        "dataset_count": len(datasets),
    }


def _iter_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AggregateCleanMetricsError(
                    f"{label} 第 {line_number} 行不是合法 JSON: {path}"
                ) from exc
            if not isinstance(payload, dict):
                _raise(f"{label} 第 {line_number} 行顶层必须是 object")
            rows.append(payload)
    return rows


def _validate_audit_corrections_requirement(
    manifest_path: Path | None,
    *,
    expected_runs: int,
    expected_algorithms: int,
    expected_datasets: int,
) -> Path | None:
    formal_contract = (
        expected_runs == DEFAULT_EXPECTED_RUNS
        and expected_algorithms == DEFAULT_EXPECTED_ALGORITHMS
        and expected_datasets == DEFAULT_EXPECTED_DATASETS
    )
    if manifest_path is None:
        if formal_contract:
            _raise(
                "正式 2250-run 聚合必须提供 audit corrections manifest，"
                "禁止绕过公式抽查修正层"
            )
        return None
    return manifest_path.resolve()


def _load_plan_with_contract(path: Path, *, label: str) -> Any:
    try:
        return load_plan_jsonl(path)
    except PlanContractError as exc:
        raise AggregateCleanMetricsError(f"{label} 契约失败: {exc}") from exc


def _validate_state_counts(
    value: object,
    *,
    context: str,
    allow_exhausted: bool = False,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        _raise(f"{context} 缺失")
    frozen = _parse_int(value.get("frozen"), context=f"{context}.frozen")
    non_applicable = _parse_int(
        value.get("non_applicable"),
        context=f"{context}.non_applicable",
    )
    state_counts = {"frozen": frozen, "non_applicable": non_applicable}
    allowed_fields = {"frozen", "non_applicable"}
    if allow_exhausted:
        state_counts["exhausted"] = _parse_int(
            value.get("exhausted"),
            context=f"{context}.exhausted",
        )
        allowed_fields.add("exhausted")
    extras = sorted(set(value) - allowed_fields)
    if extras:
        _raise(f"{context} 存在额外状态字段: {extras}")
    return state_counts


def _validate_frozen_summary(
    summary_path: Path,
    *,
    index_path: Path,
    plan: Any,
    label: str,
    allow_exhausted: bool = False,
) -> dict[str, Any]:
    payload = _read_json_object(summary_path, context=f"{label} summary")
    status = _string(payload.get("status"), context=f"{label}.summary.status")
    if status != "ok":
        _raise(f"{label}.summary.status 必须为 'ok'")
    output_jsonl = _resolve_report_path(
        payload.get("output_jsonl"),
        context=f"{label}.summary.output_jsonl",
    )
    if output_jsonl != index_path.resolve():
        _raise(f"{label}.summary.output_jsonl 与 frozen index 不一致")
    output_sha256 = _sha256_string(
        payload.get("output_sha256"),
        context=f"{label}.summary.output_sha256",
    )
    if output_sha256 != _sha256_file(index_path):
        _raise(f"{label}.summary.output_sha256 与 frozen index 文件不一致")
    plan_jsonl = _resolve_report_path(
        payload.get("plan_jsonl"),
        context=f"{label}.summary.plan_jsonl",
    )
    if plan_jsonl != plan.plan_path.resolve():
        _raise(f"{label}.summary.plan_jsonl 与 plan 文件不一致")
    plan_sha256 = _sha256_string(
        payload.get("plan_sha256"),
        context=f"{label}.summary.plan_sha256",
    )
    if plan_sha256 != plan.plan_sha256:
        _raise(f"{label}.summary.plan_sha256 与 plan 文件不一致")
    row_count = _parse_int(payload.get("row_count"), context=f"{label}.summary.row_count")
    state_counts = _validate_state_counts(
        payload.get("state_counts"),
        context=f"{label}.summary.state_counts",
        allow_exhausted=allow_exhausted,
    )
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha256_file(index_path),
        "plan_path": str(plan.plan_path.resolve()),
        "plan_sha256": plan.plan_sha256,
        "plan_row_count": len(plan.entries),
        "row_count": row_count,
        "state_counts": state_counts,
        "status": status,
    }


def _validate_frozen_result_payload(
    result_path: Path,
    *,
    logical_id: str,
    evaluation_key: str,
    task_type: str,
    plan_sha256: str,
    structured_output: Mapping[str, Any],
    label: str,
) -> None:
    payload = _read_json_object(result_path, context=f"{label} result")
    if payload.get("logical_id") != logical_id:
        _raise(f"{label} result.logical_id 与 index 不一致: {logical_id}")
    if payload.get("evaluation_key") != evaluation_key:
        _raise(f"{label} result.evaluation_key 与 index 不一致: {logical_id}")
    if payload.get("task_type") != task_type:
        _raise(f"{label} result.task_type 与 index 不一致: {logical_id}")
    payload_plan_sha256 = payload.get("plan_sha256")
    if payload_plan_sha256 is not None and _sha256_string(
        payload_plan_sha256,
        context=f"{label} result.plan_sha256",
    ) != plan_sha256:
        _raise(f"{label} result.plan_sha256 与 plan 不一致: {logical_id}")
    payload_structured_output = payload.get("structured_output")
    if not isinstance(payload_structured_output, Mapping):
        _raise(f"{label} result.structured_output 缺失: {logical_id}")
    if _canonical_json(dict(payload_structured_output)) != _canonical_json(dict(structured_output)):
        _raise(f"{label} result.structured_output 与 index 不一致: {logical_id}")


def _validate_non_applicable_evidence_payload(
    evidence_path: Path,
    *,
    plan_entry: Any,
    logical_id: str,
    task_type: str,
    reason: str,
    evidence_sha256: str,
    plan_sha256: str,
    label: str,
) -> None:
    payload = _read_json_object(evidence_path, context=f"{label} evidence")
    if payload.get("logical_id") != logical_id:
        _raise(f"{label} evidence.logical_id 与 index 不一致: {logical_id}")
    if payload.get("task_type") != task_type:
        _raise(f"{label} evidence.task_type 与 index 不一致: {logical_id}")
    if payload.get("condition") != plan_entry.definition.task_spec.condition:
        _raise(f"{label} evidence.condition 与 plan 不一致: {logical_id}")
    if payload.get("reason") != reason:
        _raise(f"{label} evidence.reason 与 index 不一致: {logical_id}")
    payload_evaluation_key = payload.get("evaluation_key")
    if payload_evaluation_key is not None and payload_evaluation_key != plan_entry.evaluation_key:
        _raise(f"{label} evidence.evaluation_key 与 plan 不一致: {logical_id}")
    payload_plan_sha256 = payload.get("plan_sha256")
    if payload_plan_sha256 is not None and _sha256_string(
        payload_plan_sha256,
        context=f"{label} evidence.plan_sha256",
    ) != plan_sha256:
        _raise(f"{label} evidence.plan_sha256 与 plan 不一致: {logical_id}")
    dependencies = payload.get("dependencies")
    if dependencies != list(plan_entry.definition.task_spec.dependencies):
        _raise(f"{label} evidence.dependencies 与 plan 不一致: {logical_id}")
    request_context = payload.get("request_context")
    if not isinstance(request_context, Mapping):
        _raise(f"{label} evidence.request_context 缺失: {logical_id}")
    expected_request_context = {
        key: value
        for key, value in plan_entry.definition.request.items()
        if key != "evidence_hash"
    }
    if dict(request_context) != expected_request_context:
        _raise(f"{label} evidence.request_context 与 plan 不一致: {logical_id}")
    expected_evidence_sha = _sha256_string(
        plan_entry.definition.request.get("evidence_hash"),
        context=f"{label} plan.request.evidence_hash",
    )
    if expected_evidence_sha != evidence_sha256:
        _raise(f"{label} evidence_sha256 与 plan.request.evidence_hash 不一致: {logical_id}")


def _validate_exhausted_attempt_payload(
    payload: Mapping[str, Any],
    *,
    logical_id: str,
    evaluation_key: str,
    task_type: str,
    attempt_id: str,
    attempt_number: int,
    error_class: str,
    retryable: bool,
    label: str,
) -> None:
    expected_attempt_id = f"{evaluation_key}.a{attempt_number:02d}"
    if attempt_id != expected_attempt_id:
        _raise(f"{label} exhausted.attempt_id 不符合规范命名: {logical_id}")
    if payload.get("attempt_id") != attempt_id:
        _raise(f"{label} exhausted.attempt_json.attempt_id 与 index 不一致: {logical_id}")
    if payload.get("evaluation_key") != evaluation_key:
        _raise(f"{label} exhausted.attempt_json.evaluation_key 与 index 不一致: {logical_id}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        _raise(f"{label} exhausted.attempt_json.metadata 缺失: {logical_id}")
    expected_pairs = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "evaluation_key": evaluation_key,
        "logical_id": logical_id,
        "task_type": task_type,
        "error_class": error_class,
    }
    for field_name, expected_value in expected_pairs.items():
        if metadata.get(field_name) != expected_value:
            _raise(f"{label} exhausted.attempt_json.metadata.{field_name} 漂移: {logical_id}")
    if _parse_bool(
        metadata.get("retryable"),
        context=f"{label}.{logical_id}.exhausted.metadata.retryable",
    ) is not retryable:
        _raise(f"{label} exhausted.attempt_json.metadata.retryable 漂移: {logical_id}")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        _raise(f"{label} exhausted.attempt_json.validation 缺失: {logical_id}")
    if _parse_bool(
        validation.get("ok"),
        context=f"{label}.{logical_id}.exhausted.validation.ok",
    ):
        _raise(f"{label} exhausted.attempt_json.validation.ok 必须为 false: {logical_id}")
    validation_error_class = _string(
        validation.get("error_class"),
        context=f"{label}.{logical_id}.exhausted.validation.error_class",
    )
    if validation_error_class != error_class:
        _raise(f"{label} exhausted.attempt_json.validation.error_class 漂移: {logical_id}")


def _validate_exhausted_payload(
    row: Mapping[str, Any],
    *,
    logical_id: str,
    evaluation_key: str,
    task_type: str,
    label: str,
) -> dict[str, Any]:
    if row.get("attempt_id") is not None:
        _raise(f"{label} {logical_id} exhausted 时 attempt_id 必须为 null")
    if row.get("structured_output") is not None:
        _raise(f"{label} {logical_id} exhausted 时 structured_output 必须为 null")
    if row.get("non_applicable") is not None:
        _raise(f"{label} {logical_id} exhausted 时 non_applicable 必须为 null")
    if row.get("result_path") is not None or row.get("result_sha256") is not None:
        _raise(f"{label} {logical_id} exhausted 时 result 绑定必须为 null")
    exhausted = row.get("exhausted")
    if not isinstance(exhausted, Mapping):
        _raise(f"{label} {logical_id}.exhausted 缺失")
    attempt_count = _parse_int(
        exhausted.get("attempt_count"),
        context=f"{label}.{logical_id}.exhausted.attempt_count",
    )
    if attempt_count != 3:
        _raise(f"{label} {logical_id}.exhausted.attempt_count 必须为 3")
    last_error_class = _string(
        exhausted.get("last_error_class"),
        context=f"{label}.{logical_id}.exhausted.last_error_class",
    )
    attempts = exhausted.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != attempt_count:
        _raise(f"{label} {logical_id}.exhausted.attempts 数量必须等于 attempt_count")
    validated_attempts: list[dict[str, Any]] = []
    for expected_number, item in enumerate(attempts, start=1):
        if not isinstance(item, Mapping):
            _raise(f"{label} {logical_id}.exhausted.attempts[{expected_number}] 必须是 object")
        attempt_id = _string(
            item.get("attempt_id"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].attempt_id",
        )
        attempt_number = _parse_int(
            item.get("attempt_number"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].attempt_number",
        )
        if attempt_number != expected_number:
            _raise(f"{label} {logical_id}.exhausted.attempt_number 序列不连续")
        status = _string(
            item.get("status"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].status",
        )
        if status != "failed":
            _raise(f"{label} {logical_id}.exhausted 仅允许 failed attempts")
        error_class = _string(
            item.get("error_class"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].error_class",
        )
        retryable = _parse_bool(
            item.get("retryable"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].retryable",
        )
        attempt_path = _resolve_report_path(
            item.get("attempt_path"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].attempt_path",
        )
        attempt_sha256 = _sha256_string(
            item.get("attempt_sha256"),
            context=f"{label}.{logical_id}.exhausted.attempts[{expected_number}].attempt_sha256",
        )
        if not attempt_path.is_file():
            _raise(f"{label} exhausted attempt 文件不存在: {attempt_path}")
        if _sha256_file(attempt_path) != attempt_sha256:
            _raise(f"{label} exhausted.attempt_sha256 与文件不一致: {logical_id}")
        payload = _read_json_object(
            attempt_path,
            context=f"{label} exhausted attempt_json[{expected_number}]",
        )
        _validate_exhausted_attempt_payload(
            payload,
            logical_id=logical_id,
            evaluation_key=evaluation_key,
            task_type=task_type,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            error_class=error_class,
            retryable=retryable,
            label=label,
        )
        validated_attempts.append(
            {
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "status": status,
                "error_class": error_class,
                "retryable": retryable,
                "attempt_path": str(attempt_path),
                "attempt_sha256": attempt_sha256,
            }
        )
    if validated_attempts[-1]["error_class"] != last_error_class:
        _raise(f"{label} {logical_id}.exhausted.last_error_class 与最后一次 attempt 不一致")
    return {
        "attempt_count": attempt_count,
        "last_error_class": last_error_class,
        "attempts": validated_attempts,
    }


def _load_frozen_index_rows(
    *,
    index_path: Path,
    summary_path: Path,
    plan_path: Path,
    expected_task_type: str,
    label: str,
    allow_exhausted: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = _load_plan_with_contract(plan_path, label=f"{label} plan")
    summary_info = _validate_frozen_summary(
        summary_path,
        index_path=index_path,
        plan=plan,
        label=label,
        allow_exhausted=allow_exhausted,
    )
    rows = _iter_jsonl(index_path, label=f"{label} index")
    if len(rows) != summary_info["row_count"]:
        _raise(
            f"{label}.summary.row_count 应为 {len(rows)}，实际记录为 {summary_info['row_count']}"
        )
    plan_entries_by_logical_id = {entry.logical_id: entry for entry in plan.entries}
    seen_evaluation_keys: set[str] = set()
    state_counts = {"frozen": 0, "non_applicable": 0}
    if allow_exhausted:
        state_counts["exhausted"] = 0
    for row in rows:
        logical_id = _string(row.get("logical_id"), context=f"{label}.logical_id")
        plan_entry = plan_entries_by_logical_id.get(logical_id)
        if plan_entry is None:
            _raise(f"{label} index 存在未注册 logical_id: {logical_id}")
        evaluation_key = _string(
            row.get("evaluation_key"),
            context=f"{label}.{logical_id}.evaluation_key",
        )
        if evaluation_key != plan_entry.evaluation_key:
            _raise(f"{label} {logical_id}.evaluation_key 与 plan 不一致")
        if evaluation_key in seen_evaluation_keys:
            _raise(f"{label} index 出现重复 evaluation_key: {evaluation_key}")
        seen_evaluation_keys.add(evaluation_key)
        task_type = _string(row.get("task_type"), context=f"{label}.{logical_id}.task_type")
        if task_type != expected_task_type:
            _raise(f"{label} {logical_id}.task_type 非法: {task_type!r}")
        if task_type != plan_entry.definition.task_spec.task_type:
            _raise(f"{label} {logical_id}.task_type 与 plan 不一致")
        row_plan_sha256 = _sha256_string(
            row.get("plan_sha256"),
            context=f"{label}.{logical_id}.plan_sha256",
        )
        if row_plan_sha256 != plan.plan_sha256:
            _raise(f"{label} {logical_id}.plan_sha256 与 plan 不一致")
        if row.get("condition") != NOISE_TAG:
            _raise(f"{label} {logical_id}.condition 不是 clean")
        if row.get("condition") != plan_entry.definition.task_spec.condition:
            _raise(f"{label} {logical_id}.condition 与 plan 不一致")
        if _parse_int(row.get("priority"), context=f"{label}.{logical_id}.priority") != plan_entry.definition.task_spec.priority:
            _raise(f"{label} {logical_id}.priority 与 plan 不一致")
        state = _string(row.get("state"), context=f"{label}.{logical_id}.state")
        if state == "frozen":
            if row.get("non_applicable") is not None:
                _raise(f"{label} {logical_id} frozen 时 non_applicable 必须为 null")
            if row.get("exhausted") is not None:
                _raise(f"{label} {logical_id} frozen 时 exhausted 必须为 null")
            structured_output = row.get("structured_output")
            if not isinstance(structured_output, Mapping):
                _raise(f"{label} {logical_id}.structured_output 缺失")
            result_path = _resolve_report_path(
                row.get("result_path"),
                context=f"{label}.{logical_id}.result_path",
            )
            result_sha256 = _sha256_string(
                row.get("result_sha256"),
                context=f"{label}.{logical_id}.result_sha256",
            )
            if not result_path.is_file():
                _raise(f"{label} 结果文件不存在: {result_path}")
            if _sha256_file(result_path) != result_sha256:
                _raise(f"{label} {logical_id}.result_sha256 与结果文件不一致")
            _validate_frozen_result_payload(
                result_path,
                logical_id=logical_id,
                evaluation_key=evaluation_key,
                task_type=task_type,
                plan_sha256=plan.plan_sha256,
                structured_output=structured_output,
                label=label,
            )
            if task_type in {"gt_simplify", "pred_simplify"}:
                _validate_simplify_structured_output(
                    row,
                    logical_id=logical_id,
                    plan_request=plan_entry.definition.request,
                    allow_missing=(task_type == "pred_simplify"),
                )
            else:
                if row.get("effective_expression") is not None:
                    _raise(f"{label} {logical_id} 非 simplify frozen 行的 effective_expression 必须为 null")
                if row.get("expression_resolution") is not None:
                    _raise(f"{label} {logical_id} 非 simplify frozen 行的 expression_resolution 必须为 null")
        elif state == "non_applicable":
            if row.get("structured_output") is not None:
                _raise(f"{label} {logical_id} non_applicable 时 structured_output 必须为 null")
            if row.get("result_path") is not None or row.get("result_sha256") is not None:
                _raise(f"{label} {logical_id} non_applicable 时 result 绑定必须为 null")
            if row.get("exhausted") is not None:
                _raise(f"{label} {logical_id} non_applicable 时 exhausted 必须为 null")
            if row.get("effective_expression") is not None:
                _raise(f"{label} {logical_id} non_applicable 时 effective_expression 必须为 null")
            if row.get("expression_resolution") is not None:
                _raise(f"{label} {logical_id} non_applicable 时 expression_resolution 必须为 null")
            non_applicable = row.get("non_applicable")
            if not isinstance(non_applicable, Mapping):
                _raise(f"{label} {logical_id}.non_applicable 缺失")
            reason = _string(
                non_applicable.get("reason"),
                context=f"{label}.{logical_id}.non_applicable.reason",
            )
            evidence_sha256 = _sha256_string(
                non_applicable.get("evidence_sha256"),
                context=f"{label}.{logical_id}.non_applicable.evidence_sha256",
            )
            evidence_path = _resolve_report_path(
                non_applicable.get("evidence_path"),
                context=f"{label}.{logical_id}.non_applicable.evidence_path",
            )
            if not evidence_path.is_file():
                _raise(f"{label} 证据文件不存在: {evidence_path}")
            if _sha256_file(evidence_path) != evidence_sha256:
                _raise(f"{label} {logical_id}.non_applicable.evidence_sha256 与证据文件不一致")
            _validate_non_applicable_evidence_payload(
                evidence_path,
                plan_entry=plan_entry,
                logical_id=logical_id,
                task_type=task_type,
                reason=reason,
                evidence_sha256=evidence_sha256,
                plan_sha256=plan.plan_sha256,
                label=label,
            )
        elif state == "exhausted":
            if not allow_exhausted:
                _raise(f"{label} {logical_id}.state 非法: {state!r}")
            if task_type != "pred_simplify":
                _raise(f"{label} {logical_id} 仅 pred_simplify 允许 exhausted")
            if row.get("effective_expression") is not None:
                _raise(f"{label} {logical_id} exhausted 时 effective_expression 必须为 null")
            if row.get("expression_resolution") is not None:
                _raise(f"{label} {logical_id} exhausted 时 expression_resolution 必须为 null")
            _validate_exhausted_payload(
                row,
                logical_id=logical_id,
                evaluation_key=evaluation_key,
                task_type=task_type,
                label=label,
            )
        else:
            _raise(f"{label} {logical_id}.state 非法: {state!r}")
        state_counts[state] += 1
    if set(plan_entries_by_logical_id) != {
        _string(row.get("logical_id"), context=f"{label}.logical_id") for row in rows
    }:
        _raise(f"{label} index 与 plan logical_id 集合不闭合")
    if state_counts != summary_info["state_counts"]:
        _raise(f"{label}.summary.state_counts 与 index 实际统计不一致")
    return rows, {
        "index_jsonl": {
            "path": summary_info["index_path"],
            "sha256": summary_info["index_sha256"],
            "row_count": len(rows),
        },
        "summary_json": {
            "path": summary_info["summary_path"],
            "sha256": summary_info["summary_sha256"],
            "status": summary_info["status"],
            "state_counts": summary_info["state_counts"],
        },
        "plan_jsonl": {
            "path": summary_info["plan_path"],
            "sha256": summary_info["plan_sha256"],
            "row_count": summary_info["plan_row_count"],
        },
        "plan_requests_by_logical_id": {
            entry.logical_id: dict(entry.definition.request) for entry in plan.entries
        },
    }


def _select_simplified_expression(
    row: Mapping[str, Any],
    *,
    logical_id: str,
    plan_request: Mapping[str, Any] | None,
    allow_missing: bool,
) -> tuple[str | None, str]:
    state = _string(row.get("state"), context=f"{logical_id}.state")
    if state == "exhausted":
        if row.get("effective_expression") is not None or row.get("expression_resolution") is not None:
            _raise(f"{logical_id} exhausted 时 effective_expression/expression_resolution 必须为 null")
        if not allow_missing:
            _raise(f"{logical_id} 不允许 exhausted")
        _validate_exhausted_payload(
            row,
            logical_id=logical_id,
            evaluation_key=_string(row.get("evaluation_key"), context=f"{logical_id}.evaluation_key"),
            task_type=_string(row.get("task_type"), context=f"{logical_id}.task_type"),
            label="pred_frozen_index",
        )
        return None, state
    if state == "non_applicable":
        if row.get("effective_expression") is not None or row.get("expression_resolution") is not None:
            _raise(f"{logical_id} non_applicable 时 effective_expression/expression_resolution 必须为 null")
        _validate_non_applicable(
            row,
            logical_id=logical_id,
            allowed_reasons=PRED_NO_CALL_REASONS if allow_missing else GT_NO_CALL_REASONS,
        )
        if not allow_missing:
            _raise(f"{logical_id} 不允许 non_applicable")
        return None, state
    if state != "frozen":
        _raise(f"{logical_id} state 必须是 frozen/non_applicable/exhausted")
    return _validate_simplify_structured_output(
        row,
        logical_id=logical_id,
        plan_request=plan_request,
        allow_missing=allow_missing,
    )


def _load_pred_simplify_identity_map(
    path: Path,
    *,
    expected_runs: int,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    rows = _iter_jsonl(path, label="simplify_plan_jsonl")
    identity_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    pred_count = 0
    for row in rows:
        task_type = _string(row.get("task_type"), context="simplify_plan.task_type")
        if task_type != "pred_simplify":
            continue
        logical_id = _string(row.get("logical_id"), context="simplify_plan.logical_id")
        request = row.get("request")
        if not isinstance(request, Mapping):
            _raise(f"{logical_id}.request 缺失")
        algorithm = _string(request.get("algorithm"), context=f"{logical_id}.request.algorithm")
        algorithm_slug = _string(
            request.get("algorithm_slug"),
            context=f"{logical_id}.request.algorithm_slug",
        )
        dataset_id = _string(request.get("dataset_id"), context=f"{logical_id}.request.dataset_id")
        dataset_index = _string(
            request.get("dataset_index"),
            context=f"{logical_id}.request.dataset_index",
        )
        if request.get("noise_tag") not in {None, NOISE_TAG}:
            _raise(f"{logical_id}.request.noise_tag 必须为 clean")
        try:
            seed = int(request.get("seed"))
        except (TypeError, ValueError) as exc:
            raise AggregateCleanMetricsError(f"{logical_id}.request.seed 非法") from exc
        parsed_identity = _parse_run_logical_id(logical_id, expected_prefix="pred_simplify")
        if parsed_identity != (algorithm_slug, dataset_index, seed):
            _raise(f"simplify plan logical_id 与 request 身份不匹配: {logical_id!r}")
        key = (algorithm_slug, dataset_index, seed)
        if key in identity_map:
            _raise(f"simplify plan 出现重复 pred 身份映射: {logical_id}")
        identity_map[key] = {
            "logical_id": logical_id,
            "algorithm": algorithm,
            "algorithm_slug": algorithm_slug,
            "dataset_id": dataset_id,
            "dataset_index": dataset_index,
            "seed": seed,
            "numeric_logical_key": _logical_key(algorithm, dataset_id, seed),
            "plan_request": dict(request),
        }
        pred_count += 1
    if pred_count != expected_runs:
        _raise(f"simplify plan 的 pred 行数应为 {expected_runs}，实际为 {pred_count}")
    return identity_map, {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "pred_row_count": pred_count,
    }


def _build_gt_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    gt_plan_requests_by_logical_id: Mapping[str, Mapping[str, Any]],
    expected_datasets: int,
) -> dict[str, dict[str, Any]]:
    gt_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_id = _string(row.get("logical_id"), context="gt.logical_id")
        dataset_id = _parse_gt_logical_id(logical_id)
        if row.get("condition") != NOISE_TAG:
            _raise(f"{logical_id} condition 不是 clean")
        plan_request = gt_plan_requests_by_logical_id.get(logical_id)
        if plan_request is None:
            _raise(f"{logical_id} 缺少 GT plan request 绑定")
        plan_dataset_id = _string(
            plan_request.get("dataset_id"),
            context=f"{logical_id}.plan_request.dataset_id",
        )
        if plan_dataset_id != dataset_id:
            _raise(f"{logical_id}.plan_request.dataset_id 与 logical_id 不一致")
        expression, state = _select_simplified_expression(
            row,
            logical_id=logical_id,
            plan_request=plan_request,
            allow_missing=False,
        )
        assert expression is not None
        try:
            artifact = build_symbolic_artifact(expression)
        except SymbolicEvidenceError as exc:
            raise AggregateCleanMetricsError(f"{logical_id} GT 简化式不可解析: {exc}") from exc
        if dataset_id in gt_map:
            _raise(f"gt_index_jsonl 出现重复 dataset_id: {dataset_id}")
        gt_map[dataset_id] = {
            "logical_id": logical_id,
            "state": state,
            "expression": expression,
            "artifact": artifact,
        }
    if len(gt_map) != expected_datasets:
        _raise(f"gt_index_jsonl 数据集数应为 {expected_datasets}，实际为 {len(gt_map)}")
    return gt_map


def _build_pred_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
    expected_runs: int,
) -> dict[str, dict[str, Any]]:
    pred_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_id = _string(row.get("logical_id"), context="pred.logical_id")
        task_type = _string(row.get("task_type"), context=f"{logical_id}.task_type")
        if task_type != "pred_simplify":
            _raise(f"{logical_id}.task_type 必须为 pred_simplify")
        if row.get("condition") != NOISE_TAG:
            _raise(f"{logical_id}.condition 不是 clean")
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            logical_id,
            expected_prefix="pred_simplify",
        )
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=logical_id,
        )
        if logical_id != identity["logical_id"]:
            _raise(f"{logical_id} 不是 pred simplify plan 的 active logical_id")
        expression, state = _select_simplified_expression(
            row,
            logical_id=logical_id,
            plan_request=identity["plan_request"],
            allow_missing=True,
        )
        artifact = None
        if expression is not None:
            try:
                artifact = build_symbolic_artifact(expression)
            except SymbolicEvidenceError as exc:
                raise AggregateCleanMetricsError(f"{logical_id} 预测简化式不可解析: {exc}") from exc
        key = str(identity["numeric_logical_key"])
        if key in pred_map:
            _raise(f"pred_index_jsonl 出现重复 logical_key: {key}")
        pred_map[key] = {
            "logical_id": logical_id,
            "state": state,
            "expression": expression,
            "artifact": artifact,
            "symbolic_valid": expression is not None,
            "algorithm_slug": algorithm_slug,
            "dataset_index": dataset_index,
        }
    if len(pred_map) != expected_runs:
        _raise(f"pred_index_jsonl 行数应为 {expected_runs}，实际为 {len(pred_map)}")
    return pred_map


def _build_equivalence_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
    expected_runs: int,
) -> dict[str, dict[str, Any]]:
    eq_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_id = _string(row.get("logical_id"), context="equivalence.logical_id")
        task_type = _string(row.get("task_type"), context=f"{logical_id}.task_type")
        if task_type != "equivalence":
            _raise(f"{logical_id}.task_type 必须为 equivalence")
        if row.get("condition") != NOISE_TAG:
            _raise(f"{logical_id}.condition 不是 clean")
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(logical_id, expected_prefix="equivalence")
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=logical_id,
        )
        state = _string(row.get("state"), context=f"{logical_id}.state")
        decision = None
        if state == "frozen":
            structured_output = row.get("structured_output")
            if not isinstance(structured_output, Mapping):
                _raise(f"{logical_id}.structured_output 缺失")
            decision = _string(structured_output.get("decision"), context=f"{logical_id}.decision")
            if decision not in EQUIVALENCE_DECISIONS:
                _raise(f"{logical_id}.decision 非法: {decision!r}")
            evidence_basis = _string(
                structured_output.get("evidence_basis"),
                context=f"{logical_id}.evidence_basis",
            )
            if evidence_basis not in EQUIVALENCE_EVIDENCE_BASES:
                _raise(f"{logical_id}.evidence_basis 非法: {evidence_basis!r}")
            if decision in {"equivalent", "not_equivalent"} and evidence_basis == "insufficient":
                _raise(f"{logical_id} 确定性判定不得使用 insufficient evidence_basis")
        elif state != "non_applicable":
            _raise(f"{logical_id}.state 非法")
        else:
            _validate_non_applicable(
                row,
                logical_id=logical_id,
                allowed_reasons=EQUIVALENCE_NO_CALL_REASONS,
            )
        key = str(identity["numeric_logical_key"])
        if key in eq_map:
            _raise(f"equivalence_index_jsonl 出现重复 logical_key: {key}")
        eq_map[key] = {"logical_id": logical_id, "state": state, "decision": decision}
    if len(eq_map) != expected_runs:
        _raise(f"equivalence_index_jsonl 行数应为 {expected_runs}，实际为 {len(eq_map)}")
    return eq_map


def _build_structure_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
    expected_rows: int,
) -> dict[tuple[str, str, tuple[int, int]], dict[str, Any]]:
    structure_map: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
    for row in rows:
        logical_id = _string(row.get("logical_id"), context="structure.logical_id")
        task_type = _string(row.get("task_type"), context=f"{logical_id}.task_type")
        if task_type != "stab_structure":
            _raise(f"{logical_id}.task_type 必须为 stab_structure")
        if row.get("condition") != NOISE_TAG:
            _raise(f"{logical_id}.condition 不是 clean")
        algorithm_slug, dataset_index, pair = _parse_structure_logical_id(logical_id)
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=pair[0],
            logical_id=logical_id,
        )
        state = _string(row.get("state"), context=f"{logical_id}.state")
        decision = None
        if state == "frozen":
            structured_output = row.get("structured_output")
            if not isinstance(structured_output, Mapping):
                _raise(f"{logical_id}.structured_output 缺失")
            decision = _string(structured_output.get("decision"), context=f"{logical_id}.decision")
            if decision not in STRUCTURE_DECISIONS:
                _raise(f"{logical_id}.decision 非法: {decision!r}")
        elif state != "non_applicable":
            _raise(f"{logical_id}.state 非法")
        else:
            _validate_non_applicable(
                row,
                logical_id=logical_id,
                allowed_reasons=STRUCTURE_NO_CALL_REASONS,
            )
        key = (str(identity["algorithm"]), str(identity["dataset_id"]), pair)
        if key in structure_map:
            _raise(f"structure_index_jsonl 出现重复 pair: {logical_id}")
        structure_map[key] = {"logical_id": logical_id, "state": state, "decision": decision}
    if len(structure_map) != expected_rows:
        _raise(f"structure_index_jsonl 行数应为 {expected_rows}，实际为 {len(structure_map)}")
    return structure_map


def _parse_evidence_key(
    row: Mapping[str, Any],
    *,
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
) -> str:
    logical_key = _optional_nonempty_string(row.get("logical_key"))
    if logical_key is not None:
        algorithm, dataset_id, seed = _parse_numeric_logical_key(logical_key)
        key_value = _logical_key(algorithm, dataset_id, seed)
    else:
        pred_logical_id = _optional_nonempty_string(row.get("pred_logical_id"))
        if pred_logical_id is None:
            _raise("deterministic evidence 缺少 logical_key/pred_logical_id")
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            pred_logical_id,
            expected_prefix="pred_simplify",
        )
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=pred_logical_id,
        )
        if pred_logical_id != identity["logical_id"]:
            _raise(f"{pred_logical_id} 不是 pred simplify plan 的 active logical_id")
        key_value = str(identity["numeric_logical_key"])
    pred_logical_id = _optional_nonempty_string(row.get("pred_logical_id"))
    if pred_logical_id is not None:
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            pred_logical_id,
            expected_prefix="pred_simplify",
        )
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=pred_logical_id,
        )
        if pred_logical_id != identity["logical_id"]:
            _raise(f"{pred_logical_id} 不是 pred simplify plan 的 active logical_id")
        expected_key = str(identity["numeric_logical_key"])
        if key_value != expected_key:
            _raise(f"deterministic evidence logical_key 漂移: {key_value!r} != {expected_key!r}")
    return key_value


def _validate_report_file_binding(
    report_path: Path,
    *,
    field_name: str,
    expected_path: Path,
    expected_sha256: str,
    expected_row_count: int,
) -> dict[str, Any]:
    payload = _read_json_object(report_path, context=field_name)
    status = _string(payload.get("status"), context=f"{field_name}.status")
    if status != "ok":
        _raise(f"{field_name}.status 必须为 'ok'")
    if not _parse_bool(payload.get("contract_ok"), context=f"{field_name}.contract_ok"):
        _raise(f"{field_name}.contract_ok 必须为 true")
    return {
        "payload": payload,
        "info": {
            "path": str(report_path.resolve()),
            "sha256": _sha256_file(report_path),
        },
    }


def _validate_path_sha_rowcount_triplet(
    payload: Mapping[str, Any],
    *,
    section: str,
    raw_path: object,
    raw_sha256: object,
    raw_row_count: object,
    expected_path: Path,
    expected_sha256: str,
    expected_row_count: int,
) -> dict[str, Any]:
    resolved_path = _resolve_report_path(raw_path, context=f"{section}.path")
    if not Path(str(raw_path)).is_absolute():
        _raise(f"{section}.path 必须是绝对路径")
    if resolved_path != expected_path.resolve():
        _raise(f"{section}.path 与目标文件不一致")
    sha256 = _sha256_string(raw_sha256, context=f"{section}.sha256")
    if sha256 != expected_sha256:
        _raise(f"{section}.sha256 与目标文件不一致")
    row_count = _parse_int(raw_row_count, context=f"{section}.row_count")
    if row_count != expected_row_count:
        _raise(f"{section}.row_count 应为 {expected_row_count}，实际为 {row_count}")
    return {"path": str(resolved_path), "sha256": sha256, "row_count": row_count}


def _validate_clean_numeric_replay_contract(
    payload: Mapping[str, Any],
    *,
    rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """验证最终数值指标全部来自同一 canonical replay 路径。"""
    if payload.get("evaluation_path") != CANONICAL_REPLAY_EVALUATION_PATH:
        _raise(
            "clean_numeric_preparation_report.evaluation_path 必须为 "
            f"{CANONICAL_REPLAY_EVALUATION_PATH}"
        )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        _raise("clean_numeric_preparation_report.counts 缺失")
    valid_outputs = _parse_int(
        counts.get("valid_outputs"),
        context="clean_numeric_preparation_report.counts.valid_outputs",
    )
    invalid_outputs = _parse_int(
        counts.get("canonical_invalid_outputs"),
        context="clean_numeric_preparation_report.counts.canonical_invalid_outputs",
    )
    replay_unavailable = _parse_int(
        counts.get("replay_unavailable"),
        context="clean_numeric_preparation_report.counts.replay_unavailable",
    )
    replay_errors = _parse_int(
        counts.get("replay_errors"),
        context="clean_numeric_preparation_report.counts.replay_errors",
    )
    if min(valid_outputs, invalid_outputs, replay_unavailable, replay_errors) < 0:
        _raise("clean_numeric_preparation_report replay 状态计数不得为负数")
    if valid_outputs + invalid_outputs + replay_unavailable != len(rows):
        _raise(
            "clean_numeric_preparation_report replay 状态计数未与 CSV 行数闭合"
        )
    if replay_unavailable != 0:
        _raise(
            "clean_numeric_preparation_report.counts.replay_unavailable 必须为 0，"
            f"实际为 {replay_unavailable}"
        )
    if replay_errors != 0:
        _raise(
            "clean_numeric_preparation_report.counts.replay_errors 必须为 0，"
            f"实际为 {replay_errors}"
        )
    row_valid_outputs = 0
    row_invalid_outputs = 0
    row_replay_unavailable = 0
    row_replay_errors = 0
    for logical_key, row in rows.items():
        if row.get("evaluation_path") != CANONICAL_REPLAY_EVALUATION_PATH:
            _raise(
                f"{logical_key} clean numeric CSV evaluation_path 必须为 "
                f"{CANONICAL_REPLAY_EVALUATION_PATH}"
            )
        evaluation_status = str(row.get("evaluation_status") or "").strip()
        if evaluation_status not in {"valid", "invalid_output", "replay_unavailable"}:
            _raise(f"{logical_key}.evaluation_status 枚举非法或缺失")
        replay_error = str(row.get("replay_error") or "").strip()
        invalid_reason = str(row.get("invalid_reason") or "").strip()
        valid_output_text = str(row.get("valid_output") or "").strip().lower()
        id_quality_text = str(row.get("id_quality") or "").strip()
        ood_quality_text = str(row.get("ood_quality") or "").strip()
        if replay_error:
            row_replay_errors += 1
            if evaluation_status != "replay_unavailable":
                _raise(f"{logical_key} clean numeric CSV replay_error 必须为空")
        if evaluation_status == "valid":
            row_valid_outputs += 1
            if valid_output_text != "true" or invalid_reason or replay_error:
                _raise(f"{logical_key} valid 状态字段不一致")
            _finite_unit_float(id_quality_text, context=f"{logical_key}.id_quality")
            _finite_unit_float(ood_quality_text, context=f"{logical_key}.ood_quality")
        elif evaluation_status == "invalid_output":
            row_invalid_outputs += 1
            if valid_output_text != "false" or not invalid_reason or replay_error:
                _raise(f"{logical_key} invalid_output 状态字段不一致")
            if (
                _finite_unit_float(id_quality_text, context=f"{logical_key}.id_quality") != 0.0
                or _finite_unit_float(ood_quality_text, context=f"{logical_key}.ood_quality")
                != 0.0
            ):
                _raise(f"{logical_key} invalid_output 的 ID/OOD 质量必须为 0")
        else:
            row_replay_unavailable += 1
            if valid_output_text or invalid_reason or not replay_error:
                _raise(f"{logical_key} replay_unavailable 状态字段不一致")
            if id_quality_text or ood_quality_text:
                _raise(f"{logical_key} replay_unavailable 不得携带 ID/OOD 质量")
    if row_valid_outputs != valid_outputs:
        _raise(
            "clean_numeric_preparation_report.counts.valid_outputs 与 CSV 不一致: "
            f"{valid_outputs} != {row_valid_outputs}"
        )
    if row_invalid_outputs != invalid_outputs:
        _raise(
            "clean_numeric_preparation_report.counts.canonical_invalid_outputs "
            f"与 CSV 不一致: {invalid_outputs} != {row_invalid_outputs}"
        )
    if row_replay_unavailable != replay_unavailable or row_replay_errors != replay_errors:
        _raise("clean numeric replay unavailable/error 计数与 CSV 不一致")
    return {
        "evaluation_path": CANONICAL_REPLAY_EVALUATION_PATH,
        "valid_outputs": valid_outputs,
        "replay_errors": replay_errors,
        "invalid_outputs": invalid_outputs,
        "replay_unavailable": replay_unavailable,
    }


def _validate_clean_numeric_preparation_report(
    report_path: Path,
    *,
    numeric_csv: Path,
    numeric_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    wrapped = _validate_report_file_binding(
        report_path,
        field_name="clean_numeric_preparation_report",
        expected_path=numeric_csv,
        expected_sha256=_sha256_file(numeric_csv),
        expected_row_count=len(numeric_rows),
    )
    payload = wrapped["payload"]
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        _raise("clean_numeric_preparation_report.inputs 缺失")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        _raise("clean_numeric_preparation_report.outputs 缺失")
    freeze_binding_path = _resolve_report_path(
        inputs.get("freeze_binding_json"),
        context="clean_numeric_preparation_report.inputs.freeze_binding_json",
    )
    if not freeze_binding_path.is_file():
        _raise("clean_numeric_preparation_report.inputs.freeze_binding_json 不存在")
    freeze_binding_sha256 = _sha256_string(
        inputs.get("freeze_binding_sha256"),
        context="clean_numeric_preparation_report.inputs.freeze_binding_sha256",
    )
    if _sha256_file(freeze_binding_path) != freeze_binding_sha256:
        _raise("clean_numeric_preparation_report.freeze_binding_sha256 与文件不一致")
    freeze_bundles = inputs.get("freeze_bundles")
    if not isinstance(freeze_bundles, list) or not freeze_bundles:
        _raise("clean_numeric_preparation_report.inputs.freeze_bundles 缺失")
    validated_bundles: list[dict[str, Any]] = []
    for index, bundle in enumerate(freeze_bundles):
        if not isinstance(bundle, Mapping):
            _raise("clean_numeric_preparation_report.inputs.freeze_bundles 条目必须为 object")
        bundle_path = _resolve_report_path(
            bundle.get("path"),
            context=f"clean_numeric_preparation_report.inputs.freeze_bundles[{index}].path",
        )
        bundle_sha256 = _sha256_string(
            bundle.get("sha256"),
            context=f"clean_numeric_preparation_report.inputs.freeze_bundles[{index}].sha256",
        )
        if not bundle_path.is_file():
            _raise(f"clean_numeric_preparation_report.freeze_bundle 不存在: {bundle_path}")
        if _sha256_file(bundle_path) != bundle_sha256:
            _raise(f"clean_numeric_preparation_report.freeze_bundle SHA 不一致: {bundle_path}")
        validated_bundles.append({"path": str(bundle_path), "sha256": bundle_sha256})
    source_runs_sha256 = _sha256_string(
        inputs.get("source_runs_sha256"),
        context="clean_numeric_preparation_report.inputs.source_runs_sha256",
    )
    source_runs_csv = _resolve_report_path(
        inputs.get("source_runs_csv"),
        context="clean_numeric_preparation_report.inputs.source_runs_csv",
    )
    if not source_runs_csv.is_file():
        _raise("clean_numeric_preparation_report.inputs.source_runs_csv 不存在")
    if _sha256_file(source_runs_csv) != source_runs_sha256:
        _raise("clean_numeric_preparation_report.source_runs_sha256 与文件不一致")
    output_info = _validate_path_sha_rowcount_triplet(
        payload,
        section="clean_numeric_preparation_report.outputs.run_csv",
        raw_path=outputs.get("run_csv"),
        raw_sha256=outputs.get("run_csv_sha256"),
        raw_row_count=outputs.get("run_csv_row_count"),
        expected_path=numeric_csv,
        expected_sha256=_sha256_file(numeric_csv),
        expected_row_count=len(numeric_rows),
    )
    replay_contract = _validate_clean_numeric_replay_contract(payload, rows=numeric_rows)
    return {
        **wrapped["info"],
        "status": "ok",
        "contract_ok": True,
        "canonical_replay": replay_contract,
        "outputs": {"run_csv": output_info},
        "inputs": {
            "freeze_binding_json": str(freeze_binding_path),
            "freeze_binding_sha256": freeze_binding_sha256,
            "freeze_bundles": validated_bundles,
            "source_runs_csv": str(source_runs_csv),
            "source_runs_sha256": source_runs_sha256,
        },
    }


def _validate_eff_preparation_report(
    report_path: Path,
    *,
    eff_csv: Path,
    eff_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    wrapped = _validate_report_file_binding(
        report_path,
        field_name="eff_preparation_report",
        expected_path=eff_csv,
        expected_sha256=_sha256_file(eff_csv),
        expected_row_count=len(eff_rows),
    )
    payload = wrapped["payload"]
    if payload.get("condition") != NOISE_TAG:
        _raise("eff_preparation_report.condition 不是 clean")
    if _parse_int(payload.get("horizon"), context="eff_preparation_report.horizon") != EFF_HORIZON:
        _raise(f"eff_preparation_report.horizon 必须为 {EFF_HORIZON}")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        _raise("eff_preparation_report.summary 缺失")
    if not _parse_bool(summary.get("full_contract_checked"), context="eff_preparation_report.summary.full_contract_checked"):
        _raise("eff_preparation_report 必须声明 full_contract_checked=true")
    unresolved_run_count = _parse_int(
        summary.get("unresolved_run_count"),
        context="eff_preparation_report.summary.unresolved_run_count",
    )
    if unresolved_run_count != 0:
        _raise(f"eff_preparation_report.unresolved_run_count 必须为 0，实际为 {unresolved_run_count}")
    unresolved = payload.get("unresolved")
    if not isinstance(unresolved, list) or unresolved:
        _raise("eff_preparation_report.unresolved 必须为空数组")
    success_count = _parse_int(summary.get("success_count"), context="eff_preparation_report.summary.success_count")
    processed_count = _parse_int(
        summary.get("processed_run_count"),
        context="eff_preparation_report.summary.processed_run_count",
    )
    if success_count != len(eff_rows):
        _raise(f"eff_preparation_report.success_count 应为 {len(eff_rows)}，实际为 {success_count}")
    if processed_count != len(eff_rows):
        _raise(f"eff_preparation_report.processed_run_count 应为 {len(eff_rows)}，实际为 {processed_count}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        _raise("eff_preparation_report.inputs 缺失")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        _raise("eff_preparation_report.outputs 缺失")
    freeze_binding = inputs.get("freeze_binding_report")
    repair_manifest = inputs.get("repair_manifest")
    freeze_records = inputs.get("freeze_records")
    freeze_reports = inputs.get("freeze_reports")
    if not isinstance(freeze_binding, Mapping) or not isinstance(repair_manifest, Mapping):
        _raise("eff_preparation_report.inputs 缺少 freeze_binding_report/repair_manifest")
    if not isinstance(freeze_records, list) or not freeze_records:
        _raise("eff_preparation_report.inputs.freeze_records 缺失")
    if not isinstance(freeze_reports, list) or not freeze_reports:
        _raise("eff_preparation_report.inputs.freeze_reports 缺失")
    freeze_binding_path = _resolve_report_path(
        freeze_binding.get("path"),
        context="eff_preparation_report.inputs.freeze_binding_report.path",
    )
    freeze_binding_sha = _sha256_string(
        freeze_binding.get("sha256"),
        context="eff_preparation_report.inputs.freeze_binding_report.sha256",
    )
    if not freeze_binding_path.is_file():
        _raise("eff_preparation_report.inputs.freeze_binding_report.path 不存在")
    if _sha256_file(freeze_binding_path) != freeze_binding_sha:
        _raise("eff_preparation_report.freeze_binding_report.sha256 与文件不一致")
    repair_manifest_path = _resolve_report_path(
        repair_manifest.get("path"),
        context="eff_preparation_report.inputs.repair_manifest.path",
    )
    repair_manifest_sha = _sha256_string(
        repair_manifest.get("sha256"),
        context="eff_preparation_report.inputs.repair_manifest.sha256",
    )
    if not repair_manifest_path.is_file():
        _raise("eff_preparation_report.inputs.repair_manifest.path 不存在")
    if _sha256_file(repair_manifest_path) != repair_manifest_sha:
        _raise("eff_preparation_report.repair_manifest.sha256 与文件不一致")
    allowed_bundle_shas: set[str] = set()
    for section_name, entries in (("freeze_records", freeze_records), ("freeze_reports", freeze_reports)):
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                _raise(f"eff_preparation_report.inputs.{section_name} 条目必须为 object")
            artifact_path = _resolve_report_path(
                entry.get("path"),
                context=f"eff_preparation_report.inputs.{section_name}[{index}].path",
            )
            artifact_sha = _sha256_string(
                entry.get("sha256"),
                context=f"eff_preparation_report.inputs.{section_name}[{index}].sha256",
            )
            if not artifact_path.is_file():
                _raise(f"eff_preparation_report.inputs.{section_name} 文件不存在: {artifact_path}")
            if _sha256_file(artifact_path) != artifact_sha:
                _raise(f"eff_preparation_report.inputs.{section_name} SHA 不一致: {artifact_path}")
            if section_name == "freeze_records":
                allowed_bundle_shas.add(artifact_sha)
            else:
                pass
    allowed_report_shas = {
        _sha256_string(
            entry.get("sha256"),
            context=f"eff_preparation_report.inputs.freeze_reports[{index}].sha256",
        )
        for index, entry in enumerate(freeze_reports)
        if isinstance(entry, Mapping)
    }
    output_info = _validate_path_sha_rowcount_triplet(
        payload,
        section="eff_preparation_report.outputs.eff_csv",
        raw_path=outputs.get("eff_csv"),
        raw_sha256=outputs.get("eff_csv_sha256"),
        raw_row_count=outputs.get("eff_csv_row_count"),
        expected_path=eff_csv,
        expected_sha256=_sha256_file(eff_csv),
        expected_row_count=len(eff_rows),
    )
    for logical_key, row in eff_rows.items():
        if row.get("freeze_binding_report_sha256") != freeze_binding_sha:
            _raise(f"{logical_key} freeze_binding_report_sha256 与 eff_preparation_report 不一致")
        if row.get("repair_manifest_sha256") != repair_manifest_sha:
            _raise(f"{logical_key} repair_manifest_sha256 与 eff_preparation_report 不一致")
        if row.get("bundle_sha256") not in allowed_bundle_shas:
            _raise(f"{logical_key} bundle_sha256 未绑定到 eff_preparation_report.inputs.freeze_records")
        if row.get("bundle_report_sha256") not in allowed_report_shas:
            _raise(f"{logical_key} bundle_report_sha256 未绑定到 eff_preparation_report.inputs.freeze_reports")
    replay_contract = _validate_canonical_replay_contract(
        summary,
        rows=eff_rows,
        report_context="eff_preparation_report.summary",
        row_point_fields=True,
    )
    readiness = _validate_eff_readiness_summary(summary, eff_rows=eff_rows)
    return {
        **wrapped["info"],
        "status": "ok",
        "contract_ok": True,
        "summary": dict(summary),
        "canonical_replay": replay_contract,
        "readiness": readiness,
        "outputs": {"eff_csv": output_info},
        "inputs": {
            "freeze_binding_report": {
                "path": str(freeze_binding_path),
                "sha256": freeze_binding_sha,
            },
            "repair_manifest": {
                "path": str(repair_manifest_path),
                "sha256": repair_manifest_sha,
            },
            "freeze_record_count": len(freeze_records),
            "freeze_report_count": len(freeze_reports),
        },
    }


def _validate_canonical_replay_contract(
    summary: Mapping[str, Any],
    *,
    rows: Mapping[str, Mapping[str, str]],
    report_context: str,
    row_point_fields: bool,
) -> dict[str, Any]:
    """阻止旧评估路径或未闭合 replay 结果进入正式聚合。"""
    evaluation_path = summary.get("evaluation_path")
    if evaluation_path != CANONICAL_REPLAY_EVALUATION_PATH:
        _raise(
            f"{report_context}.evaluation_path 必须为 "
            f"{CANONICAL_REPLAY_EVALUATION_PATH}"
        )

    attempted = _parse_int(
        summary.get("canonical_replay_attempted_points"),
        context=f"{report_context}.canonical_replay_attempted_points",
    )
    succeeded = _parse_int(
        summary.get("canonical_replay_succeeded_points"),
        context=f"{report_context}.canonical_replay_succeeded_points",
    )
    failed = _parse_int(
        summary.get("canonical_replay_failed_points"),
        context=f"{report_context}.canonical_replay_failed_points",
    )
    invalid_output = _parse_int(
        summary.get("canonical_replay_invalid_output_points"),
        context=f"{report_context}.canonical_replay_invalid_output_points",
    )
    if min(attempted, succeeded, failed, invalid_output) < 0:
        _raise(f"{report_context} canonical replay 计数不得为负数")
    if failed != 0:
        _raise(f"{report_context}.canonical_replay_failed_points 必须为 0，实际为 {failed}")
    if attempted != succeeded + failed:
        _raise(
            f"{report_context}.canonical_replay_attempted_points 未与 succeeded+failed 闭合: "
            f"{attempted} != {succeeded}+{failed}"
        )
    if invalid_output > succeeded:
        _raise(
            f"{report_context}.canonical_replay_invalid_output_points 不得超过 succeeded: "
            f"{invalid_output} > {succeeded}"
        )

    row_totals = {"attempted": 0, "succeeded": 0, "failed": 0, "invalid_output": 0}
    for logical_key, row in rows.items():
        if row.get("evaluation_path") != CANONICAL_REPLAY_EVALUATION_PATH:
            _raise(
                f"{logical_key} CSV evaluation_path 必须为 "
                f"{CANONICAL_REPLAY_EVALUATION_PATH}"
            )
        if not row_point_fields:
            continue
        row_counts = {
            name: _parse_int(
                row.get(f"canonical_replay_{name}_points"),
                context=f"{logical_key}.canonical_replay_{name}_points",
            )
            for name in ("attempted", "succeeded", "failed", "invalid_output")
        }
        if row_counts["failed"] != 0:
            _raise(f"{logical_key} CSV canonical_replay_failed_points 必须为 0")
        if row_counts["attempted"] != row_counts["succeeded"]:
            _raise(
                f"{logical_key} CSV canonical_replay_succeeded_points 未与 "
                "canonical_replay_attempted_points 闭合"
            )
        for name, value in row_counts.items():
            row_totals[name] += value

    if row_point_fields:
        expected = {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "invalid_output": invalid_output,
        }
        for name, value in row_totals.items():
            if value != expected[name]:
                _raise(
                    f"{report_context}.canonical_replay_{name}_points 与 CSV 逐行合计不一致: "
                    f"{expected[name]} != {value}"
                )

    return {
        "evaluation_path": CANONICAL_REPLAY_EVALUATION_PATH,
        "attempted_points": attempted,
        "succeeded_points": succeeded,
        "failed_points": failed,
        "invalid_output_points": invalid_output,
    }


def _validate_eff_readiness_summary(
    summary: Mapping[str, Any],
    *,
    eff_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """验证 EFF 原始缺口、审计修复与逐行产物构成精确闭环。"""
    if not _parse_bool(
        summary.get("formal_eff_ready"),
        context="eff_preparation_report.summary.formal_eff_ready",
    ):
        _raise("eff_preparation_report.summary.formal_eff_ready 必须为 true")

    raw_missing = _parse_int(
        summary.get("original_missing_points"),
        context="eff_preparation_report.summary.original_missing_points",
    )
    unresolved_missing = _parse_int(
        summary.get("missing_points_after_repairs"),
        context="eff_preparation_report.summary.missing_points_after_repairs",
    )
    repaired_missing = _parse_int(
        summary.get("audited_repair_points"),
        context="eff_preparation_report.summary.audited_repair_points",
    )
    future_ignored = _parse_int(
        summary.get("future_backfill_ignored_points"),
        context="eff_preparation_report.summary.future_backfill_ignored_points",
    )
    normalized = _parse_int(
        summary.get("checkpoint_normalization_points"),
        context="eff_preparation_report.summary.checkpoint_normalization_points",
    )
    if min(raw_missing, unresolved_missing, repaired_missing, future_ignored, normalized) < 0:
        _raise("eff_preparation_report 的 EFF 审计计数不得为负数")
    if unresolved_missing != 0:
        _raise(
            "eff_preparation_report.summary.missing_points_after_repairs 必须为 0，"
            f"实际为 {unresolved_missing}"
        )
    if repaired_missing != raw_missing:
        _raise(
            "eff_preparation_report 的原始缺口与 audited_repair_points 未精确闭合: "
            f"raw={raw_missing} repaired={repaired_missing}"
        )

    row_totals = {
        field: sum(
            _parse_int(row.get(field), context=f"{logical_key}.{field}")
            for logical_key, row in eff_rows.items()
        )
        for field in (
            "audited_repair_points",
            "future_backfill_ignored_points",
            "checkpoint_normalization_points",
        )
    }
    expected_totals = {
        "audited_repair_points": repaired_missing,
        "future_backfill_ignored_points": future_ignored,
        "checkpoint_normalization_points": normalized,
    }
    for field, expected in expected_totals.items():
        if row_totals[field] != expected:
            _raise(
                f"eff_preparation_report.summary.{field} 与 EFF CSV 逐行合计不一致: "
                f"{expected} != {row_totals[field]}"
            )

    return {
        "formal_eff_ready": True,
        "raw_missing_points": raw_missing,
        "repaired_missing_points": repaired_missing,
        "unresolved_missing_points": unresolved_missing,
        "future_backfill_ignored_points": future_ignored,
        "checkpoint_normalization_points": normalized,
    }


def _load_evidence_index(
    path: Path,
    *,
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _iter_jsonl(path, label="evidence_jsonl")
    evidence_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_key = _parse_evidence_key(row, pred_identity_map=pred_identity_map)
        if logical_key in evidence_map:
            _raise(f"evidence_jsonl 出现重复 logical_key: {logical_key}")
        tree_similarity = _finite_unit_float(
            _metric_alias(row, ("tree", "tree_similarity"), ("tree_similarity",)),
            context=f"{logical_key}.tree_similarity",
        )
        variable_f1 = _finite_unit_float(
            _metric_alias(row, ("variable", "f1"), ("variable_f1",)),
            context=f"{logical_key}.variable_f1",
        )
        operator_f1 = _finite_unit_float(
            _metric_alias(row, ("operator", "f1"), ("operator_f1",)),
            context=f"{logical_key}.operator_f1",
        )
        evidence_map[logical_key] = {
            "logical_key": logical_key,
            "gt_logical_id": None,
            "pred_logical_id": _optional_nonempty_string(row.get("pred_logical_id")),
            "tree_similarity": tree_similarity,
            "variable_f1": variable_f1,
            "operator_f1": operator_f1,
            "evidence_hash": _sha256_string(row.get("evidence_hash"), context=f"{logical_key}.evidence_hash"),
            "gt_expression": _optional_nonempty_string(
                _metric_alias(
                    row,
                    ("ground_truth", "simplified_expression"),
                    ("gt_expression",),
                )
            ),
            "pred_expression": _optional_nonempty_string(
                _metric_alias(
                    row,
                    ("prediction", "simplified_expression"),
                    ("pred_expression",),
                )
            ),
            "gt_artifact_sha256": _optional_nonempty_string(
                _metric_alias(
                    row,
                    ("lhs_artifact", "artifact_sha256"),
                    ("ground_truth", "artifact_sha256"),
                    ("gt_artifact_sha256",),
                )
            ),
            "pred_artifact_sha256": _optional_nonempty_string(
                _metric_alias(
                    row,
                    ("rhs_artifact", "artifact_sha256"),
                    ("prediction", "artifact_sha256"),
                    ("pred_artifact_sha256",),
                )
            ),
        }
        gt_logical_id = _optional_nonempty_string(row.get("gt_logical_id"))
        if gt_logical_id is not None:
            evidence_map[logical_key]["gt_logical_id"] = gt_logical_id
            _, logical_key_dataset_id, _ = _parse_numeric_logical_key(logical_key)
            gt_dataset_id = _parse_gt_logical_id(gt_logical_id)
            if gt_dataset_id != logical_key_dataset_id:
                _raise(
                    f"{logical_key} deterministic evidence 的 gt_logical_id 与 logical_key.dataset_id 不一致"
                )
    return evidence_map, {"path": str(path.resolve()), "sha256": _sha256_file(path), "row_count": len(evidence_map)}


def _validate_manifest_jsonl_record(
    record: object,
    *,
    context: str,
    expected_path: Path | None = None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(record, Mapping):
        _raise(f"{context} 缺失")
    path = _resolve_report_path(record.get("path"), context=f"{context}.path")
    if expected_path is not None and path.resolve() != expected_path.resolve():
        _raise(f"{context}.path 未绑定当前聚合输入: {path} != {expected_path.resolve()}")
    if not path.is_file():
        _raise(f"{context}.path 不存在: {path}")
    declared_sha = _sha256_string(record.get("sha256"), context=f"{context}.sha256")
    actual_sha = _sha256_file(path)
    if declared_sha != actual_sha:
        _raise(f"{context}.sha256 与文件不一致")
    rows = _iter_jsonl(path, label=context)
    declared_rows = _parse_int(record.get("row_count"), context=f"{context}.row_count")
    if declared_rows != len(rows):
        _raise(f"{context}.row_count 与文件不一致: {declared_rows} != {len(rows)}")
    return path, rows, {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "row_count": len(rows),
    }


def _rows_by_logical_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        logical_id = _string(row.get("logical_id"), context=f"{context}.logical_id")
        if logical_id in indexed:
            _raise(f"{context} logical_id 重复: {logical_id}")
        indexed[logical_id] = dict(row)
    return indexed


def _validate_output_row_sha(row: Mapping[str, Any], *, context: str) -> None:
    declared = _sha256_string(row.get("output_row_sha256"), context=f"{context}.output_row_sha256")
    actual = _sha256_text(
        _canonical_json({key: value for key, value in row.items() if key != "output_row_sha256"})
    )
    if declared != actual:
        _raise(f"{context}.output_row_sha256 校验失败")


def _original_expression_from_plan(plan_row: Mapping[str, Any], *, logical_id: str) -> str:
    request = plan_row.get("request")
    if not isinstance(request, Mapping):
        _raise(f"{logical_id} plan.request 缺失")
    for field in ("original_expression", "expression"):
        expression = _optional_nonempty_string(request.get(field))
        if expression is not None:
            return expression
    _raise(f"{logical_id} plan 缺少原始表达式")
    raise AssertionError("unreachable")


def _validate_effective_views(
    *,
    rows: Sequence[Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    plan_rows: Mapping[str, Mapping[str, Any]],
    base_file_sha256: str,
    overlay_revision_sha256: str,
    context: str,
) -> dict[str, str]:
    views = _rows_by_logical_id(rows, context=context)
    if set(views) != set(base_rows):
        _raise(
            f"{context} 必须是 base index 的完整 successor view: "
            f"missing={sorted(set(base_rows) - set(views))} extra={sorted(set(views) - set(base_rows))}"
        )
    replacements: dict[str, str] = {}
    overlay_fields = {
        "base_file_sha256",
        "base_row_sha256",
        "overlay_action",
        "overlay_revision_sha256",
        "output_row_sha256",
    }
    audit_patch_fields = {
        "source_audit_logical_id",
        "source_final_record_sha256",
        "suggested_simplified_expression",
    }
    for logical_id, view in views.items():
        base = base_rows[logical_id]
        _validate_output_row_sha(view, context=f"{context}.{logical_id}")
        if view.get("base_file_sha256") != base_file_sha256:
            _raise(f"{context}.{logical_id}.base_file_sha256 漂移")
        expected_base_row_sha = _sha256_text(_canonical_json(base))
        if view.get("base_row_sha256") != expected_base_row_sha:
            _raise(f"{context}.{logical_id}.base_row_sha256 漂移")
        if view.get("overlay_revision_sha256") != overlay_revision_sha256:
            _raise(f"{context}.{logical_id}.overlay_revision_sha256 漂移")
        extra_fields = set(view) - set(base) - overlay_fields - audit_patch_fields
        if extra_fields:
            _raise(f"{context}.{logical_id} 出现未声明字段: {sorted(extra_fields)}")
        action = _string(view.get("overlay_action"), context=f"{context}.{logical_id}.overlay_action")
        if action not in {"unchanged", "replace_expression"}:
            _raise(f"{context}.{logical_id}.overlay_action 非法: {action}")
        present_audit_fields = set(view) & audit_patch_fields
        if action == "unchanged" and present_audit_fields:
            _raise(f"{context}.{logical_id} unchanged 行不应携带审计 patch 字段")
        mutable_fields = {"effective_expression", "expression_resolution"} if action == "replace_expression" else set()
        for field, base_value in base.items():
            if field not in mutable_fields and view.get(field) != base_value:
                _raise(f"{context}.{logical_id}.{field} 未授权漂移")
        if action == "unchanged":
            continue
        if view.get("expression_resolution") != AUDIT_FALLBACK_RESOLUTION:
            _raise(f"{context}.{logical_id}.expression_resolution 不是审计原式回退")
        _string(
            view.get("source_audit_logical_id"),
            context=f"{context}.{logical_id}.source_audit_logical_id",
        )
        _sha256_string(
            view.get("source_final_record_sha256"),
            context=f"{context}.{logical_id}.source_final_record_sha256",
        )
        suggested = view.get("suggested_simplified_expression")
        if suggested is not None and not isinstance(suggested, str):
            _raise(f"{context}.{logical_id}.suggested_simplified_expression 非法")
        plan = plan_rows.get(logical_id)
        if plan is None:
            _raise(f"{context}.{logical_id} 缺少 base plan")
        original = _original_expression_from_plan(plan, logical_id=logical_id)
        effective = _string(
            view.get("effective_expression"),
            context=f"{context}.{logical_id}.effective_expression",
        )
        if effective != original:
            _raise(f"{context}.{logical_id} 未回退到 plan 原始表达式")
        if effective == base.get("effective_expression"):
            _raise(f"{context}.{logical_id} 声明 replace_expression 但表达式未变化")
        replacements[logical_id] = effective
    return replacements


def _apply_audit_corrections(
    manifest_path: Path,
    *,
    base_paths: Mapping[str, Path],
    pred_identity_map: Mapping[tuple[str, str, int], dict[str, Any]],
    gt_rows: dict[str, dict[str, Any]],
    pred_rows: dict[str, dict[str, Any]],
    eq_rows: dict[str, dict[str, Any]],
    structure_rows: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    evidence_rows: dict[str, dict[str, Any]],
    formal_contract: bool,
) -> dict[str, Any]:
    manifest = _read_json_object(manifest_path, context="audit corrections manifest")
    if manifest.get("schema_version") != AUDIT_CORRECTIONS_SCHEMA_VERSION:
        _raise("audit corrections manifest.schema_version 非法")
    if manifest.get("phase") != "final" or manifest.get("status") != "ok":
        _raise("audit corrections manifest 必须是 final/ok")
    if manifest.get("condition") != NOISE_TAG:
        _raise("audit corrections manifest.condition 不是 clean")
    declared_overlay_sha = _sha256_string(
        manifest.get("overlay_sha256"), context="audit corrections manifest.overlay_sha256"
    )
    actual_overlay_sha = _sha256_text(
        _canonical_json({key: value for key, value in manifest.items() if key != "overlay_sha256"})
    )
    if declared_overlay_sha != actual_overlay_sha:
        _raise("audit corrections manifest.overlay_sha256 校验失败")
    overlay_revision = _sha256_string(
        manifest.get("overlay_revision_sha256"),
        context="audit corrections manifest.overlay_revision_sha256",
    )
    policy = manifest.get("policy")
    expected_policy = {
        "severity_allowlist": "all",
        "undetermined_policy": "retain",
        "abstention_recovery": "apply",
        "dependent_policy": "rebuild",
    }
    if not isinstance(policy, Mapping) or dict(policy) != expected_policy:
        _raise(f"audit corrections manifest.policy 必须为正式口径: {expected_policy}")

    base_manifest = manifest.get("base_inputs")
    if not isinstance(base_manifest, Mapping):
        _raise("audit corrections manifest.base_inputs 缺失")
    base_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    base_info: dict[str, dict[str, Any]] = {}
    for name, expected_path in base_paths.items():
        _, rows, info = _validate_manifest_jsonl_record(
            base_manifest.get(name),
            context=f"audit corrections manifest.base_inputs.{name}",
            expected_path=expected_path,
        )
        base_rows_by_name[name] = rows
        base_info[name] = info

    audit_path, audit_rows, audit_info = _validate_manifest_jsonl_record(
        manifest.get("audit_final"),
        context="audit corrections manifest.audit_final",
    )
    audit_base = base_manifest.get("audit_final")
    if not isinstance(audit_base, Mapping) or dict(audit_base) != dict(manifest["audit_final"]):
        _raise("audit corrections manifest.audit_final 与 base_inputs.audit_final 不一致")
    audit_by_id: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        audit_id = _string(row.get("audit_logical_id"), context="audit_final.audit_logical_id")
        if audit_id in audit_by_id:
            _raise(f"audit_final.audit_logical_id 重复: {audit_id}")
        declared = _sha256_string(
            row.get("final_record_sha256"), context=f"{audit_id}.final_record_sha256"
        )
        actual = _sha256_text(
            _canonical_json({key: value for key, value in row.items() if key != "final_record_sha256"})
        )
        if declared != actual:
            _raise(f"{audit_id}.final_record_sha256 校验失败")
        audit_by_id[audit_id] = row
    clean_audit_count = sum(1 for row in audit_rows if row.get("condition") == NOISE_TAG)
    if formal_contract and len(audit_rows) < MINIMUM_FORMAL_AUDIT_ROWS:
        _raise(
            f"正式公式抽查至少需要 {MINIMUM_FORMAL_AUDIT_ROWS} 条，实际为 {len(audit_rows)}"
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        _raise("audit corrections manifest.outputs 缺失")
    output_rows: dict[str, list[dict[str, Any]]] = {}
    output_info: dict[str, dict[str, Any]] = {}
    for name in (
        "corrections",
        "gt_effective",
        "pred_effective",
        "equivalence_overrides",
        "evidence_rebuilt",
        "structure_stale",
        "structure_overlay_plan",
        "structure_overlay_index",
    ):
        _, rows, info = _validate_manifest_jsonl_record(
            outputs.get(name), context=f"audit corrections manifest.outputs.{name}"
        )
        output_rows[name] = rows
        output_info[name] = info

    gt_base = _rows_by_logical_id(base_rows_by_name["gt_index"], context="base.gt_index")
    pred_base = _rows_by_logical_id(base_rows_by_name["pred_index"], context="base.pred_index")
    gt_plans = _rows_by_logical_id(base_rows_by_name["gt_plan"], context="base.gt_plan")
    pred_plans = _rows_by_logical_id(base_rows_by_name["pred_plan"], context="base.pred_plan")
    gt_replacements = _validate_effective_views(
        rows=output_rows["gt_effective"],
        base_rows=gt_base,
        plan_rows=gt_plans,
        base_file_sha256=base_info["gt_index"]["sha256"],
        overlay_revision_sha256=overlay_revision,
        context="audit.gt_effective",
    )
    pred_replacements = _validate_effective_views(
        rows=output_rows["pred_effective"],
        base_rows=pred_base,
        plan_rows=pred_plans,
        base_file_sha256=base_info["pred_index"]["sha256"],
        overlay_revision_sha256=overlay_revision,
        context="audit.pred_effective",
    )

    correction_targets: set[tuple[str, str]] = set()
    for correction in output_rows["corrections"]:
        if correction.get("schema_version") != "audit_corrections.overlay_row.v1":
            _raise("audit corrections row.schema_version 非法")
        if correction.get("condition") != NOISE_TAG:
            _raise("audit corrections row.condition 不是 clean")
        if correction.get("overlay_revision_sha256") != overlay_revision:
            _raise("audit corrections row.overlay_revision_sha256 漂移")
        source_id = _string(
            correction.get("source_audit_logical_id"), context="correction.source_audit_logical_id"
        )
        source = audit_by_id.get(source_id)
        if source is None or correction.get("source_final_record_sha256") != source.get("final_record_sha256"):
            _raise(f"correction 未绑定 audit final row: {source_id}")
        target_kind = _string(correction.get("target_kind"), context="correction.target_kind")
        target_id = _string(correction.get("target_logical_id"), context="correction.target_logical_id")
        action = _string(correction.get("action"), context="correction.action")
        base_row_sha = correction.get("base_row_sha256")
        expected_correction_id = _sha256_text(
            _canonical_json(
                {
                    "source_final_record_sha256": source["final_record_sha256"],
                    "target_logical_id": target_id,
                    "action": action,
                    "base_row_sha256": base_row_sha,
                }
            )
        )
        if correction.get("correction_id") != expected_correction_id:
            _raise(f"correction_id 校验失败: {target_id}")
        target = (target_kind, target_id)
        if target in correction_targets:
            _raise(f"correction target 重复: {target}")
        correction_targets.add(target)

    expected_expression_targets = {
        *(('gt', logical_id) for logical_id in gt_replacements),
        *(('pred', logical_id) for logical_id in pred_replacements),
    }
    actual_expression_targets = {
        target for target in correction_targets if target[0] in {"gt", "pred"}
    }
    if actual_expression_targets != expected_expression_targets:
        _raise("表达式 corrections 与 effective views 不一致")

    for logical_id, expression in gt_replacements.items():
        dataset_id = _parse_gt_logical_id(logical_id)
        current = gt_rows.get(dataset_id)
        if current is None or current["logical_id"] != logical_id:
            _raise(f"GT correction 无法绑定聚合索引: {logical_id}")
        try:
            artifact = build_symbolic_artifact(expression)
        except SymbolicEvidenceError as exc:
            raise AggregateCleanMetricsError(f"{logical_id} 审计回退表达式不可解析: {exc}") from exc
        current.update(expression=expression, artifact=artifact)

    changed_pred_keys: set[str] = set()
    for logical_id, expression in pred_replacements.items():
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            logical_id, expected_prefix="pred_simplify"
        )
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=logical_id,
        )
        logical_key = str(identity["numeric_logical_key"])
        current = pred_rows.get(logical_key)
        if current is None or current["logical_id"] != logical_id:
            _raise(f"pred correction 无法绑定聚合索引: {logical_id}")
        try:
            artifact = build_symbolic_artifact(expression)
        except SymbolicEvidenceError as exc:
            raise AggregateCleanMetricsError(f"{logical_id} 审计回退表达式不可解析: {exc}") from exc
        current.update(expression=expression, artifact=artifact, symbolic_valid=True)
        changed_pred_keys.add(logical_key)

    eq_base = _rows_by_logical_id(
        base_rows_by_name["equivalence_index"], context="base.equivalence_index"
    )
    eq_targets: set[tuple[str, str]] = set()
    for override in output_rows["equivalence_overrides"]:
        target_id = _string(override.get("target_logical_id"), context="equivalence_override.target")
        source_id = _string(
            override.get("source_audit_logical_id"), context=f"{target_id}.source_audit_logical_id"
        )
        source = audit_by_id.get(source_id)
        if source is None or override.get("source_final_record_sha256") != source.get("final_record_sha256"):
            _raise(f"equivalence override 未绑定 audit final row: {target_id}")
        base = eq_base.get(target_id)
        if base is None:
            _raise(f"equivalence override target 不存在: {target_id}")
        structured = base.get("structured_output")
        if not isinstance(structured, Mapping):
            _raise(f"equivalence override target 不是 frozen: {target_id}")
        before = _string(override.get("before_decision"), context=f"{target_id}.before_decision")
        after = _string(override.get("after_decision"), context=f"{target_id}.after_decision")
        if before != structured.get("decision") or after not in EQUIVALENCE_DECISIONS or after == before:
            _raise(f"equivalence override 决策链非法: {target_id}")
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            target_id, expected_prefix="equivalence"
        )
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=seed,
            logical_id=target_id,
        )
        logical_key = str(identity["numeric_logical_key"])
        current = eq_rows.get(logical_key)
        if current is None or current["logical_id"] != target_id or current["decision"] != before:
            _raise(f"equivalence override 无法绑定聚合索引: {target_id}")
        current["decision"] = after
        eq_targets.add(("equivalence", target_id))
    actual_eq_targets = {target for target in correction_targets if target[0] == "equivalence"}
    if actual_eq_targets != eq_targets or len(correction_targets) != len(expected_expression_targets) + len(eq_targets):
        _raise("equivalence corrections 与 overrides 不一致")

    raw_evidence_by_key = {
        _parse_evidence_key(row, pred_identity_map=pred_identity_map): row
        for row in base_rows_by_name["evidence"]
    }
    rebuilt_path = Path(output_info["evidence_rebuilt"]["path"])
    rebuilt_map, _ = _load_evidence_index(rebuilt_path, pred_identity_map=pred_identity_map)
    changed_gt_datasets = {_parse_gt_logical_id(logical_id) for logical_id in gt_replacements}
    affected_evidence_keys = {
        logical_key
        for logical_key, pred in pred_rows.items()
        if pred["symbolic_valid"]
        and _parse_numeric_logical_key(logical_key)[1] in changed_gt_datasets
    } | changed_pred_keys
    if set(rebuilt_map) != affected_evidence_keys:
        _raise(
            "evidence rebuilt 影响域不闭合: "
            f"missing={sorted(affected_evidence_keys - set(rebuilt_map))} "
            f"extra={sorted(set(rebuilt_map) - affected_evidence_keys)}"
        )
    for row in output_rows["evidence_rebuilt"]:
        logical_key = _parse_evidence_key(row, pred_identity_map=pred_identity_map)
        _validate_output_row_sha(row, context=f"evidence_rebuilt.{logical_key}")
        if row.get("overlay_revision_sha256") != overlay_revision:
            _raise(f"evidence_rebuilt.{logical_key}.overlay_revision_sha256 漂移")
        base_evidence = raw_evidence_by_key.get(logical_key)
        expected_base_sha = _sha256_text(_canonical_json(base_evidence)) if base_evidence is not None else None
        if row.get("base_evidence_row_sha256") != expected_base_sha:
            _raise(f"evidence_rebuilt.{logical_key}.base_evidence_row_sha256 漂移")
    evidence_rows.update(rebuilt_map)

    expected_structure_ids: set[str] = set()
    for logical_id in pred_replacements:
        algorithm_slug, dataset_index, seed = _parse_run_logical_id(
            logical_id, expected_prefix="pred_simplify"
        )
        for left, right in SEED_PAIRS:
            if seed in {left, right}:
                expected_structure_ids.add(
                    f"stab_structure::{algorithm_slug}::{dataset_index}::s{left}-s{right}"
                )
    manifest_structure_ids = {
        _string(value, context="audit corrections manifest.structure_ids[]")
        for value in manifest.get("structure_ids", [])
    }
    overlay_structure_ids = {
        _string(row.get("logical_id"), context="structure_overlay.logical_id")
        for row in output_rows["structure_overlay_index"]
    }
    if manifest_structure_ids != expected_structure_ids or overlay_structure_ids != expected_structure_ids:
        _raise("structure overlay 影响域与 pred expression corrections 不一致")
    structure_result_bindings = manifest.get("structure_results")
    if not isinstance(structure_result_bindings, Mapping):
        _raise("audit corrections manifest.structure_results 缺失")
    for row in output_rows["structure_overlay_index"]:
        logical_id = _string(row.get("logical_id"), context="structure_overlay.logical_id")
        if row.get("overlay_revision_sha256") != overlay_revision:
            _raise(f"structure overlay revision 漂移: {logical_id}")
        if row.get("state") != "frozen" or row.get("task_type") != "stab_structure":
            _raise(f"structure overlay 终态非法: {logical_id}")
        structured = row.get("structured_output")
        if not isinstance(structured, Mapping):
            _raise(f"structure overlay structured_output 缺失: {logical_id}")
        decision = _string(structured.get("decision"), context=f"{logical_id}.decision")
        if decision not in STRUCTURE_DECISIONS:
            _raise(f"structure overlay decision 非法: {logical_id}")
        binding = structure_result_bindings.get(logical_id)
        if not isinstance(binding, Mapping):
            _raise(f"structure result binding 缺失: {logical_id}")
        if (
            row.get("evaluation_key") != binding.get("evaluation_key")
            or row.get("result_container_sha256") != binding.get("result_container_sha256")
            or _sha256_text(_canonical_json(structured)) != binding.get("structured_output_sha256")
        ):
            _raise(f"structure overlay result binding 漂移: {logical_id}")
        algorithm_slug, dataset_index, pair = _parse_structure_logical_id(logical_id)
        identity = _resolve_pred_identity(
            pred_identity_map,
            algorithm_slug=algorithm_slug,
            dataset_index=dataset_index,
            seed=pair[0],
            logical_id=logical_id,
        )
        key = (str(identity["algorithm"]), str(identity["dataset_id"]), pair)
        if key not in structure_rows:
            _raise(f"structure overlay 无法绑定聚合索引: {logical_id}")
        structure_rows[key] = {"logical_id": logical_id, "state": "frozen", "decision": decision}

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        _raise("audit corrections manifest.counts 缺失")
    expected_counts = {
        "audit_clean_rows": clean_audit_count,
        "gt_expression_fallbacks": len(gt_replacements),
        "pred_expression_fallbacks": len(pred_replacements),
        "expression_fallbacks": len(gt_replacements) + len(pred_replacements),
        "reference_overrides": len(eq_targets),
        "evidence_rebuilds": len(rebuilt_map),
        "stale_structure": len(expected_structure_ids),
        "structure_replacements": len(overlay_structure_ids),
    }
    for field, expected in expected_counts.items():
        actual = _parse_int(counts.get(field), context=f"audit corrections manifest.counts.{field}")
        if actual != expected:
            _raise(f"audit corrections manifest.counts.{field} 不一致: {actual} != {expected}")

    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256_file(manifest_path),
        "schema_version": AUDIT_CORRECTIONS_SCHEMA_VERSION,
        "status": "ok",
        "overlay_sha256": declared_overlay_sha,
        "overlay_revision_sha256": overlay_revision,
        "audit_final": audit_info,
        "counts": expected_counts,
        "outputs": output_info,
    }

def _metric_alias(row: Mapping[str, Any], *candidates: tuple[str, ...]) -> object:
    current: object = row
    for candidate in candidates:
        current = row
        ok = True
        for part in candidate:
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    return None


def _extract_eff_trajectory(row: Mapping[str, str], *, logical_key: str) -> list[float]:
    trajectory: list[float] = []
    for field in EFF_QUALITY_FIELDS:
        trajectory.append(_finite_unit_float(row.get(field), context=f"{logical_key}.{field}"))
    return trajectory


def _validate_run_distribution(
    numeric_rows: Mapping[str, Mapping[str, str]],
    *,
    expected_algorithms: int,
    expected_datasets: int,
) -> dict[str, set[str]]:
    grouped_algorithms: dict[str, set[str]] = {}
    grouped_tasks: dict[tuple[str, str], set[int]] = {}
    for row in numeric_rows.values():
        algorithm = str(row["algorithm"])
        dataset_id = str(row["dataset_id"])
        seed = int(row["seed"])
        grouped_algorithms.setdefault(algorithm, set()).add(dataset_id)
        grouped_tasks.setdefault((algorithm, dataset_id), set()).add(seed)
    if len(grouped_algorithms) != expected_algorithms:
        _raise(f"算法数应为 {expected_algorithms}，实际为 {len(grouped_algorithms)}")
    all_datasets = {dataset for datasets in grouped_algorithms.values() for dataset in datasets}
    if len(all_datasets) != expected_datasets:
        _raise(f"数据集数应为 {expected_datasets}，实际为 {len(all_datasets)}")
    for algorithm, datasets in sorted(grouped_algorithms.items()):
        if len(datasets) != expected_datasets:
            _raise(f"{algorithm} 的数据集数应为 {expected_datasets}，实际为 {len(datasets)}")
    for task_key, seeds in sorted(grouped_tasks.items()):
        if tuple(sorted(seeds)) != SEEDS:
            _raise(f"{task_key} 的 seed 集合必须是 {SEEDS}，实际为 {tuple(sorted(seeds))}")
    return grouped_algorithms


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _reportable_frozen_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """移除仅供本次校验使用、体积巨大的 plan request 内存索引。"""

    return {
        key: value
        for key, value in info.items()
        if key != "plan_requests_by_logical_id"
    }


def aggregate_clean_metrics(
    *,
    numeric_csv: Path,
    clean_numeric_preparation_report_json: Path,
    eff_csv: Path,
    eff_preparation_report_json: Path,
    gt_plan_jsonl: Path,
    gt_summary_json: Path,
    gt_index_jsonl: Path,
    pred_plan_jsonl: Path,
    pred_summary_json: Path,
    pred_index_jsonl: Path,
    equivalence_plan_jsonl: Path,
    equivalence_summary_json: Path,
    equivalence_index_jsonl: Path,
    structure_plan_jsonl: Path,
    structure_summary_json: Path,
    structure_index_jsonl: Path,
    evidence_jsonl: Path,
    clean_run_csv: Path,
    task_stability_csv: Path,
    algorithm_csv: Path,
    report_json: Path,
    audit_corrections_manifest_json: Path | None = None,
    expected_runs: int = DEFAULT_EXPECTED_RUNS,
    expected_algorithms: int = DEFAULT_EXPECTED_ALGORITHMS,
    expected_datasets: int = DEFAULT_EXPECTED_DATASETS,
) -> dict[str, Any]:
    audit_corrections_manifest_json = _validate_audit_corrections_requirement(
        audit_corrections_manifest_json,
        expected_runs=expected_runs,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
    )
    formal_contract = (
        expected_runs == DEFAULT_EXPECTED_RUNS
        and expected_algorithms == DEFAULT_EXPECTED_ALGORITHMS
        and expected_datasets == DEFAULT_EXPECTED_DATASETS
    )
    expected_task_rows = expected_algorithms * expected_datasets
    expected_structure_rows = expected_task_rows * len(SEED_PAIRS)
    pred_identity_map, simplify_plan_info = _load_pred_simplify_identity_map(
        pred_plan_jsonl,
        expected_runs=expected_runs,
    )

    numeric_rows, numeric_info = _read_csv_by_logical_key(
        numeric_csv,
        required_fields={
            "logical_key",
            "algorithm",
            "dataset_id",
            "seed",
            "noise_tag",
            "task_id",
            "host",
            "evaluation_status",
            "valid_output",
            "id_quality",
            "ood_quality",
        },
        label="clean_numeric_run_metrics.csv",
        expected_runs=expected_runs,
    )
    numeric_preparation_report_info = _validate_clean_numeric_preparation_report(
        clean_numeric_preparation_report_json,
        numeric_csv=numeric_csv,
        numeric_rows=numeric_rows,
    )
    eff_rows, eff_info = _read_csv_by_logical_key(
        eff_csv,
        required_fields={
            "logical_key",
            "algorithm",
            "dataset_id",
            "seed",
            "noise_tag",
            "task_id",
            "host",
            "best_quality",
            "m_eff",
            "audited_repair_points",
            "future_backfill_ignored_points",
            "checkpoint_normalization_points",
            "bundle_sha256",
            "bundle_report_sha256",
            "freeze_binding_report_sha256",
            "repair_manifest_sha256",
            *EFF_QUALITY_FIELDS,
        },
        label="clean_eff_run_metrics.csv",
        expected_runs=expected_runs,
    )
    eff_report_info = _validate_eff_preparation_report(
        eff_preparation_report_json,
        eff_csv=eff_csv,
        eff_rows=eff_rows,
    )
    gt_index_rows, gt_info = _load_frozen_index_rows(
        index_path=gt_index_jsonl,
        summary_path=gt_summary_json,
        plan_path=gt_plan_jsonl,
        expected_task_type="gt_simplify",
        label="gt_frozen_index",
    )
    gt_rows = _build_gt_index(
        gt_index_rows,
        gt_plan_requests_by_logical_id=gt_info["plan_requests_by_logical_id"],
        expected_datasets=expected_datasets,
    )
    pred_index_rows, pred_info = _load_frozen_index_rows(
        index_path=pred_index_jsonl,
        summary_path=pred_summary_json,
        plan_path=pred_plan_jsonl,
        expected_task_type="pred_simplify",
        label="pred_frozen_index",
        allow_exhausted=True,
    )
    pred_rows = _build_pred_index(
        pred_index_rows,
        pred_identity_map=pred_identity_map,
        expected_runs=expected_runs,
    )
    pred_exhausted_count = sum(1 for row in pred_rows.values() if row["state"] == "exhausted")
    if pred_exhausted_count:
        _raise(
            "pred_frozen_index 仍包含 exhausted 终态，clean 正式六轴聚合禁止继续；"
            f"请先完成补冻或重验证闭环（exhausted={pred_exhausted_count}）"
        )
    eq_index_rows, eq_info = _load_frozen_index_rows(
        index_path=equivalence_index_jsonl,
        summary_path=equivalence_summary_json,
        plan_path=equivalence_plan_jsonl,
        expected_task_type="equivalence",
        label="equivalence_frozen_index",
    )
    eq_rows = _build_equivalence_index(
        eq_index_rows,
        pred_identity_map=pred_identity_map,
        expected_runs=expected_runs,
    )
    structure_index_rows, structure_info = _load_frozen_index_rows(
        index_path=structure_index_jsonl,
        summary_path=structure_summary_json,
        plan_path=structure_plan_jsonl,
        expected_task_type="stab_structure",
        label="structure_frozen_index",
    )
    structure_rows = _build_structure_index(
        structure_index_rows,
        pred_identity_map=pred_identity_map,
        expected_rows=expected_structure_rows,
    )
    evidence_rows, evidence_info = _load_evidence_index(
        evidence_jsonl,
        pred_identity_map=pred_identity_map,
    )

    _validate_run_distribution(
        numeric_rows,
        expected_algorithms=expected_algorithms,
        expected_datasets=expected_datasets,
    )
    if set(numeric_rows) != set(eff_rows):
        _raise("numeric 与 eff 的 logical_key 集合不一致")
    if set(numeric_rows) != set(pred_rows):
        _raise("numeric 与 pred frozen index 的 logical_key 集合不一致")
    if set(numeric_rows) != set(eq_rows):
        _raise("numeric 与 equivalence frozen index 的 logical_key 集合不一致")
    audit_corrections_info = None
    if audit_corrections_manifest_json is not None:
        audit_corrections_info = _apply_audit_corrections(
            audit_corrections_manifest_json,
            base_paths={
                "gt_plan": gt_plan_jsonl,
                "gt_index": gt_index_jsonl,
                "pred_plan": pred_plan_jsonl,
                "pred_index": pred_index_jsonl,
                "equivalence_plan": equivalence_plan_jsonl,
                "equivalence_index": equivalence_index_jsonl,
                "structure_plan": structure_plan_jsonl,
                "structure_index": structure_index_jsonl,
                "evidence": evidence_jsonl,
            },
            pred_identity_map=pred_identity_map,
            gt_rows=gt_rows,
            pred_rows=pred_rows,
            eq_rows=eq_rows,
            structure_rows=structure_rows,
            evidence_rows=evidence_rows,
            formal_contract=formal_contract,
        )
    expected_evidence_keys = {logical_key for logical_key, row in pred_rows.items() if row["symbolic_valid"]}
    if set(evidence_rows) != expected_evidence_keys:
        missing = sorted(expected_evidence_keys - set(evidence_rows))
        extra = sorted(set(evidence_rows) - expected_evidence_keys)
        _raise(f"evidence_jsonl logical_key 集合不闭合: missing={missing} extra={extra}")

    clean_run_rows: list[dict[str, Any]] = []
    artifact_recanonicalizations: set[tuple[str, str, str, str]] = set()
    metric_recanonicalizations: set[tuple[str, str]] = set()
    run_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for logical_key in sorted(numeric_rows, key=lambda item: (
        numeric_rows[item]["algorithm"],
        numeric_rows[item]["dataset_id"],
        int(numeric_rows[item]["seed"]),
    )):
        numeric = numeric_rows[logical_key]
        eff = eff_rows[logical_key]
        pred = pred_rows[logical_key]
        equivalence = eq_rows[logical_key]

        for field in ("algorithm", "dataset_id", "seed", "task_id", "host", "noise_tag"):
            if str(numeric[field]) != str(eff[field]):
                _raise(f"{logical_key} numeric 与 eff 的 {field} 不一致")

        algorithm = str(numeric["algorithm"])
        dataset_id = str(numeric["dataset_id"])
        seed = int(numeric["seed"])
        gt = gt_rows.get(dataset_id)
        if gt is None:
            _raise(f"{logical_key} 缺少 GT frozen index")

        valid_output = _parse_bool(numeric["valid_output"], context=f"{logical_key}.valid_output")
        id_quality = _finite_unit_float(numeric["id_quality"], context=f"{logical_key}.id_quality")
        ood_quality = _finite_unit_float(numeric["ood_quality"], context=f"{logical_key}.ood_quality")
        if not valid_output and (id_quality != 0.0 or ood_quality != 0.0):
            _raise(f"{logical_key} 无效输出时 ID/OOD 质量必须为 0")
        m_eff = _finite_unit_float(eff["m_eff"], context=f"{logical_key}.m_eff")
        quality_trajectory = _extract_eff_trajectory(eff, logical_key=logical_key)
        recomputed_m_eff = efficiency_from_qualities(quality_trajectory, horizon=EFF_HORIZON)
        if not math.isclose(m_eff, recomputed_m_eff, rel_tol=1e-12, abs_tol=1e-12):
            _raise(
                f"{logical_key} 的 m_eff 与 180 个 q 点重算结果不一致: "
                f"{m_eff!r} != {recomputed_m_eff!r}"
            )
        best_quality = _finite_unit_float(eff["best_quality"], context=f"{logical_key}.best_quality")
        recomputed_best_quality = max(quality_trajectory, default=0.0)
        if not math.isclose(best_quality, recomputed_best_quality, rel_tol=1e-12, abs_tol=1e-12):
            _raise(
                f"{logical_key} 的 best_quality 与 180 个 q 点不一致: "
                f"{best_quality!r} != {recomputed_best_quality!r}"
            )

        symbolic_valid = bool(pred["symbolic_valid"])
        reference_complexity = int(gt["artifact"]["node_count"])
        predicted_complexity = 0
        equivalence_decision = "non_applicable"
        tree_similarity = 0.0
        variable_f1 = 0.0
        operator_f1 = 0.0

        if symbolic_valid:
            artifact_was_recanonicalized = False
            evidence = evidence_rows.get(logical_key)
            if evidence is None:
                _raise(f"{logical_key} 缺少 deterministic evidence")
            if evidence["gt_logical_id"] != gt["logical_id"]:
                _raise(f"{logical_key} deterministic evidence 的 gt_logical_id 漂移")
            if evidence["pred_logical_id"] != pred["logical_id"]:
                _raise(f"{logical_key} deterministic evidence 的 pred_logical_id 漂移")
            if equivalence["state"] != "frozen":
                _raise(f"{logical_key} 有可用预测式时 equivalence 不得缺失")
            assert pred["artifact"] is not None
            if evidence["gt_expression"] is not None and evidence["gt_expression"] != gt["expression"]:
                _raise(f"{logical_key} deterministic evidence 的 GT expression 漂移")
            if evidence["pred_expression"] is not None and evidence["pred_expression"] != pred["expression"]:
                _raise(f"{logical_key} deterministic evidence 的 pred expression 漂移")
            # SymPy 版本可能改变等价表达式的 canonical tree 序列化。前面已要求
            # source expression 字节一致，后面还会逐项离线复算 T/V/O；因此这里
            # 将仅哈希变化记为可审计的 recanonicalization，而不误判成公式漂移。
            if evidence["gt_artifact_sha256"] is not None and evidence["gt_artifact_sha256"] != gt["artifact"]["artifact_sha256"]:
                artifact_was_recanonicalized = True
                artifact_recanonicalizations.add(
                    (
                        "gt",
                        str(gt["logical_id"]),
                        str(evidence["gt_artifact_sha256"]),
                        str(gt["artifact"]["artifact_sha256"]),
                    )
                )
            if evidence["pred_artifact_sha256"] is not None and evidence["pred_artifact_sha256"] != pred["artifact"]["artifact_sha256"]:
                artifact_was_recanonicalized = True
                artifact_recanonicalizations.add(
                    (
                        "pred",
                        str(pred["logical_id"]),
                        str(evidence["pred_artifact_sha256"]),
                        str(pred["artifact"]["artifact_sha256"]),
                    )
                )
            if not any(
                (
                    evidence["evidence_hash"],
                    evidence["gt_expression"],
                    evidence["pred_expression"],
                    evidence["gt_artifact_sha256"],
                    evidence["pred_artifact_sha256"],
                )
            ):
                _raise(f"{logical_key} deterministic evidence 缺少表达式或 hash 绑定")
            predicted_complexity = int(pred["artifact"]["node_count"])
            equivalence_decision = str(equivalence["decision"])
            expected_tree_similarity = tree_similarity_metric(gt["artifact"], pred["artifact"])
            expected_variable_f1 = variable_f1_metric(gt["artifact"], pred["artifact"])
            expected_operator_f1 = operator_f1_metric(gt["artifact"], pred["artifact"])
            if not math.isclose(float(evidence["tree_similarity"]), expected_tree_similarity, rel_tol=1e-12, abs_tol=1e-12):
                if not artifact_was_recanonicalized:
                    _raise(f"{logical_key} deterministic evidence 的 tree_similarity 无法离线复算")
                metric_recanonicalizations.add((logical_key, "tree_similarity"))
            if not math.isclose(float(evidence["variable_f1"]), expected_variable_f1, rel_tol=1e-12, abs_tol=1e-12):
                if not artifact_was_recanonicalized:
                    _raise(f"{logical_key} deterministic evidence 的 variable_f1 无法离线复算")
                metric_recanonicalizations.add((logical_key, "variable_f1"))
            if not math.isclose(float(evidence["operator_f1"]), expected_operator_f1, rel_tol=1e-12, abs_tol=1e-12):
                if not artifact_was_recanonicalized:
                    _raise(f"{logical_key} deterministic evidence 的 operator_f1 无法离线复算")
                metric_recanonicalizations.add((logical_key, "operator_f1"))
            tree_similarity = expected_tree_similarity
            variable_f1 = expected_variable_f1
            operator_f1 = expected_operator_f1
        else:
            if equivalence["state"] != "non_applicable":
                _raise(f"{logical_key} 无可用预测式时 equivalence 必须是 non_applicable")
            if logical_key in evidence_rows:
                _raise(f"{logical_key} 无可用预测式时不应存在 deterministic evidence")

        equivalent = equivalence_decision == "equivalent"
        m_sym = symbolic_fidelity_score(
            equivalent=equivalent,
            tree_similarity=tree_similarity,
            variable_f1=variable_f1,
            operator_f1=operator_f1,
            valid=symbolic_valid,
        )
        m_min = minimality_score(
            reference_complexity,
            predicted_complexity if predicted_complexity > 0 else 1,
            valid=symbolic_valid,
        )

        row = {
            "logical_key": logical_key,
            "algorithm": algorithm,
            "dataset_id": dataset_id,
            "seed": seed,
            "noise_tag": NOISE_TAG,
            "task_id": str(numeric["task_id"]),
            "host": str(numeric["host"]),
            "valid_output": str(valid_output).lower(),
            "pred_state": str(pred["state"]),
            "equivalence_state": str(equivalence["state"]),
            "id_quality": f"{id_quality:.17g}",
            "ood_quality": f"{ood_quality:.17g}",
            "m_eff": f"{m_eff:.17g}",
            "equivalence_decision": equivalence_decision,
            "equivalent": str(equivalent).lower(),
            "tree_similarity": f"{tree_similarity:.17g}",
            "variable_f1": f"{variable_f1:.17g}",
            "operator_f1": f"{operator_f1:.17g}",
            "m_sym": f"{m_sym:.17g}",
            "reference_complexity": reference_complexity,
            "predicted_complexity": predicted_complexity,
            "m_min": f"{m_min:.17g}",
            "gt_logical_id": gt["logical_id"],
            "pred_logical_id": str(pred["logical_id"]),
            "equivalence_logical_id": str(equivalence["logical_id"]),
        }
        clean_run_rows.append(row)
        run_grouped.setdefault((algorithm, dataset_id), []).append(row)

    if len(clean_run_rows) != expected_runs:
        _raise(f"clean_run_metrics.csv 行数应为 {expected_runs}，实际为 {len(clean_run_rows)}")

    task_rows: list[dict[str, Any]] = []
    for task_key in sorted(run_grouped):
        algorithm, dataset_id = task_key
        rows = sorted(run_grouped[task_key], key=lambda item: int(item["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            _raise(f"{task_key} 的 run seed 排列不完整")
        run_qualities = [
            RunQuality(
                id_quality=float(row["id_quality"]),
                ood_quality=float(row["ood_quality"]),
                valid=_parse_bool(row["valid_output"], context=f"{row['logical_key']}.valid_output"),
            )
            for row in rows
        ]
        pair_results: list[bool] = []
        pair_flags: dict[tuple[int, int], str] = {}
        symbolic_valid_by_seed = {
            int(row["seed"]): row["pred_state"] == "frozen" and int(row["predicted_complexity"]) > 0
            for row in rows
        }
        valid_output_by_seed = {
            int(row["seed"]): _parse_bool(row["valid_output"], context=f"{row['logical_key']}.valid_output")
            for row in rows
        }
        for pair in SEED_PAIRS:
            structure = structure_rows.get((algorithm, dataset_id, pair))
            if structure is None:
                _raise(f"{algorithm}::{dataset_id} 缺少 structure pair {pair}")
            if structure["state"] == "frozen":
                if not (valid_output_by_seed[pair[0]] and valid_output_by_seed[pair[1]]):
                    _raise(
                        f"{algorithm}::{dataset_id} pair {pair} frozen structure 必须绑定两侧 valid_output=true"
                    )
                decision = str(structure["decision"])
                consistent = decision in {"mathematically_equivalent", "same_canonical_structure"}
                if not (symbolic_valid_by_seed[pair[0]] and symbolic_valid_by_seed[pair[1]]):
                    _raise(
                        f"{algorithm}::{dataset_id} pair {pair} 在无可用符号输出时不应出现 frozen structure"
                    )
            else:
                if structure["state"] != "non_applicable":
                    _raise(f"{algorithm}::{dataset_id} pair {pair} 的 structure state 非法")
                if symbolic_valid_by_seed[pair[0]] and symbolic_valid_by_seed[pair[1]] and valid_output_by_seed[pair[0]] and valid_output_by_seed[pair[1]]:
                    _raise(
                        f"{algorithm}::{dataset_id} pair {pair} 结构裁决缺失但两侧均有效"
                    )
                decision = "non_applicable"
                consistent = False
            pair_results.append(consistent)
            pair_flags[pair] = decision

        stab = stability_score(run_qualities, structural_pair_results=pair_results)
        task_rows.append(
            {
                "algorithm": algorithm,
                "dataset_id": dataset_id,
                "seed_520_valid_output": str(valid_output_by_seed[520]).lower(),
                "seed_521_valid_output": str(valid_output_by_seed[521]).lower(),
                "seed_522_valid_output": str(valid_output_by_seed[522]).lower(),
                "pair_520_521": pair_flags[(520, 521)],
                "pair_520_522": pair_flags[(520, 522)],
                "pair_521_522": pair_flags[(521, 522)],
                "numerical_consistency": f"{stab.numerical_consistency:.17g}",
                "validity": f"{stab.validity:.17g}",
                "structural_consistency": f"{stab.structural_consistency:.17g}",
                "m_stab": f"{stab.score:.17g}",
            }
        )

    if len(task_rows) != expected_task_rows:
        _raise(f"task_stability.csv 行数应为 {expected_task_rows}，实际为 {len(task_rows)}")

    algorithm_groups: dict[str, list[dict[str, Any]]] = {}
    for row in clean_run_rows:
        algorithm_groups.setdefault(str(row["algorithm"]), []).append(row)
    algorithm_task_groups: dict[str, list[dict[str, Any]]] = {}
    for row in task_rows:
        algorithm_task_groups.setdefault(str(row["algorithm"]), []).append(row)

    if len(algorithm_groups) != expected_algorithms:
        _raise(f"algorithm_six_axis.csv 算法数应为 {expected_algorithms}，实际为 {len(algorithm_groups)}")

    algorithm_rows: list[dict[str, Any]] = []
    expected_runs_per_algorithm = expected_datasets * len(SEEDS)
    for algorithm in sorted(algorithm_groups):
        run_rows = sorted(algorithm_groups[algorithm], key=lambda row: (str(row["dataset_id"]), int(row["seed"])))
        task_group = sorted(algorithm_task_groups.get(algorithm, []), key=lambda row: str(row["dataset_id"]))
        if len(run_rows) != expected_runs_per_algorithm:
            _raise(f"{algorithm} 的 run 数应为 {expected_runs_per_algorithm}，实际为 {len(run_rows)}")
        if len(task_group) != expected_datasets:
            _raise(f"{algorithm} 的 task 数应为 {expected_datasets}，实际为 {len(task_group)}")
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "run_count": len(run_rows),
                "task_count": len(task_group),
                "id_score": f"{100.0 * sum(float(row['id_quality']) for row in run_rows) / len(run_rows):.17g}",
                "ood_score": f"{100.0 * sum(float(row['ood_quality']) for row in run_rows) / len(run_rows):.17g}",
                "sym_score": f"{100.0 * sum(float(row['m_sym']) for row in run_rows) / len(run_rows):.17g}",
                "min_score": f"{100.0 * sum(float(row['m_min']) for row in run_rows) / len(run_rows):.17g}",
                "eff_score": f"{100.0 * sum(float(row['m_eff']) for row in run_rows) / len(run_rows):.17g}",
                "stab_score": f"{100.0 * sum(float(row['m_stab']) for row in task_group) / len(task_group):.17g}",
            }
        )

    clean_run_fields = [
        "logical_key",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "host",
        "valid_output",
        "pred_state",
        "equivalence_state",
        "id_quality",
        "ood_quality",
        "m_eff",
        "equivalence_decision",
        "equivalent",
        "tree_similarity",
        "variable_f1",
        "operator_f1",
        "m_sym",
        "reference_complexity",
        "predicted_complexity",
        "m_min",
        "gt_logical_id",
        "pred_logical_id",
        "equivalence_logical_id",
    ]
    task_fields = [
        "algorithm",
        "dataset_id",
        "seed_520_valid_output",
        "seed_521_valid_output",
        "seed_522_valid_output",
        "pair_520_521",
        "pair_520_522",
        "pair_521_522",
        "numerical_consistency",
        "validity",
        "structural_consistency",
        "m_stab",
    ]
    algorithm_fields = [
        "algorithm",
        "run_count",
        "task_count",
        "id_score",
        "ood_score",
        "sym_score",
        "min_score",
        "eff_score",
        "stab_score",
    ]

    _write_csv_atomic(clean_run_csv, clean_run_fields, clean_run_rows)
    _write_csv_atomic(task_stability_csv, task_fields, task_rows)
    _write_csv_atomic(algorithm_csv, algorithm_fields, algorithm_rows)

    outputs = {
        "clean_run_metrics_csv": {
            "path": str(clean_run_csv.resolve()),
            "sha256": _sha256_file(clean_run_csv),
            "row_count": len(clean_run_rows),
        },
        "task_stability_csv": {
            "path": str(task_stability_csv.resolve()),
            "sha256": _sha256_file(task_stability_csv),
            "row_count": len(task_rows),
        },
        "algorithm_six_axis_csv": {
            "path": str(algorithm_csv.resolve()),
            "sha256": _sha256_file(algorithm_csv),
            "row_count": len(algorithm_rows),
        },
    }
    summary = {
        "run_row_count": len(clean_run_rows),
        "task_row_count": len(task_rows),
        "algorithm_row_count": len(algorithm_rows),
        "expected_runs": expected_runs,
        "expected_task_rows": expected_task_rows,
        "expected_algorithm_rows": expected_algorithms,
        "judge_exhausted_count": pred_exhausted_count,
        "artifact_recanonicalization_count": len(artifact_recanonicalizations),
        "metric_recanonicalization_count": len(metric_recanonicalizations),
        "formal_clean_ready": formal_contract and audit_corrections_info is not None,
        "equivalence_decision_counts": {
            decision: sum(1 for row in clean_run_rows if row["equivalence_decision"] == decision)
            for decision in ("equivalent", "not_equivalent", "undetermined", "non_applicable")
        },
    }
    gt_report_info = _reportable_frozen_info(gt_info)
    pred_report_info = _reportable_frozen_info(pred_info)
    eq_report_info = _reportable_frozen_info(eq_info)
    structure_report_info = _reportable_frozen_info(structure_info)
    diagnostics = {
        "runtime": {
            "python_version": platform.python_version(),
            "sympy_version": sympy.__version__,
        },
        "symbolic_recanonicalization": {
            "artifact_bindings": [
                {
                    "side": side,
                    "logical_id": logical_id,
                    "stored_sha256": stored_sha,
                    "current_sha256": current_sha,
                }
                for side, logical_id, stored_sha, current_sha in sorted(artifact_recanonicalizations)
            ],
            "metric_bindings": [
                {"logical_key": logical_key, "metric": metric}
                for logical_key, metric in sorted(metric_recanonicalizations)
            ],
        },
    }
    summary_sha256 = _sha256_text(_canonical_json({"inputs": {
        "numeric_csv": numeric_info,
        "clean_numeric_preparation_report_json": numeric_preparation_report_info,
        "eff_csv": eff_info,
        "pred_plan_jsonl": simplify_plan_info,
        "eff_preparation_report_json": eff_report_info,
        "gt_frozen_index": gt_report_info,
        "pred_frozen_index": pred_report_info,
        "equivalence_frozen_index": eq_report_info,
        "structure_frozen_index": structure_report_info,
        "evidence_jsonl": evidence_info,
        "audit_corrections_manifest_json": audit_corrections_info,
    }, "outputs": outputs, "summary": summary, "diagnostics": diagnostics}))
    report = {
        "status": "ok",
        "contract_ok": True,
        "formal_clean_ready": summary["formal_clean_ready"],
        "inputs": {
            "numeric_csv": numeric_info,
            "clean_numeric_preparation_report_json": numeric_preparation_report_info,
            "eff_csv": eff_info,
            "pred_plan_jsonl": simplify_plan_info,
            "eff_preparation_report_json": eff_report_info,
            "gt_frozen_index": gt_report_info,
            "pred_frozen_index": pred_report_info,
            "equivalence_frozen_index": eq_report_info,
            "structure_frozen_index": structure_report_info,
            "evidence_jsonl": evidence_info,
            "audit_corrections_manifest_json": audit_corrections_info,
        },
        "outputs": outputs,
        "summary": summary,
        "diagnostics": diagnostics,
        "summary_sha256": summary_sha256,
    }
    _write_json_atomic(report_json, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    stage5_root = _stage5_root()
    parser = argparse.ArgumentParser(description="聚合 Stage5 clean 六轴 run/task/algorithm 指标")
    parser.add_argument(
        "--numeric-csv",
        type=Path,
        default=stage5_root / "results/clean_numeric_run_metrics.csv",
    )
    parser.add_argument(
        "--clean-numeric-preparation-report-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--eff-csv",
        type=Path,
        default=stage5_root / "results/clean_eff_run_metrics.csv",
    )
    parser.add_argument(
        "--gt-plan-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gt-summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pred-plan-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--eff-preparation-report-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pred-summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--equivalence-plan-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--equivalence-summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--structure-plan-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--structure-summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gt-index-jsonl",
        type=Path,
        default=stage5_root / "reports/clean_gt_frozen_index.jsonl",
    )
    parser.add_argument(
        "--pred-index-jsonl",
        type=Path,
        default=stage5_root / "results/clean_pred_simplify_frozen_index.jsonl",
    )
    parser.add_argument(
        "--equivalence-index-jsonl",
        type=Path,
        default=stage5_root / "results/clean_equivalence_frozen_index.jsonl",
    )
    parser.add_argument(
        "--structure-index-jsonl",
        type=Path,
        default=stage5_root / "results/clean_structure_frozen_index.jsonl",
    )
    parser.add_argument(
        "--evidence-jsonl",
        type=Path,
        default=stage5_root / "results/clean_pred_vs_gt_evidence.jsonl",
    )
    parser.add_argument(
        "--audit-corrections-manifest-json",
        type=Path,
        help="公式质量抽查发布的 final/ok corrections manifest；正式 2250-run 聚合必填",
    )
    parser.add_argument(
        "--clean-run-csv",
        type=Path,
        default=stage5_root / "results/clean_run_metrics.csv",
    )
    parser.add_argument(
        "--task-stability-csv",
        type=Path,
        default=stage5_root / "results/task_stability.csv",
    )
    parser.add_argument(
        "--algorithm-csv",
        type=Path,
        default=stage5_root / "results/algorithm_six_axis.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=stage5_root / "reports/aggregate_clean_metrics.json",
    )
    parser.add_argument("--expected-runs", type=int, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument("--expected-algorithms", type=int, default=DEFAULT_EXPECTED_ALGORITHMS)
    parser.add_argument("--expected-datasets", type=int, default=DEFAULT_EXPECTED_DATASETS)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = aggregate_clean_metrics(
            numeric_csv=args.numeric_csv.resolve(),
            clean_numeric_preparation_report_json=args.clean_numeric_preparation_report_json.resolve(),
            eff_csv=args.eff_csv.resolve(),
            eff_preparation_report_json=args.eff_preparation_report_json.resolve(),
            gt_plan_jsonl=args.gt_plan_jsonl.resolve(),
            gt_summary_json=args.gt_summary_json.resolve(),
            gt_index_jsonl=args.gt_index_jsonl.resolve(),
            pred_plan_jsonl=args.pred_plan_jsonl.resolve(),
            pred_summary_json=args.pred_summary_json.resolve(),
            pred_index_jsonl=args.pred_index_jsonl.resolve(),
            equivalence_plan_jsonl=args.equivalence_plan_jsonl.resolve(),
            equivalence_summary_json=args.equivalence_summary_json.resolve(),
            equivalence_index_jsonl=args.equivalence_index_jsonl.resolve(),
            structure_plan_jsonl=args.structure_plan_jsonl.resolve(),
            structure_summary_json=args.structure_summary_json.resolve(),
            structure_index_jsonl=args.structure_index_jsonl.resolve(),
            evidence_jsonl=args.evidence_jsonl.resolve(),
            audit_corrections_manifest_json=(
                args.audit_corrections_manifest_json.resolve()
                if args.audit_corrections_manifest_json is not None
                else None
            ),
            clean_run_csv=args.clean_run_csv.resolve(),
            task_stability_csv=args.task_stability_csv.resolve(),
            algorithm_csv=args.algorithm_csv.resolve(),
            report_json=args.report_json.resolve(),
            expected_runs=args.expected_runs,
            expected_algorithms=args.expected_algorithms,
            expected_datasets=args.expected_datasets,
        )
    except AggregateCleanMetricsError as exc:
        failure = {
            "fatal_error": str(exc),
            "summary": {
                "run_row_count": 0,
                "task_row_count": 0,
                "algorithm_row_count": 0,
                "expected_runs": args.expected_runs,
                "expected_task_rows": args.expected_algorithms * args.expected_datasets,
                "expected_algorithm_rows": args.expected_algorithms,
            },
        }
        _write_json_atomic(args.report_json.resolve(), failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.print_summary:
        print(rendered)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
