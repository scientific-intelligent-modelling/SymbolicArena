"""聚合公式盲审首轮结果，并生成受预算约束的第二轮盲审计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import (
    canonical_json,
    evaluation_key,
    validate_formula_audit_scope,
    validate_structured_output,
)
from .run_claude_plan import load_plan_jsonl
from .state import TaskSpec


JsonDict = dict[str, Any]

DEFAULT_ROUND1_TASK_COUNT = 1050
DEFAULT_MAXIMUM_TOTAL_ATTEMPTS = 1575
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

_ROUND2_REQUEST_KEYS = {
    "audit_scope",
    "audit_binding_sha256",
    "variables",
    "allowed_functions",
    "domain_assumptions",
    "original_expression",
    "candidate_simplified_expression",
    "reference_simplified_expression",
    "review_round",
    "evidence_hash",
}
_SEVERITY_RANK = {
    "critical": 0,
    "major": 1,
    "undetermined": 2,
    "minor": 3,
    "pass": 4,
}


class FormulaAuditAggregationError(RuntimeError):
    """首轮审计产物不完整、发生漂移或违反聚合契约。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormulaAuditAggregationError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaAuditAggregationError(f"{context} 必须是非空字符串")
    return value.strip()


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        raise FormulaAuditAggregationError(f"JSONL 不存在: {path}")
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FormulaAuditAggregationError(f"{path}:{line_number} 不得为空行")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormulaAuditAggregationError(
                    f"{path}:{line_number} JSON 非法"
                ) from exc
            rows.append(dict(_require_mapping(payload, context=f"{path}:{line_number}")))
    return rows


def _unique_by(
    rows: Sequence[Mapping[str, Any]], key: str, *, context: str
) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise FormulaAuditAggregationError(f"{context} 出现重复 {key}: {value}")
        result[value] = dict(row)
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FormulaAuditAggregationError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_frozen_path(raw_path: str, *, state_db: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    state_relative = state_db.parent / path
    return state_relative if state_relative.is_file() else path.resolve()


def _load_frozen_results(
    *,
    state_db: Path,
    plan_rows_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, JsonDict | None], int, dict[str, str], dict[str, int]]:
    keys = tuple(plan_rows_by_key)
    task_rows: dict[str, JsonDict] = {}
    physical_attempt_count = 0
    state_contract: dict[str, int] = {}
    try:
        with _connect_read_only(state_db) as connection:
            all_task_keys = {
                str(row[0])
                for row in connection.execute("SELECT evaluation_key FROM tasks")
            }
            if all_task_keys != set(keys):
                raise FormulaAuditAggregationError(
                    "状态库任务集合与首轮 plan 不完全一致: "
                    f"extra={len(all_task_keys - set(keys))}, "
                    f"missing={len(set(keys) - all_task_keys)}"
                )
            metadata = dict(connection.execute("SELECT key, value FROM meta"))
            for field in (
                "attempt_cap",
                "logical_task_cap",
                "max_attempts_per_task",
                "attempt_offset",
            ):
                if field not in metadata:
                    raise FormulaAuditAggregationError(f"状态库缺少预算元数据: {field}")
                state_contract[field] = int(metadata[field])
            offset_row = connection.execute(
                "SELECT value FROM meta WHERE key='attempt_offset'"
            ).fetchone()
            if offset_row is not None:
                physical_attempt_count = int(offset_row[0])
            physical_attempt_count += int(
                connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            )
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = f"""SELECT t.evaluation_key, t.logical_id, t.task_type,
                                   t.condition_name, t.priority, t.input_hash,
                                   t.prompt_version, t.schema_version,
                                   t.dependencies_json, t.spec_json, t.state,
                                   f.attempt_id, f.result_path, f.result_sha256
                            FROM tasks t
                            LEFT JOIN frozen_results f
                              ON f.evaluation_key=t.evaluation_key
                            WHERE t.evaluation_key IN ({placeholders})"""
                for row in connection.execute(query, chunk):
                    task_rows[str(row["evaluation_key"])] = dict(row)
    except sqlite3.Error as exc:
        raise FormulaAuditAggregationError(f"状态库读取失败: {exc}") from exc

    if set(task_rows) != set(keys):
        missing = sorted(set(keys) - set(task_rows))
        raise FormulaAuditAggregationError(
            f"状态库缺少首轮任务: {missing[:3]}，共 {len(missing)} 条"
        )

    frozen_payloads: dict[str, JsonDict | None] = {}
    states_by_key: dict[str, str] = {}
    for key, plan_row in plan_rows_by_key.items():
        task_row = task_rows[key]
        expected_spec = _require_mapping(plan_row.get("task_spec"), context=f"{key}.task_spec")
        try:
            stored_spec = json.loads(str(task_row["spec_json"]))
        except json.JSONDecodeError as exc:
            raise FormulaAuditAggregationError(f"{key} 的 state spec_json 非法") from exc
        if stored_spec != expected_spec:
            raise FormulaAuditAggregationError(f"{key} 的 state task_spec 与 plan 不一致")
        state = _require_string(task_row.get("state"), context=f"{key}.state")
        states_by_key[key] = state
        if state == "running":
            raise FormulaAuditAggregationError(f"首轮任务仍在运行，禁止提前聚合: {key}")
        if state not in {"frozen", "pending", "retry_wait", "exhausted"}:
            raise FormulaAuditAggregationError(f"首轮任务状态不可聚合: {key}={state}")
        if state != "frozen":
            if task_row.get("result_path") or task_row.get("result_sha256"):
                raise FormulaAuditAggregationError(f"非 frozen 任务存在结果绑定: {key}")
            frozen_payloads[key] = None
            continue
        if not task_row.get("result_path"):
            raise FormulaAuditAggregationError(f"frozen 任务缺少结果绑定: {key}")
        result_path = _resolve_frozen_path(str(task_row["result_path"]), state_db=state_db)
        if not result_path.is_file():
            raise FormulaAuditAggregationError(f"frozen result 不存在: {result_path}")
        actual_sha256 = _sha256_file(result_path)
        if actual_sha256 != task_row["result_sha256"]:
            raise FormulaAuditAggregationError(f"frozen result SHA-256 漂移: {key}")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormulaAuditAggregationError(f"frozen result 不可读: {key}") from exc
        frozen = dict(_require_mapping(payload, context=f"{key}.frozen_result"))
        expected_identity = {
            "evaluation_key": key,
            "logical_id": plan_row.get("logical_id"),
            "task_type": plan_row.get("task_type"),
            "task_kind": "formula_audit",
            "attempt_id": task_row.get("attempt_id"),
        }
        for field, expected in expected_identity.items():
            if frozen.get(field) != expected:
                raise FormulaAuditAggregationError(f"{key} 的 frozen.{field} 漂移")
        if frozen.get("request") != plan_row.get("request"):
            raise FormulaAuditAggregationError(f"{key} 的 frozen.request 与 plan 不一致")
        validation = _require_mapping(frozen.get("validation"), context=f"{key}.validation")
        if validation.get("ok") is not True:
            raise FormulaAuditAggregationError(f"{key} 的 frozen validation 不是成功状态")
        output = _require_mapping(
            frozen.get("structured_output"), context=f"{key}.structured_output"
        )
        if validation.get("structured_output") != output:
            raise FormulaAuditAggregationError(
                f"{key} 的 validation.structured_output 与 frozen 顶层不一致"
            )
        try:
            validated = validate_structured_output("formula_audit", output)
            validate_formula_audit_scope(
                _require_mapping(plan_row.get("request"), context=f"{key}.request"),
                validated,
            )
        except ValueError as exc:
            raise FormulaAuditAggregationError(f"{key} 的模型输出不满足契约: {exc}") from exc
        frozen["structured_output"] = validated
        frozen_payloads[key] = frozen
    return frozen_payloads, physical_attempt_count, states_by_key, state_contract


def _stored_simplification_decision(outcome: object, *, context: str) -> str:
    normalized = _require_string(outcome, context=context)
    if normalized in {"simplified", "unchanged"}:
        return "preserved"
    if normalized == "unable":
        return "undetermined"
    raise FormulaAuditAggregationError(f"{context} 未知值: {normalized!r}")


def _stored_reference_decision(
    decision: object, *, scope: str, context: str
) -> str:
    normalized = _require_string(decision, context=context)
    allowed = (
        {"not_applicable"}
        if scope == "ground_truth_simplification"
        else {"equivalent", "not_equivalent", "undetermined"}
    )
    if normalized not in allowed:
        raise FormulaAuditAggregationError(f"{context} 未知值: {normalized!r}")
    return normalized


def _judgment_undetermined_fields(output: Mapping[str, Any]) -> list[str]:
    fields = [
        "simplification_equivalence",
        "simplification_quality",
        "reference_equivalence",
    ]
    return [field for field in fields if output.get(field) == "undetermined"]


def _trigger_undetermined_fields(output: Mapping[str, Any]) -> list[str]:
    fields = [*_judgment_undetermined_fields(output)]
    if output.get("severity") == "undetermined":
        fields.append("severity")
    return fields


def _offline_final_severity(
    *,
    scope: str,
    output: Mapping[str, Any],
    simplification_consistent: bool,
    reference_consistent: bool,
) -> tuple[str, list[str]]:
    """按可复现规则重算严重度；模型自报 severity 不参与计算。"""

    reasons: list[str] = []
    if output["simplification_equivalence"] == "not_preserved":
        reasons.append("simplification_not_preserved")
        return "critical", reasons
    if not simplification_consistent:
        reasons.append("stored_simplification_conflict")
        return ("critical" if scope == "ground_truth_simplification" else "major"), reasons
    if not reference_consistent:
        reasons.append("stored_equivalence_conflict")
        return "major", reasons
    undetermined = _judgment_undetermined_fields(output)
    if undetermined:
        return "undetermined", [f"undetermined:{field}" for field in undetermined]
    if output["simplification_quality"] == "more_complex":
        reasons.append("candidate_more_complex")
        return ("major" if scope == "ground_truth_simplification" else "minor"), reasons
    return "pass", ["no_offline_conflict"]


def _trigger_reasons(
    *,
    scope: str,
    output: Mapping[str, Any],
    final_severity: str,
    simplification_consistent: bool,
    reference_consistent: bool,
    confidence_threshold: float,
) -> list[str]:
    reasons = [
        f"undetermined:{field}" for field in _trigger_undetermined_fields(output)
    ]
    if float(output["confidence"]) < confidence_threshold:
        reasons.append("confidence_below_threshold")
    if output["simplification_equivalence"] == "not_preserved":
        reasons.append("simplification_not_preserved")
    if not simplification_consistent:
        reasons.append("stored_simplification_conflict")
    if not reference_consistent:
        reasons.append("stored_equivalence_conflict")
    if scope == "ground_truth_simplification" and final_severity in {"major", "critical"}:
        reasons.append("ground_truth_high_offline_severity")
    # 模型 severity 只作为复判证据，不会覆盖离线 final_severity。
    if scope == "ground_truth_simplification" and output["severity"] in {"major", "critical"}:
        reasons.append("ground_truth_high_model_severity")
    return list(dict.fromkeys(reasons))


def _manifest_identity_fields(manifest: Mapping[str, Any]) -> JsonDict:
    """保留控制面来源身份，供分层汇总与结果追溯使用。"""

    return {
        key: manifest.get(key)
        for key in (
            "logical_key",
            "condition",
            "algorithm",
            "dataset_id",
            "dataset_index",
            "seed",
            "sample_role",
            "risk_reasons",
            "expression_resolution",
            "source_formula_logical_id",
            "source_formula_evaluation_key",
            "source_formula_result_sha256",
            "source_equivalence_logical_id",
            "source_gt_logical_id",
            "source_gt_artifact_sha256",
            "source_pred_artifact_sha256",
        )
    }


def _validate_audit_binding(
    *,
    scope: str,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    context: str,
) -> None:
    if scope == "ground_truth_simplification":
        binding_payload = {
            "source_evaluation_key": manifest.get("source_formula_evaluation_key"),
            "source_result_sha256": manifest.get("source_formula_result_sha256"),
            "stored_simplification_outcome": manifest.get("stored_simplification_outcome"),
        }
    else:
        binding_payload = {
            "source_prediction_evaluation_key": manifest.get(
                "source_formula_evaluation_key"
            ),
            "source_prediction_result_sha256": manifest.get(
                "source_formula_result_sha256"
            ),
            "source_gt_artifact_sha256": manifest.get("source_gt_artifact_sha256"),
            "source_prediction_artifact_sha256": manifest.get(
                "source_pred_artifact_sha256"
            ),
            "stored_simplification_outcome": manifest.get(
                "stored_simplification_outcome"
            ),
            "expression_resolution": manifest.get("expression_resolution"),
            "stored_equivalence_decision": manifest.get(
                "stored_equivalence_decision"
            ),
        }
    for field, value in binding_payload.items():
        _require_string(value, context=f"{context}.{field}")
    if request.get("audit_binding_sha256") != _sha256_json(binding_payload):
        raise FormulaAuditAggregationError(f"{context} 的 audit_binding_sha256 漂移")


def _build_round2_request(round1_request: Mapping[str, Any]) -> JsonDict:
    unexpected = set(round1_request) - _ROUND2_REQUEST_KEYS
    if unexpected:
        raise FormulaAuditAggregationError(
            f"首轮 request 含 round2 白名单外字段: {sorted(unexpected)}"
        )
    request_without_hash = {
        key: json.loads(canonical_json(value))
        for key, value in round1_request.items()
        if key not in {"evidence_hash", "review_round"}
    }
    request_without_hash["review_round"] = 2
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256_json(request_without_hash)
    if set(request) != _ROUND2_REQUEST_KEYS:
        raise FormulaAuditAggregationError("round2 request 字段集合不满足盲审契约")
    return request


def _build_round2_plan_row(
    round1_row: Mapping[str, Any],
    *,
    prompt_path: Path,
    prompt_template: str,
    prompt_sha256: str,
) -> JsonDict:
    round1_logical_id = _require_string(
        round1_row.get("logical_id"), context="round1.logical_id"
    )
    if round1_logical_id.endswith("::v2"):
        raise FormulaAuditAggregationError(f"首轮 logical_id 已是 v2: {round1_logical_id}")
    logical_id = f"{round1_logical_id}::v2"
    request = _build_round2_request(
        _require_mapping(round1_row.get("request"), context=f"{round1_logical_id}.request")
    )
    schema_sha256 = _require_string(
        round1_row.get("schema_sha256"), context=f"{round1_logical_id}.schema_sha256"
    )
    prompt_version = prompt_path.stem
    schema_version = _require_string(
        round1_row.get("schema_version"), context=f"{round1_logical_id}.schema_version"
    )
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_json(normalized_input)
    task_type = _require_string(
        round1_row.get("task_type"), context=f"{round1_logical_id}.task_type"
    )
    if task_type != "formula_audit" or round1_row.get("task_kind") != "formula_audit":
        raise FormulaAuditAggregationError(f"首轮任务不是 formula_audit: {round1_logical_id}")
    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    dependencies = round1_row.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise FormulaAuditAggregationError(f"{round1_logical_id}.dependencies 非法")
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=_require_string(
            round1_row.get("condition"), context=f"{round1_logical_id}.condition"
        ),
        priority=int(round1_row["priority"]),
        input_hash=input_hash,
        prompt_version=prompt_version,
        schema_version=schema_version,
        dependencies=tuple(dependencies),
    )
    return {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": task_type,
        "task_kind": "formula_audit",
        "condition": spec.condition,
        "priority": spec.priority,
        "input_hash": input_hash,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "dependencies": list(dependencies),
        "prompt_path": str(prompt_path.resolve()),
        "schema_path": round1_row["schema_path"],
        "prompt_template": prompt_template,
        "schema_content": round1_row["schema_content"],
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
    }


def aggregate_formula_audit(
    *,
    round1_plan_jsonl: str | Path,
    sample_manifest_jsonl: str | Path,
    state_db: str | Path,
    output_root: str | Path,
    expected_round1_task_count: int = DEFAULT_ROUND1_TASK_COUNT,
    maximum_total_attempts: int = DEFAULT_MAXIMUM_TOTAL_ATTEMPTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    round2_prompt_path: str | Path | None = None,
    external_physical_attempt_count: int = 0,
) -> JsonDict:
    """离线聚合首轮审计；只写审计产物和计划，不执行任何模型调用。"""

    if expected_round1_task_count <= 0:
        raise FormulaAuditAggregationError("expected_round1_task_count 必须为正整数")
    if maximum_total_attempts < expected_round1_task_count:
        raise FormulaAuditAggregationError("maximum_total_attempts 不得小于首轮任务数")
    if not 0 < confidence_threshold <= 1:
        raise FormulaAuditAggregationError("confidence_threshold 必须在 (0, 1] 内")
    if external_physical_attempt_count < 0:
        raise FormulaAuditAggregationError("external_physical_attempt_count 不能为负数")

    plan_path = Path(round1_plan_jsonl).resolve()
    manifest_path = Path(sample_manifest_jsonl).resolve()
    state_path = Path(state_db).resolve()
    output_path = Path(output_root).resolve()
    effective_round2_prompt_path = (
        Path(round2_prompt_path).resolve()
        if round2_prompt_path is not None
        else Path(__file__).resolve().parents[1] / "config/prompts/formula_audit.v2.txt"
    )
    if not effective_round2_prompt_path.is_file():
        raise FormulaAuditAggregationError(
            f"round2 prompt 不存在: {effective_round2_prompt_path}"
        )
    round2_prompt_bytes = effective_round2_prompt_path.read_bytes()
    round2_prompt_template = round2_prompt_bytes.decode("utf-8")
    round2_prompt_sha256 = _sha256_bytes(round2_prompt_bytes)
    loaded_plan = load_plan_jsonl(plan_path)
    raw_plan_rows = _read_jsonl(plan_path)
    if len(raw_plan_rows) != expected_round1_task_count:
        raise FormulaAuditAggregationError(
            f"首轮任务数应为 {expected_round1_task_count}，实际为 {len(raw_plan_rows)}"
        )
    plan_rows_by_key = _unique_by(
        raw_plan_rows, "evaluation_key", context="round1_plan"
    )
    if set(plan_rows_by_key) != {entry.evaluation_key for entry in loaded_plan.entries}:
        raise FormulaAuditAggregationError("首轮 plan 解析身份漂移")

    manifest_rows = _read_jsonl(manifest_path)
    manifest_by_key = _unique_by(
        manifest_rows, "audit_evaluation_key", context="sample_manifest"
    )
    if set(manifest_by_key) != set(plan_rows_by_key):
        raise FormulaAuditAggregationError("sample manifest 与 round1 plan 任务集合不一致")
    for key, manifest in manifest_by_key.items():
        if manifest.get("audit_logical_id") != plan_rows_by_key[key].get("logical_id"):
            raise FormulaAuditAggregationError(f"manifest logical_id 漂移: {key}")

    frozen_by_key, physical_attempt_count, round1_states, state_contract = _load_frozen_results(
        state_db=state_path,
        plan_rows_by_key=plan_rows_by_key,
    )
    if state_contract["attempt_cap"] != maximum_total_attempts:
        raise FormulaAuditAggregationError(
            "状态库 attempt_cap 与审计总预算不一致: "
            f"{state_contract['attempt_cap']} != {maximum_total_attempts}"
        )
    if state_contract["logical_task_cap"] < expected_round1_task_count:
        raise FormulaAuditAggregationError("状态库 logical_task_cap 小于首轮任务数")
    if state_contract["max_attempts_per_task"] not in {1, 2}:
        raise FormulaAuditAggregationError(
            "公式审计每项只允许首调加至多一次失败重试"
        )
    accounted_physical_attempt_count = (
        physical_attempt_count + external_physical_attempt_count
    )
    if accounted_physical_attempt_count > maximum_total_attempts:
        raise FormulaAuditAggregationError(
            "计入预检后物理调用已超预算: "
            f"{accounted_physical_attempt_count}/{maximum_total_attempts}"
        )
    remaining_attempt_budget = maximum_total_attempts - accounted_physical_attempt_count
    round1_state_distribution = dict(sorted(Counter(round1_states.values()).items()))
    unfinished_states = {
        state: count
        for state, count in round1_state_distribution.items()
        if state != "frozen" and count
    }
    retryable_unfinished_count = sum(
        round1_state_distribution.get(state, 0) for state in ("pending", "retry_wait")
    )
    if retryable_unfinished_count and remaining_attempt_budget > 0:
        raise FormulaAuditAggregationError(
            "首轮仍有可继续任务且物理预算未耗尽，禁止提前生成 round2: "
            f"{unfinished_states}"
        )

    comparisons: list[JsonDict] = []
    for key in sorted(plan_rows_by_key, key=lambda item: str(plan_rows_by_key[item]["logical_id"])):
        plan_row = plan_rows_by_key[key]
        manifest = manifest_by_key[key]
        scope = _require_string(manifest.get("audit_scope"), context=f"{key}.audit_scope")
        request = _require_mapping(plan_row.get("request"), context=f"{key}.request")
        if request.get("audit_scope") != scope:
            raise FormulaAuditAggregationError(f"{key} 的 manifest/request scope 不一致")
        if request.get("review_round") != 1:
            raise FormulaAuditAggregationError(f"{key} 的首轮 review_round 必须为 1")
        _validate_audit_binding(
            scope=scope,
            request=request,
            manifest=manifest,
            context=key,
        )
        stored_simplification = _stored_simplification_decision(
            manifest.get("stored_simplification_outcome"),
            context=f"{key}.stored_simplification_outcome",
        )
        stored_reference = _stored_reference_decision(
            manifest.get("stored_equivalence_decision"),
            scope=scope,
            context=f"{key}.stored_equivalence_decision",
        )
        frozen = frozen_by_key[key]
        if frozen is None:
            state = round1_states[key]
            comparisons.append(
                {
                    "audit_logical_id": plan_row["logical_id"],
                    "round1_evaluation_key": key,
                    "round1_state": state,
                    "audit_scope": scope,
                    **_manifest_identity_fields(manifest),
                    "stored_simplification_outcome": manifest["stored_simplification_outcome"],
                    "stored_simplification_decision": stored_simplification,
                    "round1_simplification_decision": None,
                    "simplification_decision_consistent": None,
                    "stored_equivalence_decision": stored_reference,
                    "round1_equivalence_decision": None,
                    "equivalence_decision_consistent": None,
                    "model_reported_severity": None,
                    "offline_final_severity": "undetermined",
                    "offline_final_severity_reasons": [f"round1_result_unavailable:{state}"],
                    "round1_confidence": None,
                    "round1_model_evidence": None,
                    "requires_round2": True,
                    "round2_trigger_reasons": [f"round1_result_unavailable:{state}"],
                }
            )
            continue
        output = _require_mapping(
            frozen["structured_output"], context=f"{key}.structured_output"
        )
        simplification_consistent = (
            output["simplification_equivalence"] == stored_simplification
        )
        reference_consistent = output["reference_equivalence"] == stored_reference
        final_severity, final_reasons = _offline_final_severity(
            scope=scope,
            output=output,
            simplification_consistent=simplification_consistent,
            reference_consistent=reference_consistent,
        )
        trigger_reasons = _trigger_reasons(
            scope=scope,
            output=output,
            final_severity=final_severity,
            simplification_consistent=simplification_consistent,
            reference_consistent=reference_consistent,
            confidence_threshold=confidence_threshold,
        )
        comparisons.append(
            {
                "audit_logical_id": plan_row["logical_id"],
                "round1_evaluation_key": key,
                "round1_state": "frozen",
                "audit_scope": scope,
                **_manifest_identity_fields(manifest),
                "stored_simplification_outcome": manifest["stored_simplification_outcome"],
                "stored_simplification_decision": stored_simplification,
                "round1_simplification_decision": output["simplification_equivalence"],
                "simplification_decision_consistent": simplification_consistent,
                "stored_equivalence_decision": stored_reference,
                "round1_equivalence_decision": output["reference_equivalence"],
                "equivalence_decision_consistent": reference_consistent,
                "model_reported_severity": output["severity"],
                "offline_final_severity": final_severity,
                "offline_final_severity_reasons": final_reasons,
                "round1_confidence": output["confidence"],
                "round1_model_evidence": dict(output),
                "requires_round2": bool(trigger_reasons),
                "round2_trigger_reasons": trigger_reasons,
            }
        )

    triggered = [row for row in comparisons if row["requires_round2"]]
    triggered.sort(
        key=lambda row: (
            _SEVERITY_RANK[str(row["offline_final_severity"])],
            -len(row["round2_trigger_reasons"]),
            str(row["audit_logical_id"]),
        )
    )
    round2_budget_sufficient = len(triggered) <= remaining_attempt_budget
    selected_keys = {
        str(row["round1_evaluation_key"])
        for row in (triggered if round2_budget_sufficient else [])
    }
    trigger_rows: list[JsonDict] = []
    round2_plan_rows: list[JsonDict] = []
    for rank, comparison in enumerate(triggered, start=1):
        key = str(comparison["round1_evaluation_key"])
        selected = key in selected_keys
        trigger_rows.append(
            {
                "priority_rank": rank,
                "audit_logical_id": comparison["audit_logical_id"],
                "round1_evaluation_key": key,
                "audit_scope": comparison["audit_scope"],
                "offline_final_severity": comparison["offline_final_severity"],
                "model_reported_severity": comparison["model_reported_severity"],
                "trigger_reasons": comparison["round2_trigger_reasons"],
                "selected_for_round2": selected,
                "not_selected_reason": (
                    None
                    if selected
                    else "round2_plan_withheld_insufficient_physical_attempt_budget"
                ),
            }
        )
        if selected:
            round2_plan_rows.append(
                _build_round2_plan_row(
                    plan_rows_by_key[key],
                    prompt_path=effective_round2_prompt_path,
                    prompt_template=round2_prompt_template,
                    prompt_sha256=round2_prompt_sha256,
                )
            )
    round2_plan_rows.sort(key=lambda row: str(row["logical_id"]))

    comparisons_path = output_path / "results/formula_audit_round1_comparisons.jsonl"
    triggers_path = output_path / "manifests/formula_audit_round2_triggers.jsonl"
    round2_plan_path = output_path / "plans/formula_audit_round2.jsonl"
    report_path = output_path / "reports/formula_audit_aggregate_report.json"
    _write_jsonl(comparisons_path, comparisons)
    _write_jsonl(triggers_path, trigger_rows)
    _write_jsonl(round2_plan_path, round2_plan_rows)

    severity_counts = Counter(str(row["offline_final_severity"]) for row in comparisons)
    trigger_reason_counts = Counter(
        reason for row in triggered for reason in row["round2_trigger_reasons"]
    )
    report: JsonDict = {
        "status": "ok" if round2_budget_sufficient else "round2_budget_insufficient",
        "contract_ok": True,
        "model_invoked": False,
        "round1_execution_classification": (
            "complete"
            if not unfinished_states
            else (
                "physical_budget_exhausted_with_unresolved_tasks"
                if remaining_attempt_budget == 0
                else "terminal_unresolved_tasks"
            )
        ),
        "inputs": {
            "round1_plan_jsonl": str(plan_path),
            "round1_plan_sha256": _sha256_file(plan_path),
            "sample_manifest_jsonl": str(manifest_path),
            "sample_manifest_sha256": _sha256_file(manifest_path),
            "state_db": str(state_path),
            "state_contract": dict(state_contract),
            "round2_prompt_path": str(effective_round2_prompt_path),
            "round2_prompt_sha256": round2_prompt_sha256,
        },
        "outputs": {
            "comparisons_jsonl": str(comparisons_path),
            "comparisons_sha256": _sha256_file(comparisons_path),
            "triggers_jsonl": str(triggers_path),
            "triggers_sha256": _sha256_file(triggers_path),
            "round2_plan_jsonl": str(round2_plan_path),
            "round2_plan_sha256": _sha256_file(round2_plan_path),
            "report_json": str(report_path),
        },
        "counts": {
            "round1_task_count": len(comparisons),
            "round2_trigger_count": len(triggered),
            "round2_planned_count": len(round2_plan_rows),
            "round2_deferred_by_budget_count": len(triggered) - len(round2_plan_rows),
            "simplification_conflict_count": sum(
                row["simplification_decision_consistent"] is False for row in comparisons
            ),
            "equivalence_conflict_count": sum(
                row["equivalence_decision_consistent"] is False for row in comparisons
            ),
            "round1_state_distribution": round1_state_distribution,
            "round1_unresolved_without_frozen_result_count": sum(unfinished_states.values()),
            "offline_final_severity_counts": dict(sorted(severity_counts.items())),
            "trigger_reason_counts": dict(sorted(trigger_reason_counts.items())),
        },
        "budget": {
            "round1_initial_task_count": expected_round1_task_count,
            "round1_physical_attempt_count": physical_attempt_count,
            "external_preflight_physical_attempt_count": external_physical_attempt_count,
            "accounted_physical_attempt_count_before_round2": (
                accounted_physical_attempt_count
            ),
            "maximum_total_physical_attempts": maximum_total_attempts,
            "remaining_physical_attempt_budget": remaining_attempt_budget,
            "round2_plan_count": len(round2_plan_rows),
            "round2_required_task_count": len(triggered),
            "round2_budget_shortfall": max(0, len(triggered) - remaining_attempt_budget),
            "round2_budget_sufficient": round2_budget_sufficient,
            "round2_plan_withheld_on_shortfall": not round2_budget_sufficient,
            "round2_within_remaining_budget": len(round2_plan_rows)
            <= remaining_attempt_budget,
        },
        "severity_policy": {
            "source": "offline_deterministic_v1",
            "model_reported_severity_is_evidence_only": True,
            "critical": "simplification not preserved, or GT stored simplification conflict",
            "major": "prediction stored-decision conflict, or GT candidate more complex",
            "undetermined": "a primary judgment is undetermined and no stronger conflict exists",
            "minor": "prediction candidate is more complex without a stronger issue",
            "pass": "no offline conflict or quality issue",
        },
        "blindness_contract": {
            "round1_structured_output_in_round2_request": False,
            "model_reported_severity_in_round2_request": False,
            "stored_simplification_outcome_in_round2_request": False,
            "stored_equivalence_decision_in_round2_request": False,
            "round2_request_keys": sorted(_ROUND2_REQUEST_KEYS),
        },
    }
    _write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合公式审计首轮结果并生成盲审复判计划")
    parser.add_argument("--round1-plan-jsonl", type=Path, required=True)
    parser.add_argument("--sample-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-round1-task-count", type=int, default=DEFAULT_ROUND1_TASK_COUNT
    )
    parser.add_argument(
        "--maximum-total-attempts", type=int, default=DEFAULT_MAXIMUM_TOTAL_ATTEMPTS
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--round2-prompt-path", type=Path, default=None)
    parser.add_argument("--external-physical-attempt-count", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = aggregate_formula_audit(
            round1_plan_jsonl=args.round1_plan_jsonl,
            sample_manifest_jsonl=args.sample_manifest_jsonl,
            state_db=args.state_db,
            output_root=args.output_root,
            expected_round1_task_count=args.expected_round1_task_count,
            maximum_total_attempts=args.maximum_total_attempts,
            confidence_threshold=args.confidence_threshold,
            round2_prompt_path=args.round2_prompt_path,
            external_physical_attempt_count=args.external_physical_attempt_count,
        )
    except (FormulaAuditAggregationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
