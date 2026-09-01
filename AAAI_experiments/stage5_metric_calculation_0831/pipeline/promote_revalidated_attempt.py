"""把因本地验证器缺陷失败的既有 Claude attempt 审计后补冻。

该流程不会发起新的模型调用，也不会增加物理 attempt 计数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_exhausted_simplifications import (
    match_audit_envelope_sanitization,
    run_isolated_simplify_semantic_validator,
    validate_envelope_with_optional_fenced_result_normalization,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
    ContractViolation,
    build_claude_command,
    canonical_json,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    SEMANTIC_VALIDATOR_VERSION,
    SEMANTIC_VALIDATOR_TRANSPORT_VERSION,
    _infer_task_kind,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskStateStore,
)


class PromotionError(RuntimeError):
    """既有 attempt 不满足可审计补冻条件。"""


_PROMOTABLE_ERROR_CLASSES = (
    "validation_failed",
    "semantic_validator_timeout",
    "semantic_validator_error",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"JSON 文件不可读: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"JSON 文件不是 object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_failed_attempt_binding(
    *,
    state_db: Path,
    attempt_id: str,
    evaluation_key: str,
) -> dict[str, object]:
    if not state_db.is_file():
        raise PromotionError(f"状态库不存在: {state_db}")
    connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """SELECT a.attempt_id, a.evaluation_key, a.attempt_number,
                      a.status AS attempt_status, a.error_class, a.retryable,
                      t.logical_id, t.task_type, t.condition_name, t.input_hash,
                      t.prompt_version, t.schema_version, t.state AS task_state
               FROM attempts AS a
               JOIN tasks AS t ON t.evaluation_key = a.evaluation_key
               WHERE a.attempt_id = ?""",
            (attempt_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise PromotionError(f"状态库中不存在 attempt: {attempt_id}")
    payload = dict(row)
    if payload["evaluation_key"] != evaluation_key:
        raise PromotionError("attempt 与状态库 evaluation_key 不一致")
    if payload["attempt_status"] != "failed":
        raise PromotionError("仅允许补冻状态库中 status=failed 的 attempt")
    if payload["task_state"] not in {"retry_wait", "exhausted"}:
        raise PromotionError(f"任务状态 {payload['task_state']!r} 不允许补冻")
    if payload["error_class"] not in _PROMOTABLE_ERROR_CLASSES:
        raise PromotionError(
            f"原 error_class={payload['error_class']!r} 不在补冻白名单内"
        )
    return payload


def _strictly_revalidate_attempt(
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    entry: object,
    binding: Mapping[str, object],
    semantic_timeout_seconds: float,
) -> tuple[dict[str, object], dict[str, object], str]:
    definition = entry.definition
    task_kind = _infer_task_kind(
        definition.task_spec.task_type,
        definition.task_kind,
    )
    if task_kind != "simplify":
        raise PromotionError("当前补冻器仅允许重验证 simplify 任务")

    attempt_id = str(binding["attempt_id"])
    attempt_number = int(binding["attempt_number"])
    expected_attempt_id = f"{entry.evaluation_key}.a{attempt_number:02d}"
    if attempt_id != expected_attempt_id or attempt_path.name != f"{attempt_id}.json":
        raise PromotionError("attempt_id 或审计文件名不是 canonical 形式")
    if attempt.get("attempt_id") != attempt_id:
        raise PromotionError("payload attempt_id 与状态库不一致")
    if attempt.get("evaluation_key") != entry.evaluation_key:
        raise PromotionError("payload evaluation_key 与冻结 plan 不一致")
    if attempt.get("request") != definition.request:
        raise PromotionError("attempt request 与冻结 plan 不一致")

    metadata = attempt.get("metadata")
    validation = attempt.get("validation")
    if not isinstance(metadata, Mapping) or not isinstance(validation, Mapping):
        raise PromotionError("attempt metadata/validation 缺失")
    expected_metadata = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "requested_model": CONTRACT_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "transport_version": CONTRACT_TRANSPORT_VERSION,
        "prompt_path": str(definition.prompt_path),
        "prompt_sha256": definition.prompt_sha256,
        "schema_path": str(definition.schema_path),
        "schema_sha256": definition.schema_sha256,
        "error_class": binding["error_class"],
        "retryable": bool(binding["retryable"]),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise PromotionError(f"metadata.{field} 与状态库/冻结 plan 不一致")
    expected_state_fields = {
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "condition_name": definition.task_spec.condition,
        "input_hash": definition.task_spec.input_hash,
        "prompt_version": definition.task_spec.prompt_version,
        "schema_version": definition.task_spec.schema_version,
    }
    for field, expected in expected_state_fields.items():
        if binding.get(field) != expected:
            raise PromotionError(f"状态库 {field} 与冻结 plan 不一致")
    if validation.get("ok") is not False:
        raise PromotionError("原 attempt 不是验证失败状态")
    if validation.get("error_class") != binding["error_class"]:
        raise PromotionError("validation.error_class 与状态库不一致")

    expected_prompt = render_prompt(
        definition.prompt_template,
        definition.request,
        definition.schema,
    )
    if attempt.get("prompt") != expected_prompt:
        raise PromotionError("prompt 与冻结 plan 不一致")
    if attempt.get("command") != build_claude_command(definition.schema):
        raise PromotionError("Claude command 与固定单轮契约不一致")
    if metadata.get("rendered_prompt_sha256") != _sha256_text(expected_prompt):
        raise PromotionError("metadata.rendered_prompt_sha256 校验失败")
    if metadata.get("request_sha256") != _sha256_text(canonical_json(definition.request)):
        raise PromotionError("metadata.request_sha256 校验失败")

    stdout = attempt.get("stdout")
    stderr = attempt.get("stderr")
    envelope = attempt.get("envelope")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise PromotionError("attempt stdout/stderr 缺失")
    if metadata.get("stdout_sha256") != _sha256_text(stdout):
        raise PromotionError("metadata.stdout_sha256 校验失败")
    if metadata.get("stderr_sha256") != _sha256_text(stderr):
        raise PromotionError("metadata.stderr_sha256 校验失败")
    try:
        stdout_envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PromotionError(f"stdout 不是合法 Claude envelope JSON: {exc}") from exc
    if not isinstance(envelope, Mapping):
        raise PromotionError("原 attempt 缺少可重解析的 Claude envelope")
    envelope_sanitization_mode = match_audit_envelope_sanitization(
        stdout_envelope,
        envelope,
    )
    if envelope_sanitization_mode is None:
        raise PromotionError("envelope 与原始 stdout 不一致")
    try:
        structured_output, result_normalization_mode = (
            validate_envelope_with_optional_fenced_result_normalization(
                stdout_envelope,
                task_kind=task_kind,
                schema=definition.schema,
            )
        )
    except ContractViolation as exc:
        raise PromotionError(f"Claude envelope 严格契约失败: {exc}") from exc
    stored_structured = validation.get("structured_output")
    if stored_structured is not None and stored_structured != structured_output:
        raise PromotionError("validation.structured_output 与 envelope 不一致")

    semantic = run_isolated_simplify_semantic_validator(
        evaluation_key=entry.evaluation_key,
        request=definition.request,
        structured_output=structured_output,
        timeout_seconds=semantic_timeout_seconds,
    )
    if semantic.get("status") != "promotable":
        raise PromotionError(
            f"隔离语义重验证未通过: {semantic.get('status')}: {semantic.get('error')}"
        )
    semantic_evidence = semantic.get("semantic_evidence")
    if not isinstance(semantic_evidence, Mapping):
        raise PromotionError("隔离语义重验证缺少 semantic_evidence")
    return (
        structured_output,
        dict(semantic_evidence),
        envelope_sanitization_mode,
        result_normalization_mode,
    )


def promote_revalidated_attempt(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempt_json: str | Path,
    frozen_dir: str | Path,
    predecessor_attempt_manifest: str | Path | None,
    audit_reason: str,
    report_json: str | Path | None = None,
    semantic_timeout_seconds: float = 90.0,
) -> dict[str, object]:
    if not audit_reason.strip():
        raise PromotionError("audit_reason 不得为空")
    if semantic_timeout_seconds <= 0:
        raise PromotionError("semantic_timeout_seconds 必须为正数")
    attempt_path = Path(attempt_json).resolve()
    attempt_sha256 = _sha256_file(attempt_path)
    attempt = _read_object(attempt_path)
    attempt_id = attempt.get("attempt_id")
    evaluation_key = attempt.get("evaluation_key")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise PromotionError("attempt_id 缺失")
    if not isinstance(evaluation_key, str) or not evaluation_key:
        raise PromotionError("evaluation_key 缺失")
    loaded_plan = load_plan_jsonl(plan_jsonl)
    by_key = {entry.evaluation_key: entry for entry in loaded_plan.entries}
    entry = by_key.get(evaluation_key)
    if entry is None:
        raise PromotionError("attempt evaluation_key 不在冻结 plan 中")
    definition = entry.definition
    task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
    state_path = Path(state_db).resolve()
    binding = _load_failed_attempt_binding(
        state_db=state_path,
        attempt_id=attempt_id,
        evaluation_key=evaluation_key,
    )
    (
        structured_output,
        semantic_evidence,
        envelope_sanitization_mode,
        result_normalization_mode,
    ) = _strictly_revalidate_attempt(
        attempt=attempt,
        attempt_path=attempt_path,
        entry=entry,
        binding=binding,
        semantic_timeout_seconds=semantic_timeout_seconds,
    )
    metadata = attempt["metadata"]
    assert isinstance(metadata, Mapping)

    predecessor = None
    if predecessor_attempt_manifest is not None:
        loaded_predecessor = _load_predecessor_attempt_manifest(
            predecessor_attempt_manifest
        )
        predecessor = PredecessorAttemptManifest(
            path=str(loaded_predecessor.path),
            sha256=loaded_predecessor.sha256,
            attempt_count=loaded_predecessor.attempt_count,
        )
    store = TaskStateStore(
        state_path,
        predecessor_attempt_manifest=predecessor,
    )
    validator_path = Path(__file__).with_name("symbolic_evidence.py")
    validator_worker_path = Path(__file__).with_name("semantic_validation_worker.py")
    promoted_at = time.time()
    promoted_metadata = dict(metadata)
    promoted_metadata.update(
        {
            "original_error_class": metadata.get("error_class"),
            "original_attempt_audit_path": str(attempt_path),
            "original_attempt_audit_sha256": attempt_sha256,
            "promotion_audit_reason": audit_reason,
            "promoted_at": promoted_at,
            "semantic_validator_version": SEMANTIC_VALIDATOR_VERSION,
            "semantic_validator_sha256": _sha256_file(validator_path),
            "semantic_validator_transport_version": SEMANTIC_VALIDATOR_TRANSPORT_VERSION,
            "semantic_validator_worker_sha256": _sha256_file(validator_worker_path),
            "semantic_validator_timeout_seconds": float(semantic_timeout_seconds),
            "source_envelope_sanitization_mode": envelope_sanitization_mode,
            "result_normalization_mode": result_normalization_mode,
            "error_class": None,
            "retryable": False,
        }
    )
    promoted_validation = {
        "ok": True,
        "error_class": None,
        "error_message": None,
        "structured_output": structured_output,
        "semantic_evidence": semantic_evidence,
        "promotion": {
            "source_attempt_audit_sha256": attempt_sha256,
            "reason": audit_reason,
            "new_model_call": False,
            "result_normalization_mode": result_normalization_mode,
        },
    }
    frozen_payload = {
        "attempt_id": attempt_id,
        "evaluation_key": evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "request": attempt.get("request"),
        "prompt": attempt.get("prompt"),
        "command": attempt.get("command"),
        "stdout": attempt.get("stdout"),
        "stderr": attempt.get("stderr"),
        "envelope": attempt.get("envelope"),
        "structured_output": structured_output,
        "validation": promoted_validation,
        "metadata": promoted_metadata,
    }
    frozen_path = Path(frozen_dir).resolve() / f"{evaluation_key}.json"
    if frozen_path.exists():
        raise PromotionError(f"目标 frozen 文件已存在: {frozen_path}")
    _atomic_write_json(frozen_path, frozen_payload)
    frozen_sha256 = _sha256_file(frozen_path)
    try:
        store.promote_failed_attempt(
            attempt_id,
            result_path=str(frozen_path),
            result_sha256=frozen_sha256,
            allowed_error_classes=_PROMOTABLE_ERROR_CLASSES,
            audit_reason=audit_reason,
            now=promoted_at,
        )
    except StateContractError as exc:
        raise PromotionError(f"状态库拒绝补冻: {exc}") from exc

    report = {
        "status": "promoted",
        "logical_id": entry.logical_id,
        "evaluation_key": evaluation_key,
        "attempt_id": attempt_id,
        "attempt_count_delta": 0,
        "new_model_call": False,
        "source_attempt_audit_path": str(attempt_path),
        "source_attempt_audit_sha256": attempt_sha256,
        "frozen_path": str(frozen_path),
        "frozen_sha256": frozen_sha256,
        "semantic_decision": semantic_evidence.get("decision"),
        "symbolic_decision": semantic_evidence.get("symbolic_decision"),
        "proof_basis": semantic_evidence.get("proof_basis"),
        "probe_count": semantic_evidence.get("probe_count"),
        "audit_reason": audit_reason,
        "semantic_timeout_seconds": float(semantic_timeout_seconds),
        "strict_contract_revalidated": True,
        "source_envelope_sanitization_mode": envelope_sanitization_mode,
        "result_normalization_mode": result_normalization_mode,
    }
    if report_json is not None:
        _atomic_write_json(Path(report_json), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补冻通过修正版验证器的既有 Claude attempt")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempt-json", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path, default=None)
    parser.add_argument("--audit-reason", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--semantic-timeout-seconds", type=float, default=90.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    promote_revalidated_attempt(
        plan_jsonl=args.plan_jsonl,
        state_db=args.state_db,
        attempt_json=args.attempt_json,
        frozen_dir=args.frozen_dir,
        predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        audit_reason=args.audit_reason,
        report_json=args.report_json,
        semantic_timeout_seconds=args.semantic_timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
