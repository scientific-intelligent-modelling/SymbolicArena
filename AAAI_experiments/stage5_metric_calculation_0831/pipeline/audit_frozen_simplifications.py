"""用当前严格契约只读复核已冻结的 simplify 结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audit_exhausted_simplifications import (
    run_isolated_simplify_semantic_validator,
)
from .claude_contract import (
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
    ContractViolation,
    build_claude_command,
    canonical_json,
    render_prompt,
    validate_claude_envelope,
)
from .claude_runner import _infer_task_kind
from .run_claude_plan import PlanContractError, load_plan_jsonl


JsonDict = dict[str, object]


class FrozenSimplificationAuditError(RuntimeError):
    """冻结 simplify 复核无法形成可信输入快照。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(canonical_json(value))


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path, *, context: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FrozenSimplificationAuditError(f"{context} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenSimplificationAuditError(f"{context} 必须是 JSON object")
    return payload


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FrozenSimplificationAuditError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _failure(base: Mapping[str, object], code: str, message: str) -> JsonDict:
    return {**base, "status": "failed", "failure_class": code, "error": message}


def _validate_state_binding(entry: object, row: Mapping[str, object]) -> str | None:
    definition = entry.definition
    expected = {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "condition_name": definition.task_spec.condition,
        "priority": definition.task_spec.priority,
        "input_hash": definition.task_spec.input_hash,
        "prompt_version": definition.task_spec.prompt_version,
        "schema_version": definition.task_spec.schema_version,
        "dependencies_json": json.dumps(
            list(definition.task_spec.dependencies),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "spec_json": definition.task_spec.canonical_json(),
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            return f"状态库 {field} 与冻结 plan 不一致"
    return None


def _audit_one_frozen(
    *,
    entry: object,
    state_row: Mapping[str, object] | None,
    attempts_dir: Path,
    frozen_dir: Path,
    semantic_timeout_seconds: float,
) -> JsonDict:
    base: JsonDict = {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "condition": entry.definition.task_spec.condition,
    }
    if state_row is None:
        return _failure(base, "state_missing", "冻结 plan 任务不在状态库")
    binding_error = _validate_state_binding(entry, state_row)
    if binding_error is not None:
        return _failure(base, "state_identity_error", binding_error)
    if state_row.get("state") != "frozen":
        return _failure(
            {**base, "task_state": state_row.get("state")},
            "task_not_frozen",
            "任务尚未处于 frozen 终态",
        )

    attempt_id = state_row.get("frozen_attempt_id")
    attempt_number = state_row.get("attempt_number")
    expected_attempt_id = (
        f"{entry.evaluation_key}.a{int(attempt_number):02d}"
        if isinstance(attempt_number, int)
        else None
    )
    bound: JsonDict = {
        **base,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
    }
    if (
        not isinstance(attempt_id, str)
        or attempt_id != expected_attempt_id
        or state_row.get("attempt_status") != "accepted"
    ):
        return _failure(bound, "attempt_identity_error", "冻结 attempt 绑定或状态不合法")

    expected_frozen_path = (frozen_dir / f"{entry.evaluation_key}.json").resolve()
    result_path_raw = state_row.get("result_path")
    if not isinstance(result_path_raw, str):
        return _failure(bound, "frozen_binding_error", "状态库缺少 result_path")
    result_path = Path(result_path_raw).resolve()
    if result_path != expected_frozen_path:
        return _failure(bound, "frozen_binding_error", "result_path 未绑定到指定 frozen_dir")
    if not result_path.is_file():
        return _failure(bound, "frozen_artifact_missing", "frozen 文件不存在")
    actual_frozen_sha256 = _sha256_file(result_path)
    if actual_frozen_sha256 != state_row.get("result_sha256"):
        return _failure(bound, "frozen_sha256_mismatch", "frozen 文件 SHA-256 与状态库不一致")
    bound.update(
        {
            "frozen_path": str(result_path),
            "frozen_sha256": actual_frozen_sha256,
        }
    )
    try:
        payload = _read_json_object(result_path, context=f"frozen {entry.logical_id}")
    except FrozenSimplificationAuditError as exc:
        return _failure(bound, "frozen_artifact_invalid", str(exc))

    definition = entry.definition
    task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
    expected_top = {
        "attempt_id": attempt_id,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "request": definition.request,
    }
    for field, wanted in expected_top.items():
        if payload.get(field) != wanted:
            return _failure(bound, "frozen_identity_error", f"frozen.{field} 与 plan/状态库不一致")

    metadata = payload.get("metadata")
    validation = payload.get("validation")
    if not isinstance(metadata, Mapping) or not isinstance(validation, Mapping):
        return _failure(bound, "frozen_artifact_invalid", "metadata/validation 缺失")
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
        "error_class": None,
        "retryable": False,
        "returncode": 0,
        "timed_out": False,
    }
    for field, wanted in expected_metadata.items():
        if metadata.get(field) != wanted:
            return _failure(bound, "metadata_identity_error", f"metadata.{field} 不符合冻结契约")
    if validation.get("ok") is not True or validation.get("error_class") is not None:
        return _failure(bound, "validation_state_error", "frozen validation 不是成功状态")

    expected_prompt = render_prompt(
        definition.prompt_template,
        definition.request,
        definition.schema,
    )
    if payload.get("prompt") != expected_prompt:
        return _failure(bound, "prompt_mismatch", "prompt 与冻结 plan 不一致")
    if payload.get("command") != build_claude_command(definition.schema):
        return _failure(bound, "command_contract_error", "Claude command 不符合固定单轮契约")
    if metadata.get("rendered_prompt_sha256") != _sha256_text(expected_prompt):
        return _failure(bound, "metadata_hash_error", "rendered_prompt_sha256 校验失败")
    if metadata.get("request_sha256") != _sha256_text(canonical_json(definition.request)):
        return _failure(bound, "metadata_hash_error", "request_sha256 校验失败")

    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    envelope = payload.get("envelope")
    if not isinstance(stdout, str) or not isinstance(stderr, str) or not isinstance(envelope, Mapping):
        return _failure(bound, "raw_output_missing", "stdout/stderr/envelope 缺失")
    if metadata.get("stdout_sha256") != _sha256_text(stdout):
        return _failure(bound, "metadata_hash_error", "stdout_sha256 校验失败")
    if metadata.get("stderr_sha256") != _sha256_text(stderr):
        return _failure(bound, "metadata_hash_error", "stderr_sha256 校验失败")
    try:
        stdout_envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return _failure(bound, "outer_json_invalid", f"stdout 不是合法 JSON: {exc}")
    if stdout_envelope != envelope:
        return _failure(bound, "raw_output_mismatch", "envelope 与原始 stdout 不一致")
    try:
        structured_output = validate_claude_envelope(
            envelope,
            task_kind=task_kind,
            schema=definition.schema,
        )
    except ContractViolation as exc:
        return _failure(bound, "claude_contract_error", str(exc))
    if payload.get("structured_output") != structured_output:
        return _failure(bound, "structured_output_mismatch", "顶层 structured_output 与 envelope 不一致")
    if validation.get("structured_output") != structured_output:
        return _failure(bound, "structured_output_mismatch", "validation.structured_output 与 envelope 不一致")

    attempt_path = (attempts_dir / f"{attempt_id}.json").resolve()
    if not attempt_path.is_file():
        return _failure(bound, "attempt_artifact_missing", "原始 attempt 审计文件不存在")
    attempt_sha256 = _sha256_file(attempt_path)
    try:
        attempt_payload = _read_json_object(attempt_path, context=f"attempt {attempt_id}")
    except FrozenSimplificationAuditError as exc:
        return _failure(bound, "attempt_artifact_invalid", str(exc))
    for field in (
        "attempt_id",
        "evaluation_key",
        "request",
        "prompt",
        "command",
        "stdout",
        "stderr",
        "envelope",
    ):
        if attempt_payload.get(field) != payload.get(field):
            return _failure(bound, "attempt_frozen_mismatch", f"attempt 与 frozen 的 {field} 不一致")

    promotion = validation.get("promotion")
    if promotion is not None:
        if not isinstance(promotion, Mapping) or promotion.get("new_model_call") is not False:
            return _failure(bound, "promotion_audit_error", "补冻标记缺失 new_model_call=false")
        if promotion.get("source_attempt_audit_sha256") != attempt_sha256:
            return _failure(bound, "promotion_audit_error", "补冻源 attempt SHA-256 不一致")
        if metadata.get("original_attempt_audit_sha256") != attempt_sha256:
            return _failure(bound, "promotion_audit_error", "metadata 补冻源 SHA-256 不一致")
        original_path = metadata.get("original_attempt_audit_path")
        if not isinstance(original_path, str) or Path(original_path).resolve() != attempt_path:
            return _failure(bound, "promotion_audit_error", "metadata 补冻源路径不一致")

    semantic = run_isolated_simplify_semantic_validator(
        evaluation_key=entry.evaluation_key,
        request=definition.request,
        structured_output=structured_output,
        timeout_seconds=semantic_timeout_seconds,
    )
    if semantic.get("status") != "promotable":
        return _failure(
            bound,
            str(semantic.get("status") or "semantic_revalidation_failed"),
            str(semantic.get("error") or "当前隔离语义验证未通过"),
        )
    semantic_evidence = semantic.get("semantic_evidence")
    if not isinstance(semantic_evidence, Mapping):
        return _failure(bound, "semantic_validator_error", "当前语义验证缺少 evidence")
    stored_semantic = validation.get("semantic_evidence")
    stored_semantic_sha256 = (
        _sha256_json(stored_semantic) if isinstance(stored_semantic, Mapping) else None
    )
    current_semantic_sha256 = _sha256_json(semantic_evidence)
    return {
        **bound,
        "status": "passed",
        "failure_class": None,
        "attempt_path": str(attempt_path),
        "attempt_sha256": attempt_sha256,
        "semantic_decision": semantic_evidence.get("decision"),
        "symbolic_decision": semantic_evidence.get("symbolic_decision"),
        "proof_basis": semantic_evidence.get("proof_basis"),
        "probe_count": semantic_evidence.get("probe_count"),
        "current_semantic_evidence_sha256": current_semantic_sha256,
        "stored_semantic_evidence_sha256": stored_semantic_sha256,
        "stored_semantic_evidence_matches_current": (
            stored_semantic_sha256 == current_semantic_sha256
            if stored_semantic_sha256 is not None
            else False
        ),
        "promoted_without_new_model_call": promotion is not None,
    }


def audit_frozen_simplifications(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    frozen_dir: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    expected_plan_count: int | None = None,
    semantic_timeout_seconds: float = 90.0,
    workers: int = 16,
) -> JsonDict:
    if semantic_timeout_seconds <= 0:
        raise FrozenSimplificationAuditError("semantic_timeout_seconds 必须为正数")
    if workers <= 0:
        raise FrozenSimplificationAuditError("workers 必须为正整数")
    if expected_plan_count is not None and expected_plan_count <= 0:
        raise FrozenSimplificationAuditError("expected_plan_count 必须为正整数")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise FrozenSimplificationAuditError(f"plan 契约失败: {exc}") from exc
    if expected_plan_count is not None and len(loaded_plan.entries) != expected_plan_count:
        raise FrozenSimplificationAuditError(
            f"plan 行数 {len(loaded_plan.entries)} != 预期 {expected_plan_count}"
        )
    task_types = {entry.definition.task_spec.task_type for entry in loaded_plan.entries}
    conditions = {entry.definition.task_spec.condition for entry in loaded_plan.entries}
    if len(task_types) != 1 or len(conditions) != 1:
        raise FrozenSimplificationAuditError("plan 的 task_type 与 condition 必须各自唯一")
    task_type = next(iter(task_types))
    condition = next(iter(conditions))
    if not task_type.endswith("simplify"):
        raise FrozenSimplificationAuditError("当前工具只允许 simplify plan")

    state_path = Path(state_db).resolve()
    connection = _connect_read_only(state_path)
    try:
        state_rows = connection.execute(
            """SELECT t.*, f.attempt_id AS frozen_attempt_id,
                      f.result_path, f.result_sha256, f.frozen_at,
                      a.attempt_number, a.status AS attempt_status,
                      a.error_class AS attempt_error_class,
                      a.retryable AS attempt_retryable
               FROM tasks AS t
               LEFT JOIN frozen_results AS f ON f.evaluation_key = t.evaluation_key
               LEFT JOIN attempts AS a ON a.attempt_id = f.attempt_id
               WHERE t.task_type = ? AND t.condition_name = ?
               ORDER BY t.logical_id""",
            (task_type, condition),
        ).fetchall()
    finally:
        connection.close()
    by_key = {str(row["evaluation_key"]): dict(row) for row in state_rows}
    plan_keys = {entry.evaluation_key for entry in loaded_plan.entries}
    extra_keys = sorted(set(by_key) - plan_keys)
    if extra_keys:
        raise FrozenSimplificationAuditError(
            f"状态库存在 {len(extra_keys)} 个不属于当前 plan 的同类任务"
        )
    state_snapshot = [by_key[key] for key in sorted(by_key)]
    state_binding_sha256 = _sha256_json(state_snapshot)
    attempts_path = Path(attempts_dir).resolve()
    frozen_path = Path(frozen_dir).resolve()

    def audit_entry(entry: object) -> JsonDict:
        try:
            return _audit_one_frozen(
                entry=entry,
                state_row=by_key.get(entry.evaluation_key),
                attempts_dir=attempts_path,
                frozen_dir=frozen_path,
                semantic_timeout_seconds=semantic_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - 完整报告优先于单行崩溃
            return {
                "evaluation_key": entry.evaluation_key,
                "logical_id": entry.logical_id,
                "task_type": entry.definition.task_spec.task_type,
                "condition": entry.definition.task_spec.condition,
                "status": "failed",
                "failure_class": "unexpected_audit_error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(audit_entry, loaded_plan.entries))
    output_path = Path(output_jsonl).resolve()
    _atomic_write_jsonl(output_path, rows)
    status_counts = Counter(str(row.get("status")) for row in rows)
    failure_counts = Counter(
        str(row.get("failure_class"))
        for row in rows
        if row.get("status") != "passed"
    )
    passed = status_counts.get("passed", 0)
    report: JsonDict = {
        "status": "ok" if passed == len(rows) else "failed",
        "model_invoked": False,
        "state_db_mutated": False,
        "artifact_files_mutated": False,
        "plan_jsonl": str(loaded_plan.plan_path.resolve()),
        "plan_sha256": loaded_plan.plan_sha256,
        "plan_count": len(loaded_plan.entries),
        "expected_plan_count": expected_plan_count,
        "state_db": str(state_path),
        "state_binding_sha256": state_binding_sha256,
        "database_task_count": len(state_rows),
        "attempts_dir": str(attempts_path),
        "frozen_dir": str(frozen_path),
        "workers": workers,
        "semantic_timeout_seconds": float(semantic_timeout_seconds),
        "semantic_validator_worker_sha256": _sha256_file(
            Path(__file__).with_name("semantic_validation_worker.py")
        ),
        "symbolic_validator_sha256": _sha256_file(
            Path(__file__).with_name("symbolic_evidence.py")
        ),
        "passed_count": passed,
        "failed_count": len(rows) - passed,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "promoted_without_new_model_call_count": sum(
            row.get("promoted_without_new_model_call") is True for row in rows
        ),
        "stored_semantic_evidence_mismatch_count": sum(
            row.get("status") == "passed"
            and row.get("stored_semantic_evidence_matches_current") is False
            for row in rows
        ),
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读复核全部 frozen simplify 结果")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-plan-count", type=int, default=None)
    parser.add_argument("--semantic-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_frozen_simplifications(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            frozen_dir=args.frozen_dir,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            expected_plan_count=args.expected_plan_count,
            semantic_timeout_seconds=args.semantic_timeout_seconds,
            workers=args.workers,
        )
    except FrozenSimplificationAuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
