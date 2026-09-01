"""只读审计 exhausted simplify 任务中可由当前验证器补冻的既有输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .run_claude_plan import PlanContractError, load_plan_jsonl
from .claude_runner import _redact_string, _sanitize_for_audit
from .semantic_validation_worker import (
    WORKER_MODULE,
    build_semantic_validation_payload,
)


JsonDict = dict[str, object]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_SECRET_KEY_PATTERN = re.compile(
    r"(token|api[_-]?key|authorization|secret|password)",
    re.IGNORECASE,
)
_TRAILING_JSON_FENCE_PATTERN = re.compile(
    r"(?P<prefix>[^{}" + "`" + r"]{0,512})"
    r"(?P<fence>```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```)\Z"
)
_FORBIDDEN_NORMALIZATION_PREFIX_PATTERN = re.compile(
    r"(?:tool[ _-]?(?:result|use)|stdout|stderr|<tool|\b(?:bash|python)\b)",
    re.IGNORECASE,
)


class ExhaustedSimplificationAuditError(RuntimeError):
    """exhausted simplify 审计无法形成可信闭环。"""


def _legacy_sanitize_for_audit(
    value: object,
    *,
    parent_key: str | None = None,
) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if _LEGACY_SECRET_KEY_PATTERN.search(key_text):
                sanitized[key_text] = "[REDACTED]"
            else:
                sanitized[key_text] = _legacy_sanitize_for_audit(
                    item,
                    parent_key=key_text,
                )
        return sanitized
    if isinstance(value, list):
        return [_legacy_sanitize_for_audit(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return [_legacy_sanitize_for_audit(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        if parent_key and _LEGACY_SECRET_KEY_PATTERN.search(parent_key):
            return "[REDACTED]"
        return _redact_string(value)
    return value


def match_audit_envelope_sanitization(
    raw_envelope: Mapping[str, object],
    stored_envelope: Mapping[str, object],
) -> str | None:
    """识别当前或已冻结的 legacy 脱敏投影，不放宽原始 envelope 契约。"""

    if _sanitize_for_audit(raw_envelope) == stored_envelope:
        return "current_boundary_keys"
    if _legacy_sanitize_for_audit(raw_envelope) == stored_envelope:
        return "legacy_token_substring"
    return None


def validate_envelope_with_optional_fenced_result_normalization(
    envelope: Mapping[str, object],
    *,
    task_kind: str,
    schema: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """先走原始严格契约；仅对受限尾部 fenced JSON 形态尝试临时归一化。"""

    try:
        return (
            validate_claude_envelope(
                envelope,
                task_kind=task_kind,
                schema=schema,
            ),
            "strict_passthrough",
        )
    except ContractViolation:
        normalized = _normalize_trailing_fenced_json_result_envelope(envelope)
        if normalized is None:
            raise
    return (
        validate_claude_envelope(
            normalized,
            task_kind=task_kind,
            schema=schema,
        ),
        "tail_fenced_json_object",
    )


def _normalize_trailing_fenced_json_result_envelope(
    envelope: Mapping[str, object],
) -> dict[str, object] | None:
    result = envelope.get("result")
    if not isinstance(result, str):
        return None
    matched = _TRAILING_JSON_FENCE_PATTERN.fullmatch(result)
    if matched is None:
        return None
    if _FORBIDDEN_NORMALIZATION_PREFIX_PATTERN.search(matched.group("prefix")):
        return None
    if result.count("```json") != 1 or result.count("```") != 2:
        return None
    body = matched.group("body")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    normalized = dict(envelope)
    normalized["result"] = body
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise ExhaustedSimplificationAuditError(f"{context} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExhaustedSimplificationAuditError(f"{context} 必须是 JSON object")
    return payload


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ExhaustedSimplificationAuditError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=0.2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def run_isolated_simplify_semantic_validator(
    *,
    evaluation_key: str,
    request: Mapping[str, object],
    structured_output: Mapping[str, object],
    timeout_seconds: float,
) -> JsonDict:
    payload = build_semantic_validation_payload(
        evaluation_key=evaluation_key,
        request=request,
        structured_output=structured_output,
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(_REPO_ROOT),
            start_new_session=True,
        )
    except OSError as exc:
        return {"status": "semantic_validator_error", "error": str(exc)}
    try:
        stdout, stderr = process.communicate(
            canonical_json(payload),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return {
            "status": "semantic_validator_timeout",
            "error": f"semantic validator 超过 {timeout_seconds:g} 秒",
        }
    if process.returncode != 0:
        return {
            "status": "semantic_validator_error",
            "error": stderr.strip() or stdout.strip() or f"rc={process.returncode}",
        }
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "semantic_validator_error",
            "error": f"semantic validator 输出非法 JSON: {exc}",
        }
    if not isinstance(response, dict):
        return {
            "status": "semantic_validator_error",
            "error": "semantic validator 输出不是 JSON object",
        }
    status = response.get("status")
    evidence = response.get("semantic_evidence")
    if not isinstance(evidence, Mapping):
        return {
            "status": "semantic_validator_error",
            "error": "semantic validator 输出缺少 semantic_evidence",
        }
    if status == "ok" and evidence.get("decision") in {
        "equivalent",
        "undetermined",
        "not_applicable",
    }:
        return {"status": "promotable", "semantic_evidence": dict(evidence)}
    return {
        "status": "semantic_rejected",
        "worker_status": status,
        "error": str(response.get("error_message") or "当前语义验证未通过"),
        "semantic_evidence": dict(evidence),
    }


def _audit_attempt(
    *,
    row: sqlite3.Row,
    entry: object,
    attempts_dir: Path,
    timeout_seconds: float,
) -> JsonDict:
    attempt_id = str(row["attempt_id"])
    attempt_number = int(row["attempt_number"])
    expected_attempt_id = f"{entry.evaluation_key}.a{attempt_number:02d}"
    base: JsonDict = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "db_error_class": row["error_class"],
        "db_retryable": bool(row["retryable"]),
    }
    if attempt_id != expected_attempt_id:
        return {**base, "status": "identity_error", "error": "attempt_id 非 canonical"}
    attempt_path = attempts_dir / f"{attempt_id}.json"
    base["attempt_path"] = str(attempt_path)
    if not attempt_path.is_file():
        return {**base, "status": "artifact_missing", "error": "attempt 审计文件不存在"}
    base["attempt_sha256"] = _sha256_file(attempt_path)
    try:
        payload = _read_json_object(attempt_path, context=attempt_id)
    except ExhaustedSimplificationAuditError as exc:
        return {**base, "status": "artifact_invalid", "error": str(exc)}
    definition = entry.definition
    metadata = payload.get("metadata")
    validation = payload.get("validation")
    if payload.get("attempt_id") != attempt_id:
        return {**base, "status": "identity_error", "error": "payload attempt_id 漂移"}
    if payload.get("evaluation_key") != entry.evaluation_key:
        return {**base, "status": "identity_error", "error": "payload evaluation_key 漂移"}
    if payload.get("request") != definition.request:
        return {**base, "status": "identity_error", "error": "request 与冻结 plan 不一致"}
    if not isinstance(metadata, Mapping) or not isinstance(validation, Mapping):
        return {**base, "status": "artifact_invalid", "error": "metadata/validation 缺失"}
    expected_metadata = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": "simplify",
        "error_class": row["error_class"],
        "retryable": bool(row["retryable"]),
        "requested_model": CONTRACT_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "transport_version": CONTRACT_TRANSPORT_VERSION,
        "prompt_path": str(definition.prompt_path),
        "prompt_sha256": definition.prompt_sha256,
        "schema_path": str(definition.schema_path),
        "schema_sha256": definition.schema_sha256,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            return {
                **base,
                "status": "identity_error",
                "error": f"metadata.{field} 与状态/plan 不一致",
            }
    if validation.get("ok") is not False:
        return {**base, "status": "artifact_invalid", "error": "失败 attempt 的 validation.ok 非 false"}
    if validation.get("error_class") != row["error_class"]:
        return {**base, "status": "identity_error", "error": "validation.error_class 与状态库不一致"}
    expected_prompt = render_prompt(
        definition.prompt_template,
        definition.request,
        definition.schema,
    )
    if payload.get("prompt") != expected_prompt:
        return {**base, "status": "strict_contract_failed", "error": "prompt 与冻结 plan 不一致"}
    if payload.get("command") != build_claude_command(definition.schema):
        return {**base, "status": "strict_contract_failed", "error": "Claude command 契约不一致"}
    if metadata.get("rendered_prompt_sha256") != _sha256_text(expected_prompt):
        return {
            **base,
            "status": "identity_error",
            "error": "metadata.rendered_prompt_sha256 校验失败",
        }
    if metadata.get("request_sha256") != _sha256_text(canonical_json(definition.request)):
        return {
            **base,
            "status": "identity_error",
            "error": "metadata.request_sha256 校验失败",
        }
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return {**base, "status": "artifact_invalid", "error": "stdout/stderr 缺失"}
    if metadata.get("stdout_sha256") != _sha256_text(stdout):
        return {**base, "status": "identity_error", "error": "stdout_sha256 校验失败"}
    if metadata.get("stderr_sha256") != _sha256_text(stderr):
        return {**base, "status": "identity_error", "error": "stderr_sha256 校验失败"}
    envelope = payload.get("envelope")
    if not isinstance(envelope, Mapping):
        return {**base, "status": "no_recoverable_output", "error": "缺少 Claude envelope"}
    try:
        stdout_envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            **base,
            "status": "strict_contract_failed",
            "error": f"stdout 不是合法 envelope JSON: {exc}",
        }
    envelope_sanitization_mode = match_audit_envelope_sanitization(
        stdout_envelope,
        envelope,
    )
    if envelope_sanitization_mode is None:
        return {**base, "status": "identity_error", "error": "envelope 与 stdout 不一致"}
    try:
        structured_output, normalization_mode = (
            validate_envelope_with_optional_fenced_result_normalization(
                stdout_envelope,
                task_kind="simplify",
                schema=definition.schema,
            )
        )
    except ContractViolation as exc:
        return {**base, "status": "strict_contract_failed", "error": str(exc)}
    stored_structured = validation.get("structured_output")
    if stored_structured is not None and stored_structured != structured_output:
        return {
            **base,
            "status": "identity_error",
            "error": "validation.structured_output 与 envelope 不一致",
        }
    semantic = run_isolated_simplify_semantic_validator(
        evaluation_key=entry.evaluation_key,
        request=definition.request,
        structured_output=structured_output,
        timeout_seconds=timeout_seconds,
    )
    return {
        **base,
        **semantic,
        "structured_output": structured_output,
        "envelope_sanitization_mode": envelope_sanitization_mode,
        "result_normalization_mode": normalization_mode,
    }


def audit_exhausted_simplifications(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    semantic_timeout_seconds: float = 90.0,
) -> JsonDict:
    if semantic_timeout_seconds <= 0:
        raise ExhaustedSimplificationAuditError("semantic_timeout_seconds 必须为正数")
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise ExhaustedSimplificationAuditError(f"plan 契约失败: {exc}") from exc
    by_key = {entry.evaluation_key: entry for entry in loaded_plan.entries}
    task_types = {entry.definition.task_spec.task_type for entry in loaded_plan.entries}
    conditions = {entry.definition.task_spec.condition for entry in loaded_plan.entries}
    if len(task_types) != 1 or len(conditions) != 1:
        raise ExhaustedSimplificationAuditError(
            "当前审计器要求 plan 中 task_type 与 condition 各自唯一"
        )
    only_task_type = next(iter(task_types))
    only_condition = next(iter(conditions))
    if not only_task_type.endswith("simplify"):
        raise ExhaustedSimplificationAuditError("当前审计器只允许 simplify plan")
    state_path = Path(state_db).resolve()
    attempts_path = Path(attempts_dir).resolve()
    connection = _connect_read_only(state_path)
    task_rows: list[JsonDict] = []
    attempt_status_counts: dict[str, int] = {}
    try:
        exhausted = connection.execute(
            """SELECT evaluation_key, logical_id, task_type, condition_name,
                      attempt_count, last_error_class
               FROM tasks
               WHERE state='exhausted' AND task_type=? AND condition_name=?
               ORDER BY logical_id""",
            (only_task_type, only_condition),
        ).fetchall()
        for task in exhausted:
            evaluation_key = str(task["evaluation_key"])
            entry = by_key.get(evaluation_key)
            if entry is None:
                raise ExhaustedSimplificationAuditError(
                    f"状态库 exhausted 任务不在当前 plan: {task['logical_id']}"
                )
            definition = entry.definition
            if task["logical_id"] != entry.logical_id:
                raise ExhaustedSimplificationAuditError(f"{entry.logical_id}.logical_id 漂移")
            if task["task_type"] != definition.task_spec.task_type:
                raise ExhaustedSimplificationAuditError(f"{entry.logical_id}.task_type 漂移")
            if task["condition_name"] != definition.task_spec.condition:
                raise ExhaustedSimplificationAuditError(f"{entry.logical_id}.condition 漂移")
            if not definition.task_spec.task_type.endswith("simplify"):
                raise ExhaustedSimplificationAuditError(
                    f"当前工具只允许 simplify exhausted: {entry.logical_id}"
                )
            attempts = connection.execute(
                """SELECT attempt_id, attempt_number, status, error_class, retryable
                   FROM attempts
                   WHERE evaluation_key=?
                   ORDER BY attempt_number""",
                (evaluation_key,),
            ).fetchall()
            if int(task["attempt_count"]) != 3 or len(attempts) != 3:
                raise ExhaustedSimplificationAuditError(
                    f"{entry.logical_id} exhausted 必须恰有 3 次尝试"
                )
            attempt_audits: list[JsonDict] = []
            for attempt in attempts:
                if attempt["status"] != "failed":
                    raise ExhaustedSimplificationAuditError(
                        f"{attempt['attempt_id']} exhausted attempt 必须为 failed"
                    )
                audited = _audit_attempt(
                    row=attempt,
                    entry=entry,
                    attempts_dir=attempts_path,
                    timeout_seconds=semantic_timeout_seconds,
                )
                status = str(audited["status"])
                attempt_status_counts[status] = attempt_status_counts.get(status, 0) + 1
                attempt_audits.append(audited)
            promotable = [
                str(item["attempt_id"])
                for item in attempt_audits
                if item["status"] == "promotable"
            ]
            task_rows.append(
                {
                    "evaluation_key": evaluation_key,
                    "logical_id": entry.logical_id,
                    "task_type": definition.task_spec.task_type,
                    "condition": definition.task_spec.condition,
                    "attempt_count": len(attempt_audits),
                    "last_error_class": task["last_error_class"],
                    "attempts": attempt_audits,
                    "promotable_attempt_ids": promotable,
                    "recommended_attempt_id": promotable[0] if promotable else None,
                    "resolution": "promotable" if promotable else "unresolved",
                }
            )
    finally:
        connection.close()

    output_path = Path(output_jsonl).resolve()
    _atomic_write_jsonl(output_path, task_rows)
    normalization_mode_counts: dict[str, int] = {}
    for row in task_rows:
        for attempt in row["attempts"]:
            mode = attempt.get("result_normalization_mode")
            if isinstance(mode, str):
                normalization_mode_counts[mode] = normalization_mode_counts.get(mode, 0) + 1

    report: JsonDict = {
        "status": "ok",
        "model_invoked": False,
        "state_db_mutated": False,
        "artifact_files_mutated": False,
        "plan_jsonl": str(loaded_plan.plan_path.resolve()),
        "plan_sha256": loaded_plan.plan_sha256,
        "state_db": str(state_path),
        "attempts_dir": str(attempts_path),
        "semantic_validator_module": WORKER_MODULE,
        "semantic_validator_worker_sha256": _sha256_file(
            Path(__file__).with_name("semantic_validation_worker.py")
        ),
        "symbolic_validator_sha256": _sha256_file(
            Path(__file__).with_name("symbolic_evidence.py")
        ),
        "semantic_timeout_seconds": float(semantic_timeout_seconds),
        "exhausted_task_count": len(task_rows),
        "promotable_task_count": sum(row["resolution"] == "promotable" for row in task_rows),
        "unresolved_task_count": sum(row["resolution"] == "unresolved" for row in task_rows),
        "attempt_status_counts": dict(sorted(attempt_status_counts.items())),
        "result_normalization_mode_counts": dict(sorted(normalization_mode_counts.items())),
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读复核 exhausted simplify 历史输出")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--semantic-timeout-seconds", type=float, default=90.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit_exhausted_simplifications(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            semantic_timeout_seconds=args.semantic_timeout_seconds,
        )
    except ExhaustedSimplificationAuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
