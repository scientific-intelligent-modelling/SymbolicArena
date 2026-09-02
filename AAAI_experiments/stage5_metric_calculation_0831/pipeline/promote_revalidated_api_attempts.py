"""严格重验证既有 Anthropic API 失败尝试并在原 attempt 上补冻。

本工具不发起模型请求，也不增加物理 attempt。它只处理冻结 plan 范围内、
状态库中已耗尽且原错误为 ``structured_output_invalid`` 的 equivalence/structure
任务。所有输入绑定、API 响应契约、Draft-07 schema 和任务输出契约都通过后，
才允许调用状态机的 ``promote_failed_attempt``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError

from .anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    STRICT_EVALUATOR_SYSTEM_PROMPT,
    AnthropicApiResponse,
    AnthropicApiRunner,
)
from .claude_contract import (
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
    ContractViolation,
    canonical_json,
    render_prompt,
)
from .claude_runner import _infer_task_kind
from .run_claude_plan import (
    PlanContractError,
    _load_predecessor_attempt_manifest,
    load_plan_jsonl,
)
from .state import PredecessorAttemptManifest, StateContractError, TaskStateStore


JsonDict = dict[str, object]
PROMOTABLE_ERROR_CLASSES = ("structured_output_invalid",)
PROMOTABLE_TASK_KINDS = {"equivalence", "structure"}
FORMAT_VERSION = "anthropic_api_failed_attempt_promotion.v1"


class ApiAttemptPromotionError(RuntimeError):
    """恢复输入或状态不满足严格补冻契约。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApiAttemptPromotionError(f"attempt 审计文件不可读: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ApiAttemptPromotionError(f"attempt 审计文件必须是 JSON object: {path}")
    return payload


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ApiAttemptPromotionError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _failed_attempt_rows(
    *,
    state_db: Path,
    evaluation_keys: set[str],
    task_type: str,
) -> list[sqlite3.Row]:
    connection = _connect_read_only(state_db)
    try:
        rows = connection.execute(
            """SELECT a.*, t.logical_id, t.task_type, t.condition_name,
                      t.input_hash, t.prompt_version, t.schema_version,
                      t.spec_json, t.state AS task_state
               FROM attempts AS a
               JOIN tasks AS t ON t.evaluation_key = a.evaluation_key
               WHERE a.status='failed'
                 AND a.error_class='structured_output_invalid'
                 AND t.task_type=?
                 AND t.state='exhausted'
               ORDER BY a.evaluation_key, a.attempt_number""",
            (task_type,),
        ).fetchall()
    finally:
        connection.close()
    return [row for row in rows if str(row["evaluation_key"]) in evaluation_keys]


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ApiAttemptPromotionError(f"{context} 必须是 JSON object")
    return value


def _require_exact(value: object, expected: object, *, context: str) -> None:
    if value != expected:
        raise ApiAttemptPromotionError(f"{context} 与状态库/冻结 plan/API 契约不一致")


def _validate_api_request(
    api_request: Mapping[str, object],
    *,
    rendered_prompt: str,
) -> None:
    _require_exact(api_request.get("model"), CONTRACT_CANONICAL_MODEL, context="api_request.model")
    _require_exact(api_request.get("stream"), False, context="api_request.stream")
    _require_exact(
        api_request.get("system"),
        STRICT_EVALUATOR_SYSTEM_PROMPT,
        context="api_request.system",
    )
    _require_exact(
        api_request.get("messages"),
        [{"role": "user", "content": rendered_prompt}],
        context="api_request.messages",
    )
    _require_exact(api_request.get("thinking"), {"type": "adaptive"}, context="api_request.thinking")
    _require_exact(
        api_request.get("output_config"),
        {"effort": CONTRACT_EFFORT},
        context="api_request.output_config",
    )
    max_tokens = api_request.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ApiAttemptPromotionError("api_request.max_tokens 必须为正整数")


def _strictly_revalidate_attempt(
    *,
    attempt: Mapping[str, object],
    attempt_path: Path,
    row: sqlite3.Row,
    entry: object,
) -> JsonDict:
    definition = entry.definition
    task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
    if task_kind not in PROMOTABLE_TASK_KINDS:
        raise ApiAttemptPromotionError(
            f"task_kind={task_kind!r} 不允许仅凭 API 结构化输出补冻"
        )
    attempt_id = str(row["attempt_id"])
    evaluation_key = str(row["evaluation_key"])
    attempt_number = int(row["attempt_number"])
    if attempt_id != f"{evaluation_key}.a{attempt_number:02d}":
        raise ApiAttemptPromotionError("状态库 attempt_id 不是 canonical 形式")
    if attempt_path.name != f"{attempt_id}.json":
        raise ApiAttemptPromotionError("attempt 审计文件名不是 canonical 形式")
    if row["task_state"] != "exhausted":
        raise ApiAttemptPromotionError(
            f"任务当前状态 {row['task_state']!r}，仅允许恢复 exhausted 任务"
        )
    if row["retryable"] != 1:
        raise ApiAttemptPromotionError("原 structured_output_invalid attempt 必须为 retryable")

    expected_state = {
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "condition_name": definition.task_spec.condition,
        "input_hash": definition.task_spec.input_hash,
        "prompt_version": definition.task_spec.prompt_version,
        "schema_version": definition.task_spec.schema_version,
        "spec_json": definition.task_spec.canonical_json(),
    }
    for field, expected in expected_state.items():
        _require_exact(row[field], expected, context=f"state.{field}")

    _require_exact(attempt.get("attempt_id"), attempt_id, context="attempt_id")
    _require_exact(attempt.get("evaluation_key"), evaluation_key, context="evaluation_key")
    _require_exact(attempt.get("request"), definition.request, context="request")
    rendered_prompt = render_prompt(
        definition.prompt_template,
        definition.request,
        definition.schema,
    )
    _require_exact(attempt.get("prompt"), rendered_prompt, context="prompt")

    validation = _require_mapping(attempt.get("validation"), context="validation")
    _require_exact(validation.get("ok"), False, context="validation.ok")
    _require_exact(
        validation.get("error_class"),
        "structured_output_invalid",
        context="validation.error_class",
    )
    _require_exact(
        validation.get("structured_output"),
        None,
        context="validation.structured_output",
    )
    metadata = _require_mapping(attempt.get("metadata"), context="metadata")
    expected_metadata = {
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "evaluation_key": evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "requested_model": CONTRACT_CANONICAL_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "transport_version": API_TRANSPORT_VERSION,
        "stream": False,
        "prompt_path": str(definition.prompt_path),
        "prompt_sha256": definition.prompt_sha256,
        "rendered_prompt_sha256": _sha256_text(rendered_prompt),
        "system_prompt_sha256": _sha256_text(STRICT_EVALUATOR_SYSTEM_PROMPT),
        "schema_path": str(definition.schema_path),
        "schema_sha256": definition.schema_sha256,
        "request_sha256": _sha256_text(canonical_json(definition.request)),
        "http_status": 200,
        "response_model": CONTRACT_CANONICAL_MODEL,
        "error_class": "structured_output_invalid",
        "retryable": True,
    }
    for field, expected in expected_metadata.items():
        _require_exact(metadata.get(field), expected, context=f"metadata.{field}")

    api_request = _require_mapping(attempt.get("api_request"), context="api_request")
    _validate_api_request(api_request, rendered_prompt=rendered_prompt)
    _require_exact(
        metadata.get("api_request_sha256"),
        _sha256_text(canonical_json(api_request)),
        context="metadata.api_request_sha256",
    )
    api_response = _require_mapping(attempt.get("api_response"), context="api_response")
    _require_exact(
        metadata.get("persisted_api_response_sha256"),
        _sha256_text(canonical_json(api_response)),
        context="metadata.persisted_api_response_sha256",
    )
    output_text = attempt.get("output_text")
    if not isinstance(output_text, str):
        raise ApiAttemptPromotionError("output_text 必须是字符串")

    try:
        Draft7Validator.check_schema(definition.schema)
    except SchemaError as exc:
        raise ApiAttemptPromotionError(f"冻结 plan 的 Draft-07 schema 非法: {exc.message}") from exc
    response = AnthropicApiResponse(
        status_code=200,
        body=dict(api_response),
        text=canonical_json(api_response),
        headers={},
    )
    try:
        structured, usage, extracted_text, response_metadata = (
            AnthropicApiRunner._extract_structured_output(
                response,
                task_kind=task_kind,
                schema=definition.schema,
            )
        )
    except ContractViolation as exc:
        raise ApiAttemptPromotionError(f"API 响应严格重验证失败: {exc}") from exc
    # 旧 runner 可能在解析异常前尚未把 content 文本回填到 output_text。
    # api_response 已通过持久化哈希校验，因此其 content 才是恢复权威来源；
    # 只有旧字段非空时，才要求它与重新提取的文本完全一致。
    if output_text:
        _require_exact(extracted_text, output_text, context="output_text")
    stored_usage = metadata.get("usage")
    if isinstance(stored_usage, Mapping):
        _require_exact(usage, dict(stored_usage), context="metadata.usage")
    recovery = response_metadata.get("structured_output_recovery")
    structured_sha256 = _sha256_text(canonical_json(structured))
    return {
        "attempt": dict(attempt),
        "attempt_id": attempt_id,
        "attempt_path": str(attempt_path),
        "attempt_sha256": _sha256_file(attempt_path),
        "evaluation_key": evaluation_key,
        "entry": entry,
        "task_kind": task_kind,
        "structured_output": structured,
        "structured_output_sha256": structured_sha256,
        "structured_output_recovery": recovery,
        "structured_output_source": "hashed_api_response.content",
        "stored_output_text_was_empty": not bool(output_text),
        "response_metadata": response_metadata,
    }


def _predecessor_manifest(
    path: str | Path | None,
) -> PredecessorAttemptManifest | None:
    if path is None:
        return None
    loaded = _load_predecessor_attempt_manifest(path)
    return PredecessorAttemptManifest(
        path=str(loaded.path),
        sha256=loaded.sha256,
        attempt_count=loaded.attempt_count,
    )


def _frozen_payload(
    candidate: Mapping[str, object],
    *,
    audit_reason: str,
    plan_path: Path,
    plan_sha256: str,
    promoted_at: float,
) -> JsonDict:
    attempt = _require_mapping(candidate["attempt"], context="candidate.attempt")
    entry = candidate["entry"]
    definition = entry.definition
    api_response = dict(_require_mapping(attempt.get("api_response"), context="api_response"))
    structured_output = dict(
        _require_mapping(candidate["structured_output"], context="structured_output")
    )
    original_metadata = dict(_require_mapping(attempt.get("metadata"), context="metadata"))
    promotion = {
        "format_version": FORMAT_VERSION,
        "new_model_call": False,
        "strict_api_contract_revalidated": True,
        "draft7_revalidated": True,
        "task_contract_revalidated": True,
        "original_error_class": "structured_output_invalid",
        "original_attempt_audit_path": candidate["attempt_path"],
        "original_attempt_audit_sha256": candidate["attempt_sha256"],
        "structured_output_sha256": candidate["structured_output_sha256"],
        "structured_output_recovery": candidate["structured_output_recovery"],
        "structured_output_source": candidate["structured_output_source"],
        "stored_output_text_was_empty": candidate["stored_output_text_was_empty"],
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "audit_reason": audit_reason,
        "promoted_at": promoted_at,
    }
    metadata = dict(original_metadata)
    metadata.update(
        {
            "original_error_class": "structured_output_invalid",
            "promotion_audit_reason": audit_reason,
            "promoted_at": promoted_at,
            "promotion_format_version": FORMAT_VERSION,
            "promotion_plan_path": str(plan_path),
            "promotion_plan_sha256": plan_sha256,
            "revalidation_response_metadata": candidate["response_metadata"],
            "structured_output_recovery": candidate["structured_output_recovery"],
            "structured_output_source": candidate["structured_output_source"],
            "stored_output_text_was_empty": candidate[
                "stored_output_text_was_empty"
            ],
            "error_class": None,
            "retryable": False,
        }
    )
    return {
        "attempt_id": candidate["attempt_id"],
        "evaluation_key": candidate["evaluation_key"],
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": candidate["task_kind"],
        "request": dict(definition.request),
        "prompt": attempt.get("prompt"),
        "command": [
            "anthropic-messages-api",
            "--model",
            CONTRACT_CANONICAL_MODEL,
            "--effort",
            CONTRACT_EFFORT,
            "--stream",
            "false",
        ],
        "stdout": canonical_json(api_response),
        "stderr": "",
        "envelope": api_response,
        "structured_output": structured_output,
        "validation": {
            "ok": True,
            "error_class": None,
            "error_message": None,
            "structured_output": structured_output,
            "promotion": promotion,
        },
        "metadata": metadata,
    }


def promote_revalidated_api_attempts(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    frozen_dir: str | Path,
    task_type: str,
    manifest_jsonl: str | Path,
    report_json: str | Path,
    audit_reason: str,
    dry_run: bool = True,
    predecessor_attempt_manifest: str | Path | None = None,
) -> JsonDict:
    """重验证 plan 内失败 API attempt；默认 dry-run，只落审计报告。"""

    if not task_type.strip():
        raise ApiAttemptPromotionError("task_type 不得为空")
    if not audit_reason.strip():
        raise ApiAttemptPromotionError("audit_reason 不得为空")
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise ApiAttemptPromotionError(f"plan 契约失败: {exc}") from exc
    plan_task_types = {
        entry.definition.task_spec.task_type for entry in loaded_plan.entries
    }
    if plan_task_types != {task_type}:
        raise ApiAttemptPromotionError(
            f"显式 task_type={task_type!r} 与 plan task_type 集合 {sorted(plan_task_types)!r} 不一致"
        )
    by_key = {entry.evaluation_key: entry for entry in loaded_plan.entries}
    for entry in loaded_plan.entries:
        task_kind = _infer_task_kind(
            entry.definition.task_spec.task_type,
            entry.definition.task_kind,
        )
        if task_kind not in PROMOTABLE_TASK_KINDS:
            raise ApiAttemptPromotionError(
                f"plan 含不允许补冻的 task_kind={task_kind!r}: {entry.evaluation_key}"
            )

    state_path = Path(state_db).resolve()
    attempts_path = Path(attempts_dir).resolve()
    frozen_path = Path(frozen_dir).resolve()
    manifest_path = Path(manifest_jsonl).resolve()
    report_path = Path(report_json).resolve()
    rows = _failed_attempt_rows(
        state_db=state_path,
        evaluation_keys=set(by_key),
        task_type=task_type,
    )
    manifest_rows: list[JsonDict] = []
    candidates_by_key: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        attempt_id = str(row["attempt_id"])
        evaluation_key = str(row["evaluation_key"])
        attempt_path = attempts_path / f"{attempt_id}.json"
        base_record: JsonDict = {
            "format_version": FORMAT_VERSION,
            "evaluation_key": evaluation_key,
            "attempt_id": attempt_id,
            "task_type": task_type,
            "plan_path": str(loaded_plan.plan_path.resolve()),
            "plan_sha256": loaded_plan.plan_sha256,
            "attempt_path": str(attempt_path),
            "new_model_call": False,
        }
        try:
            if not attempt_path.is_file():
                raise ApiAttemptPromotionError("attempt 审计文件不存在")
            candidate = _strictly_revalidate_attempt(
                attempt=_read_json_object(attempt_path),
                attempt_path=attempt_path,
                row=row,
                entry=by_key[evaluation_key],
            )
        except (ApiAttemptPromotionError, ContractViolation, OSError) as exc:
            manifest_rows.append(
                {**base_record, "status": "rejected", "reason": str(exc)}
            )
            continue
        candidates_by_key[evaluation_key].append(candidate)
        manifest_rows.append(
            {
                **base_record,
                "status": "promotable",
                "reason": None,
                "attempt_sha256": candidate["attempt_sha256"],
                "structured_output_sha256": candidate["structured_output_sha256"],
                "structured_output_recovery": candidate["structured_output_recovery"],
                "structured_output_source": candidate["structured_output_source"],
                "stored_output_text_was_empty": candidate[
                    "stored_output_text_was_empty"
                ],
            }
        )

    ambiguous_keys = {
        evaluation_key
        for evaluation_key, candidates in candidates_by_key.items()
        if len(candidates) != 1
    }
    if ambiguous_keys:
        for record in manifest_rows:
            if record["evaluation_key"] in ambiguous_keys and record["status"] == "promotable":
                record["status"] = "rejected"
                record["reason"] = "同一任务存在多个可补冻 failed attempt，拒绝歧义晋升"
        for evaluation_key in ambiguous_keys:
            candidates_by_key.pop(evaluation_key, None)

    promotable = [
        candidates[0]
        for _, candidates in sorted(candidates_by_key.items())
        if len(candidates) == 1
    ]
    attempts_before = len(rows)
    promoted_count = 0
    promotion_failures = 0
    store: TaskStateStore | None = None
    if not dry_run and promotable:
        try:
            store = TaskStateStore(
                state_path,
                predecessor_attempt_manifest=_predecessor_manifest(
                    predecessor_attempt_manifest
                ),
            )
        except (StateContractError, PlanContractError) as exc:
            raise ApiAttemptPromotionError(f"状态库恢复契约失败: {exc}") from exc
        records_by_attempt = {
            str(record["attempt_id"]): record for record in manifest_rows
        }
        for candidate in promotable:
            attempt_id = str(candidate["attempt_id"])
            evaluation_key = str(candidate["evaluation_key"])
            destination = frozen_path / f"{evaluation_key}.json"
            record = records_by_attempt[attempt_id]
            if destination.exists():
                record["status"] = "promotion_failed"
                record["reason"] = f"冻结目标已存在，拒绝覆盖: {destination}"
                promotion_failures += 1
                continue
            if store.task_state(evaluation_key) != "exhausted":
                record["status"] = "promotion_failed"
                record["reason"] = "晋升前任务状态不再是 exhausted"
                promotion_failures += 1
                continue
            promoted_at = time.time()
            frozen_payload = _frozen_payload(
                candidate,
                audit_reason=audit_reason,
                plan_path=loaded_plan.plan_path.resolve(),
                plan_sha256=loaded_plan.plan_sha256,
                promoted_at=promoted_at,
            )
            _atomic_write_json(destination, frozen_payload)
            result_sha256 = _sha256_file(destination)
            try:
                store.promote_failed_attempt(
                    attempt_id,
                    result_path=str(destination),
                    result_sha256=result_sha256,
                    allowed_error_classes=PROMOTABLE_ERROR_CLASSES,
                    audit_reason=audit_reason,
                    now=promoted_at,
                )
            except StateContractError as exc:
                record["status"] = "promotion_failed"
                record["reason"] = f"状态机拒绝晋升: {exc}"
                record["orphan_frozen_path"] = str(destination)
                record["orphan_frozen_sha256"] = result_sha256
                promotion_failures += 1
                continue
            record["status"] = "promoted"
            record["frozen_path"] = str(destination)
            record["frozen_sha256"] = result_sha256
            record["promoted_at"] = promoted_at
            promoted_count += 1

    manifest_rows.sort(key=lambda row: (str(row["evaluation_key"]), str(row["attempt_id"])))
    _atomic_write_jsonl(manifest_path, manifest_rows)
    manifest_sha256 = _sha256_file(manifest_path)
    rejected_count = sum(
        record["status"] in {"rejected", "promotion_failed"}
        for record in manifest_rows
    )
    report: JsonDict = {
        "format_version": FORMAT_VERSION,
        "generated_at": time.time(),
        "dry_run": bool(dry_run),
        "new_model_call": False,
        "audit_reason": audit_reason,
        "task_type": task_type,
        "plan_path": str(loaded_plan.plan_path.resolve()),
        "plan_sha256": loaded_plan.plan_sha256,
        "state_db": str(state_path),
        "attempts_dir": str(attempts_path),
        "frozen_dir": str(frozen_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "scoped_task_count": len(loaded_plan.entries),
        "matching_failed_attempt_count": attempts_before,
        "promotable_count": len(promotable),
        "promoted_count": promoted_count,
        "rejected_count": rejected_count,
        "promotion_failure_count": promotion_failures,
        "attempt_count_delta": 0,
        "strict_revalidation": {
            "api_message_contract": True,
            "draft7_schema": True,
            "task_output_contract": True,
            "unique_valid_structured_object_required": True,
        },
        "status_counts": {
            status: sum(record["status"] == status for record in manifest_rows)
            for status in ("promotable", "promoted", "rejected", "promotion_failed")
        },
    }
    _atomic_write_json(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-jsonl", required=True)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--attempts-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--audit-reason", required=True)
    parser.add_argument("--predecessor-attempt-manifest")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    mode.add_argument("--apply", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = promote_revalidated_api_attempts(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            frozen_dir=args.frozen_dir,
            task_type=args.task_type,
            manifest_jsonl=args.manifest_jsonl,
            report_json=args.report_json,
            audit_reason=args.audit_reason,
            dry_run=args.dry_run,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        )
    except (ApiAttemptPromotionError, OSError, StateContractError) as exc:
        print(f"API attempt 补冻失败: {exc}")
        return 2
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
