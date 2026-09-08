"""把既有、已审计的 API frozen JSON 安全绑定到另一份 Stage5 状态库。

该工具只做本地文件校验与状态迁移，不发送任何网络请求，也不把导入动作
伪装成模型调用。每次成功导入都会保留独立的 ``cache_import`` 审计记录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    STRICT_EVALUATOR_SYSTEM_PROMPT,
    AnthropicApiResponse,
    AnthropicApiRunner,
)
from .audit_frozen_simplifications import run_isolated_simplify_semantic_validator
from .claude_contract import CONTRACT_CANONICAL_MODEL, CONTRACT_EFFORT, canonical_json, render_prompt
from .claude_runner import _infer_task_kind, _redact_string
from .run_claude_plan import PlanContractError, PlannedDefinition, load_plan_jsonl
from .state import PredecessorAttemptManifest, StateContractError, TaskStateStore


FORBIDDEN_SOURCE = "all_15alg_fullcpu_v1"
AUDIT_SCHEMA_VERSION = "cache_import_attempt.v1"
REPORT_SCHEMA_VERSION = "frozen_plan_import_report.v1"


class FrozenPlanImportError(RuntimeError):
    """输入冻结结果或目标状态不满足安全导入契约。"""


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


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenPlanImportError(f"{context} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenPlanImportError(f"{context} 顶层必须是 JSON object")
    return payload


def _reject_forbidden(value: object, *, context: str) -> None:
    if FORBIDDEN_SOURCE in str(value):
        raise FrozenPlanImportError(f"{context} 命中禁止来源 {FORBIDDEN_SOURCE}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_forbidden(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, context=f"{context}[{index}]")


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrozenPlanImportError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenPlanImportError(f"{context} 必须是非空字符串")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = _require_string(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FrozenPlanImportError(f"{context} 必须是小写十六进制 SHA256")
    return text


def _read_state_meta(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FrozenPlanImportError(f"state DB 不存在或为空: {path}")
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            return dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error as exc:
        raise FrozenPlanImportError(f"state DB meta 不可读: {exc}") from exc


def _state_content_sha256(path: Path) -> str:
    """哈希 SQLite 的逻辑内容，避免只哈希主文件而漏掉 WAL。"""

    digest = hashlib.sha256()
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                if not str(row[0]).startswith("sqlite_")
            ]
            for table in tables:
                digest.update(table.encode("utf-8"))
                for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
                    digest.update(canonical_json(dict(row)).encode("utf-8"))
                    digest.update(b"\n")
    except sqlite3.Error as exc:
        raise FrozenPlanImportError(f"state DB 逻辑内容无法哈希: {exc}") from exc
    return digest.hexdigest()


def _state_store_from_meta(path: Path, meta: Mapping[str, str]) -> TaskStateStore:
    try:
        predecessor_count = int(meta.get("predecessor_attempt_count", "0"))
        predecessor_manifest = None
        if predecessor_count:
            predecessor_manifest = PredecessorAttemptManifest(
                path=meta["predecessor_attempt_manifest_path"],
                sha256=meta["predecessor_attempt_manifest_sha256"],
                attempt_count=predecessor_count,
            )
        return TaskStateStore(
            path,
            attempt_cap=int(meta["attempt_cap"]),
            logical_task_cap=int(meta["logical_task_cap"]),
            max_attempts_per_task=int(meta["max_attempts_per_task"]),
            predecessor_attempt_manifest=predecessor_manifest,
        )
    except (KeyError, TypeError, ValueError, StateContractError) as exc:
        raise FrozenPlanImportError(f"state DB 冻结容量契约无法还原: {exc}") from exc


def _task_rows(state_db: Path, entries: Sequence[PlannedDefinition]) -> dict[str, dict[str, Any]]:
    keys = [entry.evaluation_key for entry in entries]
    rows: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = (
                    "SELECT t.evaluation_key, t.spec_json, t.state, "
                    "f.attempt_id AS frozen_attempt_id, f.result_path, f.result_sha256 "
                    "FROM tasks t LEFT JOIN frozen_results f "
                    "ON f.evaluation_key=t.evaluation_key "
                    f"WHERE t.evaluation_key IN ({placeholders})"
                )
                for row in connection.execute(query, chunk).fetchall():
                    rows[str(row["evaluation_key"])] = dict(row)
    except sqlite3.Error as exc:
        raise FrozenPlanImportError(f"state DB 任务读取失败: {exc}") from exc
    return rows


def _validate_source_artifact(
    entry: PlannedDefinition,
    *,
    frozen_dir: Path,
    attempts_dir: Path,
) -> dict[str, object] | None:
    result_path = (frozen_dir / f"{entry.evaluation_key}.json").resolve()
    if not result_path.is_file():
        return None
    payload = _read_json(result_path, context=f"frozen {entry.logical_id}")
    _reject_forbidden(payload, context=f"frozen {entry.logical_id}")
    definition = entry.definition
    task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
    expected_prompt = render_prompt(definition.prompt_template, definition.request, definition.schema)
    expected_identity = {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "request": definition.request,
        "prompt": expected_prompt,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise FrozenPlanImportError(f"{entry.logical_id}: frozen.{field} 漂移")

    source_attempt_id = _require_string(payload.get("attempt_id"), context="frozen.attempt_id")
    metadata = _require_mapping(payload.get("metadata"), context="frozen.metadata")
    validation = _require_mapping(payload.get("validation"), context="frozen.validation")
    envelope = _require_mapping(payload.get("envelope"), context="frozen.envelope")
    expected_metadata = {
        "attempt_id": source_attempt_id,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "requested_model": CONTRACT_CANONICAL_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "transport_version": API_TRANSPORT_VERSION,
        "stream": False,
        "http_status": 200,
        "response_model": CONTRACT_CANONICAL_MODEL,
        "error_class": None,
        "retryable": False,
        "prompt_sha256": definition.prompt_sha256,
        "schema_sha256": definition.schema_sha256,
        "rendered_prompt_sha256": _sha256_text(expected_prompt),
        "request_sha256": _sha256_text(canonical_json(definition.request)),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise FrozenPlanImportError(f"{entry.logical_id}: metadata.{field} 漂移")
    if metadata.get("api_channel") not in {"routify", "yapi"}:
        raise FrozenPlanImportError(f"{entry.logical_id}: metadata.api_channel 非法")
    for field in ("api_base_url", "api_endpoint"):
        _require_string(metadata.get(field), context=f"metadata.{field}")
    response_metadata = _require_mapping(
        metadata.get("response_metadata"), context="metadata.response_metadata"
    )
    if response_metadata.get("model") != CONTRACT_CANONICAL_MODEL:
        raise FrozenPlanImportError(f"{entry.logical_id}: metadata.response_metadata.model 漂移")
    if Path(definition.schema_path).stem != definition.task_spec.schema_version:
        raise FrozenPlanImportError(f"{entry.logical_id}: plan schema_version 与 schema_path 漂移")
    if Path(_require_string(metadata.get("schema_path"), context="metadata.schema_path")).name != Path(
        definition.schema_path
    ).name:
        raise FrozenPlanImportError(f"{entry.logical_id}: metadata.schema_path/schema_version 漂移")
    for field in (
        "api_request_sha256",
        "api_response_sha256",
        "persisted_api_response_sha256",
        "raw_response_sha256",
    ):
        _require_sha256(metadata.get(field), context=f"metadata.{field}")
    if metadata["persisted_api_response_sha256"] != _sha256_text(canonical_json(envelope)):
        raise FrozenPlanImportError(f"{entry.logical_id}: API 响应 provenance SHA 漂移")
    if validation.get("ok") is not True or validation.get("error_class") is not None:
        raise FrozenPlanImportError(f"{entry.logical_id}: validation 不是成功状态")

    try:
        structured_output, _, output_text, _ = AnthropicApiRunner._extract_structured_output(
            AnthropicApiResponse(
                status_code=200,
                body=dict(envelope),
                text=str(payload.get("stdout") or ""),
                headers={},
            ),
            task_kind=task_kind,
            schema=definition.schema,
        )
    except Exception as exc:
        raise FrozenPlanImportError(f"{entry.logical_id}: frozen 输出未通过正式 schema 校验: {exc}") from exc
    if payload.get("structured_output") != structured_output:
        raise FrozenPlanImportError(f"{entry.logical_id}: 顶层 structured_output 与 API envelope 不一致")
    if validation.get("structured_output") != structured_output:
        raise FrozenPlanImportError(f"{entry.logical_id}: validation.structured_output 不一致")
    if task_kind == "simplify":
        semantic = run_isolated_simplify_semantic_validator(
            evaluation_key=entry.evaluation_key,
            request=definition.request,
            structured_output=structured_output,
            timeout_seconds=300.0,
        )
        if semantic.get("status") != "promotable":
            raise FrozenPlanImportError(
                f"{entry.logical_id}: simplify 独立语义复核失败: "
                f"{semantic.get('status')}: {semantic.get('error')}"
            )

    source_attempt_path = (attempts_dir / f"{source_attempt_id}.json").resolve()
    if not source_attempt_path.is_file():
        raise FrozenPlanImportError(f"{entry.logical_id}: source attempt 审计文件不存在")
    attempt_payload = _read_json(source_attempt_path, context=f"attempt {entry.logical_id}")
    _reject_forbidden(attempt_payload, context=f"attempt {entry.logical_id}")
    expected_attempt_fields = {
        "attempt_id": source_attempt_id,
        "evaluation_key": entry.evaluation_key,
        "request": definition.request,
        "prompt": expected_prompt,
        "api_response": envelope,
        "output_text": output_text,
        "validation": validation,
        "metadata": metadata,
    }
    for field, expected in expected_attempt_fields.items():
        if attempt_payload.get(field) != expected:
            raise FrozenPlanImportError(f"{entry.logical_id}: attempt.{field} 与 frozen 不一致")
    api_request = _require_mapping(attempt_payload.get("api_request"), context="attempt.api_request")
    if _sha256_text(canonical_json(api_request)) != metadata["api_request_sha256"]:
        raise FrozenPlanImportError(f"{entry.logical_id}: API request provenance SHA 漂移")
    expected_api_request = {
        "model": CONTRACT_CANONICAL_MODEL,
        "stream": False,
        "system": STRICT_EVALUATOR_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": expected_prompt}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": CONTRACT_EFFORT},
    }
    for field, expected in expected_api_request.items():
        if api_request.get(field) != expected:
            raise FrozenPlanImportError(f"{entry.logical_id}: attempt.api_request.{field} 漂移")
    max_tokens = api_request.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise FrozenPlanImportError(f"{entry.logical_id}: attempt.api_request.max_tokens 非法")
    raw_audit = result_path.read_text(encoding="utf-8") + source_attempt_path.read_text(encoding="utf-8")
    if _redact_string(raw_audit) != raw_audit or "Authorization" in raw_audit:
        raise FrozenPlanImportError(f"{entry.logical_id}: source audit 包含疑似凭据")

    return {
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path),
        "source_attempt_id": source_attempt_id,
        "source_attempt_path": str(source_attempt_path),
        "source_attempt_sha256": _sha256_file(source_attempt_path),
        "structured_output_sha256": _sha256_text(canonical_json(structured_output)),
        "model": CONTRACT_CANONICAL_MODEL,
        "api_channel": metadata["api_channel"],
        "schema_version": definition.task_spec.schema_version,
    }


def _failure_report(
    *,
    report_path: Path,
    plan_path: Path,
    plan_sha256: str,
    state_db: Path,
    frozen_dir: Path,
    attempts_dir: Path,
    planned: int,
    error: str,
) -> None:
    _atomic_write_json(
        report_path,
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
            "counts": {"planned": planned, "imported": 0, "cached": 0, "failed": planned, "missing": 0},
            "error": error,
            "inputs": {
                "plan_jsonl": str(plan_path),
                "plan_sha256": plan_sha256,
                "state_db": str(state_db),
                "frozen_dir": str(frozen_dir),
                "attempts_dir": str(attempts_dir),
            },
        },
    )


def import_frozen_plan_results(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    frozen_dir: str | Path,
    attempts_dir: str | Path,
    report_json: str | Path,
    lease_seconds: float = 300.0,
) -> dict[str, object]:
    """严格复核现有 frozen 结果，并在目标 state DB 中建立新的审计绑定。"""

    plan_path = Path(plan_jsonl).resolve()
    state_path = Path(state_db).resolve()
    frozen_path = Path(frozen_dir).resolve()
    attempts_path = Path(attempts_dir).resolve()
    report_path = Path(report_json).resolve()
    for label, path in (
        ("plan_jsonl", plan_path),
        ("state_db", state_path),
        ("frozen_dir", frozen_path),
        ("attempts_dir", attempts_path),
        ("report_json", report_path),
    ):
        _reject_forbidden(str(path), context=label)
    if lease_seconds <= 0:
        raise FrozenPlanImportError("lease_seconds 必须为正数")
    try:
        plan = load_plan_jsonl(plan_path)
    except PlanContractError as exc:
        raise FrozenPlanImportError(f"plan 契约失败: {exc}") from exc
    planned = len(plan.entries)
    _reject_forbidden(plan_path.read_text(encoding="utf-8"), context="plan_jsonl")
    state_sha256_before = _sha256_file(state_path)
    state_content_sha256_before = _state_content_sha256(state_path)

    try:
        meta = _read_state_meta(state_path)
        rows = _task_rows(state_path, plan.entries)
        artifacts: dict[str, dict[str, object] | None] = {}
        for entry in plan.entries:
            row = rows.get(entry.evaluation_key)
            if row is None:
                raise FrozenPlanImportError(f"目标 state DB 缺少任务: {entry.logical_id}")
            if row["spec_json"] != entry.definition.task_spec.canonical_json():
                raise FrozenPlanImportError(f"目标 state DB 任务契约漂移: {entry.logical_id}")
            artifacts[entry.evaluation_key] = _validate_source_artifact(
                entry,
                frozen_dir=frozen_path,
                attempts_dir=attempts_path,
            )
            if row["state"] == "frozen":
                artifact = artifacts[entry.evaluation_key]
                if artifact is None:
                    raise FrozenPlanImportError(f"cached frozen source 缺失: {entry.logical_id}")
                if row["result_path"] != artifact["result_path"]:
                    raise FrozenPlanImportError(f"cached frozen result_path 漂移: {entry.logical_id}")
                if row["result_sha256"] != artifact["result_sha256"]:
                    raise FrozenPlanImportError(f"cached frozen SHA256 漂移: {entry.logical_id}")
    except FrozenPlanImportError as exc:
        _failure_report(
            report_path=report_path,
            plan_path=plan_path,
            plan_sha256=plan.plan_sha256,
            state_db=state_path,
            frozen_dir=frozen_path,
            attempts_dir=attempts_path,
            planned=planned,
            error=str(exc),
        )
        raise

    store = _state_store_from_meta(state_path, meta)
    entries_report: list[dict[str, object]] = []
    imported = cached = failed = missing = 0
    for entry in plan.entries:
        state = store.task_state(entry.evaluation_key)
        artifact = artifacts[entry.evaluation_key]
        if state == "frozen":
            cached += 1
            entries_report.append(
                {
                    "evaluation_key": entry.evaluation_key,
                    "logical_id": entry.logical_id,
                    "status": "cached",
                    "source_frozen_sha256": artifact["result_sha256"] if artifact else None,
                    "source_attempt_sha256": artifact["source_attempt_sha256"] if artifact else None,
                }
            )
            continue
        if artifact is None:
            missing += 1
            entries_report.append(
                {
                    "evaluation_key": entry.evaluation_key,
                    "logical_id": entry.logical_id,
                    "status": "missing",
                    "failure_class": "source_frozen_missing",
                }
            )
            continue
        if state not in {"pending", "retry_wait"}:
            failed += 1
            entries_report.append(
                {
                    "evaluation_key": entry.evaluation_key,
                    "logical_id": entry.logical_id,
                    "status": "failed",
                    "failure_class": "state_not_importable",
                    "state": state,
                }
            )
            continue

        lease = None
        audit_path = None
        try:
            if _sha256_file(Path(str(artifact["result_path"]))) != artifact["result_sha256"]:
                raise FrozenPlanImportError(f"{entry.logical_id}: source frozen 在导入前发生漂移")
            if _sha256_file(Path(str(artifact["source_attempt_path"]))) != artifact["source_attempt_sha256"]:
                raise FrozenPlanImportError(f"{entry.logical_id}: source attempt 在导入前发生漂移")
            lease = store.reserve_attempt(entry.evaluation_key, lease_seconds=lease_seconds)
            audit_path = attempts_path / f"{lease.attempt_id}.cache_import.json"
            if audit_path.exists():
                raise FrozenPlanImportError(f"cache import audit 已存在，拒绝覆盖: {audit_path}")
            audit_payload: dict[str, object] = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "status": "accepted",
                "provider": "cache_import",
                "transport": "local_artifact_rebinding",
                "network_request": False,
                "attempt_id": lease.attempt_id,
                "attempt_number": lease.attempt_number,
                "evaluation_key": entry.evaluation_key,
                "logical_id": entry.logical_id,
                "task_type": entry.definition.task_spec.task_type,
                "task_kind": _infer_task_kind(
                    entry.definition.task_spec.task_type, entry.definition.task_kind
                ),
                "plan_path": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "source_attempt_id": artifact["source_attempt_id"],
                "source_attempt_path": artifact["source_attempt_path"],
                "source_attempt_sha256": artifact["source_attempt_sha256"],
                "source_frozen_path": artifact["result_path"],
                "source_frozen_sha256": artifact["result_sha256"],
                "structured_output_sha256": artifact["structured_output_sha256"],
                "source_model": artifact["model"],
                "source_api_channel": artifact["api_channel"],
                "source_schema_version": artifact["schema_version"],
                "imported_at": time.time(),
            }
            _atomic_write_json(audit_path, audit_payload)
            store.freeze_result(
                lease.attempt_id,
                result_path=str(artifact["result_path"]),
                result_sha256=str(artifact["result_sha256"]),
            )
            imported += 1
            entries_report.append(
                {
                    "evaluation_key": entry.evaluation_key,
                    "logical_id": entry.logical_id,
                    "status": "imported",
                    "attempt_id": lease.attempt_id,
                    "audit_path": str(audit_path),
                    "audit_sha256": _sha256_file(audit_path),
                    "source_frozen_sha256": artifact["result_sha256"],
                }
            )
        except Exception as exc:
            failed += 1
            if lease is not None:
                try:
                    store.finish_failure(
                        lease.attempt_id,
                        error_class="cache_import_failed",
                        retryable=True,
                    )
                except StateContractError as cleanup_exc:
                    raise FrozenPlanImportError(
                        f"{entry.logical_id}: 导入失败且 running attempt 清理失败: {cleanup_exc}"
                    ) from exc
                if audit_path is not None:
                    _atomic_write_json(
                        audit_path,
                        {
                            "schema_version": AUDIT_SCHEMA_VERSION,
                            "status": "failed",
                            "provider": "cache_import",
                            "transport": "local_artifact_rebinding",
                            "network_request": False,
                            "attempt_id": lease.attempt_id,
                            "evaluation_key": entry.evaluation_key,
                            "logical_id": entry.logical_id,
                            "error": str(exc),
                            "source_frozen_path": artifact["result_path"],
                            "source_frozen_sha256": artifact["result_sha256"],
                        },
                    )
            entries_report.append(
                {
                    "evaluation_key": entry.evaluation_key,
                    "logical_id": entry.logical_id,
                    "status": "failed",
                    "failure_class": "state_not_ready" if lease is None else "cache_import_failed",
                    "error": str(exc),
                }
            )

    counts = {
        "planned": planned,
        "imported": imported,
        "cached": cached,
        "failed": failed,
        "missing": missing,
    }
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed" if failed == 0 and missing == 0 else "incomplete",
        "counts": counts,
        "network_requests": 0,
        "inputs": {
            "plan_jsonl": str(plan_path),
            "plan_sha256": plan.plan_sha256,
            "state_db": str(state_path),
            "state_db_sha256_before": state_sha256_before,
            "state_db_content_sha256_before": state_content_sha256_before,
            "frozen_dir": str(frozen_path),
            "attempts_dir": str(attempts_path),
            "source_artifacts_sha256": _sha256_text(
                canonical_json(
                    {
                        key: None
                        if value is None
                        else {
                            "result_sha256": value["result_sha256"],
                            "source_attempt_sha256": value["source_attempt_sha256"],
                        }
                        for key, value in sorted(artifacts.items())
                    }
                )
            ),
        },
        "outputs": {
            "state_db": str(state_path),
            "state_db_sha256": _sha256_file(state_path),
            "state_db_content_sha256": _state_content_sha256(state_path),
            "report_json": str(report_path),
            "import_audits_sha256": _sha256_text(
                canonical_json(
                    [
                        {
                            "attempt_id": row.get("attempt_id"),
                            "audit_sha256": row.get("audit_sha256"),
                        }
                        for row in entries_report
                        if row.get("status") == "imported"
                    ]
                )
            ),
        },
        "entries": entries_report,
    }
    _atomic_write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="安全导入已有 Stage5 API frozen 结果")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--lease-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = import_frozen_plan_results(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            frozen_dir=args.frozen_dir,
            attempts_dir=args.attempt_dir,
            report_json=args.report_json,
            lease_seconds=args.lease_seconds,
        )
    except FrozenPlanImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
