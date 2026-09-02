"""独立复核由 Anthropic Messages API 冻结的 simplify 结果。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    AnthropicApiResponse,
    AnthropicApiRunner,
)
from .audit_frozen_simplifications import run_isolated_simplify_semantic_validator
from .claude_contract import (
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
    canonical_json,
    render_prompt,
)
from .claude_runner import _infer_task_kind, _redact_string
from .run_claude_plan import PlanContractError, load_plan_jsonl


JsonDict = dict[str, object]


class ApiFrozenAuditError(RuntimeError):
    """API 冻结结果或其状态绑定不满足审计约束。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json(path: Path, *, context: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApiFrozenAuditError(f"{context} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise ApiFrozenAuditError(f"{context} 顶层不是 JSON object")
    return payload


def _failure(entry: object, failure_class: str, error: str) -> JsonDict:
    return {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "status": "failed",
        "failure_class": failure_class,
        "error": error,
    }


def _audit_one(
    *,
    entry: object,
    state_row: Mapping[str, object],
    attempts_dir: Path,
    frozen_dir: Path,
    semantic_timeout_seconds: float,
) -> JsonDict:
    expected_result_path = (frozen_dir / f"{entry.evaluation_key}.json").resolve()
    result_path = Path(str(state_row["result_path"])).resolve()
    if result_path != expected_result_path or not result_path.is_file():
        return _failure(entry, "frozen_binding_error", "API frozen 路径绑定不合法")
    if _sha256_file(result_path) != state_row["result_sha256"]:
        return _failure(entry, "frozen_sha256_mismatch", "API frozen SHA-256 漂移")
    payload = _read_json(result_path, context=f"frozen {entry.logical_id}")
    definition = entry.definition
    task_kind = _infer_task_kind(definition.task_spec.task_type, definition.task_kind)
    expected_identity = {
        "attempt_id": state_row["attempt_id"],
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": definition.task_spec.task_type,
        "task_kind": task_kind,
        "request": definition.request,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            return _failure(entry, "frozen_identity_error", f"frozen.{field} 漂移")
    expected_prompt = render_prompt(
        definition.prompt_template,
        definition.request,
        definition.schema,
    )
    if payload.get("prompt") != expected_prompt:
        return _failure(entry, "prompt_mismatch", "API frozen prompt 与计划不一致")
    metadata = payload.get("metadata")
    validation = payload.get("validation")
    envelope = payload.get("envelope")
    if not isinstance(metadata, Mapping) or not isinstance(validation, Mapping):
        return _failure(entry, "frozen_artifact_invalid", "metadata/validation 缺失")
    if not isinstance(envelope, Mapping):
        return _failure(entry, "frozen_artifact_invalid", "API envelope 缺失")
    expected_metadata = {
        "attempt_id": state_row["attempt_id"],
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
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            return _failure(entry, "metadata_identity_error", f"metadata.{field} 漂移")
    if metadata.get("api_channel") not in {"routify", "yapi"}:
        return _failure(entry, "metadata_identity_error", "metadata.api_channel 非法")
    if validation.get("ok") is not True or validation.get("error_class") is not None:
        return _failure(entry, "validation_state_error", "API frozen validation 不是成功状态")
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
        return _failure(entry, "api_contract_error", str(exc))
    if payload.get("structured_output") != structured_output:
        return _failure(entry, "structured_output_mismatch", "顶层输出与 API envelope 不一致")
    if validation.get("structured_output") != structured_output:
        return _failure(entry, "structured_output_mismatch", "validation 输出与 API envelope 不一致")

    attempt_path = (attempts_dir / f"{state_row['attempt_id']}.json").resolve()
    if not attempt_path.is_file():
        return _failure(entry, "attempt_artifact_missing", "API attempt 审计文件不存在")
    attempt_payload = _read_json(attempt_path, context=f"attempt {entry.logical_id}")
    expected_attempt_fields = {
        "attempt_id": state_row["attempt_id"],
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
            return _failure(entry, "attempt_frozen_mismatch", f"attempt.{field} 与 frozen 不一致")
    audit_raw = attempt_path.read_text(encoding="utf-8") + result_path.read_text(encoding="utf-8")
    if _redact_string(audit_raw) != audit_raw or "Authorization" in audit_raw:
        return _failure(entry, "secret_leak", "API 审计文件包含疑似凭据")

    semantic = run_isolated_simplify_semantic_validator(
        evaluation_key=entry.evaluation_key,
        request=definition.request,
        structured_output=structured_output,
        timeout_seconds=semantic_timeout_seconds,
    )
    if semantic.get("status") != "promotable":
        return _failure(
            entry,
            str(semantic.get("status") or "semantic_revalidation_failed"),
            str(semantic.get("error") or "API simplify 独立语义复核失败"),
        )
    semantic_evidence = semantic.get("semantic_evidence")
    return {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "status": "passed",
        "failure_class": None,
        "api_channel": metadata.get("api_channel"),
        "attempt_id": state_row["attempt_id"],
        "attempt_path": str(attempt_path),
        "attempt_sha256": _sha256_file(attempt_path),
        "frozen_path": str(result_path),
        "frozen_sha256": _sha256_file(result_path),
        "semantic_decision": (
            semantic_evidence.get("decision")
            if isinstance(semantic_evidence, Mapping)
            else None
        ),
        "semantic_evidence_sha256": (
            hashlib.sha256(canonical_json(semantic_evidence).encode("utf-8")).hexdigest()
            if isinstance(semantic_evidence, Mapping)
            else None
        ),
    }


def audit_api_frozen_simplifications(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    frozen_dir: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    expected_api_count: int,
    semantic_timeout_seconds: float = 300.0,
    workers: int = 4,
) -> JsonDict:
    if expected_api_count <= 0 or workers <= 0 or semantic_timeout_seconds <= 0:
        raise ApiFrozenAuditError("数量、并发与超时参数必须为正数")
    try:
        plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise ApiFrozenAuditError(f"plan 契约失败: {exc}") from exc
    frozen_path = Path(frozen_dir).resolve()
    attempts_path = Path(attempts_dir).resolve()
    state_path = Path(state_db).resolve()
    with sqlite3.connect(f"{state_path.as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows_by_key = {
            str(row["evaluation_key"]): dict(row)
            for row in connection.execute(
                """SELECT t.evaluation_key, t.state, f.attempt_id, f.result_path,
                          f.result_sha256
                   FROM tasks t
                   LEFT JOIN frozen_results f ON f.evaluation_key=t.evaluation_key"""
            ).fetchall()
        }
    selected: list[tuple[object, Mapping[str, object]]] = []
    for entry in plan.entries:
        row = rows_by_key.get(entry.evaluation_key)
        if row is None or not isinstance(row.get("result_path"), str):
            continue
        if Path(str(row["result_path"])).resolve().parent == frozen_path:
            if row.get("state") != "frozen":
                raise ApiFrozenAuditError(f"API 结果任务未冻结: {entry.logical_id}")
            selected.append((entry, row))
    if len(selected) != expected_api_count:
        raise ApiFrozenAuditError(
            f"API frozen 数量不符: {len(selected)} != {expected_api_count}"
        )
    planned_filenames = {f"{entry.evaluation_key}.json" for entry, _ in selected}
    actual_filenames = {path.name for path in frozen_path.glob("*.json")}
    if actual_filenames != planned_filenames:
        raise ApiFrozenAuditError("API frozen 目录存在计划外文件或缺失计划文件")

    results: list[JsonDict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _audit_one,
                entry=entry,
                state_row=row,
                attempts_dir=attempts_path,
                frozen_dir=frozen_path,
                semantic_timeout_seconds=semantic_timeout_seconds,
            )
            for entry, row in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["logical_id"]))
    output_path = Path(output_jsonl).resolve()
    _atomic_write_text(
        output_path,
        "".join(canonical_json(row) + "\n" for row in results),
    )
    passed = sum(row["status"] == "passed" for row in results)
    failure_classes: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    for row in results:
        if row["status"] != "passed":
            key = str(row.get("failure_class"))
            failure_classes[key] = failure_classes.get(key, 0) + 1
        else:
            key = str(row.get("api_channel"))
            channel_counts[key] = channel_counts.get(key, 0) + 1
    report: JsonDict = {
        "status": "ok" if passed == len(results) else "failed",
        "model_invoked": False,
        "plan_jsonl": str(Path(plan_jsonl).resolve()),
        "plan_sha256": plan.plan_sha256,
        "state_db": str(state_path),
        "attempts_dir": str(attempts_path),
        "frozen_dir": str(frozen_path),
        "expected_api_count": expected_api_count,
        "audited_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "failure_class_counts": failure_classes,
        "channel_counts": channel_counts,
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "semantic_timeout_seconds": semantic_timeout_seconds,
        "workers": workers,
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-api-count", type=int, required=True)
    parser.add_argument("--semantic-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_api_frozen_simplifications(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            frozen_dir=args.frozen_dir,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            expected_api_count=args.expected_api_count,
            semantic_timeout_seconds=args.semantic_timeout_seconds,
            workers=args.workers,
        )
    except ApiFrozenAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
