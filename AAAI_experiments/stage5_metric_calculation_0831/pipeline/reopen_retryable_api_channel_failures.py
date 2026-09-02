"""审计并重开被旧 API 分类器过早耗尽的渠道级失败。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_claude_plan import _load_predecessor_attempt_manifest
from .state import PredecessorAttemptManifest, StateContractError, TaskStateStore


class ApiChannelFailureReopenError(RuntimeError):
    """状态库与 attempt 审计证据不满足重开契约。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApiChannelFailureReopenError(f"attempt 审计文件不可读: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ApiChannelFailureReopenError(f"attempt 审计文件不是 JSON object: {path}")
    return payload


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ApiChannelFailureReopenError(f"{context} 必须是 JSON object")
    return value


def _load_candidates(
    *,
    state_db: Path,
    condition: str,
    task_type: str,
) -> list[sqlite3.Row]:
    if not state_db.is_file():
        raise ApiChannelFailureReopenError(f"状态库不存在: {state_db}")
    connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """SELECT a.attempt_id, a.attempt_number, a.evaluation_key,
                      a.error_class, a.retryable, t.logical_id, t.task_type,
                      t.condition_name, t.attempt_count
               FROM tasks AS t
               JOIN attempts AS a
                 ON a.evaluation_key=t.evaluation_key
                AND a.attempt_number=t.attempt_count
               WHERE t.condition_name=?
                 AND t.task_type=?
                 AND t.state='exhausted'
                 AND t.attempt_count < 3
                 AND a.status='failed'
                 AND a.error_class IN ('api_auth_error', 'api_http_error')
               ORDER BY t.logical_id""",
            (condition, task_type),
        ).fetchall()
    finally:
        connection.close()


def _load_state_limits(state_db: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    finally:
        connection.close()
    try:
        return {
            "attempt_cap": int(meta["attempt_cap"]),
            "logical_task_cap": int(meta["logical_task_cap"]),
            "max_attempts_per_task": int(meta["max_attempts_per_task"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiChannelFailureReopenError("状态库缺少冻结预算元数据") from exc


def _validated_candidate(
    row: sqlite3.Row,
    *,
    attempts_dir: Path,
    failed_channel: str,
) -> dict[str, object] | None:
    attempt_id = str(row["attempt_id"])
    attempt_path = (attempts_dir / f"{attempt_id}.json").resolve()
    payload = _read_json_object(attempt_path)
    metadata = _require_mapping(payload.get("metadata"), context=f"{attempt_id}.metadata")
    validation = _require_mapping(
        payload.get("validation"),
        context=f"{attempt_id}.validation",
    )
    expected = {
        "attempt_id": attempt_id,
        "attempt_number": int(row["attempt_number"]),
        "evaluation_key": str(row["evaluation_key"]),
        "logical_id": str(row["logical_id"]),
        "task_type": str(row["task_type"]),
        "error_class": str(row["error_class"]),
        "retryable": False,
    }
    for field_name, expected_value in expected.items():
        if metadata.get(field_name) != expected_value:
            raise ApiChannelFailureReopenError(
                f"{attempt_id}.metadata.{field_name} 漂移: "
                f"{metadata.get(field_name)!r} != {expected_value!r}"
            )
    if payload.get("attempt_id") != attempt_id or payload.get("evaluation_key") != row["evaluation_key"]:
        raise ApiChannelFailureReopenError(f"{attempt_id}: 顶层 attempt/evaluation 绑定漂移")
    if validation.get("ok") is not False or validation.get("error_class") != row["error_class"]:
        raise ApiChannelFailureReopenError(f"{attempt_id}: validation 状态漂移")
    channel = metadata.get("api_channel")
    http_status = metadata.get("http_status")
    error_message = str(validation.get("error_message") or "")
    error_class = str(row["error_class"])
    if error_class == "api_auth_error":
        qualified = channel == failed_channel and http_status in {401, 403}
        reclassified_error_class = "api_channel_unavailable"
        qualification = "failed_channel_auth_http_401_403"
    else:
        normalized = error_message.lower()
        qualified = http_status == 400 and any(
            marker in normalized
            for marker in (
                "allmodelsfailed",
                "deployment request could not be completed",
                "555420",
            )
        )
        reclassified_error_class = "api_http_transient"
        qualification = "legacy_upstream_deployment_http_400"
    if not qualified:
        return None
    return {
        "attempt_id": attempt_id,
        "evaluation_key": str(row["evaluation_key"]),
        "logical_id": str(row["logical_id"]),
        "attempt_number": int(row["attempt_number"]),
        "original_error_class": error_class,
        "reclassified_error_class": reclassified_error_class,
        "api_channel": str(channel),
        "http_status": int(http_status),
        "qualification": qualification,
        "attempt_path": str(attempt_path),
        "attempt_sha256": _sha256_file(attempt_path),
    }


def reopen_retryable_api_channel_failures(
    *,
    state_db: str | Path,
    attempts_dir: str | Path,
    condition: str,
    task_type: str,
    failed_channel: str,
    audit_reason: str,
    apply: bool = False,
    predecessor_attempt_manifest: str | Path | None = None,
    report_json: str | Path | None = None,
) -> dict[str, object]:
    if not condition or not task_type or not failed_channel or not audit_reason:
        raise ApiChannelFailureReopenError("condition/task_type/failed_channel/audit_reason 均为必填")
    state_path = Path(state_db).resolve()
    attempt_root = Path(attempts_dir).resolve()
    raw_candidates = _load_candidates(
        state_db=state_path,
        condition=condition,
        task_type=task_type,
    )
    candidates = [
        candidate
        for row in raw_candidates
        if (
            candidate := _validated_candidate(
                row,
                attempts_dir=attempt_root,
                failed_channel=failed_channel,
            )
        )
        is not None
    ]

    loaded_predecessor = None
    if predecessor_attempt_manifest is not None:
        loaded_predecessor = _load_predecessor_attempt_manifest(predecessor_attempt_manifest)
    store = TaskStateStore(
        state_path,
        **_load_state_limits(state_path),
        predecessor_attempt_manifest=(
            PredecessorAttemptManifest(
                path=str(loaded_predecessor.path),
                sha256=loaded_predecessor.sha256,
                attempt_count=loaded_predecessor.attempt_count,
            )
            if loaded_predecessor is not None
            else None
        ),
    )
    reopened = 0
    if apply:
        for candidate in candidates:
            store.reopen_exhausted_failed_attempt(
                str(candidate["attempt_id"]),
                allowed_error_classes=(str(candidate["original_error_class"]),),
                reclassified_error_class=str(candidate["reclassified_error_class"]),
                audit_reason=audit_reason,
            )
            reopened += 1
    report: dict[str, object] = {
        "schema_version": "api_channel_failure_reopen.v1",
        "created_at": time.time(),
        "mode": "apply" if apply else "dry_run",
        "state_db": str(state_path),
        "attempts_dir": str(attempt_root),
        "condition": condition,
        "task_type": task_type,
        "failed_channel": failed_channel,
        "audit_reason": audit_reason,
        "counts": {
            "candidate": len(candidates),
            "reopened": reopened,
        },
        "candidates": candidates,
        "state_summary": store.state_summary(),
    }
    if report_json is not None:
        _atomic_write_json(Path(report_json), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--failed-channel", required=True)
    parser.add_argument("--audit-reason", required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", dest="apply", action="store_false")
    parser.set_defaults(apply=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = reopen_retryable_api_channel_failures(
            state_db=args.state_db,
            attempts_dir=args.attempts_dir,
            condition=args.condition,
            task_type=args.task_type,
            failed_channel=args.failed_channel,
            audit_reason=args.audit_reason,
            apply=args.apply,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            report_json=args.report_json,
        )
    except (ApiChannelFailureReopenError, StateContractError, OSError) as exc:
        print(f"API 渠道失败重开失败: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
