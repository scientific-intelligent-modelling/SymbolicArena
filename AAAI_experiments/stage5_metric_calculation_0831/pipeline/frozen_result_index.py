"""冻结结果索引构建器。

读取 canonical plan JSONL 与只读状态库，校验每个计划任务都已形成
可审计闭环，并输出可按 `logical_id` / `evaluation_key` 检索的规范 JSONL。
默认只接受 `frozen` 与 `non_applicable`。若显式启用 `allow_exhausted`，
则仅允许 `pred_simplify` 额外落入严格审计过的 `exhausted` 终态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    validate_structured_output,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    PlanContractError,
    load_plan_jsonl,
)


JsonDict = dict[str, object]


class FrozenResultIndexError(RuntimeError):
    """冻结结果索引构建过程中的硬错误。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenResultIndexError(f"{context} 必须是非空字符串")
    return value


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrozenResultIndexError(f"{context} 必须是 JSON object")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = _require_string(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FrozenResultIndexError(f"{context} 必须是小写十六进制 SHA256")
    return text


def _infer_task_kind(task_type: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if task_type.endswith("simplify"):
        return "simplify"
    if task_type.endswith("equivalence"):
        return "equivalence"
    if task_type.endswith("structure"):
        return "structure"
    raise FrozenResultIndexError(f"无法从 task_type 推断 task_kind: {task_type!r}")


def _read_json_object(path: Path, *, context: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FrozenResultIndexError(f"{context} 不可解析: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenResultIndexError(f"{context} 不是 JSON object")
    return payload


def _connect_read_only(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.is_file():
        raise FrozenResultIndexError(f"状态库不存在: {db_path}")
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise FrozenResultIndexError(f"状态库打开失败: {exc}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _fetch_one(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence[object],
    *,
    context: str,
) -> sqlite3.Row | None:
    try:
        return connection.execute(query, parameters).fetchone()
    except sqlite3.Error as exc:
        raise FrozenResultIndexError(f"{context} 查询失败: {exc}") from exc


def _validate_unique_entries(entries: Sequence[object]) -> None:
    seen_evaluation_keys: dict[str, str] = {}
    seen_logical_ids: dict[str, str] = {}
    for entry in entries:
        evaluation_key = entry.evaluation_key
        logical_id = entry.logical_id
        previous_logical = seen_evaluation_keys.get(evaluation_key)
        if previous_logical is not None:
            raise FrozenResultIndexError(
                f"plan 存在重复 evaluation_key: {evaluation_key} "
                f"(logical_id={previous_logical!r} 与 {logical_id!r})"
            )
        previous_key = seen_logical_ids.get(logical_id)
        if previous_key is not None:
            raise FrozenResultIndexError(
                f"plan 存在重复 logical_id: {logical_id} "
                f"(evaluation_key={previous_key!r} 与 {evaluation_key!r})"
            )
        seen_evaluation_keys[evaluation_key] = logical_id
        seen_logical_ids[logical_id] = evaluation_key


def _validate_task_row(task_row: sqlite3.Row, *, expected_spec_json: str, context: str) -> str:
    state = _require_string(task_row["state"], context=f"{context}.state")
    if state not in {"frozen", "non_applicable"}:
        raise FrozenResultIndexError(f"{context} 未形成 frozen/non_applicable 闭环: {state}")
    spec_json = _require_string(task_row["spec_json"], context=f"{context}.spec_json")
    if spec_json != expected_spec_json:
        raise FrozenResultIndexError(f"{context} 任务契约发生漂移")
    return state


def _validate_task_row_with_exhausted(
    task_row: sqlite3.Row,
    *,
    expected_spec_json: str,
    context: str,
    allow_exhausted: bool,
) -> str:
    if not allow_exhausted:
        return _validate_task_row(
            task_row,
            expected_spec_json=expected_spec_json,
            context=context,
        )
    state = _require_string(task_row["state"], context=f"{context}.state")
    if state not in {"frozen", "non_applicable", "exhausted"}:
        raise FrozenResultIndexError(
            f"{context} 未形成 frozen/non_applicable/exhausted 闭环: {state}"
        )
    spec_json = _require_string(task_row["spec_json"], context=f"{context}.spec_json")
    if spec_json != expected_spec_json:
        raise FrozenResultIndexError(f"{context} 任务契约发生漂移")
    return state


def _validate_file_sha256(
    path: Path,
    *,
    expected_sha256: str,
    context: str,
    artifact_label: str,
) -> None:
    if not path.is_file():
        raise FrozenResultIndexError(f"{context} {artifact_label}不存在: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FrozenResultIndexError(f"{context} {artifact_label} SHA256 漂移")


def _validate_optional_plan_identity(
    payload: Mapping[str, object],
    *,
    plan_sha256: str,
    context: str,
) -> None:
    plan_sha_value = payload.get("plan_sha256")
    if plan_sha_value is None:
        return
    payload_plan_sha256 = _require_sha256(
        plan_sha_value,
        context=f"{context}.plan_sha256",
    )
    if payload_plan_sha256 != plan_sha256:
        raise FrozenResultIndexError(f"{context}.plan_sha256 漂移")


def _validate_non_applicable_evidence_payload(
    payload: Mapping[str, object],
    *,
    entry: object,
    reason: str,
    evidence_sha256: str,
    plan_sha256: str,
    context: str,
) -> None:
    request = entry.definition.request
    request_evidence_hash = _require_sha256(
        request.get("evidence_hash"),
        context=f"{context}.plan_request.evidence_hash",
    )
    if request_evidence_hash != evidence_sha256:
        raise FrozenResultIndexError(
            f"{context}.plan_request.evidence_hash 与证据文件 SHA256 不一致"
        )
    expected_pairs = {
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "condition": entry.definition.task_spec.condition,
        "reason": reason,
    }
    for field_name, expected_value in expected_pairs.items():
        actual_value = _require_string(payload.get(field_name), context=f"{context}.{field_name}")
        if actual_value != expected_value:
            raise FrozenResultIndexError(f"{context}.{field_name} 漂移")
    expected_phase_by_task_type = {
        "gt_simplify": "gt",
        "pred_simplify": "pred",
        "equivalence": "equivalence",
        "stab_structure": "structure",
    }
    phase = _require_string(payload.get("phase"), context=f"{context}.phase")
    expected_phase = expected_phase_by_task_type.get(entry.definition.task_spec.task_type)
    if expected_phase is not None and phase != expected_phase:
        raise FrozenResultIndexError(f"{context}.phase 漂移")
    dependencies = payload.get("dependencies")
    expected_dependencies = list(entry.definition.task_spec.dependencies)
    if dependencies != expected_dependencies:
        raise FrozenResultIndexError(f"{context}.dependencies 漂移")
    request_context = _require_mapping(
        payload.get("request_context"),
        context=f"{context}.request_context",
    )
    expected_request_context = {
        key: value for key, value in request.items() if key != "evidence_hash"
    }
    if dict(request_context) != expected_request_context:
        raise FrozenResultIndexError(f"{context}.request_context 与 plan request 不一致")
    # evaluation_key 依赖证据文件 SHA，不能反向写进证据文件本身，否则会形成
    # 循环指纹。旧证据若携带该字段仍严格核验；新证据由 plan 顶层和状态库绑定。
    payload_evaluation_key = payload.get("evaluation_key")
    if payload_evaluation_key is not None:
        actual_evaluation_key = _require_string(
            payload_evaluation_key,
            context=f"{context}.evaluation_key",
        )
        if actual_evaluation_key != entry.evaluation_key:
            raise FrozenResultIndexError(f"{context}.evaluation_key 漂移")
    _validate_optional_plan_identity(
        payload,
        plan_sha256=plan_sha256,
        context=context,
    )


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenResultIndexError(f"{context} 必须是整数")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise FrozenResultIndexError(f"{context} 必须是布尔值")
    return value


def _require_sqlite_bool(value: object, *, context: str) -> bool:
    integer = _require_int(value, context=context)
    if integer not in {0, 1}:
        raise FrozenResultIndexError(f"{context} 必须是 SQLite 布尔值 0 或 1")
    return bool(integer)


def _build_frozen_row(
    *,
    connection: sqlite3.Connection,
    entry: object,
    plan_sha256: str,
) -> JsonDict:
    context = f"任务 {entry.logical_id}"
    frozen_row = _fetch_one(
        connection,
        """SELECT result_path, result_sha256, attempt_id
           FROM frozen_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.frozen_results",
    )
    if frozen_row is None:
        raise FrozenResultIndexError(f"{context} 缺失 frozen_results 闭环记录")
    contradictory = _fetch_one(
        connection,
        """SELECT 1
           FROM non_applicable_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.non_applicable_results",
    )
    if contradictory is not None:
        raise FrozenResultIndexError(f"{context} 同时存在 frozen 与 non_applicable 记录")

    result_path = Path(_require_string(frozen_row["result_path"], context=f"{context}.result_path"))
    expected_sha256 = _require_sha256(
        frozen_row["result_sha256"],
        context=f"{context}.result_sha256",
    )
    attempt_id = _require_string(frozen_row["attempt_id"], context=f"{context}.attempt_id")
    _validate_file_sha256(
        result_path,
        expected_sha256=expected_sha256,
        context=context,
        artifact_label="结果文件",
    )

    payload = _read_json_object(result_path, context=f"{context}.result_json")
    task_kind = _infer_task_kind(entry.definition.task_spec.task_type, entry.definition.task_kind)
    if payload.get("evaluation_key") != entry.evaluation_key:
        raise FrozenResultIndexError(f"{context} result_json.evaluation_key 漂移")
    if payload.get("logical_id") != entry.logical_id:
        raise FrozenResultIndexError(f"{context} result_json.logical_id 漂移")
    if payload.get("task_type") != entry.definition.task_spec.task_type:
        raise FrozenResultIndexError(f"{context} result_json.task_type 漂移")
    if payload.get("task_kind") != task_kind:
        raise FrozenResultIndexError(f"{context} result_json.task_kind 漂移")
    _validate_optional_plan_identity(
        payload,
        plan_sha256=plan_sha256,
        context=f"{context}.result_json",
    )

    structured_output = payload.get("structured_output")
    try:
        validated_structured_output = validate_structured_output(
            task_kind,
            _require_mapping(structured_output, context=f"{context}.structured_output"),
        )
    except Exception as exc:
        raise FrozenResultIndexError(f"{context} structured_output 非法: {exc}") from exc

    return {
        "plan_sha256": plan_sha256,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "task_kind": task_kind,
        "condition": entry.definition.task_spec.condition,
        "priority": entry.definition.task_spec.priority,
        "state": "frozen",
        "attempt_id": attempt_id,
        "result_path": str(result_path),
        "result_sha256": expected_sha256,
        "structured_output": validated_structured_output,
        "non_applicable": None,
        "exhausted": None,
    }


def _build_non_applicable_row(
    *,
    connection: sqlite3.Connection,
    entry: object,
    plan_sha256: str,
) -> JsonDict:
    context = f"任务 {entry.logical_id}"
    non_applicable_row = _fetch_one(
        connection,
        """SELECT reason, evidence_path, evidence_sha256
           FROM non_applicable_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.non_applicable_results",
    )
    if non_applicable_row is None:
        raise FrozenResultIndexError(f"{context} 缺失 non_applicable 闭环记录")
    contradictory = _fetch_one(
        connection,
        """SELECT 1
           FROM frozen_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.frozen_results",
    )
    if contradictory is not None:
        raise FrozenResultIndexError(f"{context} 同时存在 non_applicable 与 frozen 记录")

    reason = _require_string(non_applicable_row["reason"], context=f"{context}.reason")
    evidence_path = _require_string(
        non_applicable_row["evidence_path"],
        context=f"{context}.evidence_path",
    )
    evidence_sha256 = _require_sha256(
        non_applicable_row["evidence_sha256"],
        context=f"{context}.evidence_sha256",
    )
    evidence_path_obj = Path(evidence_path)
    _validate_file_sha256(
        evidence_path_obj,
        expected_sha256=evidence_sha256,
        context=context,
        artifact_label="证据文件",
    )
    evidence_payload = _read_json_object(
        evidence_path_obj,
        context=f"{context}.evidence_json",
    )
    _validate_non_applicable_evidence_payload(
        evidence_payload,
        entry=entry,
        reason=reason,
        evidence_sha256=evidence_sha256,
        plan_sha256=plan_sha256,
        context=f"{context}.evidence_json",
    )
    task_kind = _infer_task_kind(entry.definition.task_spec.task_type, entry.definition.task_kind)

    return {
        "plan_sha256": plan_sha256,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "task_kind": task_kind,
        "condition": entry.definition.task_spec.condition,
        "priority": entry.definition.task_spec.priority,
        "state": "non_applicable",
        "attempt_id": None,
        "result_path": None,
        "result_sha256": None,
        "structured_output": None,
        "non_applicable": {
            "reason": reason,
            "evidence_path": str(evidence_path_obj),
            "evidence_sha256": evidence_sha256,
        },
        "exhausted": None,
    }


def _validate_attempt_payload(
    payload: Mapping[str, object],
    *,
    entry: object,
    attempt_id: str,
    attempt_number: int,
    error_class: str,
    retryable: bool,
    context: str,
) -> None:
    expected_attempt_id = f"{entry.evaluation_key}.a{attempt_number:02d}"
    if attempt_id != expected_attempt_id:
        raise FrozenResultIndexError(
            f"{context}.attempt_id 不符合 evaluation_key + attempt_number 契约"
        )
    payload_attempt_id = _require_string(
        payload.get("attempt_id"),
        context=f"{context}.attempt_json.attempt_id",
    )
    if payload_attempt_id != attempt_id:
        raise FrozenResultIndexError(f"{context}.attempt_json.attempt_id 漂移")
    payload_evaluation_key = _require_string(
        payload.get("evaluation_key"),
        context=f"{context}.attempt_json.evaluation_key",
    )
    if payload_evaluation_key != entry.evaluation_key:
        raise FrozenResultIndexError(f"{context}.attempt_json.evaluation_key 漂移")
    metadata = _require_mapping(
        payload.get("metadata"),
        context=f"{context}.attempt_json.metadata",
    )
    expected_pairs = {
        "attempt_id": attempt_id,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "error_class": error_class,
    }
    for field_name, expected_value in expected_pairs.items():
        actual_value = _require_string(
            metadata.get(field_name),
            context=f"{context}.attempt_json.metadata.{field_name}",
        )
        if actual_value != expected_value:
            raise FrozenResultIndexError(f"{context}.attempt_json.metadata.{field_name} 漂移")
    actual_attempt_number = _require_int(
        metadata.get("attempt_number"),
        context=f"{context}.attempt_json.metadata.attempt_number",
    )
    if actual_attempt_number != attempt_number:
        raise FrozenResultIndexError(f"{context}.attempt_json.metadata.attempt_number 漂移")
    actual_retryable = _require_bool(
        metadata.get("retryable"),
        context=f"{context}.attempt_json.metadata.retryable",
    )
    if actual_retryable is not retryable:
        raise FrozenResultIndexError(f"{context}.attempt_json.metadata.retryable 漂移")
    validation = _require_mapping(
        payload.get("validation"),
        context=f"{context}.attempt_json.validation",
    )
    if validation.get("ok") is not False:
        raise FrozenResultIndexError(f"{context}.attempt_json.validation.ok 必须为 false")
    validation_error_class = _require_string(
        validation.get("error_class"),
        context=f"{context}.attempt_json.validation.error_class",
    )
    if validation_error_class != error_class:
        raise FrozenResultIndexError(
            f"{context}.attempt_json.validation.error_class 漂移"
        )


def _build_exhausted_row(
    *,
    connection: sqlite3.Connection,
    entry: object,
    plan_sha256: str,
    attempts_dir: Path | None,
    task_row: sqlite3.Row,
) -> JsonDict:
    context = f"任务 {entry.logical_id}"
    if entry.definition.task_spec.task_type != "pred_simplify":
        raise FrozenResultIndexError(f"{context} 仅 pred_simplify 允许 exhausted 终态")
    if attempts_dir is None:
        raise FrozenResultIndexError(f"{context} 启用 exhausted 索引时必须提供 attempts_dir")
    task_attempt_count = _require_int(
        task_row["attempt_count"],
        context=f"{context}.tasks.attempt_count",
    )
    if task_attempt_count != 3:
        raise FrozenResultIndexError(f"{context} exhausted 的 tasks.attempt_count 必须为 3")
    contradictory_frozen = _fetch_one(
        connection,
        """SELECT 1
           FROM frozen_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.frozen_results",
    )
    if contradictory_frozen is not None:
        raise FrozenResultIndexError(f"{context} exhausted 与 frozen 记录冲突")
    contradictory_non_applicable = _fetch_one(
        connection,
        """SELECT 1
           FROM non_applicable_results
           WHERE evaluation_key = ?""",
        (entry.evaluation_key,),
        context=f"{context}.non_applicable_results",
    )
    if contradictory_non_applicable is not None:
        raise FrozenResultIndexError(f"{context} exhausted 与 non_applicable 记录冲突")
    attempt_rows = connection.execute(
        """SELECT attempt_id, attempt_number, status, error_class, retryable
           FROM attempts
           WHERE evaluation_key = ?
           ORDER BY attempt_number ASC""",
        (entry.evaluation_key,),
    ).fetchall()
    if len(attempt_rows) != 3:
        raise FrozenResultIndexError(f"{context} exhausted 必须恰有 3 次 failed attempts")
    attempts: list[dict[str, object]] = []
    for expected_number, attempt_row in enumerate(attempt_rows, start=1):
        attempt_id = _require_string(
            attempt_row["attempt_id"],
            context=f"{context}.attempt_id[{expected_number}]",
        )
        attempt_number = _require_int(
            attempt_row["attempt_number"],
            context=f"{context}.attempt_number[{expected_number}]",
        )
        if attempt_number != expected_number:
            raise FrozenResultIndexError(f"{context} attempt_number 序列不连续")
        status = _require_string(
            attempt_row["status"],
            context=f"{context}.attempt_status[{expected_number}]",
        )
        if status != "failed":
            raise FrozenResultIndexError(f"{context} exhausted 只允许 failed attempts")
        error_class = _require_string(
            attempt_row["error_class"],
            context=f"{context}.error_class[{expected_number}]",
        )
        retryable = _require_sqlite_bool(
            attempt_row["retryable"],
            context=f"{context}.retryable[{expected_number}]",
        )
        attempt_path = attempts_dir / f"{attempt_id}.json"
        if not attempt_path.is_file():
            raise FrozenResultIndexError(f"{context} attempt 文件不存在: {attempt_path}")
        attempt_sha256 = _sha256_file(attempt_path)
        payload = _read_json_object(
            attempt_path,
            context=f"{context}.attempt_json[{expected_number}]",
        )
        _validate_attempt_payload(
            payload,
            entry=entry,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            error_class=error_class,
            retryable=retryable,
            context=context,
        )
        attempts.append(
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
    last_error_class = _require_string(
        attempt_rows[-1]["error_class"],
        context=f"{context}.last_error_class",
    )
    task_last_error_class = _require_string(
        task_row["last_error_class"],
        context=f"{context}.tasks.last_error_class",
    )
    if task_last_error_class != last_error_class:
        raise FrozenResultIndexError(f"{context}.tasks.last_error_class 与最后 attempt 不一致")
    task_kind = _infer_task_kind(entry.definition.task_spec.task_type, entry.definition.task_kind)
    return {
        "plan_sha256": plan_sha256,
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "task_kind": task_kind,
        "condition": entry.definition.task_spec.condition,
        "priority": entry.definition.task_spec.priority,
        "state": "exhausted",
        "attempt_id": None,
        "result_path": None,
        "result_sha256": None,
        "structured_output": None,
        "non_applicable": None,
        "exhausted": {
            "attempt_count": 3,
            "last_error_class": last_error_class,
            "attempts": attempts,
        },
    }


def build_frozen_result_index(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    allow_exhausted: bool = False,
    attempts_dir: str | Path | None = None,
) -> JsonDict:
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise FrozenResultIndexError(f"plan 契约错误: {exc}") from exc

    entries = sorted(loaded_plan.entries, key=lambda entry: (entry.logical_id, entry.evaluation_key))
    _validate_unique_entries(entries)

    rows: list[JsonDict] = []
    state_counts = {"frozen": 0, "non_applicable": 0}
    attempts_dir_path = Path(attempts_dir).resolve() if attempts_dir is not None else None
    if allow_exhausted:
        state_counts["exhausted"] = 0
    connection = _connect_read_only(state_db)
    try:
        for entry in entries:
            task_row = _fetch_one(
                connection,
                """SELECT logical_id, task_type, spec_json, state,
                          attempt_count, last_error_class
                   FROM tasks
                   WHERE evaluation_key = ?""",
                (entry.evaluation_key,),
                context=f"任务 {entry.logical_id}.tasks",
            )
            if task_row is None:
                raise FrozenResultIndexError(f"任务 {entry.logical_id} 缺失 tasks 闭环记录")
            if task_row["logical_id"] != entry.logical_id:
                raise FrozenResultIndexError(f"任务 {entry.logical_id} 的 logical_id 发生漂移")
            if task_row["task_type"] != entry.definition.task_spec.task_type:
                raise FrozenResultIndexError(f"任务 {entry.logical_id} 的 task_type 发生漂移")
            state = _validate_task_row_with_exhausted(
                task_row,
                expected_spec_json=entry.definition.task_spec.canonical_json(),
                context=f"任务 {entry.logical_id}",
                allow_exhausted=allow_exhausted,
            )
            if state == "frozen":
                row = _build_frozen_row(
                    connection=connection,
                    entry=entry,
                    plan_sha256=loaded_plan.plan_sha256,
                )
            elif state == "exhausted":
                row = _build_exhausted_row(
                    connection=connection,
                    entry=entry,
                    plan_sha256=loaded_plan.plan_sha256,
                    attempts_dir=attempts_dir_path,
                    task_row=task_row,
                )
            else:
                row = _build_non_applicable_row(
                    connection=connection,
                    entry=entry,
                    plan_sha256=loaded_plan.plan_sha256,
                )
            rows.append(row)
            state_counts[state] += 1
    finally:
        connection.close()

    output_path = Path(output_jsonl)
    summary_path = Path(summary_json)
    output_text = ""
    if rows:
        output_text = "\n".join(canonical_json(row) for row in rows) + "\n"
    _atomic_write_text(output_path, output_text)
    summary = {
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "plan_jsonl": str(loaded_plan.plan_path),
        "plan_sha256": loaded_plan.plan_sha256,
        "row_count": len(rows),
        "state_counts": state_counts,
        "state_db": str(Path(state_db)),
        "status": "ok",
    }
    _atomic_write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验并索引 Stage5 frozen/non_applicable/exhausted 结果")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--allow-exhausted", action="store_true")
    parser.add_argument("--attempts-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_frozen_result_index(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            allow_exhausted=bool(args.allow_exhausted),
            attempts_dir=args.attempts_dir,
        )
    except FrozenResultIndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
