"""冻结结果索引构建器。

读取 canonical plan JSONL 与只读状态库，校验每个计划任务都已形成
`frozen` 或 `non_applicable` 的闭环，并输出可按
`logical_id` / `evaluation_key` 检索的规范 JSONL。
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
    }


def build_frozen_result_index(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
) -> JsonDict:
    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        raise FrozenResultIndexError(f"plan 契约错误: {exc}") from exc

    entries = sorted(loaded_plan.entries, key=lambda entry: (entry.logical_id, entry.evaluation_key))
    _validate_unique_entries(entries)

    rows: list[JsonDict] = []
    state_counts = {"frozen": 0, "non_applicable": 0}
    connection = _connect_read_only(state_db)
    try:
        for entry in entries:
            task_row = _fetch_one(
                connection,
                """SELECT logical_id, task_type, spec_json, state
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
            state = _validate_task_row(
                task_row,
                expected_spec_json=entry.definition.task_spec.canonical_json(),
                context=f"任务 {entry.logical_id}",
            )
            if state == "frozen":
                row = _build_frozen_row(
                    connection=connection,
                    entry=entry,
                    plan_sha256=loaded_plan.plan_sha256,
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
    parser = argparse.ArgumentParser(description="校验并索引 Stage5 frozen/non_applicable 结果")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_frozen_result_index(
            plan_jsonl=args.plan_jsonl,
            state_db=args.state_db,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
        )
    except FrozenResultIndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
