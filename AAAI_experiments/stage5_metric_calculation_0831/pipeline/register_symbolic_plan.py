"""Stage5 符号计划注册执行器。

读取 `symbolic_task_builder.py` 产出的 callable plan JSONL 与
non-applicable index JSONL，先做严格契约校验，再把两类任务一起批量注册到
`TaskStateStore`，最后对 no-call 任务逐条写入 non_applicable 审计闭环。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    PlanContractError,
    PlannedDefinition,
    _load_predecessor_attempt_manifest,
    _row_to_definition,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskStateStore,
)


class RegisterSymbolicPlanError(RuntimeError):
    """符号计划注册前的输入、审计或交叉闭包校验失败。"""

    def __init__(
        self,
        message: str,
        *,
        status: str = "registration_contract_error",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.details = dict(details or {})


@dataclass(frozen=True)
class LoadedCallablePlan:
    path: Path
    sha256: str
    entries: tuple[PlannedDefinition, ...]


@dataclass(frozen=True)
class NonApplicableDefinition:
    planned: PlannedDefinition
    phase: str
    reason: str
    evidence_hash: str
    evidence_path: Path
    evidence_sha256: str
    evidence_payload: dict[str, Any]
    request_context: dict[str, Any]

    @property
    def evaluation_key(self) -> str:
        return self.planned.evaluation_key

    @property
    def logical_id(self) -> str:
        return self.planned.logical_id


@dataclass(frozen=True)
class LoadedNonApplicableIndex:
    path: Path
    sha256: str
    entries: tuple[NonApplicableDefinition, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegisterSymbolicPlanError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegisterSymbolicPlanError(f"{context} 必须是非空字符串")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = _require_string(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RegisterSymbolicPlanError(f"{context} 必须是小写十六进制 SHA256")
    return text


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegisterSymbolicPlanError(f"{context} 不可解析: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegisterSymbolicPlanError(f"{context} 不是 JSON object")
    return payload


def _read_jsonl_rows(path: Path) -> tuple[bytes, list[tuple[int, dict[str, Any]]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RegisterSymbolicPlanError(f"无法读取 JSONL 文件 {path}: {exc}") from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegisterSymbolicPlanError(f"{path}:{line_number} JSONL 解析失败: {exc}") from exc
            rows.append((line_number, dict(_require_mapping(payload, context=f"{path}:{line_number}"))))
    return raw, rows


def _resolve_input_path(path_text: str, *, base_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_path.parent / path).resolve()


def _load_callable_plan(path: str | Path) -> LoadedCallablePlan:
    plan_path = Path(path).resolve()
    raw, rows = _read_jsonl_rows(plan_path)
    entries: list[PlannedDefinition] = []
    seen_keys: dict[str, str] = {}
    seen_logical_ids: dict[str, str] = {}
    for line_number, row in rows:
        try:
            entry = _row_to_definition(row, line_number=line_number)
        except PlanContractError as exc:
            raise RegisterSymbolicPlanError(str(exc)) from exc
        previous_logical_id = seen_keys.get(entry.evaluation_key)
        if previous_logical_id is not None:
            raise RegisterSymbolicPlanError(
                f"callable plan 存在重复 evaluation_key: {entry.evaluation_key} "
                f"(logical_id={previous_logical_id!r} 与 {entry.logical_id!r})"
            )
        previous_evaluation_key = seen_logical_ids.get(entry.logical_id)
        if previous_evaluation_key is not None:
            raise RegisterSymbolicPlanError(
                f"callable plan 存在重复 logical_id: {entry.logical_id} "
                f"(evaluation_key={previous_evaluation_key!r} 与 {entry.evaluation_key!r})"
            )
        seen_keys[entry.evaluation_key] = entry.logical_id
        seen_logical_ids[entry.logical_id] = entry.evaluation_key
        entries.append(entry)
    return LoadedCallablePlan(
        path=plan_path,
        sha256=_sha256_bytes(raw),
        entries=tuple(entries),
    )


def _validate_non_applicable_payload(
    *,
    entry: PlannedDefinition,
    payload: Mapping[str, Any],
    phase: str,
    reason: str,
    request_context: Mapping[str, Any],
    context: str,
) -> None:
    schema_version = _require_string(payload.get("schema_version"), context=f"{context}.schema_version")
    if schema_version != "symbolic_non_applicable.v1":
        raise RegisterSymbolicPlanError(
            f"{context}.schema_version 必须为 symbolic_non_applicable.v1，实际为 {schema_version!r}"
        )
    expected_pairs = {
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "phase": phase,
        "condition": entry.definition.task_spec.condition,
        "reason": reason,
    }
    for field_name, expected_value in expected_pairs.items():
        actual_value = _require_string(payload.get(field_name), context=f"{context}.{field_name}")
        if actual_value != expected_value:
            raise RegisterSymbolicPlanError(
                f"{context}.{field_name} 漂移: {actual_value!r} != {expected_value!r}"
            )
    dependencies = payload.get("dependencies")
    expected_dependencies = list(entry.definition.task_spec.dependencies)
    if dependencies != expected_dependencies:
        raise RegisterSymbolicPlanError(
            f"{context}.dependencies 漂移: {dependencies!r} != {expected_dependencies!r}"
        )
    payload_request_context = dict(
        _require_mapping(payload.get("request_context"), context=f"{context}.request_context")
    )
    if payload_request_context != dict(request_context):
        raise RegisterSymbolicPlanError(f"{context}.request_context 与 request 不一致")


def _load_non_applicable_index(path: str | Path) -> LoadedNonApplicableIndex:
    index_path = Path(path).resolve()
    raw, rows = _read_jsonl_rows(index_path)
    entries: list[NonApplicableDefinition] = []
    seen_keys: dict[str, str] = {}
    seen_logical_ids: dict[str, str] = {}
    for line_number, row in rows:
        context = f"{index_path}:{line_number}"
        try:
            planned = _row_to_definition(row, line_number=line_number)
        except PlanContractError as exc:
            raise RegisterSymbolicPlanError(str(exc)) from exc
        previous_logical_id = seen_keys.get(planned.evaluation_key)
        if previous_logical_id is not None:
            raise RegisterSymbolicPlanError(
                f"non_applicable index 存在重复 evaluation_key: {planned.evaluation_key} "
                f"(logical_id={previous_logical_id!r} 与 {planned.logical_id!r})"
            )
        previous_evaluation_key = seen_logical_ids.get(planned.logical_id)
        if previous_evaluation_key is not None:
            raise RegisterSymbolicPlanError(
                f"non_applicable index 存在重复 logical_id: {planned.logical_id} "
                f"(evaluation_key={previous_evaluation_key!r} 与 {planned.evaluation_key!r})"
            )
        phase = _require_string(row.get("phase"), context=f"{context}.phase")
        reason = _require_string(row.get("reason"), context=f"{context}.reason")
        evidence_hash = _require_sha256(row.get("evidence_hash"), context=f"{context}.evidence_hash")
        evidence_sha256 = _require_sha256(
            row.get("evidence_sha256"),
            context=f"{context}.evidence_sha256",
        )
        if evidence_hash != evidence_sha256:
            raise RegisterSymbolicPlanError(f"{context} evidence_hash 与 evidence_sha256 不一致")
        request = dict(planned.definition.request)
        request_evidence_hash = _require_sha256(
            request.get("evidence_hash"),
            context=f"{context}.request.evidence_hash",
        )
        if request_evidence_hash != evidence_hash:
            raise RegisterSymbolicPlanError(f"{context}.request.evidence_hash 与证据 SHA 不一致")
        request_context = dict(
            _require_mapping(row.get("request_context"), context=f"{context}.request_context")
        )
        expected_request_context = {key: value for key, value in request.items() if key != "evidence_hash"}
        if request_context != expected_request_context:
            raise RegisterSymbolicPlanError(f"{context}.request_context 与 request 投影不一致")
        status = _require_string(row.get("status"), context=f"{context}.status")
        if status != "planned_non_applicable":
            raise RegisterSymbolicPlanError(
                f"{context}.status 必须为 planned_non_applicable，实际为 {status!r}"
            )
        evidence_payload = dict(
            _require_mapping(row.get("evidence_payload"), context=f"{context}.evidence_payload")
        )
        evidence_path = _resolve_input_path(
            _require_string(row.get("evidence_path"), context=f"{context}.evidence_path"),
            base_path=index_path,
        )
        if not evidence_path.is_file():
            raise RegisterSymbolicPlanError(f"{context}.evidence_path 对应证据文件不存在: {evidence_path}")
        file_sha256 = _sha256_file(evidence_path)
        if file_sha256 != evidence_sha256:
            raise RegisterSymbolicPlanError(f"{context}.evidence_sha256 与证据文件不一致")
        file_payload = _read_json_object(evidence_path, context=f"{context}.evidence_file")
        if file_payload != evidence_payload:
            raise RegisterSymbolicPlanError(f"{context}.evidence_payload 与证据文件内容不一致")
        _validate_non_applicable_payload(
            entry=planned,
            payload=file_payload,
            phase=phase,
            reason=reason,
            request_context=request_context,
            context=f"{context}.evidence_file",
        )
        seen_keys[planned.evaluation_key] = planned.logical_id
        seen_logical_ids[planned.logical_id] = planned.evaluation_key
        entries.append(
            NonApplicableDefinition(
                planned=planned,
                phase=phase,
                reason=reason,
                evidence_hash=evidence_hash,
                evidence_path=evidence_path,
                evidence_sha256=evidence_sha256,
                evidence_payload=file_payload,
                request_context=request_context,
            )
        )
    return LoadedNonApplicableIndex(
        path=index_path,
        sha256=_sha256_bytes(raw),
        entries=tuple(entries),
    )


def _validate_cross_plan_overlap(
    *,
    callable_plan: LoadedCallablePlan,
    non_applicable_index: LoadedNonApplicableIndex,
) -> None:
    call_by_key = {entry.evaluation_key: entry.logical_id for entry in callable_plan.entries}
    call_by_logical = {entry.logical_id: entry.evaluation_key for entry in callable_plan.entries}
    for entry in non_applicable_index.entries:
        previous_logical_id = call_by_key.get(entry.evaluation_key)
        if previous_logical_id is not None:
            raise RegisterSymbolicPlanError(
                "callable plan 与 non_applicable index 存在重叠 evaluation_key: "
                f"{entry.evaluation_key} (logical_id={previous_logical_id!r} 与 {entry.logical_id!r})"
            )
        previous_evaluation_key = call_by_logical.get(entry.logical_id)
        if previous_evaluation_key is not None:
            raise RegisterSymbolicPlanError(
                "callable plan 与 non_applicable index 存在重叠 logical_id: "
                f"{entry.logical_id} (evaluation_key={previous_evaluation_key!r} 与 {entry.evaluation_key!r})"
            )


def _load_existing_state_map(store: TaskStateStore, evaluation_keys: Sequence[str]) -> dict[str, str]:
    if not evaluation_keys:
        return {}
    state_map: dict[str, str] = {}
    with store._connect() as connection:  # type: ignore[attr-defined]
        deduplicated = tuple(dict.fromkeys(evaluation_keys))
        for start in range(0, len(deduplicated), 500):
            chunk = deduplicated[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT evaluation_key, state FROM tasks WHERE evaluation_key IN ({placeholders})"
            for row in connection.execute(query, chunk).fetchall():
                state_map[str(row["evaluation_key"])] = str(row["state"])
    return state_map


def _load_existing_non_applicable_map(
    store: TaskStateStore,
    evaluation_keys: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    if not evaluation_keys:
        return {}
    result: dict[str, tuple[str, str, str]] = {}
    with store._connect() as connection:  # type: ignore[attr-defined]
        deduplicated = tuple(dict.fromkeys(evaluation_keys))
        for start in range(0, len(deduplicated), 500):
            chunk = deduplicated[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT evaluation_key, reason, evidence_path, evidence_sha256 "
                f"FROM non_applicable_results WHERE evaluation_key IN ({placeholders})"
            )
            for row in connection.execute(query, chunk).fetchall():
                result[str(row["evaluation_key"])] = (
                    str(row["reason"]),
                    str(row["evidence_path"]),
                    str(row["evidence_sha256"]),
                )
    return result


def _preflight_existing_state(
    *,
    store: TaskStateStore,
    callable_plan: LoadedCallablePlan,
    non_applicable_index: LoadedNonApplicableIndex,
) -> tuple[int, set[str]]:
    all_keys = [entry.evaluation_key for entry in callable_plan.entries] + [
        entry.evaluation_key for entry in non_applicable_index.entries
    ]
    state_map = _load_existing_state_map(store, all_keys)
    non_applicable_map = _load_existing_non_applicable_map(
        store,
        [entry.evaluation_key for entry in non_applicable_index.entries],
    )
    for entry in callable_plan.entries:
        if state_map.get(entry.evaluation_key) == "non_applicable":
            raise StateContractError(
                f"callable 任务 {entry.logical_id!r} 已处于 non_applicable，不能与 no-call 计划分离"
            )
    already_marked: set[str] = set()
    for entry in non_applicable_index.entries:
        state = state_map.get(entry.evaluation_key)
        if state is None:
            continue
        if state == "non_applicable":
            current = non_applicable_map.get(entry.evaluation_key)
            requested = (entry.reason, str(entry.evidence_path), entry.evidence_sha256)
            if current != requested:
                raise StateContractError(f"任务 {entry.logical_id!r} 的 non_applicable 审计记录发生漂移")
            already_marked.add(entry.evaluation_key)
            continue
        if state not in {"pending", "retry_wait"}:
            raise StateContractError(
                f"任务 {entry.logical_id!r} 当前状态 {state!r}，不可标记 non_applicable"
            )
    return len(state_map), already_marked


def _count_task_types(
    callable_plan: LoadedCallablePlan,
    non_applicable_index: LoadedNonApplicableIndex,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in callable_plan.entries:
        counter[entry.definition.task_spec.task_type] += 1
    for entry in non_applicable_index.entries:
        counter[entry.planned.definition.task_spec.task_type] += 1
    return dict(sorted(counter.items()))


def _count_phases(non_applicable_index: LoadedNonApplicableIndex) -> dict[str, int]:
    counter: Counter[str] = Counter(entry.phase for entry in non_applicable_index.entries)
    return dict(sorted(counter.items()))


def _final_state_distribution(store: TaskStateStore, evaluation_keys: Sequence[str]) -> dict[str, int]:
    counter: Counter[str] = Counter(_load_existing_state_map(store, evaluation_keys).values())
    return dict(sorted(counter.items()))


def register_symbolic_plan(
    *,
    plan_jsonl: str | Path,
    non_applicable_index_jsonl: str | Path,
    state_db: str | Path,
    predecessor_attempt_manifest: str | Path | None = None,
    report_json: str | Path | None = None,
    now: float | None = None,
) -> dict[str, object]:
    started_at = time.time() if now is None else float(now)
    callable_plan = _load_callable_plan(plan_jsonl)
    non_applicable_index = _load_non_applicable_index(non_applicable_index_jsonl)
    _validate_cross_plan_overlap(
        callable_plan=callable_plan,
        non_applicable_index=non_applicable_index,
    )
    total_unique_task_count = len(callable_plan.entries) + len(non_applicable_index.entries)
    if total_unique_task_count == 0:
        raise RegisterSymbolicPlanError("callable plan 与 non_applicable index 不能同时为空")

    loaded_predecessor_manifest = None
    if predecessor_attempt_manifest is not None:
        try:
            loaded_predecessor_manifest = _load_predecessor_attempt_manifest(
                predecessor_attempt_manifest
            )
        except PlanContractError as exc:
            raise RegisterSymbolicPlanError(str(exc)) from exc
    store = TaskStateStore(
        Path(state_db).resolve(),
        predecessor_attempt_manifest=(
            PredecessorAttemptManifest(
                path=str(loaded_predecessor_manifest.path),
                sha256=loaded_predecessor_manifest.sha256,
                attempt_count=loaded_predecessor_manifest.attempt_count,
            )
            if loaded_predecessor_manifest is not None
            else None
        ),
    )
    existing_task_count, already_marked = _preflight_existing_state(
        store=store,
        callable_plan=callable_plan,
        non_applicable_index=non_applicable_index,
    )

    all_specs = [entry.definition.task_spec for entry in callable_plan.entries] + [
        entry.planned.definition.task_spec for entry in non_applicable_index.entries
    ]
    store.register_tasks(all_specs, now=started_at)

    newly_marked_count = 0
    for entry in non_applicable_index.entries:
        if entry.evaluation_key in already_marked:
            continue
        store.mark_non_applicable(
            entry.evaluation_key,
            reason=entry.reason,
            evidence_path=str(entry.evidence_path),
            evidence_sha256=entry.evidence_sha256,
            now=started_at,
        )
        newly_marked_count += 1

    all_evaluation_keys = [entry.evaluation_key for entry in callable_plan.entries] + [
        entry.evaluation_key for entry in non_applicable_index.entries
    ]
    finished_at = time.time() if now is None else float(now)
    report: dict[str, object] = {
        "status": "ok",
        "model_invoked": False,
        "state_db": str(Path(state_db).resolve()),
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "callable_plan_jsonl": {
                "path": str(callable_plan.path),
                "sha256": callable_plan.sha256,
                "row_count": len(callable_plan.entries),
            },
            "non_applicable_index_jsonl": {
                "path": str(non_applicable_index.path),
                "sha256": non_applicable_index.sha256,
                "row_count": len(non_applicable_index.entries),
            },
            "predecessor_attempt_manifest": (
                {
                    "path": str(loaded_predecessor_manifest.path),
                    "sha256": loaded_predecessor_manifest.sha256,
                    "attempt_count": loaded_predecessor_manifest.attempt_count,
                }
                if loaded_predecessor_manifest is not None
                else None
            ),
        },
        "counts": {
            "callable_task_count": len(callable_plan.entries),
            "non_applicable_task_count": len(non_applicable_index.entries),
            "total_unique_task_count": total_unique_task_count,
            "existing_task_count_before": existing_task_count,
            "newly_registered_task_count": total_unique_task_count - existing_task_count,
            "already_non_applicable_count": len(already_marked),
            "newly_marked_non_applicable_count": newly_marked_count,
            "evidence_verified_count": len(non_applicable_index.entries),
        },
        "distributions": {
            "task_type": _count_task_types(callable_plan, non_applicable_index),
            "phase": _count_phases(non_applicable_index),
            "final_state": _final_state_distribution(store, all_evaluation_keys),
        },
        "actions": {
            "used_bulk_register_tasks": True,
            "mark_non_applicable_mode": "per_row",
        },
    }
    if report_json is not None:
        _atomic_write_json(Path(report_json), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="注册 Stage5 符号 callable/no-call 计划到状态库")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--non-applicable-index-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--predecessor-attempt-manifest", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_report: dict[str, object] = {
        "model_invoked": False,
        "state_db": str(args.state_db.resolve()),
        "inputs": {
            "callable_plan_jsonl": str(args.plan_jsonl.resolve()),
            "non_applicable_index_jsonl": str(args.non_applicable_index_jsonl.resolve()),
            "predecessor_attempt_manifest": (
                str(args.predecessor_attempt_manifest.resolve())
                if args.predecessor_attempt_manifest is not None
                else None
            ),
        },
    }
    try:
        register_symbolic_plan(
            plan_jsonl=args.plan_jsonl,
            non_applicable_index_jsonl=args.non_applicable_index_jsonl,
            state_db=args.state_db,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            report_json=args.report_json,
        )
    except RegisterSymbolicPlanError as exc:
        failure_report = dict(base_report)
        failure_report.update({"status": exc.status, "error": str(exc)})
        failure_report.update(exc.details)
        _atomic_write_json(args.report_json, failure_report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except StateContractError as exc:
        failure_report = dict(base_report)
        failure_report.update({"status": "state_contract_error", "error": str(exc)})
        _atomic_write_json(args.report_json, failure_report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        failure_report = dict(base_report)
        failure_report.update({"status": "unexpected_error", "error": str(exc)})
        _atomic_write_json(args.report_json, failure_report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "ok",
        "callable_plan_jsonl": str(args.plan_jsonl.resolve()),
        "non_applicable_index_jsonl": str(args.non_applicable_index_jsonl.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
