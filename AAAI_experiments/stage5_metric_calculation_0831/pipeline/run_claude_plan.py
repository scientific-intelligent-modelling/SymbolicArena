"""Claude plan JSONL 批处理器。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    ClaudeRunResult,
    ClaudeRunner,
    ClaudeRunnerCircuitBreaker,
    TaskDefinition,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskSpec,
    TaskStateStore,
)


JsonDict = dict[str, object]


class PlanContractError(ValueError):
    """计划 JSONL、哈希或定义不满足冻结契约。"""


@dataclass(frozen=True)
class PlannedDefinition:
    logical_id: str
    evaluation_key: str
    definition: TaskDefinition


@dataclass(frozen=True)
class LoadedPlan:
    plan_path: Path
    plan_sha256: str
    entries: tuple[PlannedDefinition, ...]


@dataclass(frozen=True)
class LoadedPredecessorAttemptManifest:
    path: Path
    sha256: str
    attempt_count: int


RunnerFactory = Callable[[TaskStateStore], Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanContractError(f"{context} 不是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanContractError(f"{context} 必须是非空字符串")
    return value


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanContractError(f"{context} 必须是整数")
    return value


def _require_string_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PlanContractError(f"{context} 必须是字符串数组")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise PlanContractError(f"{context}[{index}] 必须是字符串")
        items.append(item)
    return tuple(items)


def _canonical_schema(value: Mapping[str, object]) -> JsonDict:
    payload = json.loads(canonical_json(value))
    if not isinstance(payload, dict):
        raise PlanContractError("schema_content 必须是 JSON object")
    return payload


def _load_predecessor_attempt_manifest(
    path: str | Path,
) -> LoadedPredecessorAttemptManifest:
    manifest_path = Path(path).resolve()
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise PlanContractError(
            f"无法读取 predecessor attempt manifest {manifest_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PlanContractError(
            f"predecessor attempt manifest 解析失败: {manifest_path}: {exc}"
        ) from exc
    row = _require_mapping(payload, context="predecessor_attempt_manifest")
    if row.get("schema_version") != "predecessor_attempts.v1":
        raise PlanContractError(
            "predecessor_attempt_manifest.schema_version 必须为 predecessor_attempts.v1"
        )
    attempt_ids = _require_string_list(
        row.get("attempt_ids"),
        context="predecessor_attempt_manifest.attempt_ids",
    )
    if len(set(attempt_ids)) != len(attempt_ids):
        raise PlanContractError("predecessor_attempt_manifest.attempt_ids 存在重复项")
    declared_count = row.get("attempt_count")
    if declared_count is not None:
        declared_count = _require_int(
            declared_count,
            context="predecessor_attempt_manifest.attempt_count",
        )
        if declared_count != len(attempt_ids):
            raise PlanContractError(
                "predecessor_attempt_manifest.attempt_count 与 attempt_ids 数量不一致"
            )
    source_state_db_raw = _require_string(
        row.get("source_state_db"),
        context="predecessor_attempt_manifest.source_state_db",
    )
    source_attempts_dir_raw = _require_string(
        row.get("source_attempts_dir"),
        context="predecessor_attempt_manifest.source_attempts_dir",
    )

    def resolve_source(raw_path: str) -> Path:
        source = Path(raw_path)
        if source.is_absolute():
            return source.resolve()
        cwd_candidate = (Path.cwd() / source).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (manifest_path.parent / source).resolve()

    source_state_db = resolve_source(source_state_db_raw)
    source_attempts_dir = resolve_source(source_attempts_dir_raw)
    if not source_state_db.is_file():
        raise PlanContractError(f"predecessor source_state_db 不存在: {source_state_db}")
    if not source_attempts_dir.is_dir():
        raise PlanContractError(
            f"predecessor source_attempts_dir 不存在: {source_attempts_dir}"
        )
    try:
        connection = sqlite3.connect(
            f"{source_state_db.as_uri()}?mode=ro",
            uri=True,
        )
        try:
            db_attempt_ids = tuple(
                str(record[0])
                for record in connection.execute(
                    "SELECT attempt_id FROM attempts ORDER BY attempt_id"
                ).fetchall()
            )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise PlanContractError(f"predecessor source_state_db 无法核验: {exc}") from exc
    if tuple(sorted(attempt_ids)) != db_attempt_ids:
        raise PlanContractError(
            "predecessor attempt_ids 与 source_state_db.attempts 不完全一致"
        )

    declared_hashes = _require_mapping(
        row.get("attempt_file_sha256"),
        context="predecessor_attempt_manifest.attempt_file_sha256",
    )
    if set(declared_hashes) != set(attempt_ids):
        raise PlanContractError(
            "predecessor attempt_file_sha256 键必须与 attempt_ids 完全一致"
        )
    for attempt_id in attempt_ids:
        expected_sha256 = _require_string(
            declared_hashes.get(attempt_id),
            context=f"predecessor attempt_file_sha256[{attempt_id!r}]",
        )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise PlanContractError(f"predecessor attempt SHA-256 非法: {attempt_id}")
        audit_path = source_attempts_dir / f"{attempt_id}.json"
        try:
            audit_raw = audit_path.read_bytes()
            audit_payload = json.loads(audit_raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanContractError(
                f"predecessor attempt 审计文件不可读: {audit_path}: {exc}"
            ) from exc
        if _sha256_bytes(audit_raw) != expected_sha256:
            raise PlanContractError(f"predecessor attempt 审计 SHA 漂移: {attempt_id}")
        if not isinstance(audit_payload, Mapping) or audit_payload.get("attempt_id") != attempt_id:
            raise PlanContractError(f"predecessor attempt 审计 ID 不一致: {attempt_id}")
    return LoadedPredecessorAttemptManifest(
        path=manifest_path,
        sha256=_sha256_bytes(raw),
        attempt_count=len(attempt_ids),
    )


def _default_runner_factory(
    store: TaskStateStore,
    *,
    attempts_dir: Path,
    frozen_dir: Path,
) -> ClaudeRunner:
    return ClaudeRunner(
        store,
        attempts_dir=attempts_dir,
        frozen_dir=frozen_dir,
    )


def _row_to_definition(row: Mapping[str, Any], *, line_number: int) -> PlannedDefinition:
    context = f"plan line {line_number}"
    top_key = _require_string(row.get("evaluation_key"), context=f"{context}.evaluation_key")
    logical_id = _require_string(row.get("logical_id"), context=f"{context}.logical_id")
    task_type = _require_string(row.get("task_type"), context=f"{context}.task_type")
    condition = _require_string(row.get("condition"), context=f"{context}.condition")
    priority = _require_int(row.get("priority"), context=f"{context}.priority")
    input_hash = _require_string(row.get("input_hash"), context=f"{context}.input_hash")
    prompt_version = _require_string(row.get("prompt_version"), context=f"{context}.prompt_version")
    prompt_sha256 = _require_string(row.get("prompt_sha256"), context=f"{context}.prompt_sha256")
    schema_version = _require_string(row.get("schema_version"), context=f"{context}.schema_version")
    schema_sha256 = _require_string(row.get("schema_sha256"), context=f"{context}.schema_sha256")
    dependencies = _require_string_list(row.get("dependencies"), context=f"{context}.dependencies")
    prompt_path = Path(_require_string(row.get("prompt_path"), context=f"{context}.prompt_path"))
    schema_path = Path(_require_string(row.get("schema_path"), context=f"{context}.schema_path"))
    prompt_template = _require_string(row.get("prompt_template"), context=f"{context}.prompt_template")
    request = dict(_require_mapping(row.get("request"), context=f"{context}.request"))
    normalized_input = dict(
        _require_mapping(row.get("normalized_input"), context=f"{context}.normalized_input")
    )
    schema_content = _canonical_schema(
        _require_mapping(row.get("schema_content"), context=f"{context}.schema_content")
    )
    task_spec_payload = dict(_require_mapping(row.get("task_spec"), context=f"{context}.task_spec"))

    expected_spec = TaskSpec(
        evaluation_key=top_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=condition,
        priority=priority,
        input_hash=input_hash,
        prompt_version=prompt_version,
        schema_version=schema_version,
        dependencies=dependencies,
    )
    if task_spec_payload != json.loads(expected_spec.canonical_json()):
        raise PlanContractError(f"{context}.task_spec 与顶层字段不一致")

    if not prompt_path.is_file():
        raise PlanContractError(f"{context}.prompt_path 不存在: {prompt_path}")
    if not schema_path.is_file():
        raise PlanContractError(f"{context}.schema_path 不存在: {schema_path}")

    prompt_bytes = prompt_path.read_bytes()
    schema_bytes = schema_path.read_bytes()
    if _sha256_bytes(prompt_bytes) != prompt_sha256:
        raise PlanContractError(f"{context}.prompt_sha256 与文件不一致")
    if _sha256_bytes(schema_bytes) != schema_sha256:
        raise PlanContractError(f"{context}.schema_sha256 与文件不一致")
    if prompt_bytes.decode("utf-8") != prompt_template:
        raise PlanContractError(f"{context}.prompt_template 与文件不一致")
    loaded_schema = json.loads(schema_bytes.decode("utf-8"))
    if not isinstance(loaded_schema, dict):
        raise PlanContractError(f"{context}.schema_path 不是 JSON object")
    if _canonical_schema(loaded_schema) != schema_content:
        raise PlanContractError(f"{context}.schema_content 与文件不一致")

    recomputed_rendered_prompt = render_prompt(
        prompt_template,
        request,
        schema_content,
    )
    rendered_prompt = row.get("rendered_prompt")
    if rendered_prompt is not None and rendered_prompt != recomputed_rendered_prompt:
        raise PlanContractError(f"{context}.rendered_prompt 与 request/prompt_template 不一致")

    expected_normalized_input = {
        "request": dict(request),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    if normalized_input != expected_normalized_input:
        raise PlanContractError(f"{context}.normalized_input 与冻结契约不一致")
    if _sha256_text(canonical_json(expected_normalized_input)) != input_hash:
        raise PlanContractError(f"{context}.input_hash 与 normalized_input 不一致")

    evidence_hash = _require_string(request.get("evidence_hash"), context=f"{context}.request.evidence_hash")
    recomputed_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=expected_normalized_input,
        evidence_hash=evidence_hash,
    )
    if recomputed_key != top_key:
        raise PlanContractError(f"{context}.evaluation_key 与冻结契约不一致")

    task_kind = row.get("task_kind")
    if task_kind is not None and not isinstance(task_kind, str):
        raise PlanContractError(f"{context}.task_kind 必须是字符串")

    return PlannedDefinition(
        logical_id=logical_id,
        evaluation_key=top_key,
        definition=TaskDefinition(
            task_spec=expected_spec,
            request=request,
            prompt_path=prompt_path,
            prompt_sha256=prompt_sha256,
            schema_path=schema_path,
            schema_sha256=schema_sha256,
            prompt_template=prompt_template,
            schema=schema_content,
            task_kind=task_kind,
        ),
    )


def load_plan_jsonl(path: str | Path) -> LoadedPlan:
    plan_path = Path(path)
    try:
        raw = plan_path.read_bytes()
    except OSError as exc:
        raise PlanContractError(f"无法读取 plan JSONL {plan_path}: {exc}") from exc
    entries: list[PlannedDefinition] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanContractError(f"{plan_path}:{line_number} JSONL 解析失败: {exc}") from exc
            row = _require_mapping(parsed, context=f"{plan_path}:{line_number}")
            entries.append(_row_to_definition(row, line_number=line_number))
    if not entries:
        raise PlanContractError("plan JSONL 不能为空")
    seen_keys: set[str] = set()
    seen_logical_ids: set[str] = set()
    for entry in entries:
        if entry.evaluation_key in seen_keys:
            raise PlanContractError(f"plan 存在重复 evaluation_key: {entry.evaluation_key}")
        seen_keys.add(entry.evaluation_key)
        if entry.logical_id in seen_logical_ids:
            raise PlanContractError(f"plan 存在重复 logical_id: {entry.logical_id}")
        seen_logical_ids.add(entry.logical_id)
    return LoadedPlan(plan_path=plan_path, plan_sha256=_sha256_bytes(raw), entries=tuple(entries))


def _verify_cached_frozen_entries(
    store: TaskStateStore,
    entries: Sequence[PlannedDefinition],
) -> None:
    """在跳过缓存任务前复核状态库、文件哈希和冻结身份。"""

    for entry in entries:
        if store.task_state(entry.evaluation_key) != "frozen":
            continue
        frozen = store.frozen_result(entry.evaluation_key)
        if frozen is None:
            raise PlanContractError(
                f"frozen task 缺少 frozen_results 记录: {entry.evaluation_key}"
            )
        result_path = Path(frozen["result_path"])
        if not result_path.is_file():
            raise PlanContractError(f"frozen result 文件不存在: {result_path}")
        actual_sha256 = _sha256_bytes(result_path.read_bytes())
        if actual_sha256 != frozen["result_sha256"]:
            raise PlanContractError(
                f"frozen result SHA256 漂移: {entry.evaluation_key}"
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PlanContractError(
                f"frozen result 无法解析: {entry.evaluation_key}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PlanContractError(f"frozen result 不是 JSON object: {entry.evaluation_key}")
        expected_identity = {
            "evaluation_key": entry.evaluation_key,
            "logical_id": entry.logical_id,
            "task_type": entry.definition.task_spec.task_type,
        }
        for field, expected in expected_identity.items():
            if payload.get(field) != expected:
                raise PlanContractError(
                    f"frozen result {field} 漂移: {entry.evaluation_key}"
                )
        if not isinstance(payload.get("structured_output"), Mapping):
            raise PlanContractError(
                f"frozen result structured_output 缺失或无效: {entry.evaluation_key}"
            )


def _task_state_map(store: TaskStateStore, entries: Sequence[PlannedDefinition]) -> dict[str, str]:
    return {entry.evaluation_key: store.task_state(entry.evaluation_key) for entry in entries}


def _distribution_from_state_map(state_map: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in state_map.values():
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _attempt_count_for_plan(store: TaskStateStore, entries: Sequence[PlannedDefinition]) -> int:
    keys = [entry.evaluation_key for entry in entries]
    if not keys:
        return 0
    total = 0
    connection = sqlite3.connect(store.path)
    try:
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                f"SELECT COUNT(*) FROM attempts WHERE evaluation_key IN ({placeholders})"
            )
            total += int(connection.execute(query, chunk).fetchone()[0])
    finally:
        connection.close()
    return total


def _build_report(
    *,
    loaded_plan: LoadedPlan,
    store: TaskStateStore,
    entries: Sequence[PlannedDefinition],
    selected_entries: Sequence[PlannedDefinition],
    submitted_task_count: int,
    completed_task_count: int,
    result_counts: Mapping[str, int],
    started_at_monotonic: float,
    requested_logical_ids: Sequence[str],
    limit: int | None,
    workers: int,
    stopped_by_circuit_breaker: bool,
    stopped_by_fatal_error: bool,
    recovered_expired_attempt_ids: Sequence[str],
    predecessor_attempt_manifest: LoadedPredecessorAttemptManifest | None,
    attempts_reserved_at_start: int,
    status: str,
    error: str | None = None,
) -> JsonDict:
    state_map = _task_state_map(store, entries)
    attempts_reserved_total = store.attempts_reserved()
    attempts_reserved_this_run = attempts_reserved_total - attempts_reserved_at_start
    if attempts_reserved_this_run < 0:
        raise StateContractError("运行期 attempt 计数出现倒退")
    payload: JsonDict = {
        "status": status,
        "error": error,
        "plan_jsonl": str(loaded_plan.plan_path),
        "plan_sha256": loaded_plan.plan_sha256,
        "workers": workers,
        "limit": limit,
        "requested_logical_ids": list(requested_logical_ids),
        "registered_task_count": len(entries),
        "selected_task_count": len(selected_entries),
        "submitted_task_count": submitted_task_count,
        "completed_task_count": completed_task_count,
        "result_counts": dict(result_counts),
        "state_distribution": _distribution_from_state_map(state_map),
        "attempts_reserved_total": attempts_reserved_total,
        "attempts_reserved_before_run": attempts_reserved_at_start,
        "attempts_reserved_this_run": attempts_reserved_this_run,
        "model_invoked": attempts_reserved_this_run > 0,
        "attempts_reserved_for_plan": _attempt_count_for_plan(store, entries),
        "physical_attempt_offset": store.attempt_offset,
        "predecessor_attempt_manifest_path": (
            str(predecessor_attempt_manifest.path)
            if predecessor_attempt_manifest is not None
            else None
        ),
        "predecessor_attempt_manifest_sha256": (
            predecessor_attempt_manifest.sha256
            if predecessor_attempt_manifest is not None
            else None
        ),
        "predecessor_attempt_count": (
            predecessor_attempt_manifest.attempt_count
            if predecessor_attempt_manifest is not None
            else 0
        ),
        "elapsed_seconds": max(0.0, time.monotonic() - started_at_monotonic),
        "total_cost_usd": 0.0,
        "stopped_by_circuit_breaker": stopped_by_circuit_breaker,
        "stopped_by_fatal_error": stopped_by_fatal_error,
        "recovered_expired_attempt_count": len(recovered_expired_attempt_ids),
        "recovered_expired_attempt_ids": list(recovered_expired_attempt_ids),
    }
    return payload


def _select_scope(
    entries: Sequence[PlannedDefinition],
    *,
    logical_ids: Sequence[str],
) -> list[PlannedDefinition]:
    if not logical_ids:
        return list(entries)
    wanted = list(dict.fromkeys(logical_ids))
    by_logical_id = {entry.logical_id: entry for entry in entries}
    missing = [logical_id for logical_id in wanted if logical_id not in by_logical_id]
    if missing:
        raise PlanContractError(f"计划中不存在 logical_id: {missing}")
    return [by_logical_id[logical_id] for logical_id in wanted]


def execute_plan(
    *,
    plan_jsonl: str | Path,
    state_db: str | Path,
    attempts_dir: str | Path,
    frozen_dir: str | Path,
    report_json: str | Path,
    limit: int | None = None,
    logical_ids: Sequence[str] = (),
    workers: int = 1,
    physical_attempt_offset: int = 0,
    predecessor_attempt_manifest: str | Path | None = None,
    runner_factory: Callable[[TaskStateStore, Path, Path], Any] | None = None,
) -> int:
    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须为正整数")
    if physical_attempt_offset < 0:
        raise ValueError("physical_attempt_offset 不能为负数")

    started_at_monotonic = time.monotonic()
    report_path = Path(report_json)
    result_counts = {"success": 0, "cache": 0, "failure": 0, "circuit_breaker": 0}
    loaded_plan: LoadedPlan | None = None
    loaded_predecessor_manifest: LoadedPredecessorAttemptManifest | None = None

    try:
        loaded_plan = load_plan_jsonl(plan_jsonl)
    except PlanContractError as exc:
        _atomic_write_json(
            report_path,
            {
                "status": "plan_contract_error",
                "error": str(exc),
                "plan_jsonl": str(plan_jsonl),
                "plan_sha256": None,
                "model_invoked": False,
            },
        )
        return 2
    if predecessor_attempt_manifest is not None:
        try:
            loaded_predecessor_manifest = _load_predecessor_attempt_manifest(
                predecessor_attempt_manifest
            )
        except PlanContractError as exc:
            _atomic_write_json(
                report_path,
                {
                    "status": "plan_contract_error",
                    "error": str(exc),
                    "plan_jsonl": str(plan_jsonl),
                    "plan_sha256": loaded_plan.plan_sha256,
                    "model_invoked": False,
                },
            )
            return 2

    try:
        store = TaskStateStore(
            state_db,
            attempt_offset=physical_attempt_offset,
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
        store.register_tasks(
            [entry.definition.task_spec for entry in loaded_plan.entries]
        )
        recovered_expired_attempt_ids = store.recover_expired_leases()
        attempts_reserved_at_start = store.attempts_reserved()
        _verify_cached_frozen_entries(store, loaded_plan.entries)
        selected_entries = _select_scope(loaded_plan.entries, logical_ids=logical_ids)
        selected_state_map = _task_state_map(store, selected_entries)
        runnable_entries = [
            entry
            for entry in selected_entries
            if selected_state_map[entry.evaluation_key] in {"pending", "retry_wait"}
        ]
        if limit is not None:
            runnable_entries = runnable_entries[:limit]
        result_counts["cache"] = sum(
            1
            for entry in selected_entries
            if selected_state_map[entry.evaluation_key] == "frozen"
        )
        _atomic_write_json(
            report_path,
            _build_report(
                loaded_plan=loaded_plan,
                store=store,
                entries=loaded_plan.entries,
                selected_entries=selected_entries,
                submitted_task_count=0,
                completed_task_count=0,
                result_counts=result_counts,
                started_at_monotonic=started_at_monotonic,
                requested_logical_ids=logical_ids,
                limit=limit,
                workers=workers,
                stopped_by_circuit_breaker=False,
                stopped_by_fatal_error=False,
                recovered_expired_attempt_ids=recovered_expired_attempt_ids,
                predecessor_attempt_manifest=loaded_predecessor_manifest,
                attempts_reserved_at_start=attempts_reserved_at_start,
                status="running",
            ),
        )
    except PlanContractError as exc:
        _atomic_write_json(
            report_path,
            {
                "status": "plan_contract_error",
                "error": str(exc),
                "plan_jsonl": str(plan_jsonl),
                "plan_sha256": loaded_plan.plan_sha256,
                "model_invoked": False,
            },
        )
        return 2
    except StateContractError as exc:
        _atomic_write_json(
            report_path,
            {
                "status": "state_contract_error",
                "error": str(exc),
                "plan_jsonl": str(plan_jsonl),
                "plan_sha256": loaded_plan.plan_sha256,
                "state_db": str(state_db),
                "model_invoked": False,
                "physical_attempt_offset": physical_attempt_offset,
                "predecessor_attempt_manifest_path": (
                    str(loaded_predecessor_manifest.path)
                    if loaded_predecessor_manifest is not None
                    else None
                ),
                "predecessor_attempt_manifest_sha256": (
                    loaded_predecessor_manifest.sha256
                    if loaded_predecessor_manifest is not None
                    else None
                ),
                "predecessor_attempt_count": (
                    loaded_predecessor_manifest.attempt_count
                    if loaded_predecessor_manifest is not None
                    else 0
                ),
            },
        )
        return 2

    runner = (runner_factory or _default_runner_factory)(
        store,
        attempts_dir=Path(attempts_dir),
        frozen_dir=Path(frozen_dir),
    )
    submitted_task_count = 0
    completed_task_count = 0
    total_cost_usd = 0.0
    stopped_by_circuit_breaker = False
    stopped_by_fatal_error = False
    fatal_error: str | None = None
    next_index = 0
    in_flight: dict[concurrent.futures.Future[ClaudeRunResult], PlannedDefinition] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while next_index < len(runnable_entries) and len(in_flight) < workers:
            entry = runnable_entries[next_index]
            next_index += 1
            future = executor.submit(runner.execute, entry.definition)
            in_flight[future] = entry
            submitted_task_count += 1

        while in_flight:
            done, _ = concurrent.futures.wait(
                tuple(in_flight),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                in_flight.pop(future)
                try:
                    result = future.result()
                except ClaudeRunnerCircuitBreaker as exc:
                    result_counts["circuit_breaker"] += 1
                    stopped_by_circuit_breaker = True
                    fatal_error = str(exc)
                except Exception as exc:  # pragma: no cover - 真实 runner 兜底
                    result_counts["failure"] += 1
                    fatal_error = str(exc)
                    stopped_by_fatal_error = True
                else:
                    if result.from_cache:
                        result_counts["cache"] += 1
                    elif result.state == "frozen":
                        result_counts["success"] += 1
                    else:
                        result_counts["failure"] += 1
                    if not result.from_cache and result.total_cost_usd is not None:
                        total_cost_usd += float(result.total_cost_usd)
                completed_task_count += 1
                report_payload = _build_report(
                    loaded_plan=loaded_plan,
                    store=store,
                    entries=loaded_plan.entries,
                    selected_entries=selected_entries,
                    submitted_task_count=submitted_task_count,
                    completed_task_count=completed_task_count,
                    result_counts=result_counts,
                    started_at_monotonic=started_at_monotonic,
                    requested_logical_ids=logical_ids,
                    limit=limit,
                    workers=workers,
                    stopped_by_circuit_breaker=stopped_by_circuit_breaker,
                    stopped_by_fatal_error=stopped_by_fatal_error,
                        recovered_expired_attempt_ids=recovered_expired_attempt_ids,
                        predecessor_attempt_manifest=loaded_predecessor_manifest,
                        attempts_reserved_at_start=attempts_reserved_at_start,
                        status="failed" if fatal_error else "running",
                    error=fatal_error,
                )
                report_payload["total_cost_usd"] = total_cost_usd
                _atomic_write_json(report_path, report_payload)

            while (
                not stopped_by_circuit_breaker
                and not stopped_by_fatal_error
                and next_index < len(runnable_entries)
                and len(in_flight) < workers
            ):
                entry = runnable_entries[next_index]
                next_index += 1
                future = executor.submit(runner.execute, entry.definition)
                in_flight[future] = entry
                submitted_task_count += 1

    final_status = "failed" if fatal_error else "completed"
    final_report = _build_report(
        loaded_plan=loaded_plan,
        store=store,
        entries=loaded_plan.entries,
        selected_entries=selected_entries,
        submitted_task_count=submitted_task_count,
        completed_task_count=completed_task_count,
        result_counts=result_counts,
        started_at_monotonic=started_at_monotonic,
        requested_logical_ids=logical_ids,
        limit=limit,
        workers=workers,
        stopped_by_circuit_breaker=stopped_by_circuit_breaker,
        stopped_by_fatal_error=stopped_by_fatal_error,
        recovered_expired_attempt_ids=recovered_expired_attempt_ids,
        predecessor_attempt_manifest=loaded_predecessor_manifest,
        attempts_reserved_at_start=attempts_reserved_at_start,
        status=final_status,
        error=fatal_error,
    )
    final_report["total_cost_usd"] = total_cost_usd
    _atomic_write_json(report_path, final_report)
    return 1 if fatal_error else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 Stage5 Claude plan JSONL")
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--logical-id", dest="logical_ids", action="append", default=[])
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--physical-attempt-offset", type=int, default=0)
    parser.add_argument("--predecessor-attempt-manifest", type=Path, default=None)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[[TaskStateStore, Path, Path], Any] | None = None,
) -> int:
    args = parse_args(argv)
    return execute_plan(
        plan_jsonl=args.plan_jsonl,
        state_db=args.state_db,
        attempts_dir=args.attempts_dir,
        frozen_dir=args.frozen_dir,
        report_json=args.report_json,
        limit=args.limit,
        logical_ids=tuple(args.logical_ids),
        workers=args.workers,
        physical_attempt_offset=args.physical_attempt_offset,
        predecessor_attempt_manifest=args.predecessor_attempt_manifest,
        runner_factory=runner_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
