"""Core-50 最终发布补评的 Routify 单渠道控制器。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.anthropic_api_runner import (
    API_TRANSPORT_VERSION,
    STRICT_EVALUATOR_SYSTEM_PROMPT,
    AnthropicApiResponse,
    AnthropicApiChannel,
    AnthropicApiRunner,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_api_frozen_simplifications import (
    _audit_one as _audit_one_api_frozen,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.audit_frozen_simplifications import (
    _audit_one_frozen as _audit_one_cli_frozen,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    CONTRACT_CANONICAL_MODEL,
    CONTRACT_EFFORT,
    CONTRACT_MODEL,
    CONTRACT_TRANSPORT_VERSION,
    canonical_json,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_runner import (
    _atomic_write_json,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.import_frozen_plan_results import (
    FrozenPlanImportError,
    import_frozen_plan_results,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_anthropic_api_plan import (
    load_api_channels,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    _load_predecessor_attempt_manifest,
    execute_plan,
    load_plan_jsonl,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import (
    PredecessorAttemptManifest,
    TaskStateStore,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK_ROOT = (
    REPO_ROOT
    / "AAAI_experiments/stage5_metric_calculation_0831/work"
    / "final_release_20260913/release_v2/llm"
)
DEFAULT_ROUTIFY_PROFILE = Path.home() / ".claude/settings-jyh.json"
ROUTIFY_BASE_URL = "https://routify-pub.alibaba-inc.com/protocol/anthropic"
MAX_CONCURRENCY = 32
MAX_ATTEMPTS_PER_TASK = 3
BACKOFF_SCHEDULE_SECONDS = (2.0, 8.0)
_PHASE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ReleaseControlError(ValueError):
    """发布控制参数或凭据不满足冻结约束。"""


def physical_attempt_cap(logical_total: int) -> int:
    if logical_total <= 0:
        raise ReleaseControlError("logical_total 必须是正整数")
    return math.ceil(1.5 * logical_total)


def state_attempt_cap(logical_total: int, *, cache_import_total: int = 0) -> int:
    if cache_import_total < 0:
        raise ReleaseControlError("cache_import_total 不能为负数")
    # cache_import 会占用 SQLite attempt 行，但不会发送网络请求。
    return physical_attempt_cap(logical_total) + cache_import_total


@dataclass(frozen=True)
class ReleasePaths:
    root: Path
    state_db: Path
    attempts_dir: Path
    frozen_dir: Path
    report_json: Path
    control_json: Path

    @classmethod
    def from_root(cls, root: str | Path, phase: str) -> "ReleasePaths":
        if not _PHASE_PATTERN.fullmatch(phase):
            raise ReleaseControlError("phase 只能包含小写字母、数字、下划线和连字符")
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            state_db=resolved / "release_state.sqlite3",
            attempts_dir=resolved / "attempts",
            frozen_dir=resolved / "frozen",
            report_json=resolved / "reports" / f"{phase}.json",
            control_json=resolved / "reports" / f"{phase}.control.json",
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _release_scope(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.name == "release_v2":
            return candidate
    raise ReleaseControlError("work_root 必须位于 release_v2 目录内")


def load_release_channel(
    profile: str | Path = DEFAULT_ROUTIFY_PROFILE,
) -> tuple[AnthropicApiChannel, dict[str, object]]:
    profile_path = Path(profile).expanduser().resolve()
    channels = load_api_channels(
        (f"routify={profile_path}",),
        allow_single_channel=True,
    )
    if len(channels) != 1:
        raise ReleaseControlError("Routify 发布控制器只允许一个 API 渠道")
    channel = channels[0]
    if channel.base_url.rstrip("/") != ROUTIFY_BASE_URL:
        raise ReleaseControlError("Routify profile 的 endpoint 不是冻结的正式地址")
    parsed = urlparse(channel.base_url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ReleaseControlError("Routify profile 必须使用无用户信息的标准 HTTPS 地址")
    safe_metadata = {
        "api_channel": "routify",
        "api_host": parsed.hostname,
        "api_path": parsed.path.rstrip("/"),
        "credential_loaded": bool(channel.auth_token),
    }
    return channel, safe_metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def prepare_release_state(
    *,
    work_root: str | Path,
    plan_jsonls: Sequence[str | Path],
    logical_total: int,
    cache_import_total: int,
    predecessor_attempt_manifest: str | Path | None = None,
) -> TaskStateStore:
    root = Path(work_root).expanduser().resolve()
    release_scope = _release_scope(root)
    definitions: dict[str, object] = {}
    for plan_jsonl in plan_jsonls:
        plan_path = Path(plan_jsonl).expanduser().resolve()
        if not _is_relative_to(plan_path, release_scope):
            raise ReleaseControlError("plan 必须位于本次 release_v2 工作区内")
        for entry in load_plan_jsonl(plan_path).entries:
            previous = definitions.get(entry.evaluation_key)
            if previous is not None and previous != entry.definition.task_spec:
                raise ReleaseControlError(f"跨 plan evaluation_key 漂移: {entry.evaluation_key}")
            definitions[entry.evaluation_key] = entry.definition.task_spec
    if len(definitions) > logical_total:
        raise ReleaseControlError("待注册任务数超过本次发布逻辑任务总量")
    predecessor = _predecessor_manifest(predecessor_attempt_manifest)
    store = TaskStateStore(
        root / "release_state.sqlite3",
        attempt_cap=state_attempt_cap(
            logical_total,
            cache_import_total=cache_import_total,
        ),
        logical_task_cap=logical_total,
        max_attempts_per_task=MAX_ATTEMPTS_PER_TASK,
        predecessor_attempt_manifest=predecessor,
    )
    store.register_tasks(tuple(definitions.values()))
    return store


def _copy_verified(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ReleaseControlError(f"cache source 文件不存在: {source}")
    source_sha256 = _sha256_file(source)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_sha256:
            raise ReleaseControlError(f"cache staging 已存在不同内容: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    if _sha256_file(destination) != source_sha256:
        raise ReleaseControlError(f"cache staging 复制后 SHA256 不一致: {destination}")


def bootstrap_api_frozen(
    *,
    work_root: str | Path,
    phase: str,
    plan_jsonl: str | Path,
    source_frozen_dir: str | Path,
    source_attempts_dir: str | Path,
    logical_total: int,
    cache_import_total: int,
    predecessor_attempt_manifest: str | Path | None = None,
) -> dict[str, object]:
    if cache_import_total <= 0:
        raise ReleaseControlError("bootstrap 要求 cache_import_total 为正整数")
    paths = ReleasePaths.from_root(work_root, phase)
    plan_path = Path(plan_jsonl).expanduser().resolve()
    loaded_plan = load_plan_jsonl(plan_path)
    if len(loaded_plan.entries) > cache_import_total:
        raise ReleaseControlError("bootstrap plan 数量超过冻结的 cache_import_total")
    prepare_release_state(
        work_root=paths.root,
        plan_jsonls=(plan_path,),
        logical_total=logical_total,
        cache_import_total=cache_import_total,
        predecessor_attempt_manifest=predecessor_attempt_manifest,
    )

    source_frozen = Path(source_frozen_dir).expanduser().resolve()
    source_attempts = Path(source_attempts_dir).expanduser().resolve()
    staging_root = paths.root / "imports" / phase
    staging_frozen = staging_root / "frozen"
    staging_attempts = staging_root / "attempts"
    for entry in loaded_plan.entries:
        frozen_source = source_frozen / f"{entry.evaluation_key}.json"
        payload = json.loads(frozen_source.read_text(encoding="utf-8"))
        attempt_id = payload.get("attempt_id") if isinstance(payload, Mapping) else None
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ReleaseControlError(f"source frozen attempt_id 缺失: {entry.logical_id}")
        _copy_verified(frozen_source, staging_frozen / frozen_source.name)
        _copy_verified(
            source_attempts / f"{attempt_id}.json",
            staging_attempts / f"{attempt_id}.json",
        )
    report_path = paths.root / "reports" / f"bootstrap_{phase}.json"
    report = import_frozen_plan_results(
        plan_jsonl=plan_path,
        state_db=paths.state_db,
        frozen_dir=staging_frozen,
        attempts_dir=staging_attempts,
        report_json=report_path,
    )
    if report.get("status") != "completed":
        raise ReleaseControlError(f"API frozen bootstrap 未完整完成: {report_path}")
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    return {
        "contract": "core50_release_api_cache_bootstrap.v1",
        "phase": phase,
        "network_requests": 0,
        "plan_task_count": len(loaded_plan.entries),
        "imported": counts.get("imported", 0),
        "cached": counts.get("cached", 0),
        "state_db": str(paths.state_db),
        "report_json": str(report_path),
        "report_sha256": _sha256_file(report_path),
    }


def _load_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseControlError(f"{path}:{line_number} JSONL 无效: {exc}") from exc
            if not isinstance(row, dict):
                raise ReleaseControlError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(row)
    if not rows:
        raise ReleaseControlError(f"index 不能为空: {path}")
    return rows


def _source_attempt_path(result_path: Path, attempt_id: str) -> Path:
    parent = result_path.parent
    if parent.name == "frozen_v2":
        return parent.parent / "attempts_v2" / f"{attempt_id}.json"
    if parent.name == "api_frozen_v1":
        return parent.parent / "api_attempts_v1" / f"{attempt_id}.json"
    if parent.name == "api_frozen":
        return parent.parent / "api_attempts" / f"{attempt_id}.json"
    if parent.name == "frozen":
        return parent.parent / "attempts" / f"{attempt_id}.json"
    raise ReleaseControlError(f"无法从 frozen 路径推断 source attempts: {result_path}")


def _source_rows(
    state_dbs: Sequence[Path],
    wanted_keys: set[str],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for state_db in state_dbs:
        with sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            for start in range(0, len(wanted_keys), 500):
                chunk = sorted(wanted_keys)[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                records = connection.execute(
                    """SELECT t.*, f.attempt_id AS frozen_attempt_id,
                              f.result_path, f.result_sha256,
                              a.attempt_number, a.status AS attempt_status
                         FROM tasks t
                         JOIN frozen_results f USING(evaluation_key)
                         JOIN attempts a ON a.attempt_id=f.attempt_id
                        WHERE t.evaluation_key IN ("""
                    + placeholders
                    + ")",
                    chunk,
                ).fetchall()
                for record in records:
                    row = dict(record)
                    row["source_state_db"] = str(state_db.resolve())
                    result.setdefault(str(record["evaluation_key"]), []).append(row)
    return result


def _resolve_source_binding(
    *,
    index_row: Mapping[str, object],
    entry: object,
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result_path = Path(str(index_row.get("result_path"))).expanduser().resolve()
    wanted_sha = str(index_row.get("result_sha256"))
    wanted_attempt_id = str(index_row.get("attempt_id"))
    declared_source_db = index_row.get("source_state_db")
    declared_source_db_path = (
        Path(str(declared_source_db)).expanduser().resolve()
        if isinstance(declared_source_db, str) and declared_source_db
        else None
    )
    matches = []
    for candidate in candidates:
        candidate_path = Path(str(candidate.get("result_path"))).expanduser().resolve()
        if (
            candidate.get("result_sha256") == wanted_sha
            and candidate.get("frozen_attempt_id") == wanted_attempt_id
            and candidate.get("spec_json") == entry.definition.task_spec.canonical_json()
            and candidate.get("attempt_status") == "accepted"
            and (
                declared_source_db_path is None
                or Path(str(candidate.get("source_state_db"))).resolve()
                == declared_source_db_path
            )
            and candidate_path.is_file()
            and _sha256_file(candidate_path) == wanted_sha
        ):
            matches.append(dict(candidate))
    if not matches:
        raise ReleaseControlError(f"source state 未找到完整 frozen binding: {entry.logical_id}")
    matches.sort(
        key=lambda row: (
            "/imports/" in Path(str(row.get("result_path"))).resolve().as_posix(),
            Path(str(row.get("result_path"))).resolve() != result_path,
        )
    )
    first = matches[0]
    identity_fields = ("frozen_attempt_id", "result_sha256", "spec_json")
    if any(any(row.get(field) != first.get(field) for field in identity_fields) for row in matches[1:]):
        raise ReleaseControlError(f"source state frozen binding 冲突: {entry.logical_id}")
    return first


def _audit_mixed_source(
    *,
    index_row: Mapping[str, object],
    entry: object,
    state_row: Mapping[str, object],
    release_scope: Path,
) -> dict[str, object]:
    result_path = Path(str(index_row["result_path"])).expanduser().resolve()
    if not result_path.is_file() or _sha256_file(result_path) != index_row.get("result_sha256"):
        raise ReleaseControlError(f"source frozen 文件或 SHA 漂移: {entry.logical_id}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReleaseControlError(f"source frozen 不是 object: {entry.logical_id}")
    expected_top = {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "task_type": entry.definition.task_spec.task_type,
        "task_kind": "simplify",
        "structured_output": index_row.get("structured_output"),
    }
    for field, expected in expected_top.items():
        if payload.get(field) != expected:
            raise ReleaseControlError(f"source frozen.{field} 与 index/plan 不一致: {entry.logical_id}")
    attempt_id = str(state_row["frozen_attempt_id"])
    indexed_attempt_path = index_row.get("source_attempt_path")
    attempt_path = (
        Path(str(indexed_attempt_path)).expanduser().resolve()
        if isinstance(indexed_attempt_path, str) and indexed_attempt_path
        else _source_attempt_path(result_path, attempt_id).resolve()
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ReleaseControlError(f"source frozen metadata 缺失: {entry.logical_id}")
    transport = metadata.get("transport_version")
    if transport == API_TRANSPORT_VERSION:
        api_state_row = {
            "attempt_id": attempt_id,
            "result_path": str(result_path),
            "result_sha256": state_row["result_sha256"],
        }
        def run_audit() -> Mapping[str, object]:
            return _audit_one_api_frozen(
                entry=entry,
                state_row=api_state_row,
                attempts_dir=attempt_path.parent,
                frozen_dir=result_path.parent,
                semantic_timeout_seconds=300.0,
            )

        if _is_relative_to(result_path, release_scope) and metadata.get("api_channel") != "routify":
            raise ReleaseControlError(f"本次新 frozen 不是 Routify: {entry.logical_id}")
        source_transport = "anthropic_api"
    elif transport in {CONTRACT_TRANSPORT_VERSION, None}:
        auditor_state_row = dict(state_row)
        auditor_state_row["result_path"] = str(result_path)
        auditor_state_row["result_sha256"] = index_row["result_sha256"]
        def run_audit() -> Mapping[str, object]:
            return _audit_one_cli_frozen(
                entry=entry,
                state_row=auditor_state_row,
                attempts_dir=attempt_path.parent,
                frozen_dir=result_path.parent,
                expected_rendered_prompt=render_prompt(
                    entry.definition.prompt_template,
                    entry.definition.request,
                    entry.definition.schema,
                ),
                semantic_timeout_seconds=300.0,
                allow_archived_execution_metadata=True,
                allowed_task_states=frozenset({"frozen", "superseded"}),
            )

        source_transport = "claude_code_cli"
    else:
        raise ReleaseControlError(f"未知 source transport: {entry.logical_id}: {transport!r}")
    audit: Mapping[str, object] = {}
    for _ in range(3):
        audit = run_audit()
        if audit.get("status") == "passed" or audit.get("failure_class") not in {
            "semantic_validator_error",
            "semantic_validator_timeout",
        }:
            break
    semantic_validation_mode = "current_revalidation"
    preexisting_semantic_failure = (
        index_row.get("evidence_generation") == "preexisting"
        and audit.get("failure_class")
        in {
            "semantic_rejected",
            "semantic_validator_error",
            "semantic_validator_timeout",
            "semantic_validation_failed",
        }
    )
    if preexisting_semantic_failure:
        # 旧冻结结果继续依赖当时已落盘的 validation 证据；当前验证器漂移不能触发重判。
        semantic_validation_mode = "original_frozen_validation"
    elif audit.get("status") != "passed":
        raise ReleaseControlError(
            f"source frozen 严格审计失败: {entry.logical_id}: "
            f"{audit.get('failure_class')}: {audit.get('error')}"
        )
    return {
        "evaluation_key": entry.evaluation_key,
        "logical_id": entry.logical_id,
        "source_transport": source_transport,
        "semantic_validation_mode": semantic_validation_mode,
        "source_state_db": state_row["source_state_db"],
        "source_result_path": str(result_path),
        "source_result_sha256": index_row["result_sha256"],
        "source_attempt_id": attempt_id,
        "source_attempt_path": str(attempt_path),
        "source_attempt_sha256": _sha256_file(attempt_path),
    }


def bootstrap_mixed_frozen_indexes(
    *,
    work_root: str | Path,
    index_plan_pairs: Sequence[tuple[str | Path, str | Path]],
    source_state_dbs: Sequence[str | Path],
    logical_total: int,
    cache_import_total: int,
    predecessor_attempt_manifest: str | Path,
    workers: int = 16,
) -> dict[str, object]:
    root = Path(work_root).expanduser().resolve()
    release_scope = _release_scope(root)
    if workers <= 0 or workers > 32:
        raise ReleaseControlError("mixed bootstrap workers 必须位于1到32")
    all_rows: dict[str, dict[str, object]] = {}
    entries: dict[str, object] = {}
    plan_paths: list[Path] = []
    for raw_index, raw_plan in index_plan_pairs:
        index_path = Path(raw_index).expanduser().resolve()
        plan_path = Path(raw_plan).expanduser().resolve()
        loaded = load_plan_jsonl(plan_path)
        plan_by_key = {entry.evaluation_key: entry for entry in loaded.entries}
        plan_paths.append(plan_path)
        for row in _load_jsonl_objects(index_path):
            key = str(row.get("evaluation_key"))
            entry = plan_by_key.get(key)
            if entry is None:
                raise ReleaseControlError(f"index key 不在绑定 plan: {key}")
            expected = {
                "logical_id": entry.logical_id,
                "task_type": entry.definition.task_spec.task_type,
                "condition": entry.definition.task_spec.condition,
                "priority": entry.definition.task_spec.priority,
                "plan_sha256": loaded.plan_sha256,
                "state": "frozen",
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    raise ReleaseControlError(f"index.{field} 漂移: {entry.logical_id}")
            if key in all_rows:
                raise ReleaseControlError(f"mixed index evaluation_key 重复: {key}")
            all_rows[key] = row
            entries[key] = entry
    if len(all_rows) != cache_import_total:
        raise ReleaseControlError(
            f"mixed index 数量 {len(all_rows)} != cache_import_total {cache_import_total}"
        )
    source_dbs = [Path(path).expanduser().resolve() for path in source_state_dbs]
    source_candidates = _source_rows(source_dbs, set(all_rows))

    def audit_key(key: str) -> dict[str, object]:
        entry = entries[key]
        binding = _resolve_source_binding(
            index_row=all_rows[key],
            entry=entry,
            candidates=source_candidates.get(key, ()),
        )
        return _audit_mixed_source(
            index_row=all_rows[key],
            entry=entry,
            state_row=binding,
            release_scope=release_scope,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        audited = list(executor.map(audit_key, sorted(all_rows)))
    audited_by_key = {str(row["evaluation_key"]): row for row in audited}

    store = prepare_release_state(
        work_root=root,
        plan_jsonls=tuple(plan_paths),
        logical_total=logical_total,
        cache_import_total=cache_import_total,
        predecessor_attempt_manifest=predecessor_attempt_manifest,
    )
    order = {"clean": 0, "noise001": 1, "noise005": 2}
    imported = cached = 0
    import_rows: list[dict[str, object]] = []
    for key in sorted(
        all_rows,
        key=lambda item: (
            order[entries[item].definition.task_spec.condition],
            entries[item].definition.task_spec.priority,
            entries[item].logical_id,
        ),
    ):
        entry = entries[key]
        audit = audited_by_key[key]
        if store.task_state(key) == "frozen":
            cached += 1
            continue
        source_result = Path(str(audit["source_result_path"]))
        source_attempt = Path(str(audit["source_attempt_path"]))
        staged_result = root / "imports" / "frozen" / source_result.name
        staged_attempt = root / "imports" / "source_attempts" / source_attempt.name
        _copy_verified(source_result, staged_result)
        _copy_verified(source_attempt, staged_attempt)
        lease = store.reserve_attempt(key, lease_seconds=300.0)
        import_audit = {
            "schema_version": "mixed_cache_import_attempt.v1",
            "provider": "local_cache_import",
            "network_request": False,
            "attempt_id": lease.attempt_id,
            "evaluation_key": key,
            "logical_id": entry.logical_id,
            "source_transport": audit["source_transport"],
            "source_state_db": audit["source_state_db"],
            "source_result_sha256": audit["source_result_sha256"],
            "source_attempt_sha256": audit["source_attempt_sha256"],
            "staged_result_path": str(staged_result),
            "staged_result_sha256": _sha256_file(staged_result),
        }
        audit_path = root / "imports" / "cache_audits" / f"{lease.attempt_id}.json"
        _atomic_write_json(audit_path, import_audit)
        store.freeze_result(
            lease.attempt_id,
            result_path=str(staged_result),
            result_sha256=_sha256_file(staged_result),
        )
        imported += 1
        import_rows.append(
            {
                "evaluation_key": key,
                "logical_id": entry.logical_id,
                "source_transport": audit["source_transport"],
                "source_result_sha256": audit["source_result_sha256"],
                "cache_audit_path": str(audit_path),
                "cache_audit_sha256": _sha256_file(audit_path),
            }
        )
    report = {
        "schema_version": "core50_release_mixed_cache_bootstrap.v1",
        "status": "ok",
        "network_requests": 0,
        "planned": len(all_rows),
        "audited": len(audited),
        "imported": imported,
        "cached": cached,
        "source_transport_counts": dict(
            sorted(
                {
                    name: sum(row["source_transport"] == name for row in audited)
                    for name in {str(row["source_transport"]) for row in audited}
                }.items()
            )
        ),
        "state_db": str((root / "release_state.sqlite3").resolve()),
        "entries_sha256": hashlib.sha256(canonical_json(import_rows).encode("utf-8")).hexdigest(),
    }
    report_path = root / "reports" / "mixed_cache_bootstrap.json"
    _atomic_write_json(report_path, report)
    return {**report, "report_json": str(report_path), "report_sha256": _sha256_file(report_path)}


def _safe_control_payload(
    *,
    phase: str,
    plan_path: Path,
    plan_count: int,
    paths: ReleasePaths,
    logical_total: int,
    cache_import_total: int,
    workers: int,
    limit: int | None,
    selected_condition: str | None,
    selected_task_count: int,
    channel_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "contract": "core50_final_release_routify.v1",
        "phase": phase,
        "plan_jsonl": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "plan_task_count": plan_count,
        "selected_condition": selected_condition,
        "selected_task_count": selected_task_count,
        "state_db": str(paths.state_db),
        "attempts_dir": str(paths.attempts_dir),
        "frozen_dir": str(paths.frozen_dir),
        "model": CONTRACT_CANONICAL_MODEL,
        "effort": CONTRACT_EFFORT,
        "stream": False,
        "workers": workers,
        "per_channel_concurrency": workers,
        "semantic_validation_concurrency": min(4, workers),
        "backoff_schedule_seconds": list(BACKOFF_SCHEDULE_SECONDS),
        "logical_total": logical_total,
        "physical_attempt_cap": physical_attempt_cap(logical_total),
        "cache_import_total": cache_import_total,
        "state_attempt_cap": state_attempt_cap(
            logical_total,
            cache_import_total=cache_import_total,
        ),
        "max_attempts_per_task": MAX_ATTEMPTS_PER_TASK,
        "limit": limit,
        "cost_monitoring": "attempt_metadata_and_phase_report",
        "hard_cost_cap_supported": False,
        **dict(channel_metadata),
    }


def run_phase(
    *,
    work_root: str | Path,
    phase: str,
    plan_jsonl: str | Path,
    profile: str | Path = DEFAULT_ROUTIFY_PROFILE,
    logical_total: int,
    workers: int = MAX_CONCURRENCY,
    limit: int | None = None,
    cache_import_total: int = 0,
    predecessor_attempt_manifest: str | Path | None = None,
    condition: str | None = None,
) -> int:
    if workers <= 0 or workers > MAX_CONCURRENCY:
        raise ReleaseControlError(f"workers 必须位于 1 到 {MAX_CONCURRENCY}")
    if limit is not None and limit <= 0:
        raise ReleaseControlError("limit 必须是正整数")
    paths = ReleasePaths.from_root(work_root, phase)
    plan_path = Path(plan_jsonl).expanduser().resolve()
    release_scope = _release_scope(paths.root)
    if not _is_relative_to(plan_path, release_scope):
        raise ReleaseControlError("plan 必须位于本次 release_v2 工作区内")
    loaded_plan = load_plan_jsonl(plan_path)
    if len(loaded_plan.entries) > logical_total:
        raise ReleaseControlError("当前 plan 的任务数超过本次发布逻辑任务总量")
    channel, safe_channel_metadata = load_release_channel(profile)
    if condition is not None and condition not in {"clean", "noise001", "noise005"}:
        raise ReleaseControlError(f"未知 condition: {condition}")
    selected_logical_ids = tuple(
        entry.logical_id
        for entry in loaded_plan.entries
        if condition is None or entry.definition.task_spec.condition == condition
    )
    if not selected_logical_ids:
        raise ReleaseControlError("当前 condition 在 plan 中没有任务")
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.attempts_dir.mkdir(parents=True, exist_ok=True)
    paths.frozen_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        paths.control_json,
        _safe_control_payload(
            phase=phase,
            plan_path=plan_path,
            plan_count=len(loaded_plan.entries),
            paths=paths,
            logical_total=logical_total,
            cache_import_total=cache_import_total,
            workers=workers,
            limit=limit,
            selected_condition=condition,
            selected_task_count=len(selected_logical_ids),
            channel_metadata=safe_channel_metadata,
        ),
    )

    def runner_factory(
        store: TaskStateStore,
        *,
        attempts_dir: Path,
        frozen_dir: Path,
    ) -> AnthropicApiRunner:
        return AnthropicApiRunner(
            store,
            attempts_dir=attempts_dir,
            frozen_dir=frozen_dir,
            channels=(channel,),
            timeout_seconds=300.0,
            max_tokens=4096,
            per_channel_concurrency=workers,
            semantic_validation_concurrency=min(4, workers),
            backoff_schedule_seconds=BACKOFF_SCHEDULE_SECONDS,
            allow_single_channel=True,
        )

    return execute_plan(
        plan_jsonl=plan_path,
        state_db=paths.state_db,
        attempts_dir=paths.attempts_dir,
        frozen_dir=paths.frozen_dir,
        report_json=paths.report_json,
        limit=limit,
        logical_ids=selected_logical_ids,
        workers=workers,
        attempt_cap=state_attempt_cap(
            logical_total,
            cache_import_total=cache_import_total,
        ),
        logical_task_cap=logical_total,
        max_attempts_per_task=MAX_ATTEMPTS_PER_TASK,
        predecessor_attempt_manifest=predecessor_attempt_manifest,
        runner_factory=runner_factory,
    )


def release_status(work_root: str | Path) -> dict[str, object]:
    root = Path(work_root).expanduser().resolve()
    attempts_dir = root / "attempts"
    attempt_count = 0
    estimated_cost_cny = 0.0
    error_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    for path in sorted(attempts_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            error_counts["unreadable_attempt"] = error_counts.get("unreadable_attempt", 0) + 1
            continue
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if not isinstance(metadata, Mapping):
            error_counts["missing_metadata"] = error_counts.get("missing_metadata", 0) + 1
            continue
        attempt_count += 1
        cost = metadata.get("estimated_cost_cny")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            estimated_cost_cny += float(cost)
        error_class = metadata.get("error_class")
        if isinstance(error_class, str) and error_class:
            error_counts[error_class] = error_counts.get(error_class, 0) + 1
        channel = metadata.get("api_channel")
        if isinstance(channel, str) and channel:
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

    state_counts: dict[str, int] = {}
    state_db = root / "release_state.sqlite3"
    if not state_db.is_file() and (root / "state.sqlite3").is_file():
        state_db = root / "state.sqlite3"
    if state_db.is_file():
        with sqlite3.connect(state_db) as connection:
            state_counts = {
                str(state): int(count)
                for state, count in connection.execute(
                    "SELECT state, COUNT(*) FROM tasks GROUP BY state ORDER BY state"
                )
            }
    return {
        "contract": "core50_final_release_routify_status.v1",
        "state_db": str(state_db),
        "state_db_present": state_db.is_file(),
        "attempt_file_count": attempt_count,
        "estimated_cost_cny": estimated_cost_cny,
        "state_counts": state_counts,
        "error_counts": dict(sorted(error_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "hard_cost_cap_supported": False,
    }


def export_frozen_index(
    *,
    state_db: str | Path,
    output_jsonl: str | Path,
    plan_jsonls: Sequence[str | Path] = (),
    source_state_db_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    state_path = Path(state_db).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    if not state_path.is_file():
        raise ReleaseControlError(f"state DB 不存在: {state_path}")
    rows: list[dict[str, object]] = []
    plan_bindings: dict[str, tuple[str, str]] = {}
    for plan_jsonl in plan_jsonls:
        loaded = load_plan_jsonl(plan_jsonl)
        for entry in loaded.entries:
            plan_bindings.setdefault(
                entry.evaluation_key,
                (str(loaded.plan_path), loaded.plan_sha256),
            )
    with sqlite3.connect(f"{state_path.as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        records = connection.execute(
            """SELECT t.evaluation_key, t.logical_id, t.task_type,
                      t.condition_name, t.priority, f.attempt_id,
                      f.result_path, f.result_sha256
                 FROM frozen_results f
                 JOIN tasks t USING(evaluation_key)
                ORDER BY t.priority, t.logical_id"""
        ).fetchall()
    for record in records:
        if plan_bindings and str(record["evaluation_key"]) not in plan_bindings:
            continue
        result_path = Path(str(record["result_path"])).expanduser().resolve()
        if not result_path.is_file():
            raise ReleaseControlError(f"frozen 文件不存在: {result_path}")
        actual_sha256 = _sha256_file(result_path)
        if actual_sha256 != record["result_sha256"]:
            raise ReleaseControlError(f"frozen SHA256 漂移: {record['evaluation_key']}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise ReleaseControlError(f"frozen metadata 缺失: {record['evaluation_key']}")
        source_attempt_id = payload.get("attempt_id") if isinstance(payload, Mapping) else None
        if not isinstance(source_attempt_id, str) or not source_attempt_id:
            raise ReleaseControlError(f"frozen source attempt_id 缺失: {record['evaluation_key']}")
        state_binding_attempt_id = str(record["attempt_id"])
        source_state_db = state_path
        if source_state_db_overrides and str(record["evaluation_key"]) in source_state_db_overrides:
            source_state_db = Path(
                source_state_db_overrides[str(record["evaluation_key"])]
            ).expanduser().resolve()
        rows.append(
            {
                "evaluation_key": record["evaluation_key"],
                "logical_id": record["logical_id"],
                "task_type": record["task_type"],
                "condition": record["condition_name"],
                "priority": int(record["priority"]),
                "attempt_id": source_attempt_id,
                "state_binding_attempt_id": state_binding_attempt_id,
                "cache_import_attempt_id": (
                    state_binding_attempt_id
                    if state_binding_attempt_id != source_attempt_id
                    or "/imports/" in result_path.as_posix()
                    else None
                ),
                "result_path": str(result_path),
                "result_sha256": actual_sha256,
                "source_state_db": str(source_state_db),
                "source_attempt_path": str(
                    _source_attempt_path(result_path, source_attempt_id).resolve()
                ),
                "requested_model": metadata.get("requested_model"),
                "requested_effort": metadata.get("requested_effort"),
                "api_channel": metadata.get("api_channel"),
                "stream": metadata.get("stream"),
                "estimated_cost_cny": metadata.get("estimated_cost_cny"),
                "plan_jsonl": plan_bindings.get(str(record["evaluation_key"]), (None, None))[0],
                "plan_sha256": plan_bindings.get(str(record["evaluation_key"]), (None, None))[1],
            }
        )
    if plan_jsonls:
        missing_bindings = [row["evaluation_key"] for row in rows if row["plan_sha256"] is None]
        if missing_bindings:
            raise ReleaseControlError(
                f"frozen index 有 {len(missing_bindings)} 条缺少 plan 绑定"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return {
        "schema_version": "core50_release_frozen_index.v1",
        "state_db": str(state_path),
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "frozen_count": len(rows),
        "plan_count": len(plan_jsonls),
    }


def export_predecessor_manifest(
    *,
    state_db: str | Path,
    attempts_dir: str | Path,
    output_json: str | Path,
) -> dict[str, object]:
    state_path = Path(state_db).expanduser().resolve()
    attempts_path = Path(attempts_dir).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    with sqlite3.connect(f"{state_path.as_uri()}?mode=ro", uri=True) as connection:
        attempt_ids = [
            str(row[0])
            for row in connection.execute("SELECT attempt_id FROM attempts ORDER BY attempt_id")
        ]
    attempt_hashes: dict[str, str] = {}
    for attempt_id in attempt_ids:
        attempt_path = attempts_path / f"{attempt_id}.json"
        if not attempt_path.is_file():
            raise ReleaseControlError(f"predecessor attempt 文件不存在: {attempt_path}")
        payload = json.loads(attempt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("attempt_id") != attempt_id:
            raise ReleaseControlError(f"predecessor attempt 身份不一致: {attempt_id}")
        attempt_hashes[attempt_id] = _sha256_file(attempt_path)
    manifest = {
        "schema_version": "predecessor_attempts.v1",
        "attempt_count": len(attempt_ids),
        "attempt_ids": attempt_ids,
        "attempt_file_sha256": attempt_hashes,
        "source_state_db": str(state_path),
        "source_attempts_dir": str(attempts_path),
    }
    _atomic_write_json(output_path, manifest)
    return {
        "schema_version": "core50_release_predecessor_export.v1",
        "attempt_count": len(attempt_ids),
        "output_json": str(output_path),
        "output_sha256": _sha256_file(output_path),
    }


def export_api_attempt_ledger(
    *,
    source_attempt_dirs: Sequence[str | Path],
    output_root: str | Path,
) -> dict[str, object]:
    root = Path(output_root).expanduser().resolve()
    attempts_output = root / "attempts"
    attempts_output.mkdir(parents=True, exist_ok=True)
    records: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for raw_dir in source_attempt_dirs:
        source_dir = Path(raw_dir).expanduser().resolve()
        for path in sorted(source_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            if not isinstance(metadata, Mapping) or not metadata.get("api_channel"):
                continue
            attempt_id = payload.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                raise ReleaseControlError(f"API attempt_id 缺失: {path}")
            expected = {
                "api_channel": "routify",
                "requested_model": CONTRACT_CANONICAL_MODEL,
                "requested_effort": CONTRACT_EFFORT,
                "stream": False,
            }
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ReleaseControlError(
                        f"API ledger {attempt_id} 的 {field} 不符合本次 Routify 契约"
                    )
            previous = records.get(attempt_id)
            if previous is not None and _sha256_file(previous[0]) != _sha256_file(path):
                raise ReleaseControlError(f"API ledger attempt_id 内容冲突: {attempt_id}")
            records[attempt_id] = (path, metadata)

    attempt_hashes: dict[str, str] = {}
    accepted = 0
    estimated_cost_cny = 0.0
    for attempt_id, (source, metadata) in sorted(records.items()):
        destination = attempts_output / f"{attempt_id}.json"
        _copy_verified(source, destination)
        attempt_hashes[attempt_id] = _sha256_file(destination)
        accepted += metadata.get("error_class") is None
        cost = metadata.get("estimated_cost_cny")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            estimated_cost_cny += float(cost)

    state_db = root / "attempt_ledger.sqlite3"
    temporary_db = root / ".attempt_ledger.sqlite3.tmp"
    temporary_db.unlink(missing_ok=True)
    with sqlite3.connect(temporary_db) as connection:
        connection.execute("CREATE TABLE attempts(attempt_id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO attempts(attempt_id) VALUES (?)",
            [(attempt_id,) for attempt_id in sorted(records)],
        )
    temporary_db.replace(state_db)
    manifest_path = root / "predecessor_attempts.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": "predecessor_attempts.v1",
            "attempt_count": len(records),
            "attempt_ids": sorted(records),
            "attempt_file_sha256": attempt_hashes,
            "source_state_db": str(state_db),
            "source_attempts_dir": str(attempts_output),
        },
    )
    report = {
        "schema_version": "core50_release_api_attempt_ledger.v1",
        "status": "ok",
        "attempt_count": len(records),
        "accepted_count": accepted,
        "failed_count": len(records) - accepted,
        "estimated_cost_cny": estimated_cost_cny,
        "channels": {"routify": len(records)},
        "requested_model": CONTRACT_CANONICAL_MODEL,
        "requested_effort": CONTRACT_EFFORT,
        "stream": False,
        "state_db": str(state_db),
        "manifest_json": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }
    _atomic_write_json(root / "ledger_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="只校验配置与计划，不调用 API")
    preflight.add_argument("--phase", required=True)
    preflight.add_argument("--plan-jsonl", type=Path, required=True)
    preflight.add_argument("--profile", type=Path, default=DEFAULT_ROUTIFY_PROFILE)
    preflight.add_argument("--logical-total", type=int, required=True)
    preflight.add_argument("--workers", type=int, choices=range(1, 33), default=32)
    preflight.add_argument("--cache-import-total", type=int, default=0)

    run = subparsers.add_parser("run", help="执行一个阶段；所有阶段共享状态库")
    run.add_argument("--phase", required=True)
    run.add_argument("--plan-jsonl", type=Path, required=True)
    run.add_argument("--profile", type=Path, default=DEFAULT_ROUTIFY_PROFILE)
    run.add_argument("--logical-total", type=int, required=True)
    run.add_argument("--workers", type=int, choices=range(1, 33), default=32)
    run.add_argument("--limit", type=int)
    run.add_argument("--condition", choices=("clean", "noise001", "noise005"))
    run.add_argument("--cache-import-total", type=int, default=0)
    run.add_argument("--predecessor-attempt-manifest", type=Path)

    bootstrap = subparsers.add_parser(
        "bootstrap-api",
        help="复核并导入已有 API frozen；不发送网络请求",
    )
    bootstrap.add_argument("--phase", required=True)
    bootstrap.add_argument("--plan-jsonl", type=Path, required=True)
    bootstrap.add_argument("--source-frozen-dir", type=Path, required=True)
    bootstrap.add_argument("--source-attempts-dir", type=Path, required=True)
    bootstrap.add_argument("--logical-total", type=int, required=True)
    bootstrap.add_argument("--cache-import-total", type=int, required=True)
    bootstrap.add_argument("--predecessor-attempt-manifest", type=Path)

    mixed = subparsers.add_parser(
        "bootstrap-mixed",
        help="严格审计并导入 CLI/API 混合 frozen index；不发送网络请求",
    )
    mixed.add_argument(
        "--index-plan",
        action="append",
        required=True,
        help="重复提供 index.jsonl=plan.jsonl",
    )
    mixed.add_argument("--source-state-db", type=Path, action="append", required=True)
    mixed.add_argument("--logical-total", type=int, required=True)
    mixed.add_argument("--cache-import-total", type=int, required=True)
    mixed.add_argument("--predecessor-attempt-manifest", type=Path, required=True)
    mixed.add_argument("--workers", type=int, choices=range(1, 33), default=16)

    index = subparsers.add_parser("index", help="导出 frozen 索引和物理 attempt 清单")
    index.add_argument("--state-db", type=Path)
    index.add_argument("--attempts-dir", type=Path)
    index.add_argument("--output-jsonl", type=Path)
    index.add_argument("--predecessor-manifest", type=Path)
    index.add_argument("--plan-jsonl", type=Path, action="append", default=[])

    ledger = subparsers.add_parser("ledger", help="汇总真实 Routify API attempts")
    ledger.add_argument("--source-attempts-dir", type=Path, action="append", required=True)
    ledger.add_argument("--output-root", type=Path, required=True)

    subparsers.add_parser("status", help="汇总状态与人民币估算用量，不读取凭据")
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    paths = ReleasePaths.from_root(args.work_root, args.phase)
    plan_path = args.plan_jsonl.expanduser().resolve()
    if not _is_relative_to(plan_path, _release_scope(paths.root)):
        raise ReleaseControlError("plan 必须位于本次 release_v2 工作区内")
    loaded_plan = load_plan_jsonl(plan_path)
    if len(loaded_plan.entries) > args.logical_total:
        raise ReleaseControlError("当前 plan 的任务数超过本次发布逻辑任务总量")
    _, safe_channel_metadata = load_release_channel(args.profile)
    return _safe_control_payload(
        phase=args.phase,
        plan_path=plan_path,
        plan_count=len(loaded_plan.entries),
        paths=paths,
        logical_total=args.logical_total,
        cache_import_total=args.cache_import_total,
        workers=args.workers,
        limit=None,
        selected_condition=None,
        selected_task_count=len(loaded_plan.entries),
        channel_metadata=safe_channel_metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "status":
            print(json.dumps(release_status(args.work_root), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "index":
            work_root = args.work_root.expanduser().resolve()
            state_db = args.state_db or work_root / "state.sqlite3"
            attempts_dir = args.attempts_dir or work_root / "attempts"
            frozen_index = export_frozen_index(
                state_db=state_db,
                output_jsonl=args.output_jsonl or work_root / "frozen_index.jsonl",
                plan_jsonls=args.plan_jsonl,
            )
            predecessor = export_predecessor_manifest(
                state_db=state_db,
                attempts_dir=attempts_dir,
                output_json=args.predecessor_manifest
                or work_root / "predecessor_attempts.json",
            )
            print(
                json.dumps(
                    {"frozen_index": frozen_index, "predecessor_manifest": predecessor},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "bootstrap-api":
            report = bootstrap_api_frozen(
                work_root=args.work_root,
                phase=args.phase,
                plan_jsonl=args.plan_jsonl,
                source_frozen_dir=args.source_frozen_dir,
                source_attempts_dir=args.source_attempts_dir,
                logical_total=args.logical_total,
                cache_import_total=args.cache_import_total,
                predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "bootstrap-mixed":
            pairs: list[tuple[str, str]] = []
            for spec in args.index_plan:
                index_path, separator, plan_path = spec.partition("=")
                if not separator or not index_path or not plan_path:
                    raise ReleaseControlError("--index-plan 必须采用 index.jsonl=plan.jsonl")
                pairs.append((index_path, plan_path))
            report = bootstrap_mixed_frozen_indexes(
                work_root=args.work_root,
                index_plan_pairs=pairs,
                source_state_dbs=args.source_state_db,
                logical_total=args.logical_total,
                cache_import_total=args.cache_import_total,
                predecessor_attempt_manifest=args.predecessor_attempt_manifest,
                workers=args.workers,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "ledger":
            report = export_api_attempt_ledger(
                source_attempt_dirs=args.source_attempts_dir,
                output_root=args.output_root,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "preflight":
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True))
            return 0
        return run_phase(
            work_root=args.work_root,
            phase=args.phase,
            plan_jsonl=args.plan_jsonl,
            profile=args.profile,
            logical_total=args.logical_total,
            workers=args.workers,
            limit=args.limit,
            cache_import_total=args.cache_import_total,
            predecessor_attempt_manifest=args.predecessor_attempt_manifest,
            condition=args.condition,
        )
    except (FrozenPlanImportError, ReleaseControlError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
