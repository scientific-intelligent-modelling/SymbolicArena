#!/usr/bin/env python3
"""构建 clean replacement 对应的 equivalence/structure 增量计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import canonical_json, evaluation_key, render_prompt
from .run_claude_plan import PlanContractError, load_plan_jsonl
from .state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskSpec,
    TaskStateStore,
    TaskSupersession,
)


JsonDict = dict[str, Any]
SCHEMA_VERSION = "clean_downstream_hybrid_plan.v1"
_VERSION_SUFFIX = re.compile(r"::v([1-9]\d*)$")
_EQUIVALENCE_ID = re.compile(
    r"^equivalence::[a-z0-9_]+::g\d{4}::s(?:520|521|522)::"
    r"(?P<condition>clean|noise001|noise005)$"
)
_STRUCTURE_ID = re.compile(
    r"^stab_structure::[a-z0-9_]+::g\d{4}::s(?P<seed_a>520|521|522)-"
    r"s(?P<seed_b>520|521|522)(?:::(?P<condition>noise001|noise005))?$"
)
_PRED_ID = re.compile(
    r"^pred_simplify::[a-z0-9_]+::g\d{4}::s(?:520|521|522)::"
    r"(?P<condition>clean|noise001|noise005)$"
)
_STAGE_ROOT = Path(__file__).resolve().parents[1]
_KNOWN_WRONG_PRODUCTION_DB = (_STAGE_ROOT / "state_v2.sqlite3").resolve()


class CleanDownstreamHybridPlanError(RuntimeError):
    """下游 active/fresh/dependency/state 证据不足。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json_object(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanDownstreamHybridPlanError(f"{label} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise CleanDownstreamHybridPlanError(f"{label} 顶层必须是 JSON object")
    return payload


def _require_list_of_strings(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise CleanDownstreamHybridPlanError(f"{label} 必须是非空字符串列表")
    if len(set(value)) != len(value):
        raise CleanDownstreamHybridPlanError(f"{label} 存在重复值")
    return list(value)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CleanDownstreamHybridPlanError(f"{label} 必须是小写十六进制 SHA256")
    return value


def _base_logical_id(logical_id: object) -> str:
    return _VERSION_SUFFIX.sub("", str(logical_id))


def _next_logical_id(logical_id: object) -> str:
    text = str(logical_id)
    base = _base_logical_id(text)
    match = _VERSION_SUFFIX.search(text)
    next_version = 2 if match is None else int(match.group(1)) + 1
    return f"{base}::v{next_version}"


def _required_condition(
    *, pred: set[str], equivalence: set[str], structure: set[str]
) -> str:
    conditions: set[str] = set()
    for logical_id in pred:
        match = _PRED_ID.fullmatch(logical_id)
        if match is None:
            raise CleanDownstreamHybridPlanError(
                "required pred logical id 非 canonical"
            )
        conditions.add(match.group("condition"))
    for logical_id in equivalence:
        match = _EQUIVALENCE_ID.fullmatch(logical_id)
        if match is None:
            raise CleanDownstreamHybridPlanError(
                "required equivalence logical id 非 canonical"
            )
        conditions.add(match.group("condition"))
    for logical_id in structure:
        match = _STRUCTURE_ID.fullmatch(logical_id)
        if match is None or int(match.group("seed_a")) >= int(match.group("seed_b")):
            raise CleanDownstreamHybridPlanError(
                "required structure logical id 的 seed pair 非 canonical"
            )
        conditions.add(match.group("condition") or "clean")
    if len(conditions) != 1:
        raise CleanDownstreamHybridPlanError(
            f"required logical ids 的 condition 不唯一: {sorted(conditions)}"
        )
    return next(iter(conditions))


def _read_required_sets(
    path: Path,
    *,
    expected_equivalence_count: int | None,
    expected_structure_count: int | None,
) -> tuple[set[str], set[str], set[str], str]:
    payload = _read_json_object(path, label="binding manifest")
    readiness = payload.get("aggregation_readiness")
    if not isinstance(readiness, Mapping):
        raise CleanDownstreamHybridPlanError(
            "binding manifest 缺少 aggregation_readiness"
        )
    pred = set(
        _require_list_of_strings(
            readiness.get("required_pred_simplify_logical_ids"),
            label="required pred logical ids",
        )
    )
    equivalence = set(
        _require_list_of_strings(
            readiness.get("required_equivalence_logical_ids"),
            label="required equivalence logical ids",
        )
    )
    structure = set(
        _require_list_of_strings(
            readiness.get("required_structure_logical_ids"),
            label="required structure logical ids",
        )
    )
    condition = _required_condition(
        pred=pred, equivalence=equivalence, structure=structure
    )
    manifest_structure_count = readiness.get("required_structure_count")
    if manifest_structure_count is not None and manifest_structure_count != len(structure):
        raise CleanDownstreamHybridPlanError(
            "binding manifest required_structure_count 与列表不一致"
        )
    if expected_equivalence_count is not None and len(equivalence) != expected_equivalence_count:
        raise CleanDownstreamHybridPlanError(
            f"required equivalence 数量不符: {len(equivalence)} != {expected_equivalence_count}"
        )
    if expected_structure_count is not None and len(structure) != expected_structure_count:
        raise CleanDownstreamHybridPlanError(
            f"required structure 数量不符: {len(structure)} != {expected_structure_count}"
        )
    return pred, equivalence, structure, condition


def _load_pred_supersessions(
    path: Path,
    *,
    required_pred: set[str],
) -> tuple[dict[str, str], JsonDict]:
    payload = _read_json_object(path, label="pred supersession manifest")
    if payload.get("status") != "ok":
        raise CleanDownstreamHybridPlanError("pred supersession manifest status 必须为 ok")
    _require_sha256(
        payload.get("hybrid_plan_sha256"),
        label="pred supersession manifest.hybrid_plan_sha256",
    )
    raw_rows = payload.get("supersessions")
    if not isinstance(raw_rows, list):
        raise CleanDownstreamHybridPlanError(
            "pred supersession manifest.supersessions 必须是列表"
        )
    declared_count = payload.get("counts")
    if not isinstance(declared_count, Mapping) or declared_count.get("supersessions") != len(raw_rows):
        raise CleanDownstreamHybridPlanError("pred supersession manifest 计数漂移")
    mapping: dict[str, str] = {}
    seen_new: set[str] = set()
    seen_bases: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise CleanDownstreamHybridPlanError(
                f"pred supersession[{index}] 必须是 JSON object"
            )
        base = str(raw.get("base_logical_id", ""))
        old_key = str(raw.get("predecessor_evaluation_key", ""))
        new_key = str(raw.get("successor_evaluation_key", ""))
        old_logical_id = str(raw.get("predecessor_logical_id", ""))
        new_logical_id = str(raw.get("successor_logical_id", ""))
        if (
            base not in required_pred
            or _base_logical_id(old_logical_id) != base
            or new_logical_id != _next_logical_id(old_logical_id)
        ):
            raise CleanDownstreamHybridPlanError(
                f"pred supersession[{index}] logical identity 漂移"
            )
        if not old_key or not new_key or old_key == new_key:
            raise CleanDownstreamHybridPlanError(
                f"pred supersession[{index}] evaluation key 非法"
            )
        if old_key in mapping or new_key in seen_new or base in seen_bases:
            raise CleanDownstreamHybridPlanError("pred supersession mapping 不唯一")
        mapping[old_key] = new_key
        seen_new.add(new_key)
        seen_bases.add(base)
    if seen_bases != required_pred:
        raise CleanDownstreamHybridPlanError(
            "pred supersession manifest 与 required pred 集合不闭合"
        )
    return mapping, payload


def _compose_pred_retirements(
    path: Path,
    *,
    required_pred: set[str],
    pred_mapping: dict[str, str],
) -> int:
    """把 exhausted replacement 的后继链合并为原始 pred 到当前 active pred 的映射。"""

    active_source_by_key = {value: key for key, value in pred_mapping.items()}
    if len(active_source_by_key) != len(pred_mapping):
        raise CleanDownstreamHybridPlanError("pred supersession active key 不唯一")
    seen_exhausted: set[str] = set()
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if not isinstance(row, Mapping):
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} 顶层不是 JSON object"
                    )
                if row.get("schema_version") != "stale_exhausted_retirement.v1":
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} schema_version 非法"
                    )
                base = str(row.get("revision_base", ""))
                exhausted_key = str(row.get("exhausted_evaluation_key", ""))
                replacement_key = str(row.get("replacement_evaluation_key", ""))
                replacement_logical_id = str(row.get("replacement_logical_id", ""))
                match = re.fullmatch(re.escape(base) + r"::v([1-9]\d*)", replacement_logical_id)
                if base not in required_pred or match is None or int(match.group(1)) < 2:
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} logical identity 漂移"
                    )
                source_key = active_source_by_key.get(exhausted_key)
                if source_key is None or exhausted_key in seen_exhausted:
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} exhausted binding 不在 active 链"
                    )
                if not replacement_key or replacement_key == exhausted_key:
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} replacement key 非法"
                    )
                if replacement_key in active_source_by_key:
                    raise CleanDownstreamHybridPlanError(
                        f"pred retirement manifest:{line_number} replacement key 重复"
                    )
                pred_mapping[source_key] = replacement_key
                del active_source_by_key[exhausted_key]
                active_source_by_key[replacement_key] = source_key
                seen_exhausted.add(exhausted_key)
                row_count += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanDownstreamHybridPlanError(
            f"pred retirement manifest 不可读: {exc}"
        ) from exc
    if row_count == 0:
        raise CleanDownstreamHybridPlanError("pred retirement manifest 不能为空")
    return row_count


def _load_plan_rows(
    path: Path,
    *,
    label: str,
    expected_task_type: str,
    expected_condition: str,
    expected_total_count: int | None,
) -> tuple[list[tuple[str, JsonDict]], str]:
    try:
        loaded = load_plan_jsonl(path)
    except PlanContractError as exc:
        raise CleanDownstreamHybridPlanError(f"{label} 契约失败: {exc}") from exc
    if expected_total_count is not None and len(loaded.entries) != expected_total_count:
        raise CleanDownstreamHybridPlanError(
            f"{label} 数量不符: {len(loaded.entries)} != {expected_total_count}"
        )
    expected_keys = {entry.evaluation_key for entry in loaded.entries}
    rows: list[tuple[str, JsonDict]] = []
    seen_keys: set[str] = set()
    seen_bases: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise CleanDownstreamHybridPlanError(
                        f"{label}:{line_number} 必须是 JSON object"
                    )
                if payload.get("task_type") != expected_task_type:
                    raise CleanDownstreamHybridPlanError(
                        f"{label}:{line_number}.task_type 漂移"
                    )
                if payload.get("condition") != expected_condition:
                    raise CleanDownstreamHybridPlanError(
                        f"{label}:{line_number}.condition 必须为 {expected_condition}"
                    )
                evaluation = str(payload.get("evaluation_key", ""))
                logical_id = str(payload.get("logical_id", ""))
                base = _base_logical_id(logical_id)
                if evaluation in seen_keys or base in seen_bases:
                    raise CleanDownstreamHybridPlanError(
                        f"{label} evaluation/base logical identity 不唯一"
                    )
                seen_keys.add(evaluation)
                seen_bases.add(base)
                rows.append(
                    (
                        raw_line if raw_line.endswith("\n") else raw_line + "\n",
                        payload,
                    )
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanDownstreamHybridPlanError(f"{label} 回读失败: {exc}") from exc
    if seen_keys != expected_keys or len(rows) != len(loaded.entries):
        raise CleanDownstreamHybridPlanError(f"{label} 回读 identity 漂移")
    return rows, loaded.plan_sha256


def _task_spec(row: Mapping[str, Any]) -> TaskSpec:
    payload = row.get("task_spec")
    if not isinstance(payload, Mapping):
        raise CleanDownstreamHybridPlanError(
            f"任务 {row.get('logical_id')} 缺少 task_spec"
        )
    return TaskSpec(
        evaluation_key=str(payload["evaluation_key"]),
        logical_id=str(payload["logical_id"]),
        task_type=str(payload["task_type"]),
        condition=str(payload["condition"]),
        priority=int(payload["priority"]),
        input_hash=str(payload["input_hash"]),
        prompt_version=str(payload["prompt_version"]),
        schema_version=str(payload["schema_version"]),
        dependencies=tuple(str(item) for item in payload["dependencies"]),
    )


def _successor_row(
    predecessor: Mapping[str, Any],
    fresh: Mapping[str, Any],
    *,
    expected_task_type: str,
) -> JsonDict:
    base = _base_logical_id(predecessor.get("logical_id"))
    if str(fresh.get("logical_id")) != base:
        raise CleanDownstreamHybridPlanError(
            f"fresh row 不是未版本化 base identity: {base}"
        )
    if fresh.get("status") == "planned_non_applicable":
        raise CleanDownstreamHybridPlanError(
            f"required fresh task 不能是 planned_non_applicable: {base}"
        )
    for field in ("task_type", "condition", "priority"):
        if predecessor.get(field) != fresh.get(field):
            raise CleanDownstreamHybridPlanError(f"{base} fresh.{field} 漂移")
    if fresh.get("task_type") != expected_task_type:
        raise CleanDownstreamHybridPlanError(f"{base} task_type 漂移")
    request = fresh.get("request")
    normalized = fresh.get("normalized_input")
    schema = fresh.get("schema_content")
    if not isinstance(request, Mapping) or not isinstance(normalized, Mapping) or not isinstance(schema, Mapping):
        raise CleanDownstreamHybridPlanError(f"{base} fresh request/normalized/schema 缺失")
    expected_normalized = {
        "request": dict(request),
        "prompt_sha256": str(fresh.get("prompt_sha256", "")),
        "schema_sha256": str(fresh.get("schema_sha256", "")),
    }
    if dict(normalized) != expected_normalized:
        raise CleanDownstreamHybridPlanError(f"{base} fresh normalized_input 漂移")
    input_hash = _sha256_bytes(canonical_json(expected_normalized).encode("utf-8"))
    if input_hash != fresh.get("input_hash"):
        raise CleanDownstreamHybridPlanError(f"{base} fresh input_hash 漂移")
    evidence_hash = _require_sha256(
        request.get("evidence_hash"), label=f"{base}.request.evidence_hash"
    )
    dependencies = tuple(str(item) for item in fresh.get("dependencies", []))
    logical_id = _next_logical_id(predecessor.get("logical_id"))
    task_key = evaluation_key(
        task_type=expected_task_type,
        logical_id=logical_id,
        prompt_version=str(fresh.get("prompt_version", "")),
        schema_version=str(fresh.get("schema_version", "")),
        prompt_sha256=str(fresh.get("prompt_sha256", "")),
        schema_sha256=str(fresh.get("schema_sha256", "")),
        normalized_input=expected_normalized,
        evidence_hash=evidence_hash,
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=expected_task_type,
        condition=str(fresh["condition"]),
        priority=int(fresh["priority"]),
        input_hash=input_hash,
        prompt_version=str(fresh["prompt_version"]),
        schema_version=str(fresh["schema_version"]),
        dependencies=dependencies,
    )
    successor = dict(fresh)
    successor.update(
        {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "input_hash": input_hash,
            "normalized_input": expected_normalized,
            "task_spec": json.loads(spec.canonical_json()),
            "rendered_prompt": render_prompt(
                str(fresh.get("prompt_template", "")), request, schema
            ),
        }
    )
    return successor


def _validate_dependency_change(
    *,
    base: str,
    task_type: str,
    predecessor: Mapping[str, Any],
    fresh: Mapping[str, Any],
    pred_mapping: Mapping[str, str],
) -> int:
    old_dependencies = tuple(str(item) for item in predecessor.get("dependencies", []))
    new_dependencies = tuple(str(item) for item in fresh.get("dependencies", []))
    if len(old_dependencies) != 2 or len(new_dependencies) != 2:
        raise CleanDownstreamHybridPlanError(
            f"{base} dependency 长度必须保持为 2"
        )
    changed_positions: list[int] = []
    for index, (old, new) in enumerate(
        zip(old_dependencies, new_dependencies, strict=True)
    ):
        if old == new:
            continue
        if pred_mapping.get(old) != new:
            raise CleanDownstreamHybridPlanError(
                f"{base} dependency[{index}] 错绑或顺序漂移: {old}->{new}"
            )
        changed_positions.append(index)
    if task_type == "equivalence" and changed_positions != [1]:
        raise CleanDownstreamHybridPlanError(
            f"{base} equivalence dependency 必须只改变位置 1"
        )
    if task_type == "stab_structure" and len(changed_positions) not in {1, 2}:
        raise CleanDownstreamHybridPlanError(
            f"{base} structure dependency 必须改变 1 或 2 个位置"
        )
    return len(changed_positions)


def _build_phase(
    *,
    predecessor_rows: list[tuple[str, JsonDict]],
    fresh_rows: list[tuple[str, JsonDict]],
    required: set[str],
    task_type: str,
    pred_mapping: Mapping[str, str],
) -> tuple[list[str], list[JsonDict], list[JsonDict], dict[int, int]]:
    fresh_by_base = {
        _base_logical_id(row["logical_id"]): row for _, row in fresh_rows
    }
    predecessor_bases = {
        _base_logical_id(row["logical_id"]) for _, row in predecessor_rows
    }
    if set(fresh_by_base) != predecessor_bases:
        raise CleanDownstreamHybridPlanError(
            f"{task_type} predecessor/fresh base identity 集合不一致"
        )
    if not required.issubset(predecessor_bases):
        raise CleanDownstreamHybridPlanError(
            f"{task_type} predecessor 缺少 required identity"
        )
    existing_logical_ids = {str(row["logical_id"]) for _, row in predecessor_rows}
    output_lines: list[str] = []
    output_rows: list[JsonDict] = []
    bindings: list[JsonDict] = []
    dependency_profile: dict[int, int] = {1: 0, 2: 0}
    for raw_line, predecessor in predecessor_rows:
        base = _base_logical_id(predecessor["logical_id"])
        if base not in required:
            output_lines.append(raw_line)
            output_rows.append(predecessor)
            continue
        successor_logical_id = _next_logical_id(predecessor["logical_id"])
        if successor_logical_id in existing_logical_ids:
            raise CleanDownstreamHybridPlanError(
                f"{task_type} successor 版本冲突: {successor_logical_id}"
            )
        fresh = fresh_by_base[base]
        changed_count = _validate_dependency_change(
            base=base,
            task_type=task_type,
            predecessor=predecessor,
            fresh=fresh,
            pred_mapping=pred_mapping,
        )
        dependency_profile[changed_count] += 1
        successor = _successor_row(
            predecessor, fresh, expected_task_type=task_type
        )
        output_lines.append(canonical_json(successor) + "\n")
        output_rows.append(successor)
        condition = str(successor["condition"])
        identity_prefix = (
            "clean_downstream_final_replacement"
            if condition == "clean"
            else f"{condition}_downstream_final_replacement"
        )
        bindings.append(
            {
                "base_logical_id": base,
                "task_type": task_type,
                "predecessor_evaluation_key": predecessor["evaluation_key"],
                "predecessor_logical_id": predecessor["logical_id"],
                "successor_evaluation_key": successor["evaluation_key"],
                "successor_logical_id": successor["logical_id"],
                "changed_dependency_count": changed_count,
                "old_dependencies": list(predecessor["dependencies"]),
                "new_dependencies": list(successor["dependencies"]),
                "identity": f"{identity_prefix}::{task_type}::{base}",
                "reason": (
                    f"{condition} pred replacement changed downstream evidence "
                    "and dependency"
                ),
                "successor_spec": _task_spec(successor),
            }
        )
    if len(bindings) != len(required):
        raise CleanDownstreamHybridPlanError(
            f"{task_type} changed 集合与 required 集合不一致"
        )
    logical_ids = [str(row["logical_id"]) for row in output_rows]
    evaluation_keys = [str(row["evaluation_key"]) for row in output_rows]
    if len(set(logical_ids)) != len(output_rows) or len(set(evaluation_keys)) != len(output_rows):
        raise CleanDownstreamHybridPlanError(
            f"{task_type} hybrid identity 不唯一"
        )
    return output_lines, output_rows, bindings, dependency_profile


def _validate_apply_state_db_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == _KNOWN_WRONG_PRODUCTION_DB:
        raise CleanDownstreamHybridPlanError(
            f"错误的生产 state DB 路径: {resolved}; 必须使用 llm/control/state_v2.sqlite3"
        )
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise CleanDownstreamHybridPlanError(f"apply state DB 不存在或为空: {resolved}")
    return resolved


def _state_store_from_meta(path: Path) -> TaskStateStore:
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error as exc:
        raise CleanDownstreamHybridPlanError(f"state DB meta 不可读: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    required = {"attempt_cap", "logical_task_cap", "max_attempts_per_task"}
    if not required.issubset(meta):
        raise CleanDownstreamHybridPlanError("state DB meta 缺少容量契约")
    predecessor_count = int(meta.get("predecessor_attempt_count", "0"))
    predecessor_manifest = None
    if predecessor_count:
        try:
            predecessor_manifest = PredecessorAttemptManifest(
                path=meta["predecessor_attempt_manifest_path"],
                sha256=meta["predecessor_attempt_manifest_sha256"],
                attempt_count=predecessor_count,
            )
        except KeyError as exc:
            raise CleanDownstreamHybridPlanError(
                "state DB meta 缺少 predecessor attempt manifest"
            ) from exc
    try:
        return TaskStateStore(
            path,
            attempt_cap=int(meta["attempt_cap"]),
            logical_task_cap=int(meta["logical_task_cap"]),
            max_attempts_per_task=int(meta["max_attempts_per_task"]),
            predecessor_attempt_manifest=predecessor_manifest,
        )
    except (ValueError, StateContractError, sqlite3.Error) as exc:
        raise CleanDownstreamHybridPlanError(f"state DB 契约不一致: {exc}") from exc


def _existing_mapping_count(path: Path, identities: Sequence[str]) -> int:
    if not identities:
        return 0
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        count = 0
        for start in range(0, len(identities), 500):
            chunk = identities[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            count += int(
                connection.execute(
                    f"SELECT COUNT(*) FROM task_supersessions WHERE identity IN ({placeholders})",
                    tuple(chunk),
                ).fetchone()[0]
            )
        return count
    finally:
        connection.close()


def _register_state(
    path: Path,
    *,
    bindings: Sequence[Mapping[str, Any]],
    equivalence_predecessor_sha: str,
    equivalence_successor_sha: str,
    structure_predecessor_sha: str,
    structure_successor_sha: str,
) -> JsonDict:
    identities = [str(binding["identity"]) for binding in bindings]
    before_count = _existing_mapping_count(path, identities)
    supersessions: list[TaskSupersession] = []
    for binding in bindings:
        is_equivalence = binding["task_type"] == "equivalence"
        supersessions.append(
            TaskSupersession(
                predecessor_evaluation_key=str(
                    binding["predecessor_evaluation_key"]
                ),
                successor=binding["successor_spec"],
                identity=str(binding["identity"]),
                reason=str(binding["reason"]),
                predecessor_plan_sha256=(
                    equivalence_predecessor_sha
                    if is_equivalence
                    else structure_predecessor_sha
                ),
                successor_plan_sha256=(
                    equivalence_successor_sha
                    if is_equivalence
                    else structure_successor_sha
                ),
            )
        )
    try:
        store = _state_store_from_meta(path)
        store.register_dependency_rebinding_supersession_batch(tuple(supersessions))
    except (StateContractError, sqlite3.Error) as exc:
        raise CleanDownstreamHybridPlanError(
            f"dependency rebinding state 注册失败: {exc}"
        ) from exc
    after_count = _existing_mapping_count(path, identities)
    if after_count != len(bindings):
        raise CleanDownstreamHybridPlanError(
            "dependency rebinding 注册后 supersession 数量不闭合"
        )
    return {
        "requested": True,
        "mutated": before_count != after_count,
        "existing_before": before_count,
        "registered_after": after_count,
        "state_db": str(path),
    }


def build_clean_downstream_hybrid_plan(
    *,
    predecessor_equivalence_plan_jsonl: str | Path,
    fresh_equivalence_plan_jsonl: str | Path,
    predecessor_structure_plan_jsonl: str | Path,
    fresh_structure_plan_jsonl: str | Path,
    binding_manifest_json: str | Path,
    pred_supersession_manifest_json: str | Path,
    output_equivalence_plan_jsonl: str | Path,
    output_structure_plan_jsonl: str | Path,
    supersession_manifest_json: str | Path,
    report_json: str | Path,
    pred_retirement_manifest_jsonl: str | Path | None = None,
    expected_total_count: int | None = 2250,
    expected_equivalence_count: int | None = 75,
    expected_structure_count: int | None = 87,
    expected_structure_single_dependency_count: int | None = 24,
    expected_structure_double_dependency_count: int | None = 63,
    apply_state_db: str | Path | None = None,
) -> JsonDict:
    paths = {
        "old_eq": Path(predecessor_equivalence_plan_jsonl).resolve(),
        "fresh_eq": Path(fresh_equivalence_plan_jsonl).resolve(),
        "old_structure": Path(predecessor_structure_plan_jsonl).resolve(),
        "fresh_structure": Path(fresh_structure_plan_jsonl).resolve(),
        "binding": Path(binding_manifest_json).resolve(),
        "pred_manifest": Path(pred_supersession_manifest_json).resolve(),
        "output_eq": Path(output_equivalence_plan_jsonl).resolve(),
        "output_structure": Path(output_structure_plan_jsonl).resolve(),
        "supersession_manifest": Path(supersession_manifest_json).resolve(),
        "report": Path(report_json).resolve(),
    }
    if pred_retirement_manifest_jsonl is not None:
        paths["pred_retirement_manifest"] = Path(
            pred_retirement_manifest_jsonl
        ).resolve()
    output_paths = {
        paths["output_eq"],
        paths["output_structure"],
        paths["supersession_manifest"],
        paths["report"],
    }
    if len(output_paths) != 4 or output_paths.intersection(
        {
            paths[name]
            for name in (
                "old_eq",
                "fresh_eq",
                "old_structure",
                "fresh_structure",
                "binding",
                "pred_manifest",
                "pred_retirement_manifest",
            )
            if name in paths
        }
    ):
        raise CleanDownstreamHybridPlanError("输入输出路径必须互异且禁止覆盖输入")

    required_pred, required_eq, required_structure, condition = _read_required_sets(
        paths["binding"],
        expected_equivalence_count=expected_equivalence_count,
        expected_structure_count=expected_structure_count,
    )
    pred_mapping, _ = _load_pred_supersessions(
        paths["pred_manifest"], required_pred=required_pred
    )
    pred_retirement_count = 0
    if "pred_retirement_manifest" in paths:
        pred_retirement_count = _compose_pred_retirements(
            paths["pred_retirement_manifest"],
            required_pred=required_pred,
            pred_mapping=pred_mapping,
        )
    old_eq, old_eq_sha = _load_plan_rows(
        paths["old_eq"],
        label="predecessor equivalence plan",
        expected_task_type="equivalence",
        expected_condition=condition,
        expected_total_count=expected_total_count,
    )
    fresh_eq, fresh_eq_sha = _load_plan_rows(
        paths["fresh_eq"],
        label="fresh equivalence plan",
        expected_task_type="equivalence",
        expected_condition=condition,
        expected_total_count=expected_total_count,
    )
    old_structure, old_structure_sha = _load_plan_rows(
        paths["old_structure"],
        label="predecessor structure plan",
        expected_task_type="stab_structure",
        expected_condition=condition,
        expected_total_count=expected_total_count,
    )
    fresh_structure, fresh_structure_sha = _load_plan_rows(
        paths["fresh_structure"],
        label="fresh structure plan",
        expected_task_type="stab_structure",
        expected_condition=condition,
        expected_total_count=expected_total_count,
    )
    eq_lines, eq_rows, eq_bindings, _ = _build_phase(
        predecessor_rows=old_eq,
        fresh_rows=fresh_eq,
        required=required_eq,
        task_type="equivalence",
        pred_mapping=pred_mapping,
    )
    structure_lines, structure_rows, structure_bindings, structure_profile = _build_phase(
        predecessor_rows=old_structure,
        fresh_rows=fresh_structure,
        required=required_structure,
        task_type="stab_structure",
        pred_mapping=pred_mapping,
    )
    if (
        expected_structure_single_dependency_count is not None
        and structure_profile[1] != expected_structure_single_dependency_count
    ):
        raise CleanDownstreamHybridPlanError(
            "structure 单依赖替换数量不符: "
            f"{structure_profile[1]} != {expected_structure_single_dependency_count}"
        )
    if (
        expected_structure_double_dependency_count is not None
        and structure_profile[2] != expected_structure_double_dependency_count
    ):
        raise CleanDownstreamHybridPlanError(
            "structure 双依赖替换数量不符: "
            f"{structure_profile[2]} != {expected_structure_double_dependency_count}"
        )

    _atomic_write(paths["output_eq"], "".join(eq_lines))
    _atomic_write(paths["output_structure"], "".join(structure_lines))
    try:
        verified_eq = load_plan_jsonl(paths["output_eq"])
        verified_structure = load_plan_jsonl(paths["output_structure"])
    except PlanContractError as exc:
        raise CleanDownstreamHybridPlanError(f"hybrid plan 回读失败: {exc}") from exc
    if len(verified_eq.entries) != len(eq_rows) or len(verified_structure.entries) != len(structure_rows):
        raise CleanDownstreamHybridPlanError("hybrid plan 回读数量漂移")
    eq_output_sha = verified_eq.plan_sha256
    structure_output_sha = verified_structure.plan_sha256
    bindings = [*eq_bindings, *structure_bindings]

    public_bindings = [
        {key: value for key, value in binding.items() if key != "successor_spec"}
        for binding in bindings
    ]
    supersession_manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "condition": condition,
        "counts": {
            "equivalence": len(eq_bindings),
            "structure": len(structure_bindings),
            "supersessions": len(bindings),
        },
        "plan_sha256": {
            "predecessor_equivalence": old_eq_sha,
            "successor_equivalence": eq_output_sha,
            "predecessor_structure": old_structure_sha,
            "successor_structure": structure_output_sha,
        },
        "pred_supersession_manifest_sha256": _sha256_file(paths["pred_manifest"]),
        "pred_retirement_manifest_sha256": (
            _sha256_file(paths["pred_retirement_manifest"])
            if "pred_retirement_manifest" in paths
            else None
        ),
        "binding_manifest_sha256": _sha256_file(paths["binding"]),
        "supersessions": public_bindings,
    }
    _atomic_write_json(paths["supersession_manifest"], supersession_manifest)

    input_hashes_before_apply = {
        name: _sha256_file(path)
        for name, path in paths.items()
        if name in {"old_eq", "fresh_eq", "old_structure", "fresh_structure", "binding", "pred_manifest", "pred_retirement_manifest", "output_eq", "output_structure", "supersession_manifest"}
    }
    state_registration: JsonDict = {"requested": False, "mutated": False}
    if apply_state_db is not None:
        state_path = _validate_apply_state_db_path(Path(apply_state_db))
        state_registration = _register_state(
            state_path,
            bindings=bindings,
            equivalence_predecessor_sha=old_eq_sha,
            equivalence_successor_sha=eq_output_sha,
            structure_predecessor_sha=old_structure_sha,
            structure_successor_sha=structure_output_sha,
        )
        input_hashes_after_apply = {
            name: _sha256_file(paths[name]) for name in input_hashes_before_apply
        }
        if input_hashes_before_apply != input_hashes_after_apply:
            raise CleanDownstreamHybridPlanError(
                "state apply 前后输入或生成计划 SHA256 漂移"
            )

    counts = {
        "equivalence_total": len(eq_rows),
        "equivalence_changed": len(eq_bindings),
        "equivalence_preserved": len(eq_rows) - len(eq_bindings),
        "structure_total": len(structure_rows),
        "structure_changed": len(structure_bindings),
        "structure_preserved": len(structure_rows) - len(structure_bindings),
        "structure_single_dependency_changed": structure_profile[1],
        "structure_double_dependency_changed": structure_profile[2],
        "dependency_positions_changed": len(eq_bindings)
        + structure_profile[1]
        + (2 * structure_profile[2]),
        "supersessions": len(bindings),
    }
    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "condition": condition,
        "model_invoked": False,
        "counts": counts,
        "inputs": {
            "predecessor_equivalence_plan": {"path": str(paths["old_eq"]), "sha256": old_eq_sha},
            "fresh_equivalence_plan": {"path": str(paths["fresh_eq"]), "sha256": fresh_eq_sha},
            "predecessor_structure_plan": {"path": str(paths["old_structure"]), "sha256": old_structure_sha},
            "fresh_structure_plan": {"path": str(paths["fresh_structure"]), "sha256": fresh_structure_sha},
            "binding_manifest": {"path": str(paths["binding"]), "sha256": _sha256_file(paths["binding"])},
            "pred_supersession_manifest": {"path": str(paths["pred_manifest"]), "sha256": _sha256_file(paths["pred_manifest"])},
            "pred_retirement_manifest": (
                {
                    "path": str(paths["pred_retirement_manifest"]),
                    "sha256": _sha256_file(paths["pred_retirement_manifest"]),
                    "row_count": pred_retirement_count,
                }
                if "pred_retirement_manifest" in paths
                else None
            ),
        },
        "outputs": {
            "equivalence_hybrid_plan": {"path": str(paths["output_eq"]), "sha256": eq_output_sha},
            "structure_hybrid_plan": {"path": str(paths["output_structure"]), "sha256": structure_output_sha},
            "supersession_manifest": {"path": str(paths["supersession_manifest"]), "sha256": _sha256_file(paths["supersession_manifest"])},
        },
        "state_registration": state_registration,
    }
    _atomic_write_json(paths["report"], report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建并可选注册 clean equivalence/structure replacement hybrid plans"
    )
    parser.add_argument("--predecessor-equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--predecessor-structure-plan-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-structure-plan-jsonl", type=Path, required=True)
    parser.add_argument("--binding-manifest-json", type=Path, required=True)
    parser.add_argument("--pred-supersession-manifest-json", type=Path, required=True)
    parser.add_argument("--pred-retirement-manifest-jsonl", type=Path)
    parser.add_argument("--output-equivalence-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-structure-plan-jsonl", type=Path, required=True)
    parser.add_argument("--supersession-manifest-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-total-count", type=int, default=2250)
    parser.add_argument("--expected-equivalence-count", type=int, default=75)
    parser.add_argument("--expected-structure-count", type=int, default=87)
    parser.add_argument(
        "--expected-structure-single-dependency-count", type=int, default=24
    )
    parser.add_argument(
        "--expected-structure-double-dependency-count", type=int, default=63
    )
    parser.add_argument(
        "--apply-state-db",
        type=Path,
        help="缺省不改状态库；显式传入后原子注册 dependency rebinding supersessions",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_clean_downstream_hybrid_plan(
            predecessor_equivalence_plan_jsonl=args.predecessor_equivalence_plan_jsonl,
            fresh_equivalence_plan_jsonl=args.fresh_equivalence_plan_jsonl,
            predecessor_structure_plan_jsonl=args.predecessor_structure_plan_jsonl,
            fresh_structure_plan_jsonl=args.fresh_structure_plan_jsonl,
            binding_manifest_json=args.binding_manifest_json,
            pred_supersession_manifest_json=args.pred_supersession_manifest_json,
            pred_retirement_manifest_jsonl=args.pred_retirement_manifest_jsonl,
            output_equivalence_plan_jsonl=args.output_equivalence_plan_jsonl,
            output_structure_plan_jsonl=args.output_structure_plan_jsonl,
            supersession_manifest_json=args.supersession_manifest_json,
            report_json=args.report_json,
            expected_total_count=args.expected_total_count,
            expected_equivalence_count=args.expected_equivalence_count,
            expected_structure_count=args.expected_structure_count,
            expected_structure_single_dependency_count=args.expected_structure_single_dependency_count,
            expected_structure_double_dependency_count=args.expected_structure_double_dependency_count,
            apply_state_db=args.apply_state_db,
        )
    except CleanDownstreamHybridPlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
