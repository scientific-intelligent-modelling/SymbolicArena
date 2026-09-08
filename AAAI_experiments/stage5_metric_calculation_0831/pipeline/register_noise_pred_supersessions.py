#!/usr/bin/env python3
"""把噪声预测公式 replacement 作为可审计 supersession 原子写入状态库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_claude_plan import PlanContractError, load_plan_jsonl
from .state import (
    PredecessorAttemptManifest,
    StateContractError,
    TaskSpec,
    TaskStateStore,
    TaskSupersession,
)


JsonDict = dict[str, Any]
SCHEMA_VERSION = "noise_pred_supersession_registration.v1"
MANIFEST_SCHEMA_VERSION = "noise_pred_hybrid.v1"
FORBIDDEN_SOURCE = "all_15alg_fullcpu_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_SUFFIX = re.compile(r"::v([1-9]\d*)$")
_PRED_BASE = re.compile(
    r"^pred_simplify::(?P<algorithm>[a-z0-9_]+)::(?P<dataset>g\d{4})::"
    r"s(?P<seed>520|521|522)::(?P<condition>noise001|noise005)$"
)


class NoisePredSupersessionError(RuntimeError):
    """噪声 pred successor 或状态库证据不足以注册 supersession。"""


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
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NoisePredSupersessionError(f"{label} 必须是 JSON object")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NoisePredSupersessionError(f"{label} 必须是小写十六进制 SHA256")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NoisePredSupersessionError(f"{label} 必须是非负整数")
    return value


def _read_json_object(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NoisePredSupersessionError(f"{label} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise NoisePredSupersessionError(f"{label} 顶层必须是 JSON object")
    return payload


def _reject_forbidden_text_file(path: Path, *, label: str) -> None:
    if FORBIDDEN_SOURCE in str(path):
        raise NoisePredSupersessionError(f"{label} 命中禁止来源 {FORBIDDEN_SOURCE}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if FORBIDDEN_SOURCE in line:
                    raise NoisePredSupersessionError(
                        f"{label}:{line_number} 命中禁止来源 {FORBIDDEN_SOURCE}"
                    )
    except (OSError, UnicodeError) as exc:
        raise NoisePredSupersessionError(f"{label} 不可读: {exc}") from exc


def _base_logical_id(value: object) -> str:
    return _VERSION_SUFFIX.sub("", str(value))


def _next_logical_id(value: object) -> str:
    logical_id = str(value)
    match = _VERSION_SUFFIX.search(logical_id)
    version = 2 if match is None else int(match.group(1)) + 1
    return f"{_base_logical_id(logical_id)}::v{version}"


def _load_hybrid_plan(
    path: Path,
    *,
    condition: str,
    expected_total_count: int,
) -> tuple[dict[str, TaskSpec], str]:
    _reject_forbidden_text_file(path, label="pred hybrid plan")
    try:
        loaded = load_plan_jsonl(path)
    except PlanContractError as exc:
        raise NoisePredSupersessionError(f"pred hybrid plan 契约失败: {exc}") from exc
    if len(loaded.entries) != expected_total_count:
        raise NoisePredSupersessionError(
            f"pred hybrid plan 数量不符: {len(loaded.entries)} != {expected_total_count}"
        )
    by_evaluation_key: dict[str, TaskSpec] = {}
    seen_bases: set[str] = set()
    for entry in loaded.entries:
        spec = entry.definition.task_spec
        base = _base_logical_id(spec.logical_id)
        match = _PRED_BASE.fullmatch(base)
        if (
            spec.task_type != "pred_simplify"
            or spec.condition != condition
            or match is None
            or match.group("condition") != condition
        ):
            raise NoisePredSupersessionError(
                f"pred hybrid plan 任务 condition/type/logical identity 漂移: {spec.logical_id}"
            )
        if base in seen_bases:
            raise NoisePredSupersessionError(
                f"pred hybrid plan base logical identity 重复: {base}"
            )
        if FORBIDDEN_SOURCE in spec.canonical_json():
            raise NoisePredSupersessionError(
                f"pred hybrid plan 任务命中禁止来源 {FORBIDDEN_SOURCE}: {spec.logical_id}"
            )
        seen_bases.add(base)
        by_evaluation_key[spec.evaluation_key] = spec
    return by_evaluation_key, loaded.plan_sha256


def _load_predecessor_summary(
    path: Path,
    *,
    expected_total_count: int,
) -> tuple[str, Path, dict[str, int]]:
    _reject_forbidden_text_file(path, label="predecessor pred frozen summary")
    payload = _read_json_object(path, label="predecessor pred frozen summary")
    if payload.get("status") != "ok":
        raise NoisePredSupersessionError("predecessor pred frozen summary status 必须为 ok")
    plan_sha256 = _require_sha256(
        payload.get("plan_sha256"),
        label="predecessor pred frozen summary.plan_sha256",
    )
    plan_jsonl = payload.get("plan_jsonl")
    if not isinstance(plan_jsonl, str) or not plan_jsonl:
        raise NoisePredSupersessionError(
            "predecessor pred frozen summary.plan_jsonl 必须是非空路径"
        )
    plan_path = Path(plan_jsonl).resolve()
    if FORBIDDEN_SOURCE in str(plan_path):
        raise NoisePredSupersessionError(
            f"predecessor plan 路径命中禁止来源 {FORBIDDEN_SOURCE}"
        )
    if not plan_path.is_file() or _sha256_file(plan_path) != plan_sha256:
        raise NoisePredSupersessionError(
            "predecessor pred frozen summary 引用的 plan 不存在或 SHA256 漂移"
        )
    row_count = _require_nonnegative_int(
        payload.get("row_count"),
        label="predecessor pred frozen summary.row_count",
    )
    if row_count != expected_total_count:
        raise NoisePredSupersessionError(
            f"predecessor pred frozen summary 数量不符: {row_count} != {expected_total_count}"
        )
    raw_counts = _require_mapping(
        payload.get("state_counts"),
        label="predecessor pred frozen summary.state_counts",
    )
    unknown = set(raw_counts) - {"frozen", "non_applicable"}
    if unknown:
        raise NoisePredSupersessionError(
            f"predecessor pred frozen summary 含非终态: {sorted(unknown)}"
        )
    counts = {
        str(key): _require_nonnegative_int(
            value,
            label=f"predecessor pred frozen summary.state_counts.{key}",
        )
        for key, value in raw_counts.items()
    }
    if sum(counts.values()) != row_count:
        raise NoisePredSupersessionError(
            "predecessor pred frozen summary state_counts 与 row_count 不闭合"
        )
    return plan_sha256, plan_path, counts


def _validate_manifest_plan_artifact(
    plan_meta: Mapping[str, object],
    *,
    condition: str,
    label: str,
    expected_rows: int,
    expected_sha256: str | None = None,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    raw_path = plan_meta.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label}.path 必须是非空路径"
        )
    path = Path(raw_path).resolve()
    if FORBIDDEN_SOURCE in str(path):
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label}.path 命中禁止来源 {FORBIDDEN_SOURCE}"
        )
    sha256 = _require_sha256(
        plan_meta.get("sha256"),
        label=f"manifest.conditions.{condition}.{label}.sha256",
    )
    rows = _require_nonnegative_int(
        plan_meta.get("rows"),
        label=f"manifest.conditions.{condition}.{label}.rows",
    )
    if rows != expected_rows:
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label}.rows 漂移: "
            f"{rows} != {expected_rows}"
        )
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label}.sha256 漂移"
        )
    if expected_path is not None and path != expected_path:
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label}.path 与输入路径不一致"
        )
    if not path.is_file() or _sha256_file(path) != sha256:
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.{label} 文件不存在或 SHA256 漂移"
        )
    return path, sha256


def _load_bindings(
    path: Path,
    *,
    condition: str,
    expected_count: int,
    expected_total_count: int,
    predecessor_plan_sha256: str,
    predecessor_plan_path: Path,
    hybrid_plan_sha256: str,
    hybrid_plan_path: Path,
    plan_specs: Mapping[str, TaskSpec],
) -> list[JsonDict]:
    _reject_forbidden_text_file(path, label="noise pred hybrid manifest")
    payload = _read_json_object(path, label="noise pred hybrid manifest")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION or payload.get("status") != "ok":
        raise NoisePredSupersessionError(
            "noise pred hybrid manifest schema_version/status 非法"
        )
    conditions = _require_mapping(payload.get("conditions"), label="manifest.conditions")
    selected = _require_mapping(
        conditions.get(condition),
        label=f"manifest.conditions.{condition}",
    )
    plan_metadata = {
        label: _require_mapping(
            selected.get(label),
            label=f"manifest.conditions.{condition}.{label}",
        )
        for label in ("predecessor_plan", "fresh_plan", "hybrid_plan", "refresh_plan")
    }
    _validate_manifest_plan_artifact(
        plan_metadata["predecessor_plan"],
        condition=condition,
        label="predecessor_plan",
        expected_rows=expected_total_count,
        expected_sha256=predecessor_plan_sha256,
        expected_path=predecessor_plan_path,
    )
    _validate_manifest_plan_artifact(
        plan_metadata["fresh_plan"],
        condition=condition,
        label="fresh_plan",
        expected_rows=expected_total_count,
    )
    _validate_manifest_plan_artifact(
        plan_metadata["hybrid_plan"],
        condition=condition,
        label="hybrid_plan",
        expected_rows=expected_total_count,
        expected_sha256=hybrid_plan_sha256,
        expected_path=hybrid_plan_path,
    )
    _validate_manifest_plan_artifact(
        plan_metadata["refresh_plan"],
        condition=condition,
        label="refresh_plan",
        expected_rows=expected_count,
    )

    raw_bindings = selected.get("bindings")
    if not isinstance(raw_bindings, list):
        raise NoisePredSupersessionError(
            f"manifest.conditions.{condition}.bindings 必须是 JSON array"
        )
    if len(raw_bindings) != expected_count:
        raise NoisePredSupersessionError(
            f"pred supersession 数量不符: {len(raw_bindings)} != {expected_count}"
        )

    bindings: list[JsonDict] = []
    seen_bases: set[str] = set()
    seen_predecessors: set[str] = set()
    seen_successors: set[str] = set()
    for index, raw_binding in enumerate(raw_bindings):
        binding = dict(_require_mapping(raw_binding, label=f"binding[{index}]"))
        base = str(binding.get("base_logical_id", ""))
        match = _PRED_BASE.fullmatch(base)
        predecessor_logical_id = str(binding.get("predecessor_logical_id", ""))
        successor_logical_id = str(binding.get("successor_logical_id", ""))
        predecessor_key = _require_sha256(
            binding.get("predecessor_evaluation_key"),
            label=f"binding[{index}].predecessor_evaluation_key",
        )
        successor_key = _require_sha256(
            binding.get("successor_evaluation_key"),
            label=f"binding[{index}].successor_evaluation_key",
        )
        old_source_sha = _require_sha256(
            binding.get("old_source_result_sha256"),
            label=f"binding[{index}].old_source_result_sha256",
        )
        new_source_sha = _require_sha256(
            binding.get("new_source_result_sha256"),
            label=f"binding[{index}].new_source_result_sha256",
        )
        successor_spec = plan_specs.get(successor_key)
        if (
            match is None
            or match.group("condition") != condition
            or _base_logical_id(predecessor_logical_id) != base
            or successor_logical_id != _next_logical_id(predecessor_logical_id)
            or successor_spec is None
            or successor_spec.logical_id != successor_logical_id
            or _base_logical_id(successor_spec.logical_id) != base
        ):
            raise NoisePredSupersessionError(
                f"binding[{index}] base/logical/condition/key 与 hybrid plan 不一致"
            )
        if predecessor_key == successor_key or old_source_sha == new_source_sha:
            raise NoisePredSupersessionError(
                f"binding[{index}] predecessor/successor 未发生变化"
            )
        if (
            base in seen_bases
            or predecessor_key in seen_predecessors
            or successor_key in seen_successors
        ):
            raise NoisePredSupersessionError(
                f"binding[{index}] base/evaluation identity 重复"
            )
        seen_bases.add(base)
        seen_predecessors.add(predecessor_key)
        seen_successors.add(successor_key)
        binding["successor_spec"] = successor_spec
        bindings.append(binding)
    return bindings


def _read_state_meta(path: Path) -> dict[str, str]:
    if FORBIDDEN_SOURCE in str(path):
        raise NoisePredSupersessionError(f"state DB 路径命中禁止来源 {FORBIDDEN_SOURCE}")
    if not path.is_file() or path.stat().st_size == 0:
        raise NoisePredSupersessionError(f"state DB 不存在或为空: {path}")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            return dict(connection.execute("SELECT key, value FROM meta").fetchall())
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise NoisePredSupersessionError(f"state DB meta 不可读: {exc}") from exc


def _state_store_from_meta(path: Path, meta: Mapping[str, str]) -> TaskStateStore:
    required = {"attempt_cap", "logical_task_cap", "max_attempts_per_task"}
    if not required.issubset(meta):
        raise NoisePredSupersessionError("state DB meta 缺少容量契约")
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
    except (KeyError, TypeError, ValueError, StateContractError, sqlite3.Error) as exc:
        raise NoisePredSupersessionError(f"state DB 冻结契约无法还原: {exc}") from exc


def _state_snapshot(
    path: Path,
    bindings: Sequence[Mapping[str, Any]],
    *,
    predecessor_plan_sha256: str,
    successor_plan_sha256: str,
) -> JsonDict:
    predecessor_states: Counter[str] = Counter()
    successor_states: Counter[str] = Counter()
    mapping_count = 0
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            for binding in bindings:
                predecessor_key = str(binding["predecessor_evaluation_key"])
                successor_key = str(binding["successor_evaluation_key"])
                predecessor = connection.execute(
                    "SELECT * FROM tasks WHERE evaluation_key=?", (predecessor_key,)
                ).fetchone()
                if predecessor is None:
                    raise NoisePredSupersessionError(
                        f"state DB 缺少 predecessor: {predecessor_key}"
                    )
                if (
                    str(predecessor["logical_id"]) != binding["predecessor_logical_id"]
                    or str(predecessor["task_type"]) != "pred_simplify"
                    or str(predecessor["condition_name"]) != binding["successor_spec"].condition
                    or FORBIDDEN_SOURCE in str(predecessor["spec_json"])
                ):
                    raise NoisePredSupersessionError(
                        f"state predecessor identity/provenance 漂移: {predecessor_key}"
                    )
                predecessor_state = str(predecessor["state"])
                predecessor_states[predecessor_state] += 1
                successor = connection.execute(
                    "SELECT * FROM tasks WHERE evaluation_key=?", (successor_key,)
                ).fetchone()
                mapping = connection.execute(
                    """SELECT * FROM task_supersessions
                       WHERE predecessor_evaluation_key=?""",
                    (predecessor_key,),
                ).fetchone()
                if predecessor_state in {"frozen", "non_applicable"}:
                    binding_table = (
                        "frozen_results"
                        if predecessor_state == "frozen"
                        else "non_applicable_results"
                    )
                    audit_binding = connection.execute(
                        f"SELECT 1 FROM {binding_table} WHERE evaluation_key=?",
                        (predecessor_key,),
                    ).fetchone()
                    if audit_binding is None or successor is not None or mapping is not None:
                        raise NoisePredSupersessionError(
                            f"state predecessor 终态绑定不闭合: {predecessor_key}"
                        )
                    continue
                if predecessor_state != "superseded" or successor is None or mapping is None:
                    raise NoisePredSupersessionError(
                        f"state predecessor/successor supersession 不闭合: {predecessor_key}"
                    )
                expected_spec = binding["successor_spec"].canonical_json()
                if (
                    str(successor["logical_id"]) != binding["successor_logical_id"]
                    or str(successor["spec_json"]) != expected_spec
                    or str(successor["state"]) == "superseded"
                    or str(mapping["successor_evaluation_key"]) != successor_key
                    or str(mapping["identity"])
                    != f"noise_pred_final_replacement::{binding['base_logical_id']}"
                    or str(mapping["reason"])
                    != "noise final replacement changed prediction input"
                    or str(mapping["predecessor_plan_sha256"]) != predecessor_plan_sha256
                    or str(mapping["successor_plan_sha256"]) != successor_plan_sha256
                ):
                    raise NoisePredSupersessionError(
                        f"state supersession 契约漂移: {predecessor_key}->{successor_key}"
                    )
                successor_states[str(successor["state"])] += 1
                mapping_count += 1
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise NoisePredSupersessionError(f"state DB 闭合检查失败: {exc}") from exc
    return {
        "predecessor_states": dict(sorted(predecessor_states.items())),
        "successor_states": dict(sorted(successor_states.items())),
        "mapping_count": mapping_count,
    }


def register_noise_pred_supersessions(
    *,
    condition: str,
    manifest_json: str | Path,
    hybrid_plan_jsonl: str | Path,
    predecessor_frozen_summary_json: str | Path,
    state_db: str | Path,
    report_json: str | Path,
    expected_count: int,
    expected_total_count: int = 2250,
) -> JsonDict:
    """校验 replacement 证据后，以单个 SQLite 事务注册整批 supersession。"""

    if condition not in {"noise001", "noise005"}:
        raise NoisePredSupersessionError("condition 必须是 noise001 或 noise005")
    if expected_count <= 0 or expected_total_count <= 0:
        raise NoisePredSupersessionError("expected count 必须为正整数")
    manifest_path = Path(manifest_json).resolve()
    plan_path = Path(hybrid_plan_jsonl).resolve()
    summary_path = Path(predecessor_frozen_summary_json).resolve()
    state_path = Path(state_db).resolve()
    report_path = Path(report_json).resolve()
    if report_path in {manifest_path, plan_path, summary_path, state_path}:
        raise NoisePredSupersessionError("report 输出不得覆盖输入")

    plan_specs, successor_plan_sha256 = _load_hybrid_plan(
        plan_path,
        condition=condition,
        expected_total_count=expected_total_count,
    )
    (
        predecessor_plan_sha256,
        predecessor_plan_path,
        summary_state_counts,
    ) = _load_predecessor_summary(
        summary_path,
        expected_total_count=expected_total_count,
    )
    bindings = _load_bindings(
        manifest_path,
        condition=condition,
        expected_count=expected_count,
        expected_total_count=expected_total_count,
        predecessor_plan_sha256=predecessor_plan_sha256,
        predecessor_plan_path=predecessor_plan_path,
        hybrid_plan_sha256=successor_plan_sha256,
        hybrid_plan_path=plan_path,
        plan_specs=plan_specs,
    )
    meta = _read_state_meta(state_path)
    before = _state_snapshot(
        state_path,
        bindings,
        predecessor_plan_sha256=predecessor_plan_sha256,
        successor_plan_sha256=successor_plan_sha256,
    )
    supersessions = tuple(
        TaskSupersession(
            predecessor_evaluation_key=str(binding["predecessor_evaluation_key"]),
            successor=binding["successor_spec"],
            identity=f"noise_pred_final_replacement::{binding['base_logical_id']}",
            reason="noise final replacement changed prediction input",
            predecessor_plan_sha256=predecessor_plan_sha256,
            successor_plan_sha256=successor_plan_sha256,
        )
        for binding in bindings
    )
    try:
        store = _state_store_from_meta(state_path, meta)
        store.register_dependency_rebinding_supersession_batch(supersessions)
    except (StateContractError, sqlite3.Error) as exc:
        raise NoisePredSupersessionError(
            f"dependency rebinding supersession 原子注册失败: {exc}"
        ) from exc
    after = _state_snapshot(
        state_path,
        bindings,
        predecessor_plan_sha256=predecessor_plan_sha256,
        successor_plan_sha256=successor_plan_sha256,
    )
    if after["mapping_count"] != expected_count:
        raise NoisePredSupersessionError("supersession 注册后 mapping 数量不闭合")
    if after["predecessor_states"] != {"superseded": expected_count}:
        raise NoisePredSupersessionError("supersession 注册后 predecessor 状态不闭合")

    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "model_invoked": False,
        "condition": condition,
        "inputs": {
            "manifest_json": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
            "hybrid_plan_jsonl": {
                "path": str(plan_path),
                "sha256": successor_plan_sha256,
                "row_count": len(plan_specs),
            },
            "predecessor_frozen_summary_json": {
                "path": str(summary_path),
                "sha256": _sha256_file(summary_path),
                "plan_sha256": predecessor_plan_sha256,
                "state_counts": summary_state_counts,
            },
            "state_db": str(state_path),
        },
        "state_contract": {
            "attempt_cap": int(meta["attempt_cap"]),
            "logical_task_cap": int(meta["logical_task_cap"]),
            "max_attempts_per_task": int(meta["max_attempts_per_task"]),
            "predecessor_attempt_manifest_path": meta.get(
                "predecessor_attempt_manifest_path", ""
            ),
            "predecessor_attempt_manifest_sha256": meta.get(
                "predecessor_attempt_manifest_sha256", ""
            ),
            "predecessor_attempt_count": int(meta.get("predecessor_attempt_count", "0")),
        },
        "counts": {
            "requested": expected_count,
            "predecessor_state_before": before["predecessor_states"],
            "mapping_before": before["mapping_count"],
            "mapping_after": after["mapping_count"],
            "successor_state_after": after["successor_states"],
        },
        "state_registration": {
            "atomic_batch": True,
            "dependency_rebinding": True,
            "mutated": before["mapping_count"] != after["mapping_count"],
            "state_db": str(state_path),
        },
    }
    _atomic_write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="注册噪声 pred predecessor->successor 原子 supersession"
    )
    parser.add_argument("--condition", choices=("noise001", "noise005"), required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--hybrid-plan-jsonl", type=Path, required=True)
    parser.add_argument(
        "--predecessor-frozen-summary-json",
        "--predecessor-pred-frozen-summary-json",
        dest="predecessor_frozen_summary_json",
        type=Path,
        required=True,
    )
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-total-count", type=int, default=2250)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = register_noise_pred_supersessions(
            condition=args.condition,
            manifest_json=args.manifest_json,
            hybrid_plan_jsonl=args.hybrid_plan_jsonl,
            predecessor_frozen_summary_json=args.predecessor_frozen_summary_json,
            state_db=args.state_db,
            report_json=args.report_json,
            expected_count=args.expected_count,
            expected_total_count=args.expected_total_count,
        )
    except NoisePredSupersessionError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "contract_error",
            "model_invoked": False,
            "condition": args.condition,
            "error": str(exc),
        }
        _atomic_write_json(args.report_json, failure)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
