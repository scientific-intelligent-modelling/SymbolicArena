#!/usr/bin/env python3
"""为 clean final replacement 构建 75 条增量 pred successor hybrid plan。"""

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
SCHEMA_VERSION = "clean_pred_hybrid_active_plan.v1"
_VERSION_SUFFIX = re.compile(r"::v([1-9]\d*)$")


class CleanPredHybridPlanError(RuntimeError):
    """active/fresh/binding/state 证据不足以生成严格增量计划。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json_object(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanPredHybridPlanError(f"{label} 不可读: {exc}") from exc
    if not isinstance(payload, dict):
        raise CleanPredHybridPlanError(f"{label} 顶层不是 JSON object")
    return payload


def _base_logical_id(logical_id: object) -> str:
    return _VERSION_SUFFIX.sub("", str(logical_id))


def _next_logical_id(logical_id: object) -> str:
    text = str(logical_id)
    base = _base_logical_id(text)
    match = _VERSION_SUFFIX.search(text)
    next_version = 2 if match is None else int(match.group(1)) + 1
    return f"{base}::v{next_version}"


def _load_required(path: Path, *, expected_count: int | None) -> set[str]:
    payload = _read_json_object(path, label="binding manifest")
    readiness = payload.get("aggregation_readiness")
    values = (
        readiness.get("required_pred_simplify_logical_ids")
        if isinstance(readiness, Mapping)
        else None
    )
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise CleanPredHybridPlanError("binding manifest 缺少 required pred 集合")
    required = set(values)
    if len(required) != len(values):
        raise CleanPredHybridPlanError("binding manifest required pred 集合重复")
    if any(_base_logical_id(value) != value for value in required):
        raise CleanPredHybridPlanError("binding manifest required pred 必须使用 base logical identity")
    if expected_count is not None and len(required) != expected_count:
        raise CleanPredHybridPlanError(
            f"required pred 数量不符: {len(required)} != {expected_count}"
        )
    return required


def _validated_plan(path: Path, *, label: str) -> tuple[str, set[str]]:
    try:
        loaded = load_plan_jsonl(path)
    except PlanContractError as exc:
        raise CleanPredHybridPlanError(f"{label} 契约失败: {exc}") from exc
    evaluation_keys = {entry.evaluation_key for entry in loaded.entries}
    if len(evaluation_keys) != len(loaded.entries):
        raise CleanPredHybridPlanError(f"{label} evaluation_key 重复")
    return loaded.plan_sha256, evaluation_keys


def _read_plan_rows(
    path: Path,
    *,
    label: str,
    validated_evaluation_keys: set[str],
    retain_bases: set[str] | None = None,
) -> list[tuple[str, JsonDict]]:
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
                    raise CleanPredHybridPlanError(
                        f"{label}:{line_number} 顶层不是 JSON object"
                    )
                evaluation = str(payload.get("evaluation_key", ""))
                logical_id = str(payload.get("logical_id", ""))
                base = _base_logical_id(logical_id)
                if evaluation in seen_keys:
                    raise CleanPredHybridPlanError(f"{label} evaluation_key 重复: {evaluation}")
                if base in seen_bases:
                    raise CleanPredHybridPlanError(f"{label} base logical identity 重复: {base}")
                seen_keys.add(evaluation)
                seen_bases.add(base)
                if retain_bases is None or base in retain_bases:
                    rows.append((raw_line if raw_line.endswith("\n") else raw_line + "\n", payload))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanPredHybridPlanError(f"{label} 不可读: {exc}") from exc
    if seen_keys != validated_evaluation_keys:
        raise CleanPredHybridPlanError(f"{label} 回读 evaluation_key 集合漂移")
    return rows


def _successor_row(predecessor: Mapping[str, Any], fresh: Mapping[str, Any]) -> JsonDict:
    base = _base_logical_id(predecessor["logical_id"])
    if str(fresh.get("logical_id")) != base:
        raise CleanPredHybridPlanError(f"fresh row 不是未版本化 base identity: {base}")
    for field in ("task_type", "condition", "priority", "dependencies"):
        if predecessor.get(field) != fresh.get(field):
            raise CleanPredHybridPlanError(f"{base} fresh.{field} 与 predecessor 漂移")
    request = fresh.get("request")
    schema = fresh.get("schema_content")
    if not isinstance(request, Mapping) or not isinstance(schema, Mapping):
        raise CleanPredHybridPlanError(f"{base} fresh request/schema 缺失")
    evidence_hash = request.get("evidence_hash")
    if not isinstance(evidence_hash, str) or not evidence_hash:
        raise CleanPredHybridPlanError(f"{base} fresh request.evidence_hash 缺失")
    prompt_sha = str(fresh.get("prompt_sha256", ""))
    schema_sha = str(fresh.get("schema_sha256", ""))
    prompt_template = str(fresh.get("prompt_template", ""))
    logical_id = _next_logical_id(predecessor["logical_id"])
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = _sha256_text(canonical_json(normalized_input))
    task_key = evaluation_key(
        task_type=str(fresh["task_type"]),
        logical_id=logical_id,
        prompt_version=str(fresh["prompt_version"]),
        schema_version=str(fresh["schema_version"]),
        prompt_sha256=prompt_sha,
        schema_sha256=schema_sha,
        normalized_input=normalized_input,
        evidence_hash=evidence_hash,
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=str(fresh["task_type"]),
        condition=str(fresh["condition"]),
        priority=int(fresh["priority"]),
        input_hash=input_hash,
        prompt_version=str(fresh["prompt_version"]),
        schema_version=str(fresh["schema_version"]),
        dependencies=tuple(str(value) for value in fresh["dependencies"]),
    )
    successor = dict(fresh)
    successor.update(
        {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "input_hash": input_hash,
            "normalized_input": normalized_input,
            "task_spec": json.loads(spec.canonical_json()),
            "rendered_prompt": render_prompt(prompt_template, request, schema),
        }
    )
    return successor


def _task_spec(row: Mapping[str, Any]) -> TaskSpec:
    payload = row.get("task_spec")
    if not isinstance(payload, Mapping):
        raise CleanPredHybridPlanError(f"任务 {row.get('logical_id')} 缺少 task_spec")
    return TaskSpec(
        evaluation_key=str(payload["evaluation_key"]),
        logical_id=str(payload["logical_id"]),
        task_type=str(payload["task_type"]),
        condition=str(payload["condition"]),
        priority=int(payload["priority"]),
        input_hash=str(payload["input_hash"]),
        prompt_version=str(payload["prompt_version"]),
        schema_version=str(payload["schema_version"]),
        dependencies=tuple(str(value) for value in payload["dependencies"]),
    )


def _read_state_meta(path: Path) -> tuple[dict[str, str], sqlite3.Connection]:
    if not path.is_file():
        raise CleanPredHybridPlanError(f"register state DB 不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        connection.close()
        raise
    return meta, connection


def _state_snapshot(
    path: Path,
    bindings: Sequence[Mapping[str, Any]],
    *,
    expected_before: bool,
    predecessor_plan_sha256: str | None = None,
    successor_plan_sha256: str | None = None,
) -> dict[str, str]:
    _, connection = _read_state_meta(path)
    states: dict[str, str] = {}
    try:
        for binding in bindings:
            predecessor_key = str(binding["predecessor_evaluation_key"])
            successor_key = str(binding["successor_evaluation_key"])
            predecessor = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key=?",
                (predecessor_key,),
            ).fetchone()
            successor = connection.execute(
                "SELECT state FROM tasks WHERE evaluation_key=?",
                (successor_key,),
            ).fetchone()
            if predecessor is None:
                raise CleanPredHybridPlanError(f"state DB 缺少 predecessor: {predecessor_key}")
            states[predecessor_key] = str(predecessor["state"])
            if expected_before:
                if predecessor["state"] != "frozen":
                    raise CleanPredHybridPlanError(
                        f"state predecessor 必须为 frozen: {predecessor_key}={predecessor['state']}"
                    )
                frozen = connection.execute(
                    "SELECT 1 FROM frozen_results WHERE evaluation_key=?", (predecessor_key,)
                ).fetchone()
                running = connection.execute(
                    "SELECT 1 FROM attempts WHERE evaluation_key=? AND status='running'",
                    (predecessor_key,),
                ).fetchone()
                if frozen is None or running is not None:
                    raise CleanPredHybridPlanError(
                        f"state predecessor frozen/running binding 非法: {predecessor_key}"
                    )
                if successor is not None:
                    raise CleanPredHybridPlanError(f"state DB 已存在 successor: {successor_key}")
            else:
                if predecessor["state"] != "superseded" or successor is None or successor["state"] != "pending":
                    raise CleanPredHybridPlanError(
                        f"state supersession 后置状态不闭合: {predecessor_key}->{successor_key}"
                    )
                mapping = connection.execute(
                    """SELECT predecessor_plan_sha256, successor_plan_sha256
                       FROM task_supersessions
                       WHERE predecessor_evaluation_key=? AND successor_evaluation_key=?""",
                    (predecessor_key, successor_key),
                ).fetchone()
                if mapping is None:
                    raise CleanPredHybridPlanError(
                        f"state supersession mapping 缺失: {predecessor_key}->{successor_key}"
                    )
                if (
                    str(mapping["predecessor_plan_sha256"])
                    != predecessor_plan_sha256
                    or str(mapping["successor_plan_sha256"])
                    != successor_plan_sha256
                ):
                    raise CleanPredHybridPlanError(
                        f"state supersession plan SHA256 漂移: "
                        f"{predecessor_key}->{successor_key}"
                    )
                states[successor_key] = str(successor["state"])
    finally:
        connection.close()
    return states


def _register_state(
    path: Path,
    bindings: Sequence[Mapping[str, Any]],
    *,
    predecessor_plan_sha256: str,
    successor_plan_sha256: str,
) -> dict[str, Any]:
    before = _state_snapshot(path, bindings, expected_before=True)
    meta, connection = _read_state_meta(path)
    connection.close()
    predecessor_count = int(meta.get("predecessor_attempt_count", "0"))
    predecessor_manifest = None
    if predecessor_count:
        predecessor_manifest = PredecessorAttemptManifest(
            path=meta["predecessor_attempt_manifest_path"],
            sha256=meta["predecessor_attempt_manifest_sha256"],
            attempt_count=predecessor_count,
        )
    try:
        store = TaskStateStore(
            path,
            attempt_cap=int(meta["attempt_cap"]),
            logical_task_cap=int(meta["logical_task_cap"]),
            max_attempts_per_task=int(meta["max_attempts_per_task"]),
            predecessor_attempt_manifest=predecessor_manifest,
        )
        store.register_supersession_batch(
            tuple(
                TaskSupersession(
                    predecessor_evaluation_key=str(binding["predecessor_evaluation_key"]),
                    successor=_task_spec(binding["successor_row"]),
                    identity=str(binding["identity"]),
                    reason=str(binding["reason"]),
                    predecessor_plan_sha256=predecessor_plan_sha256,
                    successor_plan_sha256=successor_plan_sha256,
                )
                for binding in bindings
            )
        )
    except (KeyError, StateContractError, sqlite3.Error) as exc:
        raise CleanPredHybridPlanError(f"state supersession 注册失败: {exc}") from exc
    after = _state_snapshot(
        path,
        bindings,
        expected_before=False,
        predecessor_plan_sha256=predecessor_plan_sha256,
        successor_plan_sha256=successor_plan_sha256,
    )
    return {"requested": True, "mutated": True, "before": before, "after": after}


def build_clean_pred_hybrid_plan(
    *,
    predecessor_plan_jsonl: str | Path,
    fresh_plan_jsonl: str | Path,
    binding_manifest_json: str | Path,
    output_plan_jsonl: str | Path,
    supersession_manifest_json: str | Path,
    report_json: str | Path,
    expected_total_count: int | None = 2250,
    expected_required_count: int | None = 75,
    register_state_db: str | Path | None = None,
) -> JsonDict:
    predecessor_path = Path(predecessor_plan_jsonl).resolve()
    fresh_path = Path(fresh_plan_jsonl).resolve()
    binding_path = Path(binding_manifest_json).resolve()
    output_path = Path(output_plan_jsonl).resolve()
    supersession_path = Path(supersession_manifest_json).resolve()
    report_path = Path(report_json).resolve()
    required = _load_required(binding_path, expected_count=expected_required_count)

    fresh_sha, fresh_keys = _validated_plan(fresh_path, label="fresh plan")
    fresh_rows = _read_plan_rows(
        fresh_path,
        label="fresh plan",
        validated_evaluation_keys=fresh_keys,
        retain_bases=required,
    )
    fresh_by_base = {_base_logical_id(row["logical_id"]): row for _, row in fresh_rows}
    if set(fresh_by_base) != required:
        raise CleanPredHybridPlanError(
            f"fresh required 集合不闭合: missing={sorted(required - set(fresh_by_base))}"
        )

    predecessor_sha, predecessor_keys = _validated_plan(
        predecessor_path, label="predecessor active plan"
    )
    predecessor_rows = _read_plan_rows(
        predecessor_path,
        label="predecessor active plan",
        validated_evaluation_keys=predecessor_keys,
    )
    if expected_total_count is not None and len(predecessor_rows) != expected_total_count:
        raise CleanPredHybridPlanError(
            f"predecessor 数量不符: {len(predecessor_rows)} != {expected_total_count}"
        )
    predecessor_bases = {_base_logical_id(row["logical_id"]) for _, row in predecessor_rows}
    if not required.issubset(predecessor_bases):
        raise CleanPredHybridPlanError("predecessor active plan 缺少 required base identity")

    output_lines: list[str] = []
    output_rows: list[JsonDict] = []
    bindings: list[JsonDict] = []
    preserved_row_hashes: list[JsonDict] = []
    existing_logical_ids = {str(row["logical_id"]) for _, row in predecessor_rows}
    for raw_line, predecessor in predecessor_rows:
        base = _base_logical_id(predecessor["logical_id"])
        if base not in required:
            output_lines.append(raw_line)
            output_rows.append(predecessor)
            preserved_row_hashes.append(
                {"logical_id": predecessor["logical_id"], "sha256": _sha256_text(raw_line.rstrip("\n"))}
            )
            continue
        successor_logical_id = _next_logical_id(predecessor["logical_id"])
        if successor_logical_id in existing_logical_ids:
            raise CleanPredHybridPlanError(f"successor 版本冲突: {successor_logical_id}")
        successor = _successor_row(predecessor, fresh_by_base[base])
        successor_line = canonical_json(successor) + "\n"
        output_lines.append(successor_line)
        output_rows.append(successor)
        bindings.append(
            {
                "base_logical_id": base,
                "predecessor_evaluation_key": predecessor["evaluation_key"],
                "predecessor_logical_id": predecessor["logical_id"],
                "successor_evaluation_key": successor["evaluation_key"],
                "successor_logical_id": successor["logical_id"],
                "identity": f"clean_pred_final_replacement::{base}",
                "reason": "clean final replacement changed prediction input",
                "successor_row": successor,
            }
        )
    if len(bindings) != len(required):
        raise CleanPredHybridPlanError("hybrid changed 集合与 required 集合不一致")
    logical_ids = [str(row["logical_id"]) for row in output_rows]
    evaluation_keys = [str(row["evaluation_key"]) for row in output_rows]
    if len(set(logical_ids)) != len(output_rows) or len(set(evaluation_keys)) != len(output_rows):
        raise CleanPredHybridPlanError("hybrid plan logical/evaluation identity 不唯一")

    _atomic_write(output_path, "".join(output_lines))
    output_sha = _sha256_file(output_path)
    try:
        verified = load_plan_jsonl(output_path)
    except PlanContractError as exc:
        raise CleanPredHybridPlanError(f"hybrid plan 回读失败: {exc}") from exc
    if len(verified.entries) != len(output_rows):
        raise CleanPredHybridPlanError("hybrid plan 回读数量漂移")

    public_bindings = [
        {key: value for key, value in binding.items() if key != "successor_row"}
        for binding in bindings
    ]
    supersession_manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "predecessor_plan_sha256": predecessor_sha,
        "fresh_plan_sha256": fresh_sha,
        "binding_manifest_sha256": _sha256_file(binding_path),
        "hybrid_plan_sha256": output_sha,
        "counts": {"supersessions": len(bindings)},
        "supersessions": public_bindings,
    }
    _atomic_write(
        supersession_path,
        json.dumps(supersession_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    state_registration: JsonDict = {"requested": False, "mutated": False}
    if register_state_db is not None:
        before_hashes = (
            _sha256_file(predecessor_path),
            _sha256_file(fresh_path),
            _sha256_file(output_path),
        )
        state_registration = _register_state(
            Path(register_state_db).resolve(),
            bindings,
            predecessor_plan_sha256=predecessor_sha,
            successor_plan_sha256=output_sha,
        )
        after_hashes = (
            _sha256_file(predecessor_path),
            _sha256_file(fresh_path),
            _sha256_file(output_path),
        )
        if before_hashes != after_hashes:
            raise CleanPredHybridPlanError("state 注册前后计划文件 SHA256 漂移")

    report: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "model_invoked": False,
        "counts": {
            "total": len(output_rows),
            "required": len(required),
            "changed": len(bindings),
            "preserved": len(preserved_row_hashes),
        },
        "required_base_logical_ids": sorted(required),
        "preserved_rows": preserved_row_hashes,
        "inputs": {
            "predecessor_plan": {"path": str(predecessor_path), "sha256": predecessor_sha},
            "fresh_plan": {"path": str(fresh_path), "sha256": fresh_sha},
            "binding_manifest": {"path": str(binding_path), "sha256": _sha256_file(binding_path)},
        },
        "outputs": {
            "hybrid_plan": {"path": str(output_path), "sha256": output_sha},
            "supersession_manifest": {
                "path": str(supersession_path),
                "sha256": _sha256_file(supersession_path),
            },
            "report_self_hash_exclusion": "report cannot contain its own SHA256",
        },
        "state_registration": state_registration,
    }
    _atomic_write(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    stage_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predecessor-plan-jsonl",
        type=Path,
        default=stage_root / "reports/clean_pred_simplify_tasks_active_v5.jsonl",
    )
    parser.add_argument(
        "--fresh-plan-jsonl",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1/pred_plan_smoke/full_plan.jsonl",
    )
    parser.add_argument(
        "--binding-manifest-json",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1/binding_manifest.json",
    )
    parser.add_argument(
        "--output-plan-jsonl",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1/clean_pred_hybrid_active_v6.jsonl",
    )
    parser.add_argument(
        "--supersession-manifest-json",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1/clean_pred_hybrid_supersessions.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=stage_root / "work/clean_final_replacement_v1/clean_pred_hybrid_report.json",
    )
    parser.add_argument("--expected-total-count", type=int, default=2250)
    parser.add_argument("--expected-required-count", type=int, default=75)
    parser.add_argument("--register-state-db", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_clean_pred_hybrid_plan(
            predecessor_plan_jsonl=args.predecessor_plan_jsonl,
            fresh_plan_jsonl=args.fresh_plan_jsonl,
            binding_manifest_json=args.binding_manifest_json,
            output_plan_jsonl=args.output_plan_jsonl,
            supersession_manifest_json=args.supersession_manifest_json,
            report_json=args.report_json,
            expected_total_count=args.expected_total_count,
            expected_required_count=args.expected_required_count,
            register_state_db=args.register_state_db,
        )
    except CleanPredHybridPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
