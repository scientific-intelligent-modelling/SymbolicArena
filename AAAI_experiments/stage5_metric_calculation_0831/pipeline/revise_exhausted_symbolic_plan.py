"""为 exhausted symbolic 任务生成同契约、仅版本递增的 successor 计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import canonical_json, evaluation_key
from .run_claude_plan import PlanContractError, load_plan_jsonl
from .state import TaskSpec


JsonDict = dict[str, object]
SUPPORTED_TASK_TYPES = {"equivalence", "stab_structure"}
SUPPORTED_CONDITIONS = {"clean", "noise001", "noise005"}
_SUFFIX_PATTERN = re.compile(r"v[1-9]\d*")
_LOGICAL_ID_PATTERNS = {
    "equivalence": re.compile(
        r"^(equivalence::[a-z0-9_]+::g\d{4}::s(?:520|521|522)::"
        r"(?:clean|noise001|noise005))(?:::(v[1-9]\d*))?$"
    ),
    "stab_structure": re.compile(
        r"^(stab_structure::[a-z0-9_]+::g\d{4}::"
        r"s(?:520|521|522)-s(?:520|521|522))(?:::(v[1-9]\d*))?$"
    ),
}


class ReviseExhaustedSymbolicPlanError(RuntimeError):
    """symbolic predecessor 状态或 successor 计划不满足冻结契约。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _read_plan_rows(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ReviseExhaustedSymbolicPlanError(
                    f"predecessor plan 第 {line_number} 行不是 JSON object"
                )
            rows.append(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviseExhaustedSymbolicPlanError(
            f"predecessor plan 不可读: {exc}"
        ) from exc
    return rows


def _successor_logical_id(
    predecessor_logical_id: str,
    *,
    task_type: str,
    logical_id_suffix: str,
) -> str:
    match = _LOGICAL_ID_PATTERNS[task_type].fullmatch(predecessor_logical_id)
    if match is None:
        raise ReviseExhaustedSymbolicPlanError(
            f"predecessor logical_id 非 canonical: {predecessor_logical_id!r}"
        )
    successor_version = int(logical_id_suffix[1:])
    predecessor_suffix = match.group(2)
    predecessor_version = (
        int(predecessor_suffix[1:]) if predecessor_suffix is not None else 1
    )
    if successor_version <= predecessor_version:
        raise ReviseExhaustedSymbolicPlanError(
            f"successor 版本必须递增: v{predecessor_version} -> {logical_id_suffix}"
        )
    return f"{match.group(1)}::{logical_id_suffix}"


def _successor_row(
    predecessor: Mapping[str, Any],
    *,
    task_type: str,
    logical_id_suffix: str,
) -> JsonDict:
    logical_id = _successor_logical_id(
        str(predecessor["logical_id"]),
        task_type=task_type,
        logical_id_suffix=logical_id_suffix,
    )
    request = predecessor["request"]
    normalized_input = predecessor["normalized_input"]
    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=str(predecessor["prompt_version"]),
        schema_version=str(predecessor["schema_version"]),
        prompt_sha256=str(predecessor["prompt_sha256"]),
        schema_sha256=str(predecessor["schema_sha256"]),
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    old_spec = predecessor["task_spec"]
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=str(predecessor["condition"]),
        priority=int(predecessor["priority"]),
        input_hash=str(predecessor["input_hash"]),
        prompt_version=str(predecessor["prompt_version"]),
        schema_version=str(predecessor["schema_version"]),
        dependencies=tuple(predecessor["dependencies"]),
    )
    if not isinstance(old_spec, Mapping):
        raise ReviseExhaustedSymbolicPlanError("predecessor task_spec 不是 object")
    expected_old_fields = set(old_spec) - {"evaluation_key", "logical_id"}
    successor_spec = json.loads(spec.canonical_json())
    if any(successor_spec[field] != old_spec[field] for field in expected_old_fields):
        raise ReviseExhaustedSymbolicPlanError(
            "successor task_spec 除版本身份外发生漂移"
        )
    successor = dict(predecessor)
    successor.update(
        {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "task_spec": successor_spec,
        }
    )
    changed_fields = {
        field
        for field in successor
        if successor.get(field) != predecessor.get(field)
    }
    if changed_fields != {"evaluation_key", "logical_id", "task_spec"}:
        raise ReviseExhaustedSymbolicPlanError(
            f"successor 出现非身份字段漂移: {sorted(changed_fields)}"
        )
    return successor


def revise_exhausted_symbolic_plan(
    *,
    predecessor_plan_jsonl: str | Path,
    state_db: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    task_type: str,
    condition: str,
    logical_id_suffix: str = "v2",
    expected_task_count: int | None = 2250,
    expected_exhausted_count: int | None = None,
) -> JsonDict:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ReviseExhaustedSymbolicPlanError(
            f"task_type 只允许 {sorted(SUPPORTED_TASK_TYPES)}"
        )
    if condition not in SUPPORTED_CONDITIONS:
        raise ReviseExhaustedSymbolicPlanError(
            f"condition 只允许 {sorted(SUPPORTED_CONDITIONS)}"
        )
    if not _SUFFIX_PATTERN.fullmatch(logical_id_suffix):
        raise ReviseExhaustedSymbolicPlanError("logical_id_suffix 格式无效")
    if int(logical_id_suffix[1:]) < 2:
        raise ReviseExhaustedSymbolicPlanError("logical_id_suffix 必须从 v2 开始")

    predecessor_path = Path(predecessor_plan_jsonl).resolve()
    try:
        loaded = load_plan_jsonl(predecessor_path)
    except PlanContractError as exc:
        raise ReviseExhaustedSymbolicPlanError(
            f"predecessor plan 契约失败: {exc}"
        ) from exc
    rows = _read_plan_rows(predecessor_path)
    if len(rows) != len(loaded.entries):
        raise ReviseExhaustedSymbolicPlanError(
            "predecessor plan 行数与已验证条目数不一致"
        )
    if expected_task_count is not None and len(rows) != expected_task_count:
        raise ReviseExhaustedSymbolicPlanError(
            f"predecessor task 数量不符: {len(rows)} != {expected_task_count}"
        )
    if any(
        entry.definition.task_spec.task_type != task_type
        or entry.definition.task_spec.condition != condition
        for entry in loaded.entries
    ):
        raise ReviseExhaustedSymbolicPlanError(
            "predecessor plan 与显式 task_type/condition 范围不一致"
        )
    row_by_key = {str(row.get("evaluation_key")): row for row in rows}
    if len(row_by_key) != len(rows) or set(row_by_key) != {
        entry.evaluation_key for entry in loaded.entries
    }:
        raise ReviseExhaustedSymbolicPlanError(
            "predecessor plan evaluation_key 不唯一或漂移"
        )

    state_path = Path(state_db).resolve()
    if not state_path.is_file():
        raise ReviseExhaustedSymbolicPlanError(f"状态库不存在: {state_path}")
    connection = sqlite3.connect(f"{state_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        task_rows = {
            str(row["evaluation_key"]): row
            for row in connection.execute(
                """SELECT evaluation_key, logical_id, task_type, condition_name,
                          state, attempt_count, spec_json
                   FROM tasks"""
            ).fetchall()
        }
        frozen_keys = {
            str(row["evaluation_key"])
            for row in connection.execute(
                "SELECT evaluation_key FROM frozen_results"
            ).fetchall()
        }
        attempt_counts = {
            str(row["evaluation_key"]): (
                int(row["count"]),
                int(row["failed_count"]),
            )
            for row in connection.execute(
                """SELECT evaluation_key, COUNT(*) AS count,
                          SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count
                   FROM attempts GROUP BY evaluation_key"""
            ).fetchall()
        }
    finally:
        connection.close()

    output_rows: list[JsonDict] = []
    successor_bindings: list[JsonDict] = []
    preserved_frozen_count = 0
    for entry in loaded.entries:
        task = task_rows.get(entry.evaluation_key)
        if task is None:
            raise ReviseExhaustedSymbolicPlanError(
                f"状态库缺少任务: {entry.logical_id}"
            )
        if (
            task["logical_id"] != entry.logical_id
            or task["task_type"] != task_type
            or task["condition_name"] != condition
            or task["spec_json"] != entry.definition.task_spec.canonical_json()
        ):
            raise ReviseExhaustedSymbolicPlanError(
                f"任务身份漂移: {entry.logical_id}"
            )
        predecessor_row = row_by_key[entry.evaluation_key]
        state = str(task["state"])
        if state == "frozen":
            if entry.evaluation_key not in frozen_keys:
                raise ReviseExhaustedSymbolicPlanError(
                    f"frozen 任务缺少结果绑定: {entry.logical_id}"
                )
            output_rows.append(dict(predecessor_row))
            preserved_frozen_count += 1
            continue
        if state != "exhausted":
            raise ReviseExhaustedSymbolicPlanError(
                f"任务尚未形成 frozen/exhausted 闭环: {entry.logical_id}={state}"
            )
        if entry.evaluation_key in frozen_keys:
            raise ReviseExhaustedSymbolicPlanError(
                f"exhausted 任务与 frozen binding 冲突: {entry.logical_id}"
            )
        counts = attempt_counts.get(entry.evaluation_key)
        if int(task["attempt_count"]) != 3 or counts != (3, 3):
            raise ReviseExhaustedSymbolicPlanError(
                f"exhausted 任务必须恰有三次 failed attempts: {entry.logical_id}"
            )
        successor = _successor_row(
            predecessor_row,
            task_type=task_type,
            logical_id_suffix=logical_id_suffix,
        )
        output_rows.append(successor)
        successor_bindings.append(
            {
                "predecessor_evaluation_key": entry.evaluation_key,
                "predecessor_logical_id": entry.logical_id,
                "successor_evaluation_key": successor["evaluation_key"],
                "successor_logical_id": successor["logical_id"],
                "predecessor_attempt_count": 3,
            }
        )

    if (
        expected_exhausted_count is not None
        and len(successor_bindings) != expected_exhausted_count
    ):
        raise ReviseExhaustedSymbolicPlanError(
            f"exhausted 数量不符: {len(successor_bindings)} != {expected_exhausted_count}"
        )
    logical_ids = [str(row["logical_id"]) for row in output_rows]
    evaluation_keys = [str(row["evaluation_key"]) for row in output_rows]
    if len(set(logical_ids)) != len(output_rows) or len(set(evaluation_keys)) != len(
        output_rows
    ):
        raise ReviseExhaustedSymbolicPlanError(
            "successor plan 出现重复任务身份"
        )

    output_path = Path(output_jsonl).resolve()
    _atomic_write_text(
        output_path,
        "".join(canonical_json(row) + "\n" for row in output_rows),
    )
    try:
        verified_output = load_plan_jsonl(output_path)
    except PlanContractError as exc:
        raise ReviseExhaustedSymbolicPlanError(
            f"successor plan 回读失败: {exc}"
        ) from exc
    if len(verified_output.entries) != len(output_rows):
        raise ReviseExhaustedSymbolicPlanError("successor plan 回读数量漂移")

    report: JsonDict = {
        "status": "ok",
        "schema_version": "exhausted_symbolic_successor_report.v1",
        "model_invoked": False,
        "state_db_mutated": False,
        "task_type": task_type,
        "condition": condition,
        "predecessor_plan_jsonl": str(predecessor_path),
        "predecessor_plan_sha256": loaded.plan_sha256,
        "state_db": str(state_path),
        "logical_id_suffix": logical_id_suffix,
        "preserved_frozen_count": preserved_frozen_count,
        "successor_task_count": len(successor_bindings),
        "output_task_count": len(output_rows),
        "output_jsonl": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "successor_bindings": successor_bindings,
    }
    _atomic_write_json(Path(report_json).resolve(), report)
    return report


def _optional_count(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--task-type", choices=sorted(SUPPORTED_TASK_TYPES), required=True)
    parser.add_argument("--condition", choices=sorted(SUPPORTED_CONDITIONS), required=True)
    parser.add_argument("--logical-id-suffix", default="v2")
    parser.add_argument("--expected-task-count", type=_optional_count, default=2250)
    parser.add_argument("--expected-exhausted-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = revise_exhausted_symbolic_plan(
            predecessor_plan_jsonl=args.predecessor_plan_jsonl,
            state_db=args.state_db,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            task_type=args.task_type,
            condition=args.condition,
            logical_id_suffix=args.logical_id_suffix,
            expected_task_count=args.expected_task_count,
            expected_exhausted_count=args.expected_exhausted_count,
        )
    except ReviseExhaustedSymbolicPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
