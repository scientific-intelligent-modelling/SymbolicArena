"""为 exhausted clean 预测化简生成版本化 recovery successor 计划。"""

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
from .state import TaskSpec


JsonDict = dict[str, object]
_SUFFIX_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


class ReviseExhaustedPredPlanError(RuntimeError):
    """predecessor 状态或 successor 计划不满足冻结契约。"""


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
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ReviseExhaustedPredPlanError(
                    f"predecessor plan 第 {line_number} 行不是 JSON object"
                )
            rows.append(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviseExhaustedPredPlanError(f"predecessor plan 不可读: {exc}") from exc
    return rows


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ReviseExhaustedPredPlanError(f"状态库不存在: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _successor_row(
    predecessor: Mapping[str, Any],
    *,
    prompt_path: Path,
    prompt_template: str,
    prompt_sha256: str,
    logical_id_suffix: str,
) -> JsonDict:
    old_logical_id = str(predecessor["logical_id"])
    logical_id = f"{old_logical_id}::{logical_id_suffix}"
    request = predecessor["request"]
    schema = predecessor["schema_content"]
    schema_sha256 = str(predecessor["schema_sha256"])
    schema_version = str(predecessor["schema_version"])
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_bytes(canonical_json(normalized_input).encode("utf-8"))
    task_key = evaluation_key(
        task_type="pred_simplify",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="pred_simplify",
        condition="clean",
        priority=int(predecessor["priority"]),
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_version,
        dependencies=tuple(predecessor["dependencies"]),
    )
    successor = dict(predecessor)
    successor.update(
        {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "input_hash": input_hash,
            "prompt_version": prompt_path.stem,
            "prompt_sha256": prompt_sha256,
            "prompt_path": str(prompt_path),
            "prompt_template": prompt_template,
            "normalized_input": normalized_input,
            "task_spec": json.loads(spec.canonical_json()),
            "rendered_prompt": render_prompt(prompt_template, request, schema),
        }
    )
    return successor


def revise_exhausted_pred_plan(
    *,
    predecessor_plan_jsonl: str | Path,
    state_db: str | Path,
    recovery_prompt_path: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    logical_id_suffix: str = "v2",
    expected_task_count: int | None = 2250,
    expected_exhausted_count: int | None = None,
) -> JsonDict:
    if not _SUFFIX_PATTERN.fullmatch(logical_id_suffix):
        raise ReviseExhaustedPredPlanError("logical_id_suffix 格式无效")
    predecessor_path = Path(predecessor_plan_jsonl).resolve()
    try:
        loaded = load_plan_jsonl(predecessor_path)
    except PlanContractError as exc:
        raise ReviseExhaustedPredPlanError(f"predecessor plan 契约失败: {exc}") from exc
    rows = _read_plan_rows(predecessor_path)
    if len(rows) != len(loaded.entries):
        raise ReviseExhaustedPredPlanError("predecessor plan 行数与已验证条目数不一致")
    if expected_task_count is not None and len(rows) != expected_task_count:
        raise ReviseExhaustedPredPlanError(
            f"predecessor task 数量不符: {len(rows)} != {expected_task_count}"
        )
    if any(
        entry.definition.task_spec.task_type != "pred_simplify"
        or entry.definition.task_spec.condition != "clean"
        for entry in loaded.entries
    ):
        raise ReviseExhaustedPredPlanError("当前工具只允许 clean pred_simplify plan")
    row_by_key = {str(row.get("evaluation_key")): row for row in rows}
    if len(row_by_key) != len(rows) or set(row_by_key) != {
        entry.evaluation_key for entry in loaded.entries
    }:
        raise ReviseExhaustedPredPlanError("predecessor plan evaluation_key 不唯一或漂移")

    prompt_path = Path(recovery_prompt_path).resolve()
    if not prompt_path.is_file():
        raise ReviseExhaustedPredPlanError(f"recovery prompt 不存在: {prompt_path}")
    prompt_bytes = prompt_path.read_bytes()
    prompt_template = prompt_bytes.decode("utf-8")
    prompt_sha256 = _sha256_bytes(prompt_bytes)
    if prompt_template.count("{{REQUEST_JSON}}") != 1:
        raise ReviseExhaustedPredPlanError(
            "recovery prompt 必须恰有一个 {{REQUEST_JSON}} 占位符"
        )

    state_path = Path(state_db).resolve()
    connection = _connect_read_only(state_path)
    try:
        task_rows = {
            str(row["evaluation_key"]): row
            for row in connection.execute(
                """SELECT evaluation_key, logical_id, task_type, condition_name,
                          state, attempt_count
                   FROM tasks"""
            ).fetchall()
        }
        frozen_keys = {
            str(row["evaluation_key"])
            for row in connection.execute("SELECT evaluation_key FROM frozen_results")
        }
        attempt_counts = {
            str(row["evaluation_key"]): (int(row["count"]), int(row["failed_count"]))
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
            raise ReviseExhaustedPredPlanError(f"状态库缺少任务: {entry.logical_id}")
        if (
            task["logical_id"] != entry.logical_id
            or task["task_type"] != "pred_simplify"
            or task["condition_name"] != "clean"
        ):
            raise ReviseExhaustedPredPlanError(f"任务身份漂移: {entry.logical_id}")
        state = str(task["state"])
        predecessor_row = row_by_key[entry.evaluation_key]
        if state == "frozen":
            if entry.evaluation_key not in frozen_keys:
                raise ReviseExhaustedPredPlanError(
                    f"frozen 任务缺少结果绑定: {entry.logical_id}"
                )
            output_rows.append(dict(predecessor_row))
            preserved_frozen_count += 1
            continue
        if state != "exhausted":
            raise ReviseExhaustedPredPlanError(
                f"任务尚未形成 frozen/exhausted 闭环: {entry.logical_id}={state}"
            )
        if entry.evaluation_key in frozen_keys:
            raise ReviseExhaustedPredPlanError(
                f"exhausted 任务与 frozen binding 冲突: {entry.logical_id}"
            )
        counts = attempt_counts.get(entry.evaluation_key)
        if int(task["attempt_count"]) != 3 or counts != (3, 3):
            raise ReviseExhaustedPredPlanError(
                f"exhausted 任务必须恰有三次 failed attempts: {entry.logical_id}"
            )
        successor = _successor_row(
            predecessor_row,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            prompt_sha256=prompt_sha256,
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

    if expected_exhausted_count is not None and len(successor_bindings) != expected_exhausted_count:
        raise ReviseExhaustedPredPlanError(
            f"exhausted 数量不符: {len(successor_bindings)} != {expected_exhausted_count}"
        )
    logical_ids = [str(row["logical_id"]) for row in output_rows]
    evaluation_keys = [str(row["evaluation_key"]) for row in output_rows]
    if len(set(logical_ids)) != len(output_rows) or len(set(evaluation_keys)) != len(output_rows):
        raise ReviseExhaustedPredPlanError("successor plan 出现重复任务身份")
    output_rows.sort(key=lambda row: (int(row["priority"]), str(row["logical_id"])))
    output_path = Path(output_jsonl).resolve()
    _atomic_write_text(
        output_path,
        "".join(canonical_json(row) + "\n" for row in output_rows),
    )
    try:
        verified_output = load_plan_jsonl(output_path)
    except PlanContractError as exc:
        raise ReviseExhaustedPredPlanError(f"successor plan 回读失败: {exc}") from exc
    if len(verified_output.entries) != len(output_rows):
        raise ReviseExhaustedPredPlanError("successor plan 回读数量漂移")

    report: JsonDict = {
        "status": "ok",
        "model_invoked": False,
        "state_db_mutated": False,
        "predecessor_plan_jsonl": str(predecessor_path),
        "predecessor_plan_sha256": loaded.plan_sha256,
        "state_db": str(state_path),
        "recovery_prompt_path": str(prompt_path),
        "recovery_prompt_version": prompt_path.stem,
        "recovery_prompt_sha256": prompt_sha256,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 exhausted clean pred recovery successor 计划")
    parser.add_argument("--predecessor-plan-jsonl", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--recovery-prompt-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--logical-id-suffix", default="v2")
    parser.add_argument("--expected-task-count", type=int, default=2250)
    parser.add_argument("--expected-exhausted-count", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        revise_exhausted_pred_plan(
            predecessor_plan_jsonl=args.predecessor_plan_jsonl,
            state_db=args.state_db,
            recovery_prompt_path=args.recovery_prompt_path,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            logical_id_suffix=args.logical_id_suffix,
            expected_task_count=args.expected_task_count,
            expected_exhausted_count=args.expected_exhausted_count,
        )
    except ReviseExhaustedPredPlanError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
