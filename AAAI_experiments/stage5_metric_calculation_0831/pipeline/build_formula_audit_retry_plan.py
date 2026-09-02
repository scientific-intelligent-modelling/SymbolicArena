"""为二审格式失败的 exhausted 任务生成单次 retry1 计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claude_contract import canonical_json, evaluation_key
from .run_claude_plan import load_plan_jsonl
from .state import TaskSpec


JsonDict = dict[str, Any]


class FormulaAuditRetryPlanError(RuntimeError):
    """二审计划、状态库或 retry 映射不满足冻结契约。"""


def _stage_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormulaAuditRetryPlanError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaAuditRetryPlanError(f"{context} 必须是非空字符串")
    return value.strip()


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        raise FormulaAuditRetryPlanError(f"JSONL 不存在: {path}")
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FormulaAuditRetryPlanError(f"{path}:{line_number} 不得为空行")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormulaAuditRetryPlanError(
                    f"{path}:{line_number} JSON 非法"
                ) from exc
            rows.append(dict(_require_mapping(payload, context=f"{path}:{line_number}")))
    if not rows:
        raise FormulaAuditRetryPlanError("round2 plan 不能为空")
    return rows


def _unique_by(
    rows: Sequence[Mapping[str, Any]], key: str, *, context: str
) -> dict[str, JsonDict]:
    result: dict[str, JsonDict] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise FormulaAuditRetryPlanError(f"{context} 出现重复 {key}: {value}")
        result[value] = dict(row)
    return result


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_terminal_states(
    *, state_db: Path, plan_by_key: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    if not state_db.is_file():
        raise FormulaAuditRetryPlanError(f"round2 state DB 不存在: {state_db}")
    try:
        connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """SELECT t.evaluation_key, t.logical_id, t.spec_json, t.state,
                              f.evaluation_key AS frozen_binding
                       FROM tasks t
                       LEFT JOIN frozen_results f
                         ON f.evaluation_key=t.evaluation_key"""
                )
            ]
    except sqlite3.Error as exc:
        raise FormulaAuditRetryPlanError(f"round2 state DB 读取失败: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    state_by_key = {str(row["evaluation_key"]): row for row in rows}
    if len(state_by_key) != len(rows):
        raise FormulaAuditRetryPlanError("round2 state DB 存在重复 evaluation_key")
    if set(state_by_key) != set(plan_by_key):
        raise FormulaAuditRetryPlanError("round2 state DB 与 plan 任务集合不一致")
    states: dict[str, str] = {}
    for key, plan_row in plan_by_key.items():
        state_row = state_by_key[key]
        try:
            stored_spec = json.loads(str(state_row["spec_json"]))
        except json.JSONDecodeError as exc:
            raise FormulaAuditRetryPlanError(f"{key} state spec_json 非法") from exc
        if stored_spec != plan_row.get("task_spec"):
            raise FormulaAuditRetryPlanError(f"{key} state task_spec 与 plan 不一致")
        if state_row.get("logical_id") != plan_row.get("logical_id"):
            raise FormulaAuditRetryPlanError(f"{key} state logical_id 与 plan 不一致")
        state = _require_string(state_row.get("state"), context=f"{key}.state")
        if state not in {"frozen", "exhausted"}:
            raise FormulaAuditRetryPlanError(
                f"round2 尚未完全收口，存在 {state} 任务: {key}"
            )
        if (state_row.get("frozen_binding") is not None) != (state == "frozen"):
            raise FormulaAuditRetryPlanError(f"{key} state 与 frozen binding 不一致")
        states[key] = state
    return states


def _retry_row(
    *,
    predecessor: Mapping[str, Any],
    prompt_path: Path,
    prompt_template: str,
    prompt_sha256: str,
) -> JsonDict:
    predecessor_logical_id = _require_string(
        predecessor.get("logical_id"), context="round2.logical_id"
    )
    if not predecessor_logical_id.endswith("::v2"):
        raise FormulaAuditRetryPlanError(
            f"round2 logical_id 必须以 ::v2 结尾: {predecessor_logical_id}"
        )
    logical_id = f"{predecessor_logical_id}::retry1"
    request = dict(
        _require_mapping(
            predecessor.get("request"), context=f"{predecessor_logical_id}.request"
        )
    )
    if request.get("review_round") != 2:
        raise FormulaAuditRetryPlanError(
            f"round2 request.review_round 必须为 2: {predecessor_logical_id}"
        )
    evidence_hash = _require_string(
        request.get("evidence_hash"), context=f"{predecessor_logical_id}.evidence_hash"
    )
    schema_sha256 = _require_string(
        predecessor.get("schema_sha256"), context=f"{predecessor_logical_id}.schema_sha256"
    )
    schema_version = _require_string(
        predecessor.get("schema_version"), context=f"{predecessor_logical_id}.schema_version"
    )
    normalized_input = {
        "request": request,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_json(normalized_input)
    task_key = evaluation_key(
        task_type="formula_audit",
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=evidence_hash,
    )
    dependencies = predecessor.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise FormulaAuditRetryPlanError(f"{predecessor_logical_id}.dependencies 非法")
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type="formula_audit",
        condition=_require_string(
            predecessor.get("condition"), context=f"{predecessor_logical_id}.condition"
        ),
        priority=int(predecessor["priority"]),
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=schema_version,
        dependencies=tuple(dependencies),
    )
    return {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": "formula_audit",
        "task_kind": "formula_audit",
        "condition": spec.condition,
        "priority": spec.priority,
        "input_hash": input_hash,
        "prompt_version": prompt_path.stem,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "dependencies": list(dependencies),
        "prompt_path": str(prompt_path),
        "schema_path": predecessor["schema_path"],
        "prompt_template": prompt_template,
        "schema_content": predecessor["schema_content"],
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
        "retry_predecessor_evaluation_key": predecessor["evaluation_key"],
    }


def build_formula_audit_retry_plan(
    *,
    round2_plan_jsonl: str | Path,
    round2_state_db: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
) -> JsonDict:
    """只为已 exhausted 的 round2 任务生成一次 v3 格式重试。"""

    plan_path = Path(round2_plan_jsonl).resolve()
    rows = _read_jsonl(plan_path)
    loaded = load_plan_jsonl(plan_path)
    plan_by_key = _unique_by(rows, "evaluation_key", context="round2_plan")
    if set(plan_by_key) != {entry.evaluation_key for entry in loaded.entries}:
        raise FormulaAuditRetryPlanError("round2 plan 解析身份漂移")
    for row in rows:
        if row.get("task_type") != "formula_audit" or row.get("task_kind") != "formula_audit":
            raise FormulaAuditRetryPlanError("round2 plan 含非 formula_audit 任务")
    state_path = Path(round2_state_db).resolve()
    states = _load_terminal_states(state_db=state_path, plan_by_key=plan_by_key)

    prompt_path = (_stage_root() / "config/prompts/formula_audit.v3.txt").resolve()
    prompt_bytes = prompt_path.read_bytes()
    prompt_template = prompt_bytes.decode("utf-8")
    prompt_sha256 = _sha256_bytes(prompt_bytes)
    retry_rows = [
        _retry_row(
            predecessor=plan_by_key[key],
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            prompt_sha256=prompt_sha256,
        )
        for key, state in sorted(
            states.items(), key=lambda item: str(plan_by_key[item[0]]["logical_id"])
        )
        if state == "exhausted"
    ]
    output_path = Path(output_jsonl).resolve()
    report_path = Path(report_json).resolve()
    _atomic_write_jsonl(output_path, retry_rows)
    report: JsonDict = {
        "status": "ok",
        "model_invoked": False,
        "inputs": {
            "round2_plan_jsonl": str(plan_path),
            "round2_plan_sha256": _sha256_file(plan_path),
            "round2_state_db": str(state_path),
        },
        "outputs": {
            "retry_plan_jsonl": str(output_path),
            "retry_plan_sha256": _sha256_file(output_path),
            "report_json": str(report_path),
        },
        "counts": {
            "round2_task_count": len(rows),
            "retry_task_count": len(retry_rows),
            "round2_state_distribution": dict(sorted(Counter(states.values()).items())),
        },
        "execution_contract": {
            "retry_label": "retry1",
            "max_attempts_per_task": 1,
            "request_review_round": 2,
            "semantic_request_unchanged": True,
            "prompt_version": prompt_path.stem,
        },
    }
    _atomic_write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建公式审计 round2 exhausted 单次重试计划")
    parser.add_argument("--round2-plan-jsonl", type=Path, required=True)
    parser.add_argument("--round2-state-db", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_formula_audit_retry_plan(
            round2_plan_jsonl=args.round2_plan_jsonl,
            round2_state_db=args.round2_state_db,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
        )
    except (FormulaAuditRetryPlanError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
