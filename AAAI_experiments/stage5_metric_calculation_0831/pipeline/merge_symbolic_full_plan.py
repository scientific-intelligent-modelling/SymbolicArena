"""将 callable symbolic plan 与 planned-non-applicable 索引合并为 full plan。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.run_claude_plan import (
    PlanContractError,
    load_plan_jsonl,
)


class MergeSymbolicFullPlanError(RuntimeError):
    """输入计划、身份集合或计数不满足 full-plan 合并契约。"""


def _require_nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MergeSymbolicFullPlanError(f"{context} 必须是非空字符串")
    return value


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MergeSymbolicFullPlanError(f"{context} 必须是整数")
    return value


def _read_jsonl_objects(
    path: Path, *, label: str
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise MergeSymbolicFullPlanError(
                        f"{label} {path}:{line_number} JSON 解析失败: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise MergeSymbolicFullPlanError(
                        f"{label} {path}:{line_number} 必须是 JSON object"
                    )
                rows.append(payload)
    except OSError as exc:
        raise MergeSymbolicFullPlanError(f"无法读取 {label} {path}: {exc}") from exc
    return rows, digest.hexdigest()


def _validate_callable_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], str, set[str], set[str]]:
    try:
        loaded = load_plan_jsonl(path)
    except PlanContractError as exc:
        raise MergeSymbolicFullPlanError(f"callable plan 契约无效: {exc}") from exc
    expected_identities = [
        (entry.evaluation_key, entry.logical_id) for entry in loaded.entries
    ]
    callable_sha256 = loaded.plan_sha256
    rows, reread_sha256 = _read_jsonl_objects(path, label="callable plan")
    if reread_sha256 != callable_sha256:
        raise MergeSymbolicFullPlanError("callable plan 在契约校验期间发生内容漂移")
    actual_identities = [
        (
            _require_nonempty_string(
                row.get("evaluation_key"),
                context=f"callable plan line {line_number}.evaluation_key",
            ),
            _require_nonempty_string(
                row.get("logical_id"),
                context=f"callable plan line {line_number}.logical_id",
            ),
        )
        for line_number, row in enumerate(rows, start=1)
    ]
    if actual_identities != expected_identities:
        raise MergeSymbolicFullPlanError("callable plan 二次读取后的身份序列漂移")
    return (
        rows,
        callable_sha256,
        {identity[0] for identity in actual_identities},
        {identity[1] for identity in actual_identities},
    )


def _validate_non_applicable_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], str, set[str], set[str]]:
    rows, non_applicable_sha256 = _read_jsonl_objects(
        path, label="non-applicable index"
    )
    evaluation_keys: set[str] = set()
    logical_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        context = f"non-applicable index line {line_number}"
        if row.get("status") != "planned_non_applicable":
            raise MergeSymbolicFullPlanError(
                f"{context}.status 必须为 planned_non_applicable"
            )
        evaluation_key = _require_nonempty_string(
            row.get("evaluation_key"), context=f"{context}.evaluation_key"
        )
        logical_id = _require_nonempty_string(
            row.get("logical_id"), context=f"{context}.logical_id"
        )
        priority = _require_int(row.get("priority"), context=f"{context}.priority")
        task_spec = row.get("task_spec")
        if not isinstance(task_spec, Mapping):
            raise MergeSymbolicFullPlanError(f"{context}.task_spec 必须是 JSON object")
        if task_spec.get("evaluation_key") != evaluation_key:
            raise MergeSymbolicFullPlanError(
                f"{context}.task_spec.evaluation_key 与顶层字段不一致"
            )
        if task_spec.get("logical_id") != logical_id:
            raise MergeSymbolicFullPlanError(
                f"{context}.task_spec.logical_id 与顶层字段不一致"
            )
        if task_spec.get("priority") != priority:
            raise MergeSymbolicFullPlanError(
                f"{context}.task_spec.priority 与顶层字段不一致"
            )
        if evaluation_key in evaluation_keys:
            raise MergeSymbolicFullPlanError(
                f"non-applicable index 存在重复 evaluation_key: {evaluation_key}"
            )
        if logical_id in logical_ids:
            raise MergeSymbolicFullPlanError(
                f"non-applicable index 存在重复 logical_id: {logical_id}"
            )
        evaluation_keys.add(evaluation_key)
        logical_ids.add(logical_id)
    return rows, non_applicable_sha256, evaluation_keys, logical_ids


def _validate_expected_count(*, label: str, expected: int | None, actual: int) -> None:
    if expected is None:
        return
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise MergeSymbolicFullPlanError(f"{label} expected count 必须是非负整数")
    if expected != actual:
        raise MergeSymbolicFullPlanError(
            f"{label} 计数漂移: 期望 {expected}，实际 {actual}"
        )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        for row in rows:
            line = (canonical_json(dict(row)) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
    return digest.hexdigest()


def _atomic_write_report(path: Path, report: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def merge_symbolic_full_plan(
    *,
    callable_plan_jsonl: str | Path,
    non_applicable_index_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path,
    expected_total_count: int | None = None,
    expected_callable_count: int | None = None,
    expected_non_applicable_count: int | None = None,
) -> dict[str, Any]:
    """严格合并两类计划行，并写出可审计、确定性的 full plan。"""

    callable_path = Path(callable_plan_jsonl).resolve()
    non_applicable_path = Path(non_applicable_index_jsonl).resolve()
    output_path = Path(output_jsonl).resolve()
    report_path = Path(report_json).resolve()
    if output_path == report_path:
        raise MergeSymbolicFullPlanError("output_jsonl 与 report_json 不能是同一路径")
    input_paths = {callable_path, non_applicable_path}
    if output_path in input_paths or report_path in input_paths:
        raise MergeSymbolicFullPlanError("输出路径不能覆盖输入文件")

    callable_rows, callable_sha256, callable_keys, callable_logical_ids = (
        _validate_callable_rows(callable_path)
    )
    (
        non_applicable_rows,
        non_applicable_sha256,
        non_applicable_keys,
        non_applicable_logical_ids,
    ) = _validate_non_applicable_rows(non_applicable_path)

    overlapping_keys = callable_keys & non_applicable_keys
    if overlapping_keys:
        raise MergeSymbolicFullPlanError(
            "callable 与 non-applicable evaluation_key 重叠: "
            f"{sorted(overlapping_keys)[0]}"
        )
    overlapping_logical_ids = callable_logical_ids & non_applicable_logical_ids
    if overlapping_logical_ids:
        raise MergeSymbolicFullPlanError(
            "callable 与 non-applicable logical_id 重叠: "
            f"{sorted(overlapping_logical_ids)[0]}"
        )

    callable_count = len(callable_rows)
    non_applicable_count = len(non_applicable_rows)
    total_count = callable_count + non_applicable_count
    _validate_expected_count(
        label="callable", expected=expected_callable_count, actual=callable_count
    )
    _validate_expected_count(
        label="non-applicable",
        expected=expected_non_applicable_count,
        actual=non_applicable_count,
    )
    _validate_expected_count(
        label="total", expected=expected_total_count, actual=total_count
    )

    rows = [*callable_rows, *non_applicable_rows]
    rows.sort(
        key=lambda row: (
            _require_int(row.get("priority"), context="priority"),
            str(row["logical_id"]),
        )
    )
    full_plan_sha256 = _atomic_write_jsonl(output_path, rows)
    report: dict[str, Any] = {
        "status": "ok",
        "model_invoked": False,
        "inputs": {
            "callable_plan_jsonl": str(callable_path),
            "callable_plan_sha256": callable_sha256,
            "non_applicable_index_jsonl": str(non_applicable_path),
            "non_applicable_index_sha256": non_applicable_sha256,
        },
        "outputs": {
            "full_plan_jsonl": str(output_path),
            "full_plan_sha256": full_plan_sha256,
            "report_json": str(report_path),
        },
        "counts": {
            "callable_count": callable_count,
            "non_applicable_count": non_applicable_count,
            "total_count": total_count,
        },
    }
    _atomic_write_report(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并 callable symbolic plan 与 planned-non-applicable 索引"
    )
    parser.add_argument("--callable-plan-jsonl", type=Path, required=True)
    parser.add_argument("--non-applicable-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-total-count", type=int)
    parser.add_argument("--expected-callable-count", type=int)
    parser.add_argument("--expected-non-applicable-count", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = merge_symbolic_full_plan(
            callable_plan_jsonl=args.callable_plan_jsonl,
            non_applicable_index_jsonl=args.non_applicable_index_jsonl,
            output_jsonl=args.output_jsonl,
            report_json=args.report_json,
            expected_total_count=args.expected_total_count,
            expected_callable_count=args.expected_callable_count,
            expected_non_applicable_count=args.expected_non_applicable_count,
        )
    except MergeSymbolicFullPlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["outputs"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
