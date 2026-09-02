"""从冻结结果与本地 best_history 唯一恢复公式参数。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence


SUPPORTED_CONDITIONS = ("clean", "noise001", "noise005")
TASK_ID_RE = re.compile(
    r"^(?P<slug>[a-z0-9]+)_s(?P<seed>520|521|522)_"
    r"(?P<condition>clean|noise001|noise005)_g(?P<dataset_index>\d{4})$"
)
SAMPLE_FILE_RE = re.compile(r"^best_sample_(?P<order>\d+)\.json$")
PARAMETER_REFERENCE_RE = re.compile(r"\bparams\s*\[")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_INDEX_FIELDS = frozenset(
    {
        "logical_key",
        "noise_tag",
        "algorithm",
        "dataset_id",
        "seed",
        "task_id",
        "local_result_path",
        "local_sha256",
        "status",
        "equation_present",
        "canonical_artifact_present",
        "metrics_match",
    }
)


class FormulaParameterRecoveryError(ValueError):
    """冻结结果、history 或既有 manifest 不符合恢复契约。"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise FormulaParameterRecoveryError(f"{context} 必须是 true/false")


def _require_sha(value: object, *, context: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise FormulaParameterRecoveryError(f"{context} 不是 SHA-256")
    return text


class _CanonicalVariableNames(ast.NodeTransformer):
    def __init__(self, names: Mapping[str, str]) -> None:
        self._names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self._names.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)


def _normalized_skeleton(source: str, *, context: str) -> tuple[str, tuple[int, ...]]:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise FormulaParameterRecoveryError(f"{context} 不是合法 Python 公式") from exc
    functions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1:
        raise FormulaParameterRecoveryError(f"{context} 必须恰好包含一个函数定义")
    function = functions[0]
    if function.args.vararg is not None or function.args.kwarg is not None or function.args.kwonlyargs:
        raise FormulaParameterRecoveryError(f"{context} 不允许可变参数或关键字专用参数")
    positional = [*function.args.posonlyargs, *function.args.args]
    parameter_args = [item for item in positional if item.arg == "params"]
    if len(parameter_args) != 1:
        raise FormulaParameterRecoveryError(f"{context} 必须恰好有一个 params 参数")
    variable_args = [item.arg for item in positional if item.arg != "params"]
    if len(set(variable_args)) != len(variable_args):
        raise FormulaParameterRecoveryError(f"{context} 变量参数重复")
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        raise FormulaParameterRecoveryError(f"{context} 必须恰好包含一个有值 return")
    expression = returns[0].value
    mapping = {name: f"x{index}" for index, name in enumerate(variable_args)}
    normalized = _CanonicalVariableNames(mapping).visit(ast.fix_missing_locations(expression))
    parameter_indices: list[int] = []
    for node in ast.walk(normalized):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "params":
            continue
        slice_node = node.slice
        if (
            isinstance(slice_node, ast.Constant)
            and not isinstance(slice_node.value, bool)
            and isinstance(slice_node.value, int)
        ):
            parameter_index = int(slice_node.value)
        elif (
            isinstance(slice_node, ast.UnaryOp)
            and isinstance(slice_node.op, (ast.USub, ast.UAdd))
            and isinstance(slice_node.operand, ast.Constant)
            and not isinstance(slice_node.operand.value, bool)
            and isinstance(slice_node.operand.value, int)
        ):
            magnitude = int(slice_node.operand.value)
            parameter_index = -magnitude if isinstance(slice_node.op, ast.USub) else magnitude
        else:
            raise FormulaParameterRecoveryError(
                f"{context} 的 params 下标必须是整数常量"
            )
        parameter_indices.append(parameter_index)
    if not parameter_indices:
        raise FormulaParameterRecoveryError(f"{context} 不含 params[...] 引用")
    return ast.dump(normalized, annotate_fields=True, include_attributes=False), tuple(sorted(set(parameter_indices)))


def _load_index_rows(
    path: Path, *, condition: str, expected_missing_count: int | None
) -> list[dict[str, Any]]:
    if condition not in SUPPORTED_CONDITIONS:
        raise FormulaParameterRecoveryError(f"未知 condition: {condition!r}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_INDEX_FIELDS.difference(reader.fieldnames or ()))
        if missing:
            raise FormulaParameterRecoveryError(f"result index 缺少字段: {missing}")
        raw_rows = [dict(row) for row in reader if row.get("noise_tag") == condition]
    targets: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in raw_rows:
        task_id = row["task_id"]
        match = TASK_ID_RE.fullmatch(task_id)
        if match is None or match.group("condition") != condition:
            raise FormulaParameterRecoveryError(f"task_id 与 condition 不一致: {task_id!r}")
        if task_id in seen_tasks:
            raise FormulaParameterRecoveryError(f"result index task_id 重复: {task_id}")
        seen_tasks.add(task_id)
        if int(row["seed"]) != int(match.group("seed")):
            raise FormulaParameterRecoveryError(f"{task_id}: seed 与 task_id 不一致")
        expected_logical_key = (
            f"{row['algorithm']}::{row['dataset_id']}::s{row['seed']}::{condition}"
        )
        if row["logical_key"] != expected_logical_key:
            raise FormulaParameterRecoveryError(f"{task_id}: logical_key 非 canonical")
        if row["status"] != "ok":
            continue
        if not _parse_bool(row["metrics_match"], context=f"{task_id}.metrics_match"):
            raise FormulaParameterRecoveryError(f"{task_id}: result index NMSE 未通过")
        result_path = Path(row["local_result_path"]).resolve()
        expected_suffix = ("results", condition, row["algorithm"], task_id, "result.json")
        if tuple(result_path.parts[-5:]) != expected_suffix:
            raise FormulaParameterRecoveryError(
                f"{task_id}: local_result_path 不符合 results/<condition>/<algorithm>/<task_id> 契约"
            )
        if not result_path.is_file():
            raise FormulaParameterRecoveryError(f"{task_id}: 本地 result.json 不存在")
        result_sha = _sha256_file(result_path)
        if _require_sha(row["local_sha256"], context=f"{task_id}.local_sha256") != result_sha:
            raise FormulaParameterRecoveryError(f"{task_id}: frozen result SHA 漂移")
        try:
            payload_raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormulaParameterRecoveryError(f"{task_id}: result.json 解析失败") from exc
        if not isinstance(payload_raw, Mapping):
            raise FormulaParameterRecoveryError(f"{task_id}: result.json 根必须是对象")
        payload = dict(payload_raw)
        if payload.get("status") != row["status"]:
            raise FormulaParameterRecoveryError(f"{task_id}: result status 与 index 不一致")
        equation = payload.get("equation")
        canonical = payload.get("canonical_artifact")
        equation_present = isinstance(equation, str) and bool(equation.strip())
        canonical_present = isinstance(canonical, Mapping)
        if _parse_bool(row["equation_present"], context=f"{task_id}.equation_present") != equation_present:
            raise FormulaParameterRecoveryError(f"{task_id}: equation_present 与 result 不一致")
        if _parse_bool(
            row["canonical_artifact_present"], context=f"{task_id}.canonical_artifact_present"
        ) != canonical_present:
            raise FormulaParameterRecoveryError(f"{task_id}: canonical_artifact_present 与 result 不一致")
        if canonical_present or not equation_present or PARAMETER_REFERENCE_RE.search(equation) is None:
            continue
        targets.append(
            {
                "task_id": task_id,
                "equation": equation,
                "equation_sha256": _sha256_bytes(equation.encode("utf-8")),
                "frozen_result_sha256": result_sha,
            }
        )
    if expected_missing_count is not None and len(targets) != expected_missing_count:
        raise FormulaParameterRecoveryError(
            f"{condition} 缺 canonical 且含 params 的结果数不符: "
            f"期望 {expected_missing_count}，实际 {len(targets)}"
        )
    return sorted(targets, key=lambda item: item["task_id"])


def _load_candidate(path: Path, *, task_id: str) -> dict[str, Any]:
    file_match = SAMPLE_FILE_RE.fullmatch(path.name)
    if file_match is None:
        raise FormulaParameterRecoveryError(f"{task_id}: history 文件名非法: {path.name}")
    raw_bytes = path.read_bytes()
    try:
        payload_raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormulaParameterRecoveryError(f"{task_id}: history JSON 解析失败: {path}") from exc
    if not isinstance(payload_raw, Mapping):
        raise FormulaParameterRecoveryError(f"{task_id}: history 根必须是对象: {path}")
    payload = dict(payload_raw)
    sample_order = payload.get("sample_order")
    if isinstance(sample_order, bool) or not isinstance(sample_order, int):
        raise FormulaParameterRecoveryError(f"{task_id}: sample_order 非整数: {path}")
    if sample_order != int(file_match.group("order")):
        raise FormulaParameterRecoveryError(f"{task_id}: sample_order 与文件名不一致: {path}")
    function = payload.get("function")
    if not isinstance(function, str) or not function.strip():
        raise FormulaParameterRecoveryError(f"{task_id}: candidate.function 缺失: {path}")
    params = payload.get("params")
    if not isinstance(params, list):
        raise FormulaParameterRecoveryError(f"{task_id}: candidate.params 不是数组: {path}")
    parsed_params: list[float] = []
    for value in params:
        if isinstance(value, bool):
            raise FormulaParameterRecoveryError(f"{task_id}: candidate.params 含布尔值: {path}")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise FormulaParameterRecoveryError(f"{task_id}: candidate.params 含非数值: {path}") from exc
        if not math.isfinite(parsed):
            raise FormulaParameterRecoveryError(f"{task_id}: candidate.params 含非有限值: {path}")
        parsed_params.append(parsed)
    raw_score = payload.get("score")
    if raw_score is None:
        score: float | None = None
    elif isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise FormulaParameterRecoveryError(f"{task_id}: candidate.score 类型非法: {path}")
    else:
        parsed_score = float(raw_score)
        score = parsed_score if math.isfinite(parsed_score) else None
    skeleton, referenced_indices = _normalized_skeleton(function, context=f"{task_id}:{path.name}")
    if any(
        index >= len(parsed_params) or index < -len(parsed_params)
        for index in referenced_indices
    ):
        raise FormulaParameterRecoveryError(f"{task_id}: candidate.params 长度不足: {path}")
    return {
        "path": path.resolve(),
        "sha256": _sha256_bytes(raw_bytes),
        "sample_order": sample_order,
        "score": score,
        "params": parsed_params,
        "skeleton": skeleton,
    }


def _entry_for_target(
    target: Mapping[str, Any], *, history_condition_root: Path
) -> dict[str, Any]:
    task_id = str(target["task_id"])
    frozen_skeleton, _ = _normalized_skeleton(
        str(target["equation"]), context=f"{task_id}.frozen_equation"
    )
    history_dir = history_condition_root / task_id
    discovered_paths = (
        sorted(history_dir.rglob("best_sample_*.json"))
        if history_dir.is_dir()
        else []
    )
    candidate_paths = [
        path for path in discovered_paths if SAMPLE_FILE_RE.fullmatch(path.name) is not None
    ]
    ignored_history_files = [
        {
            "path": str(path.resolve()),
            "reason": "noncanonical_sample_filename",
            "sha256": _sha256_file(path),
        }
        for path in discovered_paths
        if SAMPLE_FILE_RE.fullmatch(path.name) is None
    ]
    candidates = [_load_candidate(path, task_id=task_id) for path in candidate_paths]
    matches = [candidate for candidate in candidates if candidate["skeleton"] == frozen_skeleton]
    base = {
        "task_id": task_id,
        "frozen_result_sha256": target["frozen_result_sha256"],
        "equation_sha256": target["equation_sha256"],
        "ignored_history_files": ignored_history_files,
    }
    if len(matches) == 0:
        return {
            **base,
            "resolution": "unavailable",
            "reason": "no_matching_history_candidate",
            "match_count": 0,
            "history_candidate_count": len(candidates),
        }
    if len(matches) > 1:
        return {
            **base,
            "resolution": "unavailable",
            "reason": "multiple_matching_history_candidates",
            "match_count": len(matches),
            "history_candidate_count": len(candidates),
            "matching_sample_orders": sorted(candidate["sample_order"] for candidate in matches),
            "matching_candidates": [
                {
                    "candidate_path": str(candidate["path"]),
                    "candidate_sha256": candidate["sha256"],
                    "sample_order": candidate["sample_order"],
                }
                for candidate in sorted(matches, key=lambda item: item["sample_order"])
            ],
        }
    match = matches[0]
    params = list(match["params"])
    return {
        **base,
        "resolution": "recovered_params",
        "params": params,
        "source_evidence": {
            "candidate_path": str(match["path"]),
            "candidate_sha256": match["sha256"],
            "params_sha256": _sha256_bytes(_canonical_json(params).encode("utf-8")),
            "params_count": len(params),
            "sample_order": match["sample_order"],
            "score": match["score"],
            "canonical_match_method": "extract_return_ast+canonicalize_positional_variable_names_equal",
            "same_skeleton_param_group_count": 1,
            "best_choice_basis": "best_history 中唯一规范化 AST 与冻结最终公式骨架一致的候选。",
            "sha_match_verified": True,
        },
    }


def _write_json_without_drift(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FormulaParameterRecoveryError(f"拒绝覆盖漂移 manifest: {path}")
        return
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != data:
            raise FormulaParameterRecoveryError(f"并发生成了漂移 manifest: {path}")
    finally:
        tmp_path.unlink(missing_ok=True)


def build_formula_recovery_manifest(
    *,
    result_index_csv: Path,
    recovery_history_root: Path,
    condition: str,
    output_json: Path,
    expected_missing_count: int | None,
) -> dict[str, Any]:
    """只对唯一骨架匹配恢复参数，其余目标显式标记 unavailable。"""
    result_index_csv = result_index_csv.resolve()
    recovery_history_root = recovery_history_root.resolve()
    targets = _load_index_rows(
        result_index_csv,
        condition=condition,
        expected_missing_count=expected_missing_count,
    )
    entries = [
        _entry_for_target(target, history_condition_root=recovery_history_root / condition)
        for target in targets
    ]
    payload = {
        "schema_version": "formula_recovery.v1",
        "condition": condition,
        "entries": entries,
    }
    _write_json_without_drift(output_json.resolve(), payload)
    resolution_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for entry in entries:
        resolution = str(entry["resolution"])
        resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        if resolution == "unavailable":
            reason = str(entry["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "condition": condition,
        "result_index_csv": str(result_index_csv),
        "result_index_sha256": _sha256_file(result_index_csv),
        "recovery_history_root": str(recovery_history_root),
        "output_json": str(output_json.resolve()),
        "output_sha256": _sha256_file(output_json.resolve()),
        "entry_count": len(entries),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "unavailable_reason_counts": dict(sorted(reason_counts.items())),
        "status": "ok",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage5 唯一骨架公式参数恢复")
    parser.add_argument("--result-index-csv", type=Path, required=True)
    parser.add_argument("--recovery-history-root", type=Path, required=True)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-missing-count", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_formula_recovery_manifest(
        result_index_csv=args.result_index_csv,
        recovery_history_root=args.recovery_history_root,
        condition=args.condition,
        output_json=args.output_json,
        expected_missing_count=args.expected_missing_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
