"""准备 Core-50 最终发布的冻结输入与最小 Opus5 刷新计划。

本模块只整理输入，不调用模型、不重算训练，也不修改任何历史冻结目录。
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .clean_task_builder import (
    PRED_PRIORITY,
    CleanTaskBuilderError,
    _build_pred_task,
    _build_task_definition,
    _build_gt_task,
    _load_prompt_schema,
    extract_expression_body,
    load_dataset_probes,
    map_indexed_variables,
)
from .performance_replay import _corrected_artifact, load_formula_recovery_manifest
from .run_claude_plan import PlanContractError, load_plan_jsonl
from .state import TaskSpec


for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")


STAGE_ROOT_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831")
FINAL_WORK_RELATIVE = STAGE_ROOT_RELATIVE / "work/final_release_20260913"
DEFAULT_BASE = FINAL_WORK_RELATIVE / "base_candidate_v1"
DEFAULT_SYMBOLFIT = FINAL_WORK_RELATIVE / "symbolfit_noise_latest"
DEFAULT_OUTPUT = FINAL_WORK_RELATIVE / "release_v2/inputs"
DEFAULT_GT_EXTRACT = STAGE_ROOT_RELATIVE / "reports/ground_truth_extract.jsonl"
DEFAULT_PROBES = STAGE_ROOT_RELATIVE / "reports/dataset_probes.jsonl"
DEFAULT_GT_VIEW = (
    STAGE_ROOT_RELATIVE
    / "audits/formula_quality_1000_0903_v2/corrections_v2/views/gt_effective.jsonl"
)
DEFAULT_PRED_VIEW = (
    STAGE_ROOT_RELATIVE
    / "audits/formula_quality_1000_0903_v2/corrections_v2/views/pred_effective.jsonl"
)
DEFAULT_CLEAN_AUDITED_PLAN = STAGE_ROOT_RELATIVE / "reports/clean_pred_simplify_tasks_active_v5.jsonl"
DEFAULT_IDENTITY_PROMPT = STAGE_ROOT_RELATIVE / "config/prompts/simplify.identity.v1.txt"
DEFAULT_RAW_EXPORT = STAGE_ROOT_RELATIVE / "exports/Core50_raw_results_20260909/run_level"
DEFAULT_FORMULA_RECOVERY = STAGE_ROOT_RELATIVE / "manifests/formula_recovery.v1.json"
CONDITIONS = ("clean", "noise001", "noise005")
FORCED_REBUILD_ALGORITHMS = frozenset({"qlattice", "drsr", "llmsr", "imcts"})
GT_IDENTITY_DATASET = "feynman-bonus.20"
VERSION_SUFFIX_RE = re.compile(r"v([1-9]\d*)")


class FinalReleasePreparationError(ValueError):
    """最终发布输入不完整、漂移或绑定不一致。"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise FinalReleasePreparationError(f"{path}:{line_number} 不是 JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    opener = gzip.open if gzip_output else open
    with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unwrap_frozen_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    result = row.get("result")
    if not isinstance(result, Mapping):
        raise FinalReleasePreparationError("冻结行缺少 result object")
    raw_text = result.get("raw_text")
    expected_sha = result.get("sha256")
    if not isinstance(raw_text, str) or not isinstance(expected_sha, str):
        raise FinalReleasePreparationError("冻结行缺少 result.raw_text/result.sha256")
    actual_sha = _sha256_text(raw_text)
    if actual_sha != expected_sha:
        raise FinalReleasePreparationError("冻结行 raw_text SHA-256 漂移")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise FinalReleasePreparationError("冻结 result.raw_text 不是 JSON object")
    return payload, actual_sha


def _task_id(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    task_id = source.get("task_id") if isinstance(source, Mapping) else None
    if not isinstance(task_id, str) or not task_id:
        raise FinalReleasePreparationError("冻结行缺少 source.task_id")
    return task_id


def _algorithm_slug(value: object) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def _load_symbolfit_replacements(symbolfit_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = symbolfit_root / "manifests/symbolfit_noise_latest_runs.csv"
    replacements: dict[str, dict[str, Any]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            task_id = record["task_id"]
            condition = record["condition"]
            if condition not in {"noise001", "noise005"}:
                raise FinalReleasePreparationError(f"{task_id}: SymbolFit replacement condition 非法")
            if record.get("validation_status") != "passed":
                raise FinalReleasePreparationError(f"{task_id}: SymbolFit replacement 未通过验收")
            host = record["host"]
            result_path = symbolfit_root / "hosts" / host / record["frozen_result_path"]
            raw_text = result_path.read_text(encoding="utf-8")
            raw_sha = _sha256_text(raw_text)
            if raw_sha != record["frozen_result_sha256"]:
                raise FinalReleasePreparationError(f"{task_id}: SymbolFit result SHA 漂移")
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise FinalReleasePreparationError(f"{task_id}: SymbolFit result 不是 object")
            source = {
                "algorithm": "SymbolFit",
                "batch": "symbolfit_noise_internal_progress_v1_20260910_full",
                "dataset_id": record["dataset_id"],
                "host": host,
                "noise_tag": condition,
                "path": record["source_result_path"],
                "seed": record["seed"],
                "task_id": task_id,
            }
            source["source_row_sha256"] = _sha256_json(source)
            replacements[task_id] = {
                "result": {"raw_text": raw_text, "sha256": raw_sha},
                "source": source,
            }
    if len(replacements) != 300:
        raise FinalReleasePreparationError(
            f"SymbolFit replacement 必须为 300 条，实际 {len(replacements)}"
        )
    return replacements


def merge_final_rows(
    base_rows: Sequence[Mapping[str, Any]],
    replacements: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    replacement_count = 0
    for raw_row in base_rows:
        row = dict(raw_row)
        task_id = _task_id(row)
        if task_id in seen:
            raise FinalReleasePreparationError(f"{condition}: 重复 task_id: {task_id}")
        seen.add(task_id)
        replacement = replacements.get(task_id)
        if replacement is not None:
            row = dict(replacement)
            replacement_count += 1
        output.append(row)
    missing = set(replacements) - seen
    condition_missing = sorted(task for task in missing if f"_{condition}_" in task)
    if condition_missing:
        raise FinalReleasePreparationError(
            f"{condition}: replacement 未命中 base: {condition_missing[:3]}"
        )
    if len(output) != 2250:
        raise FinalReleasePreparationError(f"{condition}: final rows 不是 2250 条")
    expected = 0 if condition == "clean" else 150
    if replacement_count != expected:
        raise FinalReleasePreparationError(
            f"{condition}: replacement 期望 {expected}，实际 {replacement_count}"
        )
    return output, replacement_count


class _DropNumpyPrefix(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in {"np", "numpy", "math"}:
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node


def expression_fingerprint(expression: str) -> str:
    """只忽略格式和 np/math 限定，不把变量置换或代数近似视为相同。"""

    try:
        body = extract_expression_body(expression)
        tree = ast.parse(body, mode="eval")
        tree = ast.fix_missing_locations(_DropNumpyPrefix().visit(tree))
        return _sha256_text(ast.dump(tree, annotate_fields=True, include_attributes=False))
    except (SyntaxError, ValueError):
        return _sha256_text("".join(expression.split()))


def _artifact_expression(artifact: Mapping[str, Any], feature_names: Sequence[str]) -> str:
    expression = ""
    for name in (
        "instantiated_expression",
        "normalized_expression",
        "return_expression_source",
        "raw_equation",
    ):
        value = artifact.get(name)
        if isinstance(value, str) and value.strip():
            expression = extract_expression_body(value.strip())
            break
    if not expression:
        raise FinalReleasePreparationError("canonical artifact 缺少语义表达式")
    mapped, _ = map_indexed_variables(expression, feature_names)
    return mapped


def _load_old_opus(base: Path) -> dict[str, dict[str, Any]]:
    by_logical_id: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = base / f"results/{condition}/opus5_simplification.jsonl"
        for row in _read_jsonl(path):
            logical_id = row.get("logical_id")
            if isinstance(logical_id, str):
                canonical_id = canonical_pred_logical_id(logical_id)
                if canonical_id in by_logical_id:
                    previous = by_logical_id[canonical_id]
                    raise FinalReleasePreparationError(
                        "旧 Opus compact 出现同一预测身份的多条记录，拒绝按版本或时间猜选: "
                        f"{previous.get('logical_id')!r}, {logical_id!r}"
                    )
                by_logical_id[canonical_id] = row
    return by_logical_id


def canonical_pred_logical_id(logical_id: str) -> str:
    """去掉 pred logical id 的版本后缀，保留唯一算法-任务-种子-条件身份。"""

    parts = logical_id.split("::")
    if len(parts) < 5 or parts[0] != "pred_simplify":
        raise FinalReleasePreparationError(f"非法 pred logical_id: {logical_id!r}")
    canonical = parts[:5]
    if len(parts) > 5 and any(not suffix for suffix in parts[5:]):
        raise FinalReleasePreparationError(f"pred logical_id 含空版本段: {logical_id!r}")
    return "::".join(canonical)


def _load_numeric_artifacts(repo_root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for condition in CONDITIONS:
        path = repo_root / DEFAULT_RAW_EXPORT / condition / "numeric_run_metrics.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                artifact_sha = row.get("canonical_artifact_sha256")
                if artifact_sha:
                    output[row["task_id"]] = artifact_sha
    return output


def _logical_id(source: Mapping[str, Any], condition: str) -> str:
    task_id = str(source["task_id"])
    parts = task_id.split("_")
    algorithm = parts[0].lower()
    seed = int(source["seed"])
    dataset_index = task_id.rsplit("_g", 1)[1]
    return f"pred_simplify::{algorithm}::g{dataset_index}::s{seed}::{condition}"


def build_semantic_row(
    row: Mapping[str, Any],
    *,
    condition: str,
    old_opus: Mapping[str, Mapping[str, Any]],
    old_numeric_artifacts: Mapping[str, str],
    recovery_manifest: Any | None = None,
) -> dict[str, Any]:
    payload, raw_sha = _unwrap_frozen_row(row)
    source = row["source"]
    assert isinstance(source, Mapping)
    task_id = _task_id(row)
    algorithm = str(source.get("algorithm") or payload.get("tool") or "")
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(item, str) for item in feature_names):
        raise FinalReleasePreparationError(f"{task_id}: feature_names 非法")
    artifact, rebuilt = _corrected_artifact(
        payload,
        algorithm=algorithm,
        expected_n_features=len(feature_names),
        recovery_manifest=recovery_manifest,
        task_id=task_id,
        condition=condition,
        result_sha256=raw_sha,
    )
    artifact_sha = _sha256_json(artifact)
    semantic_expression = _artifact_expression(artifact, feature_names)
    semantic_sha = _sha256_text(semantic_expression)
    logical_id = _logical_id(source, condition)
    old = old_opus.get(logical_id)
    old_expression = old.get("input_expression") if isinstance(old, Mapping) else None
    same_binding = isinstance(old_expression, str) and (
        expression_fingerprint(old_expression) == expression_fingerprint(semantic_expression)
    )
    if same_binding:
        refresh_reason = None
    elif _algorithm_slug(algorithm) == "symbolfit" and condition != "clean":
        refresh_reason = "authorized_symbolfit_final_replacement"
    elif old is None:
        refresh_reason = "no_reusable_opus_simplification"
    else:
        refresh_reason = "corrected_semantic_ast_differs_from_old_opus_request"
    old_numeric_sha = old_numeric_artifacts.get(task_id)
    return {
        "schema_version": "core50_final_semantic_run.v2",
        "logical_key": f"{algorithm}::{source.get('dataset_id')}::s{source.get('seed')}::{condition}",
        "logical_id": logical_id,
        "task_id": task_id,
        "condition": condition,
        "algorithm": algorithm,
        "dataset_id": source.get("dataset_id"),
        "seed": int(source["seed"]),
        "source_result_sha256": raw_sha,
        "raw_equation": payload.get("equation"),
        "raw_equation_sha256": _sha256_text(payload["equation"])
        if isinstance(payload.get("equation"), str)
        else None,
        "feature_names": feature_names,
        "canonical_artifact": artifact,
        "canonical_artifact_sha256": artifact_sha,
        "canonical_artifact_rebuilt_from_raw": rebuilt,
        "effective_raw_semantic_expression": semantic_expression,
        "effective_raw_semantic_expression_sha256": semantic_sha,
        "old_opus_evaluation_key": old.get("evaluation_key") if isinstance(old, Mapping) else None,
        "old_opus_logical_id": old.get("logical_id") if isinstance(old, Mapping) else None,
        "old_opus_request_expression": old_expression,
        "old_opus_request_expression_sha256": _sha256_text(old_expression)
        if isinstance(old_expression, str)
        else None,
        "old_opus_binding_ast_equivalent": same_binding,
        "requires_pred_simplify_refresh": not same_binding,
        "pred_simplify_refresh_reason": refresh_reason,
        "old_numeric_canonical_artifact_sha256": old_numeric_sha,
        "numeric_artifact_binding_unchanged": old_numeric_sha == artifact_sha,
        "numeric_requires_replay": old_numeric_sha != artifact_sha,
    }


def _corrected_freeze_row(
    row: Mapping[str, Any], semantic_row: Mapping[str, Any]
) -> dict[str, Any]:
    """构造只供新任务规划使用的 corrected payload，保留原始 SHA 反向绑定。"""

    payload, original_sha = _unwrap_frozen_row(row)
    corrected_payload = dict(payload)
    corrected_payload["canonical_artifact"] = semantic_row["canonical_artifact"]
    corrected_payload["final_release_source_binding"] = {
        "original_result_sha256": original_sha,
        "corrected_canonical_artifact_sha256": semantic_row["canonical_artifact_sha256"],
        "effective_raw_semantic_expression_sha256": semantic_row[
            "effective_raw_semantic_expression_sha256"
        ],
    }
    raw_text = json.dumps(corrected_payload, ensure_ascii=False, sort_keys=True)
    source = dict(row["source"])
    source["original_source_row_sha256"] = source.get("source_row_sha256")
    source["source_row_sha256"] = _sha256_json(source)
    return {
        "result": {"raw_text": raw_text, "sha256": _sha256_text(raw_text)},
        "source": source,
    }


def build_incremental_pred_plans(
    *,
    repo_root: Path,
    merged_by_task: Mapping[str, Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    identity_logical_ids: set[str],
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    contract = _load_prompt_schema(repo_root)
    probes, _ = load_dataset_probes(repo_root / DEFAULT_PROBES)
    gt_rows = _read_jsonl(repo_root / DEFAULT_GT_EXTRACT)
    gt_variables = {str(row["dataset_id"]): list(row["ordered_variables"]) for row in gt_rows}
    gt_targets = {str(row["dataset_id"]): str(row["target"]) for row in gt_rows}
    plans = {condition: [] for condition in CONDITIONS}
    unresolved: list[dict[str, Any]] = []
    for semantic in semantic_rows:
        if not semantic["requires_pred_simplify_refresh"]:
            continue
        if semantic["logical_id"] in identity_logical_ids:
            continue
        task_id = str(semantic["task_id"])
        condition = str(semantic["condition"])
        freeze_row = _corrected_freeze_row(merged_by_task[task_id], semantic)
        try:
            task, no_call = _build_pred_task(
                freeze_row,
                contract=contract,
                ground_truth_variables=gt_variables,
                ground_truth_targets=gt_targets,
                recovery_entries={},
                dataset_probes=probes,
                condition=condition,
            )
        except (CleanTaskBuilderError, ValueError) as exc:
            unresolved.append(
                {
                    "task_id": task_id,
                    "logical_id": semantic["logical_id"],
                    "condition": condition,
                    "reason": f"pred_plan_build_failed:{exc}",
                }
            )
            continue
        if task is None:
            unresolved.append(
                {
                    "task_id": task_id,
                    "logical_id": semantic["logical_id"],
                    "condition": condition,
                    "reason": f"planned_no_call:{no_call.get('reason') if no_call else 'unknown'}",
                }
            )
            continue
        request = dict(task.request)
        ast_source = dict(request.get("ast_source_evidence") or {})
        derived_sha = ast_source.get("result_raw_sha256")
        ast_source["result_raw_sha256"] = semantic["source_result_sha256"]
        ast_source["derived_corrected_payload_sha256"] = derived_sha
        ast_source["corrected_canonical_artifact_sha256"] = semantic[
            "canonical_artifact_sha256"
        ]
        ast_source["effective_raw_semantic_expression_sha256"] = semantic[
            "effective_raw_semantic_expression_sha256"
        ]
        request["ast_source_evidence"] = ast_source
        request["evidence_hash"] = _sha256_json(
            {
                "source_result_sha256": semantic["source_result_sha256"],
                "corrected_canonical_artifact_sha256": semantic[
                    "canonical_artifact_sha256"
                ],
                "effective_raw_semantic_expression_sha256": semantic[
                    "effective_raw_semantic_expression_sha256"
                ],
                "deterministic_evidence": request.get("deterministic_evidence"),
                "ast_source_evidence": ast_source,
            }
        )
        task = _build_task_definition(
            logical_id=task.logical_id,
            task_type=task.task_type,
            priority=task.priority,
            request=request,
            evidence_hash=request["evidence_hash"],
            contract=contract,
            condition=condition,
        )
        if task.request.get("expression") != semantic["effective_raw_semantic_expression"]:
            raise FinalReleasePreparationError(
                f"{task_id}: API request.expression 未绑定 corrected semantic expression"
            )
        plans[condition].append(task)
    for condition in CONDITIONS:
        plans[condition].sort(key=lambda task: task.logical_id)
    return plans, unresolved


def _clone_identity_task(
    source_task: Mapping[str, Any],
    *,
    expression: str,
    contract: Any,
    reason: str,
) -> Any:
    request = dict(source_task["request"])
    request["expression"] = expression
    request["original_expression"] = expression
    evidence = dict(request.get("ast_source_evidence") or {})
    evidence["final_release_identity_confirmation"] = {
        "reason": reason,
        "effective_expression_sha256": _sha256_text(expression),
        "predecessor_evaluation_key": source_task.get("evaluation_key"),
    }
    request["ast_source_evidence"] = evidence
    request["evidence_hash"] = _sha256_json(
        {
            "logical_id": source_task["logical_id"],
            "expression": expression,
            "reason": reason,
            "source_evidence": evidence,
        }
    )
    return _build_task_definition(
        logical_id=source_task["logical_id"],
        task_type=source_task["task_type"],
        priority=int(source_task.get("priority", PRED_PRIORITY)),
        request=request,
        evidence_hash=request["evidence_hash"],
        contract=contract,
        condition=str(source_task.get("condition", "clean")),
    )


def build_identity_tasks(repo_root: Path) -> list[Any]:
    contract = _load_prompt_schema(repo_root, prompt_path=repo_root / DEFAULT_IDENTITY_PROMPT)
    probes, _ = load_dataset_probes(repo_root / DEFAULT_PROBES)
    gt_rows = _read_jsonl(repo_root / DEFAULT_GT_EXTRACT)
    gt_row = next(row for row in gt_rows if row.get("dataset_id") == GT_IDENTITY_DATASET)
    gt_task, no_call = _build_gt_task(
        gt_row,
        index=45,
        contract=contract,
        dataset_probes=probes,
        logical_id_suffix="v2",
    )
    if gt_task is None or no_call is not None:
        raise FinalReleasePreparationError("feynman-bonus.20 GT identity task 构建失败")

    pred_view = {
        row["logical_id"]: row
        for row in _read_jsonl(repo_root / DEFAULT_PRED_VIEW)
        if row.get("overlay_action") == "replace_expression"
        and row.get("expression_resolution") == "original_identity_fallback_after_audit"
    }
    source_tasks: dict[str, dict[str, Any]] = {}
    with (repo_root / DEFAULT_CLEAN_AUDITED_PLAN).open(encoding="utf-8") as handle:
        for line in handle:
            if not any(logical_id in line for logical_id in pred_view):
                continue
            row = json.loads(line)
            logical_id = row.get("logical_id")
            if logical_id in pred_view:
                source_tasks[logical_id] = row
    # 两条 iMCTS 已由新 final 覆盖；其余五条保留审计后的原式并做真实 identity 调用。
    expected = sorted(logical_id for logical_id in pred_view if "::imcts::" not in logical_id)
    if len(expected) != 5 or sorted(source_tasks) != sorted(pred_view):
        raise FinalReleasePreparationError("clean 审计 fallback 集合漂移")
    pred_tasks = [
        _clone_identity_task(
            source_tasks[logical_id],
            expression=str(pred_view[logical_id]["effective_expression"]),
            contract=contract,
            reason="audited_original_identity_fallback_confirmation",
        )
        for logical_id in expected
    ]
    return [replace(gt_task, request={
        **gt_task.request,
        "evidence_hash": _sha256_json({
            "logical_id": gt_task.logical_id,
            "expression": gt_task.request["expression"],
            "reason": "audited_ground_truth_identity_confirmation",
        }),
    }), *pred_tasks]


def _write_plans(output_root: Path, tasks: Sequence[Any], *, name: str) -> None:
    _write_jsonl(output_root / name, (task.to_json_record() for task in tasks))


def _write_full_plan(
    *,
    source_path: Path,
    output_path: Path,
    replacements: Mapping[str, Mapping[str, Any]],
    expected_count: int,
    identity_field: str = "logical_id",
) -> None:
    seen: set[str] = set()

    def rows() -> Iterable[Mapping[str, Any]]:
        with source_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                logical_id = row.get("logical_id")
                identity = (
                    row.get("request", {}).get("task_id")
                    if identity_field == "task_id"
                    else logical_id
                )
                if not isinstance(logical_id, str) or not isinstance(identity, str):
                    raise FinalReleasePreparationError(
                        f"{source_path}:{line_number} 缺少 {identity_field}"
                    )
                if identity in seen:
                    raise FinalReleasePreparationError(f"full plan 重复 {identity_field}: {identity}")
                seen.add(identity)
                yield replacements.get(identity, row)

    _write_jsonl(output_path, rows())
    if len(seen) != expected_count:
        raise FinalReleasePreparationError(
            f"{source_path}: full plan 期望 {expected_count}，实际 {len(seen)}"
        )
    missing = sorted(set(replacements) - seen)
    if missing:
        raise FinalReleasePreparationError(f"full plan replacement 未命中: {missing[:3]}")


def _load_exact_plan_rows(
    *,
    source_paths: Sequence[Path],
    required_keys: set[str],
    allow_missing: bool = False,
) -> dict[str, dict[str, Any]]:
    """按 evaluation_key 精确找历史计划；文件名或时间不得参与版本选择。"""

    found: dict[str, dict[str, Any]] = {}
    for source_path in sorted(set(source_paths)):
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("evaluation_key")
                if key not in required_keys:
                    continue
                previous = found.get(key)
                if previous is not None and _canonical_json(previous) != _canonical_json(row):
                    raise FinalReleasePreparationError(
                        f"evaluation_key={key} 在历史计划中对应不一致 payload"
                    )
                found[str(key)] = row
    missing = sorted(required_keys - set(found))
    if missing and not allow_missing:
        raise FinalReleasePreparationError(
            f"历史计划中缺少 {len(missing)} 个精确 evaluation_key: {missing[:3]}"
        )
    return found


def _reconstruct_plan_from_frozen_result(
    *, evaluation_key_value: str, repo_root: Path
) -> dict[str, Any]:
    stage_root = repo_root / STAGE_ROOT_RELATIVE
    candidates = sorted(
        {
            *stage_root.glob(f"llm/*/{evaluation_key_value}.json"),
            *stage_root.glob(f"work/*/llm/*/{evaluation_key_value}.json"),
            *stage_root.glob(f"work/*/*/llm/*/{evaluation_key_value}.json"),
        }
    )
    if len(candidates) != 1:
        raise FinalReleasePreparationError(
            f"evaluation_key={evaluation_key_value} 冻结结果候选数量不是1: {candidates}"
        )
    frozen = json.loads(candidates[0].read_text(encoding="utf-8"))
    request = frozen.get("request")
    if not isinstance(request, Mapping):
        raise FinalReleasePreparationError(f"{candidates[0]} 缺少 request")
    metadata = frozen.get("metadata")
    if not isinstance(metadata, Mapping):
        raise FinalReleasePreparationError(f"{candidates[0]} 缺少 metadata")
    prompt_name = Path(str(metadata.get("prompt_path"))).name
    prompt_path = stage_root / "config/prompts" / prompt_name
    contract = _load_prompt_schema(repo_root, prompt_path=prompt_path)
    condition = str(request.get("noise_tag"))
    task = _build_task_definition(
        logical_id=str(frozen["logical_id"]),
        task_type=str(frozen["task_type"]),
        priority=PRED_PRIORITY,
        request=dict(request),
        evidence_hash=str(request["evidence_hash"]),
        contract=contract,
        condition=condition,
    )
    if task.evaluation_key != evaluation_key_value:
        raise FinalReleasePreparationError(
            f"{candidates[0]} 无法按冻结请求重建 evaluation_key"
        )
    return task.to_json_record()


def _next_recovery_logical_id(logical_id: str) -> str:
    parts = logical_id.split("::")
    if parts[0] not in {"pred_simplify", "gt_simplify"}:
        raise FinalReleasePreparationError(f"recovery 仅允许 simplify 任务: {logical_id!r}")
    match = VERSION_SUFFIX_RE.fullmatch(parts[-1])
    if match is None:
        return f"{logical_id}::v2"
    return "::".join([*parts[:-1], f"v{int(match.group(1)) + 1}"])


def _identity_recovery_successor(
    predecessor: Mapping[str, Any], *, prompt_path: Path
) -> dict[str, Any]:
    request = predecessor.get("request")
    if not isinstance(request, Mapping):
        raise FinalReleasePreparationError("recovery predecessor 缺少 request")
    expression = request.get("expression")
    original_expression = request.get("original_expression")
    if not isinstance(expression, str) or not expression:
        raise FinalReleasePreparationError("recovery request.expression 不能为空")
    if original_expression != expression:
        raise FinalReleasePreparationError(
            "identity recovery 要求 original_expression 与 expression 逐字相同"
        )
    prompt_bytes = prompt_path.read_bytes()
    prompt_template = prompt_bytes.decode("utf-8")
    if prompt_template.count("{{REQUEST_JSON}}") != 1:
        raise FinalReleasePreparationError("identity recovery prompt 占位符数量错误")
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    schema = predecessor.get("schema_content")
    if not isinstance(schema, Mapping):
        raise FinalReleasePreparationError("recovery predecessor 缺少 schema_content")
    schema_sha = str(predecessor["schema_sha256"])
    logical_id = _next_recovery_logical_id(str(predecessor["logical_id"]))
    task_type = str(predecessor["task_type"])
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    input_hash = _sha256_json(normalized_input)
    from .claude_contract import evaluation_key, render_prompt

    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=prompt_path.stem,
        schema_version=str(predecessor["schema_version"]),
        prompt_sha256=prompt_sha,
        schema_sha256=schema_sha,
        normalized_input=normalized_input,
        evidence_hash=str(request["evidence_hash"]),
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=str(predecessor["condition"]),
        priority=int(predecessor["priority"]),
        input_hash=input_hash,
        prompt_version=prompt_path.stem,
        schema_version=str(predecessor["schema_version"]),
        dependencies=tuple(predecessor.get("dependencies", [])),
    )
    successor = dict(predecessor)
    successor.update(
        {
            "evaluation_key": task_key,
            "logical_id": logical_id,
            "input_hash": input_hash,
            "prompt_version": prompt_path.stem,
            "prompt_sha256": prompt_sha,
            "prompt_path": str(prompt_path),
            "prompt_template": prompt_template,
            "normalized_input": normalized_input,
            "task_spec": json.loads(spec.canonical_json()),
            "rendered_prompt": render_prompt(prompt_template, request, schema),
        }
    )
    return successor


def build_release_identity_recovery_plan(
    *,
    predecessor_plan_jsonl: Path,
    state_db: Path,
    recovery_prompt_path: Path,
    output_jsonl: Path,
    supersession_manifest: Path,
    condition: str | None = None,
) -> dict[str, Any]:
    """为当前真正 exhausted 的 simplify 任务生成一次性 identity successor。"""

    try:
        loaded = load_plan_jsonl(predecessor_plan_jsonl.resolve())
    except PlanContractError as exc:
        raise FinalReleasePreparationError(f"predecessor plan 契约失败: {exc}") from exc
    predecessor_rows = _read_jsonl(predecessor_plan_jsonl)
    row_by_key = {str(row["evaluation_key"]): row for row in predecessor_rows}
    if len(row_by_key) != len(predecessor_rows) or set(row_by_key) != {
        entry.evaluation_key for entry in loaded.entries
    }:
        raise FinalReleasePreparationError("predecessor plan evaluation_key 漂移或重复")

    connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tasks = {
            str(row["evaluation_key"]): dict(row)
            for row in connection.execute(
                "SELECT evaluation_key,logical_id,task_type,condition_name,state,"
                "attempt_count,last_error_class FROM tasks"
            )
        }
        frozen_keys = {
            str(row[0])
            for row in connection.execute("SELECT evaluation_key FROM frozen_results")
        }
        attempts: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute(
            "SELECT evaluation_key,attempt_number,status,error_class,retryable "
            "FROM attempts ORDER BY evaluation_key,attempt_number"
        ):
            attempts.setdefault(str(row["evaluation_key"]), []).append(dict(row))
    finally:
        connection.close()

    successors: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for entry in loaded.entries:
        task = tasks.get(entry.evaluation_key)
        if task is None:
            raise FinalReleasePreparationError(f"状态库缺少计划任务: {entry.logical_id}")
        task_condition = str(task["condition_name"])
        if condition is not None and task_condition != condition:
            continue
        state = str(task["state"])
        state_counts[state] = state_counts.get(state, 0) + 1
        if state == "frozen":
            if entry.evaluation_key not in frozen_keys:
                raise FinalReleasePreparationError(
                    f"frozen 任务缺少 frozen_results: {entry.logical_id}"
                )
            # 合法 outcome=unable 也是 frozen 终态，绝不能误当 exhausted 重跑。
            continue
        if state in {"pending", "running"}:
            continue
        if state != "exhausted":
            raise FinalReleasePreparationError(f"未知任务终态: {entry.logical_id}={state}")
        if entry.evaluation_key in frozen_keys:
            raise FinalReleasePreparationError(
                f"exhausted 与 frozen_results 冲突: {entry.logical_id}"
            )
        if task["task_type"] not in {"pred_simplify", "gt_simplify"}:
            raise FinalReleasePreparationError(
                f"exhausted 非 simplify 任务不得恢复: {entry.logical_id}"
            )
        task_attempts = attempts.get(entry.evaluation_key, [])
        if (
            not task_attempts
            or len(task_attempts) != int(task["attempt_count"])
            or any(item["status"] != "failed" for item in task_attempts)
        ):
            raise FinalReleasePreparationError(
                f"exhausted attempts 未形成全失败闭环: {entry.logical_id}"
            )
        predecessor = row_by_key[entry.evaluation_key]
        successor = _identity_recovery_successor(
            predecessor,
            prompt_path=recovery_prompt_path.resolve(),
        )
        successors.append(successor)
        request = predecessor["request"]
        ast_source = request.get("ast_source_evidence", {})
        bindings.append(
            {
                "predecessor_evaluation_key": entry.evaluation_key,
                "predecessor_logical_id": entry.logical_id,
                "successor_evaluation_key": successor["evaluation_key"],
                "successor_logical_id": successor["logical_id"],
                "task_type": task["task_type"],
                "condition": task["condition_name"],
                "predecessor_input_hash": predecessor["input_hash"],
                "request_sha256": _sha256_json(request),
                "request_expression_sha256": _sha256_text(request["original_expression"]),
                "source_result_sha256": ast_source.get("result_raw_sha256"),
                "corrected_canonical_artifact_sha256": ast_source.get(
                    "corrected_canonical_artifact_sha256"
                ),
                "effective_raw_semantic_expression_sha256": ast_source.get(
                    "effective_raw_semantic_expression_sha256"
                ),
                "predecessor_attempt_count": len(task_attempts),
                "predecessor_error_classes": sorted(
                    {str(item["error_class"]) for item in task_attempts}
                ),
                "recovery_max_attempts": 1,
            }
        )
    successors.sort(key=lambda row: (int(row["priority"]), str(row["logical_id"])))
    bindings.sort(key=lambda row: str(row["predecessor_logical_id"]))
    _write_jsonl(output_jsonl.resolve(), successors)
    manifest = {
        "schema_version": "core50_release_simplify_identity_recovery.v1",
        "status": "ready_no_api_invoked",
        "predecessor_plan": str(predecessor_plan_jsonl.resolve()),
        "predecessor_plan_sha256": loaded.plan_sha256,
        "state_db": str(state_db.resolve()),
        "condition": condition,
        "state_counts": dict(sorted(state_counts.items())),
        "recovery_prompt": str(recovery_prompt_path.resolve()),
        "recovery_prompt_sha256": _sha256_file(recovery_prompt_path.resolve()),
        "successor_count": len(successors),
        "recovery_max_attempts": 1,
        "successions": bindings,
    }
    _write_json(supersession_manifest.resolve(), manifest)
    return manifest


def materialize_effective_full_clean_plan(
    *,
    inputs_root: Path,
    state_db: Path,
    output_jsonl: Path,
    binding_manifest: Path,
    repo_root: Path,
    condition: str = "clean",
    expected_successor_count: int | None = 18,
    expected_ordinary_count: int | None = 124,
) -> dict[str, Any]:
    """将指定条件的 exhausted predecessor 原位替换为冻结 successor。"""

    if condition not in CONDITIONS:
        raise FinalReleasePreparationError(f"effective full plan condition 非法: {condition}")
    reuse_rows = _read_jsonl(inputs_root / f"pred_reused_index_{condition}.jsonl")
    regular_rows = _read_jsonl(inputs_root / f"pred_plan_{condition}.jsonl")
    identity_rows = (
        _read_jsonl(inputs_root / "pred_identity_plan_clean.jsonl")
        if condition == "clean"
        else []
    )
    successor_rows = _read_jsonl(
        inputs_root / f"simplify_identity_recovery_{condition}.jsonl"
    )
    condition_supersession = (
        inputs_root / f"simplify_identity_recovery_{condition}_supersession.json"
    )
    if condition == "clean" and not condition_supersession.is_file():
        condition_supersession = inputs_root / "simplify_identity_recovery_supersession.json"
    supersession = json.loads(
        condition_supersession.read_text(encoding="utf-8")
    )
    bindings = supersession.get("successions")
    if not isinstance(bindings, list) or len(bindings) != len(successor_rows):
        raise FinalReleasePreparationError("clean recovery supersession 数量漂移")
    predecessor_keys = {str(row["predecessor_evaluation_key"]) for row in bindings}
    successor_keys = {str(row["successor_evaluation_key"]) for row in bindings}
    if expected_successor_count is not None and (
        len(predecessor_keys) != expected_successor_count
        or len(successor_keys) != expected_successor_count
    ):
        raise FinalReleasePreparationError(
            f"{condition} recovery 必须精确为 {expected_successor_count} 对"
        )

    connection = sqlite3.connect(f"{state_db.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        states = {
            str(row["evaluation_key"]): str(row["state"])
            for row in connection.execute("SELECT evaluation_key,state FROM tasks")
        }
        frozen_keys = {
            str(row[0])
            for row in connection.execute("SELECT evaluation_key FROM frozen_results")
        }
    finally:
        connection.close()
    if any(states.get(key) != "superseded" for key in predecessor_keys):
        raise FinalReleasePreparationError(f"{condition} recovery predecessor 尚未全部 superseded")
    if any(states.get(key) != "frozen" or key not in frozen_keys for key in successor_keys):
        raise FinalReleasePreparationError(f"{condition} recovery successor 尚未全部 frozen")

    required_reuse_keys = {str(row["old_opus_evaluation_key"]) for row in reuse_rows}
    exact_reuse = _load_exact_plan_rows(
        source_paths=[
            *repo_root.joinpath(STAGE_ROOT_RELATIVE, "reports").glob(
                f"{condition}_pred_simplify_tasks*.jsonl"
            ),
            *repo_root.joinpath(STAGE_ROOT_RELATIVE, "reports").glob(
                f"{condition}_pred_simplify_full_plan*.jsonl"
            ),
        ],
        required_keys=required_reuse_keys,
        allow_missing=True,
    )
    reconstructed_reuse_keys = sorted(required_reuse_keys - set(exact_reuse))
    for key in reconstructed_reuse_keys:
        exact_reuse[key] = _reconstruct_plan_from_frozen_result(
            evaluation_key_value=key,
            repo_root=repo_root,
        )
    effective: dict[str, dict[str, Any]] = {}

    def add(row: Mapping[str, Any], *, source: str) -> None:
        task_id = row.get("request", {}).get("task_id")
        if not isinstance(task_id, str):
            raise FinalReleasePreparationError(f"{source}: plan row 缺少 task_id")
        if task_id in effective:
            raise FinalReleasePreparationError(f"{condition} effective plan 重复 task_id: {task_id}")
        effective[task_id] = dict(row)

    for reuse in reuse_rows:
        row = exact_reuse[str(reuse["old_opus_evaluation_key"])]
        if row.get("request", {}).get("task_id") != reuse["task_id"]:
            raise FinalReleasePreparationError(
                f"{reuse['task_id']}: reused evaluation_key 反向绑定错误"
            )
        add(row, source="reused")
    ordinary_regular = [
        row for row in regular_rows if str(row["evaluation_key"]) not in predecessor_keys
    ]
    if expected_ordinary_count is not None and len(ordinary_regular) != expected_ordinary_count:
        raise FinalReleasePreparationError(
            f"{condition} 普通新冻结任务期望 {expected_ordinary_count}，实际 {len(ordinary_regular)}"
        )
    if any(
        states.get(str(row["evaluation_key"])) != "frozen"
        or str(row["evaluation_key"]) not in frozen_keys
        for row in ordinary_regular
    ):
        raise FinalReleasePreparationError(f"{condition} 普通新任务尚未全部 frozen")
    expected_identity_count = 5 if condition == "clean" else 0
    if len(identity_rows) != expected_identity_count or any(
        states.get(str(row["evaluation_key"])) != "frozen"
        or str(row["evaluation_key"]) not in frozen_keys
        for row in identity_rows
    ):
        raise FinalReleasePreparationError(f"{condition} identity 任务状态不完整")
    for row in ordinary_regular:
        add(row, source="ordinary_refresh")
    for row in successor_rows:
        if str(row["evaluation_key"]) not in successor_keys:
            raise FinalReleasePreparationError("clean successor 未命中 supersession manifest")
        add(row, source="identity_recovery_successor")
    for row in identity_rows:
        add(row, source="identity_canary")
    if len(effective) != 2250:
        raise FinalReleasePreparationError(f"{condition} effective plan 不是 2250: {len(effective)}")

    _write_jsonl(output_jsonl.resolve(), (effective[key] for key in sorted(effective)))
    manifest = {
        "schema_version": "core50_full_condition_effective_plan.v1",
        "condition": condition,
        "status": "ready_no_api_invoked",
        "output_jsonl": str(output_jsonl.resolve()),
        "output_sha256": _sha256_file(output_jsonl.resolve()),
        "total_count": 2250,
        "global_reused_count": len(_read_jsonl(inputs_root / "pred_reuse_index.jsonl")),
        "condition_reused_count": len(reuse_rows),
        "condition_reused_plan_reconstructed_from_frozen_count": len(
            reconstructed_reuse_keys
        ),
        "ordinary_refresh_frozen_count": len(ordinary_regular),
        "recovery_successor_frozen_count": len(successor_rows),
        "identity_frozen_count": len(identity_rows),
        "partition_check": len(reuse_rows)
        + len(ordinary_regular)
        + len(successor_rows)
        + len(identity_rows),
        "supersession_bindings": bindings,
        "source_files": {
            "regular_plan": str((inputs_root / f"pred_plan_{condition}.jsonl").resolve()),
            "identity_plan": str((inputs_root / "pred_identity_plan_clean.jsonl").resolve())
            if condition == "clean"
            else None,
            "recovery_plan": str(
                (inputs_root / f"simplify_identity_recovery_{condition}.jsonl").resolve()
            ),
            "recovery_supersession": str(condition_supersession.resolve()),
        },
    }
    _write_json(binding_manifest.resolve(), manifest)
    return manifest


def prepare_identity_canary(*, repo_root: Path, output_root: Path) -> dict[str, Any]:
    report_path = output_root / "identity_canary_report.json"
    plan_path = output_root / "identity_canary_plan.jsonl"
    if report_path.is_file() and plan_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("task_count") != 6 or len(_read_jsonl(plan_path)) != 6:
            raise FinalReleasePreparationError("已有 identity canary 计划不完整，拒绝覆盖")
        return report
    tasks = build_identity_tasks(repo_root)
    # Rebuild the GT definition after request evidence mutation so its key binds the final payload.
    gt = tasks[0]
    contract = _load_prompt_schema(repo_root, prompt_path=repo_root / DEFAULT_IDENTITY_PROMPT)
    gt = _build_task_definition(
        logical_id=gt.logical_id,
        task_type=gt.task_type,
        priority=gt.priority,
        request=gt.request,
        evidence_hash=gt.request["evidence_hash"],
        contract=contract,
        condition=gt.condition,
    )
    tasks = [gt, *tasks[1:]]
    _write_plans(output_root, tasks, name="identity_canary_plan.jsonl")
    _write_plans(output_root, [gt], name="gt_plan.jsonl")
    _write_plans(output_root, tasks[1:], name="pred_identity_plan_clean.jsonl")
    report = {
        "schema_version": "core50_final_identity_canary.v1",
        "status": "ready_for_api_no_api_invoked",
        "task_count": len(tasks),
        "gt_task_count": 1,
        "pred_task_count": 5,
        "evaluation_keys": [task.evaluation_key for task in tasks],
    }
    _write_json(report_path, report)
    return report


def prepare_release(*, repo_root: Path, base: Path, symbolfit: Path, output_root: Path) -> dict[str, Any]:
    replacements = _load_symbolfit_replacements(symbolfit)
    old_opus = _load_old_opus(base)
    old_numeric = _load_numeric_artifacts(repo_root)
    recovery_manifests = {
        condition: load_formula_recovery_manifest(
            repo_root
            / (
                DEFAULT_FORMULA_RECOVERY
                if condition == "clean"
                else STAGE_ROOT_RELATIVE / f"manifests/{condition}_formula_recovery.v1.json"
            ),
            expected_condition=condition,
        )
        for condition in CONDITIONS
    }
    all_semantic: list[dict[str, Any]] = []
    merged_by_task: dict[str, dict[str, Any]] = {}
    replacement_counts: dict[str, int] = {}
    for condition in CONDITIONS:
        source_path = base / f"results/{condition}/final_results.jsonl.gz"
        merged, replacement_count = merge_final_rows(
            _read_jsonl(source_path), replacements, condition=condition
        )
        replacement_counts[condition] = replacement_count
        for row in merged:
            task_id = _task_id(row)
            if task_id in merged_by_task:
                raise FinalReleasePreparationError(f"跨条件重复 task_id: {task_id}")
            merged_by_task[task_id] = row
        _write_jsonl(
            output_root / f"merged_final_results_{condition}.jsonl.gz",
            merged,
            gzip_output=True,
        )
        all_semantic.extend(
            build_semantic_row(
                row,
                condition=condition,
                old_opus=old_opus,
                old_numeric_artifacts=old_numeric,
                recovery_manifest=recovery_manifests[condition],
            )
            for row in merged
        )
    if len(all_semantic) != 6750:
        raise FinalReleasePreparationError("semantic_runs 必须为 6750 条")
    _write_jsonl(output_root / "semantic_runs.jsonl", all_semantic)
    refresh = [row for row in all_semantic if row["requires_pred_simplify_refresh"]]
    numeric_replay = [row for row in all_semantic if row["numeric_requires_replay"]]
    _write_jsonl(output_root / "pred_simplify_refresh_manifest.jsonl", refresh)
    _write_jsonl(output_root / "numeric_requires_replay.jsonl", numeric_replay)
    identity_report = prepare_identity_canary(repo_root=repo_root, output_root=output_root)
    identity_tasks = _read_jsonl(output_root / "identity_canary_plan.jsonl")
    identity_logical_ids = {str(row["logical_id"]) for row in identity_tasks}
    pred_plans, pred_plan_unresolved = build_incremental_pred_plans(
        repo_root=repo_root,
        merged_by_task=merged_by_task,
        semantic_rows=all_semantic,
        identity_logical_ids=identity_logical_ids,
    )
    pred_plan_rows: list[dict[str, Any]] = []
    new_pred_by_task_id: dict[str, dict[str, Any]] = {}
    for condition, tasks in pred_plans.items():
        rows = [task.to_json_record() for task in tasks]
        pred_plan_rows.extend(rows)
        new_pred_by_task_id.update({str(row["request"]["task_id"]): row for row in rows})
        _write_jsonl(output_root / f"pred_plan_{condition}.jsonl", rows)
    for row in identity_tasks:
        if row["task_type"] == "pred_simplify":
            new_pred_by_task_id[str(row["request"]["task_id"])] = row
    combined_plan = [*identity_tasks, *pred_plan_rows]
    combined_plan.sort(key=lambda row: (int(row["priority"]), str(row["logical_id"])))
    _write_jsonl(output_root / "simplify_refresh_plan.jsonl", combined_plan)
    _write_jsonl(output_root / "pred_plan_unresolved.jsonl", pred_plan_unresolved)
    reuse_rows = [
        {
            "logical_id": row["logical_id"],
            "task_id": row["task_id"],
            "condition": row["condition"],
            "old_opus_evaluation_key": row["old_opus_evaluation_key"],
            "effective_raw_semantic_expression_sha256": row[
                "effective_raw_semantic_expression_sha256"
            ],
            "reuse_basis": "ast_equivalent_request_expression",
        }
        for row in all_semantic
        if not row["requires_pred_simplify_refresh"]
        and row["logical_id"] not in identity_logical_ids
    ]
    _write_jsonl(output_root / "pred_reuse_index.jsonl", reuse_rows)
    for condition in CONDITIONS:
        _write_jsonl(
            output_root / f"pred_reused_index_{condition}.jsonl",
            (row for row in reuse_rows if row["condition"] == condition),
        )
    reports_root = repo_root / STAGE_ROOT_RELATIVE / "reports"
    for condition in CONDITIONS:
        condition_new = {
            task_id: row
            for task_id, row in new_pred_by_task_id.items()
            if f"_{condition}_" in task_id
        }
        condition_reuse = [row for row in reuse_rows if row["condition"] == condition]
        required_keys = {str(row["old_opus_evaluation_key"]) for row in condition_reuse}
        source_paths = list(
            reports_root.glob(f"{condition}_pred_simplify_tasks*.jsonl")
        )
        source_paths.extend(reports_root.glob(f"{condition}_pred_simplify_full_plan*.jsonl"))
        exact_history = _load_exact_plan_rows(
            source_paths=source_paths,
            required_keys=required_keys,
        )
        full_by_task = dict(condition_new)
        for reuse in condition_reuse:
            key = str(reuse["old_opus_evaluation_key"])
            plan_row = exact_history[key]
            task_id = plan_row.get("request", {}).get("task_id")
            if task_id != reuse["task_id"]:
                raise FinalReleasePreparationError(
                    f"{reuse['task_id']}: compact Opus key 反向绑定到其它历史计划 {task_id!r}"
                )
            if task_id in full_by_task:
                raise FinalReleasePreparationError(f"full pred plan 重复 task_id: {task_id}")
            full_by_task[str(task_id)] = plan_row
        if len(full_by_task) != 2250:
            raise FinalReleasePreparationError(
                f"{condition}: full pred plan 期望 2250，实际 {len(full_by_task)}"
            )
        _write_jsonl(
            output_root / f"pred_plan_full_{condition}.jsonl",
            (full_by_task[task_id] for task_id in sorted(full_by_task)),
        )
    gt_identity = next(row for row in identity_tasks if row["task_type"] == "gt_simplify")
    _write_full_plan(
        source_path=repo_root / STAGE_ROOT_RELATIVE / "reports/clean_gt_simplify_tasks_v2.jsonl",
        output_path=output_root / "gt_plan_full.jsonl",
        replacements={str(gt_identity["logical_id"]): gt_identity},
        expected_count=50,
    )
    gt_reused = [
        {
            "logical_id": row["logical_id"],
            "evaluation_key": row["evaluation_key"],
            "effective_expression": row["effective_expression"],
            "effective_expression_sha256": _sha256_text(row["effective_expression"]),
            "reuse_basis": "audited_effective_gt_reference",
        }
        for row in _read_jsonl(base / "ground_truth/opus5_effective_references.jsonl")
        if row["logical_id"] != gt_identity["logical_id"]
    ]
    if len(gt_reused) != 49:
        raise FinalReleasePreparationError(f"GT reuse 期望 49，实际 {len(gt_reused)}")
    _write_jsonl(output_root / "gt_reused_index.jsonl", gt_reused)
    refresh_reason_counts: dict[str, int] = {}
    refresh_breakdown: dict[str, int] = {}
    for row in refresh:
        reason = str(row["pred_simplify_refresh_reason"])
        refresh_reason_counts[reason] = refresh_reason_counts.get(reason, 0) + 1
        key = f"{str(row['algorithm']).lower()}/{row['condition']}/{reason}"
        refresh_breakdown[key] = refresh_breakdown.get(key, 0) + 1
    report = {
        "schema_version": "core50_final_release_inputs.v2",
        "status": "inputs_ready_no_api_invoked",
        "run_count": len(all_semantic),
        "replacement_counts": replacement_counts,
        "pred_simplify_refresh_count": len(refresh),
        "numeric_requires_replay_count": len(numeric_replay),
        "identity_task_count": identity_report["task_count"],
        "executable_simplify_task_count": len(combined_plan),
        "regular_pred_plan_counts": {
            condition: len(tasks) for condition, tasks in pred_plans.items()
        },
        "pred_plan_unresolved_count": len(pred_plan_unresolved),
        "pred_reuse_count": len(reuse_rows),
        "pred_api_task_count": len(combined_plan) - 1,
        "gt_reuse_count": len(gt_reused),
        "refresh_reason_counts": dict(sorted(refresh_reason_counts.items())),
        "refresh_breakdown": dict(sorted(refresh_breakdown.items())),
        "forced_raw_rebuild_algorithms": sorted(FORCED_REBUILD_ALGORITHMS),
        "inputs": {
            "base": str(base),
            "symbolfit": str(symbolfit),
        },
    }
    _write_json(output_root / "refresh_manifest.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--base", type=Path)
    parser.add_argument("--symbolfit", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--build-recovery", action="store_true")
    parser.add_argument("--materialize-clean-effective", action="store_true")
    parser.add_argument(
        "--materialize-effective-condition", choices=CONDITIONS, default=None
    )
    parser.add_argument("--predecessor-plan", type=Path)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument(
        "--recovery-condition", choices=CONDITIONS, default=None
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = (args.output_root or (repo_root / DEFAULT_OUTPUT)).resolve()
    if args.materialize_effective_condition or args.materialize_clean_effective:
        effective_condition = args.materialize_effective_condition or "clean"
        expected_successors = {"clean": 18, "noise001": 34}.get(effective_condition)
        expected_ordinary = {"clean": 124, "noise001": 253}.get(effective_condition)
        report = materialize_effective_full_clean_plan(
            inputs_root=output_root,
            state_db=(
                args.state_db
                or repo_root / FINAL_WORK_RELATIVE / "release_v2/llm/release_state.sqlite3"
            ).resolve(),
            output_jsonl=output_root
            / f"pred_plan_full_{effective_condition}_effective.jsonl",
            binding_manifest=output_root
            / f"pred_plan_full_{effective_condition}_effective_supersession.json",
            repo_root=repo_root,
            condition=effective_condition,
            expected_successor_count=expected_successors,
            expected_ordinary_count=expected_ordinary,
        )
    elif args.build_recovery:
        recovery_name = (
            f"simplify_identity_recovery_{args.recovery_condition}"
            if args.recovery_condition
            else "simplify_identity_recovery"
        )
        report = build_release_identity_recovery_plan(
            predecessor_plan_jsonl=(
                args.predecessor_plan or output_root / "simplify_refresh_plan.jsonl"
            ).resolve(),
            state_db=(
                args.state_db
                or repo_root / FINAL_WORK_RELATIVE / "release_v2/llm/release_state.sqlite3"
            ).resolve(),
            recovery_prompt_path=(repo_root / DEFAULT_IDENTITY_PROMPT).resolve(),
            output_jsonl=output_root / f"{recovery_name}.jsonl",
            supersession_manifest=output_root / f"{recovery_name}_supersession.json",
            condition=args.recovery_condition,
        )
    elif args.identity_only:
        report = prepare_identity_canary(repo_root=repo_root, output_root=output_root)
    else:
        report = prepare_release(
            repo_root=repo_root,
            base=(args.base or (repo_root / DEFAULT_BASE)).resolve(),
            symbolfit=(args.symbolfit or (repo_root / DEFAULT_SYMBOLFIT)).resolve(),
            output_root=output_root,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
