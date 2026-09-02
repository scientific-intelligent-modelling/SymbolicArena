"""构建 Stage5 公式化简与等价裁决的独立盲审计划。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    canonical_json,
    evaluation_key,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec


SEEDS = (520, 521, 522)
CONDITIONS = ("clean", "noise001", "noise005")
AUDIT_TASK_TYPE = "formula_audit"
AUDIT_TASK_KIND = "formula_audit"
AUDIT_PRIORITY = 50
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SAMPLE_SEED = "formula-quality-audit-20260903-v1"


class FormulaAuditPlanError(ValueError):
    """输入产物或抽样计划不满足公式审计契约。"""


@dataclass(frozen=True)
class GroundTruthSource:
    dataset_id: str
    logical_id: str
    evaluation_key: str
    result_sha256: str
    original_expression: str
    effective_expression: str
    variables: tuple[str, ...]
    allowed_functions: tuple[str, ...]
    domain_assumptions: Mapping[str, Any]
    stored_outcome: str


@dataclass(frozen=True)
class PredictionSource:
    logical_key: str
    condition: str
    algorithm: str
    algorithm_slug: str
    dataset_id: str
    dataset_index: str
    seed: int
    pred_logical_id: str
    pred_evaluation_key: str
    pred_result_sha256: str
    equivalence_logical_id: str
    original_expression: str
    effective_expression: str
    reference_expression: str
    variables: tuple[str, ...]
    allowed_functions: tuple[str, ...]
    domain_assumptions: Mapping[str, Any]
    stored_simplification_outcome: str
    expression_resolution: str
    stored_equivalence_decision: str
    gt_logical_id: str
    gt_artifact_sha256: str
    pred_artifact_sha256: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage_root() -> Path:
    return _repo_root() / "AAAI_experiments/stage5_metric_calculation_0831"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormulaAuditPlanError(f"{context} 必须是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaAuditPlanError(f"{context} 必须是非空字符串")
    return value.strip()


def _require_string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FormulaAuditPlanError(f"{context} 必须是字符串数组")
    return tuple(value)


def _parse_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise FormulaAuditPlanError(f"{context} 必须为 true/false")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FormulaAuditPlanError(f"JSONL 不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FormulaAuditPlanError(f"{path}:{line_number} 不得为空行")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormulaAuditPlanError(f"{path}:{line_number} JSON 非法") from exc
            rows.append(dict(_require_mapping(row, context=f"{path}:{line_number}")))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FormulaAuditPlanError(f"CSV 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique_by(rows: Sequence[Mapping[str, Any]], key: str, *, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _require_string(row.get(key), context=f"{context}.{key}")
        if value in result:
            raise FormulaAuditPlanError(f"{context} 出现重复 {key}: {value}")
        result[value] = dict(row)
    return result


def _load_ground_truth_sources(
    *,
    plan_path: Path,
    index_path: Path,
) -> dict[str, GroundTruthSource]:
    plan_by_logical = _unique_by(_read_jsonl(plan_path), "logical_id", context="gt_plan")
    index_by_logical = _unique_by(_read_jsonl(index_path), "logical_id", context="gt_index")
    if set(plan_by_logical) != set(index_by_logical):
        raise FormulaAuditPlanError("GT plan 与 frozen index 的 logical_id 集合不一致")
    result: dict[str, GroundTruthSource] = {}
    for logical_id, plan_row in plan_by_logical.items():
        index_row = index_by_logical[logical_id]
        if index_row.get("state") != "frozen":
            raise FormulaAuditPlanError(f"GT 未冻结: {logical_id}")
        request = _require_mapping(plan_row.get("request"), context=f"{logical_id}.request")
        structured = _require_mapping(
            index_row.get("structured_output"), context=f"{logical_id}.structured_output"
        )
        dataset_id = _require_string(request.get("dataset_id"), context=f"{logical_id}.dataset_id")
        if dataset_id in result:
            raise FormulaAuditPlanError(f"GT dataset_id 重复: {dataset_id}")
        result[dataset_id] = GroundTruthSource(
            dataset_id=dataset_id,
            logical_id=logical_id,
            evaluation_key=_require_string(
                index_row.get("evaluation_key"), context=f"{logical_id}.evaluation_key"
            ),
            result_sha256=_require_string(
                index_row.get("result_sha256"), context=f"{logical_id}.result_sha256"
            ),
            original_expression=_require_string(
                request.get("expression"), context=f"{logical_id}.expression"
            ),
            effective_expression=_require_string(
                index_row.get("effective_expression"), context=f"{logical_id}.effective_expression"
            ),
            variables=_require_string_tuple(
                request.get("variables"), context=f"{logical_id}.variables"
            ),
            allowed_functions=_require_string_tuple(
                request.get("allowed_functions"), context=f"{logical_id}.allowed_functions"
            ),
            domain_assumptions=dict(
                _require_mapping(
                    request.get("domain_assumptions"),
                    context=f"{logical_id}.domain_assumptions",
                )
            ),
            stored_outcome=_require_string(
                structured.get("outcome"), context=f"{logical_id}.outcome"
            ),
        )
    if len(result) != 50:
        raise FormulaAuditPlanError(f"GT 数量应为 50，实际为 {len(result)}")
    return result


def _load_prediction_sources(
    *,
    condition: str,
    plan_path: Path,
    index_path: Path,
    evidence_path: Path,
    run_rows: Sequence[Mapping[str, str]],
) -> tuple[list[PredictionSource], list[dict[str, Any]]]:
    plan_by_logical = _unique_by(_read_jsonl(plan_path), "logical_id", context=f"{condition}_plan")
    index_by_logical = _unique_by(_read_jsonl(index_path), "logical_id", context=f"{condition}_index")
    evidence_by_key = _unique_by(
        _read_jsonl(evidence_path), "logical_key", context=f"{condition}_evidence"
    )
    sources: list[PredictionSource] = []
    excluded: list[dict[str, Any]] = []
    for row in run_rows:
        if row.get("noise_tag") != condition:
            continue
        logical_key = _require_string(row.get("logical_key"), context="run.logical_key")
        if not _parse_bool(row.get("valid_output"), context=f"{logical_key}.valid_output"):
            continue
        pred_logical_id = _require_string(
            row.get("pred_logical_id"), context=f"{logical_key}.pred_logical_id"
        )
        plan_row = plan_by_logical.get(pred_logical_id)
        index_row = index_by_logical.get(pred_logical_id)
        evidence = evidence_by_key.get(logical_key)
        if plan_row is None or index_row is None:
            raise FormulaAuditPlanError(f"{logical_key} 缺少 plan/index/evidence 关联")
        if index_row.get("state") == "non_applicable":
            if evidence is not None:
                raise FormulaAuditPlanError(f"{logical_key} non_applicable 但存在 evidence")
            non_applicable = _require_mapping(
                index_row.get("non_applicable"), context=f"{logical_key}.non_applicable"
            )
            excluded.append(
                {
                    "logical_key": logical_key,
                    "condition": condition,
                    "algorithm": _require_string(
                        row.get("algorithm"), context=f"{logical_key}.algorithm"
                    ),
                    "dataset_id": _require_string(
                        row.get("dataset_id"), context=f"{logical_key}.dataset_id"
                    ),
                    "seed": int(_require_string(row.get("seed"), context=f"{logical_key}.seed")),
                    "pred_logical_id": pred_logical_id,
                    "state": "non_applicable",
                    "reason": _require_string(
                        non_applicable.get("reason"),
                        context=f"{logical_key}.non_applicable.reason",
                    ),
                }
            )
            continue
        if index_row.get("state") != "frozen":
            raise FormulaAuditPlanError(f"{logical_key} 的 pred state 不是 frozen")
        if evidence is None:
            raise FormulaAuditPlanError(f"{logical_key} 缺少 evidence 关联")
        request = _require_mapping(plan_row.get("request"), context=f"{logical_key}.request")
        expected_identity = {
            "algorithm": _require_string(
                row.get("algorithm"), context=f"{logical_key}.algorithm"
            ),
            "dataset_id": _require_string(
                row.get("dataset_id"), context=f"{logical_key}.dataset_id"
            ),
            "seed": int(_require_string(row.get("seed"), context=f"{logical_key}.seed")),
            "noise_tag": condition,
        }
        for field, expected_value in expected_identity.items():
            if request.get(field) != expected_value:
                raise FormulaAuditPlanError(
                    f"{logical_key} 的 plan request.{field}={request.get(field)!r}，"
                    f"预期 {expected_value!r}"
                )
        if evidence.get("logical_key") != logical_key:
            raise FormulaAuditPlanError(f"{logical_key} 的 evidence logical_key 不一致")
        if evidence.get("pred_logical_id") != pred_logical_id:
            raise FormulaAuditPlanError(f"{logical_key} 的 evidence pred_logical_id 不一致")
        structured = _require_mapping(
            index_row.get("structured_output"), context=f"{logical_key}.structured_output"
        )
        prediction_evidence = _require_mapping(
            evidence.get("prediction"), context=f"{logical_key}.prediction"
        )
        ground_truth_evidence = _require_mapping(
            evidence.get("ground_truth"), context=f"{logical_key}.ground_truth"
        )
        effective_expression = _require_string(
            index_row.get("effective_expression"), context=f"{logical_key}.effective_expression"
        )
        evidence_expression = _require_string(
            prediction_evidence.get("simplified_expression"),
            context=f"{logical_key}.prediction.simplified_expression",
        )
        if effective_expression != evidence_expression:
            raise FormulaAuditPlanError(f"{logical_key} 的 index 与 evidence 预测表达式不一致")
        seed = int(_require_string(row.get("seed"), context=f"{logical_key}.seed"))
        if seed not in SEEDS:
            raise FormulaAuditPlanError(f"{logical_key} seed 非法: {seed}")
        sources.append(
            PredictionSource(
                logical_key=logical_key,
                condition=condition,
                algorithm=_require_string(row.get("algorithm"), context=f"{logical_key}.algorithm"),
                algorithm_slug=_require_string(
                    request.get("algorithm_slug"), context=f"{logical_key}.algorithm_slug"
                ),
                dataset_id=_require_string(
                    row.get("dataset_id"), context=f"{logical_key}.dataset_id"
                ),
                dataset_index=_require_string(
                    request.get("dataset_index"), context=f"{logical_key}.dataset_index"
                ),
                seed=seed,
                pred_logical_id=pred_logical_id,
                pred_evaluation_key=_require_string(
                    index_row.get("evaluation_key"), context=f"{logical_key}.evaluation_key"
                ),
                pred_result_sha256=_require_string(
                    index_row.get("result_sha256"), context=f"{logical_key}.result_sha256"
                ),
                equivalence_logical_id=_require_string(
                    row.get("equivalence_logical_id"),
                    context=f"{logical_key}.equivalence_logical_id",
                ),
                original_expression=_require_string(
                    request.get("expression"), context=f"{logical_key}.expression"
                ),
                effective_expression=effective_expression,
                reference_expression=_require_string(
                    ground_truth_evidence.get("simplified_expression"),
                    context=f"{logical_key}.ground_truth.simplified_expression",
                ),
                variables=_require_string_tuple(
                    request.get("variables"), context=f"{logical_key}.variables"
                ),
                allowed_functions=_require_string_tuple(
                    request.get("allowed_functions"), context=f"{logical_key}.allowed_functions"
                ),
                domain_assumptions=dict(
                    _require_mapping(
                        request.get("domain_assumptions"),
                        context=f"{logical_key}.domain_assumptions",
                    )
                ),
                stored_simplification_outcome=_require_string(
                    structured.get("outcome"), context=f"{logical_key}.outcome"
                ),
                expression_resolution=_require_string(
                    index_row.get("expression_resolution"),
                    context=f"{logical_key}.expression_resolution",
                ),
                stored_equivalence_decision=_require_string(
                    row.get("equivalence_decision"),
                    context=f"{logical_key}.equivalence_decision",
                ),
                gt_logical_id=_require_string(
                    evidence.get("gt_logical_id"), context=f"{logical_key}.gt_logical_id"
                ),
                gt_artifact_sha256=_require_string(
                    ground_truth_evidence.get("artifact_sha256"),
                    context=f"{logical_key}.gt_artifact_sha256",
                ),
                pred_artifact_sha256=_require_string(
                    prediction_evidence.get("artifact_sha256"),
                    context=f"{logical_key}.pred_artifact_sha256",
                ),
            )
        )
    return sources, excluded


def _risk_reasons(source: PredictionSource) -> tuple[str, ...]:
    reasons: list[str] = []
    if source.stored_simplification_outcome == "unable" or "fallback" in source.expression_resolution:
        reasons.append("simplification_fallback")
    if source.stored_equivalence_decision in {"equivalent", "undetermined"}:
        reasons.append(f"equivalence_{source.stored_equivalence_decision}")
    return tuple(reasons)


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def select_prediction_sample(
    sources: Sequence[PredictionSource],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> tuple[list[PredictionSource], dict[str, tuple[str, ...]]]:
    if sample_size < 1000:
        raise FormulaAuditPlanError("prediction sample_size 不得小于 1000")
    by_key = {source.logical_key: source for source in sources}
    if len(by_key) != len(sources):
        raise FormulaAuditPlanError("预测总体出现重复 logical_key")
    if sample_size > len(sources):
        raise FormulaAuditPlanError("sample_size 超过有效预测总体")

    reasons = {source.logical_key: _risk_reasons(source) for source in sources}
    selected: dict[str, PredictionSource] = {
        source.logical_key: source for source in sources if reasons[source.logical_key]
    }
    if len(selected) > sample_size:
        raise FormulaAuditPlanError(
            f"强制高风险样本 {len(selected)} 已超过 sample_size={sample_size}"
        )

    groups: dict[tuple[str, str, int], list[PredictionSource]] = defaultdict(list)
    for source in sources:
        if source.logical_key not in selected:
            groups[(source.condition, source.algorithm, source.seed)].append(source)
    for group in groups.values():
        group.sort(key=lambda item: _stable_rank(sample_seed, item.logical_key))
    selected_counts = Counter(
        (item.condition, item.algorithm, item.seed) for item in selected.values()
    )
    dataset_counts = Counter(item.dataset_id for item in selected.values())
    while len(selected) < sample_size:
        available = [stratum for stratum, group in groups.items() if group]
        if not available:
            raise FormulaAuditPlanError("无法从预测总体补足样本")
        stratum = min(
            available,
            key=lambda item: (
                selected_counts[item],
                _stable_rank(sample_seed, f"{item[0]}::{item[1]}::s{item[2]}"),
            ),
        )
        source = min(
            groups[stratum],
            key=lambda item: (
                dataset_counts[item.dataset_id],
                _stable_rank(sample_seed, item.logical_key),
            ),
        )
        groups[stratum].remove(source)
        selected[source.logical_key] = source
        selected_counts[stratum] += 1
        dataset_counts[source.dataset_id] += 1

    ordered = sorted(selected.values(), key=lambda item: item.logical_key)
    if {item.condition for item in ordered} != set(CONDITIONS):
        raise FormulaAuditPlanError("样本未覆盖全部 condition")
    if {item.seed for item in ordered} != set(SEEDS):
        raise FormulaAuditPlanError("样本未覆盖全部 seed")
    if len({item.algorithm for item in ordered}) != 15:
        raise FormulaAuditPlanError("样本未覆盖全部 15 个算法")
    if len({item.dataset_id for item in ordered}) != 50:
        raise FormulaAuditPlanError("样本未覆盖全部 50 个任务")
    if len({(item.condition, item.algorithm) for item in ordered}) != 45:
        raise FormulaAuditPlanError("样本未覆盖全部 algorithm-condition 分层")
    if len({(item.condition, item.algorithm, item.seed) for item in ordered}) != 135:
        raise FormulaAuditPlanError("样本未覆盖全部 algorithm-condition-seed 分层")
    return ordered, reasons


def _with_evidence_hash(request_without_hash: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(request_without_hash)
    request["evidence_hash"] = _sha256_json(request_without_hash)
    return request


def _build_plan_row(
    *,
    logical_id: str,
    condition: str,
    request: dict[str, Any],
    prompt_path: Path,
    prompt_template: str,
    prompt_sha256: str,
    schema_path: Path,
    schema_content: dict[str, Any],
    schema_sha256: str,
) -> dict[str, Any]:
    prompt_version = prompt_path.stem
    schema_version = schema_path.stem
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }
    input_hash = _sha256_json(normalized_input)
    task_key = evaluation_key(
        task_type=AUDIT_TASK_TYPE,
        logical_id=logical_id,
        prompt_version=prompt_version,
        schema_version=schema_version,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=_require_string(request.get("evidence_hash"), context="request.evidence_hash"),
    )
    spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=AUDIT_TASK_TYPE,
        condition=condition,
        priority=AUDIT_PRIORITY,
        input_hash=input_hash,
        prompt_version=prompt_version,
        schema_version=schema_version,
        dependencies=(),
    )
    return {
        "evaluation_key": task_key,
        "logical_id": logical_id,
        "task_type": AUDIT_TASK_TYPE,
        "task_kind": AUDIT_TASK_KIND,
        "condition": condition,
        "priority": AUDIT_PRIORITY,
        "input_hash": input_hash,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
        "dependencies": [],
        "prompt_path": str(prompt_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "prompt_template": prompt_template,
        "schema_content": schema_content,
        "normalized_input": normalized_input,
        "request": request,
        "task_spec": json.loads(spec.canonical_json()),
    }


def build_formula_audit_plan(
    *,
    audit_root: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    stage_root = _stage_root()
    prompt_path = stage_root / "config/prompts/formula_audit.v1.txt"
    schema_path = stage_root / "config/schemas/formula_audit.v1.json"
    prompt_bytes = prompt_path.read_bytes()
    schema_bytes = schema_path.read_bytes()
    prompt_template = prompt_bytes.decode("utf-8")
    schema_content = dict(
        _require_mapping(json.loads(schema_bytes), context=str(schema_path))
    )
    prompt_sha256 = _sha256_bytes(prompt_bytes)
    schema_sha256 = _sha256_bytes(schema_bytes)

    input_paths = {
        "gt_plan": stage_root / "reports/clean_gt_simplify_tasks_v2.jsonl",
        "gt_index": stage_root / "results/clean_gt_simplify_frozen_index_v2.jsonl",
        "run_metrics": stage_root / "results/run_metrics_all_conditions.csv",
        "clean_pred_plan": stage_root / "reports/clean_pred_simplify_tasks_active_v5.jsonl",
        "clean_pred_index": stage_root / "results/clean_pred_simplify_frozen_index_active_v5.jsonl",
        "clean_evidence": stage_root / "results/clean_pred_vs_gt_evidence.jsonl",
        "noise001_pred_plan": stage_root / "reports/noise001_pred_simplify_tasks_active_v3.jsonl",
        "noise001_pred_index": stage_root / "results/noise001_pred_simplify_frozen_index.jsonl",
        "noise001_evidence": stage_root / "results/noise001_pred_vs_gt_evidence.jsonl",
        "noise005_pred_plan": stage_root / "reports/noise005_pred_simplify_full_plan_active_v2.jsonl",
        "noise005_pred_index": stage_root / "results/noise005_pred_simplify_frozen_index.jsonl",
        "noise005_evidence": stage_root / "results/noise005_pred_vs_gt_evidence.jsonl",
    }
    gt_sources = _load_ground_truth_sources(
        plan_path=input_paths["gt_plan"], index_path=input_paths["gt_index"]
    )
    run_rows = _read_csv(input_paths["run_metrics"])
    pred_sources: list[PredictionSource] = []
    excluded_predictions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_sources, condition_excluded = _load_prediction_sources(
            condition=condition,
            plan_path=input_paths[f"{condition}_pred_plan"],
            index_path=input_paths[f"{condition}_pred_index"],
            evidence_path=input_paths[f"{condition}_evidence"],
            run_rows=run_rows,
        )
        pred_sources.extend(condition_sources)
        excluded_predictions.extend(condition_excluded)
    if len(pred_sources) != 6749:
        raise FormulaAuditPlanError(
            f"有效预测总体应为 6749，实际为 {len(pred_sources)}"
        )
    if len(excluded_predictions) != 1:
        raise FormulaAuditPlanError(
            f"不可审计预测数量应为 1，实际为 {len(excluded_predictions)}"
        )
    selected, risk_reasons = select_prediction_sample(
        pred_sources, sample_size=sample_size, sample_seed=sample_seed
    )

    plan_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for gt in sorted(gt_sources.values(), key=lambda item: item.dataset_id):
        logical_id = f"formula_audit::ground_truth::{gt.dataset_id}"
        request = _with_evidence_hash(
            {
                "audit_scope": "ground_truth_simplification",
                "audit_binding_sha256": _sha256_json(
                    {
                        "source_evaluation_key": gt.evaluation_key,
                        "source_result_sha256": gt.result_sha256,
                        "stored_simplification_outcome": gt.stored_outcome,
                    }
                ),
                "variables": list(gt.variables),
                "allowed_functions": list(gt.allowed_functions),
                "domain_assumptions": dict(gt.domain_assumptions),
                "original_expression": gt.original_expression,
                "candidate_simplified_expression": gt.effective_expression,
                "reference_simplified_expression": None,
                "review_round": 1,
            }
        )
        row = _build_plan_row(
            logical_id=logical_id,
            condition="clean",
            request=request,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            prompt_sha256=prompt_sha256,
            schema_path=schema_path,
            schema_content=schema_content,
            schema_sha256=schema_sha256,
        )
        plan_rows.append(row)
        manifest_rows.append(
            {
                "audit_logical_id": logical_id,
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": "ground_truth_simplification",
                "condition": "clean",
                "dataset_id": gt.dataset_id,
                "sample_role": "required_ground_truth",
                "risk_reasons": [],
                "source_formula_logical_id": gt.logical_id,
                "source_formula_evaluation_key": gt.evaluation_key,
                "source_formula_result_sha256": gt.result_sha256,
                "stored_simplification_outcome": gt.stored_outcome,
                "stored_equivalence_decision": "not_applicable",
            }
        )

    for source in selected:
        reasons = risk_reasons[source.logical_key]
        logical_id = (
            f"formula_audit::prediction::{source.algorithm_slug}::{source.dataset_index}::"
            f"s{source.seed}::{source.condition}"
        )
        gt = gt_sources.get(source.dataset_id)
        if gt is None:
            raise FormulaAuditPlanError(f"{source.logical_key} 缺少 GT source")
        if source.reference_expression != gt.effective_expression:
            raise FormulaAuditPlanError(f"{source.logical_key} 的 reference 与 GT index 不一致")
        request = _with_evidence_hash(
            {
                "audit_scope": "prediction_formula",
                "audit_binding_sha256": _sha256_json(
                    {
                        "source_prediction_evaluation_key": source.pred_evaluation_key,
                        "source_prediction_result_sha256": source.pred_result_sha256,
                        "source_gt_artifact_sha256": source.gt_artifact_sha256,
                        "source_prediction_artifact_sha256": source.pred_artifact_sha256,
                        "stored_simplification_outcome": source.stored_simplification_outcome,
                        "expression_resolution": source.expression_resolution,
                        "stored_equivalence_decision": source.stored_equivalence_decision,
                    }
                ),
                "variables": list(source.variables),
                "allowed_functions": list(source.allowed_functions),
                "domain_assumptions": dict(source.domain_assumptions),
                "original_expression": source.original_expression,
                "candidate_simplified_expression": source.effective_expression,
                "reference_simplified_expression": source.reference_expression,
                "review_round": 1,
            }
        )
        row = _build_plan_row(
            logical_id=logical_id,
            # 审计任务之间无条件依赖；来源 condition 仅保留在控制面 manifest。
            condition="clean",
            request=request,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            prompt_sha256=prompt_sha256,
            schema_path=schema_path,
            schema_content=schema_content,
            schema_sha256=schema_sha256,
        )
        plan_rows.append(row)
        manifest_rows.append(
            {
                "audit_logical_id": logical_id,
                "audit_evaluation_key": row["evaluation_key"],
                "audit_scope": "prediction_formula",
                "logical_key": source.logical_key,
                "condition": source.condition,
                "algorithm": source.algorithm,
                "dataset_id": source.dataset_id,
                "dataset_index": source.dataset_index,
                "seed": source.seed,
                "sample_role": "risk_forced" if reasons else "stratified_fill",
                "risk_reasons": list(reasons),
                "source_formula_logical_id": source.pred_logical_id,
                "source_formula_evaluation_key": source.pred_evaluation_key,
                "source_formula_result_sha256": source.pred_result_sha256,
                "source_equivalence_logical_id": source.equivalence_logical_id,
                "source_gt_logical_id": source.gt_logical_id,
                "source_gt_artifact_sha256": source.gt_artifact_sha256,
                "source_pred_artifact_sha256": source.pred_artifact_sha256,
                "stored_simplification_outcome": source.stored_simplification_outcome,
                "expression_resolution": source.expression_resolution,
                "stored_equivalence_decision": source.stored_equivalence_decision,
            }
        )

    plan_rows.sort(key=lambda row: str(row["logical_id"]))
    manifest_rows.sort(key=lambda row: str(row["audit_logical_id"]))
    plan_path = audit_root / "plans/formula_audit_round1.jsonl"
    manifest_path = audit_root / "manifests/formula_audit_sample.jsonl"
    excluded_path = audit_root / "manifests/formula_audit_excluded.jsonl"
    report_path = audit_root / "reports/formula_audit_plan_report.json"
    _write_jsonl(plan_path, plan_rows)
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(excluded_path, excluded_predictions)
    prediction_manifest = [row for row in manifest_rows if row["audit_scope"] == "prediction_formula"]
    report = {
        "status": "ok",
        "contract_ok": True,
        "sample_seed": sample_seed,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in input_paths.items()
        },
        "outputs": {
            "plan_jsonl": str(plan_path.resolve()),
            "plan_sha256": _sha256_file(plan_path),
            "manifest_jsonl": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "excluded_jsonl": str(excluded_path.resolve()),
            "excluded_sha256": _sha256_file(excluded_path),
        },
        "counts": {
            "ground_truth_audit_count": 50,
            "prediction_universe_count": len(pred_sources),
            "prediction_excluded_count": len(excluded_predictions),
            "prediction_sample_count": len(selected),
            "initial_api_task_count": len(plan_rows),
            "risk_forced_count": sum(row["sample_role"] == "risk_forced" for row in prediction_manifest),
            "stratified_fill_count": sum(
                row["sample_role"] == "stratified_fill" for row in prediction_manifest
            ),
            "condition_counts": dict(Counter(row["condition"] for row in prediction_manifest)),
            "seed_counts": dict(Counter(str(row["seed"]) for row in prediction_manifest)),
            "algorithm_count": len({row["algorithm"] for row in prediction_manifest}),
            "dataset_count": len({row["dataset_id"] for row in prediction_manifest}),
            "algorithm_condition_strata_count": len(
                {(row["algorithm"], row["condition"]) for row in prediction_manifest}
            ),
            "algorithm_condition_seed_strata_count": len(
                {
                    (row["algorithm"], row["condition"], row["seed"])
                    for row in prediction_manifest
                }
            ),
            "dataset_sample_count_min": min(
                Counter(row["dataset_id"] for row in prediction_manifest).values()
            ),
            "dataset_sample_count_max": max(
                Counter(row["dataset_id"] for row in prediction_manifest).values()
            ),
        },
        "budget": {
            "initial_task_count": len(plan_rows),
            "maximum_total_api_attempts": int(len(plan_rows) * 1.5),
        },
        "blindness_contract": {
            "stored_simplification_outcome_in_request": False,
            "stored_equivalence_decision_in_request": False,
            "baseline_verdicts_only_in_manifest": True,
            "source_provenance_in_request": False,
            "opaque_source_binding_in_request": True,
            "execution_condition_is_control_only": True,
        },
        "sampling_design": {
            "type": "risk_targeted_with_deterministic_stratified_fill",
            "forced_risks": [
                "simplification_fallback",
                "equivalence_equivalent",
                "equivalence_undetermined",
            ],
            "fill_stratum": "algorithm_condition_seed",
            "fill_dataset_balancing": True,
            "population_prevalence_inference_allowed": False,
        },
    }
    _write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    stage_root = _stage_root()
    parser = argparse.ArgumentParser(description="构建 Stage5 公式质量盲审计划")
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=stage_root / "audits/formula_quality_1000_0903",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_formula_audit_plan(
            audit_root=args.audit_root.resolve(),
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
        )
    except (FormulaAuditPlanError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
