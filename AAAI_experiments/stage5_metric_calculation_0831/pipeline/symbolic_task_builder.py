"""Stage5 clean 下游符号任务计划构建器。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Collection, Mapping, Sequence

from AAAI_experiments.stage5_metric_calculation_0831.pipeline.claude_contract import (
    evaluation_key,
    render_prompt,
)
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.state import TaskSpec
from AAAI_experiments.stage5_metric_calculation_0831.pipeline.symbolic_evidence import (
    SymbolicEvidenceError,
    build_pair_evidence,
)


STAGE_ROOT_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831")
EQUIVALENCE_PROMPT_RELATIVE = STAGE_ROOT_RELATIVE / "config/prompts/equivalence.v1.txt"
EQUIVALENCE_SCHEMA_RELATIVE = STAGE_ROOT_RELATIVE / "config/schemas/equivalence.v1.json"
STRUCTURE_PROMPT_RELATIVE = STAGE_ROOT_RELATIVE / "config/prompts/structure.v1.txt"
STRUCTURE_SCHEMA_RELATIVE = STAGE_ROOT_RELATIVE / "config/schemas/structure.v1.json"
DEFAULT_OUTPUT_JSONL = STAGE_ROOT_RELATIVE / "reports/clean_symbolic_tasks.jsonl"
DEFAULT_REPORT_JSON = STAGE_ROOT_RELATIVE / "reports/clean_symbolic_task_plan.json"
DEFAULT_NON_APPLICABLE_INDEX_JSONL = STAGE_ROOT_RELATIVE / "reports/clean_symbolic_non_applicable.jsonl"
DEFAULT_NON_APPLICABLE_EVIDENCE_DIR = STAGE_ROOT_RELATIVE / "reports/clean_symbolic_non_applicable"
PLAN_PHASES = ("equivalence", "structure", "all")
GT_SIMPLIFY_TASK_TYPE = "gt_simplify"
PRED_SIMPLIFY_TASK_TYPE = "pred_simplify"
EQUIVALENCE_TASK_TYPE = "equivalence"
STRUCTURE_TASK_TYPE = "stab_structure"
CONDITION = "clean"
EQUIVALENCE_PRIORITY = 30
STRUCTURE_PRIORITY = 40
SEEDS = (520, 521, 522)
SEED_SET = frozenset(SEEDS)
SEED_PAIRS = ((520, 521), (520, 522), (521, 522))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GT_LOGICAL_ID_RE = re.compile(r"^gt_simplify::([^:]+)(?:::(v2))?$")
EQUIVALENCE_GT_UNAVAILABLE = "upstream_gt_unavailable"
EQUIVALENCE_PRED_UNAVAILABLE = "upstream_pred_unavailable"
STRUCTURE_INVALID_SEED_OR_EXPRESSION = "invalid_seed_or_expression"


class SymbolicTaskBuilderError(ValueError):
    """下游符号计划输入、契约或依赖不合法。"""


@dataclass(frozen=True)
class PromptSchemaBundle:
    prompt_path: str
    prompt_version: str
    prompt_template: str
    prompt_sha256: str
    schema_path: str
    schema_version: str
    schema: dict[str, Any]
    schema_sha256: str


@dataclass(frozen=True)
class PlannedTask:
    evaluation_key: str
    logical_id: str
    task_type: str
    condition: str
    priority: int
    input_hash: str
    prompt_version: str
    prompt_sha256: str
    schema_version: str
    schema_sha256: str
    dependencies: tuple[str, ...]
    prompt_path: str
    schema_path: str
    prompt_template: str
    schema_content: dict[str, Any]
    normalized_input: dict[str, Any]
    request: dict[str, Any]

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            evaluation_key=self.evaluation_key,
            logical_id=self.logical_id,
            task_type=self.task_type,
            condition=self.condition,
            priority=self.priority,
            input_hash=self.input_hash,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            dependencies=self.dependencies,
        )

    def to_json_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dependencies"] = list(self.dependencies)
        payload["task_spec"] = json.loads(self.to_task_spec().canonical_json())
        payload["rendered_prompt"] = render_prompt(
            self.prompt_template,
            self.request,
            self.schema_content,
        )
        return payload


@dataclass(frozen=True)
class FrozenSimplifyRecord:
    evaluation_key: str
    logical_id: str
    task_type: str
    plan_sha256: str
    state: str
    simplified_status: str
    simplified_expression: str | None
    structured_output: dict[str, Any] | None
    non_applicable: dict[str, Any] | None
    result_sha256: str | None


@dataclass(frozen=True)
class SimplifyPlanRecord:
    logical_id: str
    task_type: str
    evaluation_key: str
    priority: int
    request: dict[str, Any]


@dataclass(frozen=True)
class PredGroupRecord:
    algorithm: str
    algorithm_slug: str
    dataset_id: str
    dataset_index: str
    by_seed: dict[int, SimplifyPlanRecord]


@dataclass(frozen=True)
class NumericValidityRecord:
    logical_key: str
    algorithm: str
    dataset_id: str
    seed: int
    noise_tag: str
    task_id: str
    result_sha256: str
    valid_output: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(data)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload = b"".join(_canonical_json_bytes(row) for row in rows)
    _atomic_write_bytes(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SymbolicTaskBuilderError(f"{context} 不是 JSON object")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SymbolicTaskBuilderError(f"{context} 必须是非空字符串")
    return value


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SymbolicTaskBuilderError(f"{context} 必须是整数")
    return value


def _require_string_list(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SymbolicTaskBuilderError(f"{context} 必须是非空字符串数组")
    return list(value)


def _optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SymbolicTaskBuilderError(f"{context} 必须是非空字符串或 null")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = _require_string(value, context=context)
    if SHA256_RE.fullmatch(text) is None:
        raise SymbolicTaskBuilderError(f"{context} 必须是 SHA-256 十六进制字符串")
    return text


def _validate_gt_dataset_id(dataset_id: str, *, context: str) -> str:
    if ":" in dataset_id:
        raise SymbolicTaskBuilderError(f"{context} 不能包含冒号，否则 GT logical_id 解析会歧义")
    return dataset_id


def _parse_gt_logical_id(logical_id: str) -> tuple[str, str | None]:
    match = GT_LOGICAL_ID_RE.fullmatch(logical_id)
    if match is None:
        raise SymbolicTaskBuilderError(f"GT logical_id 非 canonical: {logical_id!r}")
    return match.group(1), match.group(2)


def _resolve_maybe_relative(path_text: str, *, repo_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _domain_assumptions(expression: str) -> dict[str, Any]:
    protected_literals = sorted(
        name
        for name in ("maximum", "minimum", "clip", "where")
        if re.search(rf"\b(?:np\.)?{name}\s*\(", expression)
    )
    return {
        "number_system": "real",
        "equivalence_domain": (
            "Compare expressions on their common real-valued domain where both sides "
            "are defined and finite."
        ),
        "implicit_protected_operators": "none",
        "literal_protected_operators": protected_literals,
        "operator_semantics": [
            "Division is ordinary real division; zero denominators are outside the common domain.",
            "log, sqrt, and non-integer powers use ordinary real-domain semantics.",
            "maximum, minimum, clip, and where are literal functions only when written in the expression.",
            "Do not infer hidden clipping, epsilon guards, or fitted constants beyond the frozen expression.",
        ],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SymbolicTaskBuilderError(f"{path}:{line_number} JSONL 解析失败: {exc}") from exc
            rows.append(dict(_require_mapping(parsed, context=f"{path}:{line_number}")))
    return rows


def _load_prompt_schema(repo_root: Path, *, task_kind: str) -> PromptSchemaBundle:
    if task_kind == "equivalence":
        prompt_path = (repo_root / EQUIVALENCE_PROMPT_RELATIVE).resolve()
        schema_path = (repo_root / EQUIVALENCE_SCHEMA_RELATIVE).resolve()
    elif task_kind == "structure":
        prompt_path = (repo_root / STRUCTURE_PROMPT_RELATIVE).resolve()
        schema_path = (repo_root / STRUCTURE_SCHEMA_RELATIVE).resolve()
    else:  # pragma: no cover
        raise SymbolicTaskBuilderError(f"未知 task_kind: {task_kind!r}")
    prompt_bytes = prompt_path.read_bytes()
    schema_bytes = schema_path.read_bytes()
    schema_payload = json.loads(schema_bytes.decode("utf-8"))
    schema = dict(_require_mapping(schema_payload, context=str(schema_path)))
    return PromptSchemaBundle(
        prompt_path=str(prompt_path),
        prompt_version=prompt_path.stem,
        prompt_template=prompt_bytes.decode("utf-8"),
        prompt_sha256=_sha256_bytes(prompt_bytes),
        schema_path=str(schema_path),
        schema_version=schema_path.stem,
        schema=schema,
        schema_sha256=_sha256_bytes(schema_bytes),
    )


def _task_from_request(
    *,
    logical_id: str,
    task_type: str,
    priority: int,
    request: dict[str, Any],
    evidence_hash: str,
    contract: PromptSchemaBundle,
    dependencies: tuple[str, ...],
) -> PlannedTask:
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": contract.prompt_sha256,
        "schema_sha256": contract.schema_sha256,
    }
    input_hash = _sha256_json(normalized_input)
    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=contract.prompt_version,
        schema_version=contract.schema_version,
        prompt_sha256=contract.prompt_sha256,
        schema_sha256=contract.schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=evidence_hash,
    )
    return PlannedTask(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=CONDITION,
        priority=priority,
        input_hash=input_hash,
        prompt_version=contract.prompt_version,
        prompt_sha256=contract.prompt_sha256,
        schema_version=contract.schema_version,
        schema_sha256=contract.schema_sha256,
        dependencies=dependencies,
        prompt_path=contract.prompt_path,
        schema_path=contract.schema_path,
        prompt_template=contract.prompt_template,
        schema_content=contract.schema,
        normalized_input=normalized_input,
        request=request,
    )


def _load_simplify_plan_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen_logical_ids: set[str] = set()
    seen_keys: set[str] = set()
    for row in rows:
        logical_id = _require_string(row.get("logical_id"), context="simplify_plan.logical_id")
        evaluation_key_value = _require_string(
            row.get("evaluation_key"),
            context=f"{logical_id}.evaluation_key",
        )
        if logical_id in seen_logical_ids:
            raise SymbolicTaskBuilderError(f"simplify plan 出现重复 logical_id: {logical_id}")
        if evaluation_key_value in seen_keys:
            raise SymbolicTaskBuilderError(
                f"simplify plan 出现重复 evaluation_key: {evaluation_key_value}"
            )
        seen_logical_ids.add(logical_id)
        seen_keys.add(evaluation_key_value)
    return rows


def _simplify_plan_record(row: Mapping[str, Any]) -> dict[str, Any]:
    logical_id = _require_string(row.get("logical_id"), context="simplify_plan.logical_id")
    task_type = _require_string(row.get("task_type"), context=f"{logical_id}.task_type")
    request = dict(_require_mapping(row.get("request"), context=f"{logical_id}.request"))
    condition = _require_string(row.get("condition"), context=f"{logical_id}.condition")
    if condition != CONDITION:
        raise SymbolicTaskBuilderError(f"{logical_id} 的 condition 必须为 {CONDITION}")
    if task_type == GT_SIMPLIFY_TASK_TYPE:
        dataset_id = _validate_gt_dataset_id(
            _require_string(request.get("dataset_id"), context=f"{logical_id}.dataset_id"),
            context=f"{logical_id}.dataset_id",
        )
        parsed_dataset_id, logical_suffix = _parse_gt_logical_id(logical_id)
        if parsed_dataset_id != dataset_id:
            raise SymbolicTaskBuilderError(
                f"{logical_id}.dataset_id 漂移: {parsed_dataset_id!r} != {dataset_id!r}"
            )
        expected_logical_id = f"{GT_SIMPLIFY_TASK_TYPE}::{dataset_id}"
        if logical_suffix is not None:
            expected_logical_id = f"{expected_logical_id}::{logical_suffix}"
    elif task_type == PRED_SIMPLIFY_TASK_TYPE:
        algorithm_slug = _require_string(
            request.get("algorithm_slug"),
            context=f"{logical_id}.algorithm_slug",
        )
        dataset_index = _require_string(
            request.get("dataset_index"),
            context=f"{logical_id}.dataset_index",
        )
        if re.fullmatch(r"g\d{4}", dataset_index) is None:
            raise SymbolicTaskBuilderError(f"{logical_id}.dataset_index 必须为 g 加四位数字")
        seed = _require_int(request.get("seed"), context=f"{logical_id}.seed")
        if seed not in SEED_SET:
            raise SymbolicTaskBuilderError(f"{logical_id}.seed 必须属于 {SEEDS}")
        if request.get("noise_tag") != CONDITION:
            raise SymbolicTaskBuilderError(f"{logical_id}.noise_tag 必须为 {CONDITION}")
        expected_task_id = f"{algorithm_slug}_s{seed}_{CONDITION}_{dataset_index}"
        task_id = _require_string(request.get("task_id"), context=f"{logical_id}.task_id")
        if task_id != expected_task_id:
            raise SymbolicTaskBuilderError(
                f"{logical_id}.task_id 漂移: {task_id!r} != {expected_task_id!r}"
            )
        expected_logical_id = _pred_logical_id(algorithm_slug, dataset_index, seed)
    else:
        raise SymbolicTaskBuilderError(f"{logical_id} 的 task_type 非法: {task_type!r}")
    if logical_id != expected_logical_id:
        raise SymbolicTaskBuilderError(
            f"simplify plan logical_id 非 canonical: {logical_id!r} != {expected_logical_id!r}"
        )
    evaluation_key_value = _require_string(
        row.get("evaluation_key"),
        context=f"{logical_id}.evaluation_key",
    )
    priority = _require_int(row.get("priority"), context=f"{logical_id}.priority")
    return {
        "logical_id": logical_id,
        "task_type": task_type,
        "request": request,
        "evaluation_key": evaluation_key_value,
        "priority": priority,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SymbolicTaskBuilderError(f"{path} JSON 解析失败: {exc}") from exc
    return dict(_require_mapping(payload, context=str(path)))


def _validate_frozen_summary(
    summary_json: Path,
    *,
    repo_root: Path,
    expected_index_path: Path,
    expected_plan_path: Path,
    expected_row_count: int | None,
) -> str:
    summary = _load_json(summary_json)
    if "status" in summary and summary["status"] != "ok":
        raise SymbolicTaskBuilderError(f"{summary_json} status 必须为 ok")
    output_jsonl = _resolve_maybe_relative(
        _require_string(summary.get("output_jsonl"), context=f"{summary_json}.output_jsonl"),
        repo_root=repo_root,
    )
    if output_jsonl != expected_index_path.resolve():
        raise SymbolicTaskBuilderError(
            f"{summary_json} output_jsonl 与 frozen index 不一致: {output_jsonl} != {expected_index_path.resolve()}"
        )
    output_sha256 = _require_sha256(
        summary.get("output_sha256"),
        context=f"{summary_json}.output_sha256",
    )
    if output_sha256 != _sha256_file(expected_index_path):
        raise SymbolicTaskBuilderError(f"{summary_json} output_sha256 与 frozen index 文件不一致")
    plan_jsonl = _resolve_maybe_relative(
        _require_string(summary.get("plan_jsonl"), context=f"{summary_json}.plan_jsonl"),
        repo_root=repo_root,
    )
    if plan_jsonl != expected_plan_path.resolve():
        raise SymbolicTaskBuilderError(
            f"{summary_json} plan_jsonl 与 simplify plan 不一致: {plan_jsonl} != {expected_plan_path.resolve()}"
        )
    plan_sha256 = _require_sha256(summary.get("plan_sha256"), context=f"{summary_json}.plan_sha256")
    if plan_sha256 != _sha256_file(expected_plan_path):
        raise SymbolicTaskBuilderError(f"{summary_json} plan_sha256 与 simplify plan 文件不一致")
    if expected_row_count is not None:
        row_count = _require_int(summary.get("row_count"), context=f"{summary_json}.row_count")
        if row_count != expected_row_count:
            raise SymbolicTaskBuilderError(
                f"{summary_json} row_count 不符: 期望 {expected_row_count}，实际 {row_count}"
            )
    return plan_sha256


def _load_frozen_index(
    path: Path,
    *,
    expected_task_type: str,
    plan_by_logical_id: Mapping[str, SimplifyPlanRecord],
    expected_plan_sha256: str,
) -> dict[str, FrozenSimplifyRecord]:
    rows = _read_jsonl(path)
    result: dict[str, FrozenSimplifyRecord] = {}
    seen_eval_keys: set[str] = set()
    for row in rows:
        logical_id = _require_string(row.get("logical_id"), context=f"{path}.logical_id")
        plan_record = plan_by_logical_id.get(logical_id)
        if plan_record is None:
            raise SymbolicTaskBuilderError(f"{path} 存在未在 simplify plan 注册的 logical_id: {logical_id}")
        evaluation_key_value = _require_string(
            row.get("evaluation_key"),
            context=f"{logical_id}.evaluation_key",
        )
        if evaluation_key_value != plan_record.evaluation_key:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 evaluation_key 与 simplify plan 不一致")
        if logical_id in result:
            raise SymbolicTaskBuilderError(f"{path} 出现重复 logical_id: {logical_id}")
        if evaluation_key_value in seen_eval_keys:
            raise SymbolicTaskBuilderError(f"{path} 出现重复 evaluation_key: {evaluation_key_value}")
        seen_eval_keys.add(evaluation_key_value)
        plan_sha256 = _require_sha256(row.get("plan_sha256"), context=f"{logical_id}.plan_sha256")
        if plan_sha256 != expected_plan_sha256:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 plan_sha256 与 frozen summary 不一致")
        task_type = _require_string(row.get("task_type"), context=f"{logical_id}.task_type")
        if task_type != expected_task_type:
            raise SymbolicTaskBuilderError(
                f"{logical_id} 的 task_type 应为 {expected_task_type}，实际为 {task_type}"
            )
        if task_type != plan_record.task_type:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 task_type 与 simplify plan 不一致")
        state = _require_string(row.get("state"), context=f"{logical_id}.state")
        if state not in {"frozen", "non_applicable"}:
            raise SymbolicTaskBuilderError(f"{logical_id} frozen index state 非法: {state}")
        simplified_status = state
        simplified_expression: str | None = None
        structured_output = None
        non_applicable = None
        result_sha256 = None
        if state == "frozen":
            structured_output = dict(
                _require_mapping(row.get("structured_output"), context=f"{logical_id}.structured_output")
            )
            outcome = _require_string(structured_output.get("outcome"), context=f"{logical_id}.outcome")
            if outcome not in {"simplified", "unchanged", "unable"}:
                raise SymbolicTaskBuilderError(f"{logical_id} simplify outcome 非法: {outcome}")
            simplified_status = outcome
            expression = structured_output.get("simplified_expression")
            if outcome in {"simplified", "unchanged"}:
                simplified_expression = _require_string(
                    expression,
                    context=f"{logical_id}.simplified_expression",
                )
            elif expression is not None:
                raise SymbolicTaskBuilderError(
                    f"{logical_id} outcome=unable 时 simplified_expression 必须为 null"
                )
            result_sha256 = _require_sha256(
                row.get("result_sha256"),
                context=f"{logical_id}.result_sha256",
            )
        else:
            non_applicable = dict(
                _require_mapping(row.get("non_applicable"), context=f"{logical_id}.non_applicable")
            )
        result[logical_id] = FrozenSimplifyRecord(
            evaluation_key=evaluation_key_value,
            logical_id=logical_id,
            task_type=task_type,
            plan_sha256=plan_sha256,
            state=state,
            simplified_status=simplified_status,
            simplified_expression=simplified_expression,
            structured_output=structured_output,
            non_applicable=non_applicable,
            result_sha256=result_sha256,
        )
    return result


def _structure_logical_id(algorithm_slug: str, dataset_index: str, seed_a: int, seed_b: int) -> str:
    left = min(seed_a, seed_b)
    right = max(seed_a, seed_b)
    return f"{STRUCTURE_TASK_TYPE}::{algorithm_slug}::{dataset_index}::s{left}-s{right}"


def _pred_logical_id(algorithm_slug: str, dataset_index: str, seed: int) -> str:
    return f"{PRED_SIMPLIFY_TASK_TYPE}::{algorithm_slug}::{dataset_index}::s{seed}::{CONDITION}"


def _parse_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise SymbolicTaskBuilderError(f"{context} 必须是 true/false")


def _run_logical_key(algorithm: str, dataset_id: str, seed: int) -> str:
    return f"{algorithm}::{dataset_id}::s{seed}::{CONDITION}"


def _load_numeric_validity(
    path: Path,
    *,
    pred_plan_by_logical_id: Mapping[str, SimplifyPlanRecord],
    expected_row_count: int | None,
) -> tuple[dict[str, NumericValidityRecord], dict[str, Any]]:
    required_fields = {
        "logical_key",
        "algorithm",
        "dataset_id",
        "seed",
        "noise_tag",
        "task_id",
        "result_sha256",
        "valid_output",
    }
    result: dict[str, NumericValidityRecord] = {}
    valid_counts = {"true": 0, "false": 0}
    pred_lookup: dict[tuple[str, str, int], str] = {}
    for logical_id, plan_record in pred_plan_by_logical_id.items():
        request = plan_record.request
        pred_lookup[
            (
                _require_string(request.get("algorithm"), context=f"{logical_id}.request.algorithm"),
                _require_string(request.get("dataset_id"), context=f"{logical_id}.request.dataset_id"),
                _require_int(request.get("seed"), context=f"{logical_id}.request.seed"),
            )
        ] = logical_id
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(required_fields.difference(reader.fieldnames or ()))
        if missing_fields:
            raise SymbolicTaskBuilderError(f"clean_numeric_run_metrics.csv 缺少字段: {missing_fields}")
        for line_number, row in enumerate(reader, start=2):
            logical_key = _require_string(row.get("logical_key"), context=f"{path}:{line_number}.logical_key")
            algorithm = _require_string(row.get("algorithm"), context=f"{path}:{line_number}.algorithm")
            dataset_id = _require_string(row.get("dataset_id"), context=f"{path}:{line_number}.dataset_id")
            seed_text = _require_string(row.get("seed"), context=f"{path}:{line_number}.seed")
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise SymbolicTaskBuilderError(f"{path}:{line_number}.seed 必须是整数") from exc
            expected_logical_key = _run_logical_key(algorithm, dataset_id, seed)
            if logical_key != expected_logical_key:
                raise SymbolicTaskBuilderError(
                    f"{path}:{line_number}.logical_key 不一致: {logical_key!r} != {expected_logical_key!r}"
                )
            noise_tag = _require_string(row.get("noise_tag"), context=f"{path}:{line_number}.noise_tag")
            if noise_tag != CONDITION:
                raise SymbolicTaskBuilderError(f"{path}:{line_number}.noise_tag 必须为 clean")
            record = NumericValidityRecord(
                logical_key=logical_key,
                algorithm=algorithm,
                dataset_id=dataset_id,
                seed=seed,
                noise_tag=noise_tag,
                task_id=_require_string(row.get("task_id"), context=f"{path}:{line_number}.task_id"),
                result_sha256=_require_sha256(
                    row.get("result_sha256"),
                    context=f"{path}:{line_number}.result_sha256",
                ),
                valid_output=_parse_bool(
                    row.get("valid_output"),
                    context=f"{path}:{line_number}.valid_output",
                ),
            )
            pred_logical_id = pred_lookup.get((algorithm, dataset_id, seed))
            if pred_logical_id is None:
                raise SymbolicTaskBuilderError(
                    f"{path}:{line_number} 无法映射到 pred simplify plan: {logical_key}"
                )
            if pred_logical_id in result:
                raise SymbolicTaskBuilderError(f"clean_numeric_run_metrics.csv 出现重复 logical_key: {logical_key}")
            result[pred_logical_id] = record
            valid_counts["true" if record.valid_output else "false"] += 1
    if expected_row_count is not None and len(result) != expected_row_count:
        raise SymbolicTaskBuilderError(
            f"clean_numeric_run_metrics.csv 行数不符: 期望 {expected_row_count}，实际 {len(result)}"
        )
    for logical_id, plan_record in pred_plan_by_logical_id.items():
        if logical_id not in result:
            raise SymbolicTaskBuilderError(f"clean_numeric_run_metrics.csv 缺少 pred 记录: {logical_id}")
        record = result[logical_id]
        request = plan_record.request
        expected_task_id = _require_string(request.get("task_id"), context=f"{logical_id}.request.task_id")
        if record.task_id != expected_task_id:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 numeric task_id 与 pred plan 不一致")
        expected_algorithm = _require_string(request.get("algorithm"), context=f"{logical_id}.request.algorithm")
        expected_dataset_id = _require_string(request.get("dataset_id"), context=f"{logical_id}.request.dataset_id")
        expected_seed = _require_int(request.get("seed"), context=f"{logical_id}.request.seed")
        if (record.algorithm, record.dataset_id, record.seed) != (
            expected_algorithm,
            expected_dataset_id,
            expected_seed,
        ):
            raise SymbolicTaskBuilderError(f"{logical_id} 的 numeric identity 与 pred plan 不一致")
    report = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "row_count": len(result),
        "valid_true_count": valid_counts["true"],
        "valid_false_count": valid_counts["false"],
    }
    return result, report


def _task_json_record(task: PlannedTask) -> dict[str, Any]:
    payload = asdict(task)
    payload["dependencies"] = list(task.dependencies)
    payload["task_spec"] = json.loads(task.to_task_spec().canonical_json())
    payload["rendered_prompt"] = render_prompt(
        task.prompt_template,
        task.request,
        task.schema_content,
    )
    return payload


def _has_callable_expression(record: FrozenSimplifyRecord) -> bool:
    return record.state == "frozen" and record.simplified_status in {"simplified", "unchanged"} and bool(
        record.simplified_expression
    )


def _as_plan_records(rows: Sequence[dict[str, Any]], *, expected_task_type: str) -> dict[str, SimplifyPlanRecord]:
    result: dict[str, SimplifyPlanRecord] = {}
    for row in rows:
        logical_id = _require_string(row.get("logical_id"), context="simplify_plan.logical_id")
        task_type = _require_string(row.get("task_type"), context=f"{logical_id}.task_type")
        if task_type != expected_task_type:
            raise SymbolicTaskBuilderError(
                f"{logical_id} 的 task_type 应为 {expected_task_type}，实际为 {task_type}"
            )
        if logical_id in result:
            raise SymbolicTaskBuilderError(f"simplify plan 出现重复 logical_id: {logical_id}")
        result[logical_id] = SimplifyPlanRecord(
            logical_id=logical_id,
            task_type=task_type,
            evaluation_key=_require_string(row.get("evaluation_key"), context=f"{logical_id}.evaluation_key"),
            priority=_require_int(row.get("priority"), context=f"{logical_id}.priority"),
            request=dict(_require_mapping(row.get("request"), context=f"{logical_id}.request")),
        )
    return result


def _split_simplify_plan_rows(
    *,
    simplify_plan_jsonl: Path | None,
    gt_plan_jsonl: Path | None,
    pred_plan_jsonl: Path | None,
) -> tuple[dict[str, SimplifyPlanRecord], dict[str, SimplifyPlanRecord], Path, Path]:
    if simplify_plan_jsonl is not None:
        if gt_plan_jsonl is not None or pred_plan_jsonl is not None:
            raise SymbolicTaskBuilderError("不能同时传 simplify_plan_jsonl 与 gt/pred 分离 plan")
        rows = [_simplify_plan_record(row) for row in _load_simplify_plan_rows(simplify_plan_jsonl)]
        gt_rows = [row for row in rows if row["task_type"] == GT_SIMPLIFY_TASK_TYPE]
        pred_rows = [row for row in rows if row["task_type"] == PRED_SIMPLIFY_TASK_TYPE]
        return (
            _as_plan_records(gt_rows, expected_task_type=GT_SIMPLIFY_TASK_TYPE),
            _as_plan_records(pred_rows, expected_task_type=PRED_SIMPLIFY_TASK_TYPE),
            simplify_plan_jsonl,
            simplify_plan_jsonl,
        )
    if gt_plan_jsonl is None or pred_plan_jsonl is None:
        raise SymbolicTaskBuilderError("必须提供 simplify_plan_jsonl，或同时提供 gt_plan_jsonl 与 pred_plan_jsonl")
    gt_rows = [_simplify_plan_record(row) for row in _load_simplify_plan_rows(gt_plan_jsonl)]
    pred_rows = [_simplify_plan_record(row) for row in _load_simplify_plan_rows(pred_plan_jsonl)]
    return (
        _as_plan_records(gt_rows, expected_task_type=GT_SIMPLIFY_TASK_TYPE),
        _as_plan_records(pred_rows, expected_task_type=PRED_SIMPLIFY_TASK_TYPE),
        gt_plan_jsonl,
        pred_plan_jsonl,
    )


def _request_probe_bundle(request: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    dataset_probe = request.get("dataset_probe_evidence")
    if dataset_probe is None:
        deterministic = request.get("deterministic_evidence")
        if isinstance(deterministic, Mapping):
            dataset_probe = deterministic.get("dataset_probe")
        elif isinstance(request.get("deterministic_symbolic_evidence"), Mapping):
            dataset_probe = request["deterministic_symbolic_evidence"].get("dataset_probe")
    probe = dict(_require_mapping(dataset_probe, context=f"{context}.dataset_probe_evidence"))
    points = probe.get("points")
    if not isinstance(points, list) or not points:
        raise SymbolicTaskBuilderError(f"{context}.dataset_probe_evidence.points 缺失")
    probe_source = _require_string(
        request.get("probe_source", probe.get("schema_version")),
        context=f"{context}.probe_source",
    )
    probe_sample_sha256 = _require_sha256(
        request.get("probe_sample_sha256", probe.get("sample_sha256")),
        context=f"{context}.probe_sample_sha256",
    )
    evidence_sha256 = _require_sha256(
        probe.get("evidence_sha256"),
        context=f"{context}.dataset_probe_evidence.evidence_sha256",
    )
    variables = probe.get("variables")
    if not isinstance(variables, list) or not all(isinstance(item, str) and item for item in variables):
        raise SymbolicTaskBuilderError(f"{context}.dataset_probe_evidence.variables 非法")
    return {
        "dataset_probe": probe,
        "probe_points": list(points),
        "probe_source": probe_source,
        "probe_sample_sha256": probe_sample_sha256,
        "probe_evidence_sha256": evidence_sha256,
        "variables": list(variables),
    }


def _request_allowed_functions(request: Mapping[str, Any], *, context: str) -> list[str]:
    raw = request.get("allowed_functions")
    if raw is None:
        return []
    return _require_string_list(raw, context=f"{context}.allowed_functions")


def _phase_task_type(phase: str) -> str:
    if phase == "equivalence":
        return EQUIVALENCE_TASK_TYPE
    if phase == "structure":
        return STRUCTURE_TASK_TYPE
    raise SymbolicTaskBuilderError(f"未知 phase: {phase!r}")


def _phase_callable_tasks(tasks: Sequence[PlannedTask], *, phase: str) -> list[PlannedTask]:
    if phase == "all":
        return list(tasks)
    expected_task_type = _phase_task_type(phase)
    return [task for task in tasks if task.task_type == expected_task_type]


def _phase_no_call_records(rows: Sequence[Mapping[str, Any]], *, phase: str) -> list[dict[str, Any]]:
    if phase == "all":
        selected = [dict(row) for row in rows]
    else:
        selected = [dict(row) for row in rows if row.get("phase") == phase]
    selected.sort(key=lambda item: (int(item["priority"]), str(item["logical_id"])))
    return selected


def _phase_full_plan_rows(
    tasks: Sequence[PlannedTask],
    no_call_records: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    callable_rows = [_task_json_record(task) for task in _phase_callable_tasks(tasks, phase=phase)]
    no_call_rows = _phase_no_call_records(no_call_records, phase=phase)
    rows = [*callable_rows, *no_call_rows]
    rows.sort(key=lambda item: (int(item["priority"]), str(item["logical_id"])))
    return rows


def _request_domain_assumptions(
    request: Mapping[str, Any],
    *,
    expression: str,
    context: str,
) -> dict[str, Any]:
    raw = request.get("domain_assumptions")
    if raw is None:
        return _domain_assumptions(expression)
    return dict(_require_mapping(raw, context=f"{context}.domain_assumptions"))


def _symbolic_artifact_sha(request: Mapping[str, Any], *, context: str) -> str | None:
    for key in ("deterministic_evidence", "deterministic_symbolic_evidence"):
        raw = request.get(key)
        if not isinstance(raw, Mapping):
            continue
        artifact = raw.get("symbolic_artifact")
        if not isinstance(artifact, Mapping):
            continue
        artifact_sha = artifact.get("artifact_sha256")
        if artifact_sha is None:
            continue
        return _require_sha256(artifact_sha, context=f"{context}.{key}.symbolic_artifact.artifact_sha256")
    return None


def _shared_probe_bundle(
    left_request: Mapping[str, Any],
    right_request: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    left = _request_probe_bundle(left_request, context=f"{context}.left")
    right = _request_probe_bundle(right_request, context=f"{context}.right")
    if left["probe_evidence_sha256"] != right["probe_evidence_sha256"]:
        raise SymbolicTaskBuilderError(f"{context} 左右 dataset probe 证据不一致")
    if left["variables"] != right["variables"]:
        raise SymbolicTaskBuilderError(f"{context} 左右 dataset probe variables 不一致")
    return left


def _upstream_binding(
    *,
    plan_record: SimplifyPlanRecord,
    frozen_record: FrozenSimplifyRecord,
    role: str,
) -> dict[str, Any]:
    request = plan_record.request
    return {
        "role": role,
        "plan_logical_id": plan_record.logical_id,
        "plan_evaluation_key": plan_record.evaluation_key,
        "plan_request_evidence_hash": _require_string(
            request.get("evidence_hash"),
            context=f"{plan_record.logical_id}.request.evidence_hash",
        ),
        "plan_source_expression": _optional_string(
            request.get("expression"),
            context=f"{plan_record.logical_id}.request.expression",
        ),
        "plan_original_expression": _optional_string(
            request.get("original_expression"),
            context=f"{plan_record.logical_id}.request.original_expression",
        ),
        "plan_symbolic_artifact_sha256": _symbolic_artifact_sha(
            request,
            context=plan_record.logical_id,
        ),
        "frozen_plan_sha256": frozen_record.plan_sha256,
        "frozen_logical_id": frozen_record.logical_id,
        "frozen_evaluation_key": frozen_record.evaluation_key,
        "frozen_state": frozen_record.state,
        "frozen_simplified_status": frozen_record.simplified_status,
        "frozen_simplified_expression": frozen_record.simplified_expression,
        "frozen_structured_output_sha256": (
            _sha256_json(frozen_record.structured_output)
            if frozen_record.structured_output is not None
            else None
        ),
        "frozen_result_sha256": frozen_record.result_sha256,
    }


def _build_full_pair_evidence(
    *,
    logical_id: str,
    phase: str,
    left_plan: SimplifyPlanRecord,
    left_frozen: FrozenSimplifyRecord,
    right_plan: SimplifyPlanRecord,
    right_frozen: FrozenSimplifyRecord,
    pair_seed: int,
    pair_request_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lhs = _require_string(left_frozen.simplified_expression, context=f"{logical_id}.lhs_expression")
    rhs = _require_string(right_frozen.simplified_expression, context=f"{logical_id}.rhs_expression")
    shared_probe = _shared_probe_bundle(left_plan.request, right_plan.request, context=logical_id)
    variables = shared_probe["variables"]
    allowed_functions = sorted(
        set(_request_allowed_functions(left_plan.request, context=left_plan.logical_id))
        | set(_request_allowed_functions(right_plan.request, context=right_plan.logical_id))
    )
    pair_domain_assumptions = {
        "lhs": _request_domain_assumptions(
            left_plan.request,
            expression=lhs,
            context=left_plan.logical_id,
        ),
        "rhs": _request_domain_assumptions(
            right_plan.request,
            expression=rhs,
            context=right_plan.logical_id,
        ),
    }
    try:
        pair_evidence = build_pair_evidence(
            lhs,
            rhs,
            allowed_variables=variables,
            allowed_functions=allowed_functions,
            seed=pair_seed,
            probe_points=shared_probe["probe_points"],
            probe_source=str(shared_probe["probe_source"]),
            probe_sample_sha256=str(shared_probe["probe_sample_sha256"]),
        )
    except SymbolicEvidenceError as exc:
        raise SymbolicTaskBuilderError(f"{logical_id} 构建 pair evidence 失败: {exc}") from exc
    payload = {
        "schema_version": "symbolic_pair_evidence.v2",
        "phase": phase,
        "pair_seed": pair_seed,
        "pair_evidence": pair_evidence,
        "dataset_probe": shared_probe["dataset_probe"],
        "domain_assumptions": pair_domain_assumptions,
        "allowed_variables": variables,
        "allowed_functions": allowed_functions,
        "lhs_binding": _upstream_binding(
            plan_record=left_plan,
            frozen_record=left_frozen,
            role="lhs",
        ),
        "rhs_binding": _upstream_binding(
            plan_record=right_plan,
            frozen_record=right_frozen,
            role="rhs",
        ),
    }
    payload["evidence_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )
    return payload


def _build_non_applicable_evidence(
    *,
    logical_id: str,
    task_type: str,
    phase: str,
    reason: str,
    dependencies: tuple[str, ...],
    request_context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "symbolic_non_applicable.v1",
        "logical_id": logical_id,
        "task_type": task_type,
        "phase": phase,
        "condition": CONDITION,
        "reason": reason,
        "dependencies": list(dependencies),
        "request_context": dict(request_context),
    }


def _equivalence_no_call_reason(
    *,
    gt_record: FrozenSimplifyRecord,
    pred_record: FrozenSimplifyRecord,
) -> str | None:
    if not _has_callable_expression(gt_record):
        return EQUIVALENCE_GT_UNAVAILABLE
    if not _has_callable_expression(pred_record):
        return EQUIVALENCE_PRED_UNAVAILABLE
    return None


def _materialize_non_applicable_evidence(
    *,
    evidence_dir: Path,
    task_type: str,
    logical_id: str,
    phase: str,
    reason: str,
    request_context: Mapping[str, Any],
    dependencies: tuple[str, ...],
    contract: PromptSchemaBundle,
    write_evidence: bool,
) -> dict[str, Any]:
    evidence_payload = _build_non_applicable_evidence(
        logical_id=logical_id,
        task_type=task_type,
        phase=phase,
        reason=reason,
        dependencies=dependencies,
        request_context=request_context,
    )
    evidence_bytes = _canonical_json_bytes(evidence_payload)
    evidence_sha256 = _sha256_bytes(evidence_bytes)
    request = {**dict(request_context), "evidence_hash": evidence_sha256}
    normalized_input = {
        "request": request,
        "prompt_sha256": contract.prompt_sha256,
        "schema_sha256": contract.schema_sha256,
    }
    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=contract.prompt_version,
        schema_version=contract.schema_version,
        prompt_sha256=contract.prompt_sha256,
        schema_sha256=contract.schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=evidence_sha256,
    )
    evidence_path = (evidence_dir / f"{task_key}.json").resolve()
    if write_evidence:
        _atomic_write_bytes(evidence_path, evidence_bytes)
        written_sha = _sha256_file(evidence_path)
        if written_sha != evidence_sha256:
            raise SymbolicTaskBuilderError(f"{logical_id} non_applicable 证据文件 SHA 漂移")
    return _no_call_record(
        logical_id=logical_id,
        task_type=task_type,
        phase=phase,
        reason=reason,
        request=request,
        evidence_hash=evidence_sha256,
        evidence_payload=evidence_payload,
        contract=contract,
        dependencies=dependencies,
        evidence_path=str(evidence_path),
        evidence_sha256=evidence_sha256,
    )


def _no_call_record(
    *,
    logical_id: str,
    task_type: str,
    phase: str,
    reason: str,
    request: dict[str, Any],
    evidence_hash: str,
    evidence_payload: Mapping[str, Any],
    contract: PromptSchemaBundle,
    dependencies: tuple[str, ...],
    evidence_path: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    normalized_input = {
        "request": dict(request),
        "prompt_sha256": contract.prompt_sha256,
        "schema_sha256": contract.schema_sha256,
    }
    task_key = evaluation_key(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=contract.prompt_version,
        schema_version=contract.schema_version,
        prompt_sha256=contract.prompt_sha256,
        schema_sha256=contract.schema_sha256,
        normalized_input=normalized_input,
        evidence_hash=evidence_hash,
    )
    task_spec = TaskSpec(
        evaluation_key=task_key,
        logical_id=logical_id,
        task_type=task_type,
        condition=CONDITION,
        priority=EQUIVALENCE_PRIORITY if task_type == EQUIVALENCE_TASK_TYPE else STRUCTURE_PRIORITY,
        input_hash=_sha256_json(normalized_input),
        prompt_version=contract.prompt_version,
        schema_version=contract.schema_version,
        dependencies=dependencies,
    )
    rendered_prompt = render_prompt(
        contract.prompt_template,
        request,
        contract.schema,
    )
    return {
        "phase": phase,
        "logical_id": logical_id,
        "task_type": task_type,
        "condition": CONDITION,
        "priority": task_spec.priority,
        "reason": reason,
        "evaluation_key": task_key,
        "input_hash": task_spec.input_hash,
        "evidence_hash": evidence_hash,
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "prompt_version": contract.prompt_version,
        "prompt_sha256": contract.prompt_sha256,
        "schema_version": contract.schema_version,
        "schema_sha256": contract.schema_sha256,
        "dependencies": list(dependencies),
        "prompt_path": contract.prompt_path,
        "schema_path": contract.schema_path,
        "prompt_template": contract.prompt_template,
        "schema_content": contract.schema,
        "normalized_input": normalized_input,
        "request": dict(request),
        "request_context": {key: value for key, value in request.items() if key != "evidence_hash"},
        "task_spec": json.loads(task_spec.canonical_json()),
        "rendered_prompt": rendered_prompt,
        "evidence_payload": dict(evidence_payload),
        "status": "planned_non_applicable",
    }


def _derive_pred_groups(
    pred_plan_by_logical_id: Mapping[str, SimplifyPlanRecord],
) -> dict[tuple[str, str], PredGroupRecord]:
    grouped: dict[tuple[str, str], dict[int, SimplifyPlanRecord]] = {}
    group_meta: dict[tuple[str, str], tuple[str, str]] = {}
    for logical_id, plan_record in pred_plan_by_logical_id.items():
        request = plan_record.request
        algorithm = _require_string(request.get("algorithm"), context=f"{logical_id}.algorithm")
        algorithm_slug = _require_string(
            request.get("algorithm_slug"),
            context=f"{logical_id}.algorithm_slug",
        )
        dataset_id = _require_string(request.get("dataset_id"), context=f"{logical_id}.dataset_id")
        dataset_index = _require_string(
            request.get("dataset_index"),
            context=f"{logical_id}.dataset_index",
        )
        seed = _require_int(request.get("seed"), context=f"{logical_id}.seed")
        if seed not in SEED_SET:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 seed 必须属于 {SEEDS}")
        key = (algorithm_slug, dataset_index)
        existing_meta = group_meta.get(key)
        if existing_meta is None:
            group_meta[key] = (algorithm, dataset_id)
        elif existing_meta != (algorithm, dataset_id):
            raise SymbolicTaskBuilderError(f"{logical_id} 的 algorithm/dataset 与同组 pred plan 不一致")
        by_seed = grouped.setdefault(key, {})
        if seed in by_seed:
            raise SymbolicTaskBuilderError(f"{logical_id} 所属组出现重复 seed={seed}")
        by_seed[seed] = plan_record
    result: dict[tuple[str, str], PredGroupRecord] = {}
    for key, by_seed in grouped.items():
        seen_seeds = frozenset(by_seed)
        if seen_seeds != SEED_SET:
            raise SymbolicTaskBuilderError(
                f"{key[0]}::{key[1]} 的 seeds 必须严格为 {sorted(SEED_SET)}，实际为 {sorted(seen_seeds)}"
            )
        algorithm, dataset_id = group_meta[key]
        result[key] = PredGroupRecord(
            algorithm=algorithm,
            algorithm_slug=key[0],
            dataset_id=dataset_id,
            dataset_index=key[1],
            by_seed=dict(sorted(by_seed.items())),
        )
    return result


def _structure_pairs_from_pred_groups(
    pred_groups: Mapping[tuple[str, str], PredGroupRecord],
) -> list[tuple[PredGroupRecord, int, int]]:
    pairs: list[tuple[PredGroupRecord, int, int]] = []
    for group_key in sorted(pred_groups):
        group = pred_groups[group_key]
        for seed_a, seed_b in SEED_PAIRS:
            pairs.append((group, seed_a, seed_b))
    return pairs


def build_symbolic_task_plan(
    *,
    gt_frozen_index_jsonl: Path,
    pred_frozen_index_jsonl: Path,
    gt_frozen_summary_json: Path,
    pred_frozen_summary_json: Path,
    clean_run_metrics_csv: Path,
    non_applicable_evidence_dir: Path,
    simplify_plan_jsonl: Path | None = None,
    gt_plan_jsonl: Path | None = None,
    pred_plan_jsonl: Path | None = None,
    phase: str = "all",
    repo_root: Path | None = None,
    expected_gt_count: int | None = 50,
    expected_pred_count: int | None = 2250,
    expected_pair_count: int | None = 2250,
    write_non_applicable_evidence: bool = True,
) -> tuple[list[PlannedTask], dict[str, Any]]:
    if phase not in PLAN_PHASES:
        raise SymbolicTaskBuilderError(f"未知 phase: {phase!r}")
    if gt_frozen_summary_json is None or pred_frozen_summary_json is None:
        raise SymbolicTaskBuilderError("gt_frozen_summary_json 与 pred_frozen_summary_json 为必填")
    if clean_run_metrics_csv is None:
        raise SymbolicTaskBuilderError("clean_run_metrics_csv 为必填")
    if non_applicable_evidence_dir is None:
        raise SymbolicTaskBuilderError("non_applicable_evidence_dir 为必填")
    repo_root = repo_root.resolve() if repo_root is not None else _repo_root()
    equivalence_contract = _load_prompt_schema(repo_root, task_kind="equivalence")
    structure_contract = _load_prompt_schema(repo_root, task_kind="structure")
    gt_plan_by_logical_id, pred_plan_by_logical_id, gt_plan_path, pred_plan_path = _split_simplify_plan_rows(
        simplify_plan_jsonl=simplify_plan_jsonl,
        gt_plan_jsonl=gt_plan_jsonl,
        pred_plan_jsonl=pred_plan_jsonl,
    )
    gt_plan_by_dataset: dict[str, SimplifyPlanRecord] = {}
    for row in gt_plan_by_logical_id.values():
        dataset_id = _require_string(row.request.get("dataset_id"), context=f"{row.logical_id}.dataset_id")
        if dataset_id in gt_plan_by_dataset:
            raise SymbolicTaskBuilderError(f"simplify plan GT dataset_id 重复: {dataset_id}")
        gt_plan_by_dataset[dataset_id] = row
    gt_plan_sha256 = _validate_frozen_summary(
        gt_frozen_summary_json,
        repo_root=repo_root,
        expected_index_path=gt_frozen_index_jsonl.resolve(),
        expected_plan_path=gt_plan_path.resolve(),
        expected_row_count=expected_gt_count,
    )
    pred_plan_sha256 = _validate_frozen_summary(
        pred_frozen_summary_json,
        repo_root=repo_root,
        expected_index_path=pred_frozen_index_jsonl.resolve(),
        expected_plan_path=pred_plan_path.resolve(),
        expected_row_count=expected_pred_count,
    )

    pred_groups = _derive_pred_groups(pred_plan_by_logical_id)
    gt_frozen = _load_frozen_index(
        gt_frozen_index_jsonl,
        expected_task_type=GT_SIMPLIFY_TASK_TYPE,
        plan_by_logical_id=gt_plan_by_logical_id,
        expected_plan_sha256=gt_plan_sha256,
    )
    pred_frozen = _load_frozen_index(
        pred_frozen_index_jsonl,
        expected_task_type=PRED_SIMPLIFY_TASK_TYPE,
        plan_by_logical_id=pred_plan_by_logical_id,
        expected_plan_sha256=pred_plan_sha256,
    )
    numeric_validity_by_logical_id, numeric_validity_report = _load_numeric_validity(
        clean_run_metrics_csv.resolve(),
        pred_plan_by_logical_id=pred_plan_by_logical_id,
        expected_row_count=expected_pred_count,
    )
    non_applicable_dir = non_applicable_evidence_dir.resolve()
    derived_structure_pairs = _structure_pairs_from_pred_groups(pred_groups)

    if expected_gt_count is not None and len(gt_frozen) != expected_gt_count:
        raise SymbolicTaskBuilderError(
            f"GT frozen index 行数不符: 期望 {expected_gt_count}，实际 {len(gt_frozen)}"
        )
    if expected_pred_count is not None and len(pred_frozen) != expected_pred_count:
        raise SymbolicTaskBuilderError(
            f"pred frozen index 行数不符: 期望 {expected_pred_count}，实际 {len(pred_frozen)}"
        )
    if expected_pair_count is not None and len(derived_structure_pairs) != expected_pair_count:
        raise SymbolicTaskBuilderError(
            f"派生 structure pair 数量不符: 期望 {expected_pair_count}，实际 {len(derived_structure_pairs)}"
        )

    for dataset_id, gt_row in gt_plan_by_dataset.items():
        if gt_row.logical_id not in gt_frozen:
            raise SymbolicTaskBuilderError(f"GT frozen index 缺少上游依赖: {dataset_id}")
    for logical_id in pred_plan_by_logical_id:
        if logical_id not in pred_frozen:
            raise SymbolicTaskBuilderError(f"pred frozen index 缺少上游依赖: {logical_id}")

    equivalence_tasks: list[PlannedTask] = []
    structure_tasks: list[PlannedTask] = []
    no_call_records: list[dict[str, Any]] = []

    for logical_id in sorted(pred_plan_by_logical_id):
        pred_plan = pred_plan_by_logical_id[logical_id]
        pred_request = dict(pred_plan.request)
        dataset_id = _require_string(pred_request.get("dataset_id"), context=f"{logical_id}.dataset_id")
        dataset_index = _require_string(pred_request.get("dataset_index"), context=f"{logical_id}.dataset_index")
        algorithm = _require_string(pred_request.get("algorithm"), context=f"{logical_id}.algorithm")
        algorithm_slug = _require_string(pred_request.get("algorithm_slug"), context=f"{logical_id}.algorithm_slug")
        seed = int(pred_request["seed"])
        gt_plan = gt_plan_by_dataset.get(dataset_id)
        if gt_plan is None:
            raise SymbolicTaskBuilderError(f"{logical_id} 找不到对应 GT simplify plan: {dataset_id}")
        gt_record = gt_frozen[gt_plan.logical_id]
        pred_record = pred_frozen[logical_id]
        numeric_record = numeric_validity_by_logical_id[logical_id]
        if pred_record.result_sha256 is not None and numeric_record.result_sha256 != pred_record.result_sha256:
            raise SymbolicTaskBuilderError(f"{logical_id} 的 numeric result_sha256 与 frozen result 不一致")
        eq_logical_id = f"{EQUIVALENCE_TASK_TYPE}::{algorithm_slug}::{dataset_index}::s{seed}::{CONDITION}"
        dependencies = (gt_record.evaluation_key, pred_record.evaluation_key)
        request_context = {
            "dataset_id": dataset_id,
            "dataset_index": dataset_index,
            "algorithm": algorithm,
            "algorithm_slug": algorithm_slug,
            "seed": seed,
            "noise_tag": CONDITION,
            "variables": list(pred_request.get("variables", [])),
            "allowed_functions": list(pred_request.get("allowed_functions", [])),
            "ground_truth_logical_id": gt_plan.logical_id,
            "ground_truth_plan_evaluation_key": gt_plan.evaluation_key,
            "ground_truth_frozen_plan_sha256": gt_record.plan_sha256,
            "ground_truth_frozen_evaluation_key": gt_record.evaluation_key,
            "ground_truth_simplify_status": gt_record.simplified_status,
            "simplified_ground_truth_expression": gt_record.simplified_expression,
            "prediction_logical_id": pred_plan.logical_id,
            "prediction_plan_evaluation_key": pred_plan.evaluation_key,
            "prediction_frozen_plan_sha256": pred_record.plan_sha256,
            "prediction_frozen_evaluation_key": pred_record.evaluation_key,
            "prediction_simplify_status": pred_record.simplified_status,
            "simplified_prediction_expression": pred_record.simplified_expression,
            "prediction_task_id": numeric_record.task_id,
            "prediction_valid_output": numeric_record.valid_output,
            "prediction_result_sha256": numeric_record.result_sha256,
        }
        reason = _equivalence_no_call_reason(gt_record=gt_record, pred_record=pred_record)
        if reason is not None:
            no_call_records.append(
                _materialize_non_applicable_evidence(
                    evidence_dir=non_applicable_dir,
                    logical_id=eq_logical_id,
                    task_type=EQUIVALENCE_TASK_TYPE,
                    phase="equivalence",
                    reason=reason,
                    contract=equivalence_contract,
                    dependencies=dependencies,
                    request_context=request_context,
                    write_evidence=write_non_applicable_evidence,
                )
            )
            continue
        evidence_payload = _build_full_pair_evidence(
            logical_id=eq_logical_id,
            phase="equivalence",
            left_plan=gt_plan,
            left_frozen=gt_record,
            right_plan=pred_plan,
            right_frozen=pred_record,
            pair_seed=seed,
        )
        evidence_hash = _require_sha256(
            evidence_payload.get("evidence_sha256"),
            context=f"{eq_logical_id}.evidence_sha256",
        )
        request = {
            **request_context,
            "deterministic_evidence": evidence_payload,
            "evidence_hash": evidence_hash,
        }
        equivalence_tasks.append(
            _task_from_request(
                logical_id=eq_logical_id,
                task_type=EQUIVALENCE_TASK_TYPE,
                priority=EQUIVALENCE_PRIORITY,
                request=request,
                evidence_hash=evidence_hash,
                contract=equivalence_contract,
                dependencies=dependencies,
            )
        )

    for group, seed_a, seed_b in derived_structure_pairs:
        logical_id = _structure_logical_id(group.algorithm_slug, group.dataset_index, seed_a, seed_b)
        pred_plan_a = group.by_seed[seed_a]
        pred_plan_b = group.by_seed[seed_b]
        pred_record_a = pred_frozen[pred_plan_a.logical_id]
        pred_record_b = pred_frozen[pred_plan_b.logical_id]
        numeric_a = numeric_validity_by_logical_id[pred_plan_a.logical_id]
        numeric_b = numeric_validity_by_logical_id[pred_plan_b.logical_id]
        if pred_record_a.result_sha256 is not None and pred_record_a.result_sha256 != numeric_a.result_sha256:
            raise SymbolicTaskBuilderError(f"{pred_plan_a.logical_id} 的 numeric result_sha256 与 frozen result 不一致")
        if pred_record_b.result_sha256 is not None and pred_record_b.result_sha256 != numeric_b.result_sha256:
            raise SymbolicTaskBuilderError(f"{pred_plan_b.logical_id} 的 numeric result_sha256 与 frozen result 不一致")
        dependencies = (pred_record_a.evaluation_key, pred_record_b.evaluation_key)
        request_context = {
            "dataset_id": group.dataset_id,
            "dataset_index": group.dataset_index,
            "algorithm": group.algorithm,
            "algorithm_slug": group.algorithm_slug,
            "noise_tag": CONDITION,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "prediction_a_logical_id": pred_plan_a.logical_id,
            "prediction_a_plan_evaluation_key": pred_plan_a.evaluation_key,
            "prediction_a_frozen_plan_sha256": pred_record_a.plan_sha256,
            "prediction_a_frozen_evaluation_key": pred_record_a.evaluation_key,
            "prediction_a_simplify_status": pred_record_a.simplified_status,
            "simplified_prediction_a_expression": pred_record_a.simplified_expression,
            "prediction_a_task_id": numeric_a.task_id,
            "prediction_a_valid_output": numeric_a.valid_output,
            "prediction_a_result_sha256": numeric_a.result_sha256,
            "prediction_b_logical_id": pred_plan_b.logical_id,
            "prediction_b_plan_evaluation_key": pred_plan_b.evaluation_key,
            "prediction_b_frozen_plan_sha256": pred_record_b.plan_sha256,
            "prediction_b_frozen_evaluation_key": pred_record_b.evaluation_key,
            "prediction_b_simplify_status": pred_record_b.simplified_status,
            "simplified_prediction_b_expression": pred_record_b.simplified_expression,
            "prediction_b_task_id": numeric_b.task_id,
            "prediction_b_valid_output": numeric_b.valid_output,
            "prediction_b_result_sha256": numeric_b.result_sha256,
        }
        if not (
            numeric_a.valid_output
            and numeric_b.valid_output
            and _has_callable_expression(pred_record_a)
            and _has_callable_expression(pred_record_b)
        ):
            no_call_records.append(
                _materialize_non_applicable_evidence(
                    evidence_dir=non_applicable_dir,
                    logical_id=logical_id,
                    task_type=STRUCTURE_TASK_TYPE,
                    phase="structure",
                    reason=STRUCTURE_INVALID_SEED_OR_EXPRESSION,
                    contract=structure_contract,
                    dependencies=dependencies,
                    request_context=request_context,
                    write_evidence=write_non_applicable_evidence,
                )
            )
            continue
        evidence_payload = _build_full_pair_evidence(
            logical_id=logical_id,
            phase="structure",
            left_plan=pred_plan_a,
            left_frozen=pred_record_a,
            right_plan=pred_plan_b,
            right_frozen=pred_record_b,
            pair_seed=(seed_a * 1000) + seed_b,
        )
        evidence_hash = _require_sha256(
            evidence_payload.get("evidence_sha256"),
            context=f"{logical_id}.evidence_sha256",
        )
        request = {
            **request_context,
            "deterministic_evidence": evidence_payload,
            "deterministic_pair_evidence": evidence_payload,
            "evidence_hash": evidence_hash,
        }
        structure_tasks.append(
            _task_from_request(
                logical_id=logical_id,
                task_type=STRUCTURE_TASK_TYPE,
                priority=STRUCTURE_PRIORITY,
                request=request,
                evidence_hash=evidence_hash,
                contract=structure_contract,
                dependencies=dependencies,
            )
        )

    if phase == "equivalence":
        selected_tasks = list(equivalence_tasks)
    elif phase == "structure":
        selected_tasks = list(structure_tasks)
    else:
        selected_tasks = [*equivalence_tasks, *structure_tasks]
    selected_tasks.sort(key=lambda item: (item.priority, item.logical_id))
    no_call_records.sort(key=lambda item: (item["phase"], item["logical_id"]))
    equivalence_no_call_count = sum(1 for row in no_call_records if row["phase"] == "equivalence")
    structure_no_call_count = sum(1 for row in no_call_records if row["phase"] == "structure")
    if len(equivalence_tasks) + equivalence_no_call_count != (expected_pred_count or len(pred_plan_by_logical_id)):
        raise SymbolicTaskBuilderError("equivalence callable+no-call 总数不闭合")
    if len(structure_tasks) + structure_no_call_count != (expected_pair_count or len(derived_structure_pairs)):
        raise SymbolicTaskBuilderError("structure callable+no-call 总数不闭合")
    report = {
        "phase": phase,
        "inputs": {
            "gt_frozen_index_jsonl": str(gt_frozen_index_jsonl.resolve()),
            "pred_frozen_index_jsonl": str(pred_frozen_index_jsonl.resolve()),
            "simplify_plan_jsonl": None if simplify_plan_jsonl is None else str(simplify_plan_jsonl.resolve()),
            "gt_plan_jsonl": str(gt_plan_path.resolve()),
            "pred_plan_jsonl": str(pred_plan_path.resolve()),
            "gt_frozen_summary_json": str(gt_frozen_summary_json.resolve()),
            "pred_frozen_summary_json": str(pred_frozen_summary_json.resolve()),
            "clean_run_metrics_csv": str(clean_run_metrics_csv.resolve()),
            "non_applicable_evidence_dir": str(non_applicable_dir),
        },
        "contract": {
            "equivalence": {
                "prompt_path": equivalence_contract.prompt_path,
                "prompt_version": equivalence_contract.prompt_version,
                "prompt_sha256": equivalence_contract.prompt_sha256,
                "schema_path": equivalence_contract.schema_path,
                "schema_version": equivalence_contract.schema_version,
                "schema_sha256": equivalence_contract.schema_sha256,
            },
            "structure": {
                "prompt_path": structure_contract.prompt_path,
                "prompt_version": structure_contract.prompt_version,
                "prompt_sha256": structure_contract.prompt_sha256,
                "schema_path": structure_contract.schema_path,
                "schema_version": structure_contract.schema_version,
                "schema_sha256": structure_contract.schema_sha256,
            },
        },
        "validation": {
            "gt_frozen_count": len(gt_frozen),
            "pred_frozen_count": len(pred_frozen),
            "structure_pair_count": len(derived_structure_pairs),
            "gt_plan_count": len(gt_plan_by_dataset),
            "pred_plan_count": len(pred_plan_by_logical_id),
            "pred_group_count": len(pred_groups),
            "required_seed_set": list(SEEDS),
            "required_seed_pairs": [list(item) for item in SEED_PAIRS],
            "clean_run_metrics": numeric_validity_report,
            "equivalence_unique_logical_ids": len({task.logical_id for task in equivalence_tasks}),
            "structure_unique_logical_ids": len({task.logical_id for task in structure_tasks}),
        },
        "planning_counts": {
            "equivalence_total": len(equivalence_tasks),
            "structure_total": len(structure_tasks),
            "selected_phase_task_count": len(selected_tasks),
            "no_call_count": len(no_call_records),
            "equivalence_no_call_count": equivalence_no_call_count,
            "structure_no_call_count": structure_no_call_count,
            "equivalence_closed_total": len(equivalence_tasks) + equivalence_no_call_count,
            "structure_closed_total": len(structure_tasks) + structure_no_call_count,
        },
        "no_call_records": no_call_records,
    }
    return selected_tasks, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = _repo_root()
    stage_root = repo_root / STAGE_ROOT_RELATIVE
    parser = argparse.ArgumentParser(description="构建 Stage5 clean equivalence/structure 任务计划")
    parser.add_argument("--gt-frozen-index-jsonl", type=Path, required=True)
    parser.add_argument("--pred-frozen-index-jsonl", type=Path, required=True)
    parser.add_argument("--gt-frozen-summary-json", type=Path, required=True)
    parser.add_argument("--pred-frozen-summary-json", type=Path, required=True)
    parser.add_argument(
        "--clean-run-metrics-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--simplify-plan-jsonl", type=Path)
    parser.add_argument("--gt-plan-jsonl", type=Path)
    parser.add_argument("--pred-plan-jsonl", type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--phase", choices=PLAN_PHASES, default="all")
    parser.add_argument("--expected-gt-count", type=int, default=50)
    parser.add_argument("--expected-pred-count", type=int, default=2250)
    parser.add_argument("--expected-pair-count", type=int, default=2250)
    parser.add_argument("--output-jsonl", type=Path, default=stage_root / "reports/clean_symbolic_tasks.jsonl")
    parser.add_argument(
        "--non-applicable-index-jsonl",
        type=Path,
        default=repo_root / DEFAULT_NON_APPLICABLE_INDEX_JSONL,
    )
    parser.add_argument(
        "--non-applicable-evidence-dir",
        type=Path,
        default=repo_root / DEFAULT_NON_APPLICABLE_EVIDENCE_DIR,
    )
    parser.add_argument("--report-json", type=Path, default=stage_root / "reports/clean_symbolic_task_plan.json")
    parser.add_argument("--equivalence-output-jsonl", type=Path)
    parser.add_argument("--equivalence-non-applicable-index-jsonl", type=Path)
    parser.add_argument("--equivalence-full-plan-jsonl", type=Path)
    parser.add_argument("--structure-output-jsonl", type=Path)
    parser.add_argument("--structure-non-applicable-index-jsonl", type=Path)
    parser.add_argument("--structure-full-plan-jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_phase_materialization_request(args: argparse.Namespace) -> None:
    requested: dict[str, tuple[Path | None, Path | None, Path | None]] = {
        "equivalence": (
            args.equivalence_output_jsonl,
            args.equivalence_non_applicable_index_jsonl,
            args.equivalence_full_plan_jsonl,
        ),
        "structure": (
            args.structure_output_jsonl,
            args.structure_non_applicable_index_jsonl,
            args.structure_full_plan_jsonl,
        ),
    }
    for phase, outputs in requested.items():
        if not any(output is not None for output in outputs):
            continue
        if any(output is None for output in outputs):
            raise SymbolicTaskBuilderError(
                f"请求 {phase} 专属物化时必须同时提供 callable、non-applicable 和 full-plan 三个输出"
            )
        if args.phase not in {phase, "all"}:
            raise SymbolicTaskBuilderError(
                f"phase={args.phase!r} 时不能请求 {phase} 专属物化输出"
            )


def _materialize_phase_outputs(
    *,
    tasks: Sequence[PlannedTask],
    no_call_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    phase_outputs: dict[str, dict[str, Any]] = {}
    phase_requests = {
        "equivalence": {
            "callable": args.equivalence_output_jsonl,
            "non_applicable": args.equivalence_non_applicable_index_jsonl,
            "full_plan": args.equivalence_full_plan_jsonl,
        },
        "structure": {
            "callable": args.structure_output_jsonl,
            "non_applicable": args.structure_non_applicable_index_jsonl,
            "full_plan": args.structure_full_plan_jsonl,
        },
    }
    for phase, outputs in phase_requests.items():
        if not any(path is not None for path in outputs.values()):
            continue
        callable_tasks = _phase_callable_tasks(tasks, phase=phase)
        callable_rows = [_task_json_record(task) for task in callable_tasks]
        no_call_rows = _phase_no_call_records(no_call_records, phase=phase)
        full_rows = _phase_full_plan_rows(tasks, no_call_records, phase=phase)
        payload: dict[str, Any] = {
            "phase": phase,
            "callable_task_count": len(callable_rows),
            "non_applicable_count": len(no_call_rows),
            "full_plan_count": len(full_rows),
        }
        if outputs["callable"] is not None:
            _write_jsonl(outputs["callable"], callable_rows)
            payload["callable_output_jsonl"] = str(outputs["callable"].resolve())
            payload["callable_output_sha256"] = _sha256_file(outputs["callable"])
        if outputs["non_applicable"] is not None:
            _write_jsonl(outputs["non_applicable"], no_call_rows)
            payload["non_applicable_index_jsonl"] = str(outputs["non_applicable"].resolve())
            payload["non_applicable_index_sha256"] = _sha256_file(outputs["non_applicable"])
        if outputs["full_plan"] is not None:
            _write_jsonl(outputs["full_plan"], full_rows)
            payload["full_plan_jsonl"] = str(outputs["full_plan"].resolve())
            payload["full_plan_sha256"] = _sha256_file(outputs["full_plan"])
        phase_outputs[phase] = payload
    return phase_outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_phase_materialization_request(args)
        tasks, report = build_symbolic_task_plan(
            gt_frozen_index_jsonl=args.gt_frozen_index_jsonl.resolve(),
            pred_frozen_index_jsonl=args.pred_frozen_index_jsonl.resolve(),
            gt_frozen_summary_json=args.gt_frozen_summary_json.resolve(),
            pred_frozen_summary_json=args.pred_frozen_summary_json.resolve(),
            clean_run_metrics_csv=args.clean_run_metrics_csv.resolve(),
            non_applicable_evidence_dir=args.non_applicable_evidence_dir.resolve(),
            simplify_plan_jsonl=(
                None if args.simplify_plan_jsonl is None else args.simplify_plan_jsonl.resolve()
            ),
            gt_plan_jsonl=None if args.gt_plan_jsonl is None else args.gt_plan_jsonl.resolve(),
            pred_plan_jsonl=None if args.pred_plan_jsonl is None else args.pred_plan_jsonl.resolve(),
            phase=args.phase,
            repo_root=args.repo_root.resolve(),
            expected_gt_count=args.expected_gt_count,
            expected_pred_count=args.expected_pred_count,
            expected_pair_count=args.expected_pair_count,
            write_non_applicable_evidence=not args.dry_run,
        )
    except SymbolicTaskBuilderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(report["planning_counts"], ensure_ascii=False, sort_keys=True))
        return 0
    selected_no_call_records = _phase_no_call_records(report["no_call_records"], phase=args.phase)
    _write_jsonl(args.output_jsonl, [_task_json_record(task) for task in tasks])
    _write_jsonl(args.non_applicable_index_jsonl, selected_no_call_records)
    phase_outputs = _materialize_phase_outputs(
        tasks=tasks,
        no_call_records=report["no_call_records"],
        args=args,
    )
    if phase_outputs:
        report["phase_outputs"] = phase_outputs
    _write_json(args.report_json, report)
    print(json.dumps(report["planning_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
