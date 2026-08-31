"""Stage5 clean simplify 任务的稳定构建器。"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .claude_runner import TaskDefinition as RunnerTaskDefinition
from .claude_contract import evaluation_key, render_prompt
from .state import TaskSpec
from .trajectories import canonical_expression


STAGE_ROOT_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831")
DEFAULT_GROUND_TRUTH_JSONL = STAGE_ROOT_RELATIVE / "reports/ground_truth_extract.jsonl"
DEFAULT_FREEZE_GLOB = str(
    STAGE_ROOT_RELATIVE / "source_snapshot/trajectory_freeze/clean_freeze_*.jsonl.gz"
)
SIMPLIFY_PROMPT_RELATIVE = STAGE_ROOT_RELATIVE / "config/prompts/simplify.v1.txt"
SIMPLIFY_SCHEMA_RELATIVE = STAGE_ROOT_RELATIVE / "config/schemas/simplify.v1.json"
DEFAULT_OUTPUT_JSONL = STAGE_ROOT_RELATIVE / "reports/clean_simplify_tasks.jsonl"
DEFAULT_REPORT_JSON = STAGE_ROOT_RELATIVE / "reports/clean_simplify_task_plan.json"
TASK_PHASES = ("gt", "pred", "all")
GT_TASK_TYPE = "gt_simplify"
PRED_TASK_TYPE = "pred_simplify"
CONDITION = "clean"
GT_PRIORITY = 10
PRED_PRIORITY = 20
FUTURE_EQUIVALENCE_MAX = 2250
FUTURE_STRUCTURE_MAX = 2250
TASK_ID_PATTERN = re.compile(
    r"^(?P<algorithm_slug>[a-z0-9]+)_s(?P<seed>\d+)_clean_g(?P<dataset_index>\d{4})$"
)
CANONICAL_VARIABLE_PATTERN = re.compile(r"\bx\d+\b")


class CleanTaskBuilderError(ValueError):
    """输入冻结、契约或规划结构不合法。"""


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

    def to_runner_definition(self) -> RunnerTaskDefinition:
        return RunnerTaskDefinition(
            task_spec=self.to_task_spec(),
            request=self.request,
            prompt_path=Path(self.prompt_path),
            prompt_sha256=self.prompt_sha256,
            schema_path=Path(self.schema_path),
            schema_sha256=self.schema_sha256,
            prompt_template=self.prompt_template,
            schema=self.schema_content,
            task_kind="simplify",
        )

    def to_json_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dependencies"] = list(self.dependencies)
        payload["task_spec"] = json.loads(self.to_task_spec().canonical_json())
        payload["rendered_prompt"] = render_prompt(self.prompt_template, self.request)
        return payload


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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CleanTaskBuilderError(f"{context} 不是 JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleanTaskBuilderError(f"{path}:{line_number} JSONL 解析失败: {exc}") from exc
            rows.append(dict(_require_mapping(row, context=f"{path}:{line_number}")))
    return rows


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleanTaskBuilderError(f"{path}:{line_number} JSONL 解析失败: {exc}") from exc
            yield dict(_require_mapping(row, context=f"{path}:{line_number}"))


def _load_prompt_schema(repo_root: Path) -> PromptSchemaBundle:
    prompt_path = (repo_root / SIMPLIFY_PROMPT_RELATIVE).resolve()
    schema_path = (repo_root / SIMPLIFY_SCHEMA_RELATIVE).resolve()
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


def _build_contract_input(
    *,
    request: Mapping[str, Any],
    prompt_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    return {
        "request": dict(request),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }


def _evaluation_key_with_hashes(
    *,
    task_type: str,
    logical_id: str,
    prompt_version: str,
    prompt_sha256: str,
    schema_version: str,
    schema_sha256: str,
    normalized_input: Mapping[str, Any],
    evidence_hash: str,
) -> str:
    return str(
        evaluation_key(
            task_type=task_type,
            logical_id=logical_id,
            prompt_version=prompt_version,
            schema_version=schema_version,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            normalized_input=normalized_input,
            evidence_hash=evidence_hash,
        )
    )


def _extract_function_names(expression: str, variables: Sequence[str]) -> list[str]:
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", expression))
    names.difference_update(variables)
    names.discard("return")
    return sorted(names)


def _expression_canonical_variables(expression: str) -> list[str]:
    return sorted(set(CANONICAL_VARIABLE_PATTERN.findall(expression)))


def _resolve_request_variables(
    *,
    expression: str,
    artifact: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> list[str]:
    artifact_variables = artifact.get("variables")
    if isinstance(artifact_variables, list) and all(
        isinstance(item, str) and item for item in artifact_variables
    ):
        if artifact_variables:
            return list(artifact_variables)
        expression_variables = _expression_canonical_variables(expression)
        if expression_variables:
            return expression_variables
        return []

    expression_variables = _expression_canonical_variables(expression)
    if expression_variables:
        return expression_variables

    feature_names = payload.get("feature_names")
    if isinstance(feature_names, list) and all(isinstance(item, str) and item for item in feature_names):
        return list(feature_names)
    return []


def _first_nonempty_string(values: Iterable[tuple[str, object]]) -> tuple[str, str]:
    for source_name, value in values:
        if isinstance(value, str) and value.strip():
            return value.strip(), source_name
    return "", ""


def select_formula_with_source(payload: Mapping[str, Any]) -> tuple[str, str]:
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    selected_expression = canonical_expression(payload)
    selected_expression = selected_expression.strip()
    if not selected_expression:
        return "", ""
    expression, selected_from = _first_nonempty_string(
        (
            ("canonical_artifact.instantiated_expression", artifact.get("instantiated_expression")),
            ("canonical_artifact.normalized_expression", artifact.get("normalized_expression")),
            ("canonical_artifact.return_expression_source", artifact.get("return_expression_source")),
            ("equation", payload.get("equation")),
        )
    )
    if expression != selected_expression:
        raise CleanTaskBuilderError("公式优先级解析与 canonical_expression 不一致")
    return expression, selected_from


def _parse_task_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    task_id = source.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise CleanTaskBuilderError("freeze source.task_id 缺失")
    match = TASK_ID_PATTERN.fullmatch(task_id)
    if match is None:
        raise CleanTaskBuilderError(f"无法解析 clean task_id: {task_id!r}")
    seed_from_task = int(match.group("seed"))
    seed_raw = source.get("seed")
    try:
        seed = int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise CleanTaskBuilderError(f"freeze source.seed 非法: {seed_raw!r}") from exc
    if seed != seed_from_task:
        raise CleanTaskBuilderError(f"freeze source.seed 与 task_id 不一致: {task_id!r}")
    noise_tag = source.get("noise_tag")
    if noise_tag != CONDITION:
        raise CleanTaskBuilderError(f"检测到非 clean 轨迹混入: {task_id!r}, noise_tag={noise_tag!r}")
    return {
        "task_id": task_id,
        "algorithm_slug": match.group("algorithm_slug"),
        "dataset_index": match.group("dataset_index"),
        "seed": seed,
    }


def _validate_clean_payload(source: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    train_label_noise = payload.get("train_label_noise")
    if not isinstance(train_label_noise, Mapping):
        return
    if bool(train_label_noise.get("enabled")) or bool(train_label_noise.get("requested")):
        raise CleanTaskBuilderError(
            f"{source.get('task_id')}: train_label_noise 标记显示不是 clean"
        )
    sigma = train_label_noise.get("sigma")
    if isinstance(sigma, bool):
        raise CleanTaskBuilderError(f"{source.get('task_id')}: sigma 非法")
    if sigma not in (None, 0, 0.0):
        try:
            if float(sigma) != 0.0:
                raise CleanTaskBuilderError(
                    f"{source.get('task_id')}: sigma={sigma!r}，不是 clean 结果"
                )
        except (TypeError, ValueError) as exc:
            raise CleanTaskBuilderError(f"{source.get('task_id')}: sigma 非法") from exc


def _gt_logical_id(dataset_id: str) -> str:
    return f"{GT_TASK_TYPE}::{dataset_id}"


def _pred_logical_id(identity: Mapping[str, Any]) -> str:
    return (
        f"{PRED_TASK_TYPE}::{identity['algorithm_slug']}::g{identity['dataset_index']}"
        f"::s{identity['seed']}::{CONDITION}"
    )


def _build_task_definition(
    *,
    logical_id: str,
    task_type: str,
    priority: int,
    request: dict[str, Any],
    evidence_hash: str,
    contract: PromptSchemaBundle,
) -> PlannedTask:
    normalized_input = _build_contract_input(
        request=request,
        prompt_sha256=contract.prompt_sha256,
        schema_sha256=contract.schema_sha256,
    )
    input_hash = _sha256_json(normalized_input)
    task_key = _evaluation_key_with_hashes(
        task_type=task_type,
        logical_id=logical_id,
        prompt_version=contract.prompt_version,
        prompt_sha256=contract.prompt_sha256,
        schema_version=contract.schema_version,
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
        dependencies=(),
        prompt_path=contract.prompt_path,
        schema_path=contract.schema_path,
        prompt_template=contract.prompt_template,
        schema_content=contract.schema,
        normalized_input=normalized_input,
        request=request,
    )


def _no_call_record(
    *,
    logical_id: str,
    task_type: str,
    phase: str,
    reason: str,
    request_context: dict[str, Any],
    evidence_hash: str,
    contract: PromptSchemaBundle,
) -> dict[str, Any]:
    normalized_input = _build_contract_input(
        request=request_context,
        prompt_sha256=contract.prompt_sha256,
        schema_sha256=contract.schema_sha256,
    )
    return {
        "phase": phase,
        "logical_id": logical_id,
        "task_type": task_type,
        "condition": CONDITION,
        "reason": reason,
        "input_hash": _sha256_json(normalized_input),
        "evidence_hash": evidence_hash,
        "prompt_version": contract.prompt_version,
        "prompt_sha256": contract.prompt_sha256,
        "schema_version": contract.schema_version,
        "schema_sha256": contract.schema_sha256,
        "request_context": request_context,
        "status": "planned_no_call",
    }


def _build_gt_task(
    row: Mapping[str, Any],
    *,
    index: int,
    contract: PromptSchemaBundle,
) -> tuple[PlannedTask | None, dict[str, Any] | None]:
    dataset_id = row.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise CleanTaskBuilderError("ground_truth_extract.dataset_id 缺失")
    variables = row.get("ordered_variables")
    if not isinstance(variables, list) or not all(isinstance(item, str) and item for item in variables):
        raise CleanTaskBuilderError(f"{dataset_id}: ordered_variables 非法")
    expression = row.get("normalized_expression_input")
    original_expression = row.get("return_source")
    logical_id = _gt_logical_id(dataset_id)
    evidence_hash = row.get("evidence_sha256")
    if not isinstance(evidence_hash, str) or not evidence_hash:
        raise CleanTaskBuilderError(f"{dataset_id}: evidence_sha256 缺失")
    request_context = {
        "dataset_id": dataset_id,
        "target_name": row.get("target"),
        "variables": variables,
        "allowed_functions": _extract_function_names(str(expression or ""), variables),
        "expression": expression if isinstance(expression, str) and expression.strip() else None,
        "original_expression": original_expression
        if isinstance(original_expression, str) and original_expression.strip()
        else None,
        "ast_source_evidence": {
            "return_ast_dump": row.get("return_ast_dump"),
            "return_source": row.get("return_source"),
            "selection_reason": row.get("selection_reason"),
            "source_checksums": row.get("source_checksums"),
        },
        "evidence_hash": evidence_hash,
    }
    if not request_context["expression"]:
        no_call = _no_call_record(
            logical_id=logical_id,
            task_type=GT_TASK_TYPE,
            phase="gt",
            reason="missing_ground_truth_expression",
            request_context=request_context,
            evidence_hash=evidence_hash,
            contract=contract,
        )
        return None, no_call
    task = _build_task_definition(
        logical_id=logical_id,
        task_type=GT_TASK_TYPE,
        priority=GT_PRIORITY,
        request=request_context,
        evidence_hash=evidence_hash,
        contract=contract,
    )
    return task, None


def _build_pred_task(
    freeze_row: Mapping[str, Any],
    *,
    contract: PromptSchemaBundle,
) -> tuple[PlannedTask | None, dict[str, Any] | None]:
    source = _require_mapping(freeze_row.get("source"), context="freeze source")
    identity = _parse_task_identity(source)
    result = _require_mapping(freeze_row.get("result"), context=f"{identity['task_id']} result")
    raw_text = result.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise CleanTaskBuilderError(f"{identity['task_id']}: result.raw_text 缺失")
    raw_sha256 = _sha256_text(raw_text)
    expected_sha256 = result.get("sha256")
    if raw_sha256 != expected_sha256:
        raise CleanTaskBuilderError(f"{identity['task_id']}: result raw sha256 校验失败")
    payload = json.loads(raw_text)
    payload = dict(_require_mapping(payload, context=f"{identity['task_id']} raw payload"))
    _validate_clean_payload(source, payload)
    artifact = payload.get("canonical_artifact")
    if not isinstance(artifact, Mapping):
        artifact = {}
    expression, expression_source = select_formula_with_source(payload)
    variables = _resolve_request_variables(
        expression=expression or str(payload.get("equation") or ""),
        artifact=artifact,
        payload=payload,
    )
    evidence_payload = {
        "source": {
            "algorithm": source.get("algorithm"),
            "algorithm_slug": identity["algorithm_slug"],
            "batch": source.get("batch"),
            "dataset_id": source.get("dataset_id"),
            "dataset_index": f"g{identity['dataset_index']}",
            "host": source.get("host"),
            "noise_tag": source.get("noise_tag"),
            "path": source.get("path"),
            "seed": identity["seed"],
            "source_row_sha256": source.get("source_row_sha256"),
            "task_id": identity["task_id"],
        },
        "result": {
            "raw_sha256": raw_sha256,
            "status": payload.get("status"),
            "equation": payload.get("equation"),
            "canonical_artifact": {
                "instantiated_expression": artifact.get("instantiated_expression"),
                "normalized_expression": artifact.get("normalized_expression"),
                "return_expression_source": artifact.get("return_expression_source"),
                "raw_equation": artifact.get("raw_equation"),
                "raw_equation_kind": artifact.get("raw_equation_kind"),
                "python_function_source": artifact.get("python_function_source"),
                "variables": artifact.get("variables"),
                "ast_node_count": artifact.get("ast_node_count"),
                "tree_depth": artifact.get("tree_depth"),
                "normalization_mode": artifact.get("normalization_mode"),
                "normalization_notes": artifact.get("normalization_notes"),
            },
        },
    }
    evidence_hash = _sha256_json(evidence_payload)
    logical_id = _pred_logical_id(identity)
    request_context = {
        "dataset_id": source.get("dataset_id"),
        "dataset_index": f"g{identity['dataset_index']}",
        "algorithm": source.get("algorithm"),
        "algorithm_slug": identity["algorithm_slug"],
        "seed": identity["seed"],
        "noise_tag": CONDITION,
        "task_id": identity["task_id"],
        "variables": variables,
        "allowed_functions": _extract_function_names(expression or str(payload.get("equation") or ""), variables),
        "expression": expression or None,
        "original_expression": payload.get("equation")
        if isinstance(payload.get("equation"), str) and payload.get("equation", "").strip()
        else None,
        "ast_source_evidence": {
            "selected_expression_source": expression_source or None,
            "formula_candidates": {
                "instantiated_expression": artifact.get("instantiated_expression"),
                "normalized_expression": artifact.get("normalized_expression"),
                "return_expression_source": artifact.get("return_expression_source"),
                "equation": payload.get("equation"),
            },
            "canonical_artifact": evidence_payload["result"]["canonical_artifact"],
            "result_status": payload.get("status"),
            "result_path": source.get("path"),
            "result_raw_sha256": raw_sha256,
            "source_row_sha256": source.get("source_row_sha256"),
        },
        "evidence_hash": evidence_hash,
    }
    if not expression:
        no_call = _no_call_record(
            logical_id=logical_id,
            task_type=PRED_TASK_TYPE,
            phase="pred",
            reason="missing_final_expression",
            request_context=request_context,
            evidence_hash=evidence_hash,
            contract=contract,
        )
        return None, no_call
    task = _build_task_definition(
        logical_id=logical_id,
        task_type=PRED_TASK_TYPE,
        priority=PRED_PRIORITY,
        request=request_context,
        evidence_hash=evidence_hash,
        contract=contract,
    )
    return task, None


def _load_pred_freeze_rows(
    freeze_glob: str,
    *,
    expected_pred_count: int | None,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pattern = freeze_glob
    if not Path(pattern).is_absolute():
        pattern = str((repo_root / pattern).resolve())
    paths = [Path(item) for item in sorted(glob.glob(pattern))]
    if not paths:
        raise CleanTaskBuilderError(f"未找到 freeze 文件: {freeze_glob}")
    rows: list[dict[str, Any]] = []
    clean_keys: list[str] = []
    raw_sha_valid = 0
    formula_source_counts: dict[str, int] = {}
    tasks_with_variables = 0
    tasks_without_variables = 0
    for path in paths:
        for row in _iter_gzip_jsonl(path):
            source = _require_mapping(row.get("source"), context=f"{path} source")
            identity = _parse_task_identity(source)
            clean_key = (
                f"{identity['algorithm_slug']}::g{identity['dataset_index']}"
                f"::s{identity['seed']}::{CONDITION}"
            )
            clean_keys.append(clean_key)
            result = _require_mapping(row.get("result"), context=f"{identity['task_id']} result")
            raw_text = result.get("raw_text")
            if isinstance(raw_text, str) and raw_text:
                if _sha256_text(raw_text) == result.get("sha256"):
                    raw_sha_valid += 1
                payload = dict(
                    _require_mapping(
                        json.loads(raw_text),
                        context=f"{identity['task_id']} raw payload",
                    )
                )
                expression, expression_source = select_formula_with_source(payload)
                if expression_source:
                    formula_source_counts[expression_source] = (
                        formula_source_counts.get(expression_source, 0) + 1
                    )
                artifact = payload.get("canonical_artifact")
                artifact_mapping = artifact if isinstance(artifact, Mapping) else {}
                variables = _resolve_request_variables(
                    expression=expression or str(payload.get("equation") or ""),
                    artifact=artifact_mapping,
                    payload=payload,
                )
                if variables or not _expression_canonical_variables(expression):
                    tasks_with_variables += 1
                else:
                    tasks_without_variables += 1
            rows.append(row)
    unique_clean_keys = sorted(set(clean_keys))
    duplicate_count = len(clean_keys) - len(unique_clean_keys)
    if expected_pred_count is not None and len(rows) != expected_pred_count:
        raise CleanTaskBuilderError(
            f"clean freeze 行数不符: 期望 {expected_pred_count}，实际 {len(rows)}"
        )
    if expected_pred_count is not None and len(unique_clean_keys) != expected_pred_count:
        raise CleanTaskBuilderError(
            f"clean key 唯一数不符: 期望 {expected_pred_count}，实际 {len(unique_clean_keys)}"
        )
    if raw_sha_valid != len(rows):
        raise CleanTaskBuilderError(
            f"存在 result raw sha256 校验失败: 通过 {raw_sha_valid}/{len(rows)}"
        )
    return rows, {
        "freeze_files": [str(path.resolve()) for path in paths],
        "freeze_file_count": len(paths),
        "row_count": len(rows),
        "unique_clean_keys": len(unique_clean_keys),
        "duplicate_clean_keys": duplicate_count,
        "result_raw_sha_valid_count": raw_sha_valid,
        "formula_source_counts": dict(sorted(formula_source_counts.items())),
        "tasks_with_variables": tasks_with_variables,
        "tasks_without_variables": tasks_without_variables,
    }


def build_clean_task_plan(
    *,
    phase: str = "all",
    ground_truth_jsonl: Path | None = None,
    freeze_glob: str | None = None,
    expected_gt_count: int | None = 50,
    expected_pred_count: int | None = 2250,
    repo_root: Path | None = None,
) -> tuple[list[PlannedTask], dict[str, Any]]:
    if phase not in TASK_PHASES:
        raise CleanTaskBuilderError(f"未知 phase: {phase!r}")
    repo_root = repo_root or _repo_root()
    contract = _load_prompt_schema(repo_root)
    gt_path = (ground_truth_jsonl or (repo_root / DEFAULT_GROUND_TRUTH_JSONL)).resolve()
    pred_glob = freeze_glob or DEFAULT_FREEZE_GLOB

    gt_rows = _read_jsonl(gt_path)
    if expected_gt_count is not None and len(gt_rows) != expected_gt_count:
        raise CleanTaskBuilderError(
            f"ground_truth_extract 行数不符: 期望 {expected_gt_count}，实际 {len(gt_rows)}"
        )
    pred_rows, pred_validation = _load_pred_freeze_rows(
        pred_glob,
        expected_pred_count=expected_pred_count,
        repo_root=repo_root,
    )

    gt_tasks: list[PlannedTask] = []
    pred_tasks: list[PlannedTask] = []
    no_call_records: list[dict[str, Any]] = []

    for index, row in enumerate(gt_rows, start=1):
        task, no_call = _build_gt_task(row, index=index, contract=contract)
        if task is not None:
            gt_tasks.append(task)
        if no_call is not None:
            no_call_records.append(no_call)

    for row in pred_rows:
        task, no_call = _build_pred_task(row, contract=contract)
        if task is not None:
            pred_tasks.append(task)
        if no_call is not None:
            no_call_records.append(no_call)

    if phase == "gt":
        selected_tasks = list(gt_tasks)
    elif phase == "pred":
        selected_tasks = list(pred_tasks)
    else:
        selected_tasks = [*gt_tasks, *pred_tasks]
    selected_tasks.sort(key=lambda item: (item.priority, item.logical_id))
    no_call_records.sort(key=lambda item: (item["phase"], item["logical_id"]))
    gt_no_call_count = sum(1 for item in no_call_records if item["phase"] == "gt")
    pred_no_call_count = sum(1 for item in no_call_records if item["phase"] == "pred")

    report = {
        "phase": phase,
        "inputs": {
            "ground_truth_jsonl": str(gt_path),
            "freeze_glob": pred_glob,
        },
        "contract": {
            "prompt_path": contract.prompt_path,
            "prompt_version": contract.prompt_version,
            "prompt_sha256": contract.prompt_sha256,
            "schema_path": contract.schema_path,
            "schema_version": contract.schema_version,
            "schema_sha256": contract.schema_sha256,
        },
        "validation": {
            "ground_truth_row_count": len(gt_rows),
            "pred_freeze": pred_validation,
            "noise_condition": CONDITION,
        },
        "planning_counts": {
            "gt_simplify_total": len(gt_tasks),
            "pred_simplify_total": len(pred_tasks),
            "future_equivalence_max": FUTURE_EQUIVALENCE_MAX,
            "future_structure_max": FUTURE_STRUCTURE_MAX,
            "clean_total_max": len(gt_tasks)
            + len(pred_tasks)
            + FUTURE_EQUIVALENCE_MAX
            + FUTURE_STRUCTURE_MAX,
            "selected_phase_task_count": len(selected_tasks),
            "no_call_count": len(no_call_records),
            "gt_no_call_count": gt_no_call_count,
            "pred_no_call_count": pred_no_call_count,
        },
        "no_call_counts": {"gt": gt_no_call_count, "pred": pred_no_call_count},
        "no_call_records": no_call_records,
    }
    return selected_tasks, report


def build_argument_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="构建 Stage5 clean simplify Claude 任务")
    parser.add_argument("--phase", choices=TASK_PHASES, default="all")
    parser.add_argument(
        "--ground-truth-jsonl",
        type=Path,
        default=repo_root / DEFAULT_GROUND_TRUTH_JSONL,
    )
    parser.add_argument(
        "--freeze-glob",
        default=DEFAULT_FREEZE_GLOB,
        help="相对仓库根目录的 clean freeze glob",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_JSONL,
        help="任务 JSONL 输出路径；dry-run 时忽略",
    )
    parser.add_argument("--expected-gt-count", type=int, default=50)
    parser.add_argument("--expected-pred-count", type=int, default=2250)
    parser.add_argument(
        "--report",
        nargs="?",
        const="-",
        default=None,
        help="报告输出路径；仅传 --report 时输出到 stdout",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.report == "-" and not args.dry_run:
        raise CleanTaskBuilderError("非 dry-run 模式下 --report 输出到 stdout 会污染 JSONL")
    tasks, report = build_clean_task_plan(
        phase=args.phase,
        ground_truth_jsonl=args.ground_truth_jsonl,
        freeze_glob=args.freeze_glob,
        expected_gt_count=args.expected_gt_count,
        expected_pred_count=args.expected_pred_count,
    )
    if args.report and args.report != "-":
        _write_json(Path(args.report), report)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    task_rows = [task.to_json_record() for task in tasks]
    _write_jsonl(args.output_jsonl.resolve(), task_rows)
    if args.report == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
