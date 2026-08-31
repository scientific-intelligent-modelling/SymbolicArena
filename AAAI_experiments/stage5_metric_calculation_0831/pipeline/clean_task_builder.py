"""Stage5 clean simplify 任务的稳定构建器。"""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import glob
import gzip
import hashlib
import json
import keyword
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .claude_runner import TaskDefinition as RunnerTaskDefinition
from .claude_contract import evaluation_key, render_prompt
from .state import TaskSpec
from .symbolic_evidence import SymbolicEvidenceError, build_symbolic_artifact
from .trajectories import canonical_expression


STAGE_ROOT_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831")
DEFAULT_GROUND_TRUTH_JSONL = STAGE_ROOT_RELATIVE / "reports/ground_truth_extract.jsonl"
DEFAULT_FORMULA_RECOVERY_JSON = (
    STAGE_ROOT_RELATIVE / "manifests/formula_recovery.v1.json"
)
DEFAULT_DATASET_PROBES_JSONL = STAGE_ROOT_RELATIVE / "reports/dataset_probes.jsonl"
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
INDEXED_VARIABLE_PATTERN = re.compile(r"\b(?:x|X|col)(\d+)\b")
PARAMETER_REFERENCE_PATTERN = re.compile(r"\bparams\s*\[\s*(-?\d+)\s*\]")
PARAMETER_REFERENCE_START_PATTERN = re.compile(r"\bparams\s*\[")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DATASET_PROBE_SCHEMA_VERSION = "dataset_probes_v1"


class CleanTaskBuilderError(ValueError):
    """输入冻结、契约或规划结构不合法。"""


@dataclass(frozen=True)
class FormulaResolution:
    semantic_expression: str
    expression_body: str
    variable_mapping: dict[str, str]
    status: str
    manifest_entry: dict[str, Any] | None


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
        payload["rendered_prompt"] = render_prompt(
            self.prompt_template,
            self.request,
            self.schema_content,
        )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_recovery_entry(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _serializable_symbolic_artifact(
    expression: str,
    variables: Sequence[str],
    allowed_functions: Sequence[str],
) -> dict[str, Any]:
    return json.loads(
        _cached_symbolic_artifact(
            expression,
            tuple(variables),
            tuple(sorted(set(allowed_functions))),
        )
    )


@lru_cache(maxsize=4096)
def _cached_symbolic_artifact(
    expression: str,
    variables: tuple[str, ...],
    allowed_functions: tuple[str, ...],
) -> str:
    try:
        artifact = build_symbolic_artifact(
            expression,
            allowed_variables=variables,
            allowed_functions=allowed_functions,
        )
    except (SyntaxError, SymbolicEvidenceError) as exc:
        raise CleanTaskBuilderError(f"符号证据构建失败: {exc}") from exc
    frozen = {
        key: value
        for key, value in artifact.items()
        if key != "sympy_expression"
    }
    return _canonical_json(frozen)


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


def map_indexed_variables(
    expression: str,
    feature_names: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """把算法内部的零基变量名原子映射为数据集真实特征名。"""

    if not isinstance(expression, str):
        raise CleanTaskBuilderError("expression 必须是字符串")
    normalized_feature_names = list(feature_names)
    if not normalized_feature_names:
        raise CleanTaskBuilderError("feature_names 不能为空")
    if len(set(normalized_feature_names)) != len(normalized_feature_names):
        raise CleanTaskBuilderError("feature_names 必须唯一")
    for name in normalized_feature_names:
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            raise CleanTaskBuilderError(f"feature_names 包含非法标识符: {name!r}")

    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        index = int(match.group(1))
        if index >= len(normalized_feature_names):
            raise CleanTaskBuilderError(
                f"索引变量 {token!r} 超出 feature_names 范围 "
                f"[0, {len(normalized_feature_names) - 1}]"
            )
        mapped = normalized_feature_names[index]
        mapping[token] = mapped
        return mapped

    return INDEXED_VARIABLE_PATTERN.sub(replace, expression), dict(sorted(mapping.items()))


def _validated_feature_names(payload: Mapping[str, Any], *, task_id: str) -> list[str]:
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise CleanTaskBuilderError(f"{task_id}: feature_names 缺失或为空")
    if not all(isinstance(item, str) for item in feature_names):
        raise CleanTaskBuilderError(f"{task_id}: feature_names 包含非字符串")
    # 复用映射函数的唯一性和标识符契约，但不要求表达式实际引用索引变量。
    map_indexed_variables("", feature_names)
    return list(feature_names)


def extract_expression_body(source: str) -> str:
    """从单个表达式或单个 Python 函数中提取唯一 return 表达式。"""

    if not isinstance(source, str) or not source.strip():
        return ""
    stripped = source.strip()
    try:
        ast.parse(stripped, mode="eval")
        return stripped
    except SyntaxError:
        if not stripped.startswith("def ") and re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*\(",
            stripped,
        ):
            # 超深 prefix tree 会触发 CPython parser 的嵌套上限；后续由
            # symbolic_evidence 的迭代 token parser 完成严格校验。
            return stripped
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise CleanTaskBuilderError(f"最终公式 Python AST 解析失败: {exc}") from exc
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise CleanTaskBuilderError("最终公式必须是单个表达式或单个函数定义")
    statements = list(module.body[0].body)
    while statements:
        first = statements[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            statements.pop(0)
            continue
        break
    if len(statements) != 1 or not isinstance(statements[0], ast.Return):
        raise CleanTaskBuilderError("最终公式函数必须只包含文档字符串和一个 return")
    if statements[0].value is None:
        raise CleanTaskBuilderError("最终公式函数的 return 不能为空")
    return ast.unparse(statements[0].value)


def instantiate_parameters(expression: str, parameter_values: Sequence[object]) -> str:
    """按 Python 下标语义把 `params[i]` 替换为冻结的有限浮点常数。"""

    values: list[float] = []
    for raw_value in parameter_values:
        if isinstance(raw_value, bool):
            raise CleanTaskBuilderError("拟合参数不能是布尔值")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise CleanTaskBuilderError(f"拟合参数不是数值: {raw_value!r}") from exc
        if not math.isfinite(value):
            raise CleanTaskBuilderError(f"拟合参数不是有限值: {raw_value!r}")
        values.append(value)
    if not values:
        raise CleanTaskBuilderError("拟合参数数组不能为空")

    referenced = False

    def replace(match: re.Match[str]) -> str:
        nonlocal referenced
        referenced = True
        original_index = int(match.group(1))
        index = original_index if original_index >= 0 else len(values) + original_index
        if index < 0 or index >= len(values):
            raise CleanTaskBuilderError(
                f"参数下标 params[{original_index}] 超出长度 {len(values)}"
            )
        return f"({repr(values[index])})"

    instantiated = PARAMETER_REFERENCE_PATTERN.sub(replace, expression)
    if PARAMETER_REFERENCE_START_PATTERN.search(instantiated):
        raise CleanTaskBuilderError("存在无法静态实例化的 params 下标")
    if not referenced:
        raise CleanTaskBuilderError("公式未引用 params，拒绝套用参数恢复记录")
    return instantiated


def _canonical_mapped_ast(expression: str, feature_names: Sequence[str]) -> str:
    mapped, _ = map_indexed_variables(expression, feature_names)
    try:
        parsed = ast.parse(mapped, mode="eval")
    except SyntaxError as exc:
        raise CleanTaskBuilderError(f"恢复公式 AST 解析失败: {exc}") from exc
    return ast.dump(parsed, annotate_fields=True, include_attributes=False)


def load_dataset_probes(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    rows = _read_jsonl(path)
    by_dataset: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        context = f"dataset_probes[{index}]"
        if row.get("schema_version") != DATASET_PROBE_SCHEMA_VERSION:
            raise CleanTaskBuilderError(f"{context}: schema_version 不匹配")
        dataset_id = row.get("dataset_name")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise CleanTaskBuilderError(f"{context}: dataset_name 非法")
        if dataset_id in by_dataset:
            raise CleanTaskBuilderError(f"dataset probes 出现重复 dataset_name: {dataset_id}")
        variables = row.get("variables")
        points = row.get("points")
        if not isinstance(variables, list) or not all(
            isinstance(item, str) and item for item in variables
        ):
            raise CleanTaskBuilderError(f"{dataset_id}: probe variables 非法")
        if not isinstance(points, list) or not points:
            raise CleanTaskBuilderError(f"{dataset_id}: probe points 缺失")
        if row.get("point_count") != len(points):
            raise CleanTaskBuilderError(f"{dataset_id}: probe point_count 不匹配")
        sample_payload = {
            "schema_version": row["schema_version"],
            "dataset_name": dataset_id,
            "variables": variables,
            "points": points,
        }
        if row.get("sample_sha256") != _sha256_json(sample_payload):
            raise CleanTaskBuilderError(f"{dataset_id}: probe sample_sha256 漂移")
        evidence_payload = {
            key: value for key, value in row.items() if key != "evidence_sha256"
        }
        if row.get("evidence_sha256") != _sha256_json(evidence_payload):
            raise CleanTaskBuilderError(f"{dataset_id}: probe evidence_sha256 漂移")
        by_dataset[dataset_id] = row
    return by_dataset, _sha256_bytes(raw_bytes)


def load_formula_recovery_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanTaskBuilderError(f"公式恢复 manifest 解析失败: {path}: {exc}") from exc
    root = _require_mapping(payload, context=str(path))
    if root.get("schema_version") != "formula_recovery.v1":
        raise CleanTaskBuilderError("公式恢复 manifest schema_version 不匹配")
    if root.get("condition") != CONDITION:
        raise CleanTaskBuilderError("公式恢复 manifest condition 不是 clean")
    entries = root.get("entries")
    if not isinstance(entries, list):
        raise CleanTaskBuilderError("公式恢复 manifest.entries 不是数组")

    by_task_id: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = dict(_require_mapping(raw_entry, context=f"manifest.entries[{index}]"))
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or TASK_ID_PATTERN.fullmatch(task_id) is None:
            raise CleanTaskBuilderError(f"公式恢复 entry.task_id 非法: {task_id!r}")
        if task_id in by_task_id:
            raise CleanTaskBuilderError(f"公式恢复 manifest 存在重复 task_id: {task_id}")
        resolution = entry.get("resolution")
        if resolution not in {"recovered_params", "unavailable"}:
            raise CleanTaskBuilderError(f"{task_id}: 未知公式恢复 resolution={resolution!r}")
        for field_name in ("frozen_result_sha256", "equation_sha256"):
            value = entry.get(field_name)
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise CleanTaskBuilderError(f"{task_id}: {field_name} 非法")
        if resolution == "recovered_params":
            params = entry.get("params")
            if not isinstance(params, list):
                raise CleanTaskBuilderError(f"{task_id}: recovered_params 缺少 params 数组")
            # 仅做数值与范围校验；实际下标契约在实例化时核验。
            for raw_value in params:
                if isinstance(raw_value, bool):
                    raise CleanTaskBuilderError(f"{task_id}: params 包含布尔值")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise CleanTaskBuilderError(f"{task_id}: params 包含非数值") from exc
                if not math.isfinite(value):
                    raise CleanTaskBuilderError(f"{task_id}: params 包含非有限值")
            source_evidence = _require_mapping(
                entry.get("source_evidence"),
                context=f"{task_id}.source_evidence",
            )
            candidate_path_raw = source_evidence.get("candidate_path")
            candidate_sha256 = source_evidence.get("candidate_sha256")
            params_sha256 = source_evidence.get("params_sha256")
            if not isinstance(candidate_path_raw, str) or not candidate_path_raw:
                raise CleanTaskBuilderError(f"{task_id}: candidate_path 缺失")
            if not isinstance(candidate_sha256, str) or SHA256_PATTERN.fullmatch(candidate_sha256) is None:
                raise CleanTaskBuilderError(f"{task_id}: candidate_sha256 非法")
            if not isinstance(params_sha256, str) or SHA256_PATTERN.fullmatch(params_sha256) is None:
                raise CleanTaskBuilderError(f"{task_id}: params_sha256 非法")
            candidate_path = Path(candidate_path_raw)
            if not candidate_path.is_absolute():
                candidate_path = (_repo_root() / candidate_path).resolve()
            if not candidate_path.is_file():
                raise CleanTaskBuilderError(f"{task_id}: candidate 文件不存在: {candidate_path}")
            if _sha256_file(candidate_path) != candidate_sha256:
                raise CleanTaskBuilderError(f"{task_id}: candidate 文件 SHA 漂移")
            try:
                candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CleanTaskBuilderError(f"{task_id}: candidate JSON 解析失败") from exc
            candidate = _require_mapping(candidate_payload, context=f"{task_id}.candidate")
            if candidate.get("params") != params:
                raise CleanTaskBuilderError(f"{task_id}: manifest params 与 candidate 不一致")
            if _sha256_json(params) != params_sha256:
                raise CleanTaskBuilderError(f"{task_id}: params_sha256 漂移")
            candidate_function = candidate.get("function")
            if not isinstance(candidate_function, str) or not candidate_function.strip():
                raise CleanTaskBuilderError(f"{task_id}: candidate.function 缺失")
            entry["_candidate_function"] = candidate_function
            entry["_candidate_file_sha256"] = candidate_sha256
        else:
            if not isinstance(entry.get("reason"), str) or not entry["reason"]:
                raise CleanTaskBuilderError(f"{task_id}: unavailable 缺少 reason")
        by_task_id[task_id] = entry
    return by_task_id, _sha256_bytes(raw_bytes)


def _resolve_prediction_formula(
    *,
    task_id: str,
    selected_expression: str,
    frozen_result_sha256: str,
    frozen_equation_sha256: str | None,
    feature_names: Sequence[str],
    recovery_entries: Mapping[str, Mapping[str, Any]],
) -> FormulaResolution:
    if not selected_expression:
        return FormulaResolution("", "", {}, "missing", None)
    expression_body = extract_expression_body(selected_expression)
    entry_raw = recovery_entries.get(task_id)
    entry = dict(entry_raw) if entry_raw is not None else None
    has_parameter_reference = bool(PARAMETER_REFERENCE_START_PATTERN.search(expression_body))

    if entry is not None:
        if entry["frozen_result_sha256"] != frozen_result_sha256:
            raise CleanTaskBuilderError(f"{task_id}: 公式恢复记录的 result SHA 漂移")
        if entry["equation_sha256"] != frozen_equation_sha256:
            raise CleanTaskBuilderError(f"{task_id}: 公式恢复记录的 equation SHA 漂移")

    if has_parameter_reference:
        if entry is None:
            raise CleanTaskBuilderError(f"{task_id}: 未实例化 params 公式缺少恢复记录")
        if entry["resolution"] == "unavailable":
            return FormulaResolution(
                "",
                expression_body,
                {},
                "unavailable",
                _public_recovery_entry(entry),
            )
        candidate_function = entry.get("_candidate_function")
        if not isinstance(candidate_function, str):
            raise CleanTaskBuilderError(f"{task_id}: 恢复 candidate.function 未通过预检")
        candidate_body = extract_expression_body(candidate_function)
        if _canonical_mapped_ast(candidate_body, feature_names) != _canonical_mapped_ast(
            expression_body,
            feature_names,
        ):
            raise CleanTaskBuilderError(
                f"{task_id}: candidate 公式骨架与 frozen 最终公式不一致"
            )
        expression_body = instantiate_parameters(expression_body, entry["params"])
        status = "recovered_params"
    else:
        if entry is not None:
            raise CleanTaskBuilderError(f"{task_id}: 恢复记录存在，但公式没有 params 引用")
        status = "source_instantiated"

    semantic_expression, variable_mapping = map_indexed_variables(
        expression_body,
        feature_names,
    )
    return FormulaResolution(
        semantic_expression=semantic_expression,
        expression_body=expression_body,
        variable_mapping=variable_mapping,
        status=status,
        manifest_entry=_public_recovery_entry(entry),
    )


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


def _validated_dataset_probe(
    *,
    dataset_id: str,
    variables: Sequence[str],
    target_name: object,
    dataset_probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_probe = dataset_probes.get(dataset_id)
    if raw_probe is None:
        raise CleanTaskBuilderError(f"{dataset_id}: 缺少冻结 dataset probe")
    probe = dict(raw_probe)
    if probe.get("variables") != list(variables):
        raise CleanTaskBuilderError(
            f"{dataset_id}: probe variables 与公式变量顺序不一致"
        )
    if probe.get("target_name") != target_name:
        raise CleanTaskBuilderError(f"{dataset_id}: probe target_name 不一致")
    return probe


def _build_symbolic_request_evidence(
    *,
    expression: str,
    variables: Sequence[str],
    probe: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    allowed_functions = _extract_function_names(expression, variables)
    artifact = _serializable_symbolic_artifact(
        expression,
        variables,
        allowed_functions,
    )
    assumptions = _domain_assumptions(expression)
    deterministic_evidence = {
        "symbolic_artifact": artifact,
        "dataset_probe": dict(probe),
        "domain_assumptions": assumptions,
    }
    return allowed_functions, assumptions, deterministic_evidence


def _build_gt_task(
    row: Mapping[str, Any],
    *,
    index: int,
    contract: PromptSchemaBundle,
    dataset_probes: Mapping[str, Mapping[str, Any]],
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
    source_evidence_hash = row.get("evidence_sha256")
    if not isinstance(source_evidence_hash, str) or not source_evidence_hash:
        raise CleanTaskBuilderError(f"{dataset_id}: evidence_sha256 缺失")
    semantic_expression = expression if isinstance(expression, str) and expression.strip() else ""
    probe = _validated_dataset_probe(
        dataset_id=dataset_id,
        variables=variables,
        target_name=row.get("target"),
        dataset_probes=dataset_probes,
    )
    allowed_functions: list[str] = []
    domain_assumptions: dict[str, Any] = {}
    deterministic_evidence: dict[str, Any] = {"dataset_probe": probe}
    if semantic_expression:
        allowed_functions, domain_assumptions, deterministic_evidence = (
            _build_symbolic_request_evidence(
                expression=semantic_expression,
                variables=variables,
                probe=probe,
            )
        )
    evidence_hash = _sha256_json(
        {
            "ground_truth_source_evidence_sha256": source_evidence_hash,
            "deterministic_evidence": deterministic_evidence,
        }
    )
    request_context = {
        "dataset_id": dataset_id,
        "target_name": row.get("target"),
        "variables": variables,
        "allowed_functions": allowed_functions,
        "expression": semantic_expression or None,
        "original_expression": semantic_expression or None,
        "domain_assumptions": domain_assumptions,
        "probe_points": probe["points"],
        "probe_source": probe["schema_version"],
        "probe_sample_sha256": probe["sample_sha256"],
        "dataset_probe_evidence": probe,
        "deterministic_evidence": deterministic_evidence,
        "ast_source_evidence": {
            "return_ast_dump": row.get("return_ast_dump"),
            "return_source": row.get("return_source"),
            "selection_reason": row.get("selection_reason"),
            "source_checksums": row.get("source_checksums"),
            "ground_truth_source_evidence_sha256": source_evidence_hash,
            "normalized_expression_input": semantic_expression or None,
            "raw_return_source": original_expression
            if isinstance(original_expression, str) and original_expression.strip()
            else None,
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
    ground_truth_variables: Mapping[str, Sequence[str]],
    ground_truth_targets: Mapping[str, str],
    recovery_entries: Mapping[str, Mapping[str, Any]],
    dataset_probes: Mapping[str, Mapping[str, Any]],
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
    selected_expression, expression_source = select_formula_with_source(payload)
    feature_names = _validated_feature_names(payload, task_id=identity["task_id"])
    dataset_id = source.get("dataset_id")
    if not isinstance(dataset_id, str) or dataset_id not in ground_truth_variables:
        raise CleanTaskBuilderError(
            f"{identity['task_id']}: dataset_id 未命中 Ground Truth: {dataset_id!r}"
        )
    expected_variables = list(ground_truth_variables[dataset_id])
    if feature_names != expected_variables:
        raise CleanTaskBuilderError(
            f"{identity['task_id']}: feature_names 与 Ground Truth 变量顺序不一致: "
            f"{feature_names!r} != {expected_variables!r}"
        )
    target_name = payload.get("target_name")
    if target_name != ground_truth_targets[dataset_id]:
        raise CleanTaskBuilderError(
            f"{identity['task_id']}: target_name 与 Ground Truth 不一致: "
            f"{target_name!r} != {ground_truth_targets[dataset_id]!r}"
        )
    formula_resolution = _resolve_prediction_formula(
        task_id=identity["task_id"],
        selected_expression=selected_expression,
        frozen_result_sha256=raw_sha256,
        frozen_equation_sha256=(
            _sha256_text(payload["equation"])
            if isinstance(payload.get("equation"), str)
            else None
        ),
        feature_names=feature_names,
        recovery_entries=recovery_entries,
    )
    expression = formula_resolution.semantic_expression
    variable_mapping = formula_resolution.variable_mapping
    variables = feature_names
    probe = _validated_dataset_probe(
        dataset_id=dataset_id,
        variables=variables,
        target_name=target_name,
        dataset_probes=dataset_probes,
    )
    allowed_functions: list[str] = []
    domain_assumptions: dict[str, Any] = {}
    deterministic_symbolic_evidence: dict[str, Any] = {"dataset_probe": probe}
    if expression:
        (
            allowed_functions,
            domain_assumptions,
            deterministic_symbolic_evidence,
        ) = _build_symbolic_request_evidence(
            expression=expression,
            variables=variables,
            probe=probe,
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
            "selected_expression": selected_expression or None,
            "semantic_expression": expression or None,
            "feature_names": feature_names,
            "variable_mapping": variable_mapping,
            "formula_resolution": {
                "status": formula_resolution.status,
                "expression_body": formula_resolution.expression_body or None,
                "manifest_entry": formula_resolution.manifest_entry,
            },
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
            "deterministic_symbolic_evidence": deterministic_symbolic_evidence,
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
        "allowed_functions": allowed_functions,
        "expression": expression or None,
        "original_expression": expression or None,
        "domain_assumptions": domain_assumptions,
        "probe_points": probe["points"],
        "probe_source": probe["schema_version"],
        "probe_sample_sha256": probe["sample_sha256"],
        "dataset_probe_evidence": probe,
        "deterministic_evidence": deterministic_symbolic_evidence,
        "ast_source_evidence": {
            "selected_expression_source": expression_source or None,
            "selected_expression_before_variable_mapping": selected_expression or None,
            "extracted_expression_body": formula_resolution.expression_body or None,
            "semantic_expression_after_variable_mapping": expression or None,
            "feature_names": feature_names,
            "variable_mapping": variable_mapping,
            "formula_resolution": {
                "status": formula_resolution.status,
                "manifest_entry": formula_resolution.manifest_entry,
            },
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
        no_call_reason = (
            "unresolved_parameter_values"
            if formula_resolution.status == "unavailable"
            else "missing_final_expression"
        )
        no_call = _no_call_record(
            logical_id=logical_id,
            task_type=PRED_TASK_TYPE,
            phase="pred",
            reason=no_call_reason,
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
    formula_recovery_json: Path | None = None,
    dataset_probes_jsonl: Path | None = None,
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
    recovery_path = (
        formula_recovery_json or (repo_root / DEFAULT_FORMULA_RECOVERY_JSON)
    ).resolve()
    probes_path = (
        dataset_probes_jsonl or (repo_root / DEFAULT_DATASET_PROBES_JSONL)
    ).resolve()
    pred_glob = freeze_glob or DEFAULT_FREEZE_GLOB

    gt_rows = _read_jsonl(gt_path)
    if expected_gt_count is not None and len(gt_rows) != expected_gt_count:
        raise CleanTaskBuilderError(
            f"ground_truth_extract 行数不符: 期望 {expected_gt_count}，实际 {len(gt_rows)}"
        )
    dataset_probes, dataset_probes_sha256 = load_dataset_probes(probes_path)
    if expected_gt_count is not None and len(dataset_probes) != expected_gt_count:
        raise CleanTaskBuilderError(
            f"dataset probes 行数不符: 期望 {expected_gt_count}，实际 {len(dataset_probes)}"
        )
    ground_truth_variables: dict[str, list[str]] = {}
    ground_truth_targets: dict[str, str] = {}
    for row in gt_rows:
        dataset_id = row.get("dataset_id")
        variables = row.get("ordered_variables")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise CleanTaskBuilderError("Ground Truth dataset_id 缺失")
        if dataset_id in ground_truth_variables:
            raise CleanTaskBuilderError(f"Ground Truth dataset_id 重复: {dataset_id}")
        if not isinstance(variables, list) or not all(
            isinstance(item, str) and item for item in variables
        ):
            raise CleanTaskBuilderError(f"{dataset_id}: ordered_variables 非法")
        ground_truth_variables[dataset_id] = list(variables)
        target = row.get("target")
        if not isinstance(target, str) or not target:
            raise CleanTaskBuilderError(f"{dataset_id}: Ground Truth target 非法")
        ground_truth_targets[dataset_id] = target

    needs_pred = phase in {"pred", "all"}
    if needs_pred:
        recovery_entries, recovery_manifest_sha256 = load_formula_recovery_manifest(
            recovery_path
        )
        pred_rows, pred_validation = _load_pred_freeze_rows(
            pred_glob,
            expected_pred_count=expected_pred_count,
            repo_root=repo_root,
        )
    else:
        recovery_entries = {}
        recovery_manifest_sha256 = None
        pred_rows = []
        pred_validation = {
            "skipped": True,
            "reason": "phase_gt_does_not_touch_prediction_sources",
            "row_count": 0,
        }

    gt_tasks: list[PlannedTask] = []
    pred_tasks: list[PlannedTask] = []
    no_call_records: list[dict[str, Any]] = []

    for index, row in enumerate(gt_rows, start=1):
        task, no_call = _build_gt_task(
            row,
            index=index,
            contract=contract,
            dataset_probes=dataset_probes,
        )
        if task is not None:
            gt_tasks.append(task)
        if no_call is not None:
            no_call_records.append(no_call)

    if needs_pred:
        for row in pred_rows:
            task, no_call = _build_pred_task(
                row,
                contract=contract,
                ground_truth_variables=ground_truth_variables,
                ground_truth_targets=ground_truth_targets,
                recovery_entries=recovery_entries,
                dataset_probes=dataset_probes,
            )
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
    pred_request_contexts = [task.request for task in pred_tasks]
    pred_request_contexts.extend(
        item["request_context"]
        for item in no_call_records
        if item["phase"] == "pred"
    )
    used_recovery_ids = sorted(
        context["task_id"]
        for context in pred_request_contexts
        if context.get("ast_source_evidence", {})
        .get("formula_resolution", {})
        .get("manifest_entry")
        is not None
    )
    if len(used_recovery_ids) != len(set(used_recovery_ids)):
        raise CleanTaskBuilderError("公式恢复记录被同一 task_id 重复消费")
    unused_recovery_ids = sorted(set(recovery_entries) - set(used_recovery_ids))
    if needs_pred and expected_pred_count == 2250 and unused_recovery_ids:
        raise CleanTaskBuilderError(
            f"全量 clean 构建存在未消费公式恢复记录: {unused_recovery_ids}"
        )

    report = {
        "phase": phase,
        "inputs": {
            "ground_truth_jsonl": str(gt_path),
            "formula_recovery_json": str(recovery_path),
            "dataset_probes_jsonl": str(probes_path),
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
            "formula_recovery": {
                "manifest_sha256": recovery_manifest_sha256,
                "entry_count": len(recovery_entries),
                "used_entry_count": len(used_recovery_ids),
                "used_task_ids": used_recovery_ids,
                "unused_task_ids": unused_recovery_ids,
            },
            "dataset_probes": {
                "jsonl_sha256": dataset_probes_sha256,
                "dataset_count": len(dataset_probes),
                "schema_version": DATASET_PROBE_SCHEMA_VERSION,
            },
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
        "--formula-recovery-json",
        type=Path,
        default=repo_root / DEFAULT_FORMULA_RECOVERY_JSON,
    )
    parser.add_argument(
        "--dataset-probes-jsonl",
        type=Path,
        default=repo_root / DEFAULT_DATASET_PROBES_JSONL,
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
        formula_recovery_json=args.formula_recovery_json,
        dataset_probes_jsonl=args.dataset_probes_jsonl,
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
