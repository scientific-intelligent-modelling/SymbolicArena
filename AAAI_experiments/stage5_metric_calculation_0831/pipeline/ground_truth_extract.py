from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


MANIFEST_RELATIVE = Path("AAAI_experiments/stage5_metric_calculation_0831/manifests/ground_truth.csv")
DEFAULT_JSONL_RELATIVE = Path(
    "AAAI_experiments/stage5_metric_calculation_0831/reports/ground_truth_extract.jsonl"
)
DEFAULT_REPORT_RELATIVE = Path(
    "AAAI_experiments/stage5_metric_calculation_0831/reports/ground_truth_extract.report.json"
)
CSV_SPLIT_FIELDS = ("train_csv", "valid_csv", "id_test_csv", "ood_test_csv")
SHA_VERIFIED_FIELDS = ("metadata_yaml", "formula_py", *CSV_SPLIT_FIELDS)
REQUIRED_MANIFEST_FIELDS = {
    "core50_index",
    "dataset_name",
    "basename",
    "target_name",
    "feature_count",
    "metadata_yaml",
    "metadata_yaml_sha256",
    "train_csv",
    "train_csv_sha256",
    "valid_csv",
    "valid_csv_sha256",
    "id_test_csv",
    "id_test_csv_sha256",
    "ood_test_csv",
    "ood_test_csv_sha256",
    "formula_py",
    "formula_py_sha256",
}
DISALLOWED_EXPR_NODES = (
    ast.Await,
    ast.DictComp,
    ast.FormattedValue,
    ast.GeneratorExp,
    ast.JoinedStr,
    ast.Lambda,
    ast.ListComp,
    ast.NamedExpr,
    ast.SetComp,
    ast.Yield,
    ast.YieldFrom,
)
FORBIDDEN_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


class GroundTruthExtractionError(ValueError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _clone_expr(node: ast.expr) -> ast.expr:
    return copy.deepcopy(node)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ast.unparse(node)


def _raise(dataset_id: str, message: str) -> None:
    raise GroundTruthExtractionError(f"{dataset_id}: {message}")


class _NameSubstituter(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]) -> None:
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._mapping:
            return ast.copy_location(_clone_expr(self._mapping[node.id]), node)
        return node


class _PowCallNormalizer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "pow":
            if len(node.args) != 2 or node.keywords:
                raise GroundTruthExtractionError("pow 调用必须恰好有两个位置参数且不含关键字参数")
            return ast.copy_location(
                ast.BinOp(left=node.args[0], op=ast.Pow(), right=node.args[1]),
                node,
            )
        return node


class _TrivialCastStripper(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "np":
            return node
        if node.func.attr not in {"array", "asarray"}:
            return node
        if len(node.args) != 1:
            return node
        if any(keyword.arg != "dtype" for keyword in node.keywords):
            return node
        return ast.copy_location(node.args[0], node)


def _canonicalize_expression(expr: ast.expr) -> ast.expr:
    expr = _PowCallNormalizer().visit(_clone_expr(expr))
    expr = _TrivialCastStripper().visit(expr)
    ast.fix_missing_locations(expr)
    return expr


def _ensure_no_dynamic_constructs(expr: ast.AST, *, dataset_id: str, context: str) -> None:
    for node in ast.walk(expr):
        if isinstance(node, DISALLOWED_EXPR_NODES):
            _raise(dataset_id, f"{context} 含不支持的动态表达式节点: {type(node).__name__}")
        if isinstance(node, ast.Starred):
            _raise(dataset_id, f"{context} 不支持可变参数展开")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                _raise(dataset_id, f"{context} 含禁用调用: {node.func.id}")


def _function_arg_names(func: ast.FunctionDef, *, dataset_id: str) -> list[str]:
    if func.args.posonlyargs or func.args.vararg or func.args.kwonlyargs or func.args.kwarg:
        _raise(dataset_id, f"函数 {func.name} 含不支持的参数形式")
    return [argument.arg for argument in func.args.args]


def _load_metadata(row: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
    dataset_id = str(row["basename"])
    payload = yaml.safe_load(Path(row["metadata_yaml_abs"]).read_text(encoding="utf-8")) or {}
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        _raise(dataset_id, "metadata.yaml 缺少 dataset 根节点")
    features = dataset.get("features")
    if not isinstance(features, list) or not features:
        _raise(dataset_id, "metadata.yaml 缺少 features 列表")
    feature_names: list[str] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or not isinstance(feature.get("name"), str) or not feature["name"]:
            _raise(dataset_id, f"metadata.yaml 第 {index} 个 feature 缺少合法 name")
        feature_names.append(feature["name"])
    target = dataset.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("name"), str) or not target["name"]:
        _raise(dataset_id, "metadata.yaml 缺少 target.name")
    return dataset, feature_names, target["name"]


def _read_csv_header(path: Path, *, dataset_id: str, split_name: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            _raise(dataset_id, f"{split_name} CSV 为空")
            raise exc
    if not header:
        _raise(dataset_id, f"{split_name} CSV 表头为空")
    return header


def load_manifest_rows(manifest_csv: Path, *, repo_root: Path | None = None) -> list[dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    rows: list[dict[str, Any]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"ground_truth.csv 缺少字段: {missing}")
        for raw_row in reader:
            row = dict(raw_row)
            row["core50_index"] = int(raw_row["core50_index"])
            row["feature_count"] = int(raw_row["feature_count"])
            dataset_id = str(row["basename"])
            for field_name in SHA_VERIFIED_FIELDS:
                resolved = _resolve_repo_path(repo_root, raw_row[field_name])
                if not resolved.exists():
                    raise FileNotFoundError(f"{dataset_id}: {field_name} 不存在: {resolved}")
                actual_sha = sha256_file(resolved)
                expected_sha = raw_row[f"{field_name}_sha256"]
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"{dataset_id}: {field_name} sha256 不匹配，期望 {expected_sha}，实际 {actual_sha}"
                    )
                row[f"{field_name}_abs"] = str(resolved)
                row[f"{field_name}_verified_sha256"] = actual_sha
            rows.append(row)
    rows.sort(key=lambda item: (item["core50_index"], item["basename"], item["dataset_name"]))
    return rows


def _select_target_function(
    row: dict[str, Any],
    *,
    functions: dict[str, ast.FunctionDef],
    ordered_variables: list[str],
    metadata_target_name: str,
) -> tuple[ast.FunctionDef, str]:
    dataset_id = str(row["basename"])
    if metadata_target_name in functions:
        target_function = functions[metadata_target_name]
        if _function_arg_names(target_function, dataset_id=dataset_id) != ordered_variables:
            _raise(
                dataset_id,
                "metadata target.name 命中的函数参数与特征列顺序不一致",
            )
        return target_function, "metadata_target_name"

    candidates = [
        function
        for function in functions.values()
        if _function_arg_names(function, dataset_id=dataset_id) == ordered_variables
    ]
    if len(candidates) != 1:
        _raise(
            dataset_id,
            "候选函数不唯一，无法仅凭特征列顺序稳定定位目标函数",
        )
    return candidates[0], "unique_feature_signature"


def _extract_helper_return_expression(func: ast.FunctionDef, *, dataset_id: str) -> ast.expr:
    returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        _raise(dataset_id, f"辅助函数 {func.name} 含多个 return")
    if len(func.body) == 1 and isinstance(func.body[0], ast.Return):
        expr = func.body[0].value
        if expr is None:
            _raise(dataset_id, f"辅助函数 {func.name} 的 return 为空")
        _ensure_no_dynamic_constructs(expr, dataset_id=dataset_id, context=f"辅助函数 {func.name}")
        return _canonicalize_expression(expr)
    if len(func.body) == 1 and isinstance(func.body[0], ast.With):
        with_stmt = func.body[0]
        if len(with_stmt.body) != 1 or not isinstance(with_stmt.body[0], ast.Return):
            _raise(dataset_id, f"辅助函数 {func.name} 的 with 体不受支持")
        expr = with_stmt.body[0].value
        if expr is None:
            _raise(dataset_id, f"辅助函数 {func.name} 的 return 为空")
        _ensure_no_dynamic_constructs(expr, dataset_id=dataset_id, context=f"辅助函数 {func.name}")
        return _canonicalize_expression(expr)
    _raise(dataset_id, f"辅助函数 {func.name} 含不支持的语句结构")
    raise AssertionError("unreachable")


def _inline_expression(
    expr: ast.expr,
    *,
    dataset_id: str,
    module_assignments: dict[str, ast.expr],
    functions: dict[str, ast.FunctionDef],
    selected_function_name: str,
    stack: tuple[str, ...] = (),
) -> ast.expr:
    expr = _canonicalize_expression(expr)

    class _Inliner(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id in module_assignments:
                return ast.copy_location(_clone_expr(module_assignments[node.id]), node)
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Name):
                return node
            helper_name = node.func.id
            if helper_name == selected_function_name:
                _raise(dataset_id, "目标函数递归调用自身，无法静态展开")
            if helper_name not in functions:
                return node
            if helper_name in stack:
                _raise(dataset_id, f"辅助函数 {helper_name} 递归调用，无法静态展开")
            if node.keywords:
                _raise(dataset_id, f"辅助函数 {helper_name} 调用含关键字参数，拒绝展开")
            helper_function = functions[helper_name]
            parameters = _function_arg_names(helper_function, dataset_id=dataset_id)
            if len(parameters) != len(node.args):
                _raise(dataset_id, f"辅助函数 {helper_name} 参数个数与调用不一致")
            helper_expr = _extract_helper_return_expression(helper_function, dataset_id=dataset_id)
            substituted = _NameSubstituter(
                {name: argument for name, argument in zip(parameters, node.args)}
            ).visit(_clone_expr(helper_expr))
            return ast.copy_location(
                _inline_expression(
                    substituted,
                    dataset_id=dataset_id,
                    module_assignments=module_assignments,
                    functions=functions,
                    selected_function_name=selected_function_name,
                    stack=stack + (helper_name,),
                ),
                node,
            )

    inlined = _Inliner().visit(expr)
    ast.fix_missing_locations(inlined)
    return _canonicalize_expression(inlined)


def _extract_module_assignments(
    tree: ast.Module,
    *,
    dataset_id: str,
) -> tuple[dict[str, ast.FunctionDef], dict[str, ast.expr]]:
    functions: dict[str, ast.FunctionDef] = {}
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                _raise(dataset_id, "顶层赋值必须是单个变量名")
            _ensure_no_dynamic_constructs(node.value, dataset_id=dataset_id, context="顶层赋值")
            assignments[node.targets[0].id] = _canonicalize_expression(node.value)
            continue
        _raise(dataset_id, f"顶层节点类型不受支持: {type(node).__name__}")
    if not functions:
        _raise(dataset_id, "formula.py 中没有可用函数定义")
    return functions, assignments


def extract_ground_truth_record(row: dict[str, Any]) -> dict[str, Any]:
    dataset_id = str(row["basename"])
    _, metadata_feature_names, metadata_target_name = _load_metadata(row)

    split_headers = {
        split_name: _read_csv_header(
            Path(row[f"{split_name}_abs"]),
            dataset_id=dataset_id,
            split_name=split_name,
        )
        for split_name in CSV_SPLIT_FIELDS
    }
    header_values = list(split_headers.values())
    if any(header != header_values[0] for header in header_values[1:]):
        _raise(dataset_id, "四个 split 的 CSV 表头不一致")
    ordered_columns = header_values[0]
    if len(ordered_columns) < 2:
        _raise(dataset_id, "CSV 表头至少需要一个特征列和一个目标列")
    ordered_variables = ordered_columns[:-1]
    csv_target_name = ordered_columns[-1]
    if metadata_feature_names != ordered_variables:
        _raise(dataset_id, "metadata feature_names 与 CSV 特征列顺序不一致")
    if csv_target_name != metadata_target_name:
        _raise(dataset_id, "metadata target.name 与 CSV 目标列不一致")
    if row["feature_count"] != len(ordered_variables):
        _raise(dataset_id, "manifest feature_count 与 CSV 特征列数量不一致")
    if row["target_name"] != metadata_target_name:
        _raise(dataset_id, "manifest target_name 与 metadata target.name 不一致")

    formula_path = Path(row["formula_py_abs"])
    formula_source = formula_path.read_text(encoding="utf-8")
    tree = ast.parse(formula_source, filename=str(formula_path))
    functions, module_assignments = _extract_module_assignments(tree, dataset_id=dataset_id)
    selected_function, selection_reason = _select_target_function(
        row,
        functions=functions,
        ordered_variables=ordered_variables,
        metadata_target_name=metadata_target_name,
    )
    if not selected_function.body:
        _raise(dataset_id, f"目标函数 {selected_function.name} 为空")
    returns = [node for node in ast.walk(selected_function) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        _raise(dataset_id, f"目标函数 {selected_function.name} 含多个 return")

    local_assignments: dict[str, ast.expr] = {}
    for statement in selected_function.body[:-1]:
        if not isinstance(statement, ast.Assign):
            _raise(
                dataset_id,
                f"目标函数 {selected_function.name} 仅支持前置简单赋值，遇到 {type(statement).__name__}",
            )
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            _raise(dataset_id, f"目标函数 {selected_function.name} 赋值目标不明确")
        value = _NameSubstituter(local_assignments).visit(_clone_expr(statement.value))
        _ensure_no_dynamic_constructs(
            value,
            dataset_id=dataset_id,
            context=f"目标函数 {selected_function.name} 局部赋值",
        )
        local_assignments[statement.targets[0].id] = _canonicalize_expression(value)

    last_statement = selected_function.body[-1]
    if not isinstance(last_statement, ast.Return):
        _raise(dataset_id, f"目标函数 {selected_function.name} 结尾不是 return")
    if last_statement.value is None:
        _raise(dataset_id, f"目标函数 {selected_function.name} return 为空")
    _ensure_no_dynamic_constructs(
        last_statement.value,
        dataset_id=dataset_id,
        context=f"目标函数 {selected_function.name} return",
    )

    return_source = _source_segment(formula_source, last_statement.value)
    normalized_expr = _NameSubstituter(local_assignments).visit(_clone_expr(last_statement.value))
    normalized_expr = _inline_expression(
        normalized_expr,
        dataset_id=dataset_id,
        module_assignments=module_assignments,
        functions=functions,
        selected_function_name=selected_function.name,
    )
    normalized_expr = _canonicalize_expression(normalized_expr)

    remaining_names = sorted(
        {
            node.id
            for node in ast.walk(normalized_expr)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
    )
    allowed_names = set(ordered_variables) | {"np"}
    unknown_names = sorted(name for name in remaining_names if name not in allowed_names)
    if unknown_names:
        _raise(dataset_id, f"归一化表达式仍含未解析名称: {unknown_names}")

    record = {
        "dataset_id": dataset_id,
        "target": metadata_target_name,
        "ordered_variables": ordered_variables,
        "function_name": selected_function.name,
        "selection_reason": selection_reason,
        "return_source": return_source,
        "return_ast_dump": ast.dump(last_statement.value, annotate_fields=True, include_attributes=False),
        "normalized_expression_input": ast.unparse(normalized_expr),
        "source_checksums": {
            "metadata_yaml_sha256": row["metadata_yaml_verified_sha256"],
            "formula_py_sha256": row["formula_py_verified_sha256"],
            "train_csv_sha256": row["train_csv_verified_sha256"],
            "valid_csv_sha256": row["valid_csv_verified_sha256"],
            "id_test_csv_sha256": row["id_test_csv_verified_sha256"],
            "ood_test_csv_sha256": row["ood_test_csv_verified_sha256"],
        },
    }
    record["evidence_sha256"] = _sha256_text(
        _canonical_json(
            {
                "dataset_id": record["dataset_id"],
                "target": record["target"],
                "ordered_variables": record["ordered_variables"],
                "function_name": record["function_name"],
                "selection_reason": record["selection_reason"],
                "return_source": record["return_source"],
                "return_ast_dump": record["return_ast_dump"],
                "normalized_expression_input": record["normalized_expression_input"],
                "source_checksums": record["source_checksums"],
            }
        )
    )
    return record


def build_ground_truth_records(
    manifest_csv: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = repo_root or _repo_root()
    manifest_rows = load_manifest_rows(manifest_csv, repo_root=repo_root)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    selection_counts = {
        "metadata_target_name": 0,
        "unique_feature_signature": 0,
    }
    for row in manifest_rows:
        try:
            record = extract_ground_truth_record(row)
        except GroundTruthExtractionError as exc:
            failures.append(
                {
                    "dataset_id": str(row["basename"]),
                    "reason": str(exc),
                }
            )
            continue
        selection_counts[record["selection_reason"]] = (
            selection_counts.get(record["selection_reason"], 0) + 1
        )
        records.append(record)

    report = {
        "manifest_path": str(manifest_csv.resolve()),
        "manifest_sha256": sha256_file(manifest_csv),
        "dataset_count": len(manifest_rows),
        "success_count": len(records),
        "failure_count": len(failures),
        "selection_counts": selection_counts,
        "failures": failures,
    }
    return records, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="静态抽取 Core50 Ground Truth 公式证据")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=_repo_root() / MANIFEST_RELATIVE,
        help="ground_truth.csv 路径",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=_repo_root() / DEFAULT_JSONL_RELATIVE,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=_repo_root() / DEFAULT_REPORT_RELATIVE,
        help="输出报告 JSON 路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    records, report = build_ground_truth_records(args.manifest_csv.resolve(), repo_root=_repo_root())
    write_jsonl(args.output_jsonl.resolve(), records)
    write_json(args.report_json.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
