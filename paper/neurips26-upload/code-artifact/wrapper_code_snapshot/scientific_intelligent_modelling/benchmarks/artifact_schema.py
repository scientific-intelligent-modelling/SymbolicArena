"""统一符号工件协议。

Phase 1 只负责定义一个稳定、可 JSON 序列化的最小协议，
以及若干从现有 wrapper 输出过渡到该协议的基础工具函数。
"""

from __future__ import annotations

import ast
import re
from typing import Any


CSP_VERSION = "csp_v1"


def _normalize_float_list(values: Any) -> list[float] | None:
    """把参数列表归一化为普通 float 列表。"""
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        return None
    out: list[float] = []
    for item in values:
        try:
            out.append(float(item))
        except Exception:
            return None
    return out


def infer_raw_equation_kind(raw_equation: Any) -> str:
    """粗粒度识别原始公式形态。"""
    if not isinstance(raw_equation, str):
        return "unknown"
    text = raw_equation.strip()
    if not text:
        return "empty"
    if text.startswith("def ") or "\ndef " in text:
        return "python_function"
    if text.startswith("lambda "):
        return "python_lambda"
    if re.search(r"\b(add|sub|mul|div|pow|sin|cos|log|exp)\s*\(", text):
        return "prefix_expression"
    if "\n" in text:
        return "multiline_expression"
    return "plain_expression"


def infer_variable_names(expression: str | None) -> list[str]:
    """从表达式文本中提取变量名。

    这里不做复杂语义理解，只提取当前仓库常见的变量槽位：
    - x0, x1, ...
    - col0, col1, ...
    """
    if not isinstance(expression, str) or not expression.strip():
        return []
    names = sorted(
        {
            m.group(0)
            for m in re.finditer(r"\b(?:x\d+|col\d+)\b", expression)
        }
    )
    return names


def infer_parameter_symbols(parameter_values: list[float] | None) -> list[str]:
    """按参数个数生成统一常数符号。"""
    if not parameter_values:
        return []
    return [f"c{i}" for i in range(len(parameter_values))]


def instantiate_expression(
    expression: str | None,
    *,
    parameter_symbols: list[str] | None,
    parameter_values: list[float] | None,
) -> str | None:
    """将统一表达式中的参数符号替换成具体数值。"""
    if not isinstance(expression, str) or not expression.strip():
        return None
    text = expression.strip()
    if not parameter_symbols or not parameter_values:
        return text

    out = text
    for idx, symbol in enumerate(parameter_symbols):
        if idx >= len(parameter_values):
            break
        value = repr(float(parameter_values[idx]))
        out = re.sub(rf"\b{re.escape(symbol)}\b", value, out)
        out = re.sub(rf"(?<!\w)params\[{idx}\](?!\w)", value, out)
    return out


def infer_variable_indices(variables: list[str] | None) -> list[int]:
    """从变量名列表中提取 x{i} 的索引。"""
    if not isinstance(variables, list):
        return []
    indices: set[int] = set()
    for name in variables:
        if not isinstance(name, str):
            continue
        match = re.fullmatch(r"x(\d+)", name.strip())
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def extract_return_expression_from_python_function(source: str | None) -> str | None:
    """从 def equation(...) 代码中抽取第一条 return 表达式。"""
    if not isinstance(source, str) or not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    try:
                        return ast.unparse(stmt.value).strip()
                    except Exception:
                        return None
    return None


def build_canonical_symbolic_program(
    *,
    tool_name: str,
    raw_equation: Any,
    parameter_values: list[float] | tuple[float, ...] | None = None,
    expected_n_features: int | None = None,
    python_function_source: str | None = None,
    return_expression_source: str | None = None,
    normalized_expression: str | None = None,
    variables: list[str] | None = None,
    parameter_symbols: list[str] | None = None,
    operator_set: list[str] | None = None,
    ast_node_count: int | None = None,
    tree_depth: int | None = None,
    normalization_mode: str = "wrapper_raw",
    normalization_notes: list[str] | None = None,
    fidelity_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造最小 CanonicalSymbolicProgram。

    Phase 1 默认只保证：
    - 字段结构稳定
    - JSON 可序列化
    - 可以承载后续 normalizer 的增量信息
    """
    raw_equation_text = "" if raw_equation is None else str(raw_equation)
    raw_kind = infer_raw_equation_kind(raw_equation_text)

    function_source = python_function_source
    if function_source is None and raw_kind == "python_function":
        function_source = raw_equation_text

    extracted_return = return_expression_source
    if extracted_return is None and function_source:
        extracted_return = extract_return_expression_from_python_function(function_source)

    normalized_params = _normalize_float_list(parameter_values)
    derived_param_symbols = parameter_symbols or infer_parameter_symbols(normalized_params)

    if normalized_expression is None:
        if extracted_return:
            normalized_expression = extracted_return
        elif raw_kind in {"plain_expression", "multiline_expression", "prefix_expression"}:
            normalized_expression = raw_equation_text

    derived_variables = variables or infer_variable_names(
        extracted_return or normalized_expression or raw_equation_text
    )

    artifact = {
        "version": CSP_VERSION,
        "tool_name": str(tool_name),
        "raw_equation": raw_equation_text,
        "raw_equation_kind": raw_kind,
        "python_function_source": function_source,
        "return_expression_source": extracted_return,
        "normalized_expression": normalized_expression,
        "instantiated_expression": instantiate_expression(
            normalized_expression,
            parameter_symbols=derived_param_symbols,
            parameter_values=normalized_params,
        ),
        "variables": derived_variables,
        "parameter_symbols": derived_param_symbols,
        "parameter_values": normalized_params,
        "expected_n_features": int(expected_n_features) if expected_n_features is not None else None,
        "operator_set": list(operator_set) if operator_set else [],
        "ast_node_count": int(ast_node_count) if ast_node_count is not None else None,
        "tree_depth": int(tree_depth) if tree_depth is not None else None,
        "normalization_mode": str(normalization_mode),
        "normalization_notes": list(normalization_notes or []),
        "artifact_valid": True,
        "validation_errors": [],
        "fidelity_check": dict(fidelity_check or {}),
    }
    return validate_canonical_symbolic_program(artifact)


def validate_canonical_symbolic_program(
    artifact: dict[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """校验并返回规范化后的工件字典。"""
    if not isinstance(artifact, dict):
        raise TypeError("CanonicalSymbolicProgram 必须是 dict")

    required_fields = {
        "version": str,
        "tool_name": str,
        "raw_equation": str,
        "raw_equation_kind": str,
        "normalization_mode": str,
        "normalization_notes": list,
        "variables": list,
        "parameter_symbols": list,
        "operator_set": list,
        "artifact_valid": bool,
        "validation_errors": list,
        "fidelity_check": dict,
    }
    for field_name, field_type in required_fields.items():
        if field_name not in artifact:
            raise ValueError(f"CanonicalSymbolicProgram 缺少字段: {field_name}")
        if not isinstance(artifact[field_name], field_type):
            raise TypeError(
                f"CanonicalSymbolicProgram 字段 {field_name} 类型错误: "
                f"期望 {field_type.__name__}, 实际 {type(artifact[field_name]).__name__}"
            )

    instantiated_expression = artifact.get("instantiated_expression")
    if instantiated_expression is not None and not isinstance(instantiated_expression, str):
        raise TypeError(
            "CanonicalSymbolicProgram 字段 instantiated_expression 类型错误: "
            f"期望 str 或 None, 实际 {type(instantiated_expression).__name__}"
        )

    if artifact["version"] != CSP_VERSION:
        raise ValueError(
            f"CanonicalSymbolicProgram 版本不支持: {artifact['version']!r}, "
            f"当前仅支持 {CSP_VERSION!r}"
        )

    if require_complete:
        if not artifact["python_function_source"]:
            raise ValueError("完整工件要求 python_function_source 非空")
        if not artifact["normalized_expression"]:
            raise ValueError("完整工件要求 normalized_expression 非空")

    validation_errors = list(artifact.get("validation_errors") or [])
    expected_n_features = artifact.get("expected_n_features")
    if expected_n_features is not None:
        try:
            expected_n_features = int(expected_n_features)
        except Exception as err:
            raise TypeError(f"expected_n_features 类型错误: {err}") from err
        if expected_n_features < 0:
            raise ValueError("expected_n_features 不能小于 0")
        variable_indices = infer_variable_indices(artifact.get("variables"))
        overflow = [idx for idx in variable_indices if idx >= expected_n_features]
        if overflow:
            validation_errors.append(
                "变量索引超出输入维度: "
                f"expected_n_features={expected_n_features}, "
                f"found={artifact.get('variables')}"
            )

    artifact["validation_errors"] = validation_errors
    artifact["artifact_valid"] = len(validation_errors) == 0
    artifact["instantiated_expression"] = instantiate_expression(
        artifact.get("normalized_expression"),
        parameter_symbols=artifact.get("parameter_symbols"),
        parameter_values=artifact.get("parameter_values"),
    )

    return artifact
