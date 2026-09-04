"""将不同算法的公式输出归一化到 sympy 友好的统一表示。"""

from __future__ import annotations

import ast
import re
from typing import Any

from .artifact_schema import (
    build_canonical_symbolic_program,
    extract_return_expression_from_python_function,
    infer_parameter_symbols,
    instantiate_expression,
    validate_canonical_symbolic_program,
)

try:
    import sympy as sp
except ModuleNotFoundError:  # pragma: no cover
    sp = None


def _replace_param_tokens(
    expr: str,
    *,
    parameter_values: list[float] | None = None,
) -> str:
    """把 Python 参数下标映射为 canonical 常数槽位。

    负下标必须按 Python 列表语义结合参数长度解析，不能把 `params[-1]`
    交给 SymPy，也不能猜测它对应哪个常数。
    """

    class ParameterSubscriptNormalizer(ast.NodeTransformer):
        @staticmethod
        def literal_index(node: ast.AST) -> int | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return int(node.value)
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)
                and not isinstance(node.operand.value, bool)
            ):
                value = int(node.operand.value)
                return -value if isinstance(node.op, ast.USub) else value
            return None

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:  # noqa: N802 - ast API
            node = self.generic_visit(node)
            if not isinstance(node.value, ast.Name) or node.value.id != "params":
                return node
            raw_index = self.literal_index(node.slice)
            if raw_index is None:
                raise ValueError("params 下标必须是整数常量")
            if raw_index < 0:
                if parameter_values is None:
                    raise ValueError("负参数下标需要 parameter_values 才能确定 Python 语义")
                resolved_index = len(parameter_values) + raw_index
            else:
                resolved_index = raw_index
            if resolved_index < 0 or (
                parameter_values is not None and resolved_index >= len(parameter_values)
            ):
                raise ValueError(
                    f"参数下标越界: params[{raw_index}], 参数长度={len(parameter_values or [])}"
                )
            return ast.copy_location(ast.Name(id=f"c{resolved_index}", ctx=ast.Load()), node)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"无法解析 Python return 表达式: {exc}") from exc
    transformed = ParameterSubscriptNormalizer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed.body).strip()


def _replace_col_tokens(expr: str) -> str:
    return re.sub(r"\bcol(\d+)\b", lambda m: f"x{m.group(1)}", expr)


def _replace_x_underscore_tokens(expr: str) -> str:
    return re.sub(r"\bx_(\d+)\b", lambda m: f"x{m.group(1)}", expr)


def _replace_x_bracket_tokens(expr: str) -> str:
    return re.sub(r"\bx\[(\d+)\]", lambda m: f"x{m.group(1)}", expr)


def _replace_operon_tokens(expr: str) -> str:
    text = re.sub(r"\bX(\d+)\b", lambda m: f"x{int(m.group(1)) - 1}", expr)
    text = text.replace("^", "**")
    return text


def _replace_symbolic_tokens(expr: str) -> str:
    replacements = {
        "add": "+",
        "mul": "*",
        "sub": "-",
        "pow": "**",
        "inv": "1/",
    }
    text = expr
    for op, target in replacements.items():
        text = re.sub(rf"\b{op}\b", target, text)
    text = re.sub(r"\bpow2\b", "**2", text)
    text = re.sub(r"\bpow3\b", "**3", text)
    text = re.sub(r"\bpow4\b", "**4", text)
    return text


def _shift_one_based_x_tokens(expr: str) -> str:
    """若表达式仅出现 x1/x2/... 而不含 x0，则统一平移为零基索引。"""
    matches = sorted({int(m.group(1)) for m in re.finditer(r"\bx(\d+)\b", expr)})
    if not matches or 0 in matches:
        return expr
    return re.sub(r"\bx(\d+)\b", lambda m: f"x{int(m.group(1)) - 1}", expr)


def _replace_legacy_drsr_tokens(expr: str) -> str:
    replacements = {
        r"\bx\b": "x0",
        r"\bv\b": "x1",
        r"\bt\b": "x0",
    }
    out = expr
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out)
    return out


def _first_function_def(source: str | None) -> ast.FunctionDef | None:
    if not isinstance(source, str) or not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


class _FunctionArgumentNameNormalizer(ast.NodeTransformer):
    def __init__(self, arg_map: dict[str, str]):
        self.arg_map = arg_map

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast API
        replacement = self.arg_map.get(node.id)
        if replacement:
            return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)
        return node


def _standardize_python_function_args_in_return(
    function_source: str,
    raw_return_expr: str,
    *,
    expected_n_features: int | None = None,
) -> tuple[str, list[str]]:
    """按 equation 函数签名把任意特征名映射为 x0/x1/...。

    DRSR 会根据数据集 metadata 在函数签名中生成 `r, m1, kappa` 等变量名。
    benchmark 后处理只接受标准变量槽位，因此这里基于参数顺序做确定性映射。
    """
    func = _first_function_def(function_source)
    if func is None:
        return raw_return_expr, []

    feature_args: list[str] = []
    for arg in func.args.args:
        name = arg.arg
        if name == "params":
            continue
        feature_args.append(name)

    if expected_n_features is not None:
        try:
            feature_args = feature_args[: max(0, int(expected_n_features))]
        except Exception:
            pass

    arg_map = {name: f"x{i}" for i, name in enumerate(feature_args)}
    arg_map = {old: new for old, new in arg_map.items() if old != new}
    if not arg_map:
        return raw_return_expr, []

    try:
        expr_tree = ast.parse(raw_return_expr, mode="eval")
        transformed = _FunctionArgumentNameNormalizer(arg_map).visit(expr_tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed.body).strip(), [f"{old}->{new}" for old, new in sorted(arg_map.items())]
    except Exception:
        text = raw_return_expr
        for old, new in sorted(arg_map.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text, [f"{old}->{new}" for old, new in sorted(arg_map.items())]


def _strip_numpy_prefix(expr: str) -> str:
    expr = expr.replace("numpy.", "")
    expr = expr.replace("np.", "")
    expr = expr.replace("math.", "")
    return expr


def _sanitize_expression(expr: str, *, shift_one_based: bool = True) -> str:
    text = str(expr).strip()
    if text.startswith("lambda x:"):
        text = text.split(":", 1)[1].strip()
    text = _strip_numpy_prefix(text)
    text = _replace_x_bracket_tokens(text)
    text = _replace_x_underscore_tokens(text)
    if shift_one_based:
        text = _shift_one_based_x_tokens(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sympy_locals() -> dict[str, Any]:
    locals_map: dict[str, Any] = {}
    if sp is None:
        return locals_map
    for i in range(64):
        locals_map[f"x{i}"] = sp.Symbol(f"x{i}")
        locals_map[f"c{i}"] = sp.Symbol(f"c{i}")
        locals_map[f"X{i}"] = sp.Symbol(f"x{i}")
        locals_map[f"col{i}"] = sp.Symbol(f"x{i}")
    locals_map.update(
        {
            "Abs": sp.Abs,
            "Max": sp.Max,
            "Min": sp.Min,
            # NumPy 的 maximum/minimum 是逐元素二元算子，不是 SymPy
            # calculus.util 中求函数全局极值的 maximum/minimum。
            "maximum": sp.Max,
            "minimum": sp.Min,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            # 保留为显式函数，不能退化为 `x**(1/3)`，后者对负实数不再
            # 具有 NumPy cbrt 的实数语义。
            "cbrt": sp.Function("cbrt"),
            "square": lambda x: x**2,
            "cube": lambda x: x**3,
        }
    )
    return locals_map


def _parse_sympy_expr(expr: str):
    if sp is None:
        return None
    return sp.sympify(expr, locals=_sympy_locals())


def _count_sympy_nodes(expr) -> int | None:
    if sp is None or expr is None:
        return None
    return int(sum(1 for _ in sp.preorder_traversal(expr)))


def _sympy_tree_depth(expr) -> int | None:
    if sp is None or expr is None:
        return None
    if not getattr(expr, "args", ()):
        return 1
    return 1 + max(_sympy_tree_depth(arg) or 0 for arg in expr.args)


def _collect_operator_set(expr) -> list[str]:
    if sp is None or expr is None:
        return []
    ops: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if getattr(node, "is_Symbol", False) or getattr(node, "is_Number", False):
            continue
        func_name = getattr(getattr(node, "func", None), "__name__", None)
        if not func_name:
            continue
        ops.add(func_name.lower())
    return sorted(ops)


def _normalize_common_expression(expr: str, *, shift_one_based: bool = True) -> tuple[str, Any | None]:
    sanitized = _sanitize_expression(expr, shift_one_based=shift_one_based)
    parsed = None
    if sp is not None:
        parsed = _parse_sympy_expr(sanitized)
        sanitized = str(parsed)
    return sanitized, parsed


def _build_function_source(normalized_expression: str, variables: list[str]) -> str:
    ordered_vars = sorted(variables, key=lambda name: (len(name), name)) if variables else ["x0"]
    sig = ", ".join(ordered_vars + ["params"])
    return f"def equation({sig}):\n    return {normalized_expression}\n"


def normalize_pysr_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    normalized_expression, parsed = _normalize_common_expression(raw_equation)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="pysr",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="pysr_direct",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_external_infix_artifact(
    raw_equation: str,
    *,
    tool_name: str,
    expected_n_features: int | None = None,
    shift_one_based: bool = True,
) -> dict[str, Any]:
    """归一化外部 Python 风格中缀表达式。

    新接入的轻量 wrapper 通常只暴露一个字符串表达式。这里统一处理
    `X0`/`x_0`/`x[0]` 等变量写法，输出 benchmark runner 可消费的
    CanonicalSymbolicProgram。
    """
    raw_text = str(raw_equation)
    expr = re.sub(r"\bX(\d+)\b", lambda m: f"x{m.group(1)}", raw_text)
    expr = expr.replace("^", "**")
    normalized_expression, parsed = _normalize_common_expression(
        expr,
        shift_one_based=shift_one_based,
    )
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name=tool_name,
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode=f"{tool_name}_external_infix",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_qlattice_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    # wrapper 把 QLattice 的 DataFrame 输入列显式命名为 x0, x1, ...，因此
    # sympify 导出的变量已经是零基编号。即使公式只使用 x1，也必须保留为
    # 第二列，不能套用通用的“一基编号自动平移”启发式。
    normalized_expression, parsed = _normalize_common_expression(
        raw_equation,
        shift_one_based=False,
    )
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="QLattice",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="qlattice_direct",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def _gplearn_ast_to_infix(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        if re.fullmatch(r"X\d+", node.id):
            return f"x{node.id[1:]}"
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_gplearn_ast_to_infix(node.operand)})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        args = [_gplearn_ast_to_infix(arg) for arg in node.args]
        if fname == "add" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if fname == "sub" and len(args) == 2:
            return f"({args[0]} - {args[1]})"
        if fname == "mul" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if fname == "div" and len(args) == 2:
            return f"({args[0]} / {args[1]})"
        if fname == "pow" and len(args) == 2:
            return f"({args[0]} ** {args[1]})"
        if fname == "max" and len(args) == 2:
            return f"Max({args[0]}, {args[1]})"
        if fname == "min" and len(args) == 2:
            return f"Min({args[0]}, {args[1]})"
        if fname == "neg" and len(args) == 1:
            return f"(-{args[0]})"
        if fname == "inv" and len(args) == 1:
            return f"(1 / {args[0]})"
        if fname == "abs" and len(args) == 1:
            return f"Abs({args[0]})"
        if fname in {"sqrt", "log", "exp", "sin", "cos", "tan"} and len(args) == 1:
            return f"{fname}({args[0]})"
    raise ValueError(f"不支持的 gplearn 表达式节点: {ast.dump(node)}")


def _infer_gplearn_prefix_variables(raw_equation: str) -> list[str]:
    indices = sorted({int(m.group(1)) for m in re.finditer(r"\bX(\d+)\b", raw_equation)})
    return [f"x{idx}" for idx in indices]


def _infer_gplearn_prefix_operator_set(raw_equation: str) -> list[str]:
    return sorted({m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", raw_equation)})


def _estimate_gplearn_prefix_tree_depth(raw_equation: str) -> int | None:
    max_depth = 0
    depth = 0
    for char in raw_equation:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return max_depth + 1 if max_depth else 1


def _estimate_gplearn_prefix_node_count(raw_equation: str) -> int | None:
    tokens = re.findall(
        r"\b[A-Za-z_]\w*\b|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        raw_equation,
    )
    return len(tokens) or None


def _build_gplearn_prefix_fallback_artifact(
    raw_equation: str,
    *,
    expected_n_features: int | None,
    error: Exception,
) -> dict[str, Any]:
    """保留不可被 Python AST 解析的 gplearn prefix 表达式。

    gplearn 可以生成非常深的 prefix 表达式，`ast.parse` 会先于我们触发
    `too many nested parentheses`。这类表达式仍可用 gplearn protected 语义
    回放，因此这里构造一个最小工件，后续由 runner 的专用 evaluator 执行。
    """
    variables = _infer_gplearn_prefix_variables(raw_equation)
    artifact = build_canonical_symbolic_program(
        tool_name="gplearn",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        normalized_expression="",
        variables=variables,
        operator_set=_infer_gplearn_prefix_operator_set(raw_equation),
        ast_node_count=_estimate_gplearn_prefix_node_count(raw_equation),
        tree_depth=_estimate_gplearn_prefix_tree_depth(raw_equation),
        normalization_mode="gplearn_prefix_unparsed",
        normalization_notes=[f"sympy_parse_skipped:{error.__class__.__name__}:{error}"],
    )
    artifact["sympy_parse_ok"] = False
    artifact["sympy_expression"] = None
    return validate_canonical_symbolic_program(artifact)


def normalize_gplearn_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    raw_text = str(raw_equation)
    try:
        tree = ast.parse(raw_text, mode="eval")
    except SyntaxError as exc:
        return _build_gplearn_prefix_fallback_artifact(
            raw_text,
            expected_n_features=expected_n_features,
            error=exc,
        )
    infix = _gplearn_ast_to_infix(tree.body)
    normalized_expression, parsed = _normalize_common_expression(infix, shift_one_based=False)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="gplearn",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="gplearn_prefix_to_infix",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_e2esr_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    expr = _replace_symbolic_tokens(str(raw_equation))
    normalized_expression, parsed = _normalize_common_expression(expr)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="e2esr",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="e2esr_infix",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_tpsr_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    expr = _replace_symbolic_tokens(str(raw_equation))
    normalized_expression, parsed = _normalize_common_expression(expr)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="tpsr",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="tpsr_infix",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_operon_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    expr = _replace_operon_tokens(str(raw_equation))
    normalized_expression, parsed = _normalize_common_expression(expr)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="pyoperon",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="operon_infix",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_ragsr_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    normalized_expression, parsed = _normalize_common_expression(str(raw_equation), shift_one_based=False)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="ragsr",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="ragsr_evolutionary_forest_expr",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def normalize_dso_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    expr = _replace_symbolic_tokens(str(raw_equation))
    normalized_expression, parsed = _normalize_common_expression(expr)
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="dso",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="dso_sympy_expr",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


UDSR_TRUNK_VARIANT = "udsr_trunk_dso_poly_gp_meld"
UDSR_TRUNK_COMPONENT_FLAGS = {
    "aif": False,
    "dsr": True,
    "lspt": False,
    "gp_meld": True,
    "linear_poly": True,
}
UDSR_TRUNK_COMPONENT_NOTES = (
    "Benchmark uDSR trunk variant: DSO controller + LINEAR/poly token + "
    "GP-meld, without AIF recursive decomposition or LSPT pretraining."
)


def annotate_udsr_trunk_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """给 uDSR trunk 工件补充组件口径，避免和 full uDSR 混淆。"""
    artifact["benchmark_variant"] = UDSR_TRUNK_VARIANT
    artifact["component_flags"] = dict(UDSR_TRUNK_COMPONENT_FLAGS)
    artifact["component_notes"] = UDSR_TRUNK_COMPONENT_NOTES
    return artifact


def normalize_udsr_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    artifact = normalize_dso_artifact(raw_equation, expected_n_features=expected_n_features)
    artifact["tool_name"] = "udsr"
    artifact["normalization_mode"] = "udsr_dso_sympy_expr"
    annotate_udsr_trunk_artifact(artifact)
    return validate_canonical_symbolic_program(artifact)


def normalize_imcts_artifact(raw_equation: str, *, expected_n_features: int | None = None) -> dict[str, Any]:
    normalized_expression, parsed = _normalize_common_expression(
        raw_equation,
        shift_one_based=False,
    )
    # `lambda x: ...` / `x[...]` 是仍保留的 vector source，可直接按其原始
    # 运算树归一化；仅 simplified-only 文本缺少逆转 expand_log 的证据。
    has_vector_source = str(raw_equation).lstrip().startswith("lambda x:") or bool(
        re.search(r"\bx\s*\[\s*\d+\s*\]", str(raw_equation))
    )
    if not has_vector_source and _imcts_has_unrecoverable_expand_log_signature(parsed):
        raise ValueError(
            "iMCTS expand_log(force=True) 导出可能已改变实数域语义，且冻结结果缺少 vector expression"
        )
    variables = sorted({str(sym) for sym in getattr(parsed, "free_symbols", set())}) if parsed is not None else []
    artifact = build_canonical_symbolic_program(
        tool_name="iMCTS",
        raw_equation=raw_equation,
        expected_n_features=expected_n_features,
        python_function_source=_build_function_source(normalized_expression, variables),
        return_expression_source=normalized_expression,
        normalized_expression=normalized_expression,
        variables=variables,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode="imcts_simplified_expr",
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    return validate_canonical_symbolic_program(artifact)


def _imcts_has_unrecoverable_expand_log_signature(parsed: Any | None) -> bool:
    """识别无法从 iMCTS simplified-only 冻结结果逆转的对数展开。

    上游会对所有表达式执行 ``expand(expand_log(..., force=True))``。跨变量
    的裸对数拆分、复数单位、对数幂高度分布，或“含 log 的指数被拆成多个
    exp 因子”都说明原始 vector tree 已丢失；此时继续声称 canonical
    fidelity 会制造假一致。
    """
    if sp is None or parsed is None:
        return False
    if parsed.has(sp.I):
        return True
    log_nodes = [node for node in sp.preorder_traversal(parsed) if node.func == sp.log]
    bare_log_variables = {
        str(node.args[0])
        for node in log_nodes
        if node.args
        and isinstance(node.args[0], sp.Symbol)
        and str(node.args[0]).startswith("x")
    }
    if len(bare_log_variables) >= 2:
        return True
    for node in sp.preorder_traversal(parsed):
        if not isinstance(node, sp.Mul):
            continue
        exp_factors = [factor for factor in node.args if factor.func == sp.exp]
        if len(exp_factors) >= 2 and any(factor.has(sp.log) for factor in exp_factors):
            return True
    if isinstance(parsed, sp.Add) and len(parsed.args) >= 3:
        terms_with_log = sum(1 for term in parsed.args if term.has(sp.log))
        has_log_power = any(
            isinstance(node, sp.Pow)
            and node.base.func == sp.log
            and node.exp.is_Integer
            and node.exp >= 2
            for node in sp.preorder_traversal(parsed)
        )
        if has_log_power and terms_with_log / len(parsed.args) >= 0.8:
            return True
    return False


def _normalize_python_function_artifact(
    *,
    tool_name: str,
    raw_equation: str,
    parameter_values: list[float] | None = None,
    expected_n_features: int | None = None,
    rename_cols: bool = False,
    rename_function_args: bool = False,
) -> dict[str, Any]:
    function_source = str(raw_equation)
    raw_return_expr = extract_return_expression_from_python_function(function_source)
    if not raw_return_expr:
        raise ValueError(f"{tool_name} 原始函数中未找到 return 表达式")

    notes: list[str] = []
    return_expr = raw_return_expr
    if rename_function_args:
        return_expr, arg_notes = _standardize_python_function_args_in_return(
            function_source,
            raw_return_expr,
            expected_n_features=expected_n_features,
        )
        if arg_notes:
            notes.append("function_arg_map:" + ",".join(arg_notes))

    # Python 函数的变量槽位由函数签名唯一决定，不能再套用“没有 x0 就
    # 整体减一”的文本启发式。
    if tool_name == "drsr":
        return_expr = _replace_legacy_drsr_tokens(return_expr)
    expr = _sanitize_expression(return_expr, shift_one_based=False)
    if rename_cols:
        expr = _replace_col_tokens(expr)
    expr = _replace_param_tokens(expr, parameter_values=parameter_values)

    parameter_symbols = infer_parameter_symbols(parameter_values)
    executable_expression = instantiate_expression(
        _replace_param_tokens(return_expr, parameter_values=parameter_values),
        parameter_symbols=parameter_symbols,
        parameter_values=parameter_values,
    )
    if executable_expression and re.search(r"\bc\d+\b|\bparams\s*\[", executable_expression):
        executable_expression = None

    try:
        normalized_expression, parsed = _normalize_common_expression(
            expr,
            shift_one_based=False,
        )
    except Exception as exc:
        # display/SYM 路径不能理解某些合法 NumPy 调用（例如 linalg.norm、
        # where + 比较）。只要 executable AST 已完整冻结，就保留原树供数值
        # 回放，并明确记录 display parse 失败，不能让 SymPy 改写执行语义。
        normalized_expression = expr
        parsed = None
        notes.append(f"sympy_display_parse_failed:{exc.__class__.__name__}:{exc}")
    variables = (
        sorted({str(sym) for sym in getattr(parsed, "free_symbols", set()) if str(sym).startswith("x")})
        if parsed is not None
        else sorted(set(re.findall(r"\bx\d+\b", expr)))
    )
    artifact = build_canonical_symbolic_program(
        tool_name=tool_name,
        raw_equation=raw_equation,
        parameter_values=parameter_values,
        expected_n_features=expected_n_features,
        python_function_source=function_source,
        return_expression_source=raw_return_expr,
        normalized_expression=normalized_expression,
        variables=variables,
        parameter_symbols=parameter_symbols,
        operator_set=_collect_operator_set(parsed),
        ast_node_count=_count_sympy_nodes(parsed),
        tree_depth=_sympy_tree_depth(parsed),
        normalization_mode=f"{tool_name}_python_return",
        normalization_notes=notes,
    )
    artifact["sympy_parse_ok"] = parsed is not None
    artifact["sympy_expression"] = normalized_expression if parsed is not None else None
    artifact["executable_expression"] = executable_expression
    return validate_canonical_symbolic_program(artifact)


def normalize_llmsr_artifact(
    raw_equation: str,
    *,
    parameter_values: list[float] | None = None,
    expected_n_features: int | None = None,
) -> dict[str, Any]:
    return _normalize_python_function_artifact(
        tool_name="llmsr",
        raw_equation=raw_equation,
        parameter_values=parameter_values,
        expected_n_features=expected_n_features,
        rename_cols=False,
        rename_function_args=True,
    )


def normalize_drsr_artifact(
    raw_equation: str,
    *,
    parameter_values: list[float] | None = None,
    expected_n_features: int | None = None,
) -> dict[str, Any]:
    return _normalize_python_function_artifact(
        tool_name="drsr",
        raw_equation=raw_equation,
        parameter_values=parameter_values,
        expected_n_features=expected_n_features,
        rename_cols=True,
        rename_function_args=True,
    )
