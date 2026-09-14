"""Stage5 符号证据工具。

仅做安全 AST / 前缀表达式恢复，不使用 `eval` 或 `sympify`。
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
import random
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Collection, Mapping, Sequence

import sympy as sp


class SymbolicEvidenceError(ValueError):
    """不可信表达式或证据契约违规。"""


class SimplificationContractError(ValueError):
    """化简结果被确定性反例否定。"""

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


class real_cbrt(sp.Function):
    """保留实立方根这一元算子，避免负数被改为复数主值分数幂。"""

    nargs = 1

    @classmethod
    def eval(cls, value):
        if value.is_number and value.is_real is True:
            return sp.real_root(value, 3)

    def _eval_evalf(self, prec):
        value = self.args[0].evalf(prec)
        if value.is_real is True:
            return sp.real_root(value, 3).evalf(prec)

    def _eval_is_real(self):
        return self.args[0].is_real


@dataclass
class _BuildContext:
    allowed_variables: set[str] | None
    allowed_functions: set[str] | None
    inferred_variables: set[str]
    source_text: str
    exact_numeric_literals: bool = False
    evaluate_expressions: bool = True
    symbols: dict[str, sp.Symbol] = field(default_factory=dict)
    source_functions: set[str] = field(default_factory=set)

    def symbol(self, name: str) -> sp.Symbol:
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name)
        return self.symbols[name]


NUMPY_ATTRIBUTE_WHITELIST = {
    "abs",
    "arccos",
    "arcsin",
    "cbrt",
    "clip",
    "cos",
    "cosh",
    "divide",
    "exp",
    "log",
    "log1p",
    "maximum",
    "mean",
    "minimum",
    "pi",
    "sin",
    "sqrt",
    "tan",
    "tanh",
    "where",
}
CONSTANT_NAMES = {
    "E": sp.E,
    "I": sp.I,
    "nan": sp.nan,
    "pi": sp.pi,
    "zoo": sp.zoo,
}
OPAQUE_FUNCTIONS = {"gradient", "index", "mean", "norm"}
UNEVALUATED_AST_NODE_THRESHOLD = 160
SYMBOLIC_PROOF_NODE_LIMIT = 64
SYMBOLIC_PROOF_TIMEOUT_SECONDS = 0.25
NUMERIC_PROBE_TIMEOUT_SECONDS = 0.25
NUMERIC_PROBE_TOTAL_TIMEOUT_SECONDS = 0.25
PREFIX_TOKEN_RE = re.compile(
    r"""
    (?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    |(?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<lpar>\()
    |(?P<rpar>\))
    |(?P<comma>,)
    |(?P<space>\s+)
    """,
    re.VERBOSE,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_name(name: str) -> str:
    aliases = {
        "Abs": "abs",
        "Max": "maximum",
        "Min": "minimum",
        "Piecewise": "where",
        "piecewise": "where",
        "atan": "arctan",
        "cbrt": "real_cbrt",
    }
    return aliases.get(name, name)


def _record_function(ctx: _BuildContext, name: str) -> str:
    normalized = _normalize_name(name)
    if ctx.allowed_functions is not None and normalized not in ctx.allowed_functions:
        raise SymbolicEvidenceError(f"检测到未授权函数: {normalized}")
    ctx.source_functions.add(normalized)
    return normalized


def _resolve_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return _normalize_name(node.id)
    if isinstance(node, ast.Attribute):
        if (
            node.attr == "norm"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "linalg"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "np"
        ):
            return "norm"
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise SymbolicEvidenceError("只允许 np.<math> 形式的属性调用")
        if node.attr not in NUMPY_ATTRIBUTE_WHITELIST:
            raise SymbolicEvidenceError(f"不支持的 np 属性: np.{node.attr}")
        return _normalize_name(node.attr)
    raise SymbolicEvidenceError("只允许显式白名单函数调用")


def _resolve_constant(node: ast.AST) -> sp.Basic:
    if isinstance(node, ast.Name) and node.id in CONSTANT_NAMES:
        return CONSTANT_NAMES[node.id]
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise SymbolicEvidenceError("只允许 np.pi 常量属性")
        if node.attr != "pi":
            raise SymbolicEvidenceError(f"不支持的常量属性: np.{node.attr}")
        return sp.pi
    raise SymbolicEvidenceError("未知常量")


def _literal_subscript_index(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
            return -int(operand.value)
    raise SymbolicEvidenceError("只允许静态整数下标访问")


def _parameter_symbol_name(index: int) -> str:
    if index < 0:
        return f"params__neg_{abs(index)}"
    return f"params__{index}"


def _parse_real_number_text(text: str, *, exact_numeric_literals: bool) -> sp.Basic:
    stripped = text.strip()
    if any(character in stripped for character in ".eE"):
        value = float(stripped)
        if not math.isfinite(value):
            raise SymbolicEvidenceError(f"只允许有限常量，收到: {stripped!r}")
        if exact_numeric_literals:
            return sp.Rational(stripped)
        return sp.Float(stripped)
    return sp.Integer(int(stripped))


def _ensure_real_number(value: object) -> sp.Basic:
    if isinstance(value, bool):
        raise SymbolicEvidenceError("布尔字面量不能作为普通数值常量")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SymbolicEvidenceError(f"只允许有限常量，收到: {value!r}")
        return sp.Float(repr(value))
    raise SymbolicEvidenceError(f"不支持的字面量常量: {value!r}")


def _extract_expression_ast(source: str) -> tuple[ast.AST, str, set[str]]:
    try:
        parsed = ast.parse(source, mode="eval")
        return parsed.body, "expression", set()
    except SyntaxError:
        module = ast.parse(source, mode="exec")
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise SymbolicEvidenceError("表达式必须是单个表达式或单个函数定义")
        fn = module.body[0]
        if fn.decorator_list:
            raise SymbolicEvidenceError("不允许装饰器")
        statements = list(fn.body)
        while statements and isinstance(statements[0], ast.Expr):
            doc_value = statements[0].value
            if isinstance(doc_value, ast.Constant) and isinstance(doc_value.value, str):
                statements = statements[1:]
                continue
            break
        if len(statements) != 1 or not isinstance(statements[0], ast.Return) or statements[0].value is None:
            raise SymbolicEvidenceError("函数定义必须只包含一个 return 表达式")
        inferred_variables = {
            arg.arg
            for arg in fn.args.args
            if arg.arg != "params"
        }
        return statements[0].value, "function_def", inferred_variables


def _looks_like_prefix_expression(source: str) -> bool:
    stripped = source.lstrip()
    return (not stripped.startswith("def ")) and bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(", stripped))


@dataclass(frozen=True)
class _PrefixToken:
    kind: str
    value: str


class _PrefixTokenStream:
    def __init__(self, source: str) -> None:
        self.tokens: list[_PrefixToken] = []
        position = 0
        while position < len(source):
            match = PREFIX_TOKEN_RE.match(source, position)
            if match is None:
                raise SymbolicEvidenceError(
                    f"prefix 表达式包含非法字符: {source[position:position+16]!r}"
                )
            position = match.end()
            kind = match.lastgroup
            if kind == "space":
                continue
            assert kind is not None
            self.tokens.append(_PrefixToken(kind, match.group(kind)))
        self.index = 0

    def peek(self) -> _PrefixToken | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self) -> _PrefixToken:
        token = self.peek()
        if token is None:
            raise SymbolicEvidenceError("prefix 表达式提前结束")
        self.index += 1
        return token

    def expect(self, kind: str) -> _PrefixToken:
        token = self.pop()
        if token.kind != kind:
            raise SymbolicEvidenceError(
                f"prefix 表达式期望 {kind}，实际是 {token.kind}"
            )
        return token


def _parse_prefix_number(text: str, *, exact_numeric_literals: bool) -> sp.Basic:
    return _parse_real_number_text(text, exact_numeric_literals=exact_numeric_literals)


def _resolve_identifier(name: str, ctx: _BuildContext) -> sp.Basic:
    if name in CONSTANT_NAMES:
        return CONSTANT_NAMES[name]
    normalized = _normalize_name(name)
    if normalized in {
        "abs",
        "clip",
        "sin",
        "cos",
        "cosh",
        "tan",
        "tanh",
        "sinh",
        "exp",
        "log",
        "log1p",
        "sqrt",
        "mean",
        "norm",
        "where",
        "divide",
        "div",
        "power",
        "add",
        "sub",
        "mul",
        "maximum",
        "minimum",
            "arctan",
            "arccos",
            "arcsin",
        "gradient",
        "compare",
    }:
        raise SymbolicEvidenceError(f"函数名不能作为裸变量出现: {name}")
    if name == "params":
        raise SymbolicEvidenceError("只允许 params[整数] 形式的下标访问")
    if ctx.allowed_variables is not None and name not in ctx.allowed_variables and name not in ctx.inferred_variables:
        raise SymbolicEvidenceError(f"检测到未授权变量: {name}")
    return ctx.symbol(name)


def _require_arity(name: str, args: list[sp.Basic], expected: int) -> list[sp.Basic]:
    if len(args) != expected:
        raise SymbolicEvidenceError(f"{name} 需要 {expected} 个参数，收到 {len(args)} 个")
    return args


def _require_min_arity(name: str, args: list[sp.Basic], minimum: int) -> list[sp.Basic]:
    if len(args) < minimum:
        raise SymbolicEvidenceError(f"{name} 至少需要 {minimum} 个参数，收到 {len(args)} 个")
    return args


def _build_where(args: list[sp.Basic], ctx: _BuildContext) -> sp.Basic:
    condition, when_true, when_false = _require_arity("where", args, 3)
    if not (
        bool(getattr(condition, "is_Boolean", False))
        or bool(getattr(condition, "is_Relational", False))
    ):
        raise SymbolicEvidenceError("where 的第一个参数必须是比较条件")
    return sp.Piecewise(
        (when_true, condition),
        (when_false, True),
        evaluate=ctx.evaluate_expressions,
    )


def _call_handler(name: str, ctx: _BuildContext):
    unary = {
        "abs": sp.Abs,
        "arccos": sp.acos,
        "arcsin": sp.asin,
        "arctan": sp.atan,
        "cos": sp.cos,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "log": sp.log,
        "real_cbrt": real_cbrt,
        "sin": sp.sin,
        "sinh": sp.sinh,
        "sqrt": sp.sqrt,
        "tan": sp.tan,
        "tanh": sp.tanh,
    }
    if name in unary:
        return lambda args: unary[name](
            _require_arity(name, args, 1)[0],
            evaluate=ctx.evaluate_expressions,
        )
    if name == "log1p":
        return lambda args: sp.log(
            sp.Add(
                _require_arity(name, args, 1)[0],
                sp.Integer(1),
                evaluate=ctx.evaluate_expressions,
            ),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "clip":
        return lambda args: sp.Min(
            _require_arity(name, args, 3)[2],
            sp.Max(
                _require_arity(name, args, 3)[1],
                _require_arity(name, args, 3)[0],
                evaluate=ctx.evaluate_expressions,
            ),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "gradient":
        gradient = sp.Function("gradient")
        return lambda args: gradient(_require_arity(name, args, 1)[0])
    if name == "mean":
        mean = sp.Function("mean")
        return lambda args: mean(_require_arity(name, args, 1)[0])
    if name == "norm":
        norm = sp.Function("norm")
        return lambda args: norm(_require_arity(name, args, 1)[0])
    if name in {"divide", "div"}:
        return lambda args: sp.Mul(
            _require_arity(name, args, 2)[0],
            sp.Pow(
                _require_arity(name, args, 2)[1],
                sp.Integer(-1),
                evaluate=ctx.evaluate_expressions,
            ),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "power":
        return lambda args: sp.Pow(
            *_require_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "add":
        return lambda args: sp.Add(
            *_require_min_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "sub":
        return lambda args: sp.Add(
            _require_arity(name, args, 2)[0],
            sp.Mul(
                sp.Integer(-1),
                _require_arity(name, args, 2)[1],
                evaluate=ctx.evaluate_expressions,
            ),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "mul":
        return lambda args: sp.Mul(
            *_require_min_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "maximum":
        return lambda args: sp.Max(
            *_require_min_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "minimum":
        return lambda args: sp.Min(
            *_require_min_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    if name == "where":
        return lambda args: _build_where(args, ctx)
    if name == "compare":
        return lambda args: sp.StrictGreaterThan(
            *_require_arity(name, args, 2),
            evaluate=ctx.evaluate_expressions,
        )
    raise SymbolicEvidenceError(f"不支持的函数: {name}")


def _parse_prefix_term(stream: _PrefixTokenStream, ctx: _BuildContext) -> sp.Basic:
    token = stream.pop()
    if token.kind == "number":
        return _parse_prefix_number(
            token.value,
            exact_numeric_literals=ctx.exact_numeric_literals,
        )
    if token.kind != "ident":
        raise SymbolicEvidenceError(
            f"prefix 表达式期望 number/ident，实际是 {token.kind}"
        )
    name = token.value
    if stream.peek() is not None and stream.peek().kind == "lpar":
        stream.pop()
        args: list[sp.Basic] = []
        while stream.peek() is not None and stream.peek().kind != "rpar":
            args.append(_parse_prefix_term(stream, ctx))
            next_token = stream.peek()
            if next_token is None:
                raise SymbolicEvidenceError("prefix 表达式缺少右括号")
            if next_token.kind == "comma":
                stream.pop()
        stream.expect("rpar")
        normalized = _record_function(ctx, name)
        return _call_handler(normalized, ctx)(args)
    return _resolve_identifier(name, ctx)


def _build_prefix_expression(source: str, ctx: _BuildContext) -> sp.Basic:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    stream = _PrefixTokenStream(source)
    expr = _parse_prefix_term(stream, ctx)
    if stream.peek() is not None:
        raise SymbolicEvidenceError("prefix 表达式存在未消费尾部")
    return expr


def _convert_compare(node: ast.Compare, ctx: _BuildContext) -> sp.Basic:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise SymbolicEvidenceError("只支持单一比较运算")
    left = _convert_node(node.left, ctx)
    right = _convert_node(node.comparators[0], ctx)
    op = node.ops[0]
    if isinstance(op, ast.Gt):
        return sp.StrictGreaterThan(left, right, evaluate=ctx.evaluate_expressions)
    if isinstance(op, ast.GtE):
        return sp.GreaterThan(left, right, evaluate=ctx.evaluate_expressions)
    if isinstance(op, ast.Lt):
        return sp.StrictLessThan(left, right, evaluate=ctx.evaluate_expressions)
    if isinstance(op, ast.LtE):
        return sp.LessThan(left, right, evaluate=ctx.evaluate_expressions)
    if isinstance(op, ast.Eq):
        return sp.Eq(left, right, evaluate=ctx.evaluate_expressions)
    if isinstance(op, ast.NotEq):
        return sp.Ne(left, right, evaluate=ctx.evaluate_expressions)
    raise SymbolicEvidenceError(f"不支持的比较运算: {type(op).__name__}")


def _compare_relation(node: ast.AST, ctx: _BuildContext) -> sp.Basic:
    if isinstance(node, ast.Compare):
        return _convert_compare(node, ctx)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        left = _compare_relation(node.left, ctx)
        right = _compare_relation(node.right, ctx)
        relation_type = sp.And if isinstance(node.op, ast.BitAnd) else sp.Or
        return relation_type(left, right, evaluate=ctx.evaluate_expressions)
    raise SymbolicEvidenceError(
        "where 的条件必须是比较表达式，或由 &/| 连接的比较表达式"
    )


def _convert_piecewise_call(node: ast.Call, ctx: _BuildContext) -> sp.Basic:
    """只接受与二分支 ``where`` 等价的标准 SymPy Piecewise 写法。"""

    if len(node.args) != 2:
        raise SymbolicEvidenceError("Piecewise 仅允许两个分支")
    branches: list[tuple[sp.Basic, sp.Basic | bool]] = []
    for index, branch in enumerate(node.args):
        if not isinstance(branch, ast.Tuple) or len(branch.elts) != 2:
            raise SymbolicEvidenceError("Piecewise 分支必须是 (value, condition) 二元组")
        value_node, condition_node = branch.elts
        value = _convert_node(value_node, ctx)
        if index == 1:
            if not (
                isinstance(condition_node, ast.Constant)
                and condition_node.value is True
            ):
                raise SymbolicEvidenceError("Piecewise 最后一个分支条件必须为 True")
            condition: sp.Basic | bool = True
        else:
            condition = _compare_relation(condition_node, ctx)
        branches.append((value, condition))
    return sp.Piecewise(*branches, evaluate=ctx.evaluate_expressions)


def _convert_constant(node: ast.Constant, ctx: _BuildContext) -> sp.Basic:
    if isinstance(node.value, float) and ctx.exact_numeric_literals:
        source_text = ast.get_source_segment(ctx.source_text, node)
        if isinstance(source_text, str):
            return _parse_real_number_text(
                source_text,
                exact_numeric_literals=True,
            )
    return _ensure_real_number(node.value)


def _convert_clip_call(node: ast.Call, ctx: _BuildContext) -> sp.Basic:
    if len(node.args) > 3:
        raise SymbolicEvidenceError("clip 最多接受 3 个位置参数")
    missing = object()
    values: list[ast.AST | None | object] = [missing, missing, missing]
    for index, argument in enumerate(node.args):
        values[index] = argument
    keyword_positions = {"a": 0, "a_min": 1, "a_max": 2}
    for keyword_argument in node.keywords:
        if keyword_argument.arg not in keyword_positions:
            raise SymbolicEvidenceError(
                f"clip 不支持关键字参数: {keyword_argument.arg!r}"
            )
        position = keyword_positions[keyword_argument.arg]
        if values[position] is not missing:
            raise SymbolicEvidenceError(
                f"clip 参数重复赋值: {keyword_argument.arg}"
            )
        values[position] = keyword_argument.value
    if values[0] is missing:
        raise SymbolicEvidenceError("clip 缺少待裁剪表达式")
    if values[1] is missing or values[2] is missing:
        raise SymbolicEvidenceError("clip 必须显式提供 a_min 与 a_max")

    value_node = values[0]
    assert isinstance(value_node, ast.AST)
    result = _convert_node(value_node, ctx)

    lower_node = values[1]
    upper_node = values[2]
    lower_is_none = isinstance(lower_node, ast.Constant) and lower_node.value is None
    upper_is_none = isinstance(upper_node, ast.Constant) and upper_node.value is None
    if lower_is_none and upper_is_none:
        raise SymbolicEvidenceError("clip 的 a_min 与 a_max 不能同时为 None")
    if not lower_is_none:
        assert isinstance(lower_node, ast.AST)
        result = sp.Max(
            _convert_node(lower_node, ctx),
            result,
            evaluate=ctx.evaluate_expressions,
        )
    if not upper_is_none:
        assert isinstance(upper_node, ast.AST)
        result = sp.Min(
            _convert_node(upper_node, ctx),
            result,
            evaluate=ctx.evaluate_expressions,
        )
    return result


def _convert_node(node: ast.AST, ctx: _BuildContext) -> sp.Basic:
    if isinstance(node, ast.Constant):
        return _convert_constant(node, ctx)
    if isinstance(node, ast.Name):
        if node.id in CONSTANT_NAMES:
            return CONSTANT_NAMES[node.id]
        return _resolve_identifier(node.id, ctx)
    if isinstance(node, ast.Attribute):
        return _resolve_constant(node)
    if isinstance(node, ast.BinOp):
        left = _convert_node(node.left, ctx)
        right = _convert_node(node.right, ctx)
        if isinstance(node.op, ast.Add):
            return sp.Add(left, right, evaluate=ctx.evaluate_expressions)
        if isinstance(node.op, ast.Sub):
            return sp.Add(
                left,
                sp.Mul(sp.Integer(-1), right, evaluate=ctx.evaluate_expressions),
                evaluate=ctx.evaluate_expressions,
            )
        if isinstance(node.op, ast.Mult):
            return sp.Mul(left, right, evaluate=ctx.evaluate_expressions)
        if isinstance(node.op, ast.Div):
            return sp.Mul(
                left,
                sp.Pow(
                    right,
                    sp.Integer(-1),
                    evaluate=ctx.evaluate_expressions,
                ),
                evaluate=ctx.evaluate_expressions,
            )
        if isinstance(node.op, ast.Pow):
            return sp.Pow(left, right, evaluate=ctx.evaluate_expressions)
        raise SymbolicEvidenceError(f"不支持的二元运算: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _convert_node(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return sp.Mul(
                sp.Integer(-1),
                operand,
                evaluate=ctx.evaluate_expressions,
            )
        if isinstance(node.op, ast.UAdd):
            return operand
        raise SymbolicEvidenceError(f"不支持的一元运算: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        is_piecewise = (
            isinstance(node.func, ast.Name)
            and node.func.id in {"Piecewise", "piecewise"}
        )
        normalized = _record_function(ctx, _resolve_call_name(node.func))
        if is_piecewise:
            if node.keywords:
                raise SymbolicEvidenceError("Piecewise 不支持关键字参数")
            return _convert_piecewise_call(node, ctx)
        if normalized == "clip":
            return _convert_clip_call(node, ctx)
        if normalized == "where":
            if len(node.args) != 3:
                raise SymbolicEvidenceError("where 需要 3 个参数")
            condition = _compare_relation(node.args[0], ctx)
            args = [condition] + [_convert_node(arg, ctx) for arg in node.args[1:]]
        elif normalized == "compare":
            args = [_convert_node(arg, ctx) for arg in node.args]
        else:
            args = [_convert_node(arg, ctx) for arg in node.args]
        if node.keywords:
            raise SymbolicEvidenceError("不支持关键字参数")
        return _call_handler(normalized, ctx)(args)
    if isinstance(node, ast.Compare):
        relation = _convert_compare(node, ctx)
        return sp.Piecewise((sp.Integer(1), relation), (sp.Integer(0), True))
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise SymbolicEvidenceError("只允许命名变量的静态整数下标访问")
        index = _literal_subscript_index(node.slice)
        if node.value.id == "params":
            return ctx.symbol(_parameter_symbol_name(index))
        variable_name = node.value.id
        if (
            ctx.allowed_variables is not None
            and variable_name not in ctx.allowed_variables
            and variable_name not in ctx.inferred_variables
        ):
            raise SymbolicEvidenceError(f"检测到未授权下标变量: {variable_name}")
        ctx.source_functions.add("index")
        index_function = sp.Function("index")
        return index_function(ctx.symbol(variable_name), sp.Integer(index))

    banned = (
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.IfExp,
        ast.BoolOp,
        ast.NamedExpr,
    )
    if isinstance(node, banned):
        raise SymbolicEvidenceError(f"不支持的语法节点: {type(node).__name__}")
    raise SymbolicEvidenceError(f"不支持的 AST 节点: {type(node).__name__}")


def _normalize_sympy_name(expr: sp.Basic) -> str:
    name = type(expr).__name__
    aliases = {
        "Add": "add",
        "Mul": "mul",
        "Pow": "pow",
        "StrictGreaterThan": "gt",
        "GreaterThan": "ge",
        "StrictLessThan": "lt",
        "LessThan": "le",
        "Equality": "eq",
        "Unequality": "ne",
        "Piecewise": "piecewise",
        "Abs": "abs",
        "Max": "maximum",
        "Min": "minimum",
        "Exp1": "E",
        "Pi": "pi",
    }
    return aliases.get(name, _normalize_name(name))


def _canonical_tree(expr: sp.Basic) -> dict[str, object]:
    if not isinstance(expr, sp.Basic):
        raise SymbolicEvidenceError("只能为 SymPy 表达式生成 canonical tree")
    node_type = _normalize_sympy_name(expr)
    if not expr.args:
        return {"type": node_type, "value": sp.srepr(expr)}
    return {"type": node_type, "args": [_canonical_tree(arg) for arg in expr.args]}


def _count_nodes(expr: sp.Basic) -> int:
    if not isinstance(expr, sp.Basic):
        return 0
    return 1 + sum(_count_nodes(arg) for arg in expr.args if isinstance(arg, sp.Basic))


def _collect_operator_set(expr: sp.Basic) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(node: sp.Basic) -> None:
        if not isinstance(node, sp.Basic):
            return
        if node.args:
            names.add(_normalize_sympy_name(node))
        for child in node.args:
            if isinstance(child, sp.Basic):
                visit(child)

    visit(expr)
    return tuple(sorted(names))


def _abstract_constants(tree: dict[str, object]) -> dict[str, object]:
    node_type = str(tree["type"])
    args = tree.get("args")
    if isinstance(args, list):
        return {
            "type": node_type,
            "args": [_abstract_constants(child) for child in args],
        }
    if node_type in {"Symbol", "BooleanTrue", "BooleanFalse"}:
        return {"type": node_type, "value": tree.get("value")}
    return {"type": "Constant"}


def _build_sympy_expression(
    source: str,
    *,
    allowed_variables: Collection[str] | None = None,
    allowed_functions: Collection[str] | None = None,
    exact_numeric_literals: bool = False,
    evaluate_expressions: bool | None = None,
) -> tuple[sp.Basic, _BuildContext, str]:
    normalized_allowed_variables = (
        set(allowed_variables) if allowed_variables is not None else None
    )
    normalized_allowed_functions = (
        {_normalize_name(name) for name in allowed_functions}
        if allowed_functions is not None
        else None
    )
    ctx = _BuildContext(
        allowed_variables=normalized_allowed_variables,
        allowed_functions=normalized_allowed_functions,
        inferred_variables=set(),
        source_text=source,
        exact_numeric_literals=exact_numeric_literals,
        evaluate_expressions=(
            len(source) <= UNEVALUATED_AST_NODE_THRESHOLD * 4
            if evaluate_expressions is None
            else evaluate_expressions
        ),
    )
    try:
        root, source_kind, inferred_variables = _extract_expression_ast(source)
        ast_node_count = sum(1 for _ in ast.walk(root))
        if evaluate_expressions is None:
            ctx.evaluate_expressions = (
                ast_node_count <= UNEVALUATED_AST_NODE_THRESHOLD
            )
        ctx.inferred_variables = inferred_variables
        if (
            source_kind == "function_def"
            and normalized_allowed_variables is not None
            and not inferred_variables.issubset(normalized_allowed_variables)
        ):
            illegal = tuple(sorted(inferred_variables - normalized_allowed_variables))
            raise SymbolicEvidenceError(f"函数定义包含未授权变量: {illegal}")
        expr = _convert_node(root, ctx)
    except SyntaxError:
        if not _looks_like_prefix_expression(source):
            raise
        source_kind = "prefix_expression"
        expr = _build_prefix_expression(source, ctx)
    except SymbolicEvidenceError:
        if _looks_like_prefix_expression(source):
            source_kind = "prefix_expression"
            expr = _build_prefix_expression(source, ctx)
        else:
            raise
    return expr, ctx, source_kind


def build_symbolic_artifact(
    source: str,
    *,
    allowed_variables: Collection[str] | None = None,
    allowed_functions: Collection[str] | None = None,
    evaluate_expressions: bool | None = None,
) -> dict[str, object]:
    """把不可信表达式安全地转换为稳定的 SymPy 证据对象。"""

    if not isinstance(source, str) or not source.strip():
        raise SymbolicEvidenceError("source 必须是非空字符串")
    expr, ctx, source_kind = _build_sympy_expression(
        source,
        allowed_variables=allowed_variables,
        allowed_functions=allowed_functions,
        exact_numeric_literals=False,
        evaluate_expressions=evaluate_expressions,
    )

    canonical_expression = sp.sstr(expr, order="lex")
    canonical_tree = _canonical_tree(expr)
    constants_abstracted_canonical_tree = _abstract_constants(canonical_tree)
    variables = tuple(sorted(str(symbol) for symbol in expr.free_symbols))
    function_set = tuple(sorted(ctx.source_functions))
    operator_set = _collect_operator_set(expr)
    artifact_core = {
        "source_kind": source_kind,
        "canonical_expression": canonical_expression,
        "canonical_tree": canonical_tree,
        "node_count": _count_nodes(expr),
        "variables": variables,
        "function_set": function_set,
        "operator_set": operator_set,
    }
    construction_mode = (
        "evaluated" if ctx.evaluate_expressions else "unevaluated_large_ast"
    )
    if not ctx.evaluate_expressions:
        artifact_core["construction_mode"] = construction_mode
    artifact = {
        "source_kind": source_kind,
        "source_text": source,
        "canonical_expression": canonical_expression,
        "canonical_tree": canonical_tree,
        "node_count": artifact_core["node_count"],
        "variables": variables,
        "function_set": function_set,
        "operator_set": operator_set,
        "construction_mode": construction_mode,
        "artifact_sha256": _sha256_text(_canonical_json(artifact_core)),
        "sympy_expression": expr,
        "constants_abstracted_canonical_tree": constants_abstracted_canonical_tree,
        "constants_abstracted_tree_fingerprint": _sha256_text(
            _canonical_json(constants_abstracted_canonical_tree)
        ),
    }
    return artifact


_FrozenTree = tuple[str, str | None, tuple["_FrozenTree", ...]]


def _freeze_tree(tree: Mapping[str, object]) -> _FrozenTree:
    node_type = str(tree["type"])
    value = str(tree["value"]) if "value" in tree else None
    raw_args = tree.get("args")
    if raw_args is None:
        return (node_type, value, ())
    if not isinstance(raw_args, list):
        raise SymbolicEvidenceError("canonical_tree.args 必须是数组")
    children: list[_FrozenTree] = []
    for child in raw_args:
        if not isinstance(child, Mapping):
            raise SymbolicEvidenceError("canonical_tree.args 只能包含对象节点")
        children.append(_freeze_tree(child))
    return (node_type, value, tuple(children))


def _node_label(node: _FrozenTree) -> tuple[str, str | None]:
    return (node[0], node[1])


@lru_cache(maxsize=None)
def _subtree_size(node: _FrozenTree) -> int:
    return 1 + sum(_subtree_size(child) for child in node[2])


@dataclass(frozen=True)
class _PostorderTree:
    labels: tuple[tuple[str, str | None] | None, ...]
    leftmost_leaf: tuple[int, ...]
    keyroots: tuple[int, ...]


def _postorder_tree(root: _FrozenTree) -> _PostorderTree:
    labels: list[tuple[str, str | None] | None] = [None]
    leftmost_leaf: list[int] = [0]

    def visit(node: _FrozenTree) -> int:
        child_indices = [visit(child) for child in node[2]]
        index = len(labels)
        labels.append(_node_label(node))
        leftmost_leaf.append(
            leftmost_leaf[child_indices[0]] if child_indices else index
        )
        return index

    visit(root)
    # Zhang-Shasha keyroots：每个不同 leftmost-leaf 值取最大的后序索引。
    last_for_leaf: dict[int, int] = {}
    for index in range(1, len(labels)):
        last_for_leaf[leftmost_leaf[index]] = index
    return _PostorderTree(
        labels=tuple(labels),
        leftmost_leaf=tuple(leftmost_leaf),
        keyroots=tuple(sorted(last_for_leaf.values())),
    )


def _ordered_tree_edit_distance(lhs_root: _FrozenTree, rhs_root: _FrozenTree) -> int:
    """Zhang-Shasha 有序树编辑距离，三种基本操作的成本均为 1。"""

    lhs = _postorder_tree(lhs_root)
    rhs = _postorder_tree(rhs_root)
    lhs_size = len(lhs.labels) - 1
    rhs_size = len(rhs.labels) - 1
    tree_distance = [[0] * (rhs_size + 1) for _ in range(lhs_size + 1)]

    for lhs_root_index in lhs.keyroots:
        lhs_start = lhs.leftmost_leaf[lhs_root_index]
        for rhs_root_index in rhs.keyroots:
            rhs_start = rhs.leftmost_leaf[rhs_root_index]
            forest_rows = lhs_root_index - lhs_start + 2
            forest_cols = rhs_root_index - rhs_start + 2
            forest_distance = [[0] * forest_cols for _ in range(forest_rows)]
            for lhs_index in range(lhs_start, lhs_root_index + 1):
                row = lhs_index - lhs_start + 1
                forest_distance[row][0] = forest_distance[row - 1][0] + 1
            for rhs_index in range(rhs_start, rhs_root_index + 1):
                col = rhs_index - rhs_start + 1
                forest_distance[0][col] = forest_distance[0][col - 1] + 1

            for lhs_index in range(lhs_start, lhs_root_index + 1):
                row = lhs_index - lhs_start + 1
                for rhs_index in range(rhs_start, rhs_root_index + 1):
                    col = rhs_index - rhs_start + 1
                    delete_cost = forest_distance[row - 1][col] + 1
                    insert_cost = forest_distance[row][col - 1] + 1
                    if (
                        lhs.leftmost_leaf[lhs_index] == lhs_start
                        and rhs.leftmost_leaf[rhs_index] == rhs_start
                    ):
                        rename_cost = 0 if lhs.labels[lhs_index] == rhs.labels[rhs_index] else 1
                        replace_cost = forest_distance[row - 1][col - 1] + rename_cost
                        value = min(delete_cost, insert_cost, replace_cost)
                        forest_distance[row][col] = value
                        tree_distance[lhs_index][rhs_index] = value
                    else:
                        prefix_row = lhs.leftmost_leaf[lhs_index] - lhs_start
                        prefix_col = rhs.leftmost_leaf[rhs_index] - rhs_start
                        subtree_cost = (
                            forest_distance[prefix_row][prefix_col]
                            + tree_distance[lhs_index][rhs_index]
                        )
                        forest_distance[row][col] = min(
                            delete_cost,
                            insert_cost,
                            subtree_cost,
                        )
    return tree_distance[lhs_size][rhs_size]


def normalized_tree_edit_distance(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
) -> float:
    lhs_tree = lhs["canonical_tree"]
    rhs_tree = rhs["canonical_tree"]
    if not isinstance(lhs_tree, Mapping) or not isinstance(rhs_tree, Mapping):
        raise SymbolicEvidenceError("artifact.canonical_tree 缺失或无效")
    lhs_node = _freeze_tree(lhs_tree)
    rhs_node = _freeze_tree(rhs_tree)
    raw_distance = _ordered_tree_edit_distance(lhs_node, rhs_node)
    denominator = max(_subtree_size(lhs_node) + _subtree_size(rhs_node), 1)
    return float(raw_distance) / float(denominator)


def tree_similarity(lhs: Mapping[str, object], rhs: Mapping[str, object]) -> float:
    return 1.0 - normalized_tree_edit_distance(lhs, rhs)


def variable_f1(lhs: Mapping[str, object], rhs: Mapping[str, object]) -> float:
    lhs_set = set(lhs["variables"])
    rhs_set = set(rhs["variables"])
    if not lhs_set and not rhs_set:
        return 1.0
    return 2.0 * len(lhs_set & rhs_set) / (len(lhs_set) + len(rhs_set))


def operator_f1(lhs: Mapping[str, object], rhs: Mapping[str, object]) -> float:
    lhs_set = set(lhs["operator_set"])
    rhs_set = set(rhs["operator_set"])
    if not lhs_set and not rhs_set:
        return 1.0
    return 2.0 * len(lhs_set & rhs_set) / (len(lhs_set) + len(rhs_set))


def _format_real(value: float) -> str:
    if value == 0:
        return "0"
    text = format(float(value), ".17g")
    return "0" if text == "-0" else text


def _sympy_real_to_float(value: sp.Basic, *, digits: int = 50) -> float | None:
    try:
        numeric = complex(sp.N(value, digits))
    except Exception:
        return None
    if abs(numeric.imag) > 1e-12:
        return None
    real = float(numeric.real)
    if not math.isfinite(real):
        return None
    return real


def _probe_points(
    variable_names: tuple[str, ...],
    seed: int,
    target_count: int,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    points: list[dict[str, float]] = []
    attempts = 0
    while len(points) < target_count and attempts < max(target_count * 40, 40):
        attempts += 1
        values: dict[str, float] = {}
        for name in variable_names:
            value = rng.uniform(-2.5, 2.5)
            if abs(value) < 0.15:
                value = 0.15 if value >= 0 else -0.15
            values[name] = value
        points.append(values)
    return points


def _probe_substitutions(
    values: Mapping[str, float],
    *,
    exact_numeric_literals: bool,
) -> dict[sp.Symbol, sp.Basic]:
    substitutions: dict[sp.Symbol, sp.Basic] = {}
    for name, value in values.items():
        if exact_numeric_literals:
            substitutions[sp.Symbol(name)] = sp.Rational(str(value))
        else:
            substitutions[sp.Symbol(name)] = sp.Float(repr(value))
    return substitutions


def _evaluate_real(
    expr: sp.Basic,
    values: Mapping[str, float],
    *,
    exact_numeric_literals: bool = False,
    digits: int = 50,
) -> float | None:
    substitutions = _probe_substitutions(
        values,
        exact_numeric_literals=exact_numeric_literals,
    )
    try:
        substituted = expr.subs(substitutions)
    except Exception:
        return None
    if substituted.free_symbols:
        return None
    return _sympy_real_to_float(substituted, digits=digits)


def _format_probe_values(values: Mapping[str, float]) -> dict[str, str]:
    return {name: _format_real(values[name]) for name in sorted(values)}


def _normalize_probe_value(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise SymbolicEvidenceError(f"{context} 不能是布尔值")
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError as exc:
            raise SymbolicEvidenceError(f"{context} 不是合法数值: {value!r}") from exc
    else:
        raise SymbolicEvidenceError(f"{context} 不是合法数值: {value!r}")
    if not math.isfinite(numeric):
        raise SymbolicEvidenceError(f"{context} 必须是有限实数: {value!r}")
    return numeric


def _normalize_external_probe_points(
    probe_points: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for point_index, point in enumerate(probe_points):
        if not isinstance(point, Mapping):
            raise SymbolicEvidenceError(f"probe_points[{point_index}] 必须是对象")
        raw_values = point.get("values")
        if raw_values is None:
            raw_values = {
                key: value
                for key, value in point.items()
                if key not in {"split", "row_index"}
            }
        if not isinstance(raw_values, Mapping):
            raise SymbolicEvidenceError(f"probe_points[{point_index}].values 必须是对象")
        values: dict[str, float] = {}
        for name, value in raw_values.items():
            if not isinstance(name, str) or not name:
                raise SymbolicEvidenceError(
                    f"probe_points[{point_index}] 含非法变量名: {name!r}"
                )
            values[name] = _normalize_probe_value(
                value,
                context=f"probe_points[{point_index}].values[{name!r}]",
            )
        normalized_point: dict[str, object] = {"values": values}
        split = point.get("split")
        if split is not None:
            if not isinstance(split, str) or not split:
                raise SymbolicEvidenceError(f"probe_points[{point_index}].split 必须是非空字符串")
            normalized_point["split"] = split
        row_index = point.get("row_index")
        if row_index is not None:
            if not isinstance(row_index, int) or row_index < 0:
                raise SymbolicEvidenceError(
                    f"probe_points[{point_index}].row_index 必须是非负整数"
                )
            normalized_point["row_index"] = row_index
        normalized.append(normalized_point)
    return normalized


def _probe_point_signature(point: Mapping[str, object]) -> dict[str, object]:
    signature: dict[str, object] = {
        "values": _format_probe_values(point["values"]),  # type: ignore[arg-type]
    }
    split = point.get("split")
    if isinstance(split, str):
        signature["split"] = split
    row_index = point.get("row_index")
    if isinstance(row_index, int):
        signature["row_index"] = row_index
    return signature


def _probe_sample_sha256(points: Sequence[Mapping[str, object]]) -> str:
    return _sha256_text(_canonical_json([_probe_point_signature(point) for point in points]))


def _probe_record(
    *,
    values: Mapping[str, float],
    original_value: float,
    simplified_value: float,
    abs_error: float,
    rel_error: float,
    tolerance: float,
    split: str | None = None,
    row_index: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "values": {name: _format_real(value) for name, value in values.items()},
        "original": _format_real(original_value),
        "simplified": _format_real(simplified_value),
        "abs_error": _format_real(abs_error),
        "rel_error": _format_real(rel_error),
        "tolerance": _format_real(tolerance),
    }
    if split is not None:
        record["split"] = split
    if row_index is not None:
        record["row_index"] = row_index
    return record


def _skipped_probe_record(
    *,
    point: Mapping[str, object],
    reason: str,
    missing_variables: Sequence[str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "reason": reason,
        "values": _format_probe_values(point["values"]),  # type: ignore[arg-type]
    }
    split = point.get("split")
    if isinstance(split, str):
        record["split"] = split
    row_index = point.get("row_index")
    if isinstance(row_index, int):
        record["row_index"] = row_index
    if missing_variables:
        record["missing_variables"] = list(missing_variables)
    return record


def _max_metric(records: list[dict[str, object]], key: str) -> str:
    return _format_real(max((float(record[key]) for record in records), default=0.0))


def _proof_guard_triggered(
    original_artifact: Mapping[str, object],
    simplified_artifact: Mapping[str, object],
) -> bool:
    return max(
        int(original_artifact["node_count"]),
        int(simplified_artifact["node_count"]),
    ) > SYMBOLIC_PROOF_NODE_LIMIT


def _safe_equals_zero(expr: sp.Basic) -> bool | None:
    try:
        return expr.equals(0)
    except (RecursionError, RuntimeError):
        return None
    except Exception:
        return None


class _SymbolicProofTimedOut(RuntimeError):
    pass


class _NumericProbeTimedOut(RuntimeError):
    pass


def _bounded_symbolic_difference(lhs: sp.Basic, rhs: sp.Basic) -> sp.Basic | None:
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return None

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise _SymbolicProofTimedOut

    previous_handler = signal.getsignal(signal.SIGALRM)
    started_at = time.monotonic()
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, SYMBOLIC_PROOF_TIMEOUT_SECONDS)
    try:
        return sp.simplify(sp.together(lhs - rhs))
    except _SymbolicProofTimedOut:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(0.0, previous_timer[0] - (time.monotonic() - started_at))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _bounded_evaluate_real(
    expr: sp.Basic,
    values: Mapping[str, float],
    *,
    exact_numeric_literals: bool = False,
    digits: int = 50,
    timeout_seconds: float | None = None,
) -> tuple[float | None, bool]:
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        return None, True

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise _NumericProbeTimedOut

    previous_handler = signal.getsignal(signal.SIGALRM)
    started_at = time.monotonic()
    signal.signal(signal.SIGALRM, raise_timeout)
    effective_timeout = (
        NUMERIC_PROBE_TIMEOUT_SECONDS
        if timeout_seconds is None
        else max(0.0, min(NUMERIC_PROBE_TIMEOUT_SECONDS, timeout_seconds))
    )
    if effective_timeout <= 0:
        signal.signal(signal.SIGALRM, previous_handler)
        return None, True
    previous_timer = signal.setitimer(signal.ITIMER_REAL, effective_timeout)
    try:
        return (
            _evaluate_real(
                expr,
                values,
                exact_numeric_literals=exact_numeric_literals,
                digits=digits,
            ),
            False,
        )
    except _NumericProbeTimedOut:
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(0.0, previous_timer[0] - (time.monotonic() - started_at))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _build_exact_decimal_pair(
    original_artifact: Mapping[str, object],
    simplified_artifact: Mapping[str, object],
) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
    original_source = original_artifact.get("source_text")
    simplified_source = simplified_artifact.get("source_text")
    if not isinstance(original_source, str) or not isinstance(simplified_source, str):
        return None
    try:
        exact_original, _, _ = _build_sympy_expression(
            original_source,
            exact_numeric_literals=True,
        )
        exact_simplified, _, _ = _build_sympy_expression(
            simplified_source,
            exact_numeric_literals=True,
        )
    except (SymbolicEvidenceError, SyntaxError, ValueError):
        return None
    return (
        exact_original,
        exact_simplified,
        sp.simplify(sp.together(exact_original - exact_simplified)),
    )


def _is_nonzero_rational_function(expr: sp.Basic) -> bool:
    if expr == 0:
        return False
    free_symbols = tuple(sorted(expr.free_symbols, key=lambda symbol: str(symbol)))
    if not free_symbols:
        return bool(getattr(expr, "is_number", False)) and bool(getattr(expr, "is_finite", False))
    try:
        if not expr.is_rational_function(*free_symbols):
            return False
        numerator, _ = sp.together(expr).as_numer_denom()
        return sp.simplify(numerator) != 0
    except Exception:
        return False


def _rebuild_probe_record(
    *,
    expr_lhs: sp.Basic,
    expr_rhs: sp.Basic,
    values: Mapping[str, float],
    point: Mapping[str, object],
    abs_tolerance: float,
    rel_tolerance: float,
    exact_numeric_literals: bool,
    digits: int,
) -> dict[str, object] | None:
    lhs_value, lhs_timed_out = _bounded_evaluate_real(
        expr_lhs,
        values,
        exact_numeric_literals=exact_numeric_literals,
        digits=digits,
    )
    rhs_value, rhs_timed_out = _bounded_evaluate_real(
        expr_rhs,
        values,
        exact_numeric_literals=exact_numeric_literals,
        digits=digits,
    )
    if lhs_timed_out or rhs_timed_out or lhs_value is None or rhs_value is None:
        return None
    abs_error = abs(lhs_value - rhs_value)
    scale = max(abs(lhs_value), abs(rhs_value))
    rel_error = 0.0 if scale == 0.0 else abs_error / scale
    tolerance = abs_tolerance + rel_tolerance * scale
    return _probe_record(
        values=values,
        original_value=lhs_value,
        simplified_value=rhs_value,
        abs_error=abs_error,
        rel_error=rel_error,
        tolerance=tolerance,
        split=point.get("split") if isinstance(point.get("split"), str) else None,
        row_index=point.get("row_index") if isinstance(point.get("row_index"), int) else None,
    )


def _equivalence_core(
    original_artifact: Mapping[str, object],
    simplified_artifact: Mapping[str, object],
    *,
    seed: int,
    probe_target: int,
    abs_tolerance: float,
    rel_tolerance: float,
    probe_points: Sequence[Mapping[str, object]] | None = None,
    probe_source: str | None = None,
    probe_sample_sha256: str | None = None,
    enable_exact_decimal_rebuild: bool = True,
) -> dict[str, object]:
    original_expr = original_artifact["sympy_expression"]
    simplified_expr = simplified_artifact["sympy_expression"]
    if not isinstance(original_expr, sp.Basic) or not isinstance(simplified_expr, sp.Basic):
        raise SymbolicEvidenceError("内部错误：SymPy 表达式构造失败")

    assumptions: list[str] = []
    has_opaque_function = bool(
        (set(original_artifact["function_set"]) | set(simplified_artifact["function_set"]))
        & OPAQUE_FUNCTIONS
    )
    complexity_guard = _proof_guard_triggered(original_artifact, simplified_artifact)
    proof_basis = "none"
    exact_decimal_pair: tuple[sp.Basic, sp.Basic, sp.Basic] | None = None

    if original_artifact["artifact_sha256"] == simplified_artifact["artifact_sha256"]:
        symbolic_decision = "equivalent"
        proof_basis = "artifact_identity"
    elif has_opaque_function:
        symbolic_decision = "undetermined"
        assumptions.append("存在不透明函数，符号证明被禁用，只做数值探针排错。")
    elif complexity_guard:
        symbolic_decision = "undetermined"
        assumptions.append(
            f"表达式复杂度超过符号证明阈值 {SYMBOLIC_PROOF_NODE_LIMIT}，跳过高风险 simplify。"
        )
    else:
        difference = _bounded_symbolic_difference(original_expr, simplified_expr)
        if difference is None:
            symbolic_decision = "undetermined"
            assumptions.append(
                f"符号证明超时或当前执行线程无法安全设置超时（上限 {SYMBOLIC_PROOF_TIMEOUT_SECONDS:g} 秒），"
                "保留数值探针交由后续裁决。"
            )
        else:
            difference_is_zero = difference == 0
            equals_result = (
                _safe_equals_zero(difference)
                if enable_exact_decimal_rebuild
                else None
            )
            difference_equals_zero = equals_result is True
            if difference_is_zero or difference_equals_zero:
                symbolic_decision = "equivalent"
                proof_basis = "symbolic_difference_zero"
            else:
                if enable_exact_decimal_rebuild:
                    exact_decimal_pair = _build_exact_decimal_pair(
                        original_artifact,
                        simplified_artifact,
                    )
                if exact_decimal_pair is not None:
                    _, _, exact_difference = exact_decimal_pair
                    exact_difference_is_zero = exact_difference == 0
                    exact_difference_equals_zero = _safe_equals_zero(exact_difference) is True
                    if exact_difference_is_zero or exact_difference_equals_zero:
                        symbolic_decision = "equivalent"
                        proof_basis = "symbolic_difference_zero"
                    elif (
                        bool(getattr(exact_difference, "is_number", False))
                        and bool(getattr(exact_difference, "is_finite", False))
                        and exact_difference != 0
                    ):
                        symbolic_decision = "not_equivalent"
                        proof_basis = "symbolic_nonzero_constant_difference"
                    elif _is_nonzero_rational_function(exact_difference):
                        symbolic_decision = "not_equivalent"
                        proof_basis = "symbolic_nonzero_exact_difference"
                    elif (
                        bool(getattr(difference, "is_number", False))
                        and bool(getattr(difference, "is_finite", False))
                        and difference != 0
                    ):
                        symbolic_decision = "not_equivalent"
                        proof_basis = "symbolic_nonzero_constant_difference"
                    else:
                        symbolic_decision = "undetermined"
                elif (
                    bool(getattr(difference, "is_number", False))
                    and bool(getattr(difference, "is_finite", False))
                    and difference != 0
                ):
                    symbolic_decision = "not_equivalent"
                    proof_basis = "symbolic_nonzero_constant_difference"
                else:
                    symbolic_decision = "undetermined"

    probe_records: list[dict[str, object]] = []
    skipped_probe_records: list[dict[str, object]] = []
    variable_names = tuple(
        sorted(set(original_artifact["variables"]) | set(simplified_artifact["variables"]))
    )
    if has_opaque_function and proof_basis == "artifact_identity":
        variable_names = ()
    effective_probe_source = probe_source or ("external" if probe_points is not None else "random")
    if probe_points is None:
        candidate_points = [
            {"values": values}
            for values in _probe_points(variable_names, seed, probe_target)
        ]
        normalized_probe_points_sha256 = _probe_sample_sha256(candidate_points)
        effective_probe_sample_sha256 = normalized_probe_points_sha256
    else:
        candidate_points = _normalize_external_probe_points(probe_points)
        normalized_probe_points_sha256 = _probe_sample_sha256(candidate_points)
        effective_probe_sample_sha256 = (
            probe_sample_sha256 or normalized_probe_points_sha256
        )
    if has_opaque_function and proof_basis != "artifact_identity" and probe_points is None:
        assumptions.append("不透明函数无法稳定数值化，跳过随机探针。")
    else:
        if has_opaque_function and proof_basis != "artifact_identity" and probe_points is not None:
            assumptions.append("存在不透明函数，外部探针仅用于有限实数对比。")
        probe_deadline = time.monotonic() + NUMERIC_PROBE_TOTAL_TIMEOUT_SECONDS
        for point in candidate_points:
            values = point["values"]
            assert isinstance(values, Mapping)
            missing_variables = tuple(sorted(set(variable_names) - set(values)))
            if missing_variables:
                skipped_probe_records.append(
                    _skipped_probe_record(
                        point=point,
                        reason="missing_variables",
                        missing_variables=missing_variables,
                    )
                )
                continue
            remaining_probe_time = probe_deadline - time.monotonic()
            original_value, original_timed_out = _bounded_evaluate_real(
                original_expr,
                values,
                timeout_seconds=remaining_probe_time,
            )
            if original_timed_out:
                skipped_probe_records.append(
                    _skipped_probe_record(point=point, reason="numeric_probe_timeout")
                )
                assumptions.append(
                    "数值探针超时（单表达式上限 "
                    f"{NUMERIC_PROBE_TIMEOUT_SECONDS:g} 秒、整对上限 "
                    f"{NUMERIC_PROBE_TOTAL_TIMEOUT_SECONDS:g} 秒），"
                    "停止当前表达式对的剩余探针并交由后续裁决。"
                )
                break
            remaining_probe_time = probe_deadline - time.monotonic()
            simplified_value, simplified_timed_out = _bounded_evaluate_real(
                simplified_expr,
                values,
                timeout_seconds=remaining_probe_time,
            )
            if simplified_timed_out:
                skipped_probe_records.append(
                    _skipped_probe_record(point=point, reason="numeric_probe_timeout")
                )
                assumptions.append(
                    "数值探针超时（单表达式上限 "
                    f"{NUMERIC_PROBE_TIMEOUT_SECONDS:g} 秒、整对上限 "
                    f"{NUMERIC_PROBE_TOTAL_TIMEOUT_SECONDS:g} 秒），"
                    "停止当前表达式对的剩余探针并交由后续裁决。"
                )
                break
            if original_value is None or simplified_value is None:
                if original_value is None and simplified_value is None:
                    reason = "both_nonfinite"
                elif original_value is None:
                    reason = "original_nonfinite"
                else:
                    reason = "simplified_nonfinite"
                skipped_probe_records.append(
                    _skipped_probe_record(
                        point=point,
                        reason=reason,
                    )
                )
                continue
            abs_error = abs(original_value - simplified_value)
            scale = max(abs(original_value), abs(simplified_value))
            rel_error = 0.0 if scale == 0.0 else abs_error / scale
            tolerance = abs_tolerance + rel_tolerance * scale
            record = _probe_record(
                values=values,
                original_value=original_value,
                simplified_value=simplified_value,
                abs_error=abs_error,
                rel_error=rel_error,
                tolerance=tolerance,
                split=point.get("split") if isinstance(point.get("split"), str) else None,
                row_index=point.get("row_index") if isinstance(point.get("row_index"), int) else None,
            )
            probe_records.append(record)
            if abs_error > tolerance:
                if symbolic_decision == "equivalent" and proof_basis != "artifact_identity":
                    if exact_decimal_pair is not None:
                        exact_original_expr, exact_simplified_expr, _ = exact_decimal_pair
                        rebuilt_record = _rebuild_probe_record(
                            expr_lhs=exact_original_expr,
                            expr_rhs=exact_simplified_expr,
                            values=values,
                            point=point,
                            abs_tolerance=abs_tolerance,
                            rel_tolerance=rel_tolerance,
                            exact_numeric_literals=True,
                            digits=100,
                        )
                    else:
                        rebuilt_record = _rebuild_probe_record(
                            expr_lhs=original_expr,
                            expr_rhs=simplified_expr,
                            values=values,
                            point=point,
                            abs_tolerance=abs_tolerance,
                            rel_tolerance=rel_tolerance,
                            exact_numeric_literals=True,
                            digits=100,
                        )
                    if rebuilt_record is not None and (
                        float(rebuilt_record["abs_error"]) <= float(rebuilt_record["tolerance"])
                    ):
                        probe_records[-1] = rebuilt_record
                        if "双精度探针命中微小残差，已用十进制有理替换做高精度复核。" not in assumptions:
                            assumptions.append("双精度探针命中微小残差，已用十进制有理替换做高精度复核。")
                        continue
                return {
                    "symbolic_decision": "not_equivalent",
                    "probe_records": probe_records,
                    "skipped_probe_records": skipped_probe_records,
                    "counterexample": record,
                    "assumptions": assumptions,
                    "proof_basis": proof_basis,
                    "probe_source": effective_probe_source,
                    "probe_sample_sha256": effective_probe_sample_sha256,
                    "normalized_probe_points_sha256": normalized_probe_points_sha256,
                }
    if symbolic_decision == "equivalent" and proof_basis != "artifact_identity" and not probe_records:
        if probe_points is None:
            assumptions.append("符号差分为零，但没有收集到有限随机探针。")
        else:
            assumptions.append("符号差分为零，但外部探针未产生有限实数对比点。")
    return {
        "symbolic_decision": symbolic_decision,
        "probe_records": probe_records,
        "skipped_probe_records": skipped_probe_records,
        "counterexample": None,
        "assumptions": assumptions,
        "proof_basis": proof_basis,
        "probe_source": effective_probe_source,
        "probe_sample_sha256": effective_probe_sample_sha256,
        "normalized_probe_points_sha256": normalized_probe_points_sha256,
    }


def build_pair_evidence(
    lhs: str,
    rhs: str,
    allowed_variables: Collection[str] | None = None,
    allowed_functions: Collection[str] | None = None,
    *,
    seed: int,
    probe_target: int = 12,
    tolerance: float = 1e-9,
    rel_tolerance: float = 1e-9,
    probe_points: Sequence[Mapping[str, object]] | None = None,
    probe_source: str | None = None,
    probe_sample_sha256: str | None = None,
    include_tree_distance: bool = True,
) -> dict[str, object]:
    lhs_artifact = build_symbolic_artifact(
        lhs,
        allowed_variables=allowed_variables,
        allowed_functions=allowed_functions,
    )
    rhs_artifact = build_symbolic_artifact(
        rhs,
        allowed_variables=allowed_variables,
        allowed_functions=allowed_functions,
    )
    core = _equivalence_core(
        lhs_artifact,
        rhs_artifact,
        seed=seed,
        probe_target=probe_target,
        abs_tolerance=tolerance,
        rel_tolerance=rel_tolerance,
        probe_points=probe_points,
        probe_source=probe_source,
        probe_sample_sha256=probe_sample_sha256,
        enable_exact_decimal_rebuild=False,
    )
    decision = core["symbolic_decision"]
    probe_hash = _sha256_text(_canonical_json(core["probe_records"]))
    skipped_probe_records = list(core["skipped_probe_records"])
    skipped_probe_reasons = dict(
        sorted(Counter(record["reason"] for record in skipped_probe_records).items())
    )
    if include_tree_distance:
        normalized_edit_distance = normalized_tree_edit_distance(
            lhs_artifact,
            rhs_artifact,
        )
        tree_evidence: dict[str, object] = {
            "computed": True,
            "normalized_edit_distance": normalized_edit_distance,
            "tree_similarity": 1.0 - normalized_edit_distance,
        }
    else:
        tree_evidence = {
            "computed": False,
            "normalized_edit_distance": None,
            "tree_similarity": None,
        }
    payload = {
        "decision": decision,
        "lhs_artifact": {key: value for key, value in lhs_artifact.items() if key != "sympy_expression"},
        "rhs_artifact": {key: value for key, value in rhs_artifact.items() if key != "sympy_expression"},
        "symbolic_difference": {
            "decision": core["symbolic_decision"],
            "proof_basis": core["proof_basis"],
            "counterexample": core["counterexample"],
            "numeric_probes": list(core["probe_records"]),
            "probe_hash": probe_hash,
            "probe_source": core["probe_source"],
            "probe_sample_sha256": core["probe_sample_sha256"],
            "normalized_probe_points_sha256": core[
                "normalized_probe_points_sha256"
            ],
            "skipped_probe_count": len(skipped_probe_records),
            "skipped_probe_reasons": skipped_probe_reasons,
            "skipped_probes": skipped_probe_records,
            "max_abs_error": _max_metric(core["probe_records"], "abs_error"),
            "max_rel_error": _max_metric(core["probe_records"], "rel_error"),
            "max_tolerance": _max_metric(core["probe_records"], "tolerance"),
            "assumptions": list(core["assumptions"]),
        },
        "tree": tree_evidence,
        "variable": {"f1": variable_f1(lhs_artifact, rhs_artifact)},
        "operator": {"f1": operator_f1(lhs_artifact, rhs_artifact)},
        "probe_seed": seed,
        "probe_source": core["probe_source"],
        "probe_sample_sha256": core["probe_sample_sha256"],
        "normalized_probe_points_sha256": core[
            "normalized_probe_points_sha256"
        ],
        "probe_count": len(core["probe_records"]),
        "skipped_probe_count": len(skipped_probe_records),
        "skipped_probe_reasons": skipped_probe_reasons,
        "skipped_probes": skipped_probe_records,
        "numeric_probes": list(core["probe_records"]),
        "probe_hash": probe_hash,
        "max_abs_error": _max_metric(core["probe_records"], "abs_error"),
        "max_rel_error": _max_metric(core["probe_records"], "rel_error"),
        "max_tolerance": _max_metric(core["probe_records"], "tolerance"),
    }
    payload["evidence_sha256"] = _sha256_text(
        _canonical_json({key: value for key, value in payload.items() if key != "evidence_sha256"})
    )
    return payload


def validate_simplification(
    original: str,
    simplified: str,
    allowed_variables: Collection[str],
    allowed_functions: Collection[str],
    seed: int,
    *,
    probe_target: int = 12,
    tolerance: float = 1e-9,
    rel_tolerance: float = 1e-9,
    probe_points: Sequence[Mapping[str, object]] | None = None,
    probe_source: str | None = None,
    probe_sample_sha256: str | None = None,
    original_construction_mode: str | None = None,
) -> dict[str, object]:
    """验证化简是否保持等价，并返回可冻结的确定性证据。"""

    if original_construction_mode not in {
        None,
        "evaluated",
        "unevaluated_large_ast",
    }:
        raise SymbolicEvidenceError(
            f"未知 original construction mode: {original_construction_mode!r}"
        )
    original_artifact = build_symbolic_artifact(
        original,
        allowed_variables=allowed_variables,
        allowed_functions=allowed_functions,
        evaluate_expressions=(
            None
            if original_construction_mode is None
            else original_construction_mode == "evaluated"
        ),
    )
    if original == simplified:
        normalized_points = (
            _normalize_external_probe_points(probe_points)
            if probe_points is not None
            else []
        )
        normalized_probe_points_sha256 = _probe_sample_sha256(normalized_points)
        return {
            "decision": "equivalent",
            "symbolic_decision": "equivalent",
            "proof_basis": "artifact_identity",
            "probe_seed": seed,
            "probe_source": probe_source or (
                "external" if probe_points is not None else "not_run_source_identity"
            ),
            "probe_sample_sha256": probe_sample_sha256 or normalized_probe_points_sha256,
            "normalized_probe_points_sha256": normalized_probe_points_sha256,
            "probe_count": 0,
            "skipped_probe_count": 0,
            "skipped_probe_reasons": {},
            "skipped_probes": [],
            "probe_hash": _sha256_text(_canonical_json([])),
            "max_abs_error": None,
            "max_rel_error": None,
            "max_tolerance": None,
            "numeric_probes": [],
            "assumptions": [],
            "original_sha256": original_artifact["artifact_sha256"],
            "simplified_sha256": original_artifact["artifact_sha256"],
        }
    simplified_allowed_functions = set(allowed_functions)
    # 比较表达式在 canonical tree 中会成为 Piecewise(1/0)。允许模型使用
    # 等价的显式 Piecewise 记法，但不放宽其它新函数。
    if "piecewise" in set(original_artifact["operator_set"]):
        simplified_allowed_functions.add("where")
    simplified_artifact = build_symbolic_artifact(
        simplified,
        allowed_variables=allowed_variables,
        allowed_functions=simplified_allowed_functions,
    )
    original_variables = set(original_artifact["variables"])
    simplified_variables = set(simplified_artifact["variables"])
    if not simplified_variables.issubset(original_variables):
        raise SymbolicEvidenceError(
            f"化简结果引入了新变量: {tuple(sorted(simplified_variables - original_variables))}"
        )
    original_functions = set(original_artifact["function_set"])
    simplified_functions = set(simplified_artifact["function_set"])
    if not simplified_functions.issubset(
        original_functions | simplified_allowed_functions
    ):
        raise SymbolicEvidenceError(
            f"化简结果引入了新函数: {tuple(sorted(simplified_functions - original_functions))}"
        )

    core = _equivalence_core(
        original_artifact,
        simplified_artifact,
        seed=seed,
        probe_target=probe_target,
        abs_tolerance=tolerance,
        rel_tolerance=rel_tolerance,
        probe_points=probe_points,
        probe_source=probe_source,
        probe_sample_sha256=probe_sample_sha256,
    )
    probe_hash = _sha256_text(_canonical_json(core["probe_records"]))
    skipped_probe_records = list(core["skipped_probe_records"])
    skipped_probe_reasons = dict(
        sorted(Counter(record["reason"] for record in skipped_probe_records).items())
    )
    max_abs_error = _max_metric(core["probe_records"], "abs_error")
    max_rel_error = _max_metric(core["probe_records"], "rel_error")
    max_tolerance = _max_metric(core["probe_records"], "tolerance")
    if core["counterexample"] is not None:
        evidence = {
            "decision": "contract_error",
            "symbolic_decision": "not_equivalent",
            "proof_basis": core["proof_basis"],
            "probe_seed": seed,
            "probe_source": core["probe_source"],
            "probe_sample_sha256": core["probe_sample_sha256"],
            "normalized_probe_points_sha256": core[
                "normalized_probe_points_sha256"
            ],
            "probe_count": len(core["probe_records"]),
            "skipped_probe_count": len(skipped_probe_records),
            "skipped_probe_reasons": skipped_probe_reasons,
            "skipped_probes": skipped_probe_records,
            "probe_hash": probe_hash,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "max_tolerance": max_tolerance,
            "numeric_probes": list(core["probe_records"]),
            "counterexample": core["counterexample"],
            "assumptions": list(core["assumptions"]),
            "original_sha256": original_artifact["artifact_sha256"],
            "simplified_sha256": simplified_artifact["artifact_sha256"],
        }
        raise SimplificationContractError("化简结果存在确定性数值反例", evidence=evidence)
    if core["symbolic_decision"] == "not_equivalent":
        evidence = {
            "decision": "contract_error",
            "symbolic_decision": "not_equivalent",
            "proof_basis": core["proof_basis"],
            "probe_seed": seed,
            "probe_source": core["probe_source"],
            "probe_sample_sha256": core["probe_sample_sha256"],
            "normalized_probe_points_sha256": core[
                "normalized_probe_points_sha256"
            ],
            "probe_count": len(core["probe_records"]),
            "skipped_probe_count": len(skipped_probe_records),
            "skipped_probe_reasons": skipped_probe_reasons,
            "skipped_probes": skipped_probe_records,
            "probe_hash": probe_hash,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "max_tolerance": max_tolerance,
            "numeric_probes": list(core["probe_records"]),
            "counterexample": core["counterexample"],
            "assumptions": list(core["assumptions"]),
            "original_sha256": original_artifact["artifact_sha256"],
            "simplified_sha256": simplified_artifact["artifact_sha256"],
        }
        raise SimplificationContractError("化简结果与原式不等价", evidence=evidence)
    return {
        "decision": "equivalent" if core["symbolic_decision"] == "equivalent" else "undetermined",
        "symbolic_decision": core["symbolic_decision"],
        "proof_basis": core["proof_basis"],
        "probe_seed": seed,
        "probe_source": core["probe_source"],
        "probe_sample_sha256": core["probe_sample_sha256"],
        "normalized_probe_points_sha256": core[
            "normalized_probe_points_sha256"
        ],
        "probe_count": len(core["probe_records"]),
        "skipped_probe_count": len(skipped_probe_records),
        "skipped_probe_reasons": skipped_probe_reasons,
        "skipped_probes": skipped_probe_records,
        "probe_hash": probe_hash,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "max_tolerance": max_tolerance,
        "numeric_probes": list(core["probe_records"]),
        "assumptions": list(core["assumptions"]),
        "original_sha256": original_artifact["artifact_sha256"],
        "simplified_sha256": simplified_artifact["artifact_sha256"],
    }

__all__ = [
    "NUMPY_ATTRIBUTE_WHITELIST",
    "OPAQUE_FUNCTIONS",
    "SimplificationContractError",
    "SymbolicEvidenceError",
    "build_pair_evidence",
    "build_symbolic_artifact",
    "normalized_tree_edit_distance",
    "operator_f1",
    "tree_similarity",
    "validate_simplification",
    "variable_f1",
]
