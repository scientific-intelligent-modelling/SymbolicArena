"""gplearn 原生 prefix 的受保护语义证据，不经普通 SymPy 化简。

表示采用后序扁平节点表，避免深表达式超过 Python AST/JSON 递归深度。
`normalized_expression` 仅作诊断字段；包括其值为 ``nan`` 时，语义始终来自 raw prefix。
数值规则与 benchmarks.runner._eval_gplearn_prefix_node 的 0.001 阈值一致。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class PrefixEvidenceError(ValueError):
    """原生 prefix 或数值验证输入不满足契约。"""


PROTECTED_THRESHOLD = 0.001
OPERATOR_ARITY = {
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2,
    "max": 2, "min": 2,
    "sqrt": 1, "log": 1, "inv": 1, "abs": 1, "neg": 1,
    "sin": 1, "cos": 1, "tan": 1, "exp": 1,
    "sig": 1, "sigmoid": 1,
    "pdiv": 2, "plog": 1, "psqrt": 1, "pinv": 1,
}
PROTECTED_NAMES = {
    "div": "protected_div", "sqrt": "protected_sqrt",
    "log": "protected_log", "inv": "protected_inv",
    "pdiv": "protected_div", "psqrt": "protected_sqrt",
    "plog": "protected_log", "pinv": "protected_inv",
}
TYPED_NAME = {
    "protected_div": "pdiv", "protected_sqrt": "psqrt",
    "protected_log": "plog", "protected_inv": "pinv",
}
_TOKEN_RE = re.compile(
    r"\s*(?:([A-Za-z_]\w*)|([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)|([(),]))"
)
_VARIABLE_RE = re.compile(r"[Xx](\d+)\Z")


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(raw):
        if not raw[pos:].strip():
            break
        match = _TOKEN_RE.match(raw, pos)
        if match is None:
            raise PrefixEvidenceError(f"非法 prefix 字符: {raw[pos:pos + 24]!r}")
        tokens.append(next(group for group in match.groups() if group is not None))
        pos = match.end()
    if not tokens:
        raise PrefixEvidenceError("raw prefix 为空")
    return tokens


def _render_typed(nodes: list[dict[str, Any]], root: int) -> str:
    """用显式栈写前缀文本，避免深树的递归或二次字符串拼接。"""
    pieces: list[str] = []
    pending: list[tuple[str, int | str]] = [("node", root)]
    while pending:
        kind, value = pending.pop()
        if kind == "text":
            pieces.append(str(value))
            continue
        node = nodes[int(value)]
        if node["kind"] == "variable":
            pieces.append(f"x{node['column_index']}")
        elif node["kind"] == "constant":
            pieces.append(node["literal"])
        else:
            pieces.append(TYPED_NAME.get(node["operator"], node["operator"]) + "(")
            pending.append(("text", ")"))
            for child_position in range(len(node["children"]) - 1, -1, -1):
                pending.append(("node", node["children"][child_position]))
                if child_position:
                    pending.append(("text", ","))
    return "".join(pieces)


def build_prefix_evidence(
    raw_prefix: str,
    feature_names: Sequence[str],
    *,
    normalized_expression: str | None = None,
) -> dict[str, Any]:
    """将 gplearn 原生字符串转换为可哈希的扁平 typed tree。

    节点按后序排列，children 仅引用先前节点。常数原始字面量保留，变量同时
    绑定列索引和实际 feature name；结构指纹抽象常数但绝不抽象输入列。
    """
    if not isinstance(raw_prefix, str) or not raw_prefix.strip():
        raise PrefixEvidenceError("raw_prefix 必须是非空字符串")
    if not feature_names or any(not isinstance(name, str) or not name for name in feature_names):
        raise PrefixEvidenceError("feature_names 必须是非空的列名序列")
    if len(set(feature_names)) != len(feature_names):
        raise PrefixEvidenceError("feature_names 有重复，无法无歧义绑定输入列")
    tokens = _tokens(raw_prefix)
    nodes: list[dict[str, Any]] = []
    exact_hashes: list[str] = []
    structure_hashes: list[str] = []
    depths: list[int] = []
    frames: list[dict[str, Any]] = []
    root: int | None = None
    expect_value = True

    def append_node(node: dict[str, Any], exact: object, structure: object, depth: int) -> None:
        nonlocal root, expect_value
        index = len(nodes)
        nodes.append(node)
        exact_hashes.append(_digest(exact))
        structure_hashes.append(_digest(structure))
        depths.append(depth)
        if frames:
            frames[-1]["children"].append(index)
        elif root is None:
            root = index
        else:
            raise PrefixEvidenceError("prefix 包含多个根表达式")
        expect_value = False

    position = 0
    while position < len(tokens):
        token = tokens[position]
        if expect_value:
            if token in {"(", ")", ","}:
                raise PrefixEvidenceError(f"期望值节点，得到 {token!r}")
            is_call = position + 1 < len(tokens) and tokens[position + 1] == "("
            if is_call:
                operator = token.lower()
                if operator not in OPERATOR_ARITY:
                    raise PrefixEvidenceError(f"不支持的 gplearn 算子: {token}")
                frames.append({"operator": operator, "children": []})
                position += 2
                continue
            variable = _VARIABLE_RE.fullmatch(token)
            if variable:
                column = int(variable.group(1))
                if column >= len(feature_names):
                    raise PrefixEvidenceError(f"变量 X{column} 超过输入维度 {len(feature_names)}")
                name = feature_names[column]
                node = {"kind": "variable", "column_index": column, "feature_name": name}
                append_node(node, node, node, 1)
            else:
                try:
                    number = float(token)
                except ValueError as exc:
                    raise PrefixEvidenceError(f"不支持的 gplearn 叶子: {token!r}") from exc
                if not math.isfinite(number):
                    raise PrefixEvidenceError("非有限常数不允许作为原生 prefix 叶子")
                node = {"kind": "constant", "literal": token, "value": number}
                append_node(node, node, {"kind": "constant"}, 1)
            position += 1
            continue
        if token == ",":
            if not frames:
                raise PrefixEvidenceError("根表达式后出现逗号")
            expect_value = True
            position += 1
            continue
        if token == ")":
            if not frames:
                raise PrefixEvidenceError("出现未配对右括号")
            frame = frames.pop()
            children = frame["children"]
            operator = frame["operator"]
            if len(children) != OPERATOR_ARITY[operator]:
                raise PrefixEvidenceError(f"算子 {operator} 需要 {OPERATOR_ARITY[operator]} 个参数，得到 {len(children)}")
            typed_operator = PROTECTED_NAMES.get(operator, operator)
            node = {"kind": "operator", "operator": typed_operator, "children": children}
            append_node(
                node,
                [typed_operator, [exact_hashes[index] for index in children]],
                [typed_operator, [structure_hashes[index] for index in children]],
                1 + max(depths[index] for index in children),
            )
            position += 1
            continue
        raise PrefixEvidenceError(f"表达式值后缺少逗号或右括号: {token!r}")
    if frames or expect_value or root is None:
        raise PrefixEvidenceError("prefix 不完整")
    operators = sorted({node["operator"] for node in nodes if node["kind"] == "operator"})
    columns = sorted({node["column_index"] for node in nodes if node["kind"] == "variable"})
    return {
        "schema_version": "gplearn_typed_prefix.v1",
        "raw_prefix": raw_prefix,
        "raw_prefix_sha256": hashlib.sha256(raw_prefix.encode("utf-8")).hexdigest(),
        "normalized_expression_diagnostic": normalized_expression,
        "feature_names": list(feature_names),
        "protected_threshold": PROTECTED_THRESHOLD,
        "protected_semantics": {
            "protected_div": "abs(denominator)>0.001 ? numerator/denominator : 1.0",
            "protected_log": "abs(value)>0.001 ? log(abs(value)) : 0.0",
            "protected_sqrt": "sqrt(abs(value))",
            "protected_inv": "abs(value)>0.001 ? 1.0/value : 0.0",
        },
        "nodes": nodes,
        "root_index": root,
        "node_count": len(nodes),
        "tree_depth": depths[root],
        "operator_set": operators,
        "variable_columns": columns,
        "variables": [feature_names[index] for index in columns],
        "typed_expression": _render_typed(nodes, root),
        "exact_fingerprint": exact_hashes[root],
        "constants_abstracted_structure_fingerprint": structure_hashes[root],
    }


def build_inventory_prefix_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """从已冻结的 gplearn terminal inventory 构造并绑定原生语义证据。"""
    if str(row.get("algorithm", "")).casefold() != "gplearn":
        raise PrefixEvidenceError("inventory 行不是 gplearn")
    raw = row.get("native_prefix")
    if not isinstance(raw, str) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != row.get("native_prefix_sha256"):
        raise PrefixEvidenceError("inventory 的 native_prefix SHA 不匹配")
    feature_names = row.get("feature_names")
    if not isinstance(feature_names, list):
        raise PrefixEvidenceError("inventory 缺少 feature_names")
    for index, name in enumerate(feature_names):
        mapping = row.get("variable_mapping") or {}
        if mapping.get(f"X{index}") != name:
            raise PrefixEvidenceError(f"inventory 输入列 X{index} 映射不匹配")
    expected_key = f"gplearn::{row.get('dataset_id')}::s{row.get('seed')}::{row.get('condition')}"
    if row.get("logical_key") != expected_key or not row.get("terminal_snapshot_sha256"):
        raise PrefixEvidenceError("inventory 终点身份或 snapshot SHA 缺失")
    terminal_expression = row.get("terminal_expression")
    if not isinstance(terminal_expression, str) or hashlib.sha256(terminal_expression.encode("utf-8")).hexdigest() != row.get("terminal_expression_sha256"):
        raise PrefixEvidenceError("inventory 终点公式 SHA 不匹配")
    display = row.get("display_expression")
    evidence = build_prefix_evidence(
        raw, feature_names, normalized_expression=row.get("display_expression")
    )
    evidence["source_binding"] = {
        field: row.get(field) for field in (
            "logical_key", "terminal_snapshot_sha256", "terminal_expression_sha256",
            "selected_result_sha256", "native_prefix_sha256", "gt_frozen_evaluation_key",
            "gt_reference_source_sha256",
        )
    }
    return evidence


def evaluate_prefix_evidence(evidence: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """按 typed nodes 后序求值，完全绕开 AST 与 SymPy。"""
    data = np.asarray(X, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(evidence["feature_names"]):
        raise PrefixEvidenceError("数值数据必须为二维且列数等于 feature_names")
    values: list[Any] = []
    for index, node in enumerate(evidence["nodes"]):
        kind = node["kind"]
        if kind == "constant":
            result = float(node["value"])
        elif kind == "variable":
            result = data[:, int(node["column_index"])]
        elif kind == "operator":
            children = node["children"]
            if any(not isinstance(child, int) or child < 0 or child >= index for child in children):
                raise PrefixEvidenceError("typed tree 包含非法或前向引用")
            args = [values[child] for child in children]
            op = node["operator"]
            with np.errstate(all="ignore"):
                if op == "add": result = args[0] + args[1]
                elif op == "sub": result = args[0] - args[1]
                elif op == "mul": result = args[0] * args[1]
                elif op == "protected_div":
                    result = np.where(np.abs(args[1]) > PROTECTED_THRESHOLD, np.divide(args[0], args[1]), 1.0)
                elif op == "protected_sqrt": result = np.sqrt(np.abs(args[0]))
                elif op == "protected_log":
                    result = np.where(np.abs(args[0]) > PROTECTED_THRESHOLD, np.log(np.abs(args[0])), 0.0)
                elif op == "protected_inv":
                    result = np.where(np.abs(args[0]) > PROTECTED_THRESHOLD, np.divide(1.0, args[0]), 0.0)
                elif op == "abs": result = np.abs(args[0])
                elif op == "neg": result = -args[0]
                elif op == "sin": result = np.sin(args[0])
                elif op == "cos": result = np.cos(args[0])
                elif op == "tan": result = np.tan(args[0])
                elif op == "exp": result = np.exp(args[0])
                elif op in {"sig", "sigmoid"}: result = 1.0 / (1.0 + np.exp(-args[0]))
                elif op == "pow": result = np.power(args[0], args[1])
                elif op == "max": result = np.maximum(args[0], args[1])
                elif op == "min": result = np.minimum(args[0], args[1])
                else: raise PrefixEvidenceError(f"typed tree 包含未知算子 {op!r}")
        else:
            raise PrefixEvidenceError(f"typed tree 包含未知节点类型 {kind!r}")
        values.append(result)
        if kind == "operator":
            for child in node["children"]:
                values[child] = None
    root = evidence["root_index"]
    if not isinstance(root, int) or root != len(values) - 1:
        raise PrefixEvidenceError("typed tree 根节点非法")
    output = np.asarray(values[root], dtype=float)
    if output.ndim == 0:
        return np.full(data.shape[0], float(output))
    try:
        return np.broadcast_to(output, (data.shape[0],)).astype(float).reshape(-1)
    except ValueError as exc:
        raise PrefixEvidenceError("typed tree 输出不能按样本数广播") from exc


def _boundary_points(data: np.ndarray) -> np.ndarray:
    base = np.median(data, axis=0)
    points = []
    for column in range(data.shape[1]):
        for value in (-0.0011, -0.001, -0.0009, 0.0, 0.0009, 0.001, 0.0011):
            point = base.copy()
            point[column] = value
            points.append(point)
    return np.asarray(points, dtype=float)


def compare_with_native(
    evidence: dict[str, Any],
    X: np.ndarray,
    *,
    native_predict: Callable[[np.ndarray], np.ndarray],
    include_boundary_probes: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """用真实数值行及阈值边界行对照调用方提供的原生预测器。

    需由调用方传入 gplearn model.predict 或 runner 原生 prefix evaluator，不能把
    当前模块的 evaluator 再传回自身作为“独立验证”。非有限值一律不判通过。
    """
    data = np.asarray(X, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != len(evidence["feature_names"]):
        raise PrefixEvidenceError("验证数据形状无效")
    if not np.isfinite(data).all():
        raise PrefixEvidenceError("验证数据含非有限输入")
    if rtol < 0 or atol < 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise PrefixEvidenceError("数值容差无效")
    boundaries = _boundary_points(data) if include_boundary_probes else np.empty((0, data.shape[1]))
    probes = np.vstack((data, boundaries))
    predicted = evaluate_prefix_evidence(evidence, probes)
    try:
        native = np.asarray(native_predict(probes), dtype=float).reshape(-1)
    except Exception as exc:
        raise PrefixEvidenceError(f"原生预测器执行失败: {type(exc).__name__}") from exc
    if native.shape != predicted.shape:
        raise PrefixEvidenceError("原生预测器输出形状与 typed 表示不一致")
    finite = np.isfinite(native) & np.isfinite(predicted)
    mismatch = ~np.isclose(predicted, native, rtol=rtol, atol=atol, equal_nan=False)
    errors = np.abs(predicted[finite] - native[finite])
    scale = np.maximum(np.abs(native[finite]), atol)
    sample_sha = hashlib.sha256(np.asarray(probes, dtype="<f8").tobytes()).hexdigest()
    return {
        "schema_version": "gplearn_native_numeric_comparison.v1",
        "raw_prefix_sha256": evidence["raw_prefix_sha256"],
        "representation_fingerprint": evidence["exact_fingerprint"],
        "probe_values_sha256": sample_sha,
        "observed_count": int(data.shape[0]),
        "boundary_probe_count": int(boundaries.shape[0]),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "mismatch_count": int(mismatch.sum()),
        "mismatch_indices_first20": np.flatnonzero(mismatch)[:20].tolist(),
        "max_abs_error": float(errors.max()) if errors.size else None,
        "max_rel_error": float((errors / scale).max()) if errors.size else None,
        "rtol": rtol, "atol": atol,
        "passed": bool(finite.all() and not mismatch.any()),
        "status": "nonfinite" if not finite.all() else "mismatch" if mismatch.any() else "pass",
    }


def native_equivalence_probe(
    native_prefix: str,
    candidate_typed_expression: str,
    feature_names: Sequence[str],
    points: np.ndarray,
    *,
    native_predict: Callable[[np.ndarray], np.ndarray] | None = None,
    include_boundary_probes: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """对候选 typed-prefix 给出原生数值支持或反例，而非数学证明。

    候选必须显式写 pdiv/plog/psqrt/pinv；原始 gplearn 的 div/log/sqrt/inv
    只允许出现在 `native_prefix`。默认原生对照为 runner 的 prefix evaluator。
    """
    if not isinstance(candidate_typed_expression, str) or not candidate_typed_expression.strip():
        raise PrefixEvidenceError("候选 typed expression 为空")
    if re.search(r"\b(?:div|log|sqrt|inv)\s*\(", candidate_typed_expression, flags=re.I):
        raise PrefixEvidenceError("候选必须使用 pdiv/plog/psqrt/pinv 显式保护算子")
    original = build_prefix_evidence(native_prefix, feature_names)
    candidate = build_prefix_evidence(candidate_typed_expression, feature_names)
    if native_predict is None:
        from scientific_intelligent_modelling.benchmarks.runner import _predict_gplearn_prefix_expression

        native_predict = lambda values: _predict_gplearn_prefix_expression(native_prefix, values)
    consistency = compare_with_native(
        original, points, native_predict=native_predict,
        include_boundary_probes=include_boundary_probes, rtol=rtol, atol=atol,
    )
    data = np.asarray(points, dtype=float)
    boundaries = _boundary_points(data) if include_boundary_probes else np.empty((0, data.shape[1]))
    probes = np.vstack((data, boundaries))
    try:
        native = np.asarray(native_predict(probes), dtype=float).reshape(-1)
    except Exception as exc:
        raise PrefixEvidenceError(f"原生预测器执行失败: {type(exc).__name__}") from exc
    comparison = evaluate_prefix_evidence(candidate, probes)
    finite = np.isfinite(native) & np.isfinite(comparison)
    mismatch = ~np.isclose(native, comparison, rtol=rtol, atol=atol, equal_nan=False)
    first = np.flatnonzero(mismatch)
    status = (
        "native_representation_mismatch" if not consistency["passed"] else
        "nonfinite" if not finite.all() else
        "numeric_counterexample" if mismatch.any() else
        "numeric_support_only"
    )

    def finite_or_text(value: float) -> float | str:
        return float(value) if math.isfinite(float(value)) else str(float(value))

    return {
        "schema_version": "gplearn_native_equivalence_probe.v1",
        "native_prefix_sha256": original["raw_prefix_sha256"],
        "candidate_typed_expression_sha256": candidate["raw_prefix_sha256"],
        "native_consistency": consistency,
        "observed_count": int(data.shape[0]),
        "boundary_probe_count": int(boundaries.shape[0]),
        "nonfinite_count": int((~finite).sum()),
        "mismatch_count": int(mismatch.sum()),
        "counterexample": None if not first.size else {
            "point_index": int(first[0]),
            "values": probes[first[0]].tolist(),
            "native_value": finite_or_text(native[first[0]]),
            "candidate_value": finite_or_text(comparison[first[0]]),
        },
        "rtol": rtol, "atol": atol,
        "status": status,
        "passed": status == "numeric_support_only",
    }


def _typed_postorder(evidence: dict[str, Any]) -> tuple[list[tuple[str, str | None] | None], list[int]]:
    import sympy as sp

    labels: list[tuple[str, str | None] | None] = [None]
    leftmost = [0]
    for index, node in enumerate(evidence["nodes"], start=1):
        kind = node["kind"]
        if kind == "variable":
            name = str(node["feature_name"])
            labels.append(("Symbol", sp.srepr(sp.Symbol(name))))
            leftmost.append(index)
        elif kind == "constant":
            literal = node["literal"]
            number = sp.Float(literal) if any(mark in literal for mark in ".eE") else sp.Integer(literal)
            labels.append((type(number).__name__, sp.srepr(number)))
            leftmost.append(index)
        elif kind == "operator":
            children = node["children"]
            if not children or any(child >= index - 1 or child < 0 for child in children):
                raise PrefixEvidenceError("typed tree 子节点索引无效")
            labels.append((node["operator"], None))
            leftmost.append(leftmost[children[0] + 1])
        else:
            raise PrefixEvidenceError("typed tree 节点类型无效")
    if evidence["root_index"] != len(labels) - 2:
        raise PrefixEvidenceError("typed tree 根索引无效")
    return labels, leftmost


def _gt_postorder(tree: dict[str, Any]) -> tuple[list[tuple[str, str | None] | None], list[int]]:
    """迭代后序化 GT canonical tree，避免 Python 递归上限。"""
    labels: list[tuple[str, str | None] | None] = [None]
    leftmost = [0]
    node_index: dict[int, int] = {}
    pending: list[tuple[dict[str, Any], bool]] = [(tree, False)]
    while pending:
        node, expanded = pending.pop()
        if not isinstance(node, dict) or not isinstance(node.get("type"), str):
            raise PrefixEvidenceError("GT canonical tree 节点非法")
        children = node.get("args", [])
        if not isinstance(children, list):
            raise PrefixEvidenceError("GT canonical tree.args 必须是列表")
        if not expanded:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(children))
            continue
        index = len(labels)
        labels.append((node["type"], str(node["value"]) if "value" in node else None))
        leftmost.append(leftmost[node_index[id(children[0])]] if children else index)
        node_index[id(node)] = index
    return labels, leftmost


def _keyroots(leftmost: list[int]) -> list[int]:
    latest: dict[int, int] = {}
    for index in range(1, len(leftmost)):
        latest[leftmost[index]] = index
    return sorted(latest.values())


def _ordered_tree_distance(
    lhs_labels: list[tuple[str, str | None] | None], lhs_leftmost: list[int],
    rhs_labels: list[tuple[str, str | None] | None], rhs_leftmost: list[int],
) -> int:
    """与 symbolic_evidence 的 Zhang-Shasha 成本完全一致，且仅使用迭代 DP。"""
    lhs_size = len(lhs_labels) - 1
    rhs_size = len(rhs_labels) - 1
    distance = [[0] * (rhs_size + 1) for _ in range(lhs_size + 1)]
    for lhs_root in _keyroots(lhs_leftmost):
        lhs_start = lhs_leftmost[lhs_root]
        for rhs_root in _keyroots(rhs_leftmost):
            rhs_start = rhs_leftmost[rhs_root]
            forest = [[0] * (rhs_root - rhs_start + 2) for _ in range(lhs_root - lhs_start + 2)]
            for lhs_index in range(lhs_start, lhs_root + 1):
                row = lhs_index - lhs_start + 1
                forest[row][0] = forest[row - 1][0] + 1
            for rhs_index in range(rhs_start, rhs_root + 1):
                col = rhs_index - rhs_start + 1
                forest[0][col] = forest[0][col - 1] + 1
            for lhs_index in range(lhs_start, lhs_root + 1):
                row = lhs_index - lhs_start + 1
                for rhs_index in range(rhs_start, rhs_root + 1):
                    col = rhs_index - rhs_start + 1
                    delete = forest[row - 1][col] + 1
                    insert = forest[row][col - 1] + 1
                    if lhs_leftmost[lhs_index] == lhs_start and rhs_leftmost[rhs_index] == rhs_start:
                        rename = 0 if lhs_labels[lhs_index] == rhs_labels[rhs_index] else 1
                        value = min(delete, insert, forest[row - 1][col - 1] + rename)
                        forest[row][col] = value
                        distance[lhs_index][rhs_index] = value
                    else:
                        forest[row][col] = min(
                            delete, insert,
                            forest[lhs_leftmost[lhs_index] - lhs_start][rhs_leftmost[rhs_index] - rhs_start]
                            + distance[lhs_index][rhs_index],
                        )
    return distance[lhs_size][rhs_size]


def typed_tree_similarity(
    evidence: dict[str, Any],
    gt_canonical_tree: dict[str, Any],
    *,
    max_pred_nodes: int = 1000,
    max_ref_nodes: int = 1000,
    max_pair_cells: int = 100000,
) -> dict[str, Any]:
    """计算 typed prefix 与 GT SymPy canonical tree 的同规则 NED。

    这是混合表示的结构诊断，不是旧 SYM 的同构 SymPy-tree T。大树显式返回
    unavailable_complexity；不降采样、不用近似分数冒充正式结果。
    """
    pred_count = len(evidence["nodes"])
    ref_count = 0
    pending: list[dict[str, Any]] = [gt_canonical_tree]
    while pending and ref_count <= max_ref_nodes:
        node = pending.pop()
        if not isinstance(node, dict) or not isinstance(node.get("args", []), list):
            raise PrefixEvidenceError("GT canonical tree 节点非法")
        ref_count += 1
        pending.extend(node.get("args", []))
    if pred_count > max_pred_nodes or ref_count > max_ref_nodes or pred_count * ref_count > max_pair_cells:
        return {
            "status": "unavailable_complexity", "tree_similarity": None,
            "pred_node_count": pred_count, "ref_node_count_at_least": ref_count,
            "max_pred_nodes": max_pred_nodes, "max_ref_nodes": max_ref_nodes,
            "max_pair_cells": max_pair_cells,
            "basis": "gplearn_typed_prefix_vs_gt_sympy_canonical",
            "same_ned_cost_rule": True, "legacy_sympy_tree_equivalent": False,
        }
    lhs_labels, lhs_leftmost = _typed_postorder(evidence)
    rhs_labels, rhs_leftmost = _gt_postorder(gt_canonical_tree)
    raw_distance = _ordered_tree_distance(lhs_labels, lhs_leftmost, rhs_labels, rhs_leftmost)
    ned = raw_distance / max(pred_count + ref_count, 1)
    return {
        "status": "computed", "tree_similarity": 1.0 - ned,
        "normalized_tree_edit_distance": ned, "raw_tree_edit_distance": raw_distance,
        "pred_node_count": pred_count, "ref_node_count": ref_count,
        "basis": "gplearn_typed_prefix_vs_gt_sympy_canonical",
        "same_ned_cost_rule": True, "legacy_sympy_tree_equivalent": False,
    }
