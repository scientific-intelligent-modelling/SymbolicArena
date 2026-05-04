#!/usr/bin/env python3
"""为 Core-50 SYM-F formal judge 准备公式、变量映射和 probe 采样参数。

本脚本只做可追溯的参数抽取与审计，不调用任何大模型 API。
输出用于下一步正式 symbolic judge：
- ground truth formula 来源与标准化表达式
- metadata / CSV / formula.py 的变量映射
- 独立 probe samples 的采样范围
- 无法自动确认的风险项
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE50_CSV = REPO_ROOT / "exp-planning/04.Core50正式全量评测/core50_datasets.csv"
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "exp-planning/04.Core50正式全量评测/analysis/symf_formal_judge_params_20260504"
)

SPLIT_FILES = ["train.csv", "valid.csv", "id_test.csv", "ood_test.csv"]
PROBE_SAMPLES_PER_DATASET = 4096
PROBE_RANDOM_SEED = 20260504

SYM_LOCALS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "pi": sp.pi,
    "E": sp.E,
    # ground-truth formula.py 里的 div 是 protected division；这里先用普通除法做
    # CAS 解析，并在参数表中单独标记 protected op 风险。
    "div": lambda a, b: a / b,
    "pow": sp.Pow,
}

PROTECTED_OPS = {"div", "sqrt", "log", "abs"}


@dataclass
class FormulaInfo:
    ok: bool
    error: str | None
    target_function: str | None
    arg_names: list[str]
    local_constants: dict[str, float]
    local_aliases: dict[str, str]
    return_raw: str | None
    return_substituted: str | None
    expression_x: str | None
    sympy_sstr: str | None
    formula_hash: str | None
    operators: list[str]
    protected_ops: list[str]
    free_symbols: list[str]


class _FormulaSubstituter(ast.NodeTransformer):
    def __init__(self, constants: dict[str, float], aliases: dict[str, str]):
        self.constants = constants
        self.aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self.aliases:
            return ast.copy_location(ast.Name(id=self.aliases[node.id], ctx=node.ctx), node)
        if node.id in self.constants:
            return ast.copy_location(ast.Constant(value=self.constants[node.id]), node)
        return node


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_expr_text(expr: str) -> str:
    out = expr.replace("numpy.", "").replace("np.", "").replace("math.", "")
    out = out.replace("^", "**")
    return out.strip()


def _safe_eval_constant(node: ast.AST) -> float | None:
    try:
        expr = ast.unparse(node)
        value = eval(  # noqa: S307 - 只在本地解析受控 formula.py 常数字面量。
            compile(ast.Expression(node), "<formula-const>", "eval"),
            {"__builtins__": {}},
            {"np": np, "numpy": np, "math": math, "pi": math.pi, "E": math.e},
        )
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _array_alias(node: ast.AST, arg_names: set[str]) -> str | None:
    """识别 `alias = np.asarray(x)` / `alias = np.array(x)` / `alias = x`。"""
    if isinstance(node, ast.Name) and node.id in arg_names:
        return node.id
    if isinstance(node, ast.Call):
        func_text = ast.unparse(node.func)
        if func_text in {"np.asarray", "numpy.asarray", "asarray", "np.array", "numpy.array", "array"} and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id in arg_names:
                return first.id
    return None


def _replace_args(expr: str, arg_names: list[str]) -> str:
    out = expr
    # 长变量名优先，避免把 `alpha` 中的 `a` 误替换。
    for idx, name in sorted(enumerate(arg_names), key=lambda item: len(item[1]), reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", f"x{idx}", out)
    return out


def _operator_names(expr: str) -> list[str]:
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", expr or ""))
    return sorted(name for name in names if name not in {"where"})


def _extract_formula(path: Path, target_name: str) -> FormulaInfo:
    if not path.exists():
        return FormulaInfo(False, "missing_formula_py", None, [], {}, {}, None, None, None, None, None, [], [], [])
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        global_constants: dict[str, float] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = _safe_eval_constant(node.value)
                if value is not None:
                    global_constants[node.targets[0].id] = value

        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        func = next((node for node in funcs if node.name == target_name), None)
        if func is None and funcs:
            # 有些历史数据 target_name 与函数名不完全一致时，保守选择最后一个非 helper 函数。
            helper_names = {"div", "exp", "log", "sqrt", "sin", "cos", "tan", "abs"}
            non_helpers = [node for node in funcs if node.name not in helper_names]
            func = non_helpers[-1] if non_helpers else funcs[-1]
        if func is None:
            raise ValueError("formula.py 中未找到函数定义")

        arg_names = [arg.arg for arg in func.args.args]
        arg_name_set = set(arg_names)
        constants: dict[str, float] = dict(global_constants)
        aliases: dict[str, str] = {}
        ret_node: ast.AST | None = None
        for stmt in func.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                alias = _array_alias(stmt.value, arg_name_set)
                if alias is not None:
                    aliases[name] = alias
                else:
                    value = _safe_eval_constant(stmt.value)
                    if value is not None:
                        constants[name] = value
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                ret_node = stmt.value
                break
        if ret_node is None:
            raise ValueError("目标函数中未找到 return")

        return_raw = _clean_expr_text(ast.unparse(ret_node))
        ret_sub_ast = _FormulaSubstituter(constants, aliases).visit(ast.fix_missing_locations(ret_node))
        ast.fix_missing_locations(ret_sub_ast)
        return_sub = _clean_expr_text(ast.unparse(ret_sub_ast))
        expression_x = _replace_args(return_sub, arg_names)
        operators = _operator_names(expression_x)
        protected_ops = sorted(set(operators) & PROTECTED_OPS)

        try:
            expr = sp.sympify(expression_x, locals=SYM_LOCALS)
            expr = sp.simplify(expr)
            sympy_sstr = sp.sstr(expr)
            formula_hash = hashlib.sha256(sp.srepr(expr).encode("utf-8")).hexdigest()
            free_symbols = sorted(str(symbol) for symbol in expr.free_symbols)
        except Exception as exc:
            sympy_sstr = None
            formula_hash = None
            free_symbols = []
            return FormulaInfo(
                False,
                f"sympy_parse_failed: {exc!r}",
                func.name,
                arg_names,
                constants,
                aliases,
                return_raw,
                return_sub,
                expression_x,
                sympy_sstr,
                formula_hash,
                operators,
                protected_ops,
                free_symbols,
            )

        return FormulaInfo(
            True,
            None,
            func.name,
            arg_names,
            constants,
            aliases,
            return_raw,
            return_sub,
            expression_x,
            sympy_sstr,
            formula_hash,
            operators,
            protected_ops,
            free_symbols,
        )
    except Exception as exc:
        return FormulaInfo(False, repr(exc), None, [], {}, {}, None, None, None, None, None, [], [], [])


def _feature_names(meta: dict[str, Any]) -> list[str]:
    features = meta.get("dataset", {}).get("features") or []
    out: list[str] = []
    if isinstance(features, list):
        for item in features:
            if isinstance(item, dict) and item.get("name") is not None:
                out.append(str(item["name"]))
    return out


def _target_name(meta: dict[str, Any]) -> str | None:
    target = meta.get("dataset", {}).get("target") or {}
    return str(target.get("name")) if isinstance(target, dict) and target.get("name") is not None else None


def _flatten_range(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_range(item))
    else:
        try:
            number = float(value)
        except Exception:
            return []
        if math.isfinite(number):
            out.append(number)
    return out


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
    except Exception:
        return []
    return [part.strip() for part in line.split(",") if part.strip()]


def _split_minmax(dataset_dir: Path, feature: str) -> tuple[float | None, float | None]:
    values: list[float] = []
    for split in SPLIT_FILES:
        path = dataset_dir / split
        if not path.exists():
            continue
        try:
            series = pd.read_csv(path, usecols=[feature])[feature]
        except Exception:
            continue
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.extend([float(arr.min()), float(arr.max())])
    if not values:
        return None, None
    return min(values), max(values)


def _probe_ranges(meta: dict[str, Any], dataset_dir: Path, features: list[str]) -> tuple[list[list[float]], list[str]]:
    feature_items = meta.get("dataset", {}).get("features") or []
    item_map = {
        str(item.get("name")): item
        for item in feature_items
        if isinstance(item, dict) and item.get("name") is not None
    }
    ranges: list[list[float]] = []
    sources: list[str] = []
    for feature in features:
        item = item_map.get(feature, {})
        nums = _flatten_range(item.get("train_range")) + _flatten_range(item.get("ood_range"))
        source = "metadata_train_ood_union"
        if not nums:
            lo, hi = _split_minmax(dataset_dir, feature)
            nums = [x for x in [lo, hi] if x is not None]
            source = "split_minmax_fallback"
        if not nums:
            ranges.append([float("nan"), float("nan")])
            sources.append("missing")
            continue
        lo = min(nums)
        hi = max(nums)
        if not math.isfinite(lo) or not math.isfinite(hi):
            ranges.append([float("nan"), float("nan")])
            sources.append("nonfinite")
            continue
        if lo == hi:
            eps = max(abs(lo) * 0.01, 1e-6)
            lo -= eps
            hi += eps
            source += "_expanded_constant"
        ranges.append([float(lo), float(hi)])
        sources.append(source)
    return ranges, sources


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> None:
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(CORE50_CSV)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for _, item in core.iterrows():
        dataset_dir = REPO_ROOT / str(item["dataset_dir"])
        metadata_path = REPO_ROOT / str(item["metadata_yaml"])
        formula_path = REPO_ROOT / str(item["formula_py"])
        meta = _load_yaml(metadata_path)
        meta_features = _feature_names(meta)
        meta_target = _target_name(meta)
        target_name = str(item["target_name"])
        formula = _extract_formula(formula_path, target_name)
        train_header = _csv_header(dataset_dir / "train.csv")
        csv_features = [col for col in train_header if col != target_name]
        probe_ranges, probe_sources = _probe_ranges(meta, dataset_dir, meta_features or csv_features)
        expected_symbols = {f"x{i}" for i in range(len(meta_features or csv_features or formula.arg_names))}
        extra_free_symbols = sorted(set(formula.free_symbols) - expected_symbols)
        unused_feature_symbols = sorted(expected_symbols - set(formula.free_symbols))

        issue_list: list[str] = []
        if not metadata_path.exists():
            issue_list.append("missing_metadata_yaml")
        if not formula_path.exists():
            issue_list.append("missing_formula_py")
        if meta_target and meta_target != target_name:
            issue_list.append(f"target_mismatch: core50={target_name}, metadata={meta_target}")
        if meta_features and csv_features and meta_features != csv_features:
            issue_list.append("metadata_features_not_equal_csv_feature_order")
        if formula.arg_names and meta_features and formula.arg_names != meta_features:
            issue_list.append("formula_args_not_equal_metadata_features")
        if len(probe_ranges) != len(meta_features or csv_features):
            issue_list.append("probe_range_count_mismatch")
        if any(not (math.isfinite(r[0]) and math.isfinite(r[1])) for r in probe_ranges):
            issue_list.append("probe_range_missing_or_nonfinite")
        if not formula.ok:
            issue_list.append(f"formula_parse_not_confirmed: {formula.error}")
        if extra_free_symbols:
            issue_list.append(f"formula_has_unmapped_free_symbols: {','.join(extra_free_symbols)}")

        var_map = {
            f"x{i}": name
            for i, name in enumerate(meta_features or csv_features or formula.arg_names)
        }
        reverse_map = {v: k for k, v in var_map.items()}
        judge_policy = {
            "ground_truth_source": "formula.py",
            "target_function": formula.target_function,
            "variable_order": meta_features or csv_features or formula.arg_names,
            "anonymous_variable_map": var_map,
            "physical_to_anonymous_map": reverse_map,
            "probe_samples": PROBE_SAMPLES_PER_DATASET,
            "probe_random_seed": PROBE_RANDOM_SEED,
            "probe_range_policy": "union(metadata train_range, metadata ood_range), fallback split min/max",
            "cas_equivalence_enabled": formula.ok and not formula.protected_ops,
            "numeric_equivalence_enabled": formula.ok,
            "protected_operator_policy": (
                "CAS result treated as advisory; numeric equivalence via formula.py semantics required"
                if formula.protected_ops
                else "standard elementary operators"
            ),
        }

        row = {
            "core50_index": int(item["core50_index"]),
            "dataset": str(item["dataset_name"]),
            "gid": f"g{int(item['core50_index']):04d}",
            "dataset_dir": str(item["dataset_dir"]),
            "family": str(item["family"]),
            "target_name": target_name,
            "metadata_target_name": meta_target,
            "feature_count": int(item["feature_count"]),
            "metadata_feature_names": _json_dumps(meta_features),
            "csv_feature_names": _json_dumps(csv_features),
            "formula_target_function": formula.target_function,
            "formula_arg_names": _json_dumps(formula.arg_names),
            "variable_map_x_to_feature": _json_dumps(var_map),
            "feature_to_x_map": _json_dumps(reverse_map),
            "formula_return_raw": formula.return_raw,
            "formula_return_substituted": formula.return_substituted,
            "gt_expression_x": formula.expression_x,
            "gt_expression_sympy": formula.sympy_sstr,
            "gt_formula_hash": formula.formula_hash,
            "gt_free_symbols": _json_dumps(formula.free_symbols),
            "gt_extra_free_symbols": _json_dumps(extra_free_symbols),
            "gt_unused_feature_symbols": _json_dumps(unused_feature_symbols),
            "gt_operator_names": _json_dumps(formula.operators),
            "gt_protected_ops": _json_dumps(formula.protected_ops),
            "local_constants": _json_dumps(formula.local_constants),
            "local_aliases": _json_dumps(formula.local_aliases),
            "probe_ranges": _json_dumps(probe_ranges),
            "probe_range_sources": _json_dumps(probe_sources),
            "probe_samples": PROBE_SAMPLES_PER_DATASET,
            "probe_random_seed": PROBE_RANDOM_SEED,
            "cas_equivalence_enabled": bool(judge_policy["cas_equivalence_enabled"]),
            "numeric_equivalence_enabled": bool(judge_policy["numeric_equivalence_enabled"]),
            "judge_policy": _json_dumps(judge_policy),
            "auto_confirmed": not issue_list,
            "issues": "; ".join(issue_list),
        }
        rows.append(row)
        for issue in issue_list:
            unresolved.append(
                {
                    "core50_index": row["core50_index"],
                    "dataset": row["dataset"],
                    "dataset_dir": row["dataset_dir"],
                    "issue": issue,
                    "needs_user_confirmation": True,
                    "recommendation": "确认公式语义、变量顺序或采样范围后再纳入 formal judge",
                }
            )

    params = pd.DataFrame(rows)
    unresolved_columns = [
        "core50_index",
        "dataset",
        "dataset_dir",
        "issue",
        "needs_user_confirmation",
        "recommendation",
    ]
    unresolved_df = pd.DataFrame(unresolved, columns=unresolved_columns)
    params.to_csv(outdir / "symf_formal_judge_parameters.csv", index=False)
    unresolved_df.to_csv(outdir / "symf_formal_judge_unresolved.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "core50_csv": str(CORE50_CSV.relative_to(REPO_ROOT)),
        "datasets": int(len(params)),
        "auto_confirmed": int(params["auto_confirmed"].sum()),
        "needs_review": int((~params["auto_confirmed"]).sum()),
        "formula_parse_ok": int(params["gt_expression_sympy"].notna().sum()),
        "protected_operator_datasets": int(params["gt_protected_ops"].ne("[]").sum()),
        "probe_samples_per_dataset": PROBE_SAMPLES_PER_DATASET,
        "probe_random_seed": PROBE_RANDOM_SEED,
        "unresolved_issue_counts": dict(unresolved_df["issue"].value_counts()) if not unresolved_df.empty else {},
        "deepseek_api_needed": False,
        "notes": [
            "未使用大模型 API；仅做本地 metadata/formula/csv 审计。",
            "含 protected ops 的数据集不建议只靠 CAS，后续 formal judge 必须以 formula.py 数值语义做 numeric equivalence fallback。",
        ],
    }
    (outdir / "symf_formal_judge_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SYM-F formal judge 参数补齐审计",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Datasets: `{summary['datasets']}`",
        f"- Auto confirmed: `{summary['auto_confirmed']}`",
        f"- Needs review: `{summary['needs_review']}`",
        f"- Formula parse ok: `{summary['formula_parse_ok']}`",
        f"- Protected-operator datasets: `{summary['protected_operator_datasets']}`",
        f"- Probe samples per dataset: `{PROBE_SAMPLES_PER_DATASET}`",
        f"- Probe random seed: `{PROBE_RANDOM_SEED}`",
        "- DeepSeek / LLM API needed: `false` for current audit.",
        "",
        "## 输出文件",
        "",
        "- `symf_formal_judge_parameters.csv`: 每个数据集的公式来源、变量映射、probe 范围和 judge policy。",
        "- `symf_formal_judge_unresolved.csv`: 需要人工确认的问题项。",
        "- `symf_formal_judge_config.json`: 机器可读摘要。",
        "",
        "## 当前结论",
        "",
    ]
    if unresolved_df.empty:
        lines.append("- 所有 Core-50 数据集的必要参数均已自动确认，可以进入 formal judge 实现。")
    else:
        lines.append("- 有部分数据集需要确认，优先查看 `symf_formal_judge_unresolved.csv`。")
        lines.append("")
        lines.append(unresolved_df.head(80).to_markdown(index=False))
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
