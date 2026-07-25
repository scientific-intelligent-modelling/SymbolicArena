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

import argparse
import ast
import base64
import copy
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
    def __init__(self, replacements: dict[str, ast.AST]):
        self.replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        replacement = self.replacements.get(node.id)
        if replacement is not None and isinstance(node.ctx, ast.Load):
            return ast.copy_location(copy.deepcopy(replacement), node)
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


def _formula_arg_feature_indices(
    arg_names: list[str],
    variable_order: list[str],
) -> tuple[list[int], list[str]]:
    if not variable_order:
        return list(range(len(arg_names))), []
    indices: list[int | None] = [None] * len(arg_names)
    used: set[int] = set()
    for arg_index, name in enumerate(arg_names):
        matches = [
            index
            for index, feature in enumerate(variable_order)
            if feature == name and index not in used
        ]
        if matches:
            indices[arg_index] = matches[0]
            used.add(matches[0])

    unmatched_args = [
        index for index, feature_index in enumerate(indices)
        if feature_index is None
    ]
    unmatched_features = [
        index for index in range(len(variable_order))
        if index not in used
    ]
    issues: list[str] = []
    if len(unmatched_args) == len(unmatched_features):
        for arg_index, feature_index in zip(
            unmatched_args,
            unmatched_features,
            strict=True,
        ):
            indices[arg_index] = feature_index
            issues.append(
                "formula_arg_mapped_by_remaining_position:"
                f"{arg_names[arg_index]}->{variable_order[feature_index]}"
            )
    else:
        for arg_index in unmatched_args:
            fallback = min(arg_index, max(0, len(variable_order) - 1))
            indices[arg_index] = fallback
        if unmatched_args:
            issues.append("formula_arg_mapping_not_bijective")
    return [int(index) for index in indices if index is not None], issues


def _replace_args(
    expr: str,
    arg_names: list[str],
    variable_order: list[str] | None = None,
) -> str:
    out = expr
    indices, _ = _formula_arg_feature_indices(
        arg_names,
        variable_order or arg_names,
    )
    # 长变量名优先，避免把 `alpha` 中的 `a` 误替换。
    for arg_index, name in sorted(
        enumerate(arg_names),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        out = re.sub(
            rf"\b{re.escape(name)}\b",
            f"__symf_arg_{arg_index}__",
            out,
        )
    for arg_index, feature_index in enumerate(indices):
        out = out.replace(
            f"__symf_arg_{arg_index}__",
            f"x{feature_index}",
        )
    return out


def _operator_names(expr: str) -> list[str]:
    names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", expr or ""))
    return sorted(name for name in names if name not in {"where"})


def _extract_formula(
    path: Path,
    target_name: str,
    variable_order: list[str] | None = None,
) -> FormulaInfo:
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
            # 历史数据的 target_name 可能与函数名不完全一致。
            helper_names = {"div", "exp", "log", "sqrt", "sin", "cos", "tan", "abs"}
            non_helpers = [node for node in funcs if node.name not in helper_names]
            func = non_helpers[-1] if non_helpers else funcs[-1]
        if func is None:
            raise ValueError("formula.py 中未找到函数定义")

        arg_names = [arg.arg for arg in func.args.args]
        arg_name_set = set(arg_names)
        constants: dict[str, float] = dict(global_constants)
        aliases: dict[str, str] = {}
        replacements: dict[str, ast.AST] = {
            name: ast.Constant(value=value)
            for name, value in global_constants.items()
        }
        ret_node: ast.AST | None = None
        for stmt in func.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                alias = _array_alias(stmt.value, arg_name_set)
                if alias is not None:
                    aliases[name] = alias
                    replacements[name] = ast.Name(
                        id=alias,
                        ctx=ast.Load(),
                    )
                else:
                    value = _safe_eval_constant(stmt.value)
                    if value is not None:
                        constants[name] = value
                        replacements[name] = ast.Constant(value=value)
                    else:
                        substituted = _FormulaSubstituter(
                            replacements
                        ).visit(copy.deepcopy(stmt.value))
                        replacements[name] = ast.fix_missing_locations(
                            substituted
                        )
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                ret_node = stmt.value
                break
        if ret_node is None:
            raise ValueError("目标函数中未找到 return")

        return_raw = _clean_expr_text(ast.unparse(ret_node))
        ret_sub_ast = _FormulaSubstituter(replacements).visit(
            copy.deepcopy(ret_node)
        )
        ast.fix_missing_locations(ret_sub_ast)
        return_sub = _clean_expr_text(ast.unparse(ret_sub_ast))
        expression_x = _replace_args(
            return_sub,
            arg_names,
            variable_order,
        )
        operators = _operator_names(expression_x)
        protected_ops = sorted(set(operators) & PROTECTED_OPS)

        try:
            expr = sp.sympify(expression_x, locals=SYM_LOCALS)
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


def _catalog_value(item: pd.Series, *keys: str) -> Any:
    for key in keys:
        if key not in item:
            continue
        value = item.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        home_candidate = Path.home() / path
        if home_candidate.exists():
            return home_candidate
    return REPO_ROOT / path


def prepare_parameters(
    *,
    catalog_csv: Path,
    outdir: Path,
    expected_datasets: int | None = None,
    probe_samples: int = PROBE_SAMPLES_PER_DATASET,
    probe_random_seed: int = PROBE_RANDOM_SEED,
) -> dict[str, Any]:
    """从 Core50 或 Full664 catalog 生成统一 formal judge 参数。"""
    catalog_csv = catalog_csv.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(catalog_csv)
    if expected_datasets is not None and len(catalog) != expected_datasets:
        raise ValueError(
            f"catalog 数据集数错误: actual={len(catalog)}, "
            f"expected={expected_datasets}"
        )

    identities: list[tuple[str, str]] = []
    for _, item in catalog.iterrows():
        index_value = _catalog_value(item, "global_index", "core50_index")
        if index_value is None:
            raise ValueError("catalog 缺少 global_index/core50_index")
        global_index = int(index_value)
        gid = str(
            _catalog_value(item, "dataset_id", "gid")
            or f"g{global_index:04d}"
        )
        dataset_dir_value = _catalog_value(
            item,
            "dataset_dir",
            "dataset_rel",
        )
        if dataset_dir_value is None:
            raise ValueError(f"{gid} 缺少 dataset_dir")
        identities.append((gid, str(dataset_dir_value)))
    gids = [gid for gid, _ in identities]
    if len(gids) != len(set(gids)):
        raise ValueError("catalog 存在重复稳定 gid")
    dataset_dirs = [dataset_dir for _, dataset_dir in identities]
    if len(dataset_dirs) != len(set(dataset_dirs)):
        raise ValueError("catalog 存在重复 dataset_dir")

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for _, item in catalog.iterrows():
        global_index = int(
            _catalog_value(item, "global_index", "core50_index")
        )
        gid = str(
            _catalog_value(item, "dataset_id", "gid")
            or f"g{global_index:04d}"
        )
        dataset_name = str(
            _catalog_value(item, "dataset_name", "dataset") or gid
        )
        dataset_dir_value = str(
            _catalog_value(item, "dataset_dir", "dataset_rel")
        )
        dataset_dir = _resolve_repo_path(dataset_dir_value)
        metadata_value = _catalog_value(item, "metadata_yaml")
        metadata_path = (
            _resolve_repo_path(metadata_value)
            if metadata_value is not None
            else dataset_dir / "metadata.yaml"
        )
        formula_value = _catalog_value(item, "formula_py")
        formula_path = (
            _resolve_repo_path(formula_value)
            if formula_value is not None
            else dataset_dir / "formula.py"
        )
        formula_source = (
            formula_path.read_bytes()
            if formula_path.is_file()
            else None
        )
        formula_source_sha256 = (
            hashlib.sha256(formula_source).hexdigest()
            if formula_source is not None
            else None
        )
        formula_source_b64 = (
            base64.b64encode(formula_source).decode("ascii")
            if formula_source is not None
            else None
        )
        meta = _load_yaml(metadata_path)
        meta_features = _feature_names(meta)
        meta_target = _target_name(meta)
        target_name = str(
            _catalog_value(item, "target_name")
            or meta_target
            or ""
        )
        train_header = _csv_header(dataset_dir / "train.csv")
        csv_features = [col for col in train_header if col != target_name]
        declared_features = meta_features or csv_features
        formula = _extract_formula(
            formula_path,
            target_name,
            variable_order=declared_features or None,
        )
        features = meta_features or csv_features or formula.arg_names
        formula_arg_indices, formula_mapping_issues = (
            _formula_arg_feature_indices(
                formula.arg_names,
                features,
            )
        )
        feature_count_value = _catalog_value(item, "feature_count")
        feature_count = (
            int(feature_count_value)
            if feature_count_value is not None
            else len(features)
        )
        probe_ranges, probe_sources = _probe_ranges(
            meta,
            dataset_dir,
            features,
        )
        expected_symbols = {f"x{i}" for i in range(feature_count)}
        extra_free_symbols = sorted(set(formula.free_symbols) - expected_symbols)
        unused_feature_symbols = sorted(expected_symbols - set(formula.free_symbols))

        issue_list: list[str] = []
        if not metadata_path.exists():
            issue_list.append("missing_metadata_yaml")
        if not formula_path.exists():
            issue_list.append("missing_formula_py")
        if not target_name:
            issue_list.append("missing_target_name")
        if meta_target and meta_target != target_name:
            issue_list.append(
                f"target_mismatch: catalog={target_name}, "
                f"metadata={meta_target}"
            )
        if meta_features and csv_features and meta_features != csv_features:
            issue_list.append("metadata_features_not_equal_csv_feature_order")
        issue_list.extend(formula_mapping_issues)
        if len(probe_ranges) != feature_count:
            issue_list.append("probe_range_count_mismatch")
        if any(not (math.isfinite(r[0]) and math.isfinite(r[1])) for r in probe_ranges):
            issue_list.append("probe_range_missing_or_nonfinite")
        if not formula.ok:
            issue_list.append(f"formula_parse_not_confirmed: {formula.error}")
        if extra_free_symbols:
            issue_list.append(f"formula_has_unmapped_free_symbols: {','.join(extra_free_symbols)}")

        var_map = {
            f"x{i}": name
            for i, name in enumerate(features)
        }
        reverse_map = {v: k for k, v in var_map.items()}
        judge_policy = {
            "ground_truth_source": "formula.py source frozen in params",
            "target_function": formula.target_function,
            "variable_order": features,
            "formula_arg_feature_indices": formula_arg_indices,
            "anonymous_variable_map": var_map,
            "physical_to_anonymous_map": reverse_map,
            "probe_samples": probe_samples,
            "probe_random_seed": probe_random_seed,
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
            "core50_index": global_index,
            "global_index": global_index,
            "dataset": dataset_name,
            "gid": gid,
            "dataset_dir": dataset_dir_value,
            "dataset_rel": str(
                _catalog_value(item, "dataset_rel", "dataset_dir")
            ),
            "family": str(_catalog_value(item, "family") or ""),
            "subgroup": str(_catalog_value(item, "subgroup") or ""),
            "metadata_yaml": str(metadata_path),
            "formula_py": str(formula_path),
            "formula_source_sha256": formula_source_sha256,
            "formula_source_b64": formula_source_b64,
            "target_name": target_name,
            "metadata_target_name": meta_target,
            "feature_count": feature_count,
            "metadata_feature_names": _json_dumps(meta_features),
            "csv_feature_names": _json_dumps(csv_features),
            "formula_target_function": formula.target_function,
            "formula_arg_names": _json_dumps(formula.arg_names),
            "formula_arg_feature_indices": _json_dumps(
                formula_arg_indices
            ),
            "formula_arg_mapping_issues": _json_dumps(
                formula_mapping_issues
            ),
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
            "probe_samples": probe_samples,
            "probe_random_seed": probe_random_seed,
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
                    "gid": row["gid"],
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
        "gid",
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
        "catalog_csv": str(catalog_csv),
        "datasets": int(len(params)),
        "auto_confirmed": int(params["auto_confirmed"].sum()),
        "needs_review": int((~params["auto_confirmed"]).sum()),
        "formula_parse_ok": int(params["gt_expression_sympy"].notna().sum()),
        "protected_operator_datasets": int(params["gt_protected_ops"].ne("[]").sum()),
        "probe_samples_per_dataset": probe_samples,
        "probe_random_seed": probe_random_seed,
        "unresolved_issue_counts": (
            {
                str(issue): int(count)
                for issue, count in unresolved_df["issue"]
                .value_counts()
                .items()
            }
            if not unresolved_df.empty
            else {}
        ),
        "deepseek_api_needed": False,
        "notes": [
            "未使用大模型 API；仅做本地 metadata/formula/csv 审计。",
            "含 protected ops 的数据集不建议只靠 CAS；"
            "formal judge 必须使用 formula.py 数值语义兜底。",
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
        f"- Probe samples per dataset: `{probe_samples}`",
        f"- Probe random seed: `{probe_random_seed}`",
        "- DeepSeek / LLM API needed: `false` for current audit.",
        "",
        "## 输出文件",
        "",
        "- `symf_formal_judge_parameters.csv`: "
        "公式来源、变量映射、probe 范围和 judge policy。",
        "- `symf_formal_judge_unresolved.csv`: 需要人工确认的问题项。",
        "- `symf_formal_judge_config.json`: 机器可读摘要。",
        "",
        "## 当前结论",
        "",
    ]
    if unresolved_df.empty:
        lines.append(
            "- 所有数据集的必要参数均已自动确认，可以进入 formal judge。"
        )
    else:
        lines.append("- 有部分数据集需要确认，优先查看 `symf_formal_judge_unresolved.csv`。")
        lines.append("")
        lines.append(unresolved_df.head(80).to_markdown(index=False))
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-csv", type=Path, default=CORE50_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--expected-datasets", type=int)
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=PROBE_SAMPLES_PER_DATASET,
    )
    parser.add_argument(
        "--probe-random-seed",
        type=int,
        default=PROBE_RANDOM_SEED,
    )
    args = parser.parse_args()
    summary = prepare_parameters(
        catalog_csv=args.catalog_csv,
        outdir=args.outdir,
        expected_datasets=args.expected_datasets,
        probe_samples=args.probe_samples,
        probe_random_seed=args.probe_random_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
