#!/usr/bin/env python3
"""生成 Core-50 正式 SYM-F 指标。

输入：
- `symf_formal_judge_parameters.csv`：上一阶段补齐的 GT、变量映射、probe 范围。
- `clean_final_runs_updated.csv`：3000 条 clean final run 状态与数值结果。
- `core50_result_expressions_updated.csv`：每条 run 的最终表达式与 result.json 路径。

输出：
- `symbolic_metrics_formal.csv`：run-level 正式符号指标。
- `symbolic_metrics_formal_algorithm_summary.csv`：算法级 SYM-F 汇总。
- `symbolic_metrics_formal_dataset_summary.csv`：dataset × algorithm 汇总。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

REPO_ROOT = Path(
    os.environ.get("SIM_REPO_ROOT")
    or Path(__file__).resolve().parents[1]
).resolve()
sys.path.insert(0, str(REPO_ROOT))

CORE50_ROOT = REPO_ROOT / "exp-planning/04.Core50正式全量评测"
HEXAGON_ARTIFACT_DIR = CORE50_ROOT / "analysis/hexagon_v1_with_artifacts_20260504"
PARAM_DIR = CORE50_ROOT / "analysis/symf_formal_judge_params_20260504"
DEFAULT_PARAMS_CSV = PARAM_DIR / "symf_formal_judge_parameters.csv"
DEFAULT_CLEAN_RUNS_CSV = HEXAGON_ARTIFACT_DIR / "clean_final_runs_updated.csv"
DEFAULT_EXPRESSIONS_CSV = HEXAGON_ARTIFACT_DIR / "core50_result_expressions_updated.csv"
DEFAULT_OUTDIR = CORE50_ROOT / "analysis/symf_formal_metrics_20260504"

NUMERIC_EQ_NMSE_THRESHOLD = 1e-10
NUMERIC_EQ_REL_THRESHOLD = 1e-8
NUMERIC_EQ_ABS_THRESHOLD = 1e-10
MIN_FINITE_PROBE_POINTS = 512
CAS_TIMEOUT_SECONDS = 2
TED_TIMEOUT_SECONDS = 2
CAS_MAX_CHARS = 350
CAS_MAX_OPS = 45
NUMERIC_MAX_CHARS = 1600
NUMERIC_MAX_OPS = 220
TREE_MAX_NODES = 450
PROVENANCE_SCHEMA_VERSION = 1
RUN_LEVEL_EXPRESSION_SOURCE = "run_level.expression_canonical"
RESULT_PATH_EXPRESSION_SOURCE = "result_path_then_run_level"


class TimeoutError(RuntimeError):
    pass


class time_limit:
    def __init__(self, seconds: int):
        self.seconds = int(seconds)
        self.old_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self.old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        return False

    @staticmethod
    def _handle_timeout(signum, frame):  # noqa: ARG004
        raise TimeoutError("operation timed out")


def json_loads(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_algorithm(name: Any) -> str:
    value = str(name or "").strip().lower()
    if value in {"qlattice_wrapper"}:
        return "qlattice"
    if value in {"imcts_wrapper"}:
        return "imcts"
    return value


def stable_gid(row: Any) -> str:
    value = row.get("gid") if hasattr(row, "get") else None
    if value is not None and not (
        isinstance(value, float) and math.isnan(value)
    ):
        text = str(value).strip()
        if text:
            return text
    dataset = row.get("dataset") if hasattr(row, "get") else None
    return str(dataset or "").strip()


def resolve_dataset_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "sim-datasets-data":
        home_candidate = Path.home() / path
        if home_candidate.exists():
            return home_candidate
    return REPO_ROOT / path


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_fingerprint(
    *,
    params_csv: Path,
    run_level_csv: Path,
    generator_script: Path,
    expression_source: str,
) -> dict[str, str]:
    fingerprint = {
        "params_sha256": file_sha256(params_csv),
        "run_level_sha256": file_sha256(run_level_csv),
        "generator_sha256": file_sha256(generator_script),
        "expression_source": str(expression_source),
    }
    params_header = pd.read_csv(params_csv, nrows=0)
    formula_columns = {"gid", "formula_source_sha256"}
    if formula_columns.issubset(params_header.columns):
        formula_sources = pd.read_csv(
            params_csv,
            usecols=sorted(formula_columns),
        )
        if formula_sources.isna().any(axis=None):
            raise ValueError(
                "params 中存在未冻结的 formula.py source"
            )
        records = sorted(
            (
                str(row["gid"]).strip(),
                str(row["formula_source_sha256"]).strip(),
            )
            for _, row in formula_sources.iterrows()
        )
        fingerprint["formula_sources_sha256"] = hashlib.sha256(
            json.dumps(
                records,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        fingerprint["formula_source_count"] = str(len(records))
    return fingerprint


def assert_source_fingerprint_unchanged(
    expected: dict[str, str],
    *,
    params_csv: Path,
    run_level_csv: Path,
    generator_script: Path,
    expression_source: str,
) -> None:
    actual = build_source_fingerprint(
        params_csv=params_csv,
        run_level_csv=run_level_csv,
        generator_script=generator_script,
        expression_source=expression_source,
    )
    if actual != expected:
        raise RuntimeError(
            "SYM-F source files 在运行期间发生变化: "
            f"before={expected}, after={actual}"
        )


def load_expected_runs_from_run_level(
    path: Path,
    *,
    algorithms: set[str] | None = None,
    expected_runs: int | None = None,
) -> pd.DataFrame:
    """读取统一 run-level CSV，并执行稳定运行身份校验。"""
    frame = pd.read_csv(path)
    required = {
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "status",
        "valid_output",
        "metric_complete",
        "result_path",
        "expression_canonical",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"run-level CSV 缺少字段: {sorted(missing)}")
    frame = frame.copy()
    frame["algorithm"] = frame["algorithm"].map(normalize_algorithm)
    frame["gid"] = frame["gid"].astype(str).str.strip()
    frame["seed"] = pd.to_numeric(
        frame["seed"],
        errors="raise",
    ).astype("Int64")
    frame["valid_output"] = frame["valid_output"].map(bool_value)
    frame["metric_complete"] = frame["metric_complete"].map(bool_value)
    if algorithms is not None:
        normalized = {normalize_algorithm(value) for value in algorithms}
        frame = frame[frame["algorithm"].isin(normalized)].copy()
    keys = ["algorithm", "gid", "seed"]
    duplicate_rows = int(frame.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"run-level CSV 存在重复 algorithm/gid/seed: {duplicate_rows}"
        )
    if expected_runs is not None and len(frame) != expected_runs:
        raise ValueError(
            f"run-level 行数错误: actual={len(frame)}, "
            f"expected={expected_runs}"
        )
    return frame.reset_index(drop=True)


def sympy_locals(n_features: int = 128) -> dict[str, Any]:
    out: dict[str, Any] = {f"x{i}": sp.Symbol(f"x{i}") for i in range(n_features)}
    out.update({f"c{i}": sp.Symbol(f"c{i}") for i in range(128)})
    out.update(
        {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "Abs": sp.Abs,
            "abs": sp.Abs,
            "pow": lambda x, y: x**y,
            "power": lambda x, y: x**y,
            "div": lambda x, y: x / y,
            "pi": sp.pi,
            "E": sp.E,
            "nan": sp.nan,
            "NaN": sp.nan,
        }
    )
    return out


def sanitize_expr(expr: Any, feature_to_x_map: dict[str, str] | None = None) -> str:
    text = str(expr or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.replace("numpy.", "").replace("np.", "").replace("math.", "")
    text = text.replace("^", "**")
    text = text.replace("arcsin", "asin").replace("arccos", "acos").replace("arctan", "atan")
    text = text.replace("power", "pow")
    text = text.replace("×", "*").replace("⋅", "*")
    text = text.replace("−", "-")
    # 常见变量写法归一。
    import re

    text = re.sub(r"\bx_(\d+)\b", lambda m: f"x{m.group(1)}", text)
    text = re.sub(r"\bx\[(\d+)\]", lambda m: f"x{m.group(1)}", text)
    if feature_to_x_map:
        for name, xname in sorted(feature_to_x_map.items(), key=lambda item: len(item[0]), reverse=True):
            if name == xname:
                continue
            text = re.sub(rf"\b{re.escape(name)}\b", xname, text)
    return text


def parse_sympy(expr: Any, *, feature_count: int, feature_to_x_map: dict[str, str] | None = None) -> tuple[sp.Expr | None, str | None, str]:
    cleaned = sanitize_expr(expr, feature_to_x_map)
    if not cleaned:
        return None, "empty_expression", cleaned
    try:
        parsed = sp.sympify(cleaned, locals=sympy_locals(max(128, feature_count + 8)))
        if parsed is sp.nan or getattr(parsed, "has", lambda *_: False)(sp.nan):
            return None, "nan_expression", cleaned
        return parsed, None, cleaned
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}", cleaned


def operator_set(expr: sp.Expr | None) -> set[str]:
    if expr is None:
        return set()
    ops: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if getattr(node, "is_Symbol", False) or getattr(node, "is_Number", False):
            continue
        name = getattr(getattr(node, "func", None), "__name__", "")
        if name:
            ops.add(name.lower())
    return ops


def variable_set(expr: sp.Expr | None) -> set[str]:
    if expr is None:
        return set()
    return {str(symbol) for symbol in expr.free_symbols if str(symbol).startswith("x")}


@dataclass(frozen=True)
class FastTreeNode:
    label: str
    children: tuple["FastTreeNode", ...]


def expr_to_fast_tree(expr: sp.Expr) -> FastTreeNode:
    if getattr(expr, "is_Number", False):
        return FastTreeNode("Const", ())
    if getattr(expr, "is_Symbol", False):
        return FastTreeNode(f"Symbol:{expr}", ())
    label = getattr(getattr(expr, "func", None), "__name__", type(expr).__name__)
    children = tuple(expr_to_fast_tree(arg) for arg in getattr(expr, "args", ()))
    if label in {"Add", "Mul"}:
        children = tuple(sorted(children, key=repr))
    return FastTreeNode(label, children)


@lru_cache(maxsize=None)
def fast_tree_size(node: FastTreeNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(fast_tree_size(child) for child in node.children)


@lru_cache(maxsize=None)
def fast_tree_edit_distance(a: FastTreeNode | None, b: FastTreeNode | None) -> int:
    if a is None:
        return fast_tree_size(b)
    if b is None:
        return fast_tree_size(a)

    label_cost = 0 if a.label == b.label else 1
    a_children = a.children
    b_children = b.children
    dp = [[0] * (len(b_children) + 1) for _ in range(len(a_children) + 1)]
    for i in range(1, len(a_children) + 1):
        dp[i][0] = dp[i - 1][0] + fast_tree_size(a_children[i - 1])
    for j in range(1, len(b_children) + 1):
        dp[0][j] = dp[0][j - 1] + fast_tree_size(b_children[j - 1])
    for i in range(1, len(a_children) + 1):
        for j in range(1, len(b_children) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + fast_tree_size(a_children[i - 1]),
                dp[i][j - 1] + fast_tree_size(b_children[j - 1]),
                dp[i - 1][j - 1] + fast_tree_edit_distance(a_children[i - 1], b_children[j - 1]),
            )
    return label_cost + dp[-1][-1]


def f1_score(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def safe_simplify_equiv(pred_expr: sp.Expr, gt_expr: sp.Expr) -> tuple[bool, str | None]:
    try:
        if len(str(pred_expr)) > CAS_MAX_CHARS or int(sp.count_ops(pred_expr)) > CAS_MAX_OPS:
            return False, "skipped_complex_expression"
        with time_limit(CAS_TIMEOUT_SECONDS):
            diff = sp.simplify(sp.trigsimp(pred_expr - gt_expr))
            if diff == 0:
                return True, None
            if getattr(diff, "is_zero", False) is True:
                return True, None
            return False, None
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def safe_tree_metrics(pred_expr: sp.Expr | None, gt_expr: sp.Expr | None) -> tuple[float, int | None, int | None, float | None, str | None]:
    if pred_expr is None or gt_expr is None:
        return 0.0, None, None, None, "parse_failed"
    try:
        with time_limit(TED_TIMEOUT_SECONDS):
            pred_tree = expr_to_fast_tree(pred_expr)
            gt_tree = expr_to_fast_tree(gt_expr)
            pred_size = fast_tree_size(pred_tree)
            true_size = fast_tree_size(gt_tree)
            if pred_size > TREE_MAX_NODES:
                return 0.0, None, true_size, None, f"skipped_large_tree:{pred_size}"
            distance = fast_tree_edit_distance(pred_tree, gt_tree)
        ned = min(1.0, float(distance) / float(true_size)) if true_size > 0 else 1.0
        return max(0.0, 1.0 - min(1.0, ned)), distance, true_size, ned, None
    except Exception as exc:
        return 0.0, None, None, None, f"{exc.__class__.__name__}: {exc}"


def load_formula_function(dataset_dir: Path, target_function: str | None, target_name: str | None) -> Callable[..., Any] | None:
    formula_path = dataset_dir / "formula.py"
    if not formula_path.exists():
        return None
    module_name = f"_symf_formula_{abs(hash(formula_path))}"
    spec = importlib.util.spec_from_file_location(module_name, formula_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 兼容少数 formula.py 直接使用 np/math 但文件头未显式 import 的历史数据。
    if not hasattr(module, "np"):
        setattr(module, "np", np)
    if not hasattr(module, "numpy"):
        setattr(module, "numpy", np)
    if not hasattr(module, "math"):
        setattr(module, "math", math)
    for name in [target_function, target_name, "target", "y"]:
        if name and callable(getattr(module, name, None)):
            return getattr(module, name)
    return None


def load_formula_function_from_frozen_source(
    source_b64: Any,
    source_sha256: Any,
    target_function: str | None,
    target_name: str | None,
) -> Callable[..., Any] | None:
    encoded = str(source_b64 or "").strip()
    expected_sha256 = str(source_sha256 or "").strip()
    if not encoded or not expected_sha256:
        return None
    source = base64.b64decode(encoded, validate=True)
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "frozen formula.py source sha256 mismatch"
        )
    namespace: dict[str, Any] = {
        "__name__": "_symf_frozen_formula",
        "np": np,
        "numpy": np,
        "math": math,
    }
    exec(  # noqa: S102 - 执行的是参数准备阶段冻结并哈希校验的本地公式。
        compile(source, "<frozen-formula.py>", "exec"),
        namespace,
    )
    for name in [target_function, target_name, "target", "y"]:
        value = namespace.get(name) if name else None
        if callable(value):
            return value
    return None


def generate_probe_samples(params: pd.Series) -> np.ndarray:
    ranges = json_loads(params.get("probe_ranges"), [])
    seed = int(params.get("probe_random_seed") or 20260504) + int(params.get("core50_index") or 0)
    rng = np.random.default_rng(seed)
    cols: list[np.ndarray] = []
    for lo, hi in ranges:
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("non-finite probe range")
        if lo == hi:
            eps = max(abs(lo) * 0.01, 1e-6)
            lo -= eps
            hi += eps
        cols.append(rng.uniform(lo, hi, size=int(params.get("probe_samples") or 4096)))
    if not cols:
        raise ValueError("empty probe ranges")
    return np.column_stack(cols)


def numpy_modules() -> list[Any]:
    return [
        {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "Abs": np.abs,
            "abs": np.abs,
            "pow": np.power,
            "power": np.power,
        },
        "numpy",
    ]


def evaluate_pred_expr(expr: sp.Expr, samples: np.ndarray, feature_count: int) -> np.ndarray:
    symbols = [sp.Symbol(f"x{i}") for i in range(feature_count)]
    func = sp.lambdify(symbols, expr, modules=numpy_modules())
    values = func(*[samples[:, i] for i in range(feature_count)])
    values = np.asarray(values)
    if np.iscomplexobj(values):
        imag = np.abs(np.imag(values))
        real = np.real(values)
        scale = np.maximum(1.0, np.abs(real))
        if np.any(np.isfinite(imag) & (imag > 1e-9 * scale)):
            raise ValueError("complex_prediction")
        values = real
    values = np.asarray(values, dtype=float)
    if values.shape == ():
        values = np.full(samples.shape[0], float(values))
    return values.reshape(-1)


def nmse_and_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None, float | None]:
    residual = y_pred - y_true
    mse = float(np.mean(np.square(residual)))
    centered = y_true - float(np.mean(y_true))
    denom = float(np.mean(np.square(centered)))
    if denom <= 1e-300 or not math.isfinite(denom):
        denom = float(np.mean(np.square(y_true)))
    nmse = mse / denom if denom > 1e-300 and math.isfinite(denom) else None
    max_abs = float(np.max(np.abs(residual))) if residual.size else None
    scale = max(float(np.max(np.abs(y_true))) if y_true.size else 0.0, 1.0)
    max_rel = max_abs / scale if max_abs is not None else None
    return nmse, max_abs, max_rel


def validate_frozen_formula_sources(params_df: pd.DataFrame) -> None:
    required_columns = {
        "formula_source_b64",
        "formula_source_sha256",
    }
    missing_columns = sorted(required_columns - set(params_df.columns))
    if missing_columns:
        raise ValueError(
            "正式 SYM-F 缺少冻结公式列: "
            + ", ".join(missing_columns)
        )
    for _, row in params_df.iterrows():
        gid = stable_gid(row)
        encoded = row.get("formula_source_b64")
        expected_sha256 = row.get("formula_source_sha256")
        if (
            encoded is None
            or pd.isna(encoded)
            or not str(encoded).strip()
        ):
            raise ValueError(f"{gid}: formula_source_b64 缺失")
        if (
            expected_sha256 is None
            or pd.isna(expected_sha256)
            or not str(expected_sha256).strip()
        ):
            raise ValueError(f"{gid}: formula_source_sha256 缺失")
        try:
            source = base64.b64decode(
                str(encoded).strip(),
                validate=True,
            )
        except Exception as exc:
            raise ValueError(
                f"{gid}: formula_source_b64 无法解码"
            ) from exc
        actual_sha256 = hashlib.sha256(source).hexdigest()
        if actual_sha256 != str(expected_sha256).strip():
            raise ValueError(f"{gid}: formula_source_sha256 不匹配")


def load_gt_probe_cache(
    params_df: pd.DataFrame,
    *,
    require_frozen_formula_source: bool = False,
) -> dict[str, dict[str, Any]]:
    if require_frozen_formula_source:
        validate_frozen_formula_sources(params_df)
    cache: dict[str, dict[str, Any]] = {}
    for _, row in params_df.iterrows():
        dataset_key = stable_gid(row)
        dataset_dir = resolve_dataset_path(row["dataset_dir"])
        feature_names = json_loads(row.get("metadata_feature_names"), [])
        feature_count = int(row["feature_count"])
        try:
            frozen_source = row.get("formula_source_b64")
            if frozen_source is not None and not pd.isna(frozen_source):
                func = load_formula_function_from_frozen_source(
                    frozen_source,
                    row.get("formula_source_sha256"),
                    row.get("formula_target_function"),
                    row.get("target_name"),
                )
            else:
                func = load_formula_function(
                    dataset_dir,
                    row.get("formula_target_function"),
                    row.get("target_name"),
                )
            if func is None:
                raise ValueError("formula_function_missing")
            samples = generate_probe_samples(row)
            arg_feature_indices = json_loads(
                row.get("formula_arg_feature_indices"),
                list(range(feature_count)),
            )
            if not isinstance(arg_feature_indices, list) or not all(
                isinstance(index, int)
                and 0 <= index < feature_count
                for index in arg_feature_indices
            ):
                raise ValueError("invalid_formula_arg_feature_indices")
            with np.errstate(all="ignore"):
                y_gt = func(
                    *[
                        samples[:, feature_index]
                        for feature_index in arg_feature_indices
                    ]
                )
            y_gt = np.asarray(y_gt, dtype=float)
            if y_gt.shape == ():
                y_gt = np.full(samples.shape[0], float(y_gt))
            cache[dataset_key] = {
                "samples": samples,
                "y_gt": y_gt.reshape(-1),
                "feature_count": feature_count,
                "feature_names": feature_names,
                "error": None,
            }
        except Exception as exc:
            cache[dataset_key] = {
                "error": f"{exc.__class__.__name__}: {exc}"
            }
    return cache


def numeric_equivalence(
    pred_expr: sp.Expr | None,
    dataset: str,
    probe_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if pred_expr is None:
        return {"numeric_equiv": False, "numeric_equiv_reason": "pred_parse_failed", "numeric_error": "pred_parse_failed"}
    probe = probe_cache.get(dataset) or {}
    if probe.get("error"):
        return {"numeric_equiv": False, "numeric_equiv_reason": "probe_unavailable", "numeric_error": probe.get("error")}
    try:
        samples = probe["samples"]
        y_gt = probe["y_gt"]
        feature_count = int(probe["feature_count"])
        with np.errstate(all="ignore"):
            y_pred = evaluate_pred_expr(pred_expr, samples, feature_count)
        if y_pred.shape[0] != y_gt.shape[0]:
            return {
                "numeric_equiv": False,
                "numeric_equiv_reason": "prediction_shape_mismatch",
                "numeric_error": "prediction_shape_mismatch",
            }
        finite = np.isfinite(y_gt) & np.isfinite(y_pred)
        finite_points = int(finite.sum())
        if finite_points < MIN_FINITE_PROBE_POINTS:
            return {
                "numeric_equiv": False,
                "numeric_equiv_reason": "too_few_finite_probe_points",
                "numeric_error": "too_few_finite_probe_points",
                "probe_points": int(y_gt.size),
                "finite_probe_points": finite_points,
                "finite_probe_rate": finite_points / max(1, int(y_gt.size)),
            }
        nmse, max_abs, max_rel = nmse_and_errors(y_gt[finite], y_pred[finite])
        numeric_equiv_reasons: list[str] = []
        if nmse is not None and nmse <= NUMERIC_EQ_NMSE_THRESHOLD:
            numeric_equiv_reasons.append("nmse")
        if max_rel is not None and max_rel <= NUMERIC_EQ_REL_THRESHOLD:
            numeric_equiv_reasons.append("max_rel")
        if max_abs is not None and max_abs <= NUMERIC_EQ_ABS_THRESHOLD:
            numeric_equiv_reasons.append("max_abs")
        numeric_equiv_flag = bool(numeric_equiv_reasons)
        return {
            "numeric_equiv": numeric_equiv_flag,
            "numeric_equiv_reason": "+".join(numeric_equiv_reasons) if numeric_equiv_reasons else "not_equivalent",
            "numeric_error": None,
            "probe_nmse": nmse,
            "probe_max_abs_error": max_abs,
            "probe_max_rel_error": max_rel,
            "probe_points": int(y_gt.size),
            "finite_probe_points": finite_points,
            "finite_probe_rate": finite_points / max(1, int(y_gt.size)),
        }
    except Exception as exc:
        return {
            "numeric_equiv": False,
            "numeric_equiv_reason": f"{exc.__class__.__name__}",
            "numeric_error": f"{exc.__class__.__name__}: {exc}",
        }


def should_run_numeric_equivalence(pred_expr: sp.Expr | None, pred_vars: set[str], gt_vars: set[str]) -> tuple[bool, str | None]:
    if pred_expr is None:
        return False, "pred_parse_failed"
    extra_vars = sorted(pred_vars - gt_vars)
    if extra_vars:
        return False, f"extra_pred_variables:{','.join(extra_vars)}"
    try:
        expr_text = str(pred_expr)
        op_count = int(sp.count_ops(pred_expr))
    except Exception:
        return False, "complexity_count_failed"
    if len(expr_text) > NUMERIC_MAX_CHARS:
        return False, f"skipped_numeric_long_expression:{len(expr_text)}"
    if op_count > NUMERIC_MAX_OPS:
        return False, f"skipped_numeric_high_op_count:{op_count}"
    return True, None


def expression_from_result_path(result_path: Any) -> tuple[str | None, str, str | None]:
    text = str(result_path or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, "csv_fallback", "missing_result_json"
    path = Path(text)
    if not path.is_file():
        return None, "csv_fallback", "missing_result_json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, "csv_fallback", f"{exc.__class__.__name__}: {exc}"
    artifact = payload.get("canonical_artifact") or {}
    if isinstance(artifact, dict):
        for key in ["instantiated_expression", "normalized_expression", "return_expression_source"]:
            value = artifact.get(key)
            if isinstance(value, str) and value.strip():
                return value, f"result_json.canonical_artifact.{key}", None
    equation = payload.get("equation")
    if isinstance(equation, str) and equation.strip():
        return equation, "result_json.equation", None
    return None, "csv_fallback", "no_expression_in_result_json"


def dedupe_expressions(expr_df: pd.DataFrame) -> pd.DataFrame:
    df = expr_df.copy()
    df["algorithm"] = df["algorithm"].map(normalize_algorithm)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["status_score"] = df["status_json"].astype(str).str.lower().eq("ok").astype(int)
    df["parse_score"] = df["expression_parse_ok"].astype(bool).astype(int)
    df["expr_score"] = df["expression_canonical"].fillna("").astype(str).str.len().gt(0).astype(int)
    df["path_score"] = df["result_path"].fillna("").astype(str).map(lambda x: int(Path(x).exists()))
    df = df.sort_values(["status_score", "parse_score", "expr_score", "path_score", "result_path"], kind="mergesort")
    return df.drop_duplicates(["algorithm", "gid", "dataset", "seed"], keep="last").drop(
        columns=["status_score", "parse_score", "expr_score", "path_score"]
    )


def build_expected_runs(clean_df: pd.DataFrame, expr_df: pd.DataFrame) -> pd.DataFrame:
    clean = clean_df.copy()
    clean["algorithm"] = clean["algorithm"].map(normalize_algorithm)
    clean["seed"] = pd.to_numeric(clean["seed"], errors="coerce").astype("Int64")
    expr = dedupe_expressions(expr_df)
    keep_cols = [
        "algorithm",
        "gid",
        "dataset",
        "seed",
        "status_json",
        "result_path",
        "experiment_dir",
        "expression_raw",
        "expression_canonical",
    ]
    return clean.merge(expr[keep_cols], on=["algorithm", "gid", "dataset", "seed"], how="left")


def run_formal_metrics(
    params_df: pd.DataFrame,
    expected_runs: pd.DataFrame,
    *,
    prefer_run_level_expression: bool = False,
    require_frozen_formula_source: bool = False,
) -> pd.DataFrame:
    params_by_gid = {
        stable_gid(row): row
        for _, row in params_df.iterrows()
    }
    probe_cache = load_gt_probe_cache(
        params_df,
        require_frozen_formula_source=require_frozen_formula_source,
    )
    rows: list[dict[str, Any]] = []

    total = len(expected_runs)
    for row_idx, (_, run) in enumerate(expected_runs.iterrows(), start=1):
        if row_idx % 100 == 0:
            print(f"[symf] processed {row_idx}/{total}", file=sys.stderr, flush=True)
        dataset = str(run["dataset"])
        gid = stable_gid(run)
        params = params_by_gid.get(gid)
        if params is None:
            rows.append({"algorithm": run.get("algorithm"), "gid": run.get("gid"), "dataset": dataset, "seed": run.get("seed"), "sym_f_formal": 0.0, "failure_reason": "missing_dataset_params"})
            continue

        feature_count = int(params["feature_count"])
        feature_to_x_map = json_loads(params.get("feature_to_x_map"), {})
        valid_for_symbolic = bool(run.get("valid_output")) and bool(run.get("metric_complete"))
        gt_expr, gt_parse_error, gt_cleaned = parse_sympy(
            params.get("gt_expression_x"),
            feature_count=feature_count,
            feature_to_x_map=None,
        )

        if prefer_run_level_expression:
            result_expr = None
            result_expr_error = None
            expr_source = RUN_LEVEL_EXPRESSION_SOURCE
            pred_source_expr = run.get("expression_canonical")
        else:
            result_expr, expr_source, result_expr_error = (
                expression_from_result_path(run.get("result_path"))
            )
            pred_source_expr = result_expr or run.get(
                "expression_canonical"
            )
        pred_expr, pred_parse_error, pred_cleaned = parse_sympy(
            pred_source_expr,
            feature_count=feature_count,
            feature_to_x_map=feature_to_x_map,
        )

        pred_vars = variable_set(pred_expr)
        gt_vars = variable_set(gt_expr)
        pred_ops = operator_set(pred_expr)
        gt_ops = operator_set(gt_expr)
        var_f1 = f1_score(pred_vars, gt_vars) if pred_expr is not None and gt_expr is not None else 0.0
        op_f1 = f1_score(pred_ops, gt_ops) if pred_expr is not None and gt_expr is not None else 0.0
        sof1 = 0.5 * var_f1 + 0.5 * op_f1

        cas_enabled = bool(params.get("cas_equivalence_enabled"))
        cas_equiv = False
        cas_error = None
        if valid_for_symbolic and pred_expr is not None and gt_expr is not None and cas_enabled:
            cas_equiv, cas_error = safe_simplify_equiv(pred_expr, gt_expr)

        run_numeric, numeric_skip_reason = should_run_numeric_equivalence(pred_expr, pred_vars, gt_vars)
        if valid_for_symbolic and run_numeric:
            numeric = numeric_equivalence(pred_expr, gid, probe_cache)
        else:
            numeric = {
                "numeric_equiv": False,
                "numeric_equiv_reason": numeric_skip_reason or "invalid_or_metric_incomplete_run",
                "numeric_error": numeric_skip_reason or "invalid_or_metric_incomplete_run",
            }
        numeric_equiv = bool(numeric.get("numeric_equiv"))
        equiv_final = bool(valid_for_symbolic and (cas_equiv or numeric_equiv))

        tree_similarity, tree_edit_distance, true_tree_size, ned, ted_error = safe_tree_metrics(pred_expr, gt_expr)
        sym_score = 1.0 if equiv_final else 0.3 * tree_similarity + 0.2 * sof1
        if not valid_for_symbolic or pred_expr is None or gt_expr is None:
            sym_score = 0.0

        failure_reasons: list[str] = []
        if not valid_for_symbolic:
            failure_reasons.append("invalid_or_metric_incomplete_run")
        if result_expr_error and not result_expr:
            failure_reasons.append(result_expr_error)
        if pred_parse_error:
            failure_reasons.append(f"pred_parse_failed:{pred_parse_error}")
        if gt_parse_error:
            failure_reasons.append(f"gt_parse_failed:{gt_parse_error}")
        if numeric.get("numeric_error"):
            failure_reasons.append(f"numeric:{numeric.get('numeric_error')}")
        if cas_error:
            failure_reasons.append(f"cas:{cas_error}")
        if ted_error:
            failure_reasons.append(f"ted:{ted_error}")

        rows.append(
            {
                "algorithm": run.get("algorithm"),
                "gid": run.get("gid"),
                "dataset": dataset,
                "seed": int(run["seed"]) if not pd.isna(run.get("seed")) else None,
                "status": run.get("status"),
                "valid_output": bool(run.get("valid_output")),
                "metric_complete": bool(run.get("metric_complete")),
                "valid_for_symbolic": valid_for_symbolic,
                "expr_source": expr_source,
                "result_path": run.get("result_path"),
                "pred_expression_raw": pred_source_expr,
                "pred_expression_cleaned": pred_cleaned,
                "gt_expression_x": gt_cleaned,
                "pred_parse_ok": pred_expr is not None,
                "gt_parse_ok": gt_expr is not None,
                "cas_equivalence_enabled": cas_enabled,
                "cas_equiv": cas_equiv,
                "numeric_equiv": numeric_equiv,
                "numeric_equiv_reason": numeric.get("numeric_equiv_reason"),
                "equiv_final": equiv_final,
                "probe_nmse": numeric.get("probe_nmse"),
                "probe_max_abs_error": numeric.get("probe_max_abs_error"),
                "probe_max_rel_error": numeric.get("probe_max_rel_error"),
                "probe_points": numeric.get("probe_points"),
                "finite_probe_points": numeric.get("finite_probe_points"),
                "finite_probe_rate": numeric.get("finite_probe_rate"),
                "tree_edit_distance": tree_edit_distance,
                "true_tree_size": true_tree_size,
                "ned": ned,
                "tree_similarity": tree_similarity,
                "pred_variables": json.dumps(sorted(pred_vars), ensure_ascii=False),
                "gt_variables": json.dumps(sorted(gt_vars), ensure_ascii=False),
                "var_f1": var_f1,
                "pred_operators": json.dumps(sorted(pred_ops), ensure_ascii=False),
                "gt_operators": json.dumps(sorted(gt_ops), ensure_ascii=False),
                "op_f1": op_f1,
                "sof1": sof1,
                "sym_f_formal": float(np.clip(sym_score, 0.0, 1.0)),
                "failure_reason": "; ".join(failure_reasons),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    outdir: Path,
    metrics: pd.DataFrame,
    params_df: pd.DataFrame,
    *,
    provenance: dict[str, Any] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "symbolic_metrics_formal.csv"
    metrics.to_csv(metrics_path, index=False)
    if provenance is not None:
        provenance_payload = {
            **provenance,
            "metrics_sha256": file_sha256(metrics_path),
        }
        (
            outdir / "symbolic_metrics_formal_provenance.json"
        ).write_text(
            json.dumps(
                provenance_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    dataset_summary = (
        metrics.groupby(["algorithm", "gid", "dataset"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            valid_for_symbolic_rate=("valid_for_symbolic", "mean"),
            pred_parse_rate=("pred_parse_ok", "mean"),
            cas_equiv_rate=("cas_equiv", "mean"),
            numeric_equiv_rate=("numeric_equiv", "mean"),
            equiv_final_rate=("equiv_final", "mean"),
            mean_tree_similarity=("tree_similarity", "mean"),
            mean_var_f1=("var_f1", "mean"),
            mean_op_f1=("op_f1", "mean"),
            sym_f_formal=("sym_f_formal", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(outdir / "symbolic_metrics_formal_dataset_summary.csv", index=False)

    alg_summary = (
        dataset_summary.groupby("algorithm", dropna=False)
        .agg(
            datasets=("gid", "nunique"),
            SYM_F_formal=("sym_f_formal", lambda x: 100 * float(np.mean(x))),
            exact_equiv_rate=("equiv_final_rate", "mean"),
            cas_equiv_rate=("cas_equiv_rate", "mean"),
            numeric_equiv_rate=("numeric_equiv_rate", "mean"),
            pred_parse_rate=("pred_parse_rate", "mean"),
            mean_tree_similarity=("mean_tree_similarity", "mean"),
            mean_var_f1=("mean_var_f1", "mean"),
            mean_op_f1=("mean_op_f1", "mean"),
        )
        .reset_index()
        .sort_values("SYM_F_formal", ascending=False)
    )
    alg_summary.to_csv(outdir / "symbolic_metrics_formal_algorithm_summary.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": int(len(metrics)),
        "datasets": int(metrics["gid"].nunique()),
        "algorithms": int(metrics["algorithm"].nunique()),
        "params_datasets": int(len(params_df)),
        "numeric_eq_nmse_threshold": NUMERIC_EQ_NMSE_THRESHOLD,
        "numeric_eq_rel_threshold": NUMERIC_EQ_REL_THRESHOLD,
        "numeric_eq_abs_threshold": NUMERIC_EQ_ABS_THRESHOLD,
        "min_finite_probe_points": MIN_FINITE_PROBE_POINTS,
        "cas_timeout_seconds": CAS_TIMEOUT_SECONDS,
        "ted_timeout_seconds": TED_TIMEOUT_SECONDS,
        "valid_for_symbolic_rate": float(metrics["valid_for_symbolic"].mean()),
        "pred_parse_rate": float(metrics["pred_parse_ok"].mean()),
        "equiv_final_rate": float(metrics["equiv_final"].mean()),
        "numeric_equiv_reason_counts": metrics["numeric_equiv_reason"].fillna("").value_counts().to_dict(),
        "algorithm_summary": alg_summary.to_dict(orient="records"),
    }
    (outdir / "symbolic_metrics_formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Core-50 SYM-F formal metrics",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Runs: `{summary['runs']}`",
        f"- Datasets: `{summary['datasets']}`",
        f"- Algorithms: `{summary['algorithms']}`",
        f"- Valid-for-symbolic rate: `{summary['valid_for_symbolic_rate']:.4f}`",
        f"- Prediction parse rate: `{summary['pred_parse_rate']:.4f}`",
        f"- Exact equivalence rate: `{summary['equiv_final_rate']:.4f}`",
        "",
        "## 评分口径",
        "",
        "- `equiv_final = cas_equiv or numeric_equiv`。",
        f"- `numeric_equiv` 阈值：`NMSE <= {NUMERIC_EQ_NMSE_THRESHOLD:g}` 或 `max_rel <= {NUMERIC_EQ_REL_THRESHOLD:g}` 或 `max_abs <= {NUMERIC_EQ_ABS_THRESHOLD:g}`。",
        "- `numeric_equiv_reason` 记录每条数值等价通过或跳过的具体原因，便于审计。",
        "- 等价公式 `sym_f_formal = 1.0`。",
        "- 非等价但可解析公式 `sym_f_formal = 0.3 * tree_similarity + 0.2 * ((var_f1 + op_f1) / 2)`。",
        "- invalid / metric incomplete / unparsable run 记 `0`。",
        "- `llmsr/drsr` 优先使用 `result.json` 中的 `instantiated_expression`，避免用 `c0/c1` skeleton 做数值等价。",
        "",
        "## Algorithm Summary",
        "",
        alg_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Core-50 formal SYM-F metrics")
    parser.add_argument("--params-csv", default=str(DEFAULT_PARAMS_CSV))
    parser.add_argument("--clean-runs-csv", default=str(DEFAULT_CLEAN_RUNS_CSV))
    parser.add_argument("--expressions-csv", default=str(DEFAULT_EXPRESSIONS_CSV))
    parser.add_argument(
        "--run-level-csv",
        help="直接读取统一 run-level CSV；提供后忽略 clean/expressions 两个输入",
    )
    parser.add_argument(
        "--algorithms",
        help="仅保留逗号分隔的算法集合，例如 fepysr,jaxsr,symbolfit",
    )
    parser.add_argument(
        "--prefer-run-level-expression",
        action="store_true",
        help="只使用已冻结 run-level CSV 中的 expression_canonical",
    )
    parser.add_argument(
        "--require-frozen-formula-source",
        action="store_true",
        help="要求 params 内所有 formula.py source 已内嵌且哈希正确",
    )
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    params_path = Path(args.params_csv)
    generator_path = Path(__file__).resolve()
    expression_source = (
        RUN_LEVEL_EXPRESSION_SOURCE
        if args.prefer_run_level_expression
        else RESULT_PATH_EXPRESSION_SOURCE
    )
    if args.prefer_run_level_expression and not args.run_level_csv:
        parser.error(
            "--prefer-run-level-expression requires --run-level-csv"
        )
    run_level_path = (
        Path(args.run_level_csv) if args.run_level_csv else None
    )
    source_fingerprint = (
        build_source_fingerprint(
            params_csv=params_path,
            run_level_csv=run_level_path,
            generator_script=generator_path,
            expression_source=expression_source,
        )
        if run_level_path is not None
        else None
    )

    params_df = pd.read_csv(params_path)
    if args.run_level_csv:
        algorithms = (
            {
                item.strip()
                for item in args.algorithms.split(",")
                if item.strip()
            }
            if args.algorithms
            else None
        )
        expected_runs = load_expected_runs_from_run_level(
            run_level_path,
            algorithms=algorithms,
            expected_runs=args.expected_runs,
        )
    else:
        clean_df = pd.read_csv(args.clean_runs_csv)
        expr_df = pd.read_csv(args.expressions_csv)
        expected_runs = build_expected_runs(clean_df, expr_df)
    metrics = run_formal_metrics(
        params_df,
        expected_runs,
        prefer_run_level_expression=args.prefer_run_level_expression,
        require_frozen_formula_source=(
            args.require_frozen_formula_source
        ),
    )
    if source_fingerprint is not None:
        assert_source_fingerprint_unchanged(
            source_fingerprint,
            params_csv=params_path,
            run_level_csv=run_level_path,
            generator_script=generator_path,
            expression_source=expression_source,
        )
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": "formal_metrics",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm_keys": sorted(
            metrics["algorithm"].map(normalize_algorithm).unique().tolist()
        ),
        "runs": int(len(metrics)),
        "source_fingerprint": source_fingerprint
        or {
            "params_sha256": file_sha256(params_path),
            "generator_sha256": file_sha256(generator_path),
            "expression_source": expression_source,
        },
    }
    write_outputs(
        Path(args.outdir),
        metrics,
        params_df,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "outdir": str(Path(args.outdir).resolve()),
                "runs": len(metrics),
                "datasets": metrics["dataset"].nunique(),
                "algorithms": metrics["algorithm"].nunique(),
                "sym_f_mean": float(metrics["sym_f_formal"].mean()),
                "equiv_rate": float(metrics["equiv_final"].mean()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
