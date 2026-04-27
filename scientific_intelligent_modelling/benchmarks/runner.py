"""统一 benchmark/result runner。"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import sympy as sp

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_archive import write_result_payload
from scientific_intelligent_modelling.benchmarks.result_artifacts import (
    safe_build_canonical_artifact,
    safe_export_canonical_artifact,
)
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


_HIDDEN_PARAM_KEYS = {"api_key", "apikey", "token", "password", "secret"}
_PROGRESS_DIRNAME = "progress"
_SNAPSHOT_CAPABLE_TOOLS = {
    "llmsr",
    "drsr",
    "pysr",
    "dso",
    "udsr",
    "pyoperon",
    "gplearn",
    "e2esr",
    "iMCTS",
    "tpsr",
    "QLattice",
    "ragsr",
}
_SNAPSHOT_CAPABLE_TOOL_KEYS = {tool.lower() for tool in _SNAPSHOT_CAPABLE_TOOLS}


def _is_snapshot_capable_tool(tool_name: str) -> bool:
    return str(tool_name).strip().lower() in _SNAPSHOT_CAPABLE_TOOL_KEYS


_RUNNER_TASK_IDENTITY_PARAM_KEYS = {
    "task_label",
    "task_global_index",
    "expected_dataset_rel",
    "expected_dataset_dir",
}
_NEUTRAL_SR_BACKGROUND = (
    "This is a symbolic regression task. "
    "Find a compact mathematical equation that predicts the target from the observed variables."
)


@dataclass
class DatasetSplit:
    name: str
    X: np.ndarray
    y: np.ndarray
    rows: int


@dataclass
class LoadedDataset:
    dataset_dir: Path
    dataset_name: str
    metadata: dict[str, Any]
    target_name: str
    feature_names: list[str]
    feature_descriptions: list[str | None]
    target_description: str | None
    train: DatasetSplit
    valid: DatasetSplit | None
    id_test: DatasetSplit | None
    ood_test: DatasetSplit | None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _slug_task_component(text: Any, *, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-")
    return (slug or "task")[:max_len]


def _normalize_dataset_identity_path(path: Any) -> str | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    text = text.replace("\\", "/")
    marker = "sim-datasets-data/"
    if marker in text:
        return marker + text.split(marker, 1)[1].strip("/")
    try:
        return str(Path(text).resolve())
    except Exception:
        return text.rstrip("/")


def _build_task_label(dataset: "LoadedDataset", task_global_index: Any = None, task_label: Any = None) -> str:
    if isinstance(task_label, str) and task_label.strip():
        return _slug_task_component(task_label)
    if task_global_index not in (None, ""):
        try:
            return _slug_task_component(f"g{int(task_global_index):04d}_{dataset.dataset_name}")
        except Exception:
            return _slug_task_component(f"g{task_global_index}_{dataset.dataset_name}")
    return _slug_task_component(dataset.dataset_name)


def _split_runner_task_identity_params(params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    rest = dict(params or {})
    identity: dict[str, Any] = {}
    for key in _RUNNER_TASK_IDENTITY_PARAM_KEYS:
        if key in rest:
            identity[key] = rest.pop(key)
    return rest, identity


def _dataset_identity_check(
    dataset: "LoadedDataset",
    *,
    expected_dataset_rel: Any = None,
    expected_dataset_dir: Any = None,
) -> dict[str, Any]:
    expected = expected_dataset_rel or expected_dataset_dir
    expected_norm = _normalize_dataset_identity_path(expected)
    actual_norm = _normalize_dataset_identity_path(dataset.dataset_dir)
    if expected_norm is None:
        status = "not_provided"
        match = None
    elif expected_norm == actual_norm:
        status = "match"
        match = True
    else:
        status = "mismatch"
        match = False
    return {
        "status": status,
        "match": match,
        "expected_dataset_rel": str(expected_dataset_rel) if expected_dataset_rel not in (None, "") else None,
        "expected_dataset_dir": str(expected_dataset_dir) if expected_dataset_dir not in (None, "") else None,
        "expected_normalized": expected_norm,
        "actual_dataset_dir": str(dataset.dataset_dir),
        "actual_normalized": actual_norm,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"metadata.yaml 格式非法: {path}")
    return data


def _load_split(dataset_dir: Path, filename: str, target_name: str) -> DatasetSplit | None:
    split_path = dataset_dir / filename
    split_name = filename[:-4] if filename.endswith(".csv") else filename
    if not split_path.exists():
        return None
    df = pd.read_csv(split_path)
    if df.empty:
        return DatasetSplit(
            name=split_name,
            X=np.empty((0, 0), dtype=float),
            y=np.empty((0,), dtype=float),
            rows=0,
        )
    if target_name not in df.columns:
        raise ValueError(f"{split_path} 中缺少目标列 {target_name}")
    X = df.drop(columns=[target_name]).values
    y = df[target_name].values
    return DatasetSplit(
        name=split_name,
        X=np.asarray(X),
        y=np.asarray(y).reshape(-1),
        rows=int(len(df)),
    )


def _build_background(dataset_meta: dict[str, Any], feature_names: list[str]) -> str:
    desc = str(dataset_meta.get("description") or "").strip()
    if desc:
        return desc

    features = dataset_meta.get("features") or []
    target = dataset_meta.get("target") or {}
    target_desc = str(target.get("description") or target.get("name") or "target").strip()
    feature_descs = []
    for idx, item in enumerate(features):
        if isinstance(item, dict):
            feature_descs.append(item.get("description") or item.get("name") or feature_names[idx])
        else:
            feature_descs.append(feature_names[idx])
    if not feature_descs:
        feature_descs = feature_names
    feature_text = ", ".join(str(x) for x in feature_descs if x)
    return f"Find the mathematical function skeleton that represents {target_desc}, given data on {feature_text}."


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def load_canonical_dataset(dataset_dir: str | Path) -> LoadedDataset:
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")

    meta_root = _load_yaml(dataset_path / "metadata.yaml")
    dataset_meta = meta_root.get("dataset", meta_root)
    if not isinstance(dataset_meta, dict):
        raise ValueError(f"metadata.yaml 中 dataset 字段格式非法: {dataset_path}")

    target_meta = dataset_meta.get("target") or {}
    target_name = target_meta.get("name")
    if not isinstance(target_name, str) or not target_name.strip():
        train_path = dataset_path / "train.csv"
        if not train_path.exists():
            raise ValueError(f"metadata.yaml 缺少 target.name，且不存在 train.csv: {dataset_path}")
        train_df = pd.read_csv(train_path, nrows=1)
        if train_df.empty:
            raise ValueError(f"无法从空 train.csv 推断目标列: {train_path}")
        target_name = str(train_df.columns[-1])

    train = _load_split(dataset_path, "train.csv", target_name)
    if train is None:
        raise FileNotFoundError(f"缺少 train.csv: {dataset_path}")

    feature_names = list(pd.read_csv(dataset_path / "train.csv", nrows=1).drop(columns=[target_name]).columns)
    features_meta = dataset_meta.get("features") or []
    feature_descriptions: list[str | None] = []
    for idx, feature_name in enumerate(feature_names):
        item = features_meta[idx] if idx < len(features_meta) else {}
        if isinstance(item, dict):
            feature_descriptions.append(item.get("description") or item.get("name") or feature_name)
        else:
            feature_descriptions.append(feature_name)

    target_description = None
    if isinstance(target_meta, dict):
        target_description = target_meta.get("description") or target_meta.get("name")

    return LoadedDataset(
        dataset_dir=dataset_path,
        dataset_name=dataset_path.name,
        metadata=dataset_meta,
        target_name=target_name,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        target_description=target_description,
        train=train,
        valid=_load_split(dataset_path, "valid.csv", target_name),
        id_test=_load_split(dataset_path, "id_test.csv", target_name),
        ood_test=_load_split(dataset_path, "ood_test.csv", target_name),
    )


def _evaluate_split(regressor: SymbolicRegressor, split: DatasetSplit | None) -> dict[str, float | None] | None:
    if split is None or split.rows == 0:
        return None
    pred = np.asarray(regressor.predict(split.X)).reshape(-1)
    metrics = regression_metrics(split.y, pred, acc_threshold=0.1)
    return {
        "rmse": _safe_float(metrics["rmse"]),
        "r2": _safe_float(metrics["r2"]),
        "nmse": _safe_float(metrics["nmse"]),
        "acc_0_1": _safe_float(metrics["acc_tau"]),
    }


def _evaluate_prediction(split: DatasetSplit | None, pred: np.ndarray | None) -> dict[str, float | None] | None:
    if split is None or split.rows == 0 or pred is None:
        return None
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    metrics = regression_metrics(split.y, pred_arr, acc_threshold=0.1)
    return {
        "rmse": _safe_float(metrics["rmse"]),
        "r2": _safe_float(metrics["r2"]),
        "nmse": _safe_float(metrics["nmse"]),
        "acc_0_1": _safe_float(metrics["acc_tau"]),
    }


def _split_metrics_are_usable(split: DatasetSplit | None, metrics: dict[str, Any] | None) -> bool:
    if split is None or split.rows == 0:
        return True
    if not isinstance(metrics, dict):
        return False
    # NMSE 是后续 benchmark 排名最依赖的字段；它有限即可视为该 split 可评估。
    return _safe_float(metrics.get("nmse")) is not None


def _recovered_metrics_are_usable(
    dataset: LoadedDataset,
    valid_metrics: dict[str, Any] | None,
    id_metrics: dict[str, Any] | None,
    ood_metrics: dict[str, Any] | None,
) -> bool:
    return (
        _split_metrics_are_usable(dataset.valid, valid_metrics)
        and _split_metrics_are_usable(dataset.id_test, id_metrics)
        and _split_metrics_are_usable(dataset.ood_test, ood_metrics)
    )


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if str(key).lower() in _HIDDEN_PARAM_KEYS:
            continue
        sanitized[key] = value
    return sanitized


def _sympy_locals() -> dict[str, Any]:
    locals_map: dict[str, Any] = {
        "Abs": sp.Abs,
        "Max": sp.Max,
        "Min": sp.Min,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    for i in range(256):
        locals_map[f"x{i}"] = sp.Symbol(f"x{i}")
    return locals_map


_GPLEARN_TOKEN_RE = re.compile(
    r"\s*("
    r"[A-Za-z_]\w*"
    r"|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"|[(),]"
    r")"
)


class _GPLearnPrefixParser:
    """轻量解析 gplearn prefix 表达式，避开 Python AST 括号深度限制。"""

    def __init__(self, text: str):
        self.text = text
        self.tokens = self._tokenize(text)
        self.pos = 0

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            if not text[pos:].strip():
                break
            match = _GPLEARN_TOKEN_RE.match(text, pos)
            if not match:
                raise ValueError(f"无法解析 gplearn token: {text[pos:pos + 40]!r}")
            tokens.append(match.group(1))
            pos = match.end()
        return tokens

    def parse(self) -> Any:
        if not self.tokens:
            raise ValueError("空 gplearn 表达式")
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"gplearn 表达式存在未消费 token: {self.tokens[self.pos]!r}")
        return node

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("gplearn 表达式意外结束")
        if expected is not None and token != expected:
            raise ValueError(f"gplearn 表达式期望 {expected!r}，实际 {token!r}")
        self.pos += 1
        return token

    def _parse_expr(self) -> Any:
        token = self._consume()
        next_token = self._peek()
        if re.fullmatch(r"[A-Za-z_]\w*", token) and next_token == "(":
            self._consume("(")
            args: list[Any] = []
            if self._peek() != ")":
                while True:
                    args.append(self._parse_expr())
                    if self._peek() == ",":
                        self._consume(",")
                        continue
                    break
            self._consume(")")
            return ("call", token, args)
        if re.fullmatch(r"[Xx]\d+", token):
            return ("var", int(token[1:]))
        try:
            return ("const", float(token))
        except Exception as exc:
            raise ValueError(f"不支持的 gplearn 叶子节点: {token!r}") from exc


def _broadcast_gplearn_value(value: Any, rows: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(rows, float(arr), dtype=float)
    return np.broadcast_to(arr, (rows,)).astype(float)


def _eval_gplearn_prefix_node(node: Any, X_arr: np.ndarray) -> Any:
    kind = node[0]
    if kind == "const":
        return float(node[1])
    if kind == "var":
        idx = int(node[1])
        if idx >= X_arr.shape[1]:
            raise ValueError(f"表达式变量索引越界: x{idx}, 输入维度={X_arr.shape[1]}")
        return X_arr[:, idx]
    if kind != "call":
        raise ValueError(f"不支持的 gplearn 节点类型: {kind!r}")

    op = str(node[1]).lower()
    values = [_eval_gplearn_prefix_node(arg, X_arr) for arg in node[2]]

    with np.errstate(all="ignore"):
        if op == "add" and len(values) == 2:
            return values[0] + values[1]
        if op == "sub" and len(values) == 2:
            return values[0] - values[1]
        if op == "mul" and len(values) == 2:
            return values[0] * values[1]
        if op == "div" and len(values) == 2:
            denominator = values[1]
            return np.where(np.abs(denominator) > 0.001, np.divide(values[0], denominator), 1.0)
        if op == "sqrt" and len(values) == 1:
            return np.sqrt(np.abs(values[0]))
        if op == "log" and len(values) == 1:
            value = values[0]
            return np.where(np.abs(value) > 0.001, np.log(np.abs(value)), 0.0)
        if op == "inv" and len(values) == 1:
            value = values[0]
            return np.where(np.abs(value) > 0.001, np.divide(1.0, value), 0.0)
        if op == "abs" and len(values) == 1:
            return np.abs(values[0])
        if op == "neg" and len(values) == 1:
            return -values[0]
        if op == "sin" and len(values) == 1:
            return np.sin(values[0])
        if op == "cos" and len(values) == 1:
            return np.cos(values[0])
        if op == "tan" and len(values) == 1:
            return np.tan(values[0])
        if op == "exp" and len(values) == 1:
            return np.exp(values[0])
        if op in {"sig", "sigmoid"} and len(values) == 1:
            return 1.0 / (1.0 + np.exp(-values[0]))
        if op == "pow" and len(values) == 2:
            return np.power(values[0], values[1])
        if op == "max" and len(values) == 2:
            return np.maximum(values[0], values[1])
        if op == "min" and len(values) == 2:
            return np.minimum(values[0], values[1])

    raise ValueError(f"不支持的 gplearn 算子或参数个数: {op}/{len(values)}")


def _predict_gplearn_prefix_expression(raw_equation: str, X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("gplearn prefix 预测要求二维输入")
    required_recursion_limit = min(50000, max(10000, str(raw_equation).count("(") + 1000))
    if sys.getrecursionlimit() < required_recursion_limit:
        sys.setrecursionlimit(required_recursion_limit)
    node = _GPLearnPrefixParser(str(raw_equation)).parse()
    pred = _eval_gplearn_prefix_node(node, X_arr)
    return _broadcast_gplearn_value(pred, X_arr.shape[0]).reshape(-1)


def _predict_from_canonical_artifact(artifact: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """基于统一工件中的代值表达式做轻量预测。

    这里只服务 runner 的中间最优快照，不依赖具体算法 wrapper 或子进程环境。
    """
    if str(artifact.get("tool_name") or "").strip().lower() == "gplearn":
        raw_equation = artifact.get("raw_equation")
        if isinstance(raw_equation, str) and raw_equation.strip():
            try:
                return _predict_gplearn_prefix_expression(raw_equation, X)
            except Exception:
                if artifact.get("raw_equation_kind") == "prefix_expression":
                    raise

    expr_text = (
        artifact.get("instantiated_expression")
        or artifact.get("normalized_expression")
        or artifact.get("return_expression_source")
    )
    if not isinstance(expr_text, str) or not expr_text.strip():
        raise ValueError("canonical_artifact 中缺少可执行表达式")

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("中间快照预测要求二维输入")

    expr = sp.sympify(expr_text, locals=_sympy_locals())
    free_symbols = sorted(
        list(expr.free_symbols),
        key=lambda sym: (
            0,
            int(str(sym)[1:]),
        )
        if str(sym).startswith("x") and str(sym)[1:].isdigit()
        else (1, str(sym)),
    )

    if not free_symbols:
        value = float(expr)
        return np.full(X_arr.shape[0], value, dtype=float)

    args = []
    for sym in free_symbols:
        name = str(sym)
        if not name.startswith("x") or not name[1:].isdigit():
            raise ValueError(f"表达式包含非标准变量: {name}")
        idx = int(name[1:])
        if idx >= X_arr.shape[1]:
            raise ValueError(f"表达式变量索引越界: {name}, 输入维度={X_arr.shape[1]}")
        args.append(X_arr[:, idx])

    fn = sp.lambdify(free_symbols, expr, modules="numpy")
    pred = fn(*args)
    pred_arr = np.asarray(pred, dtype=float)
    if pred_arr.ndim == 0:
        pred_arr = np.full(X_arr.shape[0], float(pred_arr), dtype=float)
    else:
        pred_arr = np.broadcast_to(pred_arr, (X_arr.shape[0],)).astype(float)
    return pred_arr.reshape(-1)


def _is_equation_function_text(func_source: Any) -> bool:
    if not isinstance(func_source, str) or not func_source.strip():
        return False
    try:
        tree = ast.parse(func_source)
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            body = list(node.body)
            while (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None
    return False


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_llmsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    samples_dir = Path(experiment_dir) / "samples"
    if not samples_dir.is_dir():
        return None
    candidates = sorted(samples_dir.glob("top01_*.json")) or sorted(samples_dir.glob("top*.json"))
    best_key = None
    best_item = None
    for path in candidates:
        item = _read_json_file(path)
        if not item:
            continue
        func = item.get("function")
        if not _is_equation_function_text(func):
            continue
        key_val = None
        nmse = item.get("nmse")
        mse = item.get("mse")
        score = item.get("score")
        if isinstance(nmse, (int, float)):
            key_val = float(nmse)
        elif isinstance(mse, (int, float)):
            key_val = float(mse)
        elif isinstance(score, (int, float)):
            key_val = -float(score)
        if key_val is None:
            continue
        if best_key is None or key_val < best_key:
            best_key = key_val
            best_item = item
    return best_item


def _extract_drsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = []
    candidate_paths.extend(sorted((base_dir / "samples").glob("top*.json")))
    candidate_paths.extend(sorted((base_dir / "best_history").glob("best_sample_*.json")))
    candidate_paths.extend(sorted((base_dir / "samples").glob("samples_*.json")))
    candidates: list[dict[str, Any]] = []
    best_key = None
    best_item = None
    for path in candidate_paths:
        item = _read_json_file(path)
        if not item:
            continue
        func = item.get("function")
        if not _is_equation_function_text(func):
            continue
        candidates.append(item)
        score = item.get("score")
        if not isinstance(score, (int, float)):
            continue
        key_val = -float(score)
        if best_key is None or key_val < best_key:
            best_key = key_val
            best_item = item
    if best_item is None:
        return None
    return _with_drsr_candidate_params(best_item, candidates)


def _candidate_parameter_values(candidate: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(candidate, dict):
        return None
    for key in ("params", "fitted_params", "parameter_values"):
        values = candidate.get(key)
        if isinstance(values, list):
            try:
                return [float(v) for v in values]
            except Exception:
                continue
    return None


def _with_drsr_candidate_params(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """确保 DRSR best candidate 携带可用于 canonical artifact 的参数。

    历史结果中出现过 best candidate 的公式被写入 `result.json`，但对应 `params`
    没有随 canonical artifact 带出的问题。这里按 function 精确匹配同目录其它
    best/sample 文件补齐参数，避免 timeout 恢复时留下未实例化的 c0/c1 常数。
    """
    if _candidate_parameter_values(candidate) is not None:
        out = dict(candidate)
        out["params"] = _candidate_parameter_values(candidate)
        return out

    func = candidate.get("function")
    if not isinstance(func, str) or not func.strip():
        return candidate
    for item in candidates:
        if item is candidate:
            continue
        if item.get("function") != func:
            continue
        params = _candidate_parameter_values(item)
        if params is None:
            continue
        out = dict(candidate)
        out["params"] = params
        out["params_source"] = "matched_drsr_candidate"
        return out
    return candidate


def _extract_pysr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = [base_dir / "hall_of_fame.csv", base_dir / "hall_of_fame.csv.bak"]
    best_loss = None
    best_item = None
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "Equation" not in df.columns or "Loss" not in df.columns:
            continue
        for _, row in df.iterrows():
            equation = row.get("Equation")
            loss = row.get("Loss")
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                loss_val = float(loss)
            except Exception:
                continue
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_item = {
                    "equation": equation,
                    "loss": loss_val,
                    "complexity": row.get("Complexity"),
                }
    return best_item


def _extract_dso_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    state_item = _read_json_file(base_dir / ".dso_current_best.json")
    if state_item:
        equation = state_item.get("equation")
        if isinstance(equation, str) and equation.strip():
            return state_item
    candidate_paths = sorted(base_dir.glob("*_hof.csv"))
    best_key = None
    best_item = None
    for path in candidate_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        expr_col = None
        for name in ("expression", "Equation", "equation", "sympy_format"):
            if name in df.columns:
                expr_col = name
                break
        score_col = None
        for name in ("r", "score", "reward"):
            if name in df.columns:
                score_col = name
                break
        if expr_col is None or score_col is None:
            continue
        for _, row in df.iterrows():
            equation = row.get(expr_col)
            score = row.get(score_col)
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                score_val = float(score)
            except Exception:
                continue
            # DSO 的 reward 越大越好，这里统一转成“越小越优”的排序键。
            key_val = -score_val
            if best_key is None or key_val < best_key:
                best_key = key_val
                best_item = {
                    "equation": equation,
                    "score": score_val,
                    "complexity": row.get("complexity"),
                }
    return best_item


def _extract_udsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    state_item = _read_json_file(base_dir / ".udsr_current_best.json")
    if state_item:
        equation = state_item.get("equation")
        if isinstance(equation, str) and equation.strip():
            return state_item
    return _extract_dso_periodic_candidate(experiment_dir)


def _extract_pyoperon_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".pyoperon_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_gplearn_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".gplearn_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_e2esr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".e2esr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_imcts_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".imcts_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_tpsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".tpsr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_qlattice_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".qlattice_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_ragsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".ragsr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_periodic_candidate(tool_name: str, experiment_dir: str | Path) -> dict[str, Any] | None:
    tool = str(tool_name).strip().lower()
    if tool == "llmsr":
        return _extract_llmsr_periodic_candidate(experiment_dir)
    if tool == "drsr":
        return _extract_drsr_periodic_candidate(experiment_dir)
    if tool == "pysr":
        return _extract_pysr_periodic_candidate(experiment_dir)
    if tool == "dso":
        return _extract_dso_periodic_candidate(experiment_dir)
    if tool == "udsr":
        return _extract_udsr_periodic_candidate(experiment_dir)
    if tool == "pyoperon":
        return _extract_pyoperon_periodic_candidate(experiment_dir)
    if tool == "gplearn":
        return _extract_gplearn_periodic_candidate(experiment_dir)
    if tool == "e2esr":
        return _extract_e2esr_periodic_candidate(experiment_dir)
    if tool == "imcts":
        return _extract_imcts_periodic_candidate(experiment_dir)
    if tool == "tpsr":
        return _extract_tpsr_periodic_candidate(experiment_dir)
    if tool == "qlattice":
        return _extract_qlattice_periodic_candidate(experiment_dir)
    if tool == "ragsr":
        return _extract_ragsr_periodic_candidate(experiment_dir)
    return None


def _progress_snapshot_filename(payload: dict[str, Any]) -> str:
    elapsed_seconds = payload.get("elapsed_seconds")
    try:
        elapsed_minutes = max(0, int(round(float(elapsed_seconds) / 60.0)))
    except Exception:
        elapsed_minutes = 0
    return f"minute_{elapsed_minutes:04d}.json"


def _write_progress_payload(
    payload: dict[str, Any],
    *,
    primary_dir: str | Path,
    experiment_dir: str | Path | None = None,
) -> list[Path]:
    filename = _progress_snapshot_filename(payload)
    paths: list[Path] = [Path(primary_dir).resolve() / filename]
    if experiment_dir:
        paths.append(Path(experiment_dir).resolve() / _PROGRESS_DIRNAME / filename)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for path in unique_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return unique_paths


def _resolve_progress_snapshot_interval_seconds(tool_name: str, params: dict[str, Any]) -> int | None:
    raw = params.pop("progress_snapshot_interval_seconds", None)
    if raw is None:
        if _is_snapshot_capable_tool(tool_name):
            return 60
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _build_periodic_snapshot_payload(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    experiment_dir: str | Path,
    checkpoint_index: int,
) -> dict[str, Any] | None:
    candidate = _extract_periodic_candidate(tool_name, experiment_dir)
    if not candidate:
        return None

    parameter_values = _candidate_parameter_values(candidate)
    canonical_artifact, canonical_artifact_error = safe_build_canonical_artifact(
        tool_name=tool_name,
        equation=candidate.get("function") if "function" in candidate else candidate.get("equation"),
        expected_n_features=len(dataset.feature_names),
        parameter_values=parameter_values,
    )

    train_metrics = None
    valid_metrics = None
    id_metrics = None
    ood_metrics = None
    error = None
    if canonical_artifact is not None:
        try:
            train_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.train.X) if dataset.train else None
            valid_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.valid.X) if dataset.valid else None
            id_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.id_test.X) if dataset.id_test else None
            ood_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.ood_test.X) if dataset.ood_test else None
            train_metrics = _evaluate_prediction(dataset.train, train_pred)
            valid_metrics = _evaluate_prediction(dataset.valid, valid_pred)
            id_metrics = _evaluate_prediction(dataset.id_test, id_pred)
            ood_metrics = _evaluate_prediction(dataset.ood_test, ood_pred)
        except Exception as exc:
            error = repr(exc)
    else:
        error = canonical_artifact_error

    payload = build_result_payload(
        tool_name=tool_name,
        dataset=dataset,
        params=params,
        seed=seed,
        started_at=started_at,
        status="ok" if error is None else "error",
        error=error,
        equation=str(candidate.get("function") or candidate.get("equation") or ""),
        equation_count=1,
        canonical_artifact=canonical_artifact,
        canonical_artifact_error=canonical_artifact_error,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        experiment_dir=str(experiment_dir),
    )
    payload["record_type"] = "periodic_best"
    payload["checkpoint_index"] = int(checkpoint_index)
    payload["elapsed_seconds"] = round(time.time() - started_at, 3)
    payload["elapsed_minutes"] = max(0, int(round(payload["elapsed_seconds"] / 60.0)))
    payload["source_iteration"] = candidate.get("iteration")
    payload["source_sample_order"] = candidate.get("sample_order")
    payload["source_score"] = candidate.get("score")
    payload["source_loss"] = candidate.get("loss")
    payload["source_complexity"] = candidate.get("complexity")
    return payload


def _recover_timeout_payload_from_candidate(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    experiment_dir: str | Path | None,
) -> dict[str, Any] | None:
    """在训练超时后，直接从实验目录里的已落盘候选恢复可分析结果。

    对 `pysr` 来说，这里会优先利用 `hall_of_fame.csv/.bak`；
    对其他已接入周期快照的算法，则复用各自的候选提取逻辑。
    """
    if not experiment_dir:
        return None

    candidate = _extract_periodic_candidate(tool_name, experiment_dir)
    if not candidate:
        return _recover_timeout_payload_from_progress_snapshots(
            tool_name=tool_name,
            dataset=dataset,
            experiment_dir=experiment_dir,
        )

    raw_equation = candidate.get("function") if "function" in candidate else candidate.get("equation")
    equation = str(raw_equation or "").strip()
    if not equation:
        return _recover_timeout_payload_from_progress_snapshots(
            tool_name=tool_name,
            dataset=dataset,
            experiment_dir=experiment_dir,
        )

    parameter_values = _candidate_parameter_values(candidate)
    canonical_artifact, canonical_artifact_error = safe_build_canonical_artifact(
        tool_name=tool_name,
        equation=equation,
        expected_n_features=len(dataset.feature_names),
        parameter_values=parameter_values,
    )

    train_metrics = None
    valid_metrics = None
    id_metrics = None
    ood_metrics = None
    if canonical_artifact is not None:
        try:
            train_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.train.X) if dataset.train else None
            valid_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.valid.X) if dataset.valid else None
            id_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.id_test.X) if dataset.id_test else None
            ood_pred = _predict_from_canonical_artifact(canonical_artifact, dataset.ood_test.X) if dataset.ood_test else None
            train_metrics = _evaluate_prediction(dataset.train, train_pred)
            valid_metrics = _evaluate_prediction(dataset.valid, valid_pred)
            id_metrics = _evaluate_prediction(dataset.id_test, id_pred)
            ood_metrics = _evaluate_prediction(dataset.ood_test, ood_pred)
        except Exception as exc:
            if canonical_artifact_error is None:
                canonical_artifact_error = repr(exc)

    if not _recovered_metrics_are_usable(dataset, valid_metrics, id_metrics, ood_metrics):
        snapshot_payload = _recover_timeout_payload_from_progress_snapshots(
            tool_name=tool_name,
            dataset=dataset,
            experiment_dir=experiment_dir,
        )
        if snapshot_payload is not None:
            return snapshot_payload

    return {
        "equation": equation,
        "equation_count": 1,
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
    }


def _recover_timeout_payload_from_progress_snapshots(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    experiment_dir: str | Path,
) -> dict[str, Any] | None:
    """从最近的可评估分钟级快照回退恢复超时结果。

    有些工具的 current-best 会在最后一分钟更新为数值不稳定表达式，导致最终
    `result.json` 有公式但没有有限指标。此时应优先保留最近一个可有限评估的
    best-so-far 快照，而不是把整条 run 降级成无效输出。
    """
    progress_dir = Path(experiment_dir) / _PROGRESS_DIRNAME
    if not progress_dir.is_dir():
        return None
    expected_tool = str(tool_name).strip().lower()
    for path in sorted(progress_dir.glob("minute_*.json"), reverse=True):
        item = _read_json_file(path)
        if not item:
            continue
        if str(item.get("tool") or "").strip().lower() != expected_tool:
            continue
        equation = str(item.get("equation") or "").strip()
        artifact = item.get("canonical_artifact")
        if not equation or not isinstance(artifact, dict):
            continue
        valid_metrics = item.get("valid")
        id_metrics = item.get("id_test")
        ood_metrics = item.get("ood_test")
        if not _recovered_metrics_are_usable(dataset, valid_metrics, id_metrics, ood_metrics):
            continue
        train_metrics = item.get("train")
        if train_metrics is None:
            try:
                train_pred = _predict_from_canonical_artifact(artifact, dataset.train.X)
                train_metrics = _evaluate_prediction(dataset.train, train_pred)
            except Exception:
                train_metrics = None
        return {
            "equation": equation,
            "equation_count": item.get("equation_count") or 1,
            "canonical_artifact": artifact,
            "canonical_artifact_error": item.get("canonical_artifact_error"),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
        }
    return None


def _periodic_snapshot_loop(
    *,
    stop_event: threading.Event,
    interval_seconds: int,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    output_dir: Path,
    experiment_dir: str | Path,
) -> None:
    checkpoint_index = 0
    while not stop_event.wait(interval_seconds):
        checkpoint_index += 1
        payload = _build_periodic_snapshot_payload(
            tool_name=tool_name,
            dataset=dataset,
            params=params,
            seed=seed,
            started_at=started_at,
            experiment_dir=experiment_dir,
            checkpoint_index=checkpoint_index,
        )
        if payload is None:
            continue
        _write_progress_payload(
            payload,
            primary_dir=output_dir / _PROGRESS_DIRNAME,
            experiment_dir=experiment_dir,
        )


def _build_srsd_distractor_summary(
    feature_names: list[str],
    feature_descriptions: list[str | None],
) -> str | None:
    """为含 distractor 的 SRSD 数据集构建变量汇总描述。

    注意：这里不能列出哪些变量是 active feature，否则会把 dummy 变量答案直接泄露给 LLM。
    只暴露“存在候选变量/可能有 distractor”这一全局事实。
    """
    has_distractor = False
    for _, desc in zip(feature_names, feature_descriptions):
        if desc and "meaningless" in str(desc).lower():
            has_distractor = True
            break

    if not has_distractor:
        return None  # 无 distractor，无需汇总

    n_total = len(feature_names)
    return (
        f"There are {n_total} candidate variables. "
        "Some variables may be physically meaningful while others may be distractor variables. "
        "Feature-level semantic descriptions are intentionally hidden to avoid revealing which variables are active."
    )


def build_runner_params(
    tool_name: str,
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    seed: int,
    task_label: str | None = None,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).resolve()
    params = dict(params_override or {})
    # runner 在外层统一传 seed，避免 params_override 中重复注入导致构造器冲突。
    params.pop("seed", None)
    params.setdefault("exp_path", str(output_path / "experiments"))
    exp_label = task_label or dataset.dataset_name
    params.setdefault("exp_name", f"{exp_label}_{tool_name}_seed{seed}")
    # 显式注入当前任务的数据契约，避免 wrapper 只能从 X.shape[1] 隐式猜维度。
    # 这些字段属于框架元参数；具体算法可选择消费或忽略，但不应再缺席。
    params.setdefault("n_features", len(dataset.feature_names))
    params.setdefault("feature_names", list(dataset.feature_names))
    params.setdefault("target_name", dataset.target_name)

    # 变量名匿名化：将原始列名替换为 x1..xN，目标列名替换为 y。
    # 对 llmsr/drsr 生效，可通过 params_override 中的 anonymize 开关控制。
    anonymize = _as_bool(params.pop("anonymize", None), default=False)
    if anonymize:
        n = len(dataset.feature_names)
        params["feature_names"] = [f"x{i+1}" for i in range(n)]
        params["target_name"] = "y"
        params["anonymize"] = True

    if tool_name in {"llmsr", "drsr"}:
        inject_prompt_semantics = _as_bool(params.get("inject_prompt_semantics"), default=True)
        if inject_prompt_semantics:
            background = _build_background(dataset.metadata, dataset.feature_names)
            params.setdefault("background", background)
            params.setdefault("metadata_path", str(dataset.dataset_dir / "metadata.yaml"))

            # 匿名化模式下不注入变量/目标描述。
            if params.get("anonymize"):
                params.pop("feature_descriptions", None)
                params.pop("target_description", None)
            elif dataset.feature_descriptions:
                # SRSD 含 distractor 的数据集：变量描述统一标 "meaning or meaningless"，
                # 汇总信息追加到 background。
                srsd_summary = _build_srsd_distractor_summary(
                    dataset.feature_names, dataset.feature_descriptions
                )
                if srsd_summary is not None:
                    params["background"] = f"{background} {srsd_summary}"
                    params["feature_descriptions"] = ["meaning or meaningless"] * len(dataset.feature_names)
                else:
                    params.setdefault("feature_descriptions", dataset.feature_descriptions)
            else:
                params.setdefault("feature_descriptions", dataset.feature_descriptions)
            if not params.get("anonymize") and dataset.target_description:
                params.setdefault("target_description", dataset.target_description)
        else:
            params.setdefault("background", _NEUTRAL_SR_BACKGROUND)
            params.pop("metadata_path", None)
            params.pop("feature_descriptions", None)
            params.pop("target_description", None)

    return params


def build_result_payload(
    *,
    tool_name: str,
    dataset: LoadedDataset,
    params: dict[str, Any],
    seed: int,
    started_at: float,
    status: str,
    error: str | None,
    equation: str | None,
    equation_count: int | None,
    canonical_artifact: dict[str, Any] | None,
    canonical_artifact_error: str | None,
    train_metrics: dict[str, float | None] | None,
    valid_metrics: dict[str, float | None] | None,
    id_metrics: dict[str, float | None] | None,
    ood_metrics: dict[str, float | None] | None,
    experiment_dir: str | None,
    task_label: str | None = None,
    task_global_index: int | None = None,
    expected_dataset_rel: str | None = None,
    expected_dataset_dir: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "task_label": task_label,
        "task_global_index": task_global_index,
        "dataset": dataset.dataset_name,
        "dataset_dir": str(dataset.dataset_dir),
        "expected_dataset_rel": expected_dataset_rel,
        "expected_dataset_dir": expected_dataset_dir,
        "dataset_identity_check": _dataset_identity_check(
            dataset,
            expected_dataset_rel=expected_dataset_rel,
            expected_dataset_dir=expected_dataset_dir,
        ),
        "experiment_dir": str(Path(experiment_dir).resolve()) if experiment_dir else None,
        "status": status,
        "error": error,
        "seed": int(seed),
        "feature_names": dataset.feature_names,
        "target_name": dataset.target_name,
        "train_rows": dataset.train.rows,
        "valid_rows": dataset.valid.rows if dataset.valid else 0,
        "id_test_rows": dataset.id_test.rows if dataset.id_test else 0,
        "ood_test_rows": dataset.ood_test.rows if dataset.ood_test else 0,
        "seconds": round(time.time() - started_at, 3),
        "equation": equation,
        "equation_count": equation_count,
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "train": train_metrics,
        "valid": valid_metrics,
        "id_test": id_metrics,
        "ood_test": ood_metrics,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": _sanitize_params(params),
    }


def _timeout_type_for_payload(
    *,
    dataset: LoadedDataset,
    equation: str | None,
    canonical_artifact: dict[str, Any] | None,
    valid_metrics: dict[str, Any] | None,
    id_metrics: dict[str, Any] | None,
    ood_metrics: dict[str, Any] | None,
) -> str:
    if (
        isinstance(equation, str)
        and equation.strip()
        and isinstance(canonical_artifact, dict)
        and _recovered_metrics_are_usable(dataset, valid_metrics, id_metrics, ood_metrics)
    ):
        return "budget_exhausted_with_output"
    if isinstance(equation, str) and equation.strip() and isinstance(canonical_artifact, dict):
        return "partial_output"
    if isinstance(equation, str) and equation.strip():
        return "unvalidated_expression"
    return "no_valid_output"


def run_benchmark_task(
    *,
    tool_name: str,
    dataset_dir: str | Path,
    output_root: str | Path,
    seed: int = 1314,
    params_override: dict[str, Any] | None = None,
) -> Path:
    params_override_clean, task_identity = _split_runner_task_identity_params(params_override)
    dataset = load_canonical_dataset(dataset_dir)
    raw_global_index = task_identity.get("task_global_index")
    task_global_index = None
    if raw_global_index not in (None, ""):
        try:
            task_global_index = int(raw_global_index)
        except Exception:
            task_global_index = None
    task_label = _build_task_label(
        dataset,
        task_global_index=task_global_index if task_global_index is not None else raw_global_index,
        task_label=task_identity.get("task_label"),
    )
    output_dir = Path(output_root).resolve() / tool_name / task_label
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_runner_params(
        tool_name,
        dataset,
        output_dir,
        seed=seed,
        task_label=task_label,
        params_override=params_override_clean,
    )
    progress_snapshot_interval_seconds = _resolve_progress_snapshot_interval_seconds(tool_name, params)

    started_at = time.time()
    status = "ok"
    error = None
    equation = None
    equation_count = None
    canonical_artifact = None
    canonical_artifact_error = None
    train_metrics = None
    valid_metrics = None
    id_metrics = None
    ood_metrics = None
    experiment_dir = None
    budget_exhausted = False
    timeout_type = "not_timeout"
    raw_timeout_error = None

    reg = SymbolicRegressor(
        tool_name,
        problem_name=dataset.dataset_name,
        seed=seed,
        **params,
    )
    experiment_dir = getattr(reg, "experiment_dir", None)
    snapshot_stop_event: threading.Event | None = None
    snapshot_thread: threading.Thread | None = None

    if progress_snapshot_interval_seconds and experiment_dir and _is_snapshot_capable_tool(tool_name):
        snapshot_stop_event = threading.Event()
        snapshot_thread = threading.Thread(
            target=_periodic_snapshot_loop,
            kwargs={
                "stop_event": snapshot_stop_event,
                "interval_seconds": progress_snapshot_interval_seconds,
                "tool_name": tool_name,
                "dataset": dataset,
                "params": params,
                "seed": seed,
                "started_at": started_at,
                "output_dir": output_dir,
                "experiment_dir": experiment_dir,
            },
            daemon=True,
        )
        snapshot_thread.start()

    try:
        reg.fit(dataset.train.X, dataset.train.y)
        experiment_dir = getattr(reg, "experiment_dir", experiment_dir)
        equation = reg.get_optimal_equation()
        canonical_artifact, canonical_artifact_error = safe_export_canonical_artifact(reg)
        try:
            equations = reg.get_total_equations()
            equation_count = len(equations) if isinstance(equations, list) else None
        except Exception:
            equation_count = None
        train_metrics = _evaluate_split(reg, dataset.train)
        valid_metrics = _evaluate_split(reg, dataset.valid)
        id_metrics = _evaluate_split(reg, dataset.id_test)
        ood_metrics = _evaluate_split(reg, dataset.ood_test)
    except TimeoutError as exc:
        budget_exhausted = True
        raw_timeout_error = repr(exc)
        status = "timed_out"
        error = raw_timeout_error
        experiment_dir = getattr(reg, "experiment_dir", experiment_dir)
        recovered_payload = _recover_timeout_payload_from_candidate(
            tool_name=tool_name,
            dataset=dataset,
            experiment_dir=experiment_dir,
        )
        if recovered_payload is not None:
            equation = recovered_payload["equation"]
            equation_count = recovered_payload["equation_count"]
            canonical_artifact = recovered_payload["canonical_artifact"]
            canonical_artifact_error = recovered_payload["canonical_artifact_error"]
            train_metrics = recovered_payload["train_metrics"]
            valid_metrics = recovered_payload["valid_metrics"]
            id_metrics = recovered_payload["id_metrics"]
            ood_metrics = recovered_payload["ood_metrics"]
            timeout_type = _timeout_type_for_payload(
                dataset=dataset,
                equation=equation,
                canonical_artifact=canonical_artifact,
                valid_metrics=valid_metrics,
                id_metrics=id_metrics,
                ood_metrics=ood_metrics,
            )
            if timeout_type == "budget_exhausted_with_output":
                # 预算耗尽但已恢复出可评估 best-so-far，应按有效完成处理；
                # 原始超时原因保留在 raw_timeout_error，避免巡检误判为失败。
                status = "ok"
                error = None
        else:
            timeout_type = "no_valid_output"
    except Exception as exc:
        status = "error"
        error = repr(exc)
    finally:
        if snapshot_stop_event is not None:
            snapshot_stop_event.set()
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=1.0)

    result = build_result_payload(
        tool_name=tool_name,
        dataset=dataset,
        params=params,
        seed=seed,
        started_at=started_at,
        status=status,
        error=error,
        equation=equation,
        equation_count=equation_count,
        canonical_artifact=canonical_artifact,
        canonical_artifact_error=canonical_artifact_error,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        experiment_dir=experiment_dir,
        task_label=task_label,
        task_global_index=task_global_index,
        expected_dataset_rel=task_identity.get("expected_dataset_rel"),
        expected_dataset_dir=task_identity.get("expected_dataset_dir"),
    )
    result["budget_exhausted"] = bool(budget_exhausted)
    result["timeout_type"] = timeout_type
    result["raw_timeout_error"] = raw_timeout_error
    result["recovered_from_timeout"] = timeout_type == "budget_exhausted_with_output"
    if budget_exhausted:
        result["termination_reason"] = timeout_type
    else:
        result["termination_reason"] = "completed" if status == "ok" else status

    result_path = output_dir / "result.json"
    write_result_payload(result, primary_path=result_path, experiment_dir=experiment_dir)
    return result_path
