"""统一 benchmark/result runner。"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


_HIDDEN_PARAM_KEYS = {"api_key", "apikey", "token", "password", "secret"}
_PROGRESS_DIRNAME = "progress"
_SYMBOLFIT_SEARCH_BEST_FILENAME = ".symbolfit_search_best.json"
_SYMBOLFIT_SEARCH_HISTORY_FILENAME = ".symbolfit_pysr_candidates.jsonl"
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
    "jaxsr",
    "tpsr",
    "QLattice",
    "ragsr",
    "fepysr",
    "symbolfit",
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
_TRAIN_LABEL_NOISE_ENABLED_KEYS = (
    "train_label_noise_enabled",
    "label_noise_enabled",
    "add_train_label_noise",
)
_TRAIN_LABEL_NOISE_SIGMA_KEYS = (
    "train_label_noise_sigma",
    "label_noise_sigma",
    "noise_sigma",
)
_TRAIN_LABEL_NOISE_SEED_KEYS = (
    "train_label_noise_seed",
    "label_noise_seed",
    "noise_seed",
)
_TRAIN_LABEL_NOISE_PROTOCOL = (
    "y_noisy = y + sigma * std(y) * N(0, 1); clean labels are used for evaluation"
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


def _pop_first_key(params: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in params:
            return params.pop(key)
    return None


def _as_optional_nonnegative_float(value: Any, *, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} 必须是非负有限数值，当前为: {value!r}") from exc
    if math.isnan(number) or math.isinf(number) or number < 0:
        raise ValueError(f"{field_name} 必须是非负有限数值，当前为: {value!r}")
    return number


def _resolve_train_label_noise_config(
    params: dict[str, Any],
    *,
    dataset: "LoadedDataset",
    seed: int,
) -> dict[str, Any]:
    """解析训练标签噪声配置，并从算法参数中移除框架级噪声字段。

    噪声只作用于传给 `fit()` 的训练标签；dataset 内部 split 保持 clean，
    因此后续 train/valid/ID/OOD 指标仍按 clean labels 计算。
    """
    enabled_raw = _pop_first_key(params, _TRAIN_LABEL_NOISE_ENABLED_KEYS)
    sigma_raw = _pop_first_key(params, _TRAIN_LABEL_NOISE_SIGMA_KEYS)
    seed_raw = _pop_first_key(params, _TRAIN_LABEL_NOISE_SEED_KEYS)

    sigma = _as_optional_nonnegative_float(sigma_raw, field_name="train_label_noise_sigma")
    enabled = _as_bool(enabled_raw, default=(sigma is not None and sigma > 0))
    if sigma is None:
        sigma = 0.0
    if not enabled:
        sigma = 0.0

    y_clean = np.asarray(dataset.train.y, dtype=float).reshape(-1)
    y_std = float(np.std(y_clean)) if y_clean.size else 0.0
    scale = float(sigma * y_std)

    if seed_raw not in (None, ""):
        try:
            rng_seed = int(seed_raw) % (2**32)
        except Exception as exc:
            raise ValueError(f"train_label_noise_seed 必须是整数，当前为: {seed_raw!r}") from exc
    else:
        dataset_identity = _normalize_dataset_identity_path(dataset.dataset_dir) or dataset.dataset_name
        digest = hashlib.sha256(
            f"{dataset_identity}|seed={int(seed)}|sigma={sigma:.12g}|train_label_noise".encode("utf-8")
        ).digest()
        rng_seed = int.from_bytes(digest[:8], "big") % (2**32)

    return {
        "enabled": bool(enabled and sigma > 0 and scale > 0),
        "requested": bool(enabled and sigma > 0),
        "sigma": float(sigma),
        "y_std": y_std,
        "scale": scale,
        "rng_seed": int(rng_seed),
        "protocol": _TRAIN_LABEL_NOISE_PROTOCOL,
    }


def _freeze_train_label_noise_evidence(
    noise_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """复制一次运行级噪声合同，避免分钟快照持有可变配置引用。"""
    evidence = dict(noise_config or {})
    evidence.setdefault("enabled", False)
    evidence.setdefault("requested", False)
    evidence.setdefault("sigma", 0.0)
    evidence.setdefault("y_std", None)
    evidence.setdefault("scale", 0.0)
    evidence.setdefault("rng_seed", None)
    evidence.setdefault("protocol", _TRAIN_LABEL_NOISE_PROTOCOL)
    return evidence


def _condition_from_train_label_noise(noise_config: Mapping[str, Any]) -> str:
    try:
        sigma = float(noise_config.get("sigma") or 0.0)
    except (TypeError, ValueError, OverflowError):
        sigma = 0.0
    if not bool(noise_config.get("requested")) or not math.isfinite(sigma) or sigma <= 0:
        return "clean"
    return f"noise{int(round(sigma * 100)):03d}"


def _attach_progress_run_context(
    payload: Mapping[str, Any],
    *,
    train_label_noise: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contextualized = dict(payload)
    evidence = _freeze_train_label_noise_evidence(train_label_noise)
    contextualized["condition"] = _condition_from_train_label_noise(evidence)
    contextualized["train_label_noise"] = evidence
    return contextualized


def _train_labels_for_fit(split: DatasetSplit, noise_config: dict[str, Any]) -> np.ndarray:
    y_clean = np.asarray(split.y, dtype=float).reshape(-1)
    if not noise_config.get("enabled"):
        return y_clean
    rng = np.random.default_rng(int(noise_config["rng_seed"]))
    noise = rng.normal(loc=0.0, scale=float(noise_config["scale"]), size=y_clean.shape)
    return y_clean + noise


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
        "cbrt": sp.Function("cbrt"),
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    for i in range(256):
        locals_map[f"x{i}"] = sp.Symbol(f"x{i}")
    return locals_map


_EXECUTABLE_NUMPY_CALLS = {
    "np.abs",
    "np.arccos",
    "np.arcsin",
    "np.arctan",
    "np.cbrt",
    "np.clip",
    "np.cos",
    "np.exp",
    "np.gradient",
    "np.log",
    "np.log1p",
    "np.linalg.norm",
    "np.maximum",
    "np.mean",
    "np.minimum",
    "np.power",
    "np.sin",
    "np.sinh",
    "np.sqrt",
    "np.square",
    "np.tan",
    "np.tanh",
    "np.where",
}
_EXECUTABLE_BARE_CALLS = {
    "abs",
    "arccos",
    "arcsin",
    "arctan",
    "cbrt",
    "clip",
    "cos",
    "exp",
    "gradient",
    "log",
    "log1p",
    "maximum",
    "mean",
    "minimum",
    "power",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "where",
}


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _validate_executable_expression(tree: ast.AST) -> None:
    """只允许 NumPy 数值表达式 AST，拒绝任意 Python 执行能力。"""
    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Subscript,
        ast.Call,
        ast.keyword,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.BitAnd,
        ast.BitOr,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"executable_expression 含不允许的 AST 节点: {node.__class__.__name__}")
        if isinstance(node, ast.Name):
            if node.id in {"np", "pi", "abs", *_EXECUTABLE_BARE_CALLS}:
                continue
            if re.fullmatch(r"x\d+", node.id):
                continue
            raise ValueError(f"executable_expression 含非标准名称: {node.id}")
        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path not in _EXECUTABLE_NUMPY_CALLS and path not in {"np.pi", "np.linalg"}:
                raise ValueError(f"executable_expression 含不允许的属性: {path}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _EXECUTABLE_BARE_CALLS:
                    raise ValueError(f"executable_expression 含不允许的函数: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if _attribute_path(node.func) not in _EXECUTABLE_NUMPY_CALLS:
                    raise ValueError("executable_expression 含不允许的 NumPy 函数")
            else:
                raise ValueError("executable_expression 函数调用形态不受支持")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or not re.fullmatch(r"x\d+", node.value.id):
                raise ValueError("executable_expression 只允许对特征向量做常量下标访问")
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                continue
            if (
                isinstance(index, ast.UnaryOp)
                and isinstance(index.op, (ast.USub, ast.UAdd))
                and isinstance(index.operand, ast.Constant)
                and isinstance(index.operand.value, int)
            ):
                continue
            raise ValueError("executable_expression 特征下标必须是整数常量")


def _predict_executable_expression(expression: str, X: np.ndarray) -> np.ndarray:
    """按冻结 Python AST 的原始结合顺序执行 DRSR/LLMSR 公式。"""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("executable_expression 预测要求二维输入")
    tree = ast.parse(expression, mode="eval")
    _validate_executable_expression(tree)
    context: dict[str, Any] = {"np": np, "pi": np.pi, "abs": np.abs}
    for name in _EXECUTABLE_BARE_CALLS:
        if hasattr(np, name):
            context[name] = getattr(np, name)
    for idx in range(X_arr.shape[1]):
        context[f"x{idx}"] = X_arr[:, idx]
    with np.errstate(all="ignore"):
        value = eval(compile(tree, "<canonical-executable-expression>", "eval"), {"__builtins__": {}}, context)
    pred = np.asarray(value, dtype=float)
    if pred.ndim == 0:
        pred = np.full(X_arr.shape[0], float(pred), dtype=float)
    else:
        pred = np.broadcast_to(pred, (X_arr.shape[0],)).astype(float)
    return pred.reshape(-1)


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
    executable_expression = artifact.get("executable_expression")
    if isinstance(executable_expression, str) and executable_expression.strip():
        return _predict_executable_expression(executable_expression, X)

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

    def _broadcast_maximum(*values, axis=None):
        if len(values) == 1:
            value = values[0]
            if isinstance(value, np.ndarray) and value.dtype == object:
                arrays = np.broadcast_arrays(*list(value))
                return np.maximum.reduce(arrays)
            if isinstance(value, (list, tuple)):
                arrays = np.broadcast_arrays(*value)
                return np.maximum.reduce(arrays)
            return np.maximum.reduce(value, axis=axis)
        arrays = np.broadcast_arrays(*values)
        return np.maximum.reduce(arrays)

    def _broadcast_minimum(*values, axis=None):
        if len(values) == 1:
            value = values[0]
            if isinstance(value, np.ndarray) and value.dtype == object:
                arrays = np.broadcast_arrays(*list(value))
                return np.minimum.reduce(arrays)
            if isinstance(value, (list, tuple)):
                arrays = np.broadcast_arrays(*value)
                return np.minimum.reduce(arrays)
            return np.minimum.reduce(value, axis=axis)
        arrays = np.broadcast_arrays(*values)
        return np.minimum.reduce(arrays)

    fn = sp.lambdify(
        free_symbols,
        expr,
        modules=[
            {
                "Max": _broadcast_maximum,
                "Min": _broadcast_minimum,
                "amax": _broadcast_maximum,
                "amin": _broadcast_minimum,
                "cbrt": np.cbrt,
            },
            "numpy",
        ],
    )
    try:
        pred = fn(*args)
    except ValueError:
        scalar_fn = sp.lambdify(
            free_symbols,
            expr,
            modules=[{"Max": max, "Min": min, "Abs": abs, "cbrt": np.cbrt}, "math"],
        )
        values = []
        for row in X_arr:
            row_args = [float(row[int(str(sym)[1:])]) for sym in free_symbols]
            values.append(float(scalar_fn(*row_args)))
        pred = np.asarray(values, dtype=float)
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


def _read_pysr_hall_of_fame_candidates(candidate_paths: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        equation_column = next(
            (name for name in ("Equation", "PySR equation", "equation") if name in df.columns),
            None,
        )
        loss_column = next(
            (name for name in ("Loss", "PySR loss", "loss") if name in df.columns),
            None,
        )
        if equation_column is None or loss_column is None:
            continue
        for _, row in df.iterrows():
            equation = row.get(equation_column)
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                loss = float(row.get(loss_column))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(loss):
                continue
            complexity = row.get("Complexity")
            try:
                complexity = int(complexity)
            except (TypeError, ValueError, OverflowError):
                complexity = None
            iteration = None
            for iteration_column in ("Iteration", "iteration", "DiscoveryIteration"):
                if iteration_column not in df.columns:
                    continue
                try:
                    iteration = int(row.get(iteration_column))
                except (TypeError, ValueError, OverflowError):
                    iteration = None
                break
            candidates.append(
                {
                    "equation": equation.strip(),
                    "scaled_equation": equation.strip(),
                    "loss": loss,
                    "internal_loss": loss,
                    "complexity": complexity,
                    "iteration": iteration,
                    "source_path": str(path),
                    "source": "pysr_native_hall_of_fame",
                }
            )
    return candidates


def _extract_pysr_candidate_from_hall_of_fame_paths(candidate_paths: list[Path]) -> dict[str, Any] | None:
    candidates = _read_pysr_hall_of_fame_candidates(candidate_paths)
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["internal_loss"])


def _extract_pysr_periodic_candidate(
    experiment_dir: str | Path,
    *,
    snapshot_minute: int | None = None,
    snapshot_elapsed_seconds: float | None = None,
) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    candidate_paths = [base_dir / "hall_of_fame.csv", base_dir / "hall_of_fame.csv.bak"]
    candidate = _extract_pysr_candidate_from_hall_of_fame_paths(candidate_paths)
    state_path = base_dir / ".pysr_native_incumbent.json"
    incumbent = _read_json_file(state_path)
    if candidate is None:
        return incumbent
    loss = candidate.get("internal_loss")
    try:
        loss = float(loss)
    except (TypeError, ValueError, OverflowError):
        return incumbent
    if not math.isfinite(loss):
        return incumbent
    old_loss = incumbent.get("internal_loss") if isinstance(incumbent, Mapping) else None
    try:
        old_loss = float(old_loss)
    except (TypeError, ValueError, OverflowError):
        old_loss = None
    # PySR HOF loss 越小越好；相同 loss 保留第一次发现，禁止结束时覆盖时间。
    if old_loss is not None and math.isfinite(old_loss) and loss >= old_loss:
        return dict(incumbent)
    observed_at = time.time()
    elapsed = snapshot_elapsed_seconds
    try:
        elapsed = max(0.0, float(elapsed))
    except (TypeError, ValueError, OverflowError):
        elapsed = None
    minute = snapshot_minute
    try:
        minute = max(1, int(minute))
    except (TypeError, ValueError, OverflowError):
        minute = max(1, int(math.ceil((elapsed or 0.0) / 60.0)))
    equation = str(candidate["equation"]).strip()
    evidence = {
        **candidate,
        "equation": equation,
        "original_equation": equation,
        "loss": loss,
        "internal_loss": loss,
        "internal_objective": "hof_loss",
        "objective_direction": "min",
        "first_discovered_minute": minute,
        "first_discovered_elapsed_seconds": elapsed,
        "source_timestamp_unix": float(observed_at),
        "source": "pysr_native_hall_of_fame",
    }
    evidence["candidate_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "equation": equation,
                "loss": loss,
                "iteration": evidence.get("iteration"),
                "source_path": evidence.get("source_path"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_name(f".{state_path.name}.tmp")
        temporary.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(state_path)
    except Exception:
        pass
    return evidence


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


def _imcts_candidate_evidence(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    """归一化 iMCTS 原生 reward 候选的发现与排名证据。"""

    if not isinstance(candidate, Mapping):
        return {}
    aliases = {
        "source_score": ("source_score", "score"),
        "source_evaluations": ("source_evaluations", "evaluations"),
        "candidate_first_discovered_minute": (
            "candidate_first_discovered_minute",
            "first_discovered_minute",
        ),
        "candidate_first_discovered_elapsed_seconds": (
            "candidate_first_discovered_elapsed_seconds",
            "first_discovered_elapsed_seconds",
        ),
        "candidate_source": ("candidate_source", "source"),
        "expression_vector": ("expression_vector", "equation"),
        "internal_objective": ("internal_objective",),
        "internal_objective_direction": (
            "internal_objective_direction",
            "objective_direction",
        ),
        "candidate_source_timestamp_unix": (
            "candidate_source_timestamp_unix",
            "source_timestamp_unix",
        ),
        "candidate_sha256": ("candidate_sha256",),
    }
    evidence: dict[str, Any] = {}
    for target, sources in aliases.items():
        for source in sources:
            value = candidate.get(source)
            if value is not None:
                evidence[target] = value
                break
    return evidence


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


def _extract_jaxsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".jaxsr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if isinstance(equation, str) and equation.strip():
        return item
    fidelity = item.get("fidelity")
    # JAXSR 明确记录了导出失败/校验中时，这是一个真实的当前 best 状态，
    # 不能降级成 heartbeat，否则轨迹会错误 carry-forward 旧公式。
    if isinstance(fidelity, dict) and fidelity:
        return item
    return None


def _jaxsr_candidate_fidelity(
    tool_name: str,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if str(tool_name).strip().lower() != "jaxsr":
        return None, None
    fidelity = candidate.get("fidelity")
    if not isinstance(fidelity, dict) or not fidelity:
        return None, None
    status = str(fidelity.get("status") or "invalid").strip().lower()
    equation = candidate.get("equation")
    if status != "verified":
        reason = fidelity.get("reason") or "native best 没有可验证的忠实表达式"
        return fidelity, f"JAXSR export fidelity {status}: {reason}"
    if not isinstance(equation, str) or not equation.strip():
        return fidelity, "JAXSR export fidelity invalid: verified 状态缺少 equation"
    return fidelity, None


def _extract_ragsr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".ragsr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _extract_fepysr_periodic_candidate(experiment_dir: str | Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / ".fepysr_current_best.json"
    item = _read_json_file(path)
    if not item:
        return None
    equation = item.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return item


def _symbolfit_coordinate_transform(active: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(active, dict):
        return None
    nested = active.get("coordinate_transform")
    transform = dict(nested) if isinstance(nested, dict) else {}
    for key in (
        "input_rescale",
        "x_min",
        "x_max",
        "x_range",
        "y_scale",
        "y_unscale_factor",
        "scale_y_by",
        "equation_space",
    ):
        if key in active and key not in transform:
            transform[key] = active[key]
    if not transform:
        return None
    return transform


def _unscale_symbolfit_equation(
    equation: str,
    coordinate_transform: dict[str, Any] | None,
) -> str:
    """将 SymbolFit active PySR 的缩放空间公式还原到原始 X/y 坐标。"""
    if not isinstance(equation, str) or not equation.strip() or not coordinate_transform:
        return str(equation or "").strip()
    transform = coordinate_transform
    equation_space = str(transform.get("equation_space") or "")
    if equation_space.startswith("original_input") and not transform.get("input_rescale"):
        return equation.strip()
    try:
        input_rescale = bool(transform.get("input_rescale", False))
        y_unscale_factor = transform.get("y_unscale_factor")
        if y_unscale_factor is None:
            y_scale = float(transform.get("y_scale", 1.0))
            y_unscale_factor = 1.0 / y_scale if y_scale else 1.0
        y_unscale_factor = float(y_unscale_factor)
        if not math.isfinite(y_unscale_factor) or y_unscale_factor == 0.0:
            y_unscale_factor = 1.0

        text = re.sub(r"\bX(\d+)\b", lambda match: f"x{match.group(1)}", equation.strip())
        expr = sp.sympify(text, locals=_sympy_locals())
        if input_rescale:
            x_min = transform.get("x_min")
            x_range = transform.get("x_range")
            x_max = transform.get("x_max")
            if x_range is None and isinstance(x_min, list) and isinstance(x_max, list):
                x_range = [float(high) - float(low) for low, high in zip(x_min, x_max)]
            if not isinstance(x_min, list) or not isinstance(x_range, list):
                return equation.strip()
            substitutions: dict[Any, Any] = {}
            for index, (raw_min, raw_range) in enumerate(zip(x_min, x_range)):
                try:
                    minimum = float(raw_min)
                    width = float(raw_range)
                except (TypeError, ValueError, OverflowError):
                    return equation.strip()
                if not math.isfinite(minimum) or not math.isfinite(width):
                    return equation.strip()
                symbol = sp.Symbol(f"x{index}")
                if abs(width) <= 1.0e-15:
                    # 上游对常量特征的缩放本身会产生零除；在原始数据坐标
                    # 中将该缩放变量固定为 0，避免回放时凭空引入 x 依赖。
                    substitutions[symbol] = sp.Float(0.0)
                else:
                    substitutions[symbol] = (symbol - sp.Float(minimum)) / sp.Float(width)
            if substitutions:
                expr = expr.xreplace(substitutions)
        if abs(y_unscale_factor - 1.0) > 1.0e-15:
            expr = sp.Float(y_unscale_factor) * expr
        return str(expr)
    except Exception:
        return equation.strip()


def _read_symbolfit_search_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    candidates: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return candidates
    for line in lines:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        equation = item.get("scaled_equation") or item.get("equation")
        if not isinstance(equation, str) or not equation.strip():
            continue
        try:
            loss = float(item.get("internal_loss", item.get("loss")))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(loss):
            continue
        item = dict(item)
        item["scaled_equation"] = equation.strip()
        item["internal_loss"] = loss
        item["loss"] = loss
        candidates.append(item)
    return candidates


def _append_symbolfit_search_history(path: Path, candidate: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def _merge_symbolfit_search_candidate(
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    snapshot_minute: int | None,
    snapshot_elapsed_seconds: float | None,
    attempt: Any,
    coordinate_transform: dict[str, Any] | None,
) -> dict[str, Any]:
    scaled_equation = str(candidate.get("scaled_equation") or candidate.get("equation") or "").strip()
    key = str(candidate.get("candidate_key") or hashlib.sha256(scaled_equation.encode("utf-8")).hexdigest())
    try:
        loss = float(candidate.get("internal_loss", candidate.get("loss")))
    except (TypeError, ValueError, OverflowError):
        loss = math.inf
    existing = next((item for item in history if str(item.get("candidate_key")) == key), None)
    if existing is None:
        merged = dict(candidate)
        merged["candidate_key"] = key
        merged["scaled_equation"] = scaled_equation
        merged["internal_loss"] = loss
        merged["loss"] = loss
        merged["attempt"] = attempt
        merged["first_discovered_attempt"] = attempt
        merged["first_discovered_minute"] = snapshot_minute
        merged["first_discovered_elapsed_seconds"] = snapshot_elapsed_seconds
        if coordinate_transform:
            merged["coordinate_transform"] = coordinate_transform
        history.append(merged)
        return merged

    if loss < float(existing.get("internal_loss", math.inf)):
        existing["internal_loss"] = loss
        existing["loss"] = loss
    if coordinate_transform and not existing.get("coordinate_transform"):
        existing["coordinate_transform"] = coordinate_transform
    return existing


def _extract_symbolfit_periodic_candidate(
    experiment_dir: str | Path,
    *,
    snapshot_minute: int | None = None,
    snapshot_elapsed_seconds: float | None = None,
) -> dict[str, Any] | None:
    base_dir = Path(experiment_dir)
    active = _read_json_file(base_dir / ".symbolfit_active_run.json")
    coordinate_transform = _symbolfit_coordinate_transform(active)
    try:
        attempt = int(active.get("attempt")) if isinstance(active, dict) else None
    except (TypeError, ValueError):
        attempt = None

    history_path = base_dir / _SYMBOLFIT_SEARCH_HISTORY_FILENAME
    history = _read_symbolfit_search_history(history_path)
    if snapshot_elapsed_seconds is not None:
        try:
            observed_elapsed_seconds = max(0.0, float(snapshot_elapsed_seconds))
        except (TypeError, ValueError, OverflowError):
            observed_elapsed_seconds = None
    else:
        observed_elapsed_seconds = None
    observed_minute = snapshot_minute
    if observed_elapsed_seconds is not None and math.isfinite(observed_elapsed_seconds):
        observed_minute = max(1, int(math.ceil(observed_elapsed_seconds / 60.0)))
    active_candidates: list[dict[str, Any]] = []
    if isinstance(active, dict):
        work_dir = active.get("work_dir")
        if isinstance(work_dir, str) and work_dir.strip():
            work_path = Path(work_dir)
            if work_path.is_dir():
                candidate_paths = sorted(
                    [
                        *work_path.glob("outputs_tmp/*/hall_of_fame.csv"),
                        *work_path.glob("outputs_tmp/*/hall_of_fame.csv.bak"),
                    ],
                    key=lambda item_path: item_path.stat().st_mtime if item_path.exists() else 0,
                    reverse=True,
                )
                active_candidates = _read_pysr_hall_of_fame_candidates(candidate_paths)

    observed_history_rows: list[dict[str, Any]] = []
    for candidate in active_candidates:
        merged = _merge_symbolfit_search_candidate(
            history,
            candidate,
            snapshot_minute=observed_minute,
            snapshot_elapsed_seconds=observed_elapsed_seconds,
            attempt=attempt,
            coordinate_transform=coordinate_transform,
        )
        # 返回历史项保留还原后的表达式，同时留下 scaled_equation 便于追溯和重放。
        merged["equation"] = _unscale_symbolfit_equation(
            str(merged.get("scaled_equation") or ""),
            merged.get("coordinate_transform"),
        )
        merged["tool"] = "symbolfit"
        merged["source"] = "symbolfit_active_pysr_hall_of_fame"

        candidate_copy = dict(candidate)
        candidate_copy["attempt"] = attempt
        candidate_copy["first_discovered_attempt"] = merged.get(
            "first_discovered_attempt"
        )
        candidate_copy["first_discovered_minute"] = merged.get(
            "first_discovered_minute"
        )
        candidate_copy["first_discovered_elapsed_seconds"] = merged.get(
            "first_discovered_elapsed_seconds"
        )
        candidate_copy["coordinate_transform"] = coordinate_transform
        candidate_copy["candidate_key"] = merged.get("candidate_key")
        observed_history_rows.append(candidate_copy)

    if active_candidates:
        # 持久化本次观察到的候选；重复行无害，读取时会保留最早发现信息。
        for candidate_copy in observed_history_rows:
            _append_symbolfit_search_history(history_path, candidate_copy)

    # wrapper 在 attempt 结束后写入该 sidecar；它仍是 PySR 内部目标，
    # 因而优先级高于基于 refit RMSE 的 current-best 文件。
    search_best = _read_json_file(base_dir / _SYMBOLFIT_SEARCH_BEST_FILENAME)
    if isinstance(search_best, dict):
        search_best = dict(search_best)
        search_best["equation"] = _unscale_symbolfit_equation(
            str(search_best.get("scaled_equation") or search_best.get("equation") or ""),
            search_best.get("coordinate_transform") or coordinate_transform,
        )
        try:
            search_best["internal_loss"] = float(
                search_best.get("internal_loss", search_best.get("loss"))
            )
            search_best["loss"] = search_best["internal_loss"]
        except (TypeError, ValueError, OverflowError):
            search_best = None
        if search_best is not None:
            history.append(search_best)

    # wrapper 侧遥测以缩放坐标保存 PySR 表达式（scaled_equation），并在旁边
    # 保存仿射映射。选全局搜索最优前先物化原始坐标表达式，避免 canonical
    # replay 把缩放公式直接用于原始 X。
    for item in history:
        scaled_equation = item.get("scaled_equation") or item.get("equation")
        if not isinstance(scaled_equation, str) or not scaled_equation.strip():
            continue
        item["scaled_equation"] = scaled_equation.strip()
        item["equation"] = _unscale_symbolfit_equation(
            item["scaled_equation"],
            item.get("coordinate_transform") or coordinate_transform,
        )

    valid_history = [
        item
        for item in history
        if isinstance(item.get("equation"), str)
        and item.get("equation", "").strip()
        and isinstance(item.get("internal_loss"), (int, float))
        and math.isfinite(float(item["internal_loss"]))
        and (
            snapshot_minute is None
            or not isinstance(item.get("first_discovered_minute"), (int, float))
            or int(item["first_discovered_minute"]) <= snapshot_minute
        )
    ]
    if valid_history:
        best = min(valid_history, key=lambda item: float(item["internal_loss"]))
        result = dict(best)
        result["tool"] = "symbolfit"
        result["source"] = result.get("source") or "symbolfit_pysr_search_history"
        result["loss"] = float(result["internal_loss"])
        return result

    # 兼容旧 run：仅写出 refit 候选、没有 PySR 活跃遥测时回退到该文件。
    current = _read_json_file(base_dir / ".symbolfit_current_best.json")
    if current:
        equation = current.get("equation")
        if isinstance(equation, str) and equation.strip():
            current = dict(current)
            current["source"] = "symbolfit_refit_current_best"
            return current
    return None


def _extract_periodic_candidate(
    tool_name: str,
    experiment_dir: str | Path,
    *,
    snapshot_minute: int | None = None,
    snapshot_elapsed_seconds: float | None = None,
) -> dict[str, Any] | None:
    tool = str(tool_name).strip().lower()
    if tool == "llmsr":
        return _extract_llmsr_periodic_candidate(experiment_dir)
    if tool == "drsr":
        return _extract_drsr_periodic_candidate(experiment_dir)
    if tool == "pysr":
        return _extract_pysr_periodic_candidate(
            experiment_dir,
            snapshot_minute=snapshot_minute,
            snapshot_elapsed_seconds=snapshot_elapsed_seconds,
        )
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
    if tool == "jaxsr":
        return _extract_jaxsr_periodic_candidate(experiment_dir)
    if tool == "ragsr":
        return _extract_ragsr_periodic_candidate(experiment_dir)
    if tool == "fepysr":
        return _extract_fepysr_periodic_candidate(experiment_dir)
    if tool == "symbolfit":
        return _extract_symbolfit_periodic_candidate(
            experiment_dir,
            snapshot_minute=snapshot_minute,
            snapshot_elapsed_seconds=snapshot_elapsed_seconds,
        )
    return None


def _progress_snapshot_filename(
    payload: dict[str, Any],
    *,
    snapshot_minute_index: int | None = None,
) -> str:
    if snapshot_minute_index is not None:
        try:
            elapsed_minutes = max(0, int(snapshot_minute_index))
        except Exception:
            elapsed_minutes = 0
        return f"minute_{elapsed_minutes:04d}.json"

    elapsed_seconds = payload.get("elapsed_seconds")
    try:
        elapsed_minutes = max(0, int(round(float(elapsed_seconds) / 60.0)))
    except Exception:
        elapsed_minutes = 0
    return f"minute_{elapsed_minutes:04d}.json"


def _progress_minute_index_from_elapsed(
    elapsed_seconds: Any,
    *,
    interval_seconds: int,
) -> int | None:
    try:
        elapsed = float(elapsed_seconds)
        interval = int(interval_seconds)
    except Exception:
        return None
    if interval <= 0 or not math.isfinite(elapsed):
        return None
    return max(0, int(elapsed // interval))


def _progress_budget_minute_index(
    result: dict[str, Any],
    *,
    interval_seconds: int,
) -> int | None:
    params = result.get("params")
    if not isinstance(params, dict):
        return None
    try:
        timeout_seconds = float(params.get("timeout_in_seconds"))
        elapsed_seconds = float(result.get("seconds") or 0.0)
        interval = int(interval_seconds)
    except Exception:
        return None
    if interval <= 0 or timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        return None
    if not math.isfinite(elapsed_seconds):
        return None
    if elapsed_seconds < max(0.0, timeout_seconds - interval):
        return None
    return max(1, int(round(timeout_seconds / interval)))


def _build_progress_backfill_payload(
    payload: dict[str, Any],
    *,
    snapshot_minute_index: int,
    backfilled_from_minute: int,
    interval_seconds: int,
) -> dict[str, Any]:
    backfill_payload = dict(payload)
    backfill_payload["record_type"] = "periodic_backfill"
    backfill_payload["source_record_type"] = payload.get("record_type")
    backfill_payload["backfilled_from_minute"] = int(backfilled_from_minute)
    backfill_payload["backfilled_from_checkpoint_index"] = payload.get("checkpoint_index")
    backfill_payload["source_elapsed_seconds"] = payload.get("elapsed_seconds")
    backfill_payload["checkpoint_index"] = int(snapshot_minute_index)
    backfill_payload["elapsed_seconds"] = round(float(snapshot_minute_index * interval_seconds), 3)
    backfill_payload["elapsed_minutes"] = int(snapshot_minute_index)
    return backfill_payload


def _write_progress_payload(
    payload: dict[str, Any],
    *,
    primary_dir: str | Path,
    experiment_dir: str | Path | None = None,
    snapshot_minute_index: int | None = None,
) -> list[Path]:
    filename = _progress_snapshot_filename(
        payload,
        snapshot_minute_index=snapshot_minute_index,
    )
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
    task_label: str | None = None,
    task_global_index: int | None = None,
    expected_dataset_rel: str | None = None,
    expected_dataset_dir: str | None = None,
    train_label_noise: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    snapshot_elapsed_seconds = max(0.0, time.time() - started_at)
    candidate = _extract_periodic_candidate(
        tool_name,
        experiment_dir,
        snapshot_minute=checkpoint_index,
        snapshot_elapsed_seconds=snapshot_elapsed_seconds,
    )
    if not candidate:
        payload = build_result_payload(
            tool_name=tool_name,
            dataset=dataset,
            params=params,
            seed=seed,
            started_at=started_at,
            status="running",
            error=None,
            equation=None,
            equation_count=0,
            canonical_artifact=None,
            canonical_artifact_error=None,
            train_metrics=None,
            valid_metrics=None,
            id_metrics=None,
            ood_metrics=None,
            experiment_dir=str(experiment_dir),
            task_label=task_label,
            task_global_index=task_global_index,
            expected_dataset_rel=expected_dataset_rel,
            expected_dataset_dir=expected_dataset_dir,
        )
        payload["record_type"] = "periodic_heartbeat"
        payload["checkpoint_index"] = int(checkpoint_index)
        payload["elapsed_seconds"] = round(time.time() - started_at, 3)
        payload["elapsed_minutes"] = max(0, int(round(payload["elapsed_seconds"] / 60.0)))
        payload["candidate_available"] = False
        native_contract = {
            "e2esr": ("decoder_length_normalized_log_likelihood", "max"),
            "imcts": ("native_reward", "max"),
            "pysr": ("hof_loss", "min"),
        }.get(str(tool_name).strip().lower())
        if native_contract is not None:
            payload["algorithm_native_incumbent"] = False
            payload["internal_objective"] = native_contract[0]
            payload["internal_objective_direction"] = native_contract[1]
            payload["internal_objective_value"] = None
            payload["native_objective_unavailable"] = True
        return _attach_progress_run_context(
            payload,
            train_label_noise=train_label_noise,
        )

    candidate_fidelity, fidelity_export_error = _jaxsr_candidate_fidelity(
        tool_name,
        candidate,
    )
    raw_equation = candidate.get("function") if "function" in candidate else candidate.get("equation")
    if fidelity_export_error is not None:
        canonical_artifact = None
        canonical_artifact_error = fidelity_export_error
    else:
        parameter_values = _candidate_parameter_values(candidate)
        canonical_artifact, canonical_artifact_error = safe_build_canonical_artifact(
            tool_name=tool_name,
            equation=raw_equation,
            expected_n_features=len(dataset.feature_names),
            parameter_values=parameter_values,
        )
        if canonical_artifact is not None and candidate_fidelity is not None:
            canonical_artifact["fidelity_check"] = dict(candidate_fidelity)

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
        equation=str(raw_equation).strip() if raw_equation is not None else None,
        equation_count=1,
        canonical_artifact=canonical_artifact,
        canonical_artifact_error=canonical_artifact_error,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        id_metrics=id_metrics,
        ood_metrics=ood_metrics,
        experiment_dir=str(experiment_dir),
        task_label=task_label,
        task_global_index=task_global_index,
        expected_dataset_rel=expected_dataset_rel,
        expected_dataset_dir=expected_dataset_dir,
    )
    payload["record_type"] = "periodic_best"
    payload["checkpoint_index"] = int(checkpoint_index)
    payload["elapsed_seconds"] = round(time.time() - started_at, 3)
    payload["elapsed_minutes"] = max(0, int(round(payload["elapsed_seconds"] / 60.0)))
    payload["source_iteration"] = (
        candidate.get("iteration")
        if candidate.get("iteration") is not None
        else candidate.get("epoch")
    )
    payload["source_sample_order"] = candidate.get("sample_order")
    payload["source_score"] = candidate.get("score")
    payload["source_loss"] = candidate.get("loss")
    payload["source_internal_loss"] = candidate.get("internal_loss")
    payload["source_evaluations"] = candidate.get("evaluations")
    payload["source_complexity"] = candidate.get("complexity")
    payload["candidate_attempt"] = candidate.get("attempt")
    payload["candidate_first_discovered_attempt"] = candidate.get("first_discovered_attempt")
    payload["candidate_first_discovered_minute"] = candidate.get("first_discovered_minute")
    payload["candidate_first_discovered_elapsed_seconds"] = candidate.get(
        "first_discovered_elapsed_seconds"
    )
    payload["candidate_coordinate_transform"] = candidate.get("coordinate_transform")
    payload["candidate_source"] = candidate.get("source")
    payload["algorithm_native_incumbent"] = candidate.get("internal_objective") is not None
    payload["native_objective_unavailable"] = candidate.get("internal_objective") is None
    payload["internal_objective"] = candidate.get("internal_objective")
    payload["internal_objective_direction"] = candidate.get("objective_direction")
    payload["internal_objective_value"] = (
        candidate.get("internal_loss")
        if candidate.get("objective_direction") == "min"
        else candidate.get("native_model_score", candidate.get("score"))
    )
    payload["expression_vector"] = candidate.get("expression_vector")
    payload["candidate_source_timestamp_unix"] = candidate.get("source_timestamp_unix")
    payload["candidate_sha256"] = candidate.get("candidate_sha256")
    payload["candidate_original_equation"] = candidate.get("original_equation")
    payload["candidate_rank"] = candidate.get("candidate_rank")
    payload["candidate_bag_index"] = candidate.get("bag_index")
    payload["native_model_score"] = candidate.get("native_model_score")
    payload["task_identity"] = {
        "task_label": task_label,
        "task_global_index": task_global_index,
        "dataset_id": dataset.dataset_name,
        "condition": _condition_from_train_label_noise(
            _freeze_train_label_noise_evidence(train_label_noise)
        ),
        "seed": int(seed),
    }
    payload["candidate_available"] = True
    if candidate_fidelity is not None:
        payload["candidate_fidelity"] = dict(candidate_fidelity)
    return _attach_progress_run_context(
        payload,
        train_label_noise=train_label_noise,
    )


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

    candidate_fidelity, fidelity_export_error = _jaxsr_candidate_fidelity(
        tool_name,
        candidate,
    )
    if fidelity_export_error is not None:
        return None

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
    if canonical_artifact is not None and candidate_fidelity is not None:
        canonical_artifact["fidelity_check"] = dict(candidate_fidelity)

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

    recovered = {
        "equation": equation,
        "equation_count": 1,
        "canonical_artifact": canonical_artifact,
        "canonical_artifact_error": canonical_artifact_error,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics,
    }
    if str(tool_name).strip().lower() in {"imcts", "imcts_wrapper"}:
        recovered.update(_imcts_candidate_evidence(candidate))
    return recovered


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
        recovered = {
            "equation": equation,
            "equation_count": item.get("equation_count") or 1,
            "canonical_artifact": artifact,
            "canonical_artifact_error": item.get("canonical_artifact_error"),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
        }
        if expected_tool in {"imcts", "imcts_wrapper"}:
            recovered.update(_imcts_candidate_evidence(item))
        return recovered
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
    task_label: str | None = None,
    task_global_index: int | None = None,
    expected_dataset_rel: str | None = None,
    expected_dataset_dir: str | None = None,
    train_label_noise: Mapping[str, Any] | None = None,
) -> None:
    last_written_minute_index = 0
    next_target_minute_index = 1
    while True:
        target_time = started_at + next_target_minute_index * interval_seconds
        wait_seconds = max(0.0, target_time - time.time())
        if stop_event.wait(wait_seconds):
            break
        try:
            payload = _build_periodic_snapshot_payload(
                tool_name=tool_name,
                dataset=dataset,
                params=params,
                seed=seed,
                started_at=started_at,
                experiment_dir=experiment_dir,
                checkpoint_index=next_target_minute_index,
                task_label=task_label,
                task_global_index=task_global_index,
                expected_dataset_rel=expected_dataset_rel,
                expected_dataset_dir=expected_dataset_dir,
                train_label_noise=train_label_noise,
            )
            if payload is None:
                next_target_minute_index += 1
                continue
            payload = _attach_progress_run_context(
                payload,
                train_label_noise=train_label_noise,
            )
            elapsed_minute_index = _progress_minute_index_from_elapsed(
                payload.get("elapsed_seconds"),
                interval_seconds=interval_seconds,
            )
            snapshot_minute_index = max(
                next_target_minute_index,
                elapsed_minute_index if elapsed_minute_index is not None else next_target_minute_index,
            )
            for missing_minute_index in range(last_written_minute_index + 1, snapshot_minute_index):
                backfill_payload = _build_progress_backfill_payload(
                    payload,
                    snapshot_minute_index=missing_minute_index,
                    backfilled_from_minute=snapshot_minute_index,
                    interval_seconds=interval_seconds,
                )
                _write_progress_payload(
                    backfill_payload,
                    primary_dir=output_dir / _PROGRESS_DIRNAME,
                    experiment_dir=experiment_dir,
                    snapshot_minute_index=missing_minute_index,
                )
            _write_progress_payload(
                payload,
                primary_dir=output_dir / _PROGRESS_DIRNAME,
                experiment_dir=experiment_dir,
                snapshot_minute_index=snapshot_minute_index,
            )
            last_written_minute_index = snapshot_minute_index
            next_target_minute_index = max(next_target_minute_index + 1, last_written_minute_index + 1)
        except Exception:
            next_target_minute_index += 1


def _write_final_progress_payload_if_requested(
    *,
    result: dict[str, Any],
    progress_snapshot_interval_seconds: int | None,
    output_dir: Path,
    experiment_dir: str | Path | None,
    tool_name: str | None = None,
    dataset: LoadedDataset | None = None,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
    started_at: float | None = None,
    task_label: str | None = None,
    task_global_index: int | None = None,
    expected_dataset_rel: str | None = None,
    expected_dataset_dir: str | None = None,
    train_label_noise: Mapping[str, Any] | None = None,
) -> None:
    if not progress_snapshot_interval_seconds:
        return
    if str(result.get("status") or "").strip().lower() != "ok":
        return
    equation = result.get("equation")
    if not isinstance(equation, str) or not equation.strip():
        return
    if not isinstance(result.get("canonical_artifact"), dict):
        return

    snapshot_minute_index = _progress_budget_minute_index(
        result,
        interval_seconds=progress_snapshot_interval_seconds,
    )
    frozen_train_label_noise = _freeze_train_label_noise_evidence(
        train_label_noise
        if train_label_noise is not None
        else result.get("train_label_noise")
    )
    payload = None
    normalized_tool = str(tool_name or result.get("tool") or "").strip()
    if (
        normalized_tool
        and dataset is not None
        and params is not None
        and seed is not None
        and started_at is not None
        and experiment_dir is not None
        and _is_snapshot_capable_tool(normalized_tool)
    ):
        try:
            payload = _build_periodic_snapshot_payload(
                tool_name=normalized_tool,
                dataset=dataset,
                params=params,
                seed=seed,
                started_at=started_at,
                experiment_dir=experiment_dir,
                checkpoint_index=snapshot_minute_index,
                task_label=task_label,
                task_global_index=task_global_index,
                expected_dataset_rel=expected_dataset_rel,
                expected_dataset_dir=expected_dataset_dir,
                train_label_noise=frozen_train_label_noise,
            )
        except Exception:
            payload = None
        if payload is not None:
            payload["record_type"] = "budget_end_internal_best"
            payload["checkpoint_index"] = int(snapshot_minute_index)
            if normalized_tool.lower() in {"imcts", "imcts_wrapper"}:
                result.update(_imcts_candidate_evidence(payload))

    if payload is None:
        payload = dict(result)
        payload["record_type"] = "final_best"
        payload["checkpoint_index"] = "final"
        try:
            elapsed_seconds = float(result.get("seconds") or 0.0)
        except Exception:
            elapsed_seconds = 0.0
        payload["elapsed_seconds"] = round(elapsed_seconds, 3)
        payload["elapsed_minutes"] = max(0, int(round(elapsed_seconds / 60.0)))

    payload = _attach_progress_run_context(
        payload,
        train_label_noise=frozen_train_label_noise,
    )
    _write_progress_payload(
        payload,
        primary_dir=output_dir / _PROGRESS_DIRNAME,
        experiment_dir=experiment_dir,
        snapshot_minute_index=snapshot_minute_index,
    )


def _build_srsd_distractor_summary(
    feature_names: list[str],
    feature_descriptions: list[str | None],
) -> str | None:
    """为含 distractor 的 SRSD 数据集构建变量汇总描述。

    注意：这里不能列出哪个 x_i 对应哪个语义，否则会把 dummy 变量答案直接泄露给 LLM。
    允许暴露无序语义集合：有哪些物理含义、各出现多少个、dummy 有多少个。
    """
    semantic_counts: dict[str, int] = {}
    n_distractors = 0
    for desc in feature_descriptions:
        text = str(desc or "").strip()
        if not text:
            continue
        if "meaningless" in text.lower():
            n_distractors += 1
            continue
        semantic_counts[text] = semantic_counts.get(text, 0) + 1

    if n_distractors == 0:
        return None  # 无 distractor，无需汇总

    n_total = len(feature_names)
    role_parts = []
    for role, count in sorted(semantic_counts.items(), key=lambda item: item[0].lower()):
        unit = "variable" if count == 1 else "variables"
        role_parts.append(f'{count} {unit} with semantic role "{role}"')
    if n_distractors:
        unit = "variable" if n_distractors == 1 else "variables"
        role_parts.append(f"{n_distractors} distractor/meaningless {unit}")
    role_text = "; ".join(role_parts) if role_parts else f"{n_total} candidate variables"

    return (
        f"There are {n_total} candidate variables in an unknown order. "
        f"The unordered semantic-role multiset is: {role_text}. "
        "The mapping from semantic roles to variable names is intentionally hidden; "
        "do not assume which x_i corresponds to which semantic role."
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

    # `feature_names` / `target_name` 始终表示真实数据契约；
    # LLM prompt 中的变量命名另用 `prompt_feature_names` / `prompt_target_name` 显式表达。
    # 对 llmsr/drsr 可通过 params_override 中的 anonymize 开关切到 x1..xN/y 且隐藏描述。
    anonymize = _as_bool(params.pop("anonymize", None), default=False)
    if anonymize:
        params["anonymize"] = True

    if tool_name in {"llmsr", "drsr"}:
        canonical_prompt_variables = _as_bool(
            params.get("canonical_prompt_variables"),
            default=True,
        )
        if anonymize:
            canonical_prompt_variables = True
        params["canonical_prompt_variables"] = canonical_prompt_variables
        params.setdefault("original_feature_names", list(dataset.feature_names))
        params.setdefault("original_target_name", dataset.target_name)

        if anonymize:
            params["prompt_feature_names"] = [f"x{i+1}" for i in range(len(dataset.feature_names))]
            params["prompt_target_name"] = "y"
        elif canonical_prompt_variables:
            params["prompt_feature_names"] = [f"x{i}" for i in range(len(dataset.feature_names))]
            params["prompt_target_name"] = "y"
        else:
            params.setdefault("prompt_feature_names", list(dataset.feature_names))
            params.setdefault("prompt_target_name", dataset.target_name)

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
                    params["feature_descriptions"] = [
                        "candidate variable; semantic role hidden"
                    ] * len(dataset.feature_names)
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
    train_label_noise = _freeze_train_label_noise_evidence(
        _resolve_train_label_noise_config(params, dataset=dataset, seed=seed)
    )
    y_train_for_fit = _train_labels_for_fit(dataset.train, train_label_noise)

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
    raw_execution_error = None
    recovered_from_error = False
    no_valid_output_reason = None
    internal_candidate_evidence: dict[str, Any] = {}

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
                "task_label": task_label,
                "task_global_index": task_global_index,
                "expected_dataset_rel": task_identity.get("expected_dataset_rel"),
                "expected_dataset_dir": task_identity.get("expected_dataset_dir"),
                "train_label_noise": train_label_noise,
            },
            daemon=True,
        )
        snapshot_thread.start()

    try:
        reg.fit(dataset.train.X, y_train_for_fit)
        experiment_dir = getattr(reg, "experiment_dir", experiment_dir)
        equation = reg.get_optimal_equation()
        canonical_artifact, canonical_artifact_error = safe_export_canonical_artifact(reg)
        try:
            equations = reg.get_total_equations()
            equation_count = len(equations) if isinstance(equations, list) else None
        except Exception:
            equation_count = None
        try:
            train_metrics = _evaluate_split(reg, dataset.train)
            valid_metrics = _evaluate_split(reg, dataset.valid)
            id_metrics = _evaluate_split(reg, dataset.id_test)
            ood_metrics = _evaluate_split(reg, dataset.ood_test)
        except Exception as exc:
            # 训练已结束但最终表达式无法预测，属于算法无可评估输出，
            # 不应被调度器视为系统错误而阻断整轮实验。
            no_valid_output_reason = f"evaluation_failed: {exc!r}"
            raise NoValidOutputError(no_valid_output_reason) from exc
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
            if str(tool_name).strip().lower() in {"imcts", "imcts_wrapper"}:
                internal_candidate_evidence = _imcts_candidate_evidence(recovered_payload)
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
    except NoValidOutputError as exc:
        status = "no_valid_output"
        error = None
        timeout_type = "no_valid_output"
        no_valid_output_reason = str(exc)
    except Exception as exc:
        raw_execution_error = repr(exc)
        experiment_dir = getattr(reg, "experiment_dir", experiment_dir)
        recovered_payload = None
        if _is_snapshot_capable_tool(tool_name):
            recovered_payload = _recover_timeout_payload_from_candidate(
                tool_name=tool_name,
                dataset=dataset,
                experiment_dir=experiment_dir,
            )
        if recovered_payload is not None:
            if str(tool_name).strip().lower() in {"imcts", "imcts_wrapper"}:
                internal_candidate_evidence = _imcts_candidate_evidence(recovered_payload)
            equation = recovered_payload["equation"]
            equation_count = recovered_payload["equation_count"]
            canonical_artifact = recovered_payload["canonical_artifact"]
            canonical_artifact_error = recovered_payload[
                "canonical_artifact_error"
            ]
            train_metrics = recovered_payload["train_metrics"]
            valid_metrics = recovered_payload["valid_metrics"]
            id_metrics = recovered_payload["id_metrics"]
            ood_metrics = recovered_payload["ood_metrics"]
            recovered_from_error = (
                _timeout_type_for_payload(
                    dataset=dataset,
                    equation=equation,
                    canonical_artifact=canonical_artifact,
                    valid_metrics=valid_metrics,
                    id_metrics=id_metrics,
                    ood_metrics=ood_metrics,
                )
                == "budget_exhausted_with_output"
            )
        if recovered_from_error:
            status = "ok"
            error = None
        else:
            status = "error"
            error = raw_execution_error
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
    result["raw_execution_error"] = raw_execution_error
    result["recovered_from_error"] = recovered_from_error
    result["no_valid_output_reason"] = no_valid_output_reason
    result["train_label_noise"] = train_label_noise
    result["condition"] = _condition_from_train_label_noise(train_label_noise)
    if str(tool_name).strip().lower() in {"imcts", "imcts_wrapper"}:
        if not internal_candidate_evidence and experiment_dir:
            internal_candidate_evidence = _imcts_candidate_evidence(
                _extract_imcts_periodic_candidate(experiment_dir)
            )
        result.update(internal_candidate_evidence)
    if budget_exhausted:
        result["termination_reason"] = timeout_type
    elif recovered_from_error:
        result["termination_reason"] = "recovered_after_error"
    elif status == "no_valid_output":
        result["termination_reason"] = "no_valid_output"
    else:
        result["termination_reason"] = "completed" if status == "ok" else status

    _write_final_progress_payload_if_requested(
        result=result,
        progress_snapshot_interval_seconds=progress_snapshot_interval_seconds,
        output_dir=output_dir,
        experiment_dir=experiment_dir,
        tool_name=tool_name,
        dataset=dataset,
        params=params,
        seed=seed,
        started_at=started_at,
        task_label=task_label,
        task_global_index=task_global_index,
        expected_dataset_rel=task_identity.get("expected_dataset_rel"),
        expected_dataset_dir=task_identity.get("expected_dataset_dir"),
        train_label_noise=train_label_noise,
    )

    result_path = output_dir / "result.json"
    write_result_payload(result, primary_path=result_path, experiment_dir=experiment_dir)
    return result_path
