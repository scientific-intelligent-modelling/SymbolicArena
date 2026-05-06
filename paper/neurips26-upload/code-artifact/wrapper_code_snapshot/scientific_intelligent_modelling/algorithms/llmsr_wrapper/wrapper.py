# algorithms/llmsr_wrapper/wrapper.py
"""
LLMSR 的 SIM 封装：

- 使用子仓库中的 `llmsr_regressor.LLMSRRegressor` 作为真正的算法实现；
- 训练时将内存中的 (X, y) 落盘为 CSV，交给 LLMSRRegressor 运行完整流水线；
- LLMSRRegressor 自身在 `exp_path/exp_name` 下持久化实验（meta.json、samples/top*.json 等）；
- 本 wrapper 的序列化只记录「元信息 + 实验目录」，反序列化后通过 `existing_exp_dir`
  恢复 LLMSRRegressor，并利用其 `predict` 中的持久化逻辑完成预测。

注意：
- 这里不再自己解析 samples 日志、也不覆写 LLM 调用逻辑，统一交给子仓库的实现；
- 你可以通过参数覆盖 problem_name / exp_path / exp_name / llm_config_path 等。
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import glob
from typing import Any, Dict, Optional, List
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_llmsr_artifact


def _llmsr_root_dir() -> str:
    """返回子仓库 llmsr 的根目录路径。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llmsr")


def _default_llm_config_path() -> str:
    """
    默认的 llm.config 路径。

    默认固定为 benchmark 统一配置路径，保证 llmsr / drsr 共享同一份
    LLM 采样口径。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(
        repo_root,
        "exp-planning",
        "02.E1选择验证",
        "llm_configs",
        "benchmark_llm.config",
    )


def _as_bool(value: Any, default: bool = False) -> bool:
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


def _import_core_regressor():
    """
    动态导入子仓库中的 LLMSRRegressor。

    之所以不在模块顶层导入，是为了避免在环境未就绪时导入失败，同时兼容
    子进程中的 import 行为。
    """
    import sys

    root = _llmsr_root_dir()
    if root not in sys.path:
        sys.path.insert(0, root)
    from llmsr_regressor import LLMSRRegressor as CoreLLMSRRegressor  # type: ignore

    return CoreLLMSRRegressor


def _is_single_line_formula_function(func_source: Any) -> bool:
    """接受仅包含字符串说明块和单个 return 的 equation 函数。"""
    if not isinstance(func_source, str) or not func_source.strip():
        return False
    try:
        tree = ast.parse(func_source)
    except Exception:
        return False

    equation_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            equation_func = node
            break
    if equation_func is None:
        return False

    body = list(equation_func.body)
    while (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return False
    return True


def _infer_n_features_from_function_signature(func_source: Any) -> int | None:
    """从 equation 函数签名推断输入特征数。"""
    if not isinstance(func_source, str) or not func_source.strip():
        return None
    try:
        tree = ast.parse(func_source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            arg_names = [arg.arg for arg in node.args.args]
            feature_names = [name for name in arg_names if name != "params"]
            return len(feature_names)
    return None


class LLMSRRegressor(BaseWrapper):
    """
    面向 SIM 的 LLMSR 封装器。

    关键设计：
    - `fit(X, y)`：将数据写成 CSV，构造子仓库的 LLMSRRegressor 并调用其 `fit()`；
    - `predict(X)`：基于实验目录（exp_dir）恢复 LLMSRRegressor，并调用其 `predict(X)`；
    - `serialize()/deserialize()`：只序列化参数和实验目录，真正的模型持久化由
      LLMSRRegressor 在实验目录下完成。

    常用可配置参数（通过 SymbolicRegressor(..., **params) 传入）：
    - problem_name: 实验/问题名（可选，不传则自动给一个默认名）
    - background:   背景描述，会进入 prompt
    - llm_config_path: 自定义 llm.config 路径（默认使用子仓库 llm.config）
    - exp_path:     实验根目录（默认 ./experiments）
    - exp_name:     实验子目录名（可选，不传则由子仓库根据时间戳生成）
    - max_params:   最大参数个数（默认 10）
    - niterations:  迭代次数（默认 2500）
    - samples_per_iteration: 每次迭代采样数（默认 4）
    - seed:         随机种子（可选，为空则不强制）
    """

    def __init__(self, **kwargs: Any):
        # 保留原始参数，便于反序列化和重复使用
        self.params: Dict[str, Any] = dict(kwargs) if kwargs else {}
        self.params.setdefault("timeout_in_seconds", 3600)
        self.params.setdefault("max_params", 10)
        self.params.setdefault("niterations", 100000)
        self.params.setdefault("samples_per_iteration", 4)
        self.params.setdefault("inject_prompt_semantics", False)
        self.params.setdefault(
            "background",
            "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
        )
        self.params.setdefault("persist_all_samples", False)
        self.params.setdefault("llm_config_path", _default_llm_config_path())

        # 子仓库的核心回归器实例（惰性创建）
        self._core: Optional[Any] = None

        # 实验相关元信息
        self._exp_dir: Optional[str] = self.params.pop("existing_exp_dir", None) or self.params.pop("exp_dir", None)
        self._problem_name: Optional[str] = self.params.get("problem_name")
        self._n_features: Optional[int] = self.params.pop("n_features", None)
        self._feature_names: Optional[list[str]] = self.params.pop("feature_names", None)
        self._target_name: Optional[str] = self.params.pop("target_name", None)
        self._prompt_feature_names: Optional[list[str]] = self.params.pop("prompt_feature_names", None)
        self._prompt_target_name: Optional[str] = self.params.pop("prompt_target_name", None)

    # ------------------------------------------------------------------
    # 序列化 / 反序列化：只记录元信息与实验目录
    # ------------------------------------------------------------------
    def serialize(self) -> str:
        """
        将当前 wrapper 的最小必要状态序列化为 JSON 字符串。

        这里只记录：
        - params: 初始化时传入的参数字典
        - exp_dir: 子仓库 LLMSRRegressor 使用的实验目录
        - problem_name: 问题名称（便于恢复 core）
        """
        state = {
            "params": self.params,
            "exp_dir": self._exp_dir,
            "problem_name": self._problem_name,
            "n_features": self._n_features,
            "feature_names": self._feature_names,
            "target_name": self._target_name,
            "prompt_feature_names": self._prompt_feature_names,
            "prompt_target_name": self._prompt_target_name,
        }
        return json.dumps(state, ensure_ascii=False)

    @classmethod
    def deserialize(cls, payload: str) -> "LLMSRRegressor":
        """
        从 JSON 字符串恢复 wrapper。

        注意：并不会立刻重新跑实验，只会恢复元信息。
        真正需要预测时，会基于 exp_dir 创建 core，并利用 existing_exp_dir
        进入「只预测模式」。
        """
        obj = json.loads(payload)
        inst = cls(**obj.get("params", {}))
        inst._exp_dir = obj.get("exp_dir")
        inst._problem_name = obj.get("problem_name") or inst.params.get("problem_name")
        inst._n_features = obj.get("n_features")
        inst._feature_names = obj.get("feature_names")
        inst._target_name = obj.get("target_name")
        inst._prompt_feature_names = obj.get("prompt_feature_names")
        inst._prompt_target_name = obj.get("prompt_target_name")
        return inst

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def _resolve_prompt_columns(self, n_features: int) -> tuple[list[str], str]:
        """解析传给 LLMSR 子仓库的 CSV 列名。

        `feature_names` / `target_name` 是真实数据契约；这里统一只使用
        `prompt_feature_names` / `prompt_target_name`，避免 LLMSR 和 DRSR
        在 prompt 变量命名上走不同隐式路径。
        """
        prompt_feature_names = self._prompt_feature_names
        if not isinstance(prompt_feature_names, list) or len(prompt_feature_names) != n_features:
            prompt_feature_names = [f"x{i}" for i in range(n_features)]
        prompt_target_name = self._prompt_target_name
        if not isinstance(prompt_target_name, str) or not prompt_target_name.strip():
            prompt_target_name = "y"
        return list(prompt_feature_names), prompt_target_name.strip()

    def _can_reuse_existing_experiment(self) -> bool:
        """判断是否可以直接复用已有实验目录而跳过在线训练。"""
        if not self._exp_dir:
            return False
        exp_dir = os.path.abspath(self._exp_dir)
        return (
            os.path.isdir(exp_dir)
            and os.path.isfile(os.path.join(exp_dir, "meta.json"))
            and os.path.isdir(os.path.join(exp_dir, "samples"))
        )

    def fit(self, X, y):
        """
        训练 LLMSR 模型。

        步骤：
        1. 将 (X, y) 写入临时 CSV；
        2. 构造子仓库 LLMSRRegressor 并调用其 fit()；
        3. 记录实验目录 exp_dir，后续通过 serialize()/deserialize() 持久化。
        """
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._n_features,
            feature_names=self._feature_names,
            target_name=self._target_name,
            context="LLMSRRegressor.fit",
        )
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).reshape(-1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X 与 y 的样本数量不一致: X.shape={X_arr.shape}, y.shape={y_arr.shape}")

        # 若显式传入了 existing_exp_dir，则优先离线复用已有实验目录。
        if self._can_reuse_existing_experiment():
            self._exp_dir = os.path.abspath(self._exp_dir)
            self._n_features = int(X_arr.shape[1])
            self._core = None
            return self

        # 推导 problem_name（允许用户通过参数显式指定）
        problem_name = (self._problem_name or "").strip()
        if not problem_name:
            problem_name = "llmsr_problem"
        self._problem_name = problem_name

        # 1) 将数据写入临时 CSV（只在本次训练中使用）
        tmp_dir = tempfile.mkdtemp(prefix="llmsr_data_")
        try:
            n_features = X_arr.shape[1]
            self._n_features = int(n_features)
            prompt_feature_names, prompt_target_name = self._resolve_prompt_columns(n_features)
            columns = prompt_feature_names + [prompt_target_name]
            data = np.column_stack([X_arr, y_arr])
            df = pd.DataFrame(data, columns=columns)
            csv_path = os.path.join(tmp_dir, f"{problem_name}.csv")
            df.to_csv(csv_path, index=False)

            # 2) 构造子仓库 LLMSRRegressor
            Core = _import_core_regressor()

            llm_config_path = self.params.get("llm_config_path") or _default_llm_config_path()
            exp_path = self.params.get("exp_path") or os.path.join(os.getcwd(), "experiments")
            exp_name = self.params.get("exp_name")

            max_params = int(self.params.get("max_params", 10))
            niterations = int(self.params.get("niterations", 2500))
            samples_per_iter = int(self.params.get("samples_per_iteration", 4))
            seed = self.params.get("seed")

            wandb_cfg = None
            if self.params.get("use_wandb"):
                # 从上游参数中获取数据集路径与名称（若有）
                dataset_path = self.params.get("train_path")
                dataset_name = self.params.get("dataset_name")
                prompts_type = self.params.get("prompts_type")
                wandb_cfg = {
                    "project": self.params.get("wandb_project"),
                    "entity": self.params.get("wandb_entity"),
                    "name": self.params.get("wandb_name"),
                    "group": self.params.get("wandb_group"),
                    "tags": self.params.get("wandb_tags"),
                }
                # 可选：记录当前使用的 prompts 类型/版本，便于在 WandB 中区分不同提示词设置
                if prompts_type is not None:
                    wandb_cfg["prompts_type"] = prompts_type
                if dataset_path is not None:
                    wandb_cfg["dataset_path"] = dataset_path
                if dataset_name is not None:
                    wandb_cfg["dataset_name"] = dataset_name

            core = Core(
                problem_name=problem_name,
                data_csv=csv_path,
                llm_config_path=llm_config_path,
                background=self.params.get("background", "") or "",
                exp_path=exp_path,
                exp_name=exp_name,
                max_params=max_params,
                niterations=niterations,
                samples_per_iteration=samples_per_iter,
                persist_all_samples=bool(self.params.get("persist_all_samples", False)),
                timeout_in_seconds=self.params.get("timeout_in_seconds"),
                seed=seed,
                metadata_path=self.params.get("metadata_path"),
                feature_descriptions=self.params.get("feature_descriptions"),
                target_description=self.params.get("target_description"),
                # wrapper 已经显式写好 prompt CSV 列名，不再依赖子仓库二次匿名化。
                anonymize=False,
                wandb_config=wandb_cfg,
            )
            core.fit()

            # 记录实验目录，后续序列化时只需要带上这个路径即可
            self._core = core
            self._exp_dir = getattr(core, "exp_dir_", None)
            if not self._exp_dir:
                # 理论上 core.fit() 会设置 exp_dir_，这里再做一次兜底
                self._exp_dir = os.path.join(exp_path, exp_name or problem_name)

        finally:
            # 临时数据只作为 fit 的输入，实验本身由 core 在 exp_path 下持久化
            import shutil

            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        return self

    # ------------------------------------------------------------------
    # 预测与方程获取
    # ------------------------------------------------------------------
    def _ensure_core(self) -> Any:
        """
        确保 self._core 可用。

        - 若已在当前进程中训练过，则直接复用；
        - 若是从序列化状态恢复，则基于 exp_dir 构造一个
          `existing_exp_dir` 模式的 LLMSRRegressor，仅用于预测和读取结果。
        """
        if self._core is not None:
            return self._core

        if not self._exp_dir:
            raise RuntimeError("LLMSRRegressor: 缺少实验目录 exp_dir，无法恢复模型状态")

        Core = _import_core_regressor()

        llm_config_path = self.params.get("llm_config_path") or _default_llm_config_path()
        problem_name = (self._problem_name or self.params.get("problem_name") or "llmsr_problem").strip()

        # 这里 data_csv 对预测模式并不重要，LLMSRRegressor 会从 meta.json 中恢复
        core = Core(
            problem_name=problem_name,
            data_csv="",
            llm_config_path=llm_config_path,
            background=self.params.get("background", "") or "",
            exp_path=os.path.dirname(self._exp_dir),
            exp_name=os.path.basename(self._exp_dir),
            max_params=int(self.params.get("max_params", 10)),
            niterations=int(self.params.get("niterations", 2500)),
            samples_per_iteration=int(self.params.get("samples_per_iteration", 4)),
            persist_all_samples=bool(self.params.get("persist_all_samples", False)),
            timeout_in_seconds=self.params.get("timeout_in_seconds"),
            seed=self.params.get("seed"),
            metadata_path=self.params.get("metadata_path"),
            feature_descriptions=self.params.get("feature_descriptions"),
            target_description=self.params.get("target_description"),
            existing_exp_dir=self._exp_dir,
        )
        # existing_exp_dir 模式下，LLMSRRegressor 会跳过部分元信息恢复判断，
        # 为保证 predict/方程恢复稳定，这里显式补齐关键元信息。
        try:
            if os.path.isfile(os.path.join(self._exp_dir, "meta.json")):
                with open(os.path.join(self._exp_dir, "meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if core.feature_names_ is None and meta.get("feature_names"):
                    core.feature_names_ = meta.get("feature_names")
                if core.target_name_ is None and meta.get("target_name"):
                    core.target_name_ = meta.get("target_name")
        except Exception:
            pass
        # 避免既有现成实验被当作已拟合模型直接跳过元信息/方程恢复路径
        core.is_fitted_ = False
        self._core = core
        return core

    def predict(self, X):
        """使用 LLMSR 已搜索到的最佳方程进行预测。"""
        core = self._ensure_core()
        X_arr = np.asarray(X)
        return core.predict(X_arr)

    # ------------------ 方程读取：从 samples/top*.json 中解析 ------------------
    def _load_best_sample(self) -> Optional[Dict[str, Any]]:
        """
        从实验目录的 samples 子目录中读取最优样本（参照子仓库 LLMSRRegressor 的逻辑）。

        返回:
            包含 'function'、'params'、'nmse'/'mse' 等字段的字典；若失败则返回 None。
        """
        if not self._exp_dir:
            return None

        samples_dir = os.path.join(self._exp_dir, "samples")
        if not os.path.isdir(samples_dir):
            return None

        candidates: List[str] = glob.glob(os.path.join(samples_dir, "top01_*.json"))
        if not candidates:
            candidates = glob.glob(os.path.join(samples_dir, "top*.json"))
        if not candidates:
            return None

        best_key: Optional[float] = None
        best_data: Optional[Dict[str, Any]] = None

        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            func = d.get("function")
            if not _is_single_line_formula_function(func):
                continue

            key_val: Optional[float] = None
            nmse = d.get("nmse")
            mse = d.get("mse")
            if isinstance(nmse, (int, float)):
                key_val = float(nmse)
            elif isinstance(mse, (int, float)):
                key_val = float(mse)
            else:
                score = d.get("score")
                if isinstance(score, (int, float)):
                    key_val = -float(score)

            if key_val is None:
                continue
            if best_key is None or key_val < best_key:
                best_key = key_val
                best_data = d

        return best_data

    def get_optimal_equation(self):
        """
        返回最优方程的函数字符串（即 top 样本中的 'function' 字段）。
        """
        best = self._load_best_sample()
        if not best:
            return ""
        func = best.get("function") or ""
        return str(func)

    def get_total_equations(self, n: Optional[int] = None):
        """
        返回所有候选 top*.json 中的函数字符串列表（按误差从小到大排序）。
        """
        if not self._exp_dir:
            return []

        samples_dir = os.path.join(self._exp_dir, "samples")
        if not os.path.isdir(samples_dir):
            return []

        paths = glob.glob(os.path.join(samples_dir, "top*.json"))
        items: List[Dict[str, Any]] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue

            key_val: Optional[float] = None
            nmse = d.get("nmse")
            mse = d.get("mse")
            if isinstance(nmse, (int, float)):
                key_val = float(nmse)
            elif isinstance(mse, (int, float)):
                key_val = float(mse)
            else:
                score = d.get("score")
                if isinstance(score, (int, float)):
                    key_val = -float(score)
            if key_val is None:
                continue
            d["_sort_key"] = key_val
            items.append(d)

        # 按误差从小到大排序
        items.sort(key=lambda d: d.get("_sort_key", float("inf")))

        if n is not None:
            try:
                n = int(n)
                items = items[: max(0, n)]
            except Exception:
                pass

        eqs: List[str] = []
        for d in items:
            func = d.get("function")
            if _is_single_line_formula_function(func):
                eqs.append(func)
        return eqs

    def export_canonical_symbolic_program(self):
        best = self._load_best_sample()
        if not best:
            raise ValueError("未找到可用的 LLMSR 候选样本")
        func = best.get("function") or ""
        if not _is_single_line_formula_function(func):
            raise ValueError("LLMSR 当前最优候选不是单行公式函数")
        params = best.get("params")
        if not isinstance(params, list):
            params = None
        expected_n_features = self._n_features
        if expected_n_features is None:
            expected_n_features = _infer_n_features_from_function_signature(func)
        return normalize_llmsr_artifact(
            str(func),
            parameter_values=params,
            expected_n_features=expected_n_features,
        )
