from __future__ import annotations

import ast
import os
import json
from datetime import datetime
import time
from typing import Optional, Union, List, Dict, Any

import numpy as np
import pandas as pd
import yaml

import llm
from llmsr import pipeline
from llmsr import config as config_mod
from llmsr import sampler
from llmsr import evaluator
from llmsr import prompts


class LLMSRRegressor:
    """LLM-SR 风格的回归器封装。

    设计目标：
    - __init__ 接收所有配置参数；
    - fit() 负责启动一次完整的搜索实验（等价于原 main.py 的逻辑）；
    - predict(X) 从实验目录中加载最佳方程进行预测。
    """

    def __init__(
        self,
        problem_name: str,
        data_csv: str,
        llm_config_path: str,
        background: str = "",
        exp_path: str = "./experiments",
        exp_name: Optional[str] = None,
        max_params: int = 10,
        niterations: int = 2500,
        samples_per_iteration: int = 4,
        timeout_in_seconds: Optional[int] = None,
        seed: Optional[int] = None,
        existing_exp_dir: Optional[str] = None,
        anonymize: bool = False,
        metadata_path: Optional[str] = None,
        feature_descriptions: Optional[List[Optional[str]]] = None,
        target_description: Optional[str] = None,
        persist_all_samples: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        # 训练相关配置
        self.problem_name = problem_name
        self.data_csv = data_csv
        self.llm_config_path = llm_config_path
        self.background = background
        self.exp_path = exp_path
        self.exp_name = exp_name
        self.max_params = max_params
        self.niterations = niterations
        self.samples_per_iteration = samples_per_iteration
        self.timeout_in_seconds = timeout_in_seconds
        self.seed = seed
        self.persist_all_samples = bool(persist_all_samples)
        self.wandb_config = wandb_config or {}
        self.metadata_path = metadata_path
        self.feature_descriptions_ = feature_descriptions
        self.target_description_ = target_description

        # 实验目录 / 元信息
        self.exp_dir_: Optional[str] = existing_exp_dir
        self.feature_names_: Optional[List[str]] = None
        self.target_name_: Optional[str] = None
        self.is_fitted_: bool = existing_exp_dir is not None

        # 是否对变量名进行匿名化（使用 x1,x2,...,y）
        self.anonymize: bool = anonymize

        # 预测时缓存的方程与参数
        self._equation_func = None
        self.params_: Optional[List[float]] = None
        self.best_nmse_: Optional[float] = None
        self.best_mse_: Optional[float] = None
        self.best_equation_str_: Optional[str] = None

        # WandB 运行句柄
        self._wandb_run = None

    # --------------------
    # 内部工具方法
    # --------------------
    @staticmethod
    def _is_single_line_formula_function(func_source: Any) -> bool:
        """接受仅包含字符串说明块和单个 return 的 equation 候选。"""
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

    def _build_dataset_and_spec(self) -> tuple[dict, str]:
        """读取 CSV，构造 dataset 与 specification 文本。"""
        df = pd.read_csv(self.data_csv)
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("CSV 至少需要 2 列（前 n-1 列为特征，最后 1 列为目标）")

        raw_feature_names = cols[:-1]
        raw_target_name = cols[-1]

        # 根据 anonymize 开关决定使用原始列名还是匿名变量名
        if self.anonymize:
            n = len(cols)
            # 特征列统一命名为 x1, x2, ..., x{n-1}，目标列为 y
            new_cols = [f"x{i+1}" for i in range(n - 1)] + ["y"]
            df.columns = new_cols
            self.feature_names_ = new_cols[:-1]
            self.target_name_ = new_cols[-1]
        else:
            self.feature_names_ = cols[:-1]
            self.target_name_ = cols[-1]

        feature_descriptions = self.feature_descriptions_
        target_description = self.target_description_
        if (feature_descriptions is None or target_description is None) and self.metadata_path:
            try:
                meta_root = yaml.safe_load(open(self.metadata_path, encoding="utf-8"))
                dataset_meta = meta_root.get("dataset", meta_root)
                feature_meta = dataset_meta.get("features") or []
                feature_by_name = {
                    str(item.get("name")): item.get("description")
                    for item in feature_meta
                    if isinstance(item, dict)
                }
                if feature_descriptions is None:
                    feature_descriptions = [feature_by_name.get(name) for name in raw_feature_names]
                if target_description is None:
                    target_meta = dataset_meta.get("target") or {}
                    if isinstance(target_meta, dict):
                        target_description = target_meta.get("description")
            except Exception:
                pass
        self.feature_descriptions_ = feature_descriptions
        self.target_description_ = target_description

        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        dataset = {"data": {"inputs": X, "outputs": y}}

        specification = prompts.build_specification(
            self.background or "",
            self.feature_names_,
            self.target_name_,
            max_params=self.max_params,
            problem=self.problem_name,
            feature_descriptions=self.feature_descriptions_,
            target_description=self.target_description_,
        )
        return dataset, specification

    def _prepare_exp_dir(self) -> str:
        """创建实验目录并返回路径。"""
        if self.exp_dir_ is not None:
            os.makedirs(self.exp_dir_, exist_ok=True)
            return self.exp_dir_

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_name = f"{self.problem_name}_{ts}" if self.problem_name else f"exp_{ts}"
        exp_name = self.exp_name or default_name
        exp_dir = os.path.join(self.exp_path, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        self.exp_dir_ = exp_dir
        return exp_dir

    def _build_llm_client(self):
        """根据 llm.config 构造 LLM 客户端。"""
        with open(self.llm_config_path, encoding="utf-8") as f:
            llm_cfg = json.load(f)

        client = llm.ClientFactory.from_config(llm_cfg)
        for k in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "n",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "stop",
        ):
            if k in llm_cfg:
                client.kwargs[k] = llm_cfg[k]
        return client

    def _save_meta(self, specification: str):
        """在实验目录下保存动态规格与元信息，便于后续只预测模式使用。"""
        if not self.exp_dir_:
            return

        # 尝试读取 llm.config 中的模型信息（不保存 api_key 等敏感字段）
        llm_model = None
        try:
            with open(self.llm_config_path, encoding="utf-8") as f:
                _llm_cfg = json.load(f)
            llm_model = _llm_cfg.get("model")
        except Exception:
            pass

        # 保存动态规格
        spec_path = os.path.join(self.exp_dir_, "spec_dynamic.txt")
        with open(spec_path, "w", encoding="utf-8") as fw:
            fw.write(specification)

        # 保存元信息
        meta = {
            "problem_name": self.problem_name,
            "data_csv": os.path.abspath(self.data_csv),
            "llm_config_path": os.path.abspath(self.llm_config_path),
            "llm_model": llm_model,
            "background": self.background,
            "exp_path": os.path.abspath(self.exp_path),
            "exp_name": self.exp_name or os.path.basename(self.exp_dir_),
            "exp_dir": os.path.abspath(self.exp_dir_),
            "feature_names": self.feature_names_,
            "feature_descriptions": self.feature_descriptions_,
            "target_name": self.target_name_,
            "target_description": self.target_description_,
            "max_params": self.max_params,
            "niterations": self.niterations,
            "samples_per_iteration": self.samples_per_iteration,
            "persist_all_samples": self.persist_all_samples,
            "timeout_in_seconds": self.timeout_in_seconds,
            "seed": self.seed,
        }
        meta_path = os.path.join(self.exp_dir_, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fw:
            json.dump(meta, fw, ensure_ascii=False, indent=2)

    def _init_wandb(self, dataset: Dict[str, Any]):
        """初始化 WandB 运行（若启用）。"""
        if not self.wandb_config or not self.wandb_config.get("project"):
            return
        try:
            import wandb
        except Exception:
            print("[LLMSR] 未安装 wandb，跳过 WandB 记录。")
            return

        tags = self.wandb_config.get("tags")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        # prompts_type：用于标记当前使用的提示词/规格类型/版本
        prompts_type = self.wandb_config.get("prompts_type")
        # 若提供了 prompts_type，则将其加入 tags（避免重复）
        if prompts_type:
            if tags is None:
                tags = [prompts_type]
            else:
                try:
                    if isinstance(tags, (list, tuple)) and prompts_type not in tags:
                        tags = list(tags) + [prompts_type]
                except Exception:
                    pass

        inputs = dataset.get("data", {}).get("inputs")
        outputs = dataset.get("data", {}).get("outputs")
        num_samples = int(outputs.shape[0]) if hasattr(outputs, "shape") else None
        num_features = int(inputs.shape[1]) if hasattr(inputs, "shape") else None

        # 数据集相关信息（基础统计 + 路径/名称）
        dataset_info: Dict[str, Any] = {
            "num_samples": num_samples,
            "num_features": num_features,
        }
        ds_path = self.wandb_config.get("dataset_path")
        if ds_path is not None:
            dataset_info["path"] = ds_path
        ds_name = self.wandb_config.get("dataset_name")
        if ds_name is not None:
            dataset_info["name"] = ds_name

        config_payload = {
            "algorithm": "llmsr",
            "problem_name": self.problem_name,
            "background": self.background,
            "llm_config_path": os.path.abspath(self.llm_config_path),
            "exp_path": os.path.abspath(self.exp_path),
            "exp_name": self.exp_name,
            "max_params": self.max_params,
            "niterations": self.niterations,
            "samples_per_iteration": self.samples_per_iteration,
            "persist_all_samples": self.persist_all_samples,
            "seed": self.seed,
            "anonymize": self.anonymize,
            "dataset": dataset_info,
        }
        # 将 prompts_type 作为单独字段写入 config，便于后续分析/筛选
        if prompts_type:
            config_payload["prompts_type"] = prompts_type

        try:
            self._wandb_run = wandb.init(
                project=self.wandb_config.get("project"),
                entity=self.wandb_config.get("entity"),
                name=self.wandb_config.get("name"),
                group=self.wandb_config.get("group"),
                tags=tags,
                config=config_payload,
            )
        except Exception as e:
            print(f"[LLMSR] 初始化 WandB 失败，跳过记录: {e}")
            self._wandb_run = None

    def _load_best_equation(self):
        """从 samples 目录中加载当前实验的最优方程。"""
        if self.exp_dir_ is None:
            raise RuntimeError("exp_dir_ 未设置，无法加载最佳方程")

        samples_dir = os.path.join(self.exp_dir_, "samples")
        if not os.path.isdir(samples_dir):
            raise RuntimeError(f"找不到 samples 目录: {samples_dir}")

        # 优先尝试 top01_*.json；如果没有，则在所有 top*.json 中按 NMSE/MSE 最小选择
        import glob

        candidates = glob.glob(os.path.join(samples_dir, "top01_*.json"))
        if not candidates:
            candidates = glob.glob(os.path.join(samples_dir, "top*.json"))
        if not candidates:
            raise RuntimeError("在 samples 目录下没有找到任何 top*.json，说明搜索可能完全失败")

        best_key = None
        best_data = None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            func_str = d.get("function", "")
            if not self._is_single_line_formula_function(func_str):
                continue
            # 新格式优先：nmse / mse
            key_val = None
            nmse = d.get("nmse")
            mse = d.get("mse")
            if isinstance(nmse, (int, float)):
                key_val = float(nmse)
            elif isinstance(mse, (int, float)):
                key_val = float(mse)
            else:
                # 兼容旧格式：score = -MSE，score 越大越好
                score = d.get("score")
                if isinstance(score, (int, float)):
                    key_val = -float(score)

            if key_val is None:
                continue
            if best_key is None or key_val < best_key:
                best_key = key_val
                best_data = d

        if best_data is None:
            raise RuntimeError("未能在 top*.json 中找到可用的样本（mse/nmse/score 均缺失）")

        func_str = best_data.get("function", "")
        self.params_ = best_data.get("params")
        nmse = best_data.get("nmse")
        mse = best_data.get("mse")
        if nmse is None and mse is not None and self.target_name_:
            nmse = mse
        self.best_nmse_ = nmse
        self.best_mse_ = mse
        self.best_equation_str_ = func_str

        # 动态执行方程定义
        global_ns = {"np": np}
        local_ns: dict = {}
        exec(func_str, global_ns, local_ns)
        equation = local_ns.get("equation") or global_ns.get("equation")
        if equation is None:
            raise RuntimeError("从 function 字符串中没有解析出 equation 函数")
        self._equation_func = equation

    # --------------------
    # 公共接口
    # --------------------
    def fit(self):
        """启动一次完整的搜索实验。"""
        start_time = time.time()
        # 设置本地随机种子（与 main.py 保持一致，只影响本地数值库）
        if self.seed is not None:
            try:
                import random as _random

                os.environ["PYTHONHASHSEED"] = str(self.seed)
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
                _random.seed(self.seed)
                np.random.seed(self.seed)
            except Exception:
                pass

        # 重置本次实验的大模型统计信息
        try:
            llm.reset_global_tokens()
            llm.reset_global_time()
        except Exception:
            pass

        # 数据与规格
        dataset, specification = self._build_dataset_and_spec()
        self._init_wandb(dataset)

        # 实验目录
        exp_dir = self._prepare_exp_dir()
        self._save_meta(specification)

        # LLM 客户端与配置
        client = self._build_llm_client()
        class_config = config_mod.ClassConfig(
            llm_class=sampler.LocalLLM,
            sandbox_class=evaluator.LocalSandbox,
        )
        cfg = config_mod.Config(
            samples_per_prompt=self.samples_per_iteration,
            wall_time_limit_seconds=self._resolve_wall_time_limit_seconds(),
        )

        niterations = int(max(1, self.niterations))
        samples_per_iter = int(max(1, self.samples_per_iteration))
        max_samples = niterations * samples_per_iter

        # 启动流水线
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=cfg,
            max_sample_nums=max_samples,
            class_config=class_config,
            log_dir=exp_dir,
            persist_all_samples=self.persist_all_samples,
            llm_client=client,
            seed=self.seed,
            wandb_run=self._wandb_run,
        )

        # 搜索结束后加载最佳方程
        self._load_best_equation()
        self.is_fitted_ = True

        # 结束时记录摘要指标
        runtime_seconds = round(time.time() - start_time, 2)
        try:
            tokens = llm.get_global_tokens()
        except Exception:
            tokens = None
        try:
            total_llm_time = llm.get_global_time()
        except Exception:
            total_llm_time = None

        if self._wandb_run:
            summary_payload: Dict[str, Any] = {
                "best_nmse": self.best_nmse_,
                "best_mse": self.best_mse_,
                "best_equation": self.best_equation_str_ or "",
                "runtime_seconds": runtime_seconds,
            }
            if tokens is not None:
                summary_payload["llm_tokens"] = tokens
            if total_llm_time is not None:
                summary_payload["total_llm_time_seconds"] = round(float(total_llm_time), 2)
            try:
                self._wandb_run.log(summary_payload)
                self._wandb_run.finish()
            except Exception as e:
                print(f"[LLMSR] WandB 记录失败（summary），已忽略: {e}")

        return self

    def _resolve_wall_time_limit_seconds(self) -> Optional[int]:
        try:
            raw = int(self.timeout_in_seconds) if self.timeout_in_seconds is not None else None
        except Exception:
            return None
        return raw if raw and raw > 0 else None

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """使用已搜索到的最佳方程，对新样本进行预测。"""
        if not self.is_fitted_:
            if self.exp_dir_ is None:
                raise RuntimeError("模型尚未 fit 且未指定 existing_exp_dir")
            # 仅预测模式下，需要从磁盘恢复元信息与方程
            meta_path = os.path.join(self.exp_dir_, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.feature_names_ = meta.get("feature_names") or self.feature_names_
                    self.target_name_ = meta.get("target_name") or self.target_name_
                except Exception:
                    pass
            self._load_best_equation()
            self.is_fitted_ = True

        if self._equation_func is None:
            self._load_best_equation()

        if self.feature_names_ is None:
            raise RuntimeError("缺少 feature_names_ 信息，无法确定 X 的列顺序")

        # 统一转换成 ndarray，并保证列顺序一致
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names_].to_numpy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError("X 必须是二维数组或 DataFrame")
            if X_arr.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"预测特征维度不匹配：X.shape[1]={X_arr.shape[1]}, expected={len(self.feature_names_)}"
                )

        features_for_equation = [X_arr[:, i] for i in range(X_arr.shape[1])]
        params = np.array(self.params_) if self.params_ is not None else None

        if params is not None:
            y_pred = self._equation_func(*features_for_equation, params)
        else:
            y_pred = self._equation_func(*features_for_equation)

        return np.asarray(y_pred)
