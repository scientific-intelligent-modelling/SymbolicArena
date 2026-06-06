import os
import sys
import json
import glob
import tempfile
import time
import ast
import re
from typing import List, Optional
import textwrap

import numpy as np
import yaml

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_drsr_artifact
from scientific_intelligent_modelling.srkit import llm as srkit_llm
from scientific_intelligent_modelling.srkit.llm import ClientFactory, parse_provider_model
from scientific_intelligent_modelling.srkit.spec_builder import build_specification as build_shared_specification
from typing import Tuple

try:
    # 可选：用于参数拟合（与评估一致的 BFGS）
    from scipy.optimize import minimize
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


def _default_benchmark_llm_config_path() -> str:
    """返回 drsr / llmsr 共享的 benchmark LLM 配置路径。"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(
        repo_root,
        "exp-planning",
        "02.E1选择验证",
        "llm_configs",
        "benchmark_llm.config",
    )


class DRSRRegressor(BaseWrapper):
    _DEFAULT_MAX_PARAMS = 10
    """
    DRSR 封装对外接口对齐到 llmsr：
    - llm_config_path
    - background
    - metadata_path
    - niterations
    - samples_per_iteration
    - seed
    - problem_name
    - exp_path
    - exp_name

    内部预算映射：
    - max_samples = niterations * samples_per_iteration
    - samples_per_prompt = samples_per_iteration

    兼容说明：
    - 旧参数 max_samples / samples_per_prompt / workdir 仅作为 fallback 保留；
      如果显式提供了 niterations / samples_per_iteration / exp_path / exp_name，
      一律优先使用新接口。
    """

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
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

    def __init__(self, **kwargs):
        self.params = dict(kwargs) if kwargs else {}
        self.params.setdefault("timeout_in_seconds", 3600)
        self.params.setdefault("niterations", 100000)
        self.params.setdefault("samples_per_iteration", 4)
        self.params.setdefault("max_params", self._DEFAULT_MAX_PARAMS)
        self.params.setdefault("persist_all_samples", False)
        self.params.setdefault(
            "background",
            "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.",
        )
        self.params.setdefault("llm_config_path", _default_benchmark_llm_config_path())
        self.model_ready = False
        self._workdir: Optional[str] = None
        self._existing_exp_dir: Optional[str] = self.params.get("existing_exp_dir") or self.params.get("exp_dir")
        self._equation_body: Optional[str] = None
        self._equation_func = None
        self._all_bodies: List[str] = []
        self._equation_entries: List[dict] = []
        self._best_params: Optional[np.ndarray] = None
        self._n_features: Optional[int] = self.params.pop("n_features", None)  # 记录特征数量
        self._feature_names: Optional[List[str]] = self.params.pop("feature_names", None)
        self._target_name: Optional[str] = self.params.pop("target_name", None)
        self._prompt_feature_names: Optional[List[str]] = self.params.pop("prompt_feature_names", None)
        self._prompt_target_name: Optional[str] = self.params.pop("prompt_target_name", None)

        # 变量名匿名化只影响 prompt 命名；真实数据契约仍保存在 feature_names/target_name。
        self._anonymize: bool = self._as_bool(self.params.pop("anonymize", False), default=False)
        if self._anonymize and not self._prompt_feature_names:
            n = len(self._feature_names) if self._feature_names else (self._n_features or 0)
            self._prompt_feature_names = [f"x{i+1}" for i in range(n)]
            self._prompt_target_name = "y"

    def _resolve_experiment_layout(self) -> Tuple[str, str, str]:
        """
        统一解析 DRSR 的实验目录布局，优先对齐 llmsr 的 exp_path / exp_name。

        返回：
        - experiments_root: 实验根目录
        - exp_name: 实验名
        - workdir: drsr 实际写入目录
        """
        exp_path = self.params.get("exp_path")
        exp_name = self.params.get("exp_name")
        if isinstance(exp_path, str) and exp_path.strip() and isinstance(exp_name, str) and exp_name.strip():
            experiments_root = os.path.abspath(exp_path.strip())
            resolved_exp_name = exp_name.strip()
            workdir = os.path.join(experiments_root, resolved_exp_name)
            return experiments_root, resolved_exp_name, workdir

        user_workdir = self.params.get("workdir")
        if isinstance(user_workdir, str) and user_workdir.strip():
            workdir = os.path.abspath(user_workdir.strip())
            experiments_root = os.path.dirname(workdir)
            resolved_exp_name = os.path.basename(workdir)
            return experiments_root, resolved_exp_name, workdir

        default_base = os.getcwd()
        problem = str(self.params.get("problem_name") or "problem").strip()
        ts = time.strftime('%Y%m%d-%H%M%S')
        resolved_exp_name = f"drsr_{problem}_{ts}"
        experiments_root = os.path.join(default_base, "experiments")
        workdir = os.path.join(experiments_root, resolved_exp_name)
        return experiments_root, resolved_exp_name, workdir

    def _resolve_search_budget(self) -> Tuple[int, int, int]:
        """
        统一解析对外预算语义。

        主接口：
        - niterations
        - samples_per_iteration

        内部派生：
        - max_samples = niterations * samples_per_iteration
        - samples_per_prompt = samples_per_iteration
        """
        niterations = self.params.get("niterations")
        samples_per_iteration = self.params.get("samples_per_iteration")

        if niterations is not None or samples_per_iteration is not None:
            resolved_niterations = int(niterations if niterations is not None else 50)
            resolved_samples_per_iteration = int(samples_per_iteration if samples_per_iteration is not None else 4)
            if resolved_niterations <= 0:
                raise ValueError("DRSRRegressor: niterations 必须大于 0")
            if resolved_samples_per_iteration <= 0:
                raise ValueError("DRSRRegressor: samples_per_iteration 必须大于 0")
            max_samples = resolved_niterations * resolved_samples_per_iteration
            return resolved_niterations, resolved_samples_per_iteration, max_samples

        # fallback：兼容旧接口
        max_samples = int(self.params.get("max_samples", 2))
        samples_per_prompt = int(self.params.get("samples_per_prompt", 4))
        if samples_per_prompt <= 0:
            raise ValueError("DRSRRegressor: samples_per_prompt 必须大于 0")
        resolved_niterations = max(1, max_samples // samples_per_prompt)
        return resolved_niterations, samples_per_prompt, max_samples

    def _resolve_prompt_semantics(self, n_features: int) -> Tuple[List[str], List[Optional[str]], Optional[str]]:
        """
        为 spec 与外层 prompt 统一解析变量命名与物理语义。

        约定：
        - 外层 prompt 与 spec 一律使用 x0/x1/.../y
        - 若 metadata 中存在 description，则优先使用 description
        - 否则退化到 name
        """
        # 匿名化模式：prompt 使用 x1..xN/y，不加载任何描述；真实数据契约不改名。
        if self._anonymize:
            prompt_names = self._prompt_feature_names
            if not isinstance(prompt_names, list) or len(prompt_names) != n_features:
                prompt_names = [f"x{i+1}" for i in range(n_features)]
            return prompt_names, None, None

        feature_names = self._prompt_feature_names
        if not isinstance(feature_names, list) or len(feature_names) != n_features:
            feature_names = self._feature_names
        if not isinstance(feature_names, list) or len(feature_names) != n_features:
            feature_names = [f"x{i}" for i in range(n_features)]
        feature_descriptions = self.params.get("feature_descriptions")
        target_description = self.params.get("target_description")
        metadata_path = self.params.get("metadata_path")

        if (feature_descriptions is None or target_description is None) and metadata_path:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_root = yaml.safe_load(f)
                dataset_meta = meta_root.get("dataset", meta_root)
                features_meta = dataset_meta.get("features") or []
                if feature_descriptions is None:
                    feature_descriptions = []
                    for idx in range(n_features):
                        item = features_meta[idx] if idx < len(features_meta) else {}
                        if isinstance(item, dict):
                            feature_descriptions.append(item.get("description") or item.get("name"))
                        else:
                            feature_descriptions.append(None)
                if target_description is None:
                    target_meta = dataset_meta.get("target") or {}
                    if isinstance(target_meta, dict):
                        target_description = target_meta.get("description") or target_meta.get("name")
            except Exception:
                pass

        if feature_descriptions is None:
            normalized_feature_descriptions = None
        else:
            normalized_feature_descriptions = list(feature_descriptions[:n_features]) + [None] * max(0, n_features - len(feature_descriptions))
            if not any(item for item in normalized_feature_descriptions) and not target_description:
                normalized_feature_descriptions = None

        return feature_names, normalized_feature_descriptions, target_description

    def _resolve_prompt_target_name(self) -> str:
        prompt_target_name = self._prompt_target_name
        if isinstance(prompt_target_name, str) and prompt_target_name.strip():
            return prompt_target_name.strip()
        if self._anonymize:
            return "y"
        target_name = self._target_name
        if isinstance(target_name, str) and target_name.strip():
            return target_name.strip()
        return "y"

    @staticmethod
    def _resolve_api_key_from_config(api_key_cfg, model_name: str, provider: str):
        """兼容 llm.config 中 api_key 为字符串或字典两种形式。"""
        if isinstance(api_key_cfg, dict):
            def _get_case_insensitive(d: dict, key: str):
                for kk, vv in d.items():
                    try:
                        if str(kk).lower() == str(key).lower():
                            return vv
                    except Exception:
                        pass
                return None
            return _get_case_insensitive(api_key_cfg, model_name) or _get_case_insensitive(api_key_cfg, provider)
        if isinstance(api_key_cfg, str):
            return api_key_cfg
        return None

    @staticmethod
    def _normalize_base_url(base_url: Optional[str], host: Optional[str]) -> Optional[str]:
        """兼容 llm.config 中使用 host 或 base_url 两种字段。"""
        if isinstance(base_url, str) and base_url.strip():
            return base_url.strip()
        if not isinstance(host, str) or not host.strip():
            return None
        host = host.strip()
        if host.startswith("http://") or host.startswith("https://"):
            return host if host.rstrip("/").endswith("/v1") else host.rstrip("/") + "/v1"
        return f"https://{host}/v1"

    def _resolve_llm_client_config(self) -> dict:
        """
        统一解析 drsr 的 LLM 配置。

        优先级：
        1. 显式传入的 wrapper 参数
        2. llm_config_path 指向的 JSON 配置
        3. 环境变量兜底
        """
        cfg = {}
        llm_config_path = self.params.get("llm_config_path")
        if isinstance(llm_config_path, str) and llm_config_path.strip():
            with open(llm_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        model_name = self.params.get("api_model") or cfg.get("model")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("DRSRRegressor: 缺少 LLM 模型配置，请提供 api_model 或 llm_config_path")

        provider, _ = parse_provider_model(model_name)
        api_key = self.params.get("api_key")
        if not api_key:
            api_key = self._resolve_api_key_from_config(cfg.get("api_key"), model_name, provider)

        base_url = self._normalize_base_url(
            self.params.get("api_base") or cfg.get("base_url"),
            cfg.get("host"),
        )

        client_config = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
        }

        generation_overrides = {}
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "n",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "logprobs",
        ):
            if key in cfg:
                generation_overrides[key] = cfg[key]

        if self.params.get("temperature") is not None:
            generation_overrides["temperature"] = self.params["temperature"]
        if isinstance(self.params.get("api_params"), dict):
            generation_overrides.update(self.params["api_params"])

        return {
            "client_config": client_config,
            "generation_overrides": generation_overrides,
        }

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._n_features,
            feature_names=self._feature_names,
            target_name=self._target_name,
            context="DRSRRegressor.fit",
        )
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        
        # 记录特征数量
        self._n_features = X.shape[1] if X.ndim == 2 else 1

        _, _, self._workdir = self._resolve_experiment_layout()
        os.makedirs(self._workdir, exist_ok=True)
        try:
            print(f"[DRSR] 使用工作目录: {os.path.abspath(self._workdir)}")
        except Exception:
            pass

        niterations, samples_per_iteration, max_samples = self._resolve_search_budget()

        # 导入 drsr_420 模块
        drsr_dir = os.path.join(os.path.dirname(__file__), "drsr")
        sys.path.insert(0, drsr_dir)
        from drsr_420 import pipeline, config as config_lib, sampler, evaluator, evaluate_on_problems, data_analyse_real, prompt_config as prompt_config_lib

        # 优先从已有实验目录复用结果，避免再次调用 LLM/API（用于离线验收和快速恢复）。
        existing_exp_dir = self._existing_exp_dir
        if isinstance(existing_exp_dir, str):
            existing_exp_dir = existing_exp_dir.strip()
        if existing_exp_dir:
            self._workdir = os.path.abspath(existing_exp_dir)
            try:
                bodies, entries = self._read_experiment_outputs(self._workdir)
                if self._restore_from_experiences(X, y, bodies, entries):
                    print(f"[DRSR] 离线复用实验: {self._workdir}")
                    self.model_ready = True
                    return self
            except Exception:
                # 兼容路径不完整/内容不规范时，回退到正常训练流程
                pass
            print(f"[DRSR] 未找到可恢复的实验结果，回退到全量训练: {self._workdir}")

        # 规范文本：
        # 优先使用用户提供的背景描述自动生成通用 spec；否则回退到默认/显式 spec_path
        specification = None
        background = self.params.get("background")
        if isinstance(background, str) and background.strip():
            specification = self._build_spec_from_background(X, y, background)
            # 将生成的 spec 保存到工作目录，便于复现
            try:
                gen_spec_dir = os.path.join(self._workdir, "specs")
                os.makedirs(gen_spec_dir, exist_ok=True)
                gen_spec_path = os.path.join(gen_spec_dir, "generated_spec.txt")
                with open(gen_spec_path, "w", encoding="utf-8") as fp:
                    fp.write(specification)
            except Exception:
                pass
        else:
            spec_path = self.params.get("spec_path")
            if not spec_path:
                spec_path = os.path.join(drsr_dir, "specs", "specification_oscillator1_numpy.txt")
            with open(spec_path, "r", encoding="utf-8") as f:
                specification = f.read()

        # 数据（直接传给 pipeline，避免文件依赖）
        dataset = {"data": {"inputs": X, "outputs": y}}

        # 由 Wrapper 构建并注入单例 LLM 客户端；LocalLLM 仅负责 prompt 组织
        try:
            srkit_llm.reset_global_tokens()
            srkit_llm.reset_global_time()
        except Exception:
            pass
        llm_runtime = self._resolve_llm_client_config()
        client = ClientFactory.from_config(llm_runtime["client_config"])
        if isinstance(llm_runtime["generation_overrides"], dict):
            client.kwargs.update(llm_runtime["generation_overrides"])

        # 注入到 drsr 的 sampler 和 analyzer。
        # 兼容不同版本 drsr：若存在旧版本 set_shared_* 接口则使用，不存在则走现代 pipeline llm_client 注入路径。
        for _mod in (sampler, data_analyse_real):
            setter = getattr(_mod, "set_shared_llm_client", None)
            if callable(setter):
                try:
                    setter(client)
                except Exception:
                    pass
        llm_class = sampler.LocalLLM

        # 经验文件预创建（相对 self._workdir）
        exp_dir = os.path.join(self._workdir, "equation_experiences")
        os.makedirs(exp_dir, exist_ok=True)
        exp_file = os.path.join(exp_dir, "experiences.json")
        if not os.path.exists(exp_file):
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump({"None": [], "Good": [], "Bad": []}, f)

        # 组装配置并运行。
        # DRSR 现在默认不再使用 wall_time_limit_seconds 截断实验，
        # 统一按预算（niterations * samples_per_iteration）跑完。
        cls_cfg = config_lib.ClassConfig(llm_class=llm_class, sandbox_class=evaluator.LocalSandbox)
        cfg = config_lib.Config(
            num_samplers=1,
            num_evaluators=1,
            samples_per_prompt=samples_per_iteration,
            evaluate_timeout_seconds=int(self.params.get("evaluate_timeout_seconds", 10)),
            results_root=self._workdir,
            wall_time_limit_seconds=self._resolve_wall_time_limit_seconds(),
        )

        feature_names, feature_descriptions, target_description = self._resolve_prompt_semantics(self._n_features or 0)
        prompt_target_name = self._resolve_prompt_target_name()
        prompt_ctx = prompt_config_lib.PromptContext(
            n_features=self._n_features or 0,
            feature_names=feature_names or None,
            dependent_name=prompt_target_name,
            problem_name=self.params.get("problem_name"),
            background=background,
            feature_descriptions=feature_descriptions,
            target_description=target_description,
            max_params=self._max_params(),
        )

        # 切换 cwd 到工作目录，确保 drsr 相对路径输出写入其中
        cwd_backup = os.getcwd()
        os.chdir(self._workdir)
        try:
            pipeline.main(
                specification=specification,
                inputs=dataset,
                config=cfg,
                max_sample_nums=max_samples,
                class_config=cls_cfg,
                log_dir=self.params.get("log_dir") or os.path.join(self._workdir, "logs"),
                llm_client=client,
                prompt_ctx=prompt_ctx,
                persist_all_samples=bool(self.params.get("persist_all_samples", False)),
            )
        finally:
            os.chdir(cwd_backup)

        # 读取经验，选取最佳/首个方程
        bodies: List[str] = []
        try:
            bodies, entries = self._read_experiment_outputs(self._workdir)
            self._restore_from_experiences(X, y, bodies, entries)
            if self._equation_body:
                self._all_bodies = bodies
                if self._equation_func is not None:
                    self.model_ready = True
                    return self
        except Exception:
            pass

        self._equation_body = (
            "    return params[0] * x + params[1] * v + params[2]\n"
        )
        self._all_bodies = bodies or [self._equation_body]
        self._equation_func = self._compile_equation(self._equation_body, self._n_features)
        self._best_params = None
        try:
            print("[DRSR Wrapper] 采用内置默认方程，正在拟合参数...")
            self._best_params = self._fit_params(X, y, n_params=self._max_params(), n_starts=3)
            print("[DRSR Wrapper] 参数拟合完成")
        except Exception as e:
            print(f"[DRSR Wrapper] 参数拟合失败: {e}")
            self._best_params = np.ones(self._max_params())
        
        self.model_ready = True
        return self

    def _resolve_wall_time_limit_seconds(self):
        raw = self.params.get("timeout_in_seconds")
        try:
            raw = int(raw)
        except Exception:
            return None
        return raw if raw > 0 else None

    def serialize(self):
        state = {
            'params': self.params,
            'equation_body': self._equation_body,
            'all_bodies': self._all_bodies,
            'best_params': self._best_params.tolist() if isinstance(self._best_params, np.ndarray) else None,
            'n_features': self._n_features,
            'feature_names': self._feature_names,
            'target_name': self._target_name,
            'prompt_feature_names': self._prompt_feature_names,
            'prompt_target_name': self._prompt_target_name,
            'anonymize': self._anonymize,
        }
        return json.dumps(state)

    @staticmethod
    def _extract_equation_body(text: str) -> str:
        """兼容“仅函数体”和“完整 def equation(...) 定义”两种持久化格式。"""
        if not isinstance(text, str):
            return ""

        stripped = textwrap.dedent(text).strip("\n")
        if not stripped:
            return ""

        lines = stripped.splitlines()
        while lines and lines[0].lstrip().startswith("@"):
            lines = lines[1:]

        if lines and lines[0].lstrip().startswith("def "):
            try:
                tree = ast.parse(stripped)
                func = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
                if func is not None:
                    for node in reversed(func.body):
                        if isinstance(node, ast.Return):
                            expr = ast.get_source_segment(stripped, node.value)
                            if expr:
                                return f"return {expr.strip()}\n"
            except Exception:
                pass

            body = "\n".join(lines[1:])
            return textwrap.dedent(body).rstrip("\n") + "\n"

        return stripped.rstrip("\n") + "\n"

    @staticmethod
    def _sort_entries(entries: List[dict]) -> None:
        category_priority = {"Good": 2, "None": 1, "Bad": 0}
        entries.sort(
            key=lambda e: (
                category_priority.get(e.get("category"), 0),
                float("-inf") if e.get("score") is None else e.get("score"),
            ),
            reverse=True,
        )

    def _build_entry(self, item: dict, category: str) -> Optional[dict]:
        raw_equation = item.get("equation")
        if raw_equation is None:
            raw_equation = item.get("function")
        equation = self._extract_equation_body(raw_equation)
        if not equation.strip():
            return None

        params = item.get("fitted_params")
        if params is None:
            params = item.get("params")
        try:
            params_arr = np.asarray(params, dtype=float).tolist() if params is not None else None
        except Exception:
            params_arr = None

        return {
            "equation": equation,
            "params": params_arr,
            "score": item.get("score"),
            "category": category,
            "sample_order": item.get("sample_order"),
        }

    def _read_experience_entries(self, exp_file: str) -> tuple[List[str], List[dict]]:
        with open(exp_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        bodies: List[str] = []
        entries: List[dict] = []
        for key in ("Good", "None", "Bad"):
            for item in data.get(key, []):
                entry = self._build_entry(item, key)
                if entry is None:
                    continue
                entries.append(entry)
                bodies.append(entry["equation"])

        self._sort_entries(entries)
        return bodies, entries

    def _read_sample_json_entries(self, paths: List[str], category: str) -> tuple[List[str], List[dict]]:
        bodies: List[str] = []
        entries: List[dict] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                continue

            entry = self._build_entry(item, category)
            if entry is None:
                continue

            entries.append(entry)
            bodies.append(entry["equation"])

        self._sort_entries(entries)
        return bodies, entries

    def _merge_entry_groups(self, groups: List[tuple[List[str], List[dict]]]) -> tuple[List[str], List[dict]]:
        bodies: List[str] = []
        entries: List[dict] = []
        seen = set()
        for group_bodies, group_entries in groups:
            for body in group_bodies:
                if body not in bodies:
                    bodies.append(body)
            for entry in group_entries:
                dedup_key = (entry.get("sample_order"), entry.get("equation"))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                entries.append(entry)

        self._sort_entries(entries)
        return bodies, entries

    def _read_experiment_outputs(self, base_dir: str) -> tuple[List[str], List[dict]]:
        """兼容 DRSR 多种结果落盘路径与 JSON 协议。"""
        groups: List[tuple[List[str], List[dict]]] = []

        aggregate_files = [
            os.path.join(base_dir, "experiences.json"),
            os.path.join(base_dir, "equation_experiences", "experiences.json"),
        ]
        for path in aggregate_files:
            if os.path.isfile(path):
                try:
                    groups.append(self._read_experience_entries(path))
                except Exception:
                    continue

        flat_specs = [
            (sorted(glob.glob(os.path.join(base_dir, "best_history", "best_sample_*.json"))), "Good"),
            (sorted(glob.glob(os.path.join(base_dir, "samples", "top*.json"))), "Good"),
            (sorted(glob.glob(os.path.join(base_dir, "samples", "samples_*.json"))), "None"),
        ]
        for paths, category in flat_specs:
            if not paths:
                continue
            groups.append(self._read_sample_json_entries(paths, category))

        return self._merge_entry_groups(groups)

    def _restore_from_experiences(self, X: np.ndarray, y: np.ndarray, bodies: List[str], entries: List[dict]) -> bool:
        self._equation_entries = entries
        if not entries:
            return False

        self._all_bodies = bodies
        best_params = None

        for entry in entries:
            equation = entry.get("equation")
            if not isinstance(equation, str) or not equation.strip():
                continue
            try:
                cleaned_body = self._clean_equation_body(equation)
                canonical_body = self._canonicalize_equation_variable_names(cleaned_body)
                candidate_params = entry.get("params")
                candidate_params_arr = None
                if isinstance(candidate_params, list):
                    candidate_params_arr = np.asarray(candidate_params, dtype=float)
                candidate_func = self._compile_equation(canonical_body, self._n_features)
                if not self._validate_compiled_equation(candidate_func, X, candidate_params_arr):
                    continue
                entry["equation"] = canonical_body
                self._equation_body = canonical_body
                self._equation_func = candidate_func
                if candidate_params_arr is not None:
                    best_params = candidate_params_arr
                break
            except Exception:
                continue

        if self._equation_func is None:
            return False

        if best_params is not None:
            self._best_params = best_params
        elif _SCIPY_OK:
            try:
                print("[DRSR Wrapper] 未找到训练期参数，正在重新拟合...")
                self._best_params = self._fit_params(X, y, n_params=self._max_params(), n_starts=3)
                print(f"[DRSR Wrapper] 参数拟合完成")
            except Exception as e:
                print(f"[DRSR Wrapper] 参数拟合失败: {e}")
                self._best_params = None
        if not isinstance(self._best_params, np.ndarray):
            self._best_params = np.ones(self._max_params())
        self.model_ready = True
        return True

    @classmethod
    def deserialize(cls, payload: str):
        obj = json.loads(payload)
        params = dict(obj.get('params', {}))
        if obj.get('anonymize') is not None:
            params['anonymize'] = obj.get('anonymize')
        inst = cls(**params)
        inst._equation_body = obj.get('equation_body')
        inst._all_bodies = obj.get('all_bodies', [])
        inst._n_features = obj.get('n_features')
        inst._feature_names = obj.get('feature_names') or inst._feature_names
        inst._target_name = obj.get('target_name') or inst._target_name
        inst._prompt_feature_names = obj.get('prompt_feature_names') or inst._prompt_feature_names
        inst._prompt_target_name = obj.get('prompt_target_name') or inst._prompt_target_name
        inst._anonymize = cls._as_bool(obj.get('anonymize'), default=getattr(inst, "_anonymize", False))
        best_params = obj.get('best_params')
        inst._best_params = np.array(best_params) if best_params is not None else None
        if inst._equation_body:
            try:
                inst._equation_func = inst._compile_equation(
                    inst._equation_body,
                    inst._n_features,
                    feature_names=inst._feature_names,
                    prompt_feature_names=inst._prompt_feature_names,
                )
                inst.model_ready = True
            except Exception:
                inst._equation_func = None
                inst.model_ready = False
        return inst

    def predict(self, X):
        if not self.model_ready or self._equation_func is None:
            raise ValueError("模型尚未训练或方程不可用")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("DRSR 预测需要二维输入数组")
        if not isinstance(self._best_params, np.ndarray):
            raise RuntimeError("DRSRRegressor: 未找到训练期最优参数（fitted_params）。请检查 experiences.json 是否包含 fitted_params，或训练流程是否按期望运行。")
        params = self._best_params
        # 动态传递所有列
        return self._equation_func(*X.T, params)

    @staticmethod
    def _validate_compiled_equation(equation_func, X: np.ndarray, params: Optional[np.ndarray]) -> bool:
        """
        校验候选方程在 wrapper 的最终执行语境下能否独立预测。

        只要编译后的方程在一个很小的样本切片上无法正常执行、返回形状异常、
        或产生非有限值，就视为坏候选并直接淘汰。
        """
        if not callable(equation_func):
            return False

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            return False

        sample_rows = min(8, X.shape[0])
        sample_X = X[:sample_rows]
        if isinstance(params, np.ndarray):
            sample_params = params
        else:
            sample_params = np.ones(DRSRRegressor._DEFAULT_MAX_PARAMS, dtype=float)

        y_pred = np.asarray(equation_func(*sample_X.T, sample_params)).reshape(-1)
        if y_pred.shape != (sample_rows,):
            return False
        return bool(np.all(np.isfinite(y_pred)))

    def get_optimal_equation(self):
        if not self._equation_body:
            return ""
        # 清理方程体再包装显示
        cleaned_body = self._clean_equation_body(self._equation_body)
        return self._wrap_equation(cleaned_body, self._n_features)

    def _max_params(self) -> int:
        try:
            return int(self.params.get("max_params", self._DEFAULT_MAX_PARAMS))
        except Exception:
            return self._DEFAULT_MAX_PARAMS

    def get_total_equations(self, n=None):
        eqs = []
        for b in (self._all_bodies or []):
            if isinstance(b, str) and b.strip():
                cleaned_b = self._clean_equation_body(b)
                eqs.append(self._wrap_equation(cleaned_b, self._n_features))
        if n is not None:
            try:
                n = int(n)
                eqs = eqs[:max(0, n)]
            except Exception:
                pass
        return eqs

    def get_total_equations_with_params(self, n: Optional[int] = None) -> List[dict]:
        """
        返回包含方程、训练期拟合参数、分数等的列表（按分数降序）。
        每个元素包含：{'equation': def字符串, 'params': List[float]|None, 'score': float|None, 'category': str, 'sample_order': int|None}
        """
        items = self._equation_entries or []
        if n is not None:
            try:
                n = int(n)
                items = items[:max(0, n)]
            except Exception:
                pass
        out: List[dict] = []
        for e in items:
            out.append({
                'equation': self._wrap_equation(e.get('equation', '')),
                'params': e.get('params'),
                'score': e.get('score'),
                'category': e.get('category'),
                'sample_order': e.get('sample_order')
            })
        return out

    def get_fitted_params(self):
        """返回最佳方程的训练期拟合参数（列表形式），若不存在则返回 None。"""
        try:
            if isinstance(self._best_params, np.ndarray):
                return self._best_params.tolist()
        except Exception:
            pass
        # 尝试从 entries 的首项获取
        if getattr(self, '_equation_entries', None):
            p = self._equation_entries[0].get('params')
            return p
        return None

    def export_canonical_symbolic_program(self):
        eq_str = self.get_optimal_equation()
        if not eq_str:
            raise ValueError("DRSR 当前没有可导出的最优方程")
        return normalize_drsr_artifact(
            eq_str,
            parameter_values=self.get_fitted_params(),
            expected_n_features=self._n_features,
        )

    def __str__(self) -> str:
        lines = [f"DRSRRegressor(tool='drsr')"]
        try:
            eq_str = self.get_optimal_equation()
            if eq_str:
                lines.append("最佳方程:")
                lines.append(eq_str.rstrip())
        except Exception:
            pass
        try:
            if isinstance(self._best_params, np.ndarray):
                np.set_printoptions(precision=8, suppress=False)
                lines.append(f"最佳参数: {np.array2string(self._best_params, precision=8, suppress_small=False)}")
        except Exception:
            pass
        if self._equation_entries:
            k = min(3, len(self._equation_entries))
            lines.append(f"Top-{k} 候选（方程+分数简要）:")
            for i, e in enumerate(self._equation_entries[:k], 1):
                score = e.get('score')
                eq = e.get('equation') or ''
                eq_one_line = " ".join(eq.strip().split())
                if len(eq_one_line) > 120:
                    eq_one_line = eq_one_line[:120] + '...'
                lines.append(f"  {i}. score={score}, eq={eq_one_line}")
        return "\n".join(lines)

    @staticmethod
    def _wrap_equation(body: str, n_features: Optional[int] = None) -> str:
        """将方程体包装为完整的函数定义，动态适配特征数量。

        重要：统一使用 (col0, col1, ..., params) 的签名，以与采样产生的变量名保持一致，
        避免 body 中引用 col0/col1 而签名却是 (x, v) 导致的 NameError。
        """
        body = textwrap.dedent(body.rstrip("\n"))
        indented_lines = []
        for line in body.splitlines():
            if line.strip():
                indented_lines.append("    " + line)
            else:
                indented_lines.append("")
        body = "\n".join(indented_lines).rstrip("\n") + "\n"
        n = 2 if (n_features is None or n_features <= 0) else int(n_features)
        feature_names = ', '.join([f'col{i}' for i in range(n)])
        return f"def equation({feature_names}, params):\n" + body

    @staticmethod
    def _clean_equation_body(body: str) -> str:
        """
        清理方程体，只接受“单行 return 公式”。

        约束：
        - 仅允许一个有效语句；
        - 该语句必须是单行 `return ...`；
        - 多行 return、赋值后再 return、示例代码等一律拒绝。
        """
        if not isinstance(body, str):
            return ""

        lines = body.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            cleaned_lines.append(line)

        if len(cleaned_lines) != 1:
            return ""

        only_line = cleaned_lines[0].strip()
        if not only_line.startswith("return "):
            return ""
        if only_line == "return" or only_line == "return None":
            return ""

        return only_line + '\n'

    @staticmethod
    def _inject_feature_aliases(
        body: str,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        prompt_feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        当已有方程体仍使用旧版变量名（x/v/x0/x1）时，注入别名变量提升兼容性。
        """
        if not isinstance(body, str):
            return body
        n = 0 if (n_features is None or n_features <= 0) else int(n_features)
        variable_names = DRSRRegressor._collect_variable_names(body) or set()
        x_indices = sorted(
            {
                int(match.group(1))
                for name in variable_names
                for match in [re.fullmatch(r"x(\d+)", name)]
                if match is not None
            }
        )
        # 匿名化或部分上游 prompt 会生成 x1..xN。若表达式没有 x0，
        # 按 one-based 约定平移到 col0..colN-1，避免最后一个变量无法回放。
        one_based_x = bool(x_indices) and 0 not in x_indices and min(x_indices) >= 1 and max(x_indices) <= n
        aliases = []
        alias_map = {}
        if n >= 1:
            alias_map["x"] = "col0"
            alias_map["x0"] = "col0"
        if n >= 2:
            alias_map["v"] = "col1"
        if one_based_x:
            for i in range(1, n + 1):
                alias_map[f"x{i}"] = f"col{i - 1}"
        else:
            for i in range(n):
                alias_map[f"x{i}"] = f"col{i}"

        for candidate_names in (feature_names, prompt_feature_names):
            if not isinstance(candidate_names, list):
                continue
            for i, name in enumerate(candidate_names[:n]):
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if not name or not name.isidentifier() or name in {"np", "numpy", "params"}:
                    continue
                alias_map[name] = f"col{i}"

        for old_name, new_name in alias_map.items():
            if old_name in variable_names and old_name != new_name:
                aliases.append(f"{old_name} = {new_name}")
        # 去重，保持固定注入顺序
        aliases = list(dict.fromkeys(aliases))
        if not aliases:
            return body

        lines = body.split('\n')
        if not lines:
            return '\n'.join(aliases) + body

        # 若开头是 docstring，优先在 docstring 后注入别名，避免污染注释
        insert_at = 0
        first_non_empty = 0
        while first_non_empty < len(lines) and lines[first_non_empty].strip() == "":
            first_non_empty += 1
        if first_non_empty < len(lines):
            first = lines[first_non_empty].lstrip()
            if first.startswith('"""') or first.startswith("'''"):
                quote = '"""' if first.startswith('"""') else "'''"
                if first.count(quote) >= 2:
                    insert_at = first_non_empty + 1
                else:
                    end_idx = first_non_empty + 1
                    while end_idx < len(lines):
                        if quote in lines[end_idx]:
                            break
                        end_idx += 1
                    insert_at = min(end_idx + 1, len(lines))

        return "\n".join(lines[:insert_at] + aliases + lines[insert_at:])

    @staticmethod
    def _collect_variable_names(body: str) -> set:
        """提取方程体中读取的变量名（仅局部变量表达），用于判断别名注入。"""
        if not isinstance(body, str):
            return set()
        try:
            dedented = textwrap.dedent(body)
            tree = ast.parse(dedented)
        except Exception:
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
        return names

    def _canonicalize_equation_variable_names(self, body: str) -> str:
        """把候选中的真实特征名规范成 col0/col1/...，保证导出的方程自洽可执行。"""
        if not isinstance(body, str):
            return ""

        n = 0 if (self._n_features is None or self._n_features <= 0) else int(self._n_features)
        variable_names = self._collect_variable_names(body) or set()
        replacements: dict[str, str] = {}

        def add_names(names: Optional[List[str]]) -> None:
            if not isinstance(names, list):
                return
            for i, name in enumerate(names[:n]):
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if not name or not name.isidentifier() or name in {"np", "numpy", "params"}:
                    continue
                replacements[name] = f"col{i}"

        add_names(self._feature_names)
        add_names(self._prompt_feature_names)

        x_indices = sorted(
            {
                int(match.group(1))
                for name in variable_names
                for match in [re.fullmatch(r"x(\d+)", name)]
                if match is not None
            }
        )
        one_based_x = bool(x_indices) and 0 not in x_indices and min(x_indices) >= 1 and max(x_indices) <= n
        if n >= 1:
            replacements.setdefault("x", "col0")
            replacements.setdefault("x0", "col0")
        if n >= 2:
            replacements.setdefault("v", "col1")
        if one_based_x:
            for i in range(1, n + 1):
                replacements.setdefault(f"x{i}", f"col{i - 1}")
        else:
            for i in range(n):
                replacements.setdefault(f"x{i}", f"col{i}")

        active = {old: new for old, new in replacements.items() if old in variable_names and old != new}
        if not active:
            return body

        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(active, key=len, reverse=True)) + r")\b")
        return pattern.sub(lambda match: active[match.group(1)], body)

    @staticmethod
    def _compile_equation(
        body: str,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        prompt_feature_names: Optional[List[str]] = None,
    ):
        """编译方程，动态适配特征数量"""
        # 清理方程体，移除测试代码
        cleaned_body = DRSRRegressor._clean_equation_body(body)
        body_with_aliases = DRSRRegressor._inject_feature_aliases(
            cleaned_body,
            n_features,
            feature_names=feature_names,
            prompt_feature_names=prompt_feature_names,
        )
        code = DRSRRegressor._wrap_equation(body_with_aliases, n_features)
        ns = {}
        # 提供 numpy 命名以支持方程体中的 np.sin/np.cos 等写法
        try:
            import numpy as _np
            ns["np"] = _np
        except Exception:
            pass
        exec(code, ns)
        return ns["equation"]

    def _build_spec_from_background(self, X: np.ndarray, y: np.ndarray, background: str) -> str:
        """
        基于用户的背景知识动态生成通用 spec 文本（无需为每个数据集手写 spec 文件）。

        约定：
        - evaluate.run 不直接执行，在当前实现中 evaluator 使用统一的 `evaluate_on_problems.evaluate`，
          但仍需提供以满足解析与流程约束。
        - equation 接口为 `equation(*cols, params)`，pipeline/评估会以 `equation(*X.T, params)` 调用，
          从而适配任意特征维度。
        - 初始骨架为“少量线性项 + 偏置”的可运行实现，供 LLM 在此基础上演化。
        """
        try:
            n_features = int(X.shape[1]) if isinstance(X, np.ndarray) and X.ndim == 2 else 2
        except Exception:
            n_features = 2
        
        feature_names, feature_descriptions, target_description = self._resolve_prompt_semantics(n_features)
        return build_shared_specification(
            background=background,
            features=feature_names,
            target=self._resolve_prompt_target_name(),
            max_params=self._max_params(),
            problem=self.params.get("problem_name"),
            evaluate_style="drsr",
            feature_descriptions=feature_descriptions,
            target_description=target_description,
        )

    def _fit_params(self, X: np.ndarray, y: np.ndarray, n_params: int = 10, n_starts: int = 5) -> np.ndarray:
        """
        使用与评估相同思想的 BFGS 在训练集上拟合参数，并返回最优参数。
        动态适配特征数量。
        """
        eq = self._equation_func
        if not callable(eq):
            return np.ones(n_params)

        def loss_fn(p: np.ndarray) -> float:
            try:
                # 动态传递所有列
                y_pred = eq(*X.T, p)
                return float(np.mean((y_pred - y) ** 2))
            except Exception:
                # 不可导的坏点，返回大损失
                return 1e6

        best_p = None
        best_loss = None
        rng = np.random.default_rng(0)
        for _ in range(max(1, n_starts)):
            x0 = rng.uniform(low=-1.0, high=1.0, size=n_params)
            try:
                res = minimize(loss_fn, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10, 'eps': 1e-12, 'disp': False})
                cur_loss = float(res.fun)
                if best_loss is None or cur_loss < best_loss:
                    best_loss = cur_loss
                    best_p = np.array(res.x)
            except Exception:
                continue
        return best_p if isinstance(best_p, np.ndarray) else np.ones(n_params)
