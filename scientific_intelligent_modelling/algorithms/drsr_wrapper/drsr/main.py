
import json
import os
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sys
import logging as pylogging

from drsr_420 import pipeline
from drsr_420 import config
from drsr_420 import sampler
from drsr_420 import evaluator
import llm as llm_mod


parser = ArgumentParser()
# 共有参数
parser.add_argument('--problem_name', type=str, default="problem")
parser.add_argument('--data_csv', type=str, required=True, help='含表头的 CSV，前 n-1 列为特征，最后一列为因变量')
parser.add_argument('--experiment_dir', type=str, default=None, help='实验目录（可为绝对/相对路径）。如提供，将直接使用此目录')
parser.add_argument('--niterations', type=int, default=10, help='搜索迭代轮数；若提供，将按 niterations * num_samplers * samples_per_iteration 计算最大采样数')
parser.add_argument('--timeout_in_seconds', type=int, default=None, help='总超时时间（秒）。到达即停止训练（即便未达迭代数）')
parser.add_argument('--seed', type=int, default=None, help='随机种子（影响 Python 与 NumPy 的随机性）')
parser.add_argument('--num_islands', type=int, default=None, help='经验缓冲的岛屿数量（覆盖默认 10）')

# 算法私有参数
parser.add_argument('--llm_config', type=str, default='llm.config', help='LLM 配置文件路径（JSON 格式）')
parser.add_argument('--background', type=str, default=None, help='背景知识（可选）')
parser.add_argument('--samples_per_iteration', type=int, default=None, help='每轮生成的候选数量（覆盖 config 默认值）')


args = parser.parse_args()

if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)

    # 计算实验总时长（秒）
    wall_limit_seconds = int(args.timeout_in_seconds) if args.timeout_in_seconds and args.timeout_in_seconds > 0 else None

    # 设定随机种子（Python/NumPy）
    try:
        if args.seed is not None:
            import random as _random
            import numpy as _np
            os.environ["PYTHONHASHSEED"] = str(args.seed)
            _random.seed(args.seed)
            _np.random.seed(args.seed)
            print(f"[INFO] Random seed set: {args.seed}")
    except Exception as _e:
        print(f"[WARN] Failed to set seed: {_e}")


    # 构建统一的结果目录（新方案）：
    # 优先使用 --experiment_root；否则使用 {experiments_base}/{experiment_name}
    # 默认 experiments_base = 'experiments'，experiment_name = '{problem_name}_{timestamp}'
    if args.experiment_dir:
        results_root = args.experiment_dir
    else:
        ts = time.strftime('%Y%m%d-%H%M%S')
        exp_name = f"{args.problem_name}_{ts}"
        results_root = os.path.join('experiments', exp_name)
    os.makedirs(results_root, exist_ok=True)
    # 所有产物直接放在实验根目录（不再创建 logs 子目录）

    # 将标准输出和错误输出同时写入结果目录，便于统一归档
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    out_path = os.path.join(results_root, "run.out")
    err_path = os.path.join(results_root, "run.err")
    out_fp = open(out_path, "a", encoding="utf-8")
    err_fp = open(err_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, out_fp)
    sys.stderr = _Tee(sys.stderr, err_fp)
    print(f"[INFO] Results root: {results_root}")

    # 不再设置独立日志目录

    # 统一配置 Python 日志格式，包含时间戳，影响 absl 日志输出
    try:
        pylogging.basicConfig(
            level=pylogging.INFO,
            format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True,
        )
    except TypeError:
        # 兼容旧版 Python（无 force 参数）
        pylogging.basicConfig(
            level=pylogging.INFO,
            format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    # 允许从命令行覆盖 samples_per_iteration（映射到 Config.samples_per_prompt）
    # 根据命令行构造 ExperienceBuffer 配置（允许覆盖岛屿数量）
    eb_cfg = config.ExperienceBufferConfig(
        num_islands=int(args.num_islands) if args.num_islands and args.num_islands > 0 else config.ExperienceBufferConfig().num_islands
    )

    if args.samples_per_iteration is not None and args.samples_per_iteration > 0:
        config = config.Config(
            results_root=results_root,
            samples_per_prompt=int(args.samples_per_iteration),
            wall_time_limit_seconds=wall_limit_seconds,
            experience_buffer=eb_cfg,
        )
    else:
        config = config.Config(
            results_root=results_root,
            wall_time_limit_seconds=wall_limit_seconds,
            experience_buffer=eb_cfg,
        )
    # 读取 LLM 配置（仅从 llm.config 文件加载模型名，不再支持命令行覆盖）
    import json as _json
    if not os.path.exists(args.llm_config):
        try:
            with open(args.llm_config, 'w', encoding='utf-8') as f:
                _json.dump({
                    # api_key 支持按提供商或完整模型配置不同 key；可留空
                    'api_key': {
                        'cstcloud': '',
                        'deepseek': '',
                        'siliconflow': '',
                        'blt': ''
                    },
                    # 强制要求带提供商前缀
                    'model': 'CSTCloud/gpt-oss-120b',
                    'max_tokens': 1024,
                    'temperature': 0.6,
                    'top_p': 0.3,
                    'top_k': 30
                }, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Generated default LLM config at {args.llm_config}")
        except Exception as e:
            print(f"[WARN] Failed to create default llm.config: {e}")
    with open(args.llm_config, 'r', encoding='utf-8') as f:
        llm_config = _json.load(f)
    # 构造一次性的 LLM 客户端实例（按任务传递，避免并行任务相互干扰）
    # 模型名格式：provider/model，例如 CSTCloud/gpt-oss-120b
    model_name = llm_config.get('model')
    if not model_name or '/' not in model_name:
        raise ValueError("缺少模型提供商：请在 llm.config 的 model 字段使用 'provider/model' 格式，例如 'CSTCloud/gpt-oss-120b'")
    provider, pure_model = llm_mod.parse_provider_model(model_name)

    # 解析 api_key：支持字符串与字典（按 provider 或完整 model）
    api_key_cfg = llm_config.get('api_key', '')
    api_key = ''
    if isinstance(api_key_cfg, dict):
        def _get_case_insensitive(d: dict, k: str):
            for kk, vv in d.items():
                try:
                    if str(kk).lower() == str(k).lower():
                        return vv
                except Exception:
                    pass
            return None
        api_key = _get_case_insensitive(api_key_cfg, model_name) or _get_case_insensitive(api_key_cfg, provider) or ''
    elif isinstance(api_key_cfg, str):
        api_key = api_key_cfg
    provider = (provider or 'bltcy').lower()
    client = None
    try:
        if provider in ('bltcy', 'blt'):
            client = llm_mod.BltClient(api_key=api_key, model=pure_model)
        elif provider in ('deepseek',):
            client = llm_mod.DeepSeekClient(api_key=api_key, model=pure_model)
        elif provider in ('siliconflow', 'sliconflow'):
            client = llm_mod.SiliconflowClient(api_key=api_key, model=pure_model)
        elif provider in ('deepinfra', 'deep-infra'):
            client = llm_mod.DeepInfraClient(
                api_key=api_key,
                model=pure_model,
                base_url=llm_config.get('base_url') or 'https://api.deepinfra.com/v1/openai',
            )
        elif provider in ('ollama', 'local'):
            client = llm_mod.OllamaClient(api_key=api_key, model=pure_model)
        elif provider in ('cstcloud', 'cst', 'cst-cloud', 'keji', 'keji-yun'):
            client = llm_mod.CSTCloudClient(api_key=api_key, model=pure_model)
        else:
            # 默认走 BLT 网关（OpenAI 兼容）
            client = llm_mod.BltClient(api_key=api_key, model=pure_model)
        # 将部分生成参数写入 client.kwargs
        client.kwargs.update({
            'max_tokens': int(llm_config.get('max_tokens', 1024) or 1024),
            'temperature': float(llm_config.get('temperature', 0.6) or 0.6),
            'top_p': float(llm_config.get('top_p', 0.3) or 0.3),
            'top_k': int(llm_config.get('top_k', 30) or 30),
        })
        print(f"[INFO] LLM client initialized: provider={provider}, model={pure_model}")
    except Exception as e:
        print(f"[WARN] Failed to init LLM client: {e}")
    # 计算最大采样数量：优先由 --niterations 推导；否则使用默认 1000
    if args.niterations is not None and args.niterations > 0:
        try:
            global_max_sample_num = int(args.niterations) * int(getattr(config, 'num_samplers', 1)) * int(getattr(config, 'samples_per_prompt', 4))
        except Exception:
            global_max_sample_num = int(args.niterations) * 1 * 4
    else:
        # global_max_sample_num = 10000
        global_max_sample_num = 1000


    # ===============
    # 工具：从 CSV 读取数据与列名
    # ===============
    def _load_from_csv(path: str):
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError('CSV 至少需要两列（>=1 特征 + 1 因变量）')
        cols = list(df.columns)
        feature_names = cols[:-1]
        y_name = cols[-1]
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy().reshape(-1)
        return X, y, feature_names, y_name

    # ===============
    # 工具：渲染 specification（NumPy 版）
    # ===============
    DEFAULT_BACKGROUND = "The physical properties of this equation are unknown and need to be analyzed based on experience."
    SPEC_TEMPLATE_NUMPY = '''\
"""
Find the mathematical function skeleton that fits the data.

Background:
{BACKGROUND}

Variables:
- Independents: {FEATURE_DOC}
- Dependent: {DEPENDENT}
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = {MAX_NPARAMS}
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    else:
        return -loss_val

@equation.evolve
def equation({FEATURE_SIG}, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    {BACKGROUND}

    Variables:
    - Independents: {FEATURE_DOC}
    - Dependent: {DEPENDENT}

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.
    """
    return {LINEAR_SEED}
'''
    # 初始线性骨架（可运行，便于快速启动；LLM 会逐步替换）

    def _ensure_feature_names(n, names):
        if names is None:
            return [f"x{i+1}" for i in range(n)]
        if len(names) != n:
            raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
        return names

    def _render_spec_numpy(n_features, feature_names=None, dependent_name=None, background=None, problem=None, max_nparams=10):
        feats = _ensure_feature_names(n_features, feature_names)
        dep = (dependent_name or 'y').strip()
        bg = (background or DEFAULT_BACKGROUND).strip()
        feature_sig = ', '.join([f"{name}: np.ndarray" for name in feats])
        feature_doc = ', '.join(feats)
        idx_c = min(3, n_features)
        terms = [f"params[{i}]*{feats[i]}" for i in range(min(3, n_features))]
        terms.append(f"params[{idx_c}]")
        linear_seed = ' + '.join(terms)
        return SPEC_TEMPLATE_NUMPY.format(
            BACKGROUND=bg,
            FEATURE_SIG=feature_sig,
            FEATURE_DOC=feature_doc,
            DEPENDENT=dep,
            MAX_NPARAMS=max_nparams,
            LINEAR_SEED=linear_seed,
        )

    # 加载 CSV（强制使用 data_csv 模式）
    X, y, feature_names, y_name = _load_from_csv(args.data_csv)
    background = args.background

    # 渲染 specification
    specification = _render_spec_numpy(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=y_name,
        background=background,
        problem=args.problem_name,
        max_nparams=10,
    )

    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}

    # 将动态渲染的 specification 保存到本次实验目录，便于调试
    try:
        spec_out_path = os.path.join(results_root, "spec_dynamic.txt")
        with open(spec_out_path, "w", encoding="utf-8") as f:
            f.write(specification)
        print(f"[INFO] Saved dynamic spec to: {spec_out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save dynamic spec: {e}")





##################################################
    # 定义 JSON 文件路径（放在实验根目录）
    json_experience_file = os.path.join(results_root, "experiences.json")

    # 如果文件不存在，则创建初始结构
    if not os.path.exists(json_experience_file):
        initial_experiences = {
            "None": [], 
            "Good": [], 
            "Bad": []
        }
        
        try:
            with open(json_experience_file, "w", encoding="utf-8") as f:
                json.dump(initial_experiences, f, ensure_ascii=False, indent=2)
            print(f"成功创建初始经验 JSON 文件: {json_experience_file}")
        except Exception as e:
            print(f"创建 JSON 文件时出错: {str(e)}")
    else:
        print(f"经验 JSON 文件已存在: {json_experience_file}")
##################################################
    
    
    # PromptContext：保证提示一致性（变量名/背景）
    from drsr_420 import prompt_config as pc
    prompt_ctx = pc.PromptContext(
        n_features=X.shape[1],
        feature_names=feature_names,
        dependent_name=y_name,
        problem_name=args.problem_name,
        background=background,
        max_params=10,
    )

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        results_root=results_root,
        prompt_ctx=prompt_ctx,
        llm_client=client,
        llm_config=llm_config,
        seed=args.seed,
    )
