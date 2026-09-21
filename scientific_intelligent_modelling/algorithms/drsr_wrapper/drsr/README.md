DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience
=============================================

> **本仓库是论文 [DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience](https://arxiv.org/abs/2506.04282) 的官方开源实现。**
>
> **Authors:** Runxiang Wang, Boxiao Wang, Kai Li, Yifan Zhang, Jian Cheng
>
> **论文链接:** https://arxiv.org/abs/2506.04282

本项目实现了一个"基于大模型的方程结构搜索 + 数据拟合"的工作流（类似符号回归）：

- 使用 LLM 从代码模板（spec）中补全目标函数的函数体（equation）。
- 对每个候选方程，使用 BFGS 在给定数据上优化参数并打分（默认负 MSE / 可推导 R²）。
- 经验缓冲区（Experience Buffer）保留更优样本，持续迭代搜索更好结构。
- 训练产物（样本、最优 spec、经验与残差分析等）统一落盘，便于复现与调试。


依赖安装
--------

```bash
pip install -r requirements.txt
```

- 仅依赖 NumPy、SciPy、Pandas 等。已移除 torch/transformers 等大包。


LLM 配置（llm.config）
----------------------

根目录提供 `llm.config`（JSON），用于配置大模型访问与采样参数：

```json
{
  "host": "api.bltcy.ai",
  "api_key": "xxx",
  "model": "bltcy/gpt-3.5-turbo",
  "max_tokens": 1024,
  "temperature": 0.6,
  "top_p": 0.3
}
```

说明：

- `api_key` 请替换为真实密钥，否则会报“未提供令牌”。
- `model` 建议使用 `provider/model` 形式（如 `bltcy/gpt-3.5-turbo`）。 目前支持Deepseek，SiliconFlow，柏拉图，Ollama，具体支持列表请查看`llm.py`
- 如需切换模型，请直接修改 `llm.config` 的相应字段，温度等配置信息也在此处进行修改。
- 运行时每个任务实例化一个 LLM Client，全程复用；并行任务互不影响。


快速开始
------------------------

CSV 需带表头：前 n-1 列为特征，最后一列为因变量。可用迭代与每轮候选数控制搜索规模。

```bash
python3 main.py \
--problem_name oscillator1 \
--llm_config llm.config \
 --data_csv ./data/oscillator1/train.csv \
 --background 'Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity.' 

```

运行后将在 `results/oscillator1_时间戳/` 下生成所有产物。

可选调参示例：

```bash
# 迭代轮数与每轮候选数（近似：最大采样数 = niterations * num_samplers * samples_per_iteration）
python3 main.py \
  --problem_name oscillator1 \
  --llm_config llm.config \
  --data_csv ./data/oscillator1/train.csv \
  --niterations 50 \
  --samples_per_iteration 8
```


批量示例（example.sh）
----------------------

根目录的 `example.sh` 给出 12 个数据集的一键命令，已内置对应背景：

```bash
bash example.sh
```


结果产物与目录结构
--------------------

以 `results/oscillator1_20250101-120000/` 为例：

- `run.out`, `run.err`：标准输出/错误输出。
- `spec_dynamic.txt`：本次运行的动态 spec（便于复现）。
- `experiences.json`：采样过程中的“经验/总结”。
- `residual_analyze.json`：残差分析结果。
- `samples/`：每次评分的样本 JSON，形如 `samples_32.json`，包含：
  - `score`：分数（越大越好）
  - `function`：完整函数源码（含 def、docstring、body）
  - `params`：BFGS 优化得到的参数（若成功）


如何找到“当前最优方程”
------------------------

扫描 `samples/` 下分数最高的样本即可：

```bash
python3 - << 'PY2'
import json, glob, os
results_root = "results/oscillator1_20250101-120000"  # 改为你的目录
best = None
for p in glob.glob(os.path.join(results_root, "samples", "samples_*.json")):
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        s = d.get("score")
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (s, p, d.get("function",""), d.get("params"))
    except Exception:
        continue
if best is None:
    print("没有找到有效样本。")
else:
    score, path, func, params = best
    print(f"[BEST] score={score}
file={path}
params={params}

function:
{func}")
PY2
```


可配置项与自定义
------------------

- `llm.config`：LLM 访问与采样参数（host/api_key/model/max_tokens/temperature/top_p）。
- `drsr_420/config.py`：采样与评估资源配置（`samples_per_prompt`、`evaluate_timeout_seconds`、并行数等）。
- 运行时可通过 `--samples_per_iteration` 覆盖每轮候选数量（内部映射为 `samples_per_prompt`）。
- 运行时可通过 `--niterations` 控制“迭代轮数”（最大采样数 = niterations × num_samplers × samples_per_iteration）。
- `drsr_420/evaluate_on_problems.py`：BFGS 拟合与指标（返回 `(score, result_matrix, optimized_params)`）。
- `drsr_420/prompt_config.py` 与 `PromptContext`：采样提示词的模板与动态渲染（变量名、背景）。


提示词与背景
--------------
提示词模板如下：
``` text
Find the mathematical function skeleton that represents {PROBLEM}.

Background:
{BACKGROUND}

Variables:
- Independents: {FEATURE_DOC}
- Dependent: {DEPENDENT}
```

显然在背景里面进行进行知识注入是极好的，并且最好可以精确到每一个因变量的含义
如： col1 为电场强度单位是(N/C)  col2为电荷数单位库仑(C)

- 背景：由 `--background` 传入，并写入动态 spec 顶部注释与 equation docstring，同时注入采样提示。
- 变量名：从 CSV 表头解析（前 n-1 列为自变量名，最后一列为因变量名）。



仓库结构
--------
- `main.py`：入口（CSV 动态模式）。
- `llm.config`：LLM 访问与采样参数。
- `example.sh`：13 个数据集的运行示例。
- `drsr_420/`：核心模块
  - `pipeline.py`：调度 Evaluator/Sampler，注入 LLM Client，触发初次数据分析。
  - `sampler.py`：采样器（全程使用注入的 Client 发起 LLM 请求）。
  - `evaluator.py`：运行候选方程、BFGS 拟合与打分。
  - `evaluate_on_problems.py`：BFGS 与评分逻辑（返回拟合参数）。
  - `buffer.py`：经验缓冲（多岛与聚类抽样）。
  - `code_manipulation.py`：AST 解析与函数/程序拼装、调用重命名等。
  - `prompt_config.py`：提示词模板与 `PromptContext`。
  - `profile.py`：轻量记录（写 `samples/*.json`），已移除 TensorBoard。
- `specs/`：历史静态 spec（动态模式无需）。
- `results/{problem}_{timestamp}/`：本次运行产物。


引用
----

如果本项目对您的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{wang2025drsr,
  title={DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience},
  author={Wang, Runxiang and Wang, Boxiao and Li, Kai and Zhang, Yifan and Cheng, Jian},
  journal={arXiv preprint arXiv:2506.04282},
  year={2025}
}
```

**论文链接:** https://arxiv.org/abs/2506.04282

**项目地址:** https://github.com/scientific-intelligent-modelling/drsr
