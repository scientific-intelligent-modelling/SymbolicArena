# Semantic-200 LLM 物理背景重跑计划

## 目标

当前 `E1 12算法` 汇总表中的 `llmsr` 与 `drsr` 来自旧版无物理背景实验，不适合与“有物理背景”的 LLM 类算法设定混用。

本轮重跑只覆盖：

- `llmsr`
- `drsr`
- Candidate-200 全部数据集
- `seed = 1314`
- 每个任务 `timeout_in_seconds = 3600`

重跑结果后续应替换汇总表中的旧版 `llmsr/drsr` 行，并在结果目录中明确标注为 `physics-aware / semantic prompt` 版本。

## Prompt 口径

本轮不是直接暴露真实 CSV 列名，而是采用统一 prompt 变量：

- 自变量：`x0, x1, ...`
- 因变量：`y`
- 物理背景：来自 `metadata.yaml`
- 变量语义：来自 `metadata.yaml` 的 feature / target description

对 `SRSD dummy` 数据集，变量顺序必须隐藏：

- prompt 仍使用 `x0, x1, ...`
- 不告诉模型哪个 `x_i` 对应哪个具体物理变量
- 只告诉模型候选变量中有哪些语义角色，以及有多少个 `meaningless / distractor` 变量

这对应 runner 中的 `inject_prompt_semantics=true` 和 `canonical_prompt_variables=true`。

## 已核验内容

- Candidate-200: `200 / 200` 数据集均存在 `metadata.yaml`
- `200 / 200` 数据集均有 dataset description
- `200 / 200` 数据集均有 target description 或 target name
- `200 / 200` 数据集均有完整 feature description 或 feature name
- `SRSD`: `70` 个
- `SRSD dummy`: `54` 个，runner 会注入“语义角色多重集 + 隐藏变量映射”的背景描述

抽样验证：

- `CRK22` 会生成类似：`Calculate Rate of change of concentration in chemistry reaction kinetics given Time and Concentration at time t`
- `feynman-iii.15.12` 会生成 target/feature 物理语义，并追加 “unknown order / semantic-role multiset / distractor variables” 约束

## 资产位置

```text
exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v1/
```

关键文件：

```text
params/llmsr_semantic.json
params/drsr_semantic.json
slices/iaaccn22.csv
slices/iaaccn23.csv
remote_jobs/llmsr_iaaccn22.sh
remote_jobs/llmsr_iaaccn23.sh
remote_jobs/drsr_iaaccn22.sh
remote_jobs/drsr_iaaccn23.sh
launch/run_semantic200_llm_queue.sh
```

## 机器分配

| wave | algorithm | host | datasets | workers |
|---|---:|---:|---:|---:|
| 1 | `llmsr` | `iaaccn22` | 100 | 50 |
| 1 | `llmsr` | `iaaccn23` | 100 | 50 |
| 2 | `drsr` | `iaaccn22` | 100 | 50 |
| 2 | `drsr` | `iaaccn23` | 100 | 50 |

## DeepInfra 模型分流

为提高并发承载，本轮不再让两个 host 都打到同一个 DeepInfra 模型名，而是按 host 做 1:1 分流：

| host | datasets | model | config |
|---|---:|---|---|
| `iaaccn22` | 100 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct` | `benchmark_llm_deepinfra_llama31_8b.config` |
| `iaaccn23` | 100 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | `benchmark_llm_deepinfra_llama31_8b_turbo.config` |

这个分流对 `llmsr` 和 `drsr` 同时生效：

- `llmsr_semantic_iaaccn22.json` 使用普通 `8B-Instruct`
- `llmsr_semantic_iaaccn23.json` 使用 `8B-Instruct-Turbo`
- `drsr_semantic_iaaccn22.json` 使用普通 `8B-Instruct`
- `drsr_semantic_iaaccn23.json` 使用 `8B-Instruct-Turbo`

两个模型按当前实验口径视为同一 Llama-3.1-8B-Instruct 家族，只用于分摊 DeepInfra 并发；最终汇总时需要在 run metadata 中保留 `llm_model_assignment` 字段，方便审计。

执行顺序：

1. 先并发跑 `llmsr` 的 `iaaccn22 + iaaccn23`
2. 等 `llmsr` 两台都结束
3. 再并发跑 `drsr` 的 `iaaccn22 + iaaccn23`

## 安全发布约束

本轮实验涉及远端 LLM 调用和 400 个正式任务，不允许自动发布。

启动脚本已加确认令牌。没有显式设置以下环境变量时，脚本会直接拒绝启动：

```bash
export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v1
```

确认发布后，从远端项目根目录执行：

```bash
cd /home/zhangziwen/projects/scientific-intelligent-modelling
export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v1
bash exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v1/launch/run_semantic200_llm_queue.sh
```

## 验收标准

启动前：

- 代码与 `generated/semantic200_llm_physics_v1/` 已同步到 `iaaccn22`
- `iaaccn22` 能通过内网访问 `10.10.100.23`
- 两台机器都存在真实数据目录 `/home/zhangziwen/sim-datasets-data`
- 两台机器都存在非 Git 跟踪的真实 LLM 配置：
  `exp-planning/02.E1选择验证/llm_configs/benchmark_llm_deepinfra_llama31_8b.config`
  和
  `exp-planning/02.E1选择验证/llm_configs/benchmark_llm_deepinfra_llama31_8b_turbo.config`

运行中：

- `__launcher__/task_status.jsonl` 正常增长
- 每个任务目录有 `progress/minute_*.json`
- `llmsr/drsr` 的 `result.json` 中应保留 `params.background` 或可追溯的 prompt metadata

完成后：

- `llmsr`: 200 条 result
- `drsr`: 200 条 result
- 汇总时生成新版 12 算法 CSV，并显式标注 `llmsr/drsr` 为 `physics-aware`

## 当前状态

已完成本地资产生成与静态校验。

尚未发布远端实验；发布前必须由用户明确确认。
