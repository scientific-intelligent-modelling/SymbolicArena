# Semantic-200 LLM Physics v2 4-host Deployment Report

## 部署时间

2026-04-29

## 本地提交

- `4a3bfdc` `[update] 准备LLM物理背景四机资产`
- `bcf29a8` `[fix] 增加四机LLM鉴权预检`

## 目标

将 `llmsr` / `drsr` 的 Candidate-200 有物理背景重跑部署到四台 CPU 机器：

- `iaaccn23`
- `iaaccn24`
- `iaaccn25`
- `iaaccn26`

每台 50 个数据集、50 workers。

## 机器分配

| host | slice | datasets | workers | model |
|---|---|---:|---:|---|
| `iaaccn23` | `iaaccn23.csv` | 50 | 50 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `iaaccn24` | `iaaccn24.csv` | 50 | 50 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| `iaaccn25` | `iaaccn25.csv` | 50 | 50 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `iaaccn26` | `iaaccn26.csv` | 50 | 50 | `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |

## 已同步内容

远端根目录：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling
```

已同步：

- `check/`
- `scientific_intelligent_modelling/`
- `pyproject.toml`
- `exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/`
- `exp-planning/02.E1选择验证/llm_configs/benchmark_llm_deepinfra_llama31_8b.config`
- `exp-planning/02.E1选择验证/llm_configs/benchmark_llm_deepinfra_llama31_8b_turbo.config`

同步方式：

- 本地先同步到 `iaaccn23`
- 再由 `iaaccn23` 通过内网同步到 `10.10.100.24~26`

## 验收结果

四台均通过：

- asset 目录存在
- `/home/zhangziwen/sim-datasets-data` 存在
- 每个 slice 为 50 条
- host-specific params JSON 可解析
- launch / remote job shell 语法检查通过

## 当前阻塞

实验尚未启动。

原因：四台都缺少 DeepInfra 鉴权。

检查结果：

- `DEEPINFRA_API_KEY`: missing
- `benchmark_llm_deepinfra_llama31_8b.config`: exists, but no `api_key`
- `benchmark_llm_deepinfra_llama31_8b_turbo.config`: exists, but no `api_key`

带确认令牌执行启动脚本时，已按预期在启动任务前失败：

```text
LLM_AUTH_FAIL no_api_key:benchmark_llm_deepinfra_llama31_8b.config,no_api_key:benchmark_llm_deepinfra_llama31_8b_turbo.config
EXIT_CODE=2
```

确认没有任何 `semantic200` tmux 会话在运行。

## 补齐 key 后的启动方式

在四台机器的两份 ignored config 中补入 DeepInfra `api_key`，或在四台机器的运行环境中提供 `DEEPINFRA_API_KEY`。

然后从 `iaaccn23` 启动：

```bash
cd /home/zhangziwen/projects/scientific-intelligent-modelling
export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v2_4host
bash exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/launch/run_semantic200_llm_queue.sh
```

启动脚本会先做四机鉴权预检；预检不通过不会创建任务 tmux。
