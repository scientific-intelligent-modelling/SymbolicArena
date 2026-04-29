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

## 启动状态

DeepInfra `api_key` 已写入四台机器的 ignored config 文件。

已完成四机 API smoke：

- `iaaccn23`: `Meta-Llama-3.1-8B-Instruct`, HTTP 200
- `iaaccn24`: `Meta-Llama-3.1-8B-Instruct-Turbo`, HTTP 200
- `iaaccn25`: `Meta-Llama-3.1-8B-Instruct`, HTTP 200
- `iaaccn26`: `Meta-Llama-3.1-8B-Instruct-Turbo`, HTTP 200

启动命令：

```bash
cd /home/zhangziwen/projects/scientific-intelligent-modelling
export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v2_4host
bash exp-planning/02.E1选择验证/generated/semantic200_llm_physics_v2_4host/launch/run_semantic200_llm_queue.sh
```

实际通过 `tmux` 队列控制器启动：

```text
tmux session: semantic200_v2_queue
batch: semantic200_llm_physics_v2_4host_seed1314_20260429-205250
queue log: /home/zhangziwen/projects/scientific-intelligent-modelling/experiments/semantic200_llm_physics_v2_4host_seed1314_20260429-205250.queue.log
```

当前正在运行 `llmsr` wave：

| host | tmux session | logs | progress snapshots |
|---|---|---:|---:|
| `iaaccn23` | `semantic200_llmsr_iaaccn23` | 50 | 200 |
| `iaaccn24` | `semantic200_llmsr_iaaccn24` | 50 | 198 |
| `iaaccn25` | `semantic200_llmsr_iaaccn25` | 50 | 198 |
| `iaaccn26` | `semantic200_llmsr_iaaccn26` | 50 | 198 |

`drsr` wave 会在四台 `llmsr` 任务全部结束后由队列控制器自动启动。
