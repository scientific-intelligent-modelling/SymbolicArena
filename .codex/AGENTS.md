# 项目级实验说明

这个项目是一个**符号回归工具集**：

## 总体原则

0. 开始实验分发之前，先检查远程的代码是否和本地的保持完全一致，若没有，需让我确认是否把远程的进行同步。

1. 做实验前，先确认当前实验目标：
   - 是流程验证
   - 还是固定超参数进行实验分发

2. 远程批量实验默认按**每个数据集一个 tmux 会话**执行。

3. 不要把 API key 写入仓库文件或提交到 Git。
   - `llmsr` / `drsr` 默认通过运行时生成的 `llm.config` 或临时环境变量注入。

4. `drsr` 现在对外参数接口已经对齐到 `llmsr`：
   - 主参数使用 `niterations`
   - 主参数使用 `samples_per_iteration`
   - 不再主推 `max_samples` / `samples_per_prompt`

5. 如果要读取实验超参数，不要在这里硬编码。
   - **统一去看**：
     [EXPERIMENT_HYPERPARAMETERS.md](/home/family/workplace/scientific-intelligent-modelling/.codex/EXPERIMENT_HYPERPARAMETERS.md)

6. Routify 的 RPM 配额按模型分别计算。
   - 多模型混合并发时，分别统计每个模型的请求速率和 `429`，不得把不同模型的 RPM 相加后与单模型上限比较。
   - 若某个模型自身接近上限，只对该模型限流，不对其它模型施加无依据的全局限流。

## 数据与路径约定

### 远程项目根目录

```bash
/home/zhangziwen/projects/scientific-intelligent-modelling
```

### LLM-SRBench 数据路径

```bash
.sim_datasets/llm-srbench/<domain>/<dataset>
```

例如 `matsci`：

```bash
.sim_datasets/llm-srbench/matsci/MatSci0
```

## 实验执行约定

### 统一批量入口

默认使用：

```bash
check/run_phys_osc_task.py
```

虽然文件名还是 `run_phys_osc_task.py`，但它现在已经是一个**通用单任务批量入口**，可用于：

- `pysr`
- `llmsr`
- `drsr`

### 环境约定

- `pysr`：`sim_base`
- `llmsr`：`sim_llm`
- `drsr`：`sim_llm`

### 远程调度方式

1. 每个数据集一个独立 `tmux` 会话
2. 每个数据集单独日志
3. 每个数据集单独 `result.json`
4. 会话命名要带预算前缀，避免混批次

## DRSR 特殊说明

### 接口语义

`drsr` 对外已经按 `llmsr` 风格使用：

- `llm_config_path`
- `background`
- `metadata_path`
- `niterations`
- `samples_per_iteration`
- `seed`
- `problem_name`
- `exp_path`
- `exp_name`

其内部会自动映射为：

```text
max_samples = niterations * samples_per_iteration
samples_per_prompt = samples_per_iteration
```

### Prompt 规范

`drsr` 的外层 prompt 已做清理，后续实验要默认保持：

- 不使用 `with driving force`
- 不再向模型暴露旧的 `col0/col1` 外层变量命名
- 统一使用 `x0/x1/.../y`
- 如果有 `metadata.yaml`，优先带入物理语义描述


## 监控约定

### 看 tmux 会话

```bash
tmux ls
```

### 看单个日志

```bash
tail -f bench_results/<batch>/logs/<dataset>.log
```

### 看结果是否开始落盘

```bash
find bench_results/<batch> -name result.json
```

## 当前建议

1. 想看真实中间过程：
   - 优先开日志
   - `pysr` 要打开 `progress=True, verbosity=1`

2. 想做公平比较：
   - 先查超参数文档
   - 再确认两边总预算一致

3. 想重跑 `drsr`：
   - 先确认旧批次是否仍在跑
   - 再决定是否停旧批次并起新批次

4. 想新增实验方案：
   - 先把方案写入超参数文件
   - 再执行批量下发
