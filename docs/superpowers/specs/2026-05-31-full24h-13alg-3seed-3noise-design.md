# Full 24h 13 算法噪声实验设计

## 目标

本设计定义下一阶段正式全量 benchmark：

```text
13 algorithms × SSR50 × 3 seeds × 3 noise levels × 24h
```

这是 1h compliance 通过后的正式实验批次。目标不是再次验证“能否跑满 1h”，而是收集可用于后续论文或 benchmark 维护的 24h 稳定结果，同时保留足够的中间过程文件用于失败诊断和复现。

## 实验网格

### 算法范围

保留非大模型算法 13 个：

```text
gplearn
pyoperon
pysr
dso
tpsr
e2esr
fepysr
jaxsr
qlattice
imcts
udsr
ragsr
symbolfit
```

排除两个大模型算法：

```text
llmsr
drsr
```

排除原因：用户本轮明确要求“除了两个大模型算法之外”先跑 24h 正式批次；这两个算法保留在后续独立 LLM 成本与环境评估批次中。

### 数据集范围

数据集固定为：

```text
sim-datasets-data/ssr50
```

该目录必须包含 50 个 SSR50 数据集。远端运行时真实数据根目录仍按项目约定优先解析为：

```text
/home/zhangziwen/sim-datasets-data
```

### Seed 范围

固定使用 3 个 seed：

```text
520
521
522
```

### 噪声范围

固定使用 3 档训练标签噪声：

```text
clean: 0.0
noise001: 0.01
noise005: 0.05
```

噪声只作用于传给 wrapper `fit()` 的训练标签；valid、id_test、ood_test 评估仍使用 clean labels。

当前 runner 已支持该协议：

- `scientific_intelligent_modelling/benchmarks/runner.py:275` 解析噪声参数。
- `scientific_intelligent_modelling/benchmarks/runner.py:324` 生成 noisy train labels。
- `scientific_intelligent_modelling/benchmarks/runner.py:1648` 在构造 wrapper 后、调用 `fit()` 前应用噪声。

参数写法：

```json
{
  "train_label_noise_sigma": 0.01
}
```

clean 档可以省略该字段，也可以显式写：

```json
{
  "train_label_noise_enabled": false,
  "train_label_noise_sigma": 0.0
}
```

推荐显式写入 clean 档，原因是 manifest、params、audit 可以统一记录三档噪声，避免“clean 是否漏配”的歧义。

### 任务规模

```text
13 × 50 × 3 × 3 = 5850 tasks
```

单任务预算：

```text
timeout_in_seconds = 86400
min_runtime_seconds = 82800
progress_snapshot_interval_seconds = 60
```

`min_runtime_seconds=82800` 等价于 23 小时。除非任务以 `budget_exhausted_with_output` 正常收口，否则低于该阈值应被审计为 early stop。

## 输出目录

正式 24h 运行产物不进入 `frozen-results/`。推荐新目录：

```text
benchmark-runs/formal24h/
  formal24h_13alg_ssr50_seed520-522_noise0-001-005_YYYYMMDD-HHMMSS/
```

`benchmark-runs/` 保持 Git ignore。Git 只提交控制面代码、计划、goal、schema 和轻量 run summary。

批次内部结构：

```text
manifest/
  algorithms.json
  datasets.csv
  noise_levels.csv
  budget.json
  tasks.csv
  git_revision.txt
params/
  gplearn__clean.json
  gplearn__noise001.json
  gplearn__noise005.json
  ...
queues/
  ssr50_source.csv
  smoke_2datasets_source.csv
  load_queue_smoke/
  load_queue_full/
runs/
  <algorithm>/
    seed<seed>/
      <noise_tag>/
        <dataset_id>/
audit/
repair/
deploy/
heartbeat.json
```

## 任务标识

manifest 任务 ID：

```text
<algorithm>__seed<seed>__<noise_tag>__<dataset_id>
```

队列调度 task ID：

```text
<tool>_s<seed>_<noise_tag>_g<global_index>
```

示例：

```text
pysr__seed520__noise001__CRK0
pysr_s520_noise001_g0001
```

任务 ID 必须包含 `noise_tag`，否则三档噪声会在 harvest/audit 阶段互相覆盖。

## 调度策略

推荐将噪声作为 scheduler 的一等任务维度，而不是手工起 3 个互不知情的 batch。

原因：

1. 一个 manifest 能完整表达 5850 个任务。
2. `heartbeat.json` 能正确汇总全部任务，而不是只看到某个 sigma 子批次。
3. `failure_cases.csv` 可直接定位到 `(algorithm, seed, noise_sigma, dataset)`。
4. rerun 只重跑失败项，不需要人工判断属于哪个 sigma。

调度器需要支持：

```bash
--noise-sigmas 0,0.01,0.05
--params-root benchmark-runs/formal24h/latest/params
--tools gplearn pyoperon pysr dso tpsr e2esr fepysr jaxsr qlattice imcts udsr ragsr symbolfit
--seeds 520 521 522
```

## 参数生成策略

不要直接修改 `exp-planning/02.E1选择验证/generated/params/*.json`。该目录仍作为 1h 基准参数源。

正式 24h 批次应在 `benchmark-runs/formal24h/<batch_id>/params/` 生成派生参数文件：

1. 读取原始 tool params。
2. 写入 `timeout_in_seconds=86400`。
3. 写入 `progress_snapshot_interval_seconds=60`。
4. 按 noise tag 写入 `train_label_noise_enabled` 和 `train_label_noise_sigma`。
5. 保留每个算法原有超大迭代预算。

当前抽样显示这些参数源已经包含大迭代预算：

- `pysr.json`：`niterations=10000000`
- `dso.json`：`training.n_samples=2000000`
- `qlattice.json`：`n_epochs=100000`
- `symbolfit.json`：`niterations=1000000`

实现时仍要对全部 13 个算法做自动校验：如果存在显式迭代预算小于 24h 经验阈值，应在 preflight 中 fail fast，而不是启动正式任务。

## 审计策略

`audit/task_audit.csv` 必须新增噪声字段：

```text
noise_tag
noise_sigma
```

审计主键从：

```text
task_id
```

扩展为仍以 `task_id` 为主键，但 `task_id` 必须包含噪声维度。审计行必须能还原：

```text
algorithm, dataset_id, seed, noise_tag, noise_sigma
```

合规条件：

1. 5850 个任务全部进入 `audit/task_audit.csv`。
2. `audit/failure_cases.csv` 为空。
3. 每个任务存在最终 `result.json`。
4. 每个任务存在分钟级过程快照。
5. 每个任务 runtime 达到 `>=82800s`，或明确标记为预算耗尽后正常恢复。
6. valid/id_test/ood_test 指标存在且有限。
7. equation 或 canonical artifact 可解析。
8. `heartbeat.json` 中 `needs_codex=false`。

## 磁盘与 inode 风险

若保留每分钟快照：

```text
5850 tasks × 1440 snapshots = 8424000 minute snapshots
```

这还不包含各算法自身的 checkpoint、hall of fame、日志和中间缓存。

正式启动前必须做远端 preflight：

1. 每台机器 `df -h`。
2. 每台机器 `df -i`。
3. 估算当前 batch 输出目录所在分区的可用 inode。
4. 估算 `experiments/<batch_id>/` 预计文件数量。
5. 若 inode 风险过高，优先保留每分钟轻量 JSON，但将算法自身大中间缓存按算法配置关闭或压缩归档；不要降低外层 `progress_snapshot_interval_seconds=60`，除非用户再次确认。

## 执行阶段

正式 goal 分三层：

```text
prepare → preflight → smoke → full → audit → repair/rerun
```

### prepare

只生成本地控制面：

- manifest
- params
- queue source
- deploy scripts
- readiness summary

不启动远端任务。

### preflight

只读检查远端：

- SSH 连通性
- 代码版本
- 参数文件
- SSR50 数据路径
- conda 环境
- 磁盘空间和 inode
- 已存在 tmux/session 是否会冲突

### smoke

建议 smoke 不用完整 24h，而是使用独立 smoke budget：

```text
13 algorithms × 2 datasets × 3 seeds × 3 noise levels × 10min
```

任务数：

```text
702
```

smoke 目标不是验证 24h runtime，而是验证：

- noise 参数能正确进入 result payload。
- 三档 noise 输出目录不互相覆盖。
- harvest/audit/rerun 能识别 noise 维度。
- 每个算法能写出 progress 和 result。

### full

只有 smoke audit 全绿后才启动：

```text
5850 tasks × 24h
```

full 期间 Codex 只做 observe/audit/repair/rerun，不删除历史批次，不清理数据盘，不提交 `benchmark-runs/`。

## 成功条件

正式批次完成时必须同时满足：

```text
manifest/tasks.csv rows = 5850
audit/task_audit.csv rows = 5850
audit/failure_cases.csv rows = 0
heartbeat.json needs_codex = false
```

同时生成轻量摘要：

```text
benchmark-control/compliance/run-summaries/<batch_id>.md
```

摘要进入 Git；完整实验产物不进入 Git。

