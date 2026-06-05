# 3h 13 算法三 seed 三噪声正式实验 Goal

## 目标

完成正式 3h benchmark 批次：

```text
13 algorithms × SSR50 × seeds(520,521,522) × noise(0,0.01,0.05) × 3h
```

总任务数：

```text
5850
```

该目标替代原 24h 正式实验目标。历史 24h 批次保留为可回收数据源，不作为当前目标的完成口径。

## 当前可回收证据

已检查现有 24h 批次：

```text
formal24h_13alg_ssr50_seed520-522_noise0-001-005_20260531-230535
```

真实远端结果根目录位于各机器的：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal24h_13alg_ssr50_seed520-522_noise0-001-005_20260531-230535
```

可以从每个任务的：

```text
progress/minute_0180.json
```

抽取 3h 结果。当前只读审计结果如下：

```text
iaaccn22: direct=248, valid=242, invalid=6
iaaccn23: direct=390, valid=390, invalid=0
iaaccn24: direct=405, valid=405, invalid=0
iaaccn25: direct=237, valid=234, invalid=3
iaaccn26: direct=354, valid=353, invalid=1
iaaccn27: direct=398, valid=397, invalid=1
iaaccn28: direct=378, valid=378, invalid=0
iaaccn29: direct=357, valid=356, invalid=1
```

合计：

```text
direct minute_0180 snapshots = 2767
valid 3h recoverable snapshots = 2755
invalid snapshots = 12
```

有效快照的验收条件：

- `status == "ok"`。
- 存在 `elapsed_seconds`、`seconds` 或 `runtime_seconds`，且运行时间约为 10800 秒。
- 存在 `valid`、`id_test`、`ood_test` 指标。
- 存在 `canonical_artifact` 或可解析 `equation`。

无效快照不得直接计入 3h 正式结果，必须进入缺口补跑或修复队列。

## 实验范围

### 保留算法

```text
gplearn
pyoperon
pysr
dso
tpsr
e2esr
fepysr
jaxsr
QLattice
iMCTS
udsr
ragsr
symbolfit
```

### 排除算法

```text
llmsr
drsr
```

这两个大模型算法不进入本批次。

### Seed

```text
520
521
522
```

### 噪声

```text
clean    = 0.0
noise001 = 0.01
noise005 = 0.05
```

runner 噪声协议：

```text
y_noisy = y + sigma * std(train_y) * N(0, 1)
```

噪声只作用于训练标签；valid/id_test/ood_test 使用 clean labels 评估。

## 预算口径

正式 full 参数：

```text
timeout_in_seconds = 10800
min_runtime_seconds = 10500
progress_snapshot_interval_seconds = 60
```

smoke 参数：

```text
timeout_in_seconds = 600
progress_snapshot_interval_seconds = 60
```

`min_runtime_seconds=10500` 等价于 2 小时 55 分钟。低于该阈值且没有明确预算耗尽恢复证据时，应审计为 `early_stop`。

## 执行策略

1. 不删除历史 24h 批次。
2. 不清理数据盘。
3. 不停止 24h 远端任务，除非用户另行明确确认。
4. 不提交 `benchmark-runs/`。
5. 先从现有 24h 批次恢复有效 `minute_0180.json`。
6. 对已恢复任务写入 `recovered_from_24h` 元数据。
7. 对未恢复或无效快照任务生成 missing-only 队列。
8. 新 3h full 只运行缺口任务。
9. smoke 仍使用 2 个数据集，验证 13 算法 × 3 seed × 3 noise 的控制面。
10. full 运行期间持续 observe/audit/repair/rerun，直到 `heartbeat.json` 标记 `needs_codex=false`。

## 推荐批次名

```text
formal3h_13alg_ssr50_seed520-522_noise0-001-005_YYYYMMDD-HHMMSS
```

推荐 latest 链接：

```text
benchmark-runs/formal3h/latest
```

完整产物目录：

```text
benchmark-runs/formal3h/<batch_id>
```

## 恢复产物布局

恢复后的结果必须落在 3h 批次自己的审计布局中：

```text
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/result.json
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/progress/minute_0180.json
benchmark-runs/formal3h/latest/runs/<algorithm>/seed<seed>/<noise_tag>/<dataset_id>/recovery_source.json
```

`recovery_source.json` 必须记录：

- `recovered_from_24h=true`
- `source_batch`
- `source_host`
- `source_path`
- `elapsed_seconds`
- `recovered_at`

## 成功条件

- `manifest/tasks.csv` 有 5850 行任务。
- `recovery/recovered_tasks.csv` 记录所有有效恢复任务。
- `recovery/missing_tasks.csv` 记录所有需补跑任务。
- `audit/task_audit.csv` 有 5850 行审计记录。
- `audit/failure_cases.csv` 为空。
- 所有任务 runtime 达到 `>=10500s`，或明确标记为预算耗尽并成功恢复。
- 每个任务有最终 `result.json`。
- 每个任务有分钟级 progress 快照。
- 每个任务有有限 valid/id_test/ood_test 指标。
- 每个任务有可解析 equation 或 canonical artifact。
- `heartbeat.json` 标记 `needs_codex=false`。

## Codex 接管规则

1. 读取 `heartbeat.json` 判断阶段。
2. 若阶段是 `preparing`，先跑本地 readiness，不启动远端 full。
3. 若阶段是 `recovering`，只从 24h 批次抽取 3h 快照并生成缺口清单。
4. 若阶段是 `preflight`，只做远端只读检查。
5. 若阶段是 `smoke`，等待 smoke 完成并审计。
6. 若 smoke 失败，基于日志修复代码、提交、重部署，只重跑失败项。
7. smoke audit 全绿后才能进入 full。
8. full 只运行 `recovery/missing_tasks.csv` 中的缺口任务。
9. full 运行期间只 observe/audit/repair/rerun，不删除历史批次。
10. 修复必须基于 `audit/failure_cases.csv`、`task_status.jsonl`、单任务日志和 result payload。
11. 修复后必须提交 Git，但不得提交 `benchmark-runs/`。
12. 只有 `heartbeat.json needs_codex=false` 且 audit 全绿时，才能宣布 goal 完成。

## 禁止事项

- 不使用 `frozen-results/` 存放本批次产物。
- 不直接运行旧 1h stage1 full 脚本替代本 goal。
- 不直接运行旧 24h full 脚本替代本 goal。
- 不在未通过 smoke 前启动 full 缺口任务。
- 不重复计算已经有效恢复的 3h 快照任务。
- 不删除历史批次。
- 不清理数据盘。
- 不提交 `benchmark-runs/`。
- 不绕过审计器手工标记通过。
