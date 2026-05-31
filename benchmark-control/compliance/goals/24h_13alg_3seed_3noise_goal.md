# 24h 13 算法三 seed 三噪声正式实验 Goal

## 目标

完成正式 24h benchmark 批次：

```text
13 algorithms × SSR50 × seeds(520,521,522) × noise(0,0.01,0.05) × 24h
```

总任务数：

```text
5850
```

## 当前状态

本 goal 是下一阶段正式实验目标。现有 1h compliance 控制面已经证明基础闭环可行，但当前代码仍有若干 1h 硬编码：

- `benchmark-control/compliance/lib/models.py` 固定 `seed=520`、`timeout=3600`。
- `benchmark-control/compliance/lib/manifest.py` 固定 15 算法、单 seed、无 noise 维度。
- `benchmark-control/compliance/lib/readiness.py` 固定 750 任务、15 算法、seed520、3600 秒。
- `benchmark-control/compliance/launchers/write_stage1_queue_commands.py` 固定 1h stage1 命令。
- `check/run_e1_candidate200_12alg_load_queue.py` 当前调度粒度是 `dataset × tool × seed`，还没有一等 `noise_sigma` 维度。

因此，在执行本 goal 前，必须先完成：

```text
docs/superpowers/plans/2026-05-31-full24h-13alg-3seed-3noise.md
```

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
clean   = 0.0
noise001 = 0.01
noise005 = 0.05
```

runner 噪声协议：

```text
y_noisy = y + sigma * std(train_y) * N(0, 1)
```

噪声只作用于训练标签；valid/id_test/ood_test 使用 clean labels 评估。

## 成功条件

- `manifest/tasks.csv` 有 5850 行任务。
- `audit/task_audit.csv` 有 5850 行审计记录。
- `audit/failure_cases.csv` 为空。
- 所有任务 runtime 达到 `>=82800s`，或明确标记为预算耗尽并成功恢复。
- 每个任务有最终 `result.json`。
- 每个任务有分钟级 progress 快照。
- 每个任务有有限 valid/id_test/ood_test 指标。
- 每个任务有可解析 equation 或 canonical artifact。
- `heartbeat.json` 标记 `needs_codex=false`。

## Codex 接管规则

1. 读取 `heartbeat.json` 判断阶段。
2. 若阶段是 `preparing`，先跑本地 readiness，不启动远端 full。
3. 若阶段是 `preflight`，只做远端只读检查。
4. 若阶段是 `smoke`，等待 smoke 完成并审计。
5. 若 smoke 失败，基于日志修复代码、提交、重部署，只重跑失败项。
6. smoke audit 全绿后才能进入 full。
7. full 运行期间只 observe/audit/repair/rerun，不删除历史批次。
8. 修复必须基于 `audit/failure_cases.csv`、`task_status.jsonl`、单任务日志和 result payload。
9. 修复后必须提交 Git，但不得提交 `benchmark-runs/`。
10. 只有 `heartbeat.json needs_codex=false` 且 audit 全绿时，才能宣布 goal 完成。

## 禁止事项

- 不使用 `frozen-results/` 存放本批次产物。
- 不直接运行旧 1h stage1 full 脚本替代本 goal。
- 不在未通过 smoke 前启动 5850 个 24h full 任务。
- 不删除历史批次。
- 不清理数据盘。
- 不提交 `benchmark-runs/`。
- 不绕过审计器手工标记通过。

## 推荐批次名

```text
formal24h_13alg_ssr50_seed520-522_noise0-001-005_YYYYMMDD-HHMMSS
```

推荐 latest 链接：

```text
benchmark-runs/formal24h/latest
```

完整产物目录：

```text
benchmark-runs/formal24h/<batch_id>
```

