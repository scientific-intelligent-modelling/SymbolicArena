# Benchmark 合规闭环设计

## 目标

本设计用于把当前符号回归 benchmark 的实验执行、预算合规、结果落盘、失败诊断和修复重跑收口成一个可恢复闭环。阶段一只覆盖合规验证，不直接进入最终 24 小时正式实验。

阶段一范围固定为：

```text
15 algorithms × SSR50 × seed520 × 1h
```

其中：

- `15 algorithms` 来自主仓库 `scientific_intelligent_modelling/config/toolbox_config.json`。
- `SSR50` 来自 `sim-datasets-data/ssr50`。
- `seed520` 是阶段一唯一 seed。
- `1h` 表示每个任务 `timeout_in_seconds=3600`。

阶段一通过后，再扩展到：

```text
15 algorithms × SSR50 × 3 seeds × 24h × noise
```

## 设计原则

1. 实验运行可以长时间无人值守，但状态必须落盘。
2. Codex 通过 `/goal` 接管闭环，但不依赖 Codex 自身常驻记忆。
3. 真实实验产物不进入 Git；控制逻辑、规则、轻量摘要进入 Git。
4. 修复必须基于审计证据，不凭猜测修改 wrapper。
5. 已合规任务不重复全量重跑，只重跑失败项。

## 目录结构

控制面目录进入 Git：

```text
benchmark-control/
  compliance/
    README.md
    goals/
      1h_compliance_goal.md
    manifests/
    templates/
    audit-schemas/
    repair-notes/
    launchers/
    run-summaries/
```

实验产物目录不进入 Git：

```text
benchmark-runs/
  compliance/
    compliance_15alg_ssr50_seed520_1h_YYYYMMDD-HHMMSS/
      manifest/
      queues/
      runs/
      audit/
      repair/
      deploy/
      heartbeat.json
```

职责边界：

- `benchmark-control/` 保存控制逻辑、goal、schema、模板、轻量审计摘要和修复摘要。
- `benchmark-runs/` 保存每轮真实运行产物，包括方程、分钟快照、日志、checkpoint、hall of fame 和原生中间文件。
- `frozen-results/` 继续只表示论文冻结结果，不承载新的 benchmark 运维批次。

## 算法清单

阶段一使用当前 `toolbox_config.json` 中的 15 个算法：

```text
QLattice
drsr
dso
e2esr
fepysr
gplearn
iMCTS
jaxsr
llmsr
pyoperon
pysr
ragsr
symbolfit
tpsr
udsr
```

## 控制器流程

整体流程为：

```text
prepare → deploy → dispatch → monitor → audit → classify → repair → rerun
```

### prepare

生成标准批次：

- 读取 15 个算法配置。
- 读取 SSR50 的 50 个数据集。
- 固定 `seed=520`、`timeout_in_seconds=3600`、`progress_snapshot_interval_seconds=60`。
- 生成 750 条任务。
- 写入：
  - `manifest/tasks.csv`
  - `manifest/algorithms.json`
  - `manifest/datasets.csv`
  - `manifest/budget.json`
  - `manifest/git_revision.txt`

### deploy

部署策略：

- 本地先同步到 `iaaccn22`。
- 再由 `iaaccn22` 通过内网同步到 `iaaccn23~29`。
- 远端数据目录固定优先使用 `/home/zhangziwen/sim-datasets-data`。
- 每台机器写入部署探测结果：
  - 主仓库 commit
  - 子模块 commit
  - 数据目录可见性
  - conda 环境可用性
  - 磁盘空间

### dispatch

任务粒度为：

```text
(algorithm, dataset_id, seed, budget)
```

任务 ID 格式为：

```text
<algorithm>__seed520__<dataset_id>
```

任务输出目录为：

```text
benchmark-runs/compliance/<batch_id>/runs/<algorithm>/seed520/<dataset_id>/
```

分发可复用已有远端队列和 host 负载逻辑，但必须保留统一 `task_id` 和标准输出目录。

### monitor

控制器周期性读取：

- `task_status.jsonl`
- `controller.log`
- `progress/minute_*.json`
- 任务目录更新时间

monitor 阶段只记录状态，不修改代码，不判断根因。状态集合为：

```text
pending
running
completed
failed
stale
timeout
```

### audit

审计器只依赖落盘产物，不依赖 tmux 是否存在。

每个任务必须检查：

- 是否存在最终结果。
- 是否存在分钟级过程快照。
- wall time 是否达到阈值。
- final metrics 是否存在且有限。
- equation 或 canonical artifact 是否可解析。
- launcher 状态是否可分类。

审计输出：

```text
audit/task_audit.csv
audit/budget_compliance_summary.csv
audit/failure_cases.csv
audit/early_stop_cases.csv
audit/missing_artifact_cases.csv
audit/metric_failure_cases.csv
```

### classify

失败分类固定为：

```text
early_stop
missing_result
missing_progress
metric_invalid
artifact_invalid
timeout_unrecovered
runtime_crash
dispatch_failure
unknown
```

`unknown` 只能作为临时状态。若持续出现 `unknown`，说明审计器缺少规则，需要先补审计器而不是直接修 wrapper。

### repair

Codex 每轮只修复一类问题或一个算法族问题。修复必须写入：

```text
repair/round_N/
  diagnosis.md
  changed_files.txt
  commit.txt
  rerun_tasks.csv
  rerun_result_summary.csv
```

每轮修复必须记录：

- 失败证据。
- 根因判断。
- 修改文件。
- commit。
- 验证命令。
- rerun 范围。

### rerun

重跑只针对失败项，不全量重跑。

输入来自：

```text
audit/failure_cases.csv
```

生成：

```text
repair/round_N/rerun_tasks.csv
```

重跑完成后再次进入 `audit → classify`。直到 `failure_cases.csv` 为空。

## 合规判定标准

阶段一每个任务必须满足：

- 预算合规：运行时间 `>= 3300s`，或明确标记为 timeout 后成功恢复。
- 结果合规：存在 `result.json` 或标准等价结果。
- 过程合规：存在分钟级 `progress/minute_*.json`，且至少有早期快照和后期快照。
- 指标合规：`valid`、`id_test`、`ood_test` 的核心指标非空、有限、可解析。
- artifact 合规：最终 equation 或 canonical artifact 可被 evaluator / normalizer 接受。
- 状态合规：任务状态可归类，不能只有无法解释的缺失。

批次通过条件：

- `750 / 750` 任务完成审计。
- `audit/failure_cases.csv` 为空。
- `audit/budget_compliance_summary.csv` 显示 15 个算法全部合规。
- `heartbeat.json` 中 `needs_codex=false`。

## Heartbeat 协议

每个批次维护：

```text
benchmark-runs/compliance/<batch_id>/heartbeat.json
```

字段定义：

```json
{
  "batch_id": "compliance_15alg_ssr50_seed520_1h_20260529-120000",
  "phase": "running",
  "updated_at": "2026-05-29T12:30:00+08:00",
  "total_tasks": 750,
  "pending": 0,
  "running": 128,
  "finished": 615,
  "failed": 7,
  "stale": 0,
  "needs_codex": true,
  "codex_reason": "early_stop detected for e2esr",
  "latest_audit": "audit/failure_cases.csv",
  "latest_rerun_queue": "repair/rerun_after_repair_001.csv"
}
```

`phase` 允许值：

```text
preparing
deploying
running
audit
repair
rerun
done
blocked
```

`needs_codex` 为 `true` 时，表示 `/goal` 接管后应优先读取 `codex_reason`、`latest_audit` 和 `latest_rerun_queue`。

## Goal 接管协议

阶段一 goal 文件为：

```text
benchmark-control/compliance/goals/1h_compliance_goal.md
```

Codex 每轮通过 `/goal` 接管时执行：

1. 读取 `benchmark-runs/compliance/latest` 指向的批次。
2. 读取 `heartbeat.json`。
3. 读取 `audit/task_audit.csv`。
4. 读取 `audit/failure_cases.csv`。
5. 判断下一步属于：
   - 继续等待
   - 重新审计
   - 修复代码
   - 重新部署
   - 只重跑失败项
   - 阻塞并请求人工决策
6. 如果修改代码，必须 commit。
7. 同步到 `iaaccn22~29`。
8. 只重跑失败项。
9. 更新 `repair/repair_log.md` 和轻量摘要。

Codex 介入等级：

```text
observe
audit
repair_required
blocked
```

禁止事项：

- 不全量重跑已合规任务。
- 不删除历史批次。
- 不把 `benchmark-runs/` 提交进 Git。
- 不在没有日志证据时修改 wrapper。
- 不绕过审计器直接宣布通过。

## Git 策略

进入 Git：

- `benchmark-control/`
- goal 文件
- manifest / audit / repair 模板
- 控制器脚本
- 审计脚本
- 轻量摘要
- 设计文档

不进入 Git：

- `benchmark-runs/`
- 方程结果
- 每分钟快照
- stdout / stderr
- checkpoint
- hall of fame
- 算法原生中间结果

## 实施边界

本设计文档只定义系统边界和闭环协议。后续实现应拆成独立计划：

1. 目录和 goal 骨架。
2. manifest 生成器。
3. 审计器。
4. heartbeat 写入器。
5. 远端分发适配。
6. rerun 生成器。
7. `/goal` 执行说明。

实现前应基于本设计另写详细实施计划。
