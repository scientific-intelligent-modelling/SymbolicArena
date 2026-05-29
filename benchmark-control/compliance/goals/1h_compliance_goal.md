# 1h Benchmark Compliance Goal

## 目标

完成阶段一 benchmark 合规验证：

```text
15 algorithms × SSR50 × seed520 × 1h
```

目标不是追求最终指标最优，而是验证所有算法都能打满预算、正常落盘、正常产出可审计结果。

## 成功条件

- 750 个任务全部完成审计。
- `audit/failure_cases.csv` 为空。
- 每个任务运行时间达到合规阈值，默认 `>=3300s`，或明确标记为 timeout 后成功恢复。
- 每个任务存在最终结果、分钟级过程快照、有效指标和可解析 artifact。
- `audit/budget_compliance_summary.csv` 显示 15 个算法全部合规。
- `heartbeat.json` 标记 `needs_codex=false`。

## 当前批次发现方式

默认读取：

```text
benchmark-runs/compliance/latest
```

`latest` 应指向当前正在执行或最近一次执行的批次目录。

## Codex 每轮接管步骤

1. 读取 `heartbeat.json`，判断当前阶段。
2. 读取 `audit/task_audit.csv`，确认任务级状态。
3. 读取 `audit/failure_cases.csv`，确认失败项和失败类别。
4. 判断下一步动作：
   - `observe`：任务仍在正常运行，继续等待。
   - `audit`：任务结束或状态变化，需要重新审计。
   - `repair_required`：已有证据表明需要修改代码。
   - `blocked`：环境、权限、磁盘或数据问题需要人工决策。
5. 如果需要修复代码，先根据日志和审计证据定位根因。
6. 修改代码后运行针对性验证并提交 commit。
7. 同步代码到 `iaaccn22~29`。
8. 用过滤 `rsync` 收集远端 `experiments/<batch_id>/` 审计所需轻量产物。
9. 将收集到的产物 harvest 到 `benchmark-runs/compliance/latest/runs/`。
10. 只重跑失败项，不重跑已合规任务。
11. 更新 `repair/repair_log.md` 和轻量摘要。

## 禁止事项

- 不全量重跑已合规任务。
- 不删除历史批次。
- 不提交 `benchmark-runs/`。
- 不在没有日志证据时修改 wrapper。
- 不绕过审计器直接宣布通过。
- 不把 `frozen-results/` 用作新的合规实验产物目录。

## 停止条件

当以下条件同时满足时，本 goal 完成：

```text
audit/failure_cases.csv 为空
budget_compliance_summary.csv 显示 15 个算法全部合规
heartbeat.json 中 needs_codex=false
```

## 标准命令

以下命令默认在仓库根目录执行。`benchmark-runs/` 是 Git ignore 的运行产物目录；这里的 prepare 和 audit 都只操作本地控制面文件，不会启动远端全量实验。

准备批次：

```bash
BATCH_ID="compliance_15alg_ssr50_seed520_1h_$(date +%Y%m%d-%H%M%S)"
mkdir -p "benchmark-runs/compliance/${BATCH_ID}"
python benchmark-control/compliance/launchers/prepare_batch.py \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/compliance/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/write_remote_sync_commands.py \
  --batch-dir benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/write_stage1_queue_commands.py \
  --batch-dir benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/check_stage1_readiness.py \
  --batch-dir benchmark-runs/compliance/latest
```

审计批次：

```bash
BATCH_ID="$(basename "$(readlink -f benchmark-runs/compliance/latest)")"

python benchmark-control/compliance/launchers/collect_remote_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --batch-id "${BATCH_ID}" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --controller-host iaaccn22 \
  --use-internal-ips

python benchmark-control/compliance/launchers/harvest_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn22 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn23 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn24 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn25 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn26 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn27 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn28 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn29

python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --write-heartbeat \
  --write-rerun \
  --round-id 1
```

远端 dispatch 只能在 smoke 验证通过且获得明确确认后执行；上面的命令不会启动任何远端实验。
