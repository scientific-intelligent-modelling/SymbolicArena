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
8. 只重跑失败项，不重跑已合规任务。
9. 更新 `repair/repair_log.md` 和轻量摘要。

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
