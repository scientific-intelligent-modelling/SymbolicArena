# Benchmark Compliance Control

本目录保存 benchmark 合规闭环的控制面文件。这里的内容进入 Git，用于记录实验如何生成、如何分发、如何审计、如何修复和如何通过 `/goal` 被 Codex 接管。

真实实验产物不放在本目录，而是放在被 Git 忽略的：

```text
benchmark-runs/
```

## 子目录

- `goals/`：Codex `/goal` 接管说明。
- `manifests/`：算法、数据集、预算和任务清单模板或轻量快照。
- `templates/`：批次、队列、部署和重跑模板。
- `audit-schemas/`：审计输出字段和失败分类定义。
- `repair-notes/`：修复摘要和可提交记录。
- `launchers/`：远端分发和控制器脚本。
- `run-summaries/`：从 `benchmark-runs/` 抽取出的轻量批次摘要。

## 阶段一范围

```text
15 algorithms × SSR50 × seed520 × 1h
```

阶段一只验证预算合规和落盘合规。通过后再扩展到 `3 seeds × 24h × noise`。
