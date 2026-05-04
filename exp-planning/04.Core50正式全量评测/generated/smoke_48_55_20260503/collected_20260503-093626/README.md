# Core50 12算法 48~55 5分钟 smoke 汇总

- 批次：`core50_12alg_5min_smoke_20260503-093626`
- 数据集：`Nguyen-6` (`core50_index=9`)
- 规模：`8 hosts x 12 algorithms = 96 runs`
- 预算：`timeout_in_seconds=300`，外层保护 `480s`

## 按算法

| algorithm | total | ok/ret0 | timed_out/ret0 | none/ret124 | other |
|---|---:|---:|---:|---:|---:|
| drsr | 8 | 0 | 0 | 8 | 0 |
| dso | 8 | 8 | 0 | 0 | 0 |
| e2esr | 8 | 8 | 0 | 0 | 0 |
| gplearn | 8 | 8 | 0 | 0 | 0 |
| imcts | 8 | 8 | 0 | 0 | 0 |
| llmsr | 8 | 8 | 0 | 0 | 0 |
| pyoperon | 8 | 8 | 0 | 0 | 0 |
| pysr | 8 | 8 | 0 | 0 | 0 |
| qlattice | 8 | 8 | 0 | 0 | 0 |
| ragsr | 8 | 1 | 7 | 0 | 0 |
| tpsr | 8 | 8 | 0 | 0 | 0 |
| udsr | 8 | 8 | 0 | 0 | 0 |

## 按机器

| host | total | ok/ret0 | timed_out/ret0 | none/ret124 | other |
|---|---:|---:|---:|---:|---:|
| iaaccn48 | 12 | 11 | 0 | 1 | 0 |
| iaaccn49 | 12 | 10 | 1 | 1 | 0 |
| iaaccn50 | 12 | 10 | 1 | 1 | 0 |
| iaaccn51 | 12 | 10 | 1 | 1 | 0 |
| iaaccn52 | 12 | 10 | 1 | 1 | 0 |
| iaaccn53 | 12 | 10 | 1 | 1 | 0 |
| iaaccn54 | 12 | 10 | 1 | 1 | 0 |
| iaaccn55 | 12 | 10 | 1 | 1 | 0 |

## 关键结论

- `gplearn/pysr/pyoperon/llmsr/dso/udsr/tpsr/e2esr/qlattice/imcts`：8 台均能正常跑通并落盘。
- `ragsr`：8 台均能启动并按 300s 预算收口；其中 7 台 launcher 状态为 `timed_out`，这是短预算 smoke 的正常信号，已有分钟级快照。
- `drsr`：8 台均产生了分钟级快照和有效候选，但 wrapper 未按 300s 预算退出，最终被外层 480s 保护超时杀掉；需要修 DRSR 的超时收口/恢复链。

## DRSR 补充核验

- 8 台机器均无残留 `drsr` / smoke 进程。
- 8 台机器均写出了 DRSR 分钟级快照，最新快照为 `minute_0007.json`。
- 最新快照均包含可评估候选；多数有 `valid/id/ood` NMSE，说明算法本体和 LLM 调用在跑，问题集中在 wrapper 超时收口没有把快照恢复成最终 `result.json`。
