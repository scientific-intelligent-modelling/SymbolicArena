# SymbolFit 噪声内部搜索轨迹重跑

本批次只重跑 SymbolFit 的 `noise001` 与 `noise005`，用于替换旧的
`observed_numeric_best_so_far_native_snapshot.v1` 轨迹。旧实验结果保持只读。

## 固定实验网格

- 50 个 SSR-50 数据集
- seeds：520、521、522
- noise：0.01、0.05
- 共 `50 x 3 x 2 = 300` 个任务
- 每个任务 10800 秒，分钟快照间隔 60 秒
- 每个任务单 worker、单 Julia/PySR 进程

## 调度边界

- batch：`symbolfit_noise_internal_progress_v1_20260910_full`
- controller：`iaaccn22`
- workers：`iaaccn22,23,24,26,27,28,29`
- `iaaccn25` 因实时审计发现异常系统负载和遗留 FePySR 进程，本批次排除
- 输出写入全新的 `experiments/<batch>/...`，不覆盖 clean 或旧 noise 结果
- 每台最多 64 个任务，CPU 权重上限 95%，内存使用上限 90%

## 执行顺序

1. 在 `iaaccn22` 仓库根目录执行 `commands/preflight_full.sh`。
2. 检查 `preflight_report.json` 中七台机器全部通过。
3. 通过 tmux 直接执行 `commands/run_full_controller.sh`。
4. 启动后检查 state 为 300 个任务，两个 noise 条件各 150 个。

最终验收必须确认每个任务具有 1 至 180 分钟快照，终点为内部搜索候选，
并包含 `condition`、`train_label_noise`、`source_internal_loss`、候选首次发现时间和
SymbolFit 坐标变换证据。
