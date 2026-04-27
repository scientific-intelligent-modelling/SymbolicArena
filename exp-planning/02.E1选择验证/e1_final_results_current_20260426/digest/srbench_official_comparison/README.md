# SRBench 官方结果对比初版

## 数据源

- 官方页面：`https://cavalab.org/srbench/` 与 `https://cavalab.org/srbench/results/`。
- 官方汇总 CSV：`docs/csv/groundtruth.csv`、`docs/csv/blackbox_results.csv`、`docs/csv/pareto.csv`。
- 官方方法配置：`experiment/methods/PySRRegressor.py`、`experiment/methods/OperonRegressor.py`。
- 本地对比：`one_seed_probe_task_results.csv` 与 E1 `e1_result_table.csv`。

## 可直接对齐的范围

- `docs/csv/groundtruth.csv` 当前只含 legacy 14 个算法；与 E1 7 算法直接交集是 `gplearn -> gplearn` 和 `pyoperon -> Operon`。
- `PySR` 在 SRBench 官网方法列表中存在，但不在 `docs/csv/groundtruth.csv` 小表里；需要完整 `/results/*.feather` 才能做官方 PySR per-dataset 对齐。当前网络下大文件下载未完成。
- 因此本报告把 PySR 分成两块：本地 `srbench1.0/srbench2025` 自查，以及官方 PySR 配置差异解释。

## 官方 ground-truth overlap 摘要

| method   | official_algorithm   |   overlap_rows |   official_accuracy_solution_count |   official_symbolic_rate_mean |   local_id_accuracy_solution_count |   local_id_nmse_gt1_count | id_change_counts                                                                                                                   |   local_ood_accuracy_solution_count |   local_ood_nmse_gt1_count | ood_change_counts                                                                                                                    |
|:---------|:---------------------|---------------:|-----------------------------------:|------------------------------:|-----------------------------------:|--------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------|------------------------------------:|---------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------|
| gplearn  | gplearn              |             33 |                                  7 |                       13.6364 |                                  4 |                         2 | {'neither_accuracy_solution': 25, 'official_good__local_id_not': 4, 'both_accuracy_solution': 3, 'local_id_good__official_not': 1} |                                   4 |                          3 | {'neither_accuracy_solution': 25, 'official_good__local_ood_not': 4, 'both_accuracy_solution': 3, 'local_ood_good__official_not': 1} |
| pyoperon | Operon               |             33 |                                 31 |                       14.8485 |                                  1 |                         0 | {'official_good__local_id_not': 30, 'neither_accuracy_solution': 2, 'both_accuracy_solution': 1}                                   |                                   0 |                          2 | {'official_good__local_ood_not': 31, 'neither_accuracy_solution': 2}                                                                 |

## 本地 PySR 在 SRBench 来源数据上的炸点

- `srbench1.0`: 0 / 33 个 `valid/id/ood` 任一 NMSE > 1。
- `srbench2025`: 1 / 6 个 `valid/id/ood` 任一 NMSE > 1。

本地结论：`srbench1.0` 上 PySR 没有炸；`srbench2025` 只有 `first_principles_kepler` 的 `valid_nmse > 1`，但 `id/ood` 不炸。

## PySR 本地 probe vs E1

在 `srbench1.0/srbench2025` 来源任务上，PySR 的 probe 和 E1 对比为：

| change_type   |   count |
|:--------------|--------:|
| neither_gt1   |      38 |
| both_gt1      |       1 |

结论：PySR 在真正 `srbench1.0` 来源上 probe/E1 都没有系统性炸；你看到的严重爆炸主要来自 `srsd` 版本的 Feynman，而不是 SRBench 官方/PMLB 版本。

## 主要不一致原因

1. **数据源不等价**：SRBench 官方 Feynman 用的是 PMLB/SRBench 标准化范围；我们的 `srsd/feynman-*` 用原始物理尺度，输入可到 `1e25`、目标可到 `1e-38`。同名公式跨来源不能视为同一难度。
2. **PySR 官方配置更重**：官方 PySR 是约 2 小时预算、`population_size=100`、`populations=max(15, cpu_count()*2)`、`maxsize=40`、`maxdepth=20`、`batching=True`、`turbo=True`、全 CPU 并行；E1 是 1 小时、串行、`population_size=64`、`populations=8`、`maxsize=30`、`maxdepth=10`。
3. **指标口径不完全一致**：SRBench 汇总小表给的是 `rmse_test/log_mse_test/accuracy_solution/symbolic_solution_rate`；E1 主表是 `valid/id/ood NMSE/R2`，且有 OOD split。
4. **E1 是单 seed**：官方 ground-truth 汇总是多 trial/多 noise 汇总；E1 当前是一个 seed，所以局部翻转不应直接解释成官方结果错或我们结果错。
5. **官方汇总 CSV 不含 PySR per-dataset 明细**：PySR 官方完整对齐需要继续拉取 `/results/ground-truth_results.feather`。

## 生成文件

- `official_groundtruth_overlap_comparison.csv`
- `official_groundtruth_overlap_summary.csv`
- `local_pysr_srbench_family_check.csv`
- `local_pysr_cross_source_feynman_pairs.csv`
