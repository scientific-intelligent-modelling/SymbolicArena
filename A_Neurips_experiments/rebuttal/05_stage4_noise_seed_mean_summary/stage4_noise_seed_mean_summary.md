# NeurIPS Stage 4 clean/noise 多种子均值结果

## 结论

本次整理只复用冻结的 Stage 4 run-level 数据，未重跑算法。完整网格为
`12 × 50 × 5 × 4 = 12000` 条记录，每个算法、任务、条件均恰好包含五个种子。

按用户要求对五个种子的逐 run 质量分取算术均值后，ROBU 前两名为
**DSO（46.74）**和
**PySR（44.29）**。seed-mean ROBU 与论文正式
seed-median ROBU 的算法排序 Spearman 相关系数为
**0.972**；平均绝对分数差为
**0.878** 分，最大差为
**3.023** 分。

这说明两种种子聚合口径总体一致，但不能互换：seed-mean 下的部分中游算法会发生名次变化。
论文和 rebuttal 的正式 ROBU 数字仍应使用 seed-median 原口径；本表适合回答“多个种子求均值”
的补充分析需求。

## 完整性

| Condition | Observed | Usable | Penalized | Usable rate |
| --- | --- | --- | --- | --- |
| Clean | 3000 | 2864 | 136 | 95.47% |
| Noise 0.01 | 3000 | 2874 | 126 | 95.80% |
| Noise 0.05 | 3000 | 2889 | 111 | 96.30% |
| Noise 0.10 | 3000 | 2873 | 127 | 95.77% |

`Penalized` 包括失败、无有效输出、指标不完整或 ID/OOD NMSE 非有限的 run。
这些 run 按正式 failure semantics 使用 `NMSE=100`，没有从均值中删除。

## ROBU 对照

| Mean rank | Algorithm | ROBU seed-mean | Formal rank | ROBU formal median | Difference |
| --- | --- | --- | --- | --- | --- |
| 1 | DSO | 46.74 | 1 | 48.30 | -1.55 |
| 2 | PYSR | 44.29 | 2 | 43.88 | +0.40 |
| 3 | UDSR | 42.19 | 4 | 41.14 | +1.05 |
| 4 | GPLEARN | 42.06 | 3 | 41.38 | +0.68 |
| 5 | IMCTS | 41.51 | 7 | 38.49 | +3.02 |
| 6 | QLATTICE | 39.73 | 5 | 39.70 | +0.02 |
| 7 | LLMSR | 39.69 | 6 | 38.89 | +0.80 |
| 8 | DRSR | 37.70 | 8 | 36.46 | +1.24 |
| 9 | PYOPERON | 31.47 | 9 | 32.61 | -1.14 |
| 10 | TPSR | 27.38 | 10 | 27.64 | -0.26 |
| 11 | E2ESR | 25.91 | 11 | 26.21 | -0.30 |
| 12 | RAGSR | 11.39 | 12 | 11.34 | +0.05 |

## 文件说明

- 逐算法、逐条件五项统计：`stage4_condition_metrics_seed_mean_wide.csv`
- 逐算法、逐任务、逐条件聚合：`stage4_dataset_condition_seed_mean.csv`
- 完整表图：`stage4_noise_seed_mean_table.png` 和 `.pdf`
- 正式口径对照：`stage4_robu_seed_mean_vs_formal.csv`
- 完整复现信息：`reproducibility.json`
