# Core50 四探针：上一轮664结果 vs 当前重跑结果排序对比

口径：只比较 `udsr / imcts / dso / pyoperon`；上一轮从 664 结果中按 `dataset_rel` 限定到当前 Core50 的 50 个数据集；指标为 dataset×algorithm 的 seed-median OOD log10 NMSE，再跨 50 数据集取平均，越低越好。上一轮无指标 run 按 12 惩罚。

## 排序

| scenario | rank | algorithm | mean OOD log NMSE | median OOD log NMSE |
|---|---:|---|---:|---:|
| current_core50_5seed | 1 | udsr | -5.646 | -4.737 |
| current_core50_5seed | 2 | imcts | -3.960 | -7.812 |
| current_core50_5seed | 3 | dso | -2.458 | -0.632 |
| current_core50_5seed | 4 | pyoperon | 1.079 | -0.075 |
| previous_probe4_664_restricted_core50_3seed | 1 | udsr | -5.431 | -4.539 |
| previous_probe4_664_restricted_core50_3seed | 2 | imcts | -4.284 | -11.380 |
| previous_probe4_664_restricted_core50_3seed | 3 | dso | -1.848 | -0.512 |
| previous_probe4_664_restricted_core50_3seed | 4 | pyoperon | 0.820 | -0.203 |

## 一致性

- 上一轮排序：`udsr > imcts > dso > pyoperon`
- 当前重跑排序：`udsr > imcts > dso > pyoperon`
- Spearman：`1.000`
- Kendall tau：`1.000`
- 算法级 pairwise agreement：`6/6 = 1.000`
- 数据集级 pairwise agreement：`271/300 = 0.903`
