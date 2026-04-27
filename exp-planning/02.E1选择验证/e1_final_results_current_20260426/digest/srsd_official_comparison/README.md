# SRSD 官方 benchmark 与本地 E1 对比

## 官方口径

- 官方仓库：`https://github.com/omron-sinicx/srsd-benchmark`。
- 官方论文/报告：`Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery`。
- 官方 SRSD 创建了 `120` 个非 dummy SRSD-Feynman 数据集，并另外创建 `120` 个带 dummy variables 的版本。
- 官方主要报告的是 `R2 > 0.999` accuracy、solution rate、NED；不是按 `NMSE > 1` 的爆炸表。

## 官方结果摘录

非 dummy SRSD-Feynman 上，官方 PySR 并不是全炸：

| Group | PySR accuracy R2>0.999 | PySR solution rate |
|---|---:|---:|
| Easy | 66.7% | 60.0% |
| Medium | 45.0% | 30.0% |
| Hard | 38.0% | 4.00% |

带 dummy variables 后，官方 PySR 明显下降：

| Group | PySR accuracy R2>0.999 | PySR solution rate |
|---|---:|---:|
| Easy + dummy | 20.0% | 20.0% |
| Medium + dummy | 10.0% | 5.00% |
| Hard + dummy | 2.00% | 0.00% |

官方还特别指出 dummy variables 让 SRSD-Feynman 极具挑战，且所有基线都不能稳健过滤 dummy variables。

## 本地 E1 的 SRSD 子集

本地 E1 不是全量 SRSD，而是 Candidate-200 中的 `70` 个 SRSD 数据集，分布如下：

| difficulty | dummy | count |
|---|---:|---:|
| easy | false | 1 |
| easy | true | 16 |
| medium | false | 1 |
| medium | true | 21 |
| hard | false | 14 |
| hard | true | 17 |

也就是说，我们这 70 个明显偏向 `dummy` 和 `hard/medium`，不是官方全量均匀口径。

## 本地 E1 按算法汇总

| method   |   runs |   train_gt1 |   id_gt1 |   ood_gt1 |   any_gt1 |   id_r2_gt_0.999 |   ood_r2_gt_0.999 |   median_id_nmse |   median_ood_nmse |
|:---------|-------:|------------:|---------:|----------:|----------:|-----------------:|------------------:|-----------------:|------------------:|
| drsr     |     70 |          55 |       58 |        58 |        61 |                2 |                 2 |      1.93062e+24 |       2.35411e+24 |
| dso      |     70 |           0 |        9 |        17 |        21 |               18 |                15 |      0.555067    |       0.678739    |
| gplearn  |     70 |           8 |       20 |        28 |        32 |               13 |                 9 |      0.987949    |       1           |
| llmsr    |     70 |          60 |       62 |        62 |        64 |                2 |                 2 |      1.31931e+34 |       5.98204e+33 |
| pyoperon |     70 |          17 |        4 |        15 |        28 |                5 |                 3 |      0.908627    |       1           |
| pysr     |     70 |          25 |       26 |        31 |        31 |               24 |                22 |      0.679277    |       1           |
| tpsr     |     70 |          35 |       34 |        39 |        42 |                0 |                 0 |      1.00021     |       1.00004     |

## 本地 E1 按组拆分

| method   | difficulty   | dummy   |   n |   id_gt1 |   ood_gt1 |   any_gt1 |   id_r2_gt_0.999 |   ood_r2_gt_0.999 |   median_id_nmse |   median_ood_nmse |
|:---------|:-------------|:--------|----:|---------:|----------:|----------:|-----------------:|------------------:|-----------------:|------------------:|
| drsr     | easy         | False   |   1 |        1 |         1 |         1 |                0 |                 0 |      9.52627e+26 |       9.3487e+20  |
| drsr     | easy         | True    |  16 |       14 |        15 |        15 |                0 |                 0 |      3.23496e+29 |       3.19521e+32 |
| drsr     | hard         | False   |  14 |       10 |        10 |        11 |                1 |                 1 |      1.37377e+24 |       1.59229e+25 |
| drsr     | hard         | True    |  17 |       14 |        14 |        15 |                1 |                 1 |      1.13738e+20 |       4.68488e+24 |
| drsr     | medium       | False   |   1 |        1 |         1 |         1 |                0 |                 0 |      1.02572e+16 |       2.79566e+14 |
| drsr     | medium       | True    |  21 |       18 |        17 |        18 |                0 |                 0 |      6.73798e+20 |       4.6923e+21  |
| gplearn  | easy         | False   |   1 |        0 |         0 |         0 |                0 |                 0 |      1           |       1           |
| gplearn  | easy         | True    |  16 |        8 |         9 |         9 |                4 |                 3 |      1           |       1.00574     |
| gplearn  | hard         | False   |  14 |        3 |         6 |         7 |                2 |                 2 |      0.917901    |       1           |
| gplearn  | hard         | True    |  17 |        3 |         5 |         7 |                2 |                 2 |      0.835429    |       0.999945    |
| gplearn  | medium       | False   |   1 |        1 |         1 |         1 |                0 |                 0 | 406637           |    3092.6         |
| gplearn  | medium       | True    |  21 |        5 |         7 |         8 |                5 |                 2 |      0.55422     |       0.910087    |
| llmsr    | easy         | False   |   1 |        1 |         1 |         1 |                0 |                 0 |      3.44219e+37 |       2.65469e+31 |
| llmsr    | easy         | True    |  16 |       14 |        15 |        15 |                0 |                 0 |      2.52068e+38 |       7.24123e+40 |
| llmsr    | hard         | False   |  14 |       11 |        10 |        12 |                1 |                 1 |      3.49361e+32 |       4.06619e+33 |
| llmsr    | hard         | True    |  17 |       16 |        16 |        16 |                1 |                 1 |      2.24332e+32 |       4.89288e+33 |
| llmsr    | medium       | False   |   1 |        1 |         1 |         1 |                0 |                 0 |      1.91352e+24 |       1.32233e+23 |
| llmsr    | medium       | True    |  21 |       19 |        19 |        19 |                0 |                 0 |      2.57762e+34 |       2.91856e+38 |
| pysr     | easy         | False   |   1 |        1 |         1 |         1 |                0 |                 0 |   1798.23        |      81.3036      |
| pysr     | easy         | True    |  16 |        9 |         9 |         9 |                6 |                 6 |      1.11967     |       1.01946     |
| pysr     | hard         | False   |  14 |        4 |         6 |         6 |                4 |                 4 |      0.611086    |       1           |
| pysr     | hard         | True    |  17 |        6 |         8 |         8 |                3 |                 2 |      0.0710095   |       0.592112    |
| pysr     | medium       | False   |   1 |        0 |         0 |         0 |                0 |                 0 |      0.811654    |       0.930838    |
| pysr     | medium       | True    |  21 |        6 |         7 |         7 |               11 |                10 |      7.43954e-06 |       0.00152684  |

## 判断

1. 官方 SRSD benchmark 不是“都炸了”。PySR/uDSR 在非 dummy SRSD 上有相当比例达到 `R2 > 0.999`，尤其 Easy/Medium。
2. 官方也确认 SRSD 比原 Feynman/SRBench 更难；带 dummy 后，PySR 的 solution rate 在 Hard + dummy 是 `0%`。
3. 我们 E1 的 SRSD 子集更偏难，而且很多是经过 PySR/LLMSR 差异筛选进入 Candidate-200 的，所以本地爆炸比例比官方全量口径更高是合理的。
4. 但本地 `llmsr/drsr/tpsr` 的 `NMSE` 爆到极大值，不等同于官方 “未达到 R2>0.999”；这是更强的数值尺度失败，需要在 leaderboard 里用 `log_nmse / clipped score / explosion flag` 单独处理。
5. 若要做完全公平的 SRSD 官方复现，需要跑官方全量 120/240，而不是只看当前 Candidate-200 中的 70 个。
