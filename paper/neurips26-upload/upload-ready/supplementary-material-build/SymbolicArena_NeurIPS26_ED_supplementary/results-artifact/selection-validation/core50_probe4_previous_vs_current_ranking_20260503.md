# Core-50 Probe-4: Previous GT-Reservoir-664 results vs current rerun ranking comparison

Protocol: compare only `udsr / imcts / dso / pyoperon`; previous 664-task results are restricted by `dataset_rel` to the current Core-50 tasks; the metric is dataset-by-algorithm seed-median OOD log10 NMSE, averaged over 50 datasets, where lower is better. Previous runs without metrics are penalized as 12.

## Ranking

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

## Consistency

- noteRanking `udsr > imcts > dso > pyoperon`
- current noteRanking `udsr > imcts > dso > pyoperon`
- Spearman `1.000`
- Kendall tau `1.000`
- Algorithm-level pairwise agreement `6/6 = 1.000`
- Dataset-level pairwise agreement `271/300 = 0.903`
