# Probe-4 Selection Metrics

## Scope

- Input: `exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv`
- Candidate metadata: `exp-planning/02.E1选择验证/generated/candidate200_unified.csv`
- Excluded algorithms: `drsr`, `llmsr`
- Current report is `NMSE-only`: latest 20260429 digest has no runtime/status/failure fields.
- Practical cost is therefore set to a neutral constant in combo scoring.

## Output Files

- `probe4_algorithm_scores_nmse_only.csv`
- `probe4_pairwise_complementarity_nmse_only.csv`
- `probe4_combo_scores_nmse_only.csv`
- `probe4_family_coverage_nmse_only.csv`

## Top Algorithms By Available Metric Score

| rank | algorithm | taxonomy | score | finite_id_ood | explosion_gt_100 | discrimination | coverage |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | pysr | evolutionary_gp | 0.6004 | 0.985 | 0.205 | 0.984 | 0.867 |
| 2 | imcts | mcts | 0.4812 | 0.990 | 0.055 | 0.574 | 0.891 |
| 3 | udsr | rl_hybrid | 0.4690 | 0.990 | 0.015 | 0.572 | 0.782 |
| 4 | gplearn | classic_gp | 0.4488 | 1.000 | 0.140 | 0.408 | 1.000 |
| 5 | tpsr | pretrained_neural | 0.4479 | 0.975 | 0.250 | 0.442 | 0.976 |
| 6 | dso | rl_policy | 0.4406 | 0.990 | 0.050 | 0.442 | 0.891 |
| 7 | ragsr | rag_hybrid | 0.4148 | 0.885 | 0.585 | 0.649 | 0.539 |
| 8 | pyoperon | evolutionary_gp | 0.3687 | 0.975 | 0.035 | 0.179 | 0.976 |
| 9 | e2esr | pretrained_neural | 0.3655 | 0.695 | 0.140 | 0.601 | 0.555 |
| 10 | qlattice | graph_hybrid | 0.3610 | 0.950 | 0.030 | 0.228 | 0.844 |

## Top 4-Algorithm Combos

| rank | combo | score | stability | discrimination | complementarity | coverage | finite_id_ood |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `imcts;pysr;ragsr;udsr` | 0.7895 | 0.960 | 0.695 | 0.793 | 0.770 | 0.963 |
| 2 | `imcts;pysr;ragsr;tpsr` | 0.7835 | 0.956 | 0.662 | 0.790 | 0.818 | 0.959 |
| 3 | `pysr;ragsr;tpsr;udsr` | 0.7812 | 0.956 | 0.662 | 0.793 | 0.791 | 0.959 |
| 4 | `gplearn;imcts;pysr;ragsr` | 0.7732 | 0.962 | 0.654 | 0.748 | 0.824 | 0.965 |
| 5 | `gplearn;pysr;ragsr;udsr` | 0.7718 | 0.962 | 0.653 | 0.754 | 0.797 | 0.965 |
| 6 | `dso;pysr;ragsr;udsr` | 0.7710 | 0.960 | 0.662 | 0.759 | 0.770 | 0.963 |
| 7 | `e2esr;imcts;pysr;ragsr` | 0.7694 | 0.886 | 0.702 | 0.816 | 0.713 | 0.889 |
| 8 | `imcts;pysr;qlattice;ragsr` | 0.7684 | 0.950 | 0.609 | 0.816 | 0.785 | 0.953 |
| 9 | `imcts;pysr;tpsr;udsr` | 0.7678 | 0.982 | 0.643 | 0.673 | 0.879 | 0.985 |
| 10 | `dso;imcts;pysr;ragsr` | 0.7648 | 0.960 | 0.662 | 0.722 | 0.797 | 0.963 |

## Caveats

- This should be used as the first pass for Probe-4 selection, not as the final decision.
- Runtime/status based operational stability should be added once raw result tables are available.
- `valid_extreme_error` is treated as an informative finite result; missing/nonfinite metrics are penalized.
