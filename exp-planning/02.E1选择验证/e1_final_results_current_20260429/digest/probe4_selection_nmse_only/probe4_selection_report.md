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

| rank | algorithm | taxonomy | score | finite_id_ood | explosion_1e12 | discrimination | coverage |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | pysr | evolutionary_gp | 0.5998 | 0.985 | 0.120 | 0.984 | 0.867 |
| 2 | imcts | mcts | 0.4809 | 0.990 | 0.020 | 0.574 | 0.891 |
| 3 | udsr | rl_hybrid | 0.4690 | 0.990 | 0.005 | 0.572 | 0.782 |
| 4 | gplearn | classic_gp | 0.4482 | 1.000 | 0.075 | 0.408 | 1.000 |
| 5 | tpsr | pretrained_neural | 0.4476 | 0.975 | 0.115 | 0.442 | 0.976 |
| 6 | dso | rl_policy | 0.4403 | 0.990 | 0.010 | 0.442 | 0.891 |
| 7 | ragsr | rag_hybrid | 0.4148 | 0.885 | 0.195 | 0.649 | 0.539 |
| 8 | pyoperon | evolutionary_gp | 0.3687 | 0.975 | 0.000 | 0.179 | 0.976 |
| 9 | e2esr | pretrained_neural | 0.3655 | 0.695 | 0.065 | 0.601 | 0.555 |
| 10 | qlattice | graph_hybrid | 0.3610 | 0.950 | 0.000 | 0.228 | 0.844 |

## Top 4-Algorithm Combos

| rank | combo | score | stability | discrimination | complementarity | coverage | finite_id_ood |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `imcts;pysr;ragsr;udsr` | 0.7893 | 0.958 | 0.695 | 0.793 | 0.770 | 0.963 |
| 2 | `imcts;pysr;ragsr;tpsr` | 0.7832 | 0.954 | 0.662 | 0.790 | 0.818 | 0.959 |
| 3 | `pysr;ragsr;tpsr;udsr` | 0.7810 | 0.954 | 0.662 | 0.793 | 0.791 | 0.959 |
| 4 | `gplearn;imcts;pysr;ragsr` | 0.7729 | 0.960 | 0.654 | 0.748 | 0.824 | 0.965 |
| 5 | `gplearn;pysr;ragsr;udsr` | 0.7715 | 0.960 | 0.653 | 0.754 | 0.797 | 0.965 |
| 6 | `dso;pysr;ragsr;udsr` | 0.7707 | 0.958 | 0.662 | 0.759 | 0.770 | 0.963 |
| 7 | `e2esr;imcts;pysr;ragsr` | 0.7692 | 0.884 | 0.702 | 0.816 | 0.713 | 0.889 |
| 8 | `imcts;pysr;qlattice;ragsr` | 0.7681 | 0.948 | 0.609 | 0.816 | 0.785 | 0.953 |
| 9 | `imcts;pysr;tpsr;udsr` | 0.7675 | 0.980 | 0.643 | 0.673 | 0.879 | 0.985 |
| 10 | `dso;imcts;pysr;ragsr` | 0.7645 | 0.958 | 0.662 | 0.722 | 0.797 | 0.963 |

## Caveats

- This should be used as the first pass for Probe-4 selection, not as the final decision.
- Runtime/status based operational stability should be added once raw result tables are available.
- `valid_extreme_error` is treated as an informative finite result; missing/nonfinite metrics are penalized.
