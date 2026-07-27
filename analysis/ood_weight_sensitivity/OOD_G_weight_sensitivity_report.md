# OOD-G Weight Sensitivity Report

This report is a local post-processing analysis. No symbolic-regression
algorithm was restarted. The analysis input replaces only the original DRSR
and LLM-SR rows with their existing model-split reruns.

## 1. Data audit

- Original rows: `3000`
- Replacement rows: `500`
- Final rows: `3000`
- Algorithms: `12`
- Core-50 tasks: `50`
- Missing run keys: `0`
- Duplicate run keys: `0`
- Every algorithm-task has five seeds: `True`
- Status counts: `{"no_valid_output": 9, "ok": 2982, "timed_out": 9}`
- `valid_output=False`: `13`
- `timed_out`: `9`
- `no_valid_output`: `9`
- ID NaN / Inf: `101 / 0`
- OOD NaN / Inf: `117 / 0`
- `metric_complete=False`: `136`
- Runs receiving the `e=1e2` metric penalty: `136`

The final grid is constructed as `2500 original + 500 DRSR/LLM-SR rerun`
records. Replacement keys match the original `(algorithm, gid, dataset, seed)`
keys exactly.

## 2. Exact reproduction of w=0.7

The implementation extracts the exact `phi_from_nmse` and `safe_nmse`
definitions from `check/plot_core50_hexagon_metrics.py` without importing its
unrelated symbolic-analysis dependencies. It first applies run-level failure
semantics, then takes the five-seed ID/OOD NMSE medians, and finally computes
task components.

- Maximum algorithm OOD-G error: `2.132e-14`
- Maximum task OOD-G-component error: `1.260e-14`
- Required tolerance: `<1e-6`

| algorithm | computed_OOD_G_w070 | formal_OOD_G_w070 | abs_error |
| --- | --- | --- | --- |
| drsr | 45.97710705 | 45.97710705 | 0.00000000 |
| dso | 48.04119691 | 48.04119691 | 0.00000000 |
| e2esr | 33.76039770 | 33.76039770 | 0.00000000 |
| gplearn | 40.72096337 | 40.72096337 | 0.00000000 |
| imcts | 63.98946974 | 63.98946974 | 0.00000000 |
| llmsr | 42.68755461 | 42.68755461 | 0.00000000 |
| pyoperon | 33.55405867 | 33.55405867 | 0.00000000 |
| pysr | 60.71762139 | 60.71762139 | 0.00000000 |
| qlattice | 35.69644212 | 35.69644212 | 0.00000000 |
| ragsr | 27.63244732 | 27.63244732 | 0.00000000 |
| tpsr | 34.02903279 | 34.02903279 | 0.00000000 |
| udsr | 59.86208575 | 59.86208575 | 0.00000000 |

## 3. Mathematical decomposition

For each algorithm, `Q` and `R` are fixed after the task aggregation:

`OOD_G(a,w) = 100 * [R(a) + w * (Q(a)-R(a))]`.

The verified slope is `100*(Q-R)`, so every score-weight curve is exactly
linear.

| algorithm | Q | R | slope | score_w0700 | score_w0850 | score_w0900 | score_w1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drsr | 0.3361 | 0.7484 | -41.2309 | 45.9771 | 39.7925 | 37.7309 | 33.6078 |
| dso | 0.3278 | 0.8365 | -50.8634 | 48.0412 | 40.4117 | 37.8685 | 32.7822 |
| e2esr | 0.1430 | 0.7916 | -64.8557 | 33.7604 | 24.0320 | 20.7893 | 14.3037 |
| gplearn | 0.2291 | 0.8229 | -59.3820 | 40.7210 | 31.8137 | 28.8446 | 22.9064 |
| imcts | 0.5681 | 0.8075 | -23.9471 | 63.9895 | 60.3974 | 59.2000 | 56.8053 |
| llmsr | 0.3105 | 0.6984 | -38.7934 | 42.6876 | 36.8685 | 34.9289 | 31.0495 |
| pyoperon | 0.1615 | 0.7415 | -57.9978 | 33.5541 | 24.8544 | 21.9545 | 16.1547 |
| pysr | 0.5052 | 0.8452 | -33.9985 | 60.7176 | 55.6178 | 53.9179 | 50.5181 |
| qlattice | 0.2213 | 0.6734 | -45.2110 | 35.6964 | 28.9148 | 26.6542 | 22.1331 |
| ragsr | 0.0636 | 0.7728 | -70.9189 | 27.6324 | 16.9946 | 13.4487 | 6.3568 |
| tpsr | 0.1357 | 0.8176 | -68.1840 | 34.0290 | 23.8014 | 20.3922 | 13.5738 |
| udsr | 0.5485 | 0.7155 | -16.7012 | 59.8621 | 57.3569 | 56.5219 | 54.8517 |

## 4. Score sensitivity

The complete scan covers `w=0.00,0.01,...,1.00`, plus the pre-specified
`w=0.875`. Scores and ranks are in `scores_by_weight.csv` and
`ranks_by_weight.csv`. Full-span plots use the untruncated 0--100 score axis.

## 5. Ranking sensitivity

| interval | weights_evaluated | min_spearman_vs_w070 | min_kendall_vs_w070 | min_top1_overlap | min_top3_overlap | min_top5_overlap | max_pairwise_order_flip_count | changed_algorithms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [0.70,0.80] | 11 | 0.9650 | 0.8788 | 1.0000 | 1.0000 | 1.0000 | 4 | e2esr;pyoperon;pysr;tpsr;udsr |
| [0.80,0.90] | 12 | 0.9650 | 0.8788 | 1.0000 | 1.0000 | 1.0000 | 4 | pyoperon;pysr;tpsr;udsr |
| [0.85,0.95] | 12 | 0.9580 | 0.8485 | 1.0000 | 1.0000 | 1.0000 | 5 | drsr;dso;pyoperon;pysr;tpsr;udsr |
| [0.70,0.95] | 27 | 0.9580 | 0.8485 | 1.0000 | 1.0000 | 1.0000 | 5 | drsr;dso;e2esr;pyoperon;pysr;tpsr;udsr |

Top-1, Top-3, Top-5 overlap, pairwise flips, and changed algorithms for every
weight are recorded in `rank_stability_by_weight.csv`.

Across `[0.70,0.95]`, Top-1, Top-3, and Top-5 memberships never change.
Algorithms with unchanged exact ranks are
`gplearn;iMCTS;LLM-SR;QLattice;RAG-SR`. The largest focused-range rank changes
belong to `PyOperon;TPSR`; the complete per-algorithm
summary is in `algorithm_rank_sensitivity.csv`.

## 6. Pairwise crossing points

- Crossings in `[0.70,0.95]`: `5`
- Crossings in `[0.80,0.90]`: `0`
- Crossings involving a baseline Top-5 algorithm: `17`

Nearest ten crossings to `w=0.7`:

| algorithm_a | algorithm_b | crossing_weight | rank_order_below_crossing | rank_order_above_crossing | involves_baseline_top5 |
| --- | --- | --- | --- | --- | --- |
| e2esr | pyoperon | 0.730088 | e2esr>pyoperon | pyoperon>e2esr | False |
| pyoperon | tpsr | 0.746629 | tpsr>pyoperon | pyoperon>tpsr | False |
| pysr | udsr | 0.749460 | pysr>udsr | udsr>pysr | True |
| qlattice | tpsr | 0.627419 | tpsr>qlattice | qlattice>tpsr | False |
| e2esr | tpsr | 0.780713 | tpsr>e2esr | e2esr>tpsr | False |
| gplearn | llmsr | 0.604481 | gplearn>llmsr | llmsr>gplearn | False |
| e2esr | qlattice | 0.601447 | e2esr>qlattice | qlattice>e2esr | False |
| pyoperon | qlattice | 0.532454 | pyoperon>qlattice | qlattice>pyoperon | False |
| drsr | dso | 0.914284 | dso>drsr | drsr>dso | True |
| drsr | gplearn | 0.410422 | gplearn>drsr | drsr>gplearn | True |

All crossings involving a baseline Top-5 algorithm:

| algorithm_a | algorithm_b | crossing_weight | rank_order_below_crossing | rank_order_above_crossing |
| --- | --- | --- | --- | --- |
| imcts | tpsr | 0.022727 | tpsr>imcts | imcts>tpsr |
| gplearn | imcts | 0.043344 | gplearn>imcts | imcts>gplearn |
| pyoperon | udsr | 0.062950 | pyoperon>udsr | udsr>pyoperon |
| drsr | ragsr | 0.082084 | ragsr>drsr | drsr>ragsr |
| ragsr | udsr | 0.105551 | ragsr>udsr | udsr>ragsr |
| dso | imcts | 0.107486 | dso>imcts | imcts>dso |
| drsr | udsr | 0.133953 | drsr>udsr | udsr>drsr |
| e2esr | udsr | 0.157960 | e2esr>udsr | udsr>e2esr |
| drsr | e2esr | 0.182887 | e2esr>drsr | drsr>e2esr |
| tpsr | udsr | 0.198220 | tpsr>udsr | udsr>tpsr |
| gplearn | udsr | 0.251529 | gplearn>udsr | udsr>gplearn |
| drsr | tpsr | 0.256709 | tpsr>drsr | drsr>tpsr |
| dso | udsr | 0.353978 | dso>udsr | udsr>dso |
| imcts | pysr | 0.374489 | pysr>imcts | imcts>pysr |
| drsr | gplearn | 0.410422 | gplearn>drsr | drsr>gplearn |
| pysr | udsr | 0.749460 | pysr>udsr | udsr>pysr |
| drsr | dso | 0.914284 | dso>drsr | drsr>dso |

## 7. Bootstrap uncertainty

The analysis uses `1000` synchronized
task-bootstrap replicates with seed `20260728`.
Each sampled task retains all five seeds. Percentile intervals are task-sampling
uncertainty intervals, not seed-resampling intervals.

| weight | algorithm | top1_probability | top3_probability | mean_rank | median_rank |
| --- | --- | --- | --- | --- | --- |
| 0.7000 | imcts | 0.6050 | 0.9990 | 1.4630 | 1.0000 |
| 0.8000 | imcts | 0.6190 | 0.9990 | 1.4460 | 1.0000 |
| 0.8500 | imcts | 0.6090 | 0.9990 | 1.4610 | 1.0000 |
| 0.8750 | imcts | 0.6030 | 0.9990 | 1.4670 | 1.0000 |
| 0.9000 | imcts | 0.6010 | 0.9990 | 1.4670 | 1.0000 |
| 0.9500 | imcts | 0.5940 | 0.9990 | 1.4740 | 1.0000 |
| 1.0000 | imcts | 0.5810 | 0.9990 | 1.4850 | 1.0000 |

Although iMCTS has the highest point estimate at every candidate weight, its
Top-1 probability ranges only from `0.581` to
`0.619`. Therefore the analysis does not support
describing the first-place algorithm as statistically decisive. By contrast,
the minimum Top-3 probability among the baseline Top-3 algorithms is
`0.976`, so Top-3 membership is much more robust
than internal Top-3 ordering.

Score intervals, rank probabilities, and all pairwise winning probabilities
are saved separately. The table below counts an algorithm pair as
directionally stable when the larger of `P(A>B)` and `P(A<B)` is at least
`0.95`. This is a pre-defined bootstrap evidence threshold, not a classical
hypothesis-test significance level.

| weight | all_pair_stable_count | all_pair_count | baseline_adjacent_pair_stable_count | baseline_adjacent_pair_count | baseline_top5_pair_stable_count | baseline_top5_pair_count | minimum_adjacent_directional_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7000 | 52 | 66 | 3 | 11 | 6 | 10 | 0.5470 |
| 0.8000 | 54 | 66 | 3 | 11 | 6 | 10 | 0.5060 |
| 0.8500 | 55 | 66 | 3 | 11 | 6 | 10 | 0.5440 |
| 0.8750 | 55 | 66 | 3 | 11 | 6 | 10 | 0.5390 |
| 0.9000 | 56 | 66 | 4 | 11 | 6 | 10 | 0.5300 |
| 0.9500 | 56 | 66 | 4 | 11 | 6 | 10 | 0.5040 |
| 1.0000 | 57 | 66 | 4 | 11 | 6 | 10 | 0.5440 |

## 8. Invalid-output and gated-retention diagnostics

There are `46` algorithm-task pairs with `q_ood=0` and
`r_ood=1`. Their original OOD-G contribution at `w=0.7` is:

| algorithm | case_count | total_OOD_G_point_contribution |
| --- | --- | --- |
| drsr | 2 | 1.2000 |
| dso | 1 | 0.6000 |
| e2esr | 12 | 7.2000 |
| gplearn | 0 | 0.0000 |
| imcts | 3 | 1.8000 |
| llmsr | 1 | 0.6000 |
| pyoperon | 6 | 3.6000 |
| pysr | 4 | 2.4000 |
| qlattice | 1 | 0.6000 |
| ragsr | 14 | 8.4000 |
| tpsr | 2 | 1.2000 |
| udsr | 0 | 0.0000 |

The diagnostic gate is exactly:

`r_gated = r_ood if q_id>0 and q_ood>0 else 0`.

| weight | spearman_original_vs_gated | kendall_original_vs_gated | top3_overlap_original_vs_gated | most_benefited_algorithm | max_algorithm_score_benefit |
| --- | --- | --- | --- | --- | --- |
| 0.7000 | 0.9860 | 0.9394 | 1.0000 | ragsr | 10.9074 |
| 0.8000 | 0.9790 | 0.9394 | 1.0000 | ragsr | 7.2716 |
| 0.8500 | 0.9930 | 0.9697 | 1.0000 | ragsr | 5.4537 |
| 0.8750 | 0.9930 | 0.9697 | 1.0000 | ragsr | 4.5447 |
| 0.9000 | 0.9930 | 0.9697 | 1.0000 | ragsr | 3.6358 |
| 0.9500 | 0.9930 | 0.9697 | 1.0000 | ragsr | 1.8179 |
| 1.0000 | 1.0000 | 1.0000 | 1.0000 | drsr | 0.0000 |

The ungated definition remains the paper's primary result.

## 9. Quality-reversal diagnostics

A reversal occurs when algorithm A has at least the stated OOD-NMSE advantage
over B on a task, but A receives a lower mixed task OOD-G component because of
retention. The summary below uses the 10x threshold; the CSV also reports
100x, 1000x, and 10000x thresholds.

| weight | reversal_count | max_ood_nmse_ratio | aggregate_pair_order_reversal_count | reversals_with_both_algorithms_in_current_top5 |
| --- | --- | --- | --- | --- |
| 0.7000 | 98 | 209944427590264224664873074688.0000 | 44 | 7 |
| 0.8000 | 52 | 209944427590264224664873074688.0000 | 24 | 3 |
| 0.8500 | 35 | 209944427590264224664873074688.0000 | 19 | 2 |
| 0.8750 | 24 | 209944427590264224664873074688.0000 | 15 | 2 |
| 0.9000 | 18 | 209944427590264224664873074688.0000 | 12 | 2 |
| 0.9500 | 14 | 209944427590264224664873074688.0000 | 9 | 2 |
| 1.0000 | 0 | 0.0000 | 0 | 0 |

## 10. Candidate-weight comparison

The pre-defined decision checks are:

1. minimum Kendall tau within `w +/- 0.05` is at least `0.90`;
2. Top-3 overlap throughout that neighborhood is `1.0`;
3. original-vs-gated Spearman correlation is at least `0.90`;
4. an interpretable quality-dominant candidate limits full retention
   compensation to at most 2.5 OOD-NMSE orders while retaining a non-zero
   retention term.

The 2.5-order bound is an explicit operational rule for this analysis, not an
empirically optimized constant. The selection never uses the identity of the
winning algorithm.

## 11. Recommended weight

**Decision:** `recommend_0.85`

0.85 is the closest pre-specified candidate that limits full retention compensation to at most 2.5 OOD-NMSE orders while passing local rank and gated-retention stability checks.

## 12. Limitations

1. The bootstrap quantifies sensitivity to the 50-task sample, not uncertainty
   over independently rerun seeds or hardware.
2. Only DRSR and LLM-SR use the existing model-split rerun snapshot; the other
   ten algorithms use the original clean archive.
3. The gated score is an anomaly diagnostic, not a retroactive replacement for
   the submitted definition.
4. Pairwise bootstrap probabilities are uncertainty summaries and should not
   be described as classical hypothesis-test p-values.

## Candidate-weight summary

| weight | quality_weight | retention_weight | equivalent_log_order_compensation | spearman_vs_w070 | kendall_vs_w070 | top3_overlap | number_of_pairwise_flips | number_of_crossings_within_plus_minus_0_05 | number_of_quality_reversals | bootstrap_top1_algorithm | bootstrap_top1_probability | original_vs_gated_rank_correlation | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7000 | 0.7000 | 0.3000 | 6.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 3 | 98 | imcts | 0.6050 | 0.9860 | not selected |
| 0.8000 | 0.8000 | 0.2000 | 3.5000 | 0.9650 | 0.8788 | 1.0000 | 4 | 1 | 52 | imcts | 0.6190 | 0.9790 | not selected |
| 0.8500 | 0.8500 | 0.1500 | 2.4706 | 0.9650 | 0.8788 | 1.0000 | 4 | 0 | 35 | imcts | 0.6090 | 0.9930 | recommend_0.85 |
| 0.8750 | 0.8750 | 0.1250 | 2.0000 | 0.9650 | 0.8788 | 1.0000 | 4 | 1 | 24 | imcts | 0.6030 | 0.9930 | not selected |
| 0.9000 | 0.9000 | 0.1000 | 1.5556 | 0.9650 | 0.8788 | 1.0000 | 4 | 1 | 18 | imcts | 0.6010 | 0.9930 | not selected |
| 0.9500 | 0.9500 | 0.0500 | 0.7368 | 0.9580 | 0.8485 | 1.0000 | 5 | 1 | 14 | imcts | 0.5940 | 0.9930 | not selected |
| 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9580 | 0.8485 | 1.0000 | 5 | 0 | 0 | imcts | 0.5810 | 1.0000 | not selected |

## Reproducibility

- Git commit before analysis: `0ad9e0b2dfdac72e14a95ea4eb023e144401116a`
- Python: `3.11.7 (main, Dec 15 2023, 18:12:31) [GCC 11.2.0]`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Package versions: `{"matplotlib": "3.8.0", "numpy": "1.26.4", "pandas": "2.1.4"}`
- Input files and SHA-256: see `reproducibility.json`

Commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python analysis/ood_weight_sensitivity/analyze_ood_weight_sensitivity.py \
  --bootstrap-reps 1000 \
  --bootstrap-seed 20260728

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout 60 python -m pytest -vv -s \
  analysis/ood_weight_sensitivity/test_analyze_ood_weight_sensitivity.py
```
