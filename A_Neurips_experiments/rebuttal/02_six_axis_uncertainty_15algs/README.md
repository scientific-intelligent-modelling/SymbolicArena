# Six-axis uncertainty for 15 algorithms

This directory reports task-bootstrap uncertainty and direct paired
algorithm comparisons for the 15-method Core-50 rebuttal evaluation.

## Protocol

- All scores use one-hour evaluations over the same 50 tasks.
- The primary analysis uses five runs for the original 12 methods and three
  runs for the three rebuttal-added methods.
- Marginal 95% intervals use 20,000 synchronized task-bootstrap
  resamples. They quantify task-sampling uncertainty and do not resample seeds.
- Pairwise inference uses the same resampled task set for both methods,
  20,000 task-level sign-flip permutations, and Holm correction
  across all 105 method pairs within each axis.
- A pair is called statistically distinguishable only when both the paired
  95% bootstrap interval excludes zero and the Holm-adjusted p-value is below
  0.05.
- The matched-three-run sensitivity uses three runs for every method. ROBU
  uses the two common noise levels. STAB is excluded from this sensitivity
  because a matched seed-level structural proxy is not included in the
  sensitivity artifact.

Overlapping marginal intervals are diagnostics only. The paired-difference
tables and matrix are the authoritative outputs for algorithm comparisons.

## Primary scores

| display_name   |   ID_Q |   OOD_G |   SYM_F |   EFF |   ROB |   STAB |   axis_mean |
|:---------------|-------:|--------:|--------:|------:|------:|-------:|------------:|
| iMCTS          |  58.07 |   63.99 |   41.50 | 46.75 | 38.49 |  42.79 |       48.60 |
| uDSR           |  63.76 |   59.86 |   36.83 | 41.05 | 41.14 |  46.74 |       48.23 |
| PySR           |  51.33 |   60.72 |   43.62 | 17.82 | 43.88 |  47.05 |       44.07 |
| DSO            |  37.03 |   48.04 |   31.26 | 24.18 | 48.30 |  44.78 |       38.93 |
| FePySR         |  41.38 |   48.14 |   32.32 | 13.43 | 41.35 |  41.59 |       36.37 |
| JAXSR          |  34.68 |   41.50 |   29.33 | 10.79 | 41.18 |  59.46 |       36.16 |
| DRSR           |  41.70 |   45.98 |   26.84 | 13.18 | 36.46 |  35.15 |       33.22 |
| LLM-SR         |  40.41 |   42.69 |   27.30 | 12.51 | 38.89 |  34.88 |       32.78 |
| QLattice       |  31.95 |   35.70 |   17.81 | 26.08 | 39.70 |  33.81 |       30.84 |
| gplearn        |  28.66 |   40.72 |   20.02 |  9.02 | 41.38 |  35.40 |       29.20 |
| PyOperon       |  23.28 |   33.55 |   11.30 | 19.68 | 32.61 |  30.18 |       25.10 |
| TPSR           |  17.14 |   34.03 |   15.24 | 11.51 | 27.64 |  37.02 |       23.76 |
| E2ESR          |  17.59 |   33.76 |   12.44 | 14.49 | 26.21 |  29.00 |       22.25 |
| RAG-SR         |   6.21 |   27.63 |   15.72 |  3.23 | 11.34 |  24.46 |       14.77 |
| SymbolFit      |   3.84 |   27.29 |   15.13 |  1.45 |  8.50 |  29.33 |       14.26 |

## Distinguishable pair counts

| axis   |   sum |   count |
|:-------|------:|--------:|
| EFF    |    56 |     105 |
| ID_Q   |    65 |     105 |
| OOD_G  |    51 |     105 |
| ROB    |    51 |     105 |
| STAB   |    48 |     105 |
| SYM_F  |    57 |     105 |

## Adjacent-ranking result

| axis   |   sum |   count |
|:-------|------:|--------:|
| EFF    |     1 |      14 |
| ID_Q   |     1 |      14 |
| OOD_G  |     1 |      14 |
| ROB    |     1 |      14 |
| STAB   |     0 |      14 |
| SYM_F  |     0 |      14 |

Only 0 of the 24 adjacent comparisons across the six
axis-specific Top-5 rankings are statistically distinguishable.

## Matched-three-run sensitivity

| axis   |   rank_correlation |   mean_absolute_score_change |   max_absolute_score_change | top1_primary   | top1_matched3   | top1_unchanged   |   top3_overlap |
|:-------|-------------------:|-----------------------------:|----------------------------:|:---------------|:----------------|:-----------------|---------------:|
| ID_Q   |              0.993 |                         0.63 |                        3.6  | udsr           | udsr            | True             |              3 |
| OOD_G  |              0.979 |                         0.64 |                        4.05 | imcts          | imcts           | True             |              3 |
| SYM_F  |              0.986 |                         0.52 |                        1.7  | pysr           | pysr            | True             |              3 |
| EFF    |              0.996 |                         0.3  |                        1.38 | imcts          | imcts           | True             |              3 |
| ROB    |              0.975 |                         0.69 |                        2.82 | dso            | dso             | True             |              2 |

## Public outputs

- `six_axis_uncertainty_15algs.png/.pdf/.jpg`
- `pairwise_significance_top5.png/.pdf/.jpg`
- `matched3_sensitivity.png/.pdf/.jpg`
- `six_axis_task_components_15algs.csv`
- `matched3_task_components_15algs.csv`
- `six_axis_means_ci_15algs.csv`
- `six_axis_pairwise_inference.csv`
- `top5_pairwise_inference.csv`
- `adjacent_pairwise_inference.csv`
- `matched3_means_ci_15algs.csv`
- `matched3_sensitivity_detail.csv`
- `matched3_sensitivity_summary.csv`
- `reviewer_p5tg_statistical_significance.md`
- `analysis_summary.json`
