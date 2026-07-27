# Six-axis uncertainty for 15 algorithms

This directory adds FePySR, JAXSR, and SymbolFit to the frozen NeurIPS
12-algorithm Core-50 six-axis components.

## Protocol

- Original 12 methods: frozen NeurIPS task components, 1h, 5 clean seeds, and
  ROBU from noise sigma 0.01/0.05/0.10.
- Added 3 methods: AAAI three-hour trajectories read at `minute_0060`, 3 seeds,
  with ROBU from the available sigma 0.01/0.05 runs.
- SYM-F for the added methods is recomputed from the exact one-hour checkpoint
  expressions using the frozen formal judge.
- Marginal 95% intervals use 20000 synchronized task bootstrap
  resamples over the 50 shared tasks.
- Pairwise inference uses paired bootstrap intervals for score differences and
  20000 task-level sign-flip permutations, with Holm correction
  within each axis.

Marginal interval overlap is not used as the significance decision. The
pairwise difference table is the authoritative source for close comparisons.
This is a rebuttal supplement rather than a retrospective protocol change:
the original task components and formal judge remain frozen. EFF and STAB
retain the archived proxy-component definitions, and fine-grained ROBU
comparisons across the two cohorts should account for the different noise
grids stated above.

## Frozen-judge audit note

The frozen formal judge re-applies `feature_to_x_map` to run-level canonical
expressions that already use anonymous `x0`, `x1`, ... variables. For example,
on `g0001` (`Keijzer-11`) a raw `x0*x1` term is cleaned as `x0*x0`. This
supplement deliberately preserves that frozen behavior so the original 12
scores do not drift. A corrected judge must be followed by a complete
15-method SYM-F/STAB recomputation; corrected and frozen scores must not be
mixed in one figure.

## Scores

| display_name   |   ID_Q |   OOD_G |   SYM_F |   EFF |   ROB |   STAB |   six_axis_mean |
|:---------------|-------:|--------:|--------:|------:|------:|-------:|----------------:|
| iMCTS          |  58.07 |   63.99 |   41.50 | 46.75 | 38.49 |  48.26 |           49.51 |
| uDSR           |  63.76 |   59.86 |   36.83 | 41.05 | 41.14 |  45.86 |           48.08 |
| PySR           |  51.33 |   60.72 |   43.62 | 17.82 | 43.88 |  45.66 |           43.84 |
| DSO            |  37.03 |   48.04 |   31.26 | 24.18 | 48.30 |  41.95 |           38.46 |
| FePySR         |  41.38 |   48.14 |   32.32 | 13.43 | 41.35 |  41.59 |           36.37 |
| JAXSR          |  34.68 |   41.50 |   29.33 | 10.79 | 41.18 |  59.46 |           36.16 |
| DRSR           |  41.70 |   45.98 |   26.84 | 13.18 | 36.46 |  32.16 |           32.72 |
| LLM-SR         |  40.41 |   42.69 |   27.30 | 12.51 | 38.89 |  32.13 |           32.32 |
| QLattice       |  31.95 |   35.70 |   17.81 | 26.08 | 39.70 |  39.24 |           31.75 |
| gplearn        |  28.66 |   40.72 |   20.02 |  9.02 | 41.38 |  34.16 |           29.00 |
| PyOperon       |  23.28 |   33.55 |   11.30 | 19.68 | 32.61 |  29.09 |           24.92 |
| TPSR           |  17.14 |   34.03 |   15.24 | 11.51 | 27.64 |  32.74 |           23.05 |
| E2ESR          |  17.59 |   33.76 |   12.44 | 14.49 | 26.21 |  25.66 |           21.69 |
| SymbolFit      |   3.84 |   27.29 |   15.13 |  1.45 |  8.50 |  29.33 |           14.26 |
| RAG-SR         |   6.21 |   27.63 |   15.72 |  3.23 | 11.34 |  19.14 |           13.88 |

## Holm-significant pair counts

| axis   |   sum |   count |
|:-------|------:|--------:|
| EFF    |    56 |     105 |
| ID_Q   |    65 |     105 |
| OOD_G  |    51 |     105 |
| ROB    |    51 |     105 |
| STAB   |    56 |     105 |
| SYM_F  |    57 |     105 |

## Outputs

- `six_axis_uncertainty_15algs.png/.pdf/.jpg`
- `six_axis_task_components_15algs.csv`
- `six_axis_means_ci_15algs.csv`
- `six_axis_pairwise_inference.csv`
- `analysis_summary.json`
- `reviewer_p5tg_statistical_significance.md`
