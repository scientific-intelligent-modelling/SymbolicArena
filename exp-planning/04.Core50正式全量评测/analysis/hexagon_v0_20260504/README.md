# Core-50 hexagon v0 figures

This directory contains a first executable version of the Core-50 six-axis plotting pipeline.

## Metric status

- `ID-Q`: formal, computed from median seed clean ID NMSE and the fixed phi mapping.
- `OOD-G`: formal, computed from clean OOD quality plus ID-to-OOD retention.
- `SYM-F`: proxy, using parseability, variable/operator F1, operator-variable Jaccard, and ID/OOD ultra-low-NMSE numeric-equivalence proxy.
- `EFF`: proxy, because local clean Core50 results do not contain minute snapshots. The proxy combines final ID/OOD quality and median runtime.
- `ROB`: pending, requires noisy-training results.
- `STAB`: proxy, using numeric IQR, valid rate, and exact skeleton consistency; formal symbolic consistency can replace it later.

## Files

- `clean_final_runs.csv`: run-level clean metrics with failure-as-zero NMSE policy.
- `core50_result_expressions.csv`: extracted final expressions and parse metadata from result.json.
- `symbolic_metrics_proxy.csv`: seed-level symbolic proxy metrics.
- `dataset_axis_components.csv`: dataset x algorithm components.
- `hexagon_scores.csv`: algorithm-level scores.
- `hexagon_scores_with_ci.csv`: algorithm-level scores plus bootstrap CI over datasets.
- `fig_hexagon_heatmap.png`: 12 algorithms x 6 axes heatmap.
- `fig_radar_top4.png`: radar chart for top-4 by clean/proxy HexaScore.
- `fig_tradeoff_idq_oodg.png`, `fig_tradeoff_idq_symf.png`, `fig_tradeoff_eff_stab.png`: trade-off scatters.

## Current ranking by clean/proxy HexaScore

| algorithm   |   ID_Q |   OOD_G |   SYM_F |   EFF |   ROB |   STAB |   HexaScore_clean_proxy |
|:------------|-------:|--------:|--------:|------:|------:|-------:|------------------------:|
| imcts       |  58.07 |   63.99 |   72.20 | 46.75 |   nan |  48.26 |                   57.07 |
| udsr        |  63.76 |   59.86 |   35.20 | 41.05 |   nan |  45.86 |                   47.93 |
| pysr        |  51.33 |   60.72 |   41.55 | 17.82 |   nan |  45.66 |                   40.23 |
| qlattice    |  31.95 |   35.70 |   50.00 | 26.08 |   nan |  39.24 |                   35.74 |
| dso         |  37.03 |   48.04 |   18.47 | 24.18 |   nan |  41.95 |                   31.96 |
| drsr        |  40.49 |   43.83 |   11.20 | 12.63 |   nan |  33.59 |                   24.28 |
| llmsr       |  37.57 |   38.89 |    7.37 | 11.36 |   nan |  35.94 |                   21.31 |
| gplearn     |  28.66 |   40.72 |   12.15 |  9.02 |   nan |  34.16 |                   21.29 |
| pyoperon    |  23.28 |   33.55 |    5.38 | 19.68 |   nan |  29.09 |                   18.89 |
| e2esr       |  17.59 |   33.76 |    0.00 | 14.49 |   nan |  25.66 |                    1.86 |
| tpsr        |  17.14 |   34.03 |    0.00 | 11.51 |   nan |  32.74 |                    1.86 |
| ragsr       |   6.21 |   27.63 |    0.00 |  3.23 |   nan |  19.14 |                    1.01 |
