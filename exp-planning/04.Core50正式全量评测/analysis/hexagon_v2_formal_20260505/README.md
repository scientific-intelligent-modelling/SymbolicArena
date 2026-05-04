# Core-50 hexagon v2 formal figures

- Created at: `2026-05-05T00:41:47`
- `SYM-F` uses formal judge: CAS equivalence, independent probe numeric equivalence, variable/operator F1, and fast tree similarity.
- `ID-Q`, `OOD-G`, `EFF`, `ROB`, `STAB` inherit the v1 clean/noise artifacts.
- `EFF` and `STAB` are still proxy axes until full clean minute-level AUC and formal seed-level structural consistency are wired.

## Score Table

| algorithm   |   ID_Q |   OOD_G |   SYM_F |   EFF |   ROB |   STAB |   HexaScore_formal_with_ROB |   SYM_F_proxy |
|:------------|-------:|--------:|--------:|------:|------:|-------:|----------------------------:|--------------:|
| imcts       |  58.07 |   63.99 |   41.50 | 46.75 | 38.49 |  48.26 |                       48.73 |         72.20 |
| udsr        |  63.76 |   59.86 |   36.83 | 41.05 | 41.14 |  45.86 |                       47.08 |         35.20 |
| pysr        |  51.33 |   60.72 |   43.62 | 17.82 | 43.88 |  45.66 |                       41.15 |         41.55 |
| dso         |  37.03 |   48.04 |   31.26 | 24.18 | 48.30 |  41.95 |                       37.37 |         18.47 |
| qlattice    |  31.95 |   35.70 |   17.81 | 26.08 | 39.70 |  39.24 |                       30.63 |         50.00 |
| drsr        |  41.70 |   45.98 |   26.84 | 13.18 | 36.46 |  32.16 |                       30.44 |         11.20 |
| llmsr       |  40.41 |   42.69 |   27.30 | 12.51 | 38.89 |  32.13 |                       30.05 |         12.00 |
| gplearn     |  28.66 |   40.72 |   20.02 |  9.02 | 41.38 |  34.16 |                       25.85 |         12.15 |
| pyoperon    |  23.28 |   33.55 |   11.30 | 19.68 | 32.61 |  29.09 |                       23.42 |          5.38 |
| tpsr        |  17.14 |   34.03 |   15.24 | 11.51 | 27.64 |  32.74 |                       21.27 |          0.00 |
| e2esr       |  17.59 |   33.76 |   12.44 | 14.49 | 26.21 |  25.66 |                       20.40 |          0.00 |
| ragsr       |   6.21 |   27.63 |   15.72 |  3.23 | 11.34 |  19.14 |                       11.12 |          0.00 |

## Figures

- `fig_hexagon_heatmap_formal.png`
- `fig_radar_top4_formal.png`
- `fig_hexascore_formal_bar.png`
- `fig_axis_bars_formal.png`
- `fig_tradeoff_idq_oodg.png`
- `fig_tradeoff_idq_symf_formal.png`
- `fig_tradeoff_oodg_rob.png`
- `fig_tradeoff_symf_eff.png`
- `fig_tradeoff_eff_stab.png`
- `fig_symf_proxy_vs_formal.png`
- `fig_symbolic_components_heatmap.png`
- `fig_symbolic_equiv_rates_bar.png`
- `fig_noise_rob_by_sigma.png`
- `fig_noise_quality_by_sigma_heatmap.png`
- `fig_noise_valid_rate_by_sigma_heatmap.png`
- `fig_valid_parse_rates.png`