# Core-50 SYM-F sensitivity analysis

## Scope

- Input: `A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h/formal_analysis/symbolic_metrics_formal.csv`
- Input SHA256: `d1e4f2ab22d555ea2871e39ed6d1fc49501438b3d2851a86ebb247bb11f8db6d`
- Grid: `3000` runs =
  `12 algorithms x 50 datasets x 5 seeds`.
- Baseline non-exact score:
  `0.3 * TreeSim + 0.1 * variable-F1 + 0.1 * operator-F1`.
- Exact formulas remain `1`; invalid or unparsable runs remain `0` in every
  configuration.
- This analysis tests the **aggregation weights and non-exact cap**. It does
  not test the CAS/numerical equivalence detector or its `1e-10` threshold.
- Correlations are descriptive dependence diagnostics, not proofs of
  statistical or causal independence.

Reproduce from the repository root:

```bash
python check/analyze_symf_weight_sensitivity.py
```

## Main findings

1. **Local perturbations are fully stable.** Across all
   `25` combinations formed by independently changing
   the `0.3` TreeSim and `0.2` SOF1 weights by up to `+/-20%`, all 12 algorithm
   ranks and all 66 pairwise orderings are unchanged. The minimum score
   Pearson correlation is `0.999548` and the minimum
   rank Spearman correlation is `1.000000`.

2. **The 0.5 non-exact cap is not a ranking breakpoint.** With the original
   60/40 TreeSim/SOF1 mixture, the complete ranking is unchanged for all
   tested caps in `[0.01, 0.83]` at
   `0.01` resolution. The Top-3 set and
   winner remain unchanged throughout the full diagnostic cap sweep
   `[0, 1]`. In particular, every tested cap from `0.25` through `0.75`
   preserves the full ranking.

3. **Broad stress tests preserve the headline conclusion.** Across
   `441` combinations spanning caps `[0.25, 0.75]` and
   TreeSim shares `[0, 1]`, the minimum rank Spearman is
   `0.881`, minimum pairwise agreement is
   `0.833`, maximum rank movement is
   `3`, and the Top-3 set is unchanged in
   `100.0%` of configurations.
   These extreme endpoints deliberately include dropping TreeSim or SOF1
   completely and should be treated as stress tests, not equally plausible
   defaults.

4. **The components are not empirically redundant at run level.** Among
   `2491` non-exact valid runs, TreeSim and
   SOF1 have Pearson correlation
   `-0.010` and Spearman correlation
   `-0.086`. Variable-F1 and operator-F1 have
   Pearson correlation `0.028`. Regressing
   TreeSim on both F1 components explains only
   `10.1%` of its variance.

5. **TreeSim is complementary but sparse.** It is zero on
   `86.2%` of non-exact
   valid runs. Consequently, despite receiving 60% of the nominal partial
   credit, it contributes only
   `1.036` of the
   average `24.990` SYM-F
   points, versus
   `11.754` points
   from variable/operator F1. This is a limitation to disclose: the
   components are not double-counted, but their empirical activation rates
   differ substantially.

## Baseline decomposition

| Rank | Algorithm | SYM-F | Exact pts | Tree pts | Variable pts | Operator pts |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | PySR | 43.617 | 34.000 | 1.875 | 3.574 | 4.168 |
| 2 | iMCTS | 41.503 | 30.800 | 1.634 | 4.050 | 5.020 |
| 3 | uDSR | 36.834 | 24.000 | 1.026 | 6.129 | 5.679 |
| 4 | DSO | 31.255 | 18.000 | 1.816 | 6.177 | 5.262 |
| 5 | LLM-SR | 27.295 | 12.800 | 1.239 | 6.930 | 6.327 |
| 6 | DRSR | 26.838 | 12.000 | 1.242 | 6.983 | 6.613 |
| 7 | gplearn | 20.016 | 8.000 | 0.992 | 6.352 | 4.672 |
| 8 | QLattice | 17.815 | 4.800 | 0.763 | 6.230 | 6.022 |
| 9 | RAG-SR | 15.720 | 1.600 | 0.000 | 8.606 | 5.514 |
| 10 | TPSR | 15.240 | 0.400 | 1.064 | 7.282 | 6.495 |
| 11 | E2ESR | 12.440 | 0.000 | 0.024 | 6.902 | 5.513 |
| 12 | PyOperon | 11.305 | 0.000 | 0.759 | 4.937 | 5.608 |

## Suggested rebuttal text

> We added a run-level sensitivity analysis over all 3,000 Core-50 symbolic
> evaluations. Independently perturbing the TreeSim and SOF1 weights by
> +/-20% leaves all 12 ranks and all 66 pairwise method orderings unchanged
> (minimum score Pearson r=0.9995; rank
> Spearman rho=1.000). Holding the original
> component ratio fixed, every non-exact cap from 0.25 to 0.75 also preserves
> the full ranking. A broader 441-setting stress grid, including the extreme
> removal of either TreeSim or SOF1, retains the same Top-3 set throughout
> (minimum rho=0.881). Finally, TreeSim and SOF1
> are nearly uncorrelated across non-exact valid runs
> (Pearson r=-0.010), arguing against obvious
> duplicate credit. We will report these results and clarify that TreeSim is
> complementary but sparse, so the F1 terms supply most observed partial
> credit.

## Files

- `algorithm_baseline_contributions.csv`
- `local_weight_sensitivity.csv`
- `broad_weight_stress_grid.csv`
- `nonexact_cap_sweep.csv`
- `tree_share_sweep.csv`
- `ablation_summary.csv`
- `ablation_algorithm_scores.csv`
- `component_correlations.csv`
- `component_redundancy_regressions.csv`
- `sensitivity_summary.json`
- `symf_weight_sensitivity.png/.pdf`
