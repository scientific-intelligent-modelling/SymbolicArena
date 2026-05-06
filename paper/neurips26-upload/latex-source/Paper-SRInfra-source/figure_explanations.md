# SymbolicArena Figure Explanations

note `imgs/` note note 

- `Files` note 
- `Question` note 
- `How to read` note 
- `Caption draft` note caption note 

## Figure 1: Pipeline

- Files: `imgs/figure01_pipeline.png`, `imgs/figure01_pipeline.pdf`
- Question: SymbolicArena note Core-50 note leaderboard?
- How to read: note note note ground-truth reservoir dual-probe scan Candidate-200 Probe-4 note Core-50 note 12 note leaderboard note 
- Caption draft: Overview of the SymbolicArena evaluation pipeline. A ground-truth reservoir is filtered and distilled through dual-probe screening, 12-method calibration, Probe-4 full-reservoir validation, and Core-50 selection before running the final multi-axis leaderboard.

## Figure 2: GT-Reservoir composition

- Files: `imgs/figure02_reservoir_composition.png`, `imgs/figure02_reservoir_composition.pdf`
- Question: GT-Reservoir-664 note note?
- How to read: note note GT-Reservoir-664 note benchmark family operator group formula complexity Probe4-derived difficulty note x note note note  reservoir note  note claim 
- Caption draft: Composition of the GT-Reservoir-664 across four dataset characteristics: benchmark family, operator group, formula-complexity bin, and Probe4-derived difficulty. Each category is explicitly labeled on the x-axis with dataset counts annotated above the bars.

## Figure 3: PySR vs LLM-SR gap scatter

- Files: `imgs/figure03_dual_probe_scatter.png`, `imgs/figure03_dual_probe_scatter.pdf`
- Question: Candidate-200 note note?
- How to read: note note PySR note LLM-SR note clipped log NMSE note note probe note note Candidate-200 
- Caption draft: Dual-probe behavior on the 664-task reservoir. Candidate-200 is enriched for datasets where PySR and LLM-SR exhibit informative disagreement under the same evaluation protocol.

## Figure 4: overall_gap_score distribution

- Files: `imgs/figure04_gap_score_distribution.png`, `imgs/figure04_gap_score_distribution.pdf`
- Question: Candidate-200 note?
- How to read: note selected note non-selected note gap score note Candidate-200 note note 
- Caption draft: Distribution of dual-probe gap scores for selected and non-selected datasets. Candidate-200 increases informativeness while retaining non-extreme calibration cases.

## Figure 5: Candidate-200 composition

- Files: `imgs/figure05_candidate200_composition_modes.png`, `imgs/figure05_candidate200_composition_modes.pdf`, `imgs/figure05_candidate200_family_distribution.png`, `imgs/figure05_candidate200_family_distribution.pdf`
- Question: Candidate-200 note note family note selection mode note?
- How to read: note high-discrimination mid-gap one-sided note selection mode note family note note Candidate-200 note 
- Caption draft: Candidate-200 composition by selection mode and benchmark family. The candidate pool combines high-discrimination, mid-gap, and one-sided cases under family-level coverage constraints.

## Figure 6: 12 algorithm health

- Files: `imgs/figure06_algorithm_health.png`, `imgs/figure06_algorithm_health.pdf`
- Question: 12 note note Probe-4 note?
- How to read: note Candidate-200 note valid rate finite metric rate note failure rate note leaderboard note probe 
- Caption draft: Health statistics of the 12 integrated symbolic-regression algorithms on Candidate-200. These diagnostics separate leaderboard coverage from probe suitability.

## Figure 7: Algorithm behavior correlation

- Files: `imgs/figure07_algorithm_behavior_correlation.png`, `imgs/figure07_algorithm_behavior_correlation.pdf`
- Question: Probe-4 note note?
- How to read: noteresponse note Probe-4 note 
- Caption draft: Behavioral correlation among the 12 algorithms based on dataset-level performance profiles. The selected Probe-4 covers complementary algorithmic responses rather than redundant solvers.

## Figure 8: Probe-4 selection score decomposition

- Files: `imgs/figure08_probe4_score_decomposition.png`, `imgs/figure08_probe4_score_decomposition.pdf`
- Question: note DSO PyOperon iMCTS uDSR note Probe-4?
- How to read: note top probe combinations note health panel fidelity complementarity coverage note note note trade-off note 
- Caption draft: Decomposition of Probe-4 selection scores across top candidate combinations. The selected panel balances health, full-panel fidelity, behavioral complementarity, and dataset coverage.

## Figure 9: Panel Fidelity scatter

- Files: `imgs/figure09_panel_fidelity_scatter.png`, `imgs/figure09_panel_fidelity_scatter.pdf`
- Question: Probe-4 note 12 note panel?
- How to read: note Candidate-200 note note 12 note panel note informativeness note Probe-4 note informativeness note Probe-4 note 
- Caption draft: Dataset informativeness estimated by the selected Probe-4 versus the full 12-method panel. High rank agreement supports using Probe-4 for full-reservoir distillation.

## Figure 10: Core-50 coverage map

- Files: `imgs/figure10_core50_coverage_map.png`, `imgs/figure10_core50_coverage_map.pdf`
- Question: Core-50 note GT-Reservoir-664 noteresponse note?
- How to read: note 664 note note Core-50 note Core-50 note note note 
- Caption draft: Coverage map of Core-50 within the 664-task reservoir using PCA over structural and Probe-4 response features. Core-50 covers multiple regions of the reservoir rather than a narrow cluster.

## Figure 11: Core-50 vs baselines quality radar

- Files: `imgs/figure11_core50_vs_baselines_radar.png`, `imgs/figure11_core50_vs_baselines_radar.pdf`
- Question: Core-50 note random metadata-only top-info note baseline note?
- How to read: note note benchmark note Core-50 note note coverage fidelity stability non-redundancy note 
- Caption draft: Benchmark-quality comparison between Core-50 and subset-selection baselines. Core-50 provides a balanced trade-off among coverage, fidelity, informativeness, stability, and non-redundancy.

## Figure 12: Rank fidelity comparison

- Files: `imgs/figure12_rank_fidelity_comparison.png`, `imgs/figure12_rank_fidelity_comparison.pdf`
- Question: Core-50 note full-reservoir note Probe-4 noteRanking?
- How to read: note Spearman Kendall pairwise win agreement note aggregate score error Core-50 note random note metadata-only baseline note Full-664 
- Caption draft: Rank-fidelity comparison of Core-50 against alternative 50-task subsets. Core-50 better preserves full-reservoir method ordering and aggregate scores.

## Figure 13: Distribution matching

- Files: `imgs/figure13_distribution_matching.png`, `imgs/figure13_distribution_matching.pdf`
- Question: Core-50 note family note failure mode note?
- How to read: note Full-664 Core-50 note baseline note note Core-50 note note 
- Caption draft: Distribution matching between Full-664, Core-50, and subset baselines across family, difficulty, and failure-mode labels.

## Figure 14: K-scaling curve

- Files: `imgs/figure14_k_scaling_curve.png`, `imgs/figure14_k_scaling_curve.pdf`
- Question: note Core note 50 note 20 30 100?
- How to read: note subset size K note coverage rank fidelity pairwise agreement note K=50 note 
- Caption draft: Subset-size scaling analysis. Core-50 is selected as a practical elbow point where fidelity and coverage gains begin to saturate relative to evaluation cost.

## Figure 15: 12 algorithms x 6 axes heatmap

- Files: `imgs/figure15_hexagon_heatmap_formal.png`, `imgs/figure15_hexagon_heatmap_formal.pdf`
- Question: 12 note?
- How to read: note note ID-Q OOD-G SYM-F EFF ROBU STAB note note leaderboard note 
- Caption draft: Six-axis Core-50 leaderboard over 12 symbolic-regression algorithms. Scores are fixed absolute 0--100 metrics and do not depend on the current set of compared algorithms.

## Figure 16: Top-4 radar

- Files: `imgs/figure16_radar_top4_formal.png`, `imgs/figure16_radar_top4_formal.pdf`
- Question: note?
- How to read: note note note note  note  
- Caption draft: Radar visualization of representative top algorithms under the six-axis Core-50 protocol, showing distinct capability profiles across numerical quality, generalization, fidelity, efficiency, robustness, and stability.

## Figure 17: ID-Q vs SYM-F

- Files: `imgs/figure17_idq_vs_symf.png`, `imgs/figure17_idq_vs_symf.pdf`
- Question: ID note?
- How to read: note ID-Q note SYM-F note ID note note SR benchmark note 
- Caption draft: Trade-off between in-distribution numerical quality and symbolic fidelity. Strong ID performance does not necessarily imply recovery of the ground-truth expression.

## Figure 18: OOD-G vs SYM-F scatter

- Files: `imgs/figure18_symf_vs_oodg.png`, `imgs/figure18_symf_vs_oodg.pdf`
- Question: note OOD note?
- How to read: note SYM-F note OOD-G note OOD note SYM note note 
- Caption draft: Relationship between symbolic fidelity and out-of-distribution generalization. The two axes capture related but non-identical aspects of symbolic-regression quality.

## Figure 19: Low-NMSE non-equivalent cases

- Files: `imgs/figure19_low_nmse_non_equivalent_cases.png`, `imgs/figure19_low_nmse_non_equivalent_cases.pdf`
- Question: note NMSE note?
- How to read: note run note  note NMSE note SR note  
- Caption draft: Examples of low-NMSE but non-equivalent formulas. These cases demonstrate why numerical accuracy alone is insufficient for symbolic-regression evaluation.

## Figure 20: Equivalence rate by NMSE threshold

- Files: `imgs/figure20_equivalence_rate_by_nmse_threshold.png`, `imgs/figure20_equivalence_rate_by_nmse_threshold.pdf`
- Question: note NMSE Thresholdnote note?
- How to read: note NMSE threshold note note  note  note  note  note 
- Caption draft: Symbolic equivalence rate as a function of NMSE threshold. Different algorithms exhibit different relationships between numerical accuracy and exact symbolic recovery.

## Figure 21: Anytime performance curve

- Files: `imgs/figure21_anytime_performance_curve.png`, `imgs/figure21_anytime_performance_curve.pdf`
- Question: note 1 note?
- How to read: note note best-so-far quality note note 
- Caption draft: Anytime best-so-far performance curves derived from minute-level snapshots. The curves reveal fast starters, slow improvers, and stagnating methods.
- Note: current note `generated_proxy` note clean LLM note snapshot note clean trace note 

## Figure 22: Time-to-threshold survival curve

- Files: `imgs/figure22_time_to_threshold_survival.png`, `imgs/figure22_time_to_threshold_survival.pdf`
- Question: noteThresholdnote?
- How to read: note note run noteThreshold note 
- Caption draft: Time-to-threshold survival analysis from minute-level best-so-far traces, quantifying how quickly each method reaches a target quality level.
- Note: current note `generated_proxy` note clean Core-50 trace note 

## Figure 23: Complexity over time

- Files: `imgs/figure23_complexity_over_time.png`, `imgs/figure23_complexity_over_time.pdf`
- Question: note?
- How to read: note note note note 
- Caption draft: Evolution of best-so-far expression complexity over time. This trace helps identify methods that trade expression complexity for numerical gains.
- Note: current note `generated_proxy` note minute snapshots note 

## Figure 24: Early vs final performance

- Files: `imgs/figure24_early_vs_final_performance.png`, `imgs/figure24_early_vs_final_performance.pdf`
- Question: note note?
- How to read: note note note note note 
- Caption draft: Early-versus-final performance comparison from best-so-far traces. The plot separates fast-and-strong methods from slow improvers and early stagnation.
- Note: current note `generated_proxy` notecurrent trace note 

## Figure 25: Noise robustness curve

- Files: `imgs/figure25_noise_robustness_curve.png`, `imgs/figure25_noise_robustness_curve.pdf`
- Question: note note clean-test note?
- How to read: note note quality note note 
- Caption draft: Noise-robustness curves under noisy-training and clean-test evaluation. Robust methods retain higher quality as label noise increases.

## Figure 26: ROBU heatmap

- Files: `imgs/figure26_noise_quality_heatmap.png`, `imgs/figure26_noise_quality_heatmap.pdf`
- Question: note 1% 5% 10% note?
- How to read: note note ROBU note note 12 note 
- Caption draft: Heatmap of noisy-training performance across algorithms and noise levels, summarizing the ROBU extension track.

## Figure 27: STAB component chart

- Files: `imgs/figure27_stab_component_chart.png`, `imgs/figure27_stab_component_chart.pdf`
- Question: STAB note seed note?
- How to read: note numerical stability valid rate structural consistency performance correction note note 
- Caption draft: Decomposition of effective stability into numerical consistency, valid-output rate, structural consistency, and performance correction.

## Figure 28: Score drift comparison

- Files: `imgs/figure28_score_drift_comparison.png`, `imgs/figure28_score_drift_comparison.pdf`
- Question: note rank-normalized note?
- How to read: note note leaderboard note 
- Caption draft: Score-drift comparison under simulated leaderboard composition changes. Fixed absolute metrics remain stable, whereas rank-normalized scores drift when the algorithm set changes.

## Figure 29: Axis correlation matrix

- Files: `imgs/figure29_axis_correlation_matrix.png`, `imgs/figure29_axis_correlation_matrix.pdf`
- Question: note note?
- How to read: note note note note 
- Caption draft: Correlation matrix of the six Core-50 evaluation axes. The axes capture related but distinct aspects of symbolic-regression performance.

## Figure 30: Bootstrap confidence interval

- Files: `imgs/figure30_bootstrap_ci.png`, `imgs/figure30_bootstrap_ci.pdf`
- Question: note Core-50 note?
- How to read: note note dataset bootstrap note noteRankingnote 
- Caption draft: Bootstrap confidence intervals for six-axis leaderboard scores obtained by resampling Core-50 datasets.

## Figure 31: Ablation summary

- Files: `imgs/figure31_ablation_summary.png`, `imgs/figure31_ablation_summary.pdf`
- Question: pipeline note?
- How to read: note baseline / ablation note fidelity coverage stability note note  note  
- Caption draft: Summary of selection and metric ablations. The full SymbolicArena design improves the balance between fidelity, coverage, stability, and non-redundancy over simpler alternatives.
- Note: current note subset baselines note full objective ablation note 

## Usage notes

- noteprefer Figure 1 3 8 10 12 14 15 17 21 25 30 
- Figure 21--24 current note `generated_proxy` note clean 12-algorithm minute snapshots note note 
- Figure 5 note note multi-panel figure 
- Figure 15 note Figure 16 note note heatmap radar note appendix 
- Figure 19 note case-study note note note appendix 

## Additional Figure: Hexagon small multiples

- Files: `imgs/fig_hexagon_small_multiples_neurips.png`, `imgs/fig_hexagon_small_multiples_neurips.pdf`
- Question: 12 note profile note note?
- How to read: note 3 note 4 note note panel note note HexaScore noteRanking note panel note note axis-key note note 9.5% note 12 note current panel note 20% note note `r=score/100` note note ID-Q OOD-G SYM-F EFF ROBU STAB 
- Caption draft: Small-multiple visualization of Core-50 six-axis profiles. The 12 panels are ordered by HexaScore and overlay all algorithms at low opacity, while the highlighted outline marks the current algorithm. Axis order is fixed clockwise from the top: ID-Q, OOD-G, SYM-F, EFF, ROBU, and STAB.
- Note: note note note appendix note Figure 16 note 
