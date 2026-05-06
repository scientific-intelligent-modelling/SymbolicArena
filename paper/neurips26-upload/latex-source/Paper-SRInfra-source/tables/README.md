# Paper Table Bank

This directory contains paper-ready tables collected from the current SR-Infra experimental artifacts.
Each table is exported as CSV, Markdown, and LaTeX.

| table | rows | purpose |
| --- | ---: | --- |
| `table01_dataset_reservoir_stats` | 8 | Dataset statistics for GT-Reservoir-664 by family. |
| `table02_candidate200_family_summary` | 7 | Candidate-200 family composition and dual-probe gap summary. |
| `table03_candidate200_selection_modes` | 4 | Candidate-200 selection modes and gap statistics. |
| `table04_probe4_algorithm_health` | 12 | Candidate-200 health-gate audit for Probe-4 selection. |
| `table05_probe4_top_combinations` | 8 | Probe-4 candidate-panel audit with health-gated alternatives and relaxed-policy stress checks. |
| `table06_probe4_full_completion` | 4 | Probe4-Full completion and valid-output rates on GT-Reservoir-664. |
| `table07_probe4_dataset_labels` | 16 | Probe4-Full dataset labels, eligibility classes, and run outcomes. |
| `table08_core50_baseline_quality` | 6 | Core-50 representativeness compared with subset-selection baselines. |
| `table09_k_scaling_metrics` | 14 | Subset-size scaling metrics for Core-K selection. |
| `table10_clean_core50_leaderboard` | 12 | Clean Core-50 leaderboard with execution, ID/OOD numerical quality, and symbolic-fidelity diagnostics. |
| `table11_hexagon_scores_formal` | 12 | Formal Core-50 six-axis scores with dataset-bootstrap 95% confidence intervals. |
| `table12_symbolic_fidelity_summary` | 12 | Formal symbolic-fidelity metrics by algorithm. |
| `table13_noise_robustness_summary` | 12 | Noise robustness summary by algorithm. |
| `table14_noise_by_sigma` | 36 | Noise-track completion and median clean-test NMSE by noise level. |
| `table15_low_nmse_non_equiv_cases` | 8 | Low-NMSE but non-equivalent symbolic-regression cases. |
| `table16_artifact_checklist` | 10 | Paper artifact and reproducibility table. |
| `table17_core50_algorithm_hyperparameters` | 12 | Core-50 12-method hyperparameter and execution-contract summary. |
| `Appendix/table18_core50_ground_truth_manifest` | 50 | Frozen Core-50 source family, variable names, and ground-truth formulas. |
| `table19_core50_ablation_summary` | 7 | Core-50 ablation against deterministic and random 50-task selectors. |

Recommended main-paper tables: `table01`, `table05`, `table08`, `table10`, `table11`, `table12`, and `table16`.
Recommended appendix tables: all remaining tables, especially detailed noise-by-sigma, low-NMSE non-equivalent cases, baseline hyperparameters, and the Core-50 formula manifest.
