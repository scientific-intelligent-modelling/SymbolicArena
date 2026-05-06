# SymbolicArena NeurIPS 2026 E&D Results Artifact

This directory contains lightweight final-result tables used to reproduce and audit the paper claims.

It intentionally excludes heavyweight raw experiment directories, remote scheduler state, tmux/SSH logs, host preflight files, and full minute-level progress snapshots.

Machine-local paths, internal hostnames, private IPs, and LLM configuration file locations have been anonymized in this upload copy. The anonymization does not change dataset IDs, method names, scores, expressions, or aggregate metrics.

## Included

### `clean-core50/`

Final clean Core-50 12-algorithm results:

- `core50_12alg_run_level_log_nmse_20260503.csv`
  - One row per algorithm × dataset × seed final run where available.
  - Includes log-NMSE summaries used for clean leaderboard auditing.

- `core50_12alg_dataset_algorithm_median_log_nmse_20260503.csv`
  - Dataset × algorithm aggregation over seeds.

- `core50_12alg_leaderboard_ood_log_nmse_20260503.csv`
  - OOD-log-NMSE clean leaderboard summary.

- `clean_final_runs_updated.csv`
  - Updated final clean run table used by the hexagon scoring pipeline.

- `clean_final_runs_scored.csv`
  - Clean run table after score component derivation.

- `core50_result_expressions_updated.csv`
  - Final expression table used for formula replay and symbolic analysis.

- `llmsr_drsr_iteration_summary*.{json,csv}`
  - Iteration budget summaries for LLM-SR and DRSR.

### `symbolic-formal/`

Formal symbolic-fidelity metrics:

- `symbolic_metrics_formal.csv`
  - Run-level symbolic-fidelity judgments and features.

- `symbolic_metrics_formal_algorithm_summary.csv`
  - Algorithm-level symbolic-fidelity summary.

- `symbolic_metrics_formal_dataset_summary.csv`
  - Dataset-level symbolic-fidelity summary.

- `symbolic_metrics_formal_summary.json`
  - Machine-readable summary.

### `hexagon/`

Final hexagonal leaderboard components:

- `hexagon_scores_formal.csv`
- `hexagon_scores_formal_with_ci.csv`
- `dataset_axis_components_formal.csv`
- `metric_update_summary.json`

### `noise-robustness/`

Noisy-train / clean-test robustness summaries:

- `noise_final_runs.csv`
- `noise_completion_summary.csv`
- `robustness_components.csv`

The raw noise experiment artifact directory is approximately 18GB and is not included in this upload package. It should be released through the anonymous dataset/artifact hosting URL if needed.

### `ablation/`

Core-50 validation and ablation tables:

- `core50_ablation_metrics.csv`
- `core50_ablation_subset_membership.csv`
- `core50_ablation_family_distribution.csv`
- `core50_baseline_quality_metrics.csv`
- `core50_baseline_subset_membership.csv`
- `core50_random_baseline_draw_membership.csv`
- `figure14_k_scaling_metrics.csv`
- `figure19_low_nmse_non_equivalent_cases.csv`
- `figure20_equivalence_rate_by_nmse_threshold.csv`
- `minute_snapshot_sample_for_figures.csv`

### `selection-validation/`

Probe-4 and Core-50 validation checks:

- `core50_probe4_previous_vs_current_dataset_algorithm_20260503.csv`
- `core50_probe4_previous_vs_current_pairwise_20260503.csv`
- `core50_probe4_previous_vs_current_ranking_20260503.csv`
- `core50_vs_candidate200_iteration_comparison_20260503.csv`

### `manifests/`

Core-50 dataset, formula, hyperparameter, and paper-table manifests used to connect result rows to benchmark tasks and paper tables.

## Excluded by design

- Full raw experiment directories under `experiments/`.
- Remote scheduling state and host preflight logs.
- SSH/tmux process logs.
- Full minute-level progress snapshots.
- Full dataset split files; these should be served via the anonymous dataset URL.
- API keys, LLM provider config files, and machine-local paths.

## Integrity

The parent upload package contains `SHA256SUMS.txt` and `FILE_MANIFEST.txt` for the entire upload directory.
