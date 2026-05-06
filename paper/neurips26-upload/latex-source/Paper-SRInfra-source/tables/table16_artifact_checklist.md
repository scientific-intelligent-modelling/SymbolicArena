# Paper artifact and reproducibility table.

| Component                   | Format                 | Path                                                                                                                 | Role                              |
|:----------------------------|:-----------------------|:---------------------------------------------------------------------------------------------------------------------|:----------------------------------|
| GT-Reservoir-664 metadata   | CSV + dataset metadata | exp-planning/01.dual_probe_experiment/datasets_runnable.csv                                                                     | Reservoir definition              |
| Candidate-200 selection log | CSV                    | experiment-results/benchmark_selection_dossier_20260422/tables/stage1_candidate200_flat.csv                          | Dual-probe candidate construction |
| E1 12-algorithm calibration | CSV                    | exp-planning/02.e1_selection_validation/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv           | Probe-4 selection                 |
| Probe4-Full run-level table | CSV                    | exp-planning/03.probe4_full664_3seed_validation/generated/postprocess_final_20260501-105508/probe4_postprocess_run_level.csv | Core-50 distillation              |
| Core-50 dataset manifest    | CSV                    | exp-planning/04.core50_formal_full_evaluation/core50_datasets.csv                                                               | Frozen task list                  |
| Clean Core-50 final runs    | CSV                    | exp-planning/04.core50_formal_full_evaluation/analysis/hexagon_v1_with_artifacts_20260504/clean_final_runs_updated.csv          | Leaderboard scoring               |
| Formal symbolic metrics     | CSV                    | exp-planning/04.core50_formal_full_evaluation/analysis/symf_formal_metrics_20260504/symbolic_metrics_formal.csv                 | SYM-F axis                        |
| Noise robustness runs       | CSV                    | exp-planning/04.core50_formal_full_evaluation/analysis/hexagon_v1_with_artifacts_20260504/noise_final_runs.csv                  | ROBU axis                          |
| Final six-axis scores       | CSV                    | exp-planning/04.core50_formal_full_evaluation/analysis/hexagon_v2_formal_20260505/hexagon_scores_formal_with_ci.csv             | Main leaderboard table            |
| Paper figures               | PNG/PDF                | paper/Paper-SRInfra/imgs                                                                                             | Publication figures               |
