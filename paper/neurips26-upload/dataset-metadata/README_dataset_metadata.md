# Dataset and Benchmark Metadata

This directory contains metadata needed for the NeurIPS 2026 E&D artifact submission.

## Files

- `core50_ground_truth_manifest.csv`
  - Frozen Core-50 task manifest with source family, subgroup, target, features, and ground-truth formulas.

- `table18_core50_ground_truth_manifest.csv`
  - Paper table version of the Core-50 formula manifest.

- `core50_algorithm_hyperparameters_full.json`
  - Machine-readable hyperparameter manifest for the 12-algorithm Core-50 leaderboard.

- `tables/*.csv`
  - CSV versions of all paper tables used for audit and reproducibility.

- `croissant.TEMPLATE_NEEDS_URL.json`
  - Croissant metadata template.
  - This is not final until `TODO_REPLACE_*` fields are replaced with the anonymous dataset URL, final license, and hosting metadata.

## Final upload requirement

For E&D submission, the dataset or benchmark resource should be accessible to reviewers by URL, and the corresponding Croissant metadata file should validate. If hosted on Hugging Face, Kaggle, Dataverse, or OpenML, use the platform-generated Croissant file and then add the minimal Responsible-AI metadata required by NeurIPS E&D.
