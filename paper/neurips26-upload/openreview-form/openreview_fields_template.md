# OpenReview Field Template

## Track

NeurIPS 2026 Evaluations & Datasets Track

## Review mode

Default target: double-blind.

The manuscript uses:

```tex
\usepackage[eandd]{neurips_2026}
```

If the dataset or benchmark cannot be anonymized, select single-blind in the OpenReview form and document the reason.

## Dataset URL

TODO_REPLACE_WITH_ANONYMOUS_DATASET_URL

Recommended options:

- Hugging Face dataset repository with anonymous organization/account
- Kaggle dataset with link sharing enabled
- Dataverse private preview URL
- OpenML dataset
- Anonymous self-hosted URL with validated Croissant metadata

## Code URL

TODO_REPLACE_WITH_ANONYMOUS_CODE_URL

Recommended:

- anonymous GitHub mirror via anonymous.4open.science or equivalent
- no author names in README, commit history, paths, or package metadata
- no API keys or SSH configs

## Dataset license

TODO_REPLACE_WITH_LICENSE

Use a clear SPDX-compatible license URL when possible.

## Artifact statement

This submission introduces SymbolicArena, a symbolic-regression evaluation and benchmark infrastructure built around a standardized GT-Reservoir-664 task pool, a frozen Core-50 benchmark subset, a 12-algorithm leaderboard protocol, and machine-readable manifests for formulas, splits, hyperparameters, and results.

## Data access statement

TODO_REPLACE_AFTER_HOSTING

The benchmark data are accessible to reviewers at the dataset URL above. The hosted artifact includes Core-50 task manifests, standardized train/validation/ID/OOD splits, formula metadata, source-family metadata, and documentation of intended use and limitations.

## Code access statement

TODO_REPLACE_AFTER_HOSTING

The code artifact includes the benchmark runner, wrapper contracts for the evaluated algorithms, result schema, postprocessing scripts, and smoke-test instructions.
