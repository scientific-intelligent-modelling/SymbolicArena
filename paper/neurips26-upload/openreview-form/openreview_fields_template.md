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

https://osf.io/5hdqa/overview?view_only=2a5790c6516f4c8f88fd74474c1d3978

Recommended options:

- Hugging Face dataset repository with anonymous organization/account
- Kaggle dataset with link sharing enabled
- Dataverse private preview URL
- OpenML dataset
- Anonymous self-hosted URL with validated Croissant metadata

## Code URL

https://anonymous.4open.science/r/SymbolicArenaCode-A3C0/

Recommended:

- anonymous GitHub mirror via anonymous.4open.science or equivalent
- no author names in README, commit history, paths, or package metadata
- no API keys or SSH configs

## Dataset license

CC BY 4.0

Use a clear SPDX-compatible license URL when possible.

## Artifact statement

This submission introduces SymbolicArena, a symbolic-regression evaluation and benchmark infrastructure built around a standardized GT-Reservoir-664 task pool, a frozen Core-50 benchmark subset, a 12-algorithm leaderboard protocol, and machine-readable manifests for formulas, splits, hyperparameters, and results.

## Data access statement

The benchmark data are accessible to reviewers at the Dataset URL above. The hosted artifact includes the full standardized dataset payload, Core-50 task manifests, standardized train/validation/ID/OOD splits, formula metadata, source-family metadata, checksums, and documentation of intended use and limitations.

## Code access statement

The code artifact is accessible to reviewers at the Code URL above. It includes the benchmark runner, wrapper contracts for the evaluated algorithms, result schema, postprocessing scripts, and smoke-test instructions.

## External raw results URL

https://osf.io/qvs5g/overview?view_only=496e544380f04bf5a97fb4ab9ea2b95f
