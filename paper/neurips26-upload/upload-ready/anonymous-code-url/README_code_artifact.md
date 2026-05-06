# Code Artifact Notes

This directory contains a lightweight code snapshot for audit.

## Included

- `wrapper_code_snapshot/scientific_intelligent_modelling/`
  - Benchmark runner, pipeline, SR kit, config, and wrapper Python files.

- `wrapper_code_snapshot/check/`
  - Analysis and figure/table generation scripts used by the paper.

- `environment.yml`
  - Conda environment specification from the main repository.

- `toolbox_config.json`
  - Toolbox configuration snapshot.

## Excluded

The full repository contains large vendored upstream algorithm assets, model files, experiment outputs, and local machine paths. These should not be uploaded blindly as an OpenReview supplementary zip.

For final E&D submission, provide an anonymous full code URL with:

- complete runnable repository,
- installation instructions,
- Core-50 smoke test,
- leaderboard reproduction command,
- data download instructions,
- no API keys,
- no SSH configs,
- no private local paths,
- no git history exposing author identity if double-blind.

## Local zip

`../SymbolicArena_NeurIPS26_ED_code_snapshot_minimal.zip` is a compact snapshot for audit. It is not a replacement for the full anonymous code repository if reviewers are expected to reproduce all experiments.
