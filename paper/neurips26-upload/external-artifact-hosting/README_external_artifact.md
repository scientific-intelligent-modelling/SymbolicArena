# SymbolicArena External Artifact Hosting Package

This directory stages the files that should be uploaded to an anonymous external artifact host, not directly to OpenReview.

The OpenReview upload package should remain lightweight. This external package is for large raw experiment artifacts, especially the Core-50 clean runs and noise robustness runs.

## Recommended Upload Target

Use one anonymous external hosting location for this package:

- Hugging Face Dataset with anonymous authoring, if review anonymity can be preserved.
- OSF anonymous view-only project.
- Zenodo anonymous/private review link, if available for the venue workflow.

After upload, put the anonymous URL into:

- `paper/neurips26-upload/README_upload_package.md`
- `paper/neurips26-upload/dataset-metadata/croissant.TEMPLATE_NEEDS_URL.json`
- the final paper artifact/reproducibility section, if the venue allows URLs.

## What Goes Here

The full external artifact should include:

- `results_slim/`
  - Lightweight final result tables already included in the OpenReview package.
- `raw_results/clean_core50_12alg_5seed/`
  - Raw Core-50 clean 12-algorithm 5-seed final outputs.
- `raw_results/noise_robustness/`
  - Raw noise robustness outputs, excluding minute-level snapshots.
- `derived_analysis/core50_formal_analysis/`
  - Derived formal metrics, symbolic fidelity, hexagon scores, and ablation analysis.
- `dataset_metadata/`
  - Core-50 manifests, ground-truth formulas, table CSVs, and hyperparameter manifests.
- `code_snapshot/`
  - Minimal wrapper and benchmark code snapshot used to interpret artifacts.

## What Must Not Go Here

Do not upload:

- API keys or real LLM provider config files.
- Conda environments or model cache directories.
- SSH keys, host preflight dumps, tmux process state, and scheduler-only logs.
- Git directories or Git LFS internals.
- Non-anonymized local paths, usernames, private IPs, or internal hostnames.

## Build Commands

From the repository root:

```bash
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py inventory
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py prepare
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py package
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py checksum
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py scan
```

Or run the full pipeline:

```bash
python paper/neurips26-upload/external-artifact-hosting/scripts/build_external_artifact.py all
```

The generated large files are written under:

```text
paper/neurips26-upload/external-artifact-hosting/dist/
```

`dist/` and `staging/` are intentionally git-ignored.

## Expected Output

The main upload file should be:

```text
dist/SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
```

Upload this file plus:

```text
dist/SHA256SUMS.txt
dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
```

Keep `README_external_artifact.md` visible on the hosting page.

The uncompressed file manifest is intentionally generated in `dist/` for local inspection, but the compressed `.zst` copy is the preferred upload file.

## Size Expectation

The source raw noise artifact is approximately 18GB before compression. The external upload artifact excludes minute-level JSON snapshots by default and keeps final results, manifests, and derived analysis.
