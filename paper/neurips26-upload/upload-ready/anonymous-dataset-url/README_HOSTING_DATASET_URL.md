# Metadata-Only Helper Files

This directory contains metadata helper files only. It is not the full dataset payload.

For the OpenReview `Dataset URL`, upload the full standardized dataset archive instead:

```text
../SymbolicArena_NeurIPS26_ED_full_dataset_payload.tar.zst
../SymbolicArena_NeurIPS26_ED_full_dataset_payload.tar.zst.sha256
```

The full archive expands to:

```text
dataset-url-full-standardized/
```

It contains the real standardized dataset files for 793 dataset directories.

Recommended hosts:

- Anonymous Hugging Face Dataset repository.
- OSF anonymous view-only project.
- Dataverse private preview URL.
- Another anonymous static host accepted by the venue.

## Helper Files

```text
README_dataset_metadata.md
core50_ground_truth_manifest.csv
core50_algorithm_hyperparameters_full.json
croissant.TEMPLATE_NEEDS_URL.json
```

These files are already included inside the full dataset archive under `metadata/`.

## After Upload

1. Replace `TODO_REPLACE_WITH_ANONYMOUS_DATASET_URL` in the Croissant file.
2. Replace `TODO_REPLACE_WITH_LICENSE_URL`.
3. Replace citation and maintenance-plan placeholders.
4. Validate the Croissant file.
5. Upload the validated Croissant file to the OpenReview `Croissant File` field.
6. Paste the anonymous host URL into the OpenReview `Dataset URL` field.

## Anonymity Requirements

- Do not use author names, institutional names, personal emails, internal hostnames, local paths, or private IPs.
- Do not expose Git history, upload account identity, access logs, SSH configs, API keys, or machine names.
- Keep reviewer access enabled during the full review period.
