# SymbolicArena NeurIPS 2026 Upload-Ready Map

This directory reorganizes the anonymized submission assets by upload destination.

## 1. OpenReview Direct Upload

Use files under:

```text
upload-ready/openreview-direct/
```

OpenReview fields:

- `PDF`: upload `SymbolicArena_NeurIPS26_ED_main.pdf`.
- `Croissant File`: upload `croissant.OPENREVIEW_NEEDS_URL.json` after replacing URL/license placeholders and validating it.
- `Supplementary Material`: upload `SymbolicArena_NeurIPS26_ED_supplementary_material.zip`.

The supplementary zip is intentionally small and contains only slim result tables, dataset metadata, manifests, and checksums. It excludes raw run directories, minute-level snapshots, conda environments, and large external artifacts.

## 2. Dataset URL

Use files under:

```text
upload-ready/anonymous-dataset-url/
```

Upload these files to an anonymous dataset-hosting location, for example an anonymous Hugging Face Dataset, OSF anonymous project, or another reviewer-accessible anonymous host.

After hosting, fill the OpenReview `Dataset URL` field with that anonymous URL and update the Croissant file.

## 3. Code URL

Use files under:

```text
upload-ready/anonymous-code-url/
```

This is a minimal anonymized code snapshot. For the final Code URL, prefer an anonymous Git repository containing the runnable benchmark infrastructure, installation instructions, smoke tests, and reproduction commands. This snapshot can be used as the initial seed or as a small fallback artifact.

## 4. External Full Artifact URL

Use files under:

```text
upload-ready/anonymous-external-artifact-url/
```

The actual external raw-results archive is generated under:

```text
external-artifact-hosting/dist/
```

Upload the following files to anonymous external artifact hosting:

```text
SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
SHA256SUMS.txt
README_external_artifact.md
EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv
external_artifact_croissant.TEMPLATE.json
```

Do not upload the uncompressed `staging/` directory.

## Remaining Manual Steps

- Replace all `TODO_REPLACE_*` placeholders with anonymous URLs and license metadata.
- Validate Croissant JSON before uploading it to OpenReview.
- Ensure the hosting account, repository names, commit history, README text, and downloadable files do not expose author identity.
- Confirm the OpenReview form uses the corrected plain-text abstract and TL;DR from `openreview-direct/openreview_fields_ready_NEEDS_URL.md`.
