# NeurIPS 2026 E&D Upload Package

This directory collects the files prepared for the NeurIPS 2026 Evaluations & Datasets submission.

## What to upload directly

- `paper-pdf/SymbolicArena_NeurIPS26_ED_main.pdf`
  - Main paper PDF.
  - Built from `paper/Paper-SRInfra/main.tex`.
  - Uses `\usepackage[eandd]{neurips_2026}`.

- `SymbolicArena_NeurIPS26_ED_latex_source.zip`
  - Minimal LaTeX source bundle for archival or camera-ready transfer.
  - Includes `main.tex`, `main.bbl`, `references.bib`, `neurips_2026.sty`, `checklist.tex`, `sections/`, `tables/`, and paper figures.
  - Excludes git metadata, build cache, Chinese draft, and unused trash figures.

## What needs URL-based submission

- Dataset / benchmark artifact:
  - Use an anonymous reviewer-accessible hosting URL.
  - Provide a validated Croissant metadata file on OpenReview.
  - Current local template: `dataset-metadata/croissant.TEMPLATE_NEEDS_URL.json`.

- Code artifact:
  - Use an anonymous code hosting URL for the full runnable repository.
  - This upload package only contains a minimal wrapper/runner code snapshot:
    `SymbolicArena_NeurIPS26_ED_code_snapshot_minimal.zip`.
  - The full repository contains large vendored algorithm assets and should not be blindly uploaded as a supplementary zip.

- Full experiment artifact:
  - Use an anonymous external artifact URL for the large raw experiment outputs.
  - Local staging instructions and build outputs are under:
    `external-artifact-hosting/`.
  - Upload the generated `dist/SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst`,
    `dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst`, and `dist/SHA256SUMS.txt`
    to the external host, not to OpenReview.

## Contents

- `paper-pdf/`
  - Main submission PDF.

- `latex-source/`
  - Unzipped LaTeX source tree.

- `dataset-metadata/`
  - Core-50 manifests, result summary tables, hyperparameter manifests, and Croissant template.

- `code-artifact/`
  - Minimal code snapshot and environment/config files.

- `openreview-form/`
  - Text snippets to paste into OpenReview fields.

- `external-artifact-hosting/`
  - Instructions, source manifest, anonymization/build script, and local external-hosting staging metadata.
  - Its generated `dist/` and `staging/` directories are intentionally git-ignored.

- `FILE_MANIFEST.txt`
  - File list for this upload package.

- `SHA256SUMS.txt`
  - Checksums for integrity checks.

## Blocking items before final upload

- Replace placeholder dataset URL in `dataset-metadata/croissant.TEMPLATE_NEEDS_URL.json`.
- Replace placeholder license in Croissant metadata.
- Decide whether review is double-blind or single-blind in OpenReview.
- If double-blind, ensure dataset/code hosting accounts and URLs do not expose identities.
- Validate the final Croissant JSON before upload.
- Confirm OpenReview file-size limits for any supplemental zip.
