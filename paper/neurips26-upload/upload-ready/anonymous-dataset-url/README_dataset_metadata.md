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
  - Croissant 1.1 metadata template for a future host or release.
  - Host-specific `TODO_REPLACE_*` values still need replacement, but the Responsible AI and PROV-O fields are complete and must not be replaced with placeholders.

- `croissant.OPENREVIEW_READY.json`
  - OpenReview-ready Croissant 1.1 metadata for the anonymous OSF release.
  - Uses direct OSF download URLs, archive checksums, archive-contained `FileSet` entries, and field-level CSV extraction definitions.

## Responsible AI and provenance coverage

The ready file and template both include the exact NeurIPS 2026 minimal fields:

- `rai:dataLimitations`
- `rai:dataBiases`
- `rai:personalSensitiveInformation`
- `rai:dataUseCases`
- `rai:dataSocialImpact`
- `rai:hasSyntheticData`
- `prov:wasDerivedFrom`
- `prov:wasGeneratedBy`

`prov:wasDerivedFrom` identifies the upstream SRSD, LLM-SRBench, SRBench, SRBench2025, and classical formula/domain resources. `prov:wasGeneratedBy` is a machine-readable `prov:Activity` describing standardization, split materialization, validation, duplicate control, and Core-50 distillation.

## Final upload requirement

Upload `croissant.OPENREVIEW_READY.json`, not the template. Validate it before upload:

```bash
mlcroissant validate \
  --jsonld paper/neurips26-upload/dataset-metadata/croissant.OPENREVIEW_READY.json

pytest -q tests/test_neurips_croissant_metadata.py
```

The repository regression test checks the required Responsible AI and provenance fields, minimal Croissant structure, absence of unresolved placeholders in the ready file, and byte-equivalent JSON content in the direct-upload mirror.
