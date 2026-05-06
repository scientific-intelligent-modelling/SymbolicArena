# Anonymous External Artifact URL Payload

Use this directory together with generated files under:

```text
paper/neurips26-upload/external-artifact-hosting/dist/
```

Upload the following generated files to an anonymous external artifact host:

```text
SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
SHA256SUMS.txt
```

Also keep these documentation files visible on the hosting page:

```text
README_external_artifact.md
UPLOAD_CHECKLIST.md
EXTERNAL_ARTIFACT_BUILD_SUMMARY.md
EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv
external_artifact_croissant.TEMPLATE.json
```

## Recommended Hosts

- Anonymous Hugging Face Dataset repository.
- OSF anonymous view-only project.
- Zenodo private review link, if compatible with the venue workflow.
- Figshare private/reviewer link.

## Do Not Upload

- `external-artifact-hosting/staging/`
- `minute_*.json`
- `.log` files
- scheduler directories such as `__launcher__`
- conda environments
- model caches
- local SSH configs or LLM provider configs

## OpenReview Usage

If the OpenReview form has only one Dataset URL field, use it for the dataset host and mention this external artifact URL in the paper artifact/reproducibility section or supplementary README. If a large-dataset/sample URL field is appropriate, use it for the external artifact or a representative sample page.
