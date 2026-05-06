# Upload These Files

Use this as the operational checklist.

## OpenReview Form

Upload directly:

```text
openreview-direct/SymbolicArena_NeurIPS26_ED_main.pdf
openreview-direct/SymbolicArena_NeurIPS26_ED_supplementary_material.zip
```

Upload the Croissant file only after replacing URL/license placeholders:

```text
openreview-direct/croissant.OPENREVIEW_NEEDS_URL.json
```

## Dataset URL

Upload these to OSF or another anonymous dataset host:

```text
SymbolicArena_NeurIPS26_ED_full_dataset_payload.tar.zst
SymbolicArena_NeurIPS26_ED_full_dataset_payload.tar.zst.sha256
```

Then paste that anonymous host page into OpenReview:

```text
Dataset URL
```

## Code URL

Upload this to an anonymous code host, or use it to seed an anonymous Git repository:

```text
anonymous-code-url-payload.zip
```

Then paste the anonymous code repository/page into OpenReview:

```text
Code URL
```

## External Raw Results URL

Upload these files to the same OSF project or another anonymous artifact host:

```text
../external-artifact-hosting/dist/SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
../external-artifact-hosting/dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
../external-artifact-hosting/dist/SHA256SUMS.txt
```

This is not the Dataset URL. It is the full raw-results artifact link for reproducibility.
