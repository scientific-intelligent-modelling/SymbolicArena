# OpenReview Direct Upload Checklist

Upload these files directly in the OpenReview form.

## PDF

```text
SymbolicArena_NeurIPS26_ED_main.pdf
```

## Croissant File

```text
croissant.OPENREVIEW_READY.json
```

The ready file contains the anonymous dataset URL, direct archive download URLs, archive checksums, license, anonymized citation, code URL, and release-maintenance policy. It also contains the complete NeurIPS 2026 minimal Responsible AI and provenance block:

- `rai:dataLimitations`
- `rai:dataBiases`
- `rai:personalSensitiveInformation`
- `rai:dataUseCases`
- `rai:dataSocialImpact`
- `rai:hasSyntheticData`
- `prov:wasDerivedFrom`
- `prov:wasGeneratedBy`

Before upload, validate the exact direct-upload file:

```bash
mlcroissant validate --jsonld croissant.OPENREVIEW_READY.json
```

Do not upload any older placeholder Croissant template.

## Supplementary Material

```text
SymbolicArena_NeurIPS26_ED_supplementary_material.zip
```

This zip is below the 100MB OpenReview supplementary limit. It contains slim result tables, dataset metadata, manifests, and checksums only.

## Optional Archival Source

```text
SymbolicArena_NeurIPS26_ED_latex_source.zip
```

Upload only if the form or venue workflow asks for LaTeX source. The main required file is the PDF.

## Not Uploaded Here

- Full raw run artifact: upload via the external artifact URL.
- Full runnable repository: provide through the Code URL.
- Large dataset files: provide through the Dataset URL.
- Minute-level progress snapshots: intentionally excluded.
