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

The ready file already contains the anonymous Dataset URL, license URL, anonymized citation, code URL, and maintenance plan.

Before upload, run a Croissant validator if possible.

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
