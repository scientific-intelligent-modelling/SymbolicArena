# External Artifact Build Summary

Generated local staging path:

```text
paper/neurips26-upload/external-artifact-hosting/staging/SymbolicArena_NeurIPS26_ED_external_full_artifact
```

Generated upload files:

```text
paper/neurips26-upload/external-artifact-hosting/dist/SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
paper/neurips26-upload/external-artifact-hosting/dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt
paper/neurips26-upload/external-artifact-hosting/dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
paper/neurips26-upload/external-artifact-hosting/dist/SHA256SUMS.txt
```

Build status:

- Source inventory: all required sources found.
- Prepared payload files: 84,488.
- Prepared payload size: approximately 988MB after excluding scheduler-only directories, logs, and minute-level snapshots.
- Main archive size: approximately 33MB.
- Plain file manifest size: approximately 22MB.
- Compressed file manifest size: approximately 460KB.

Anonymization status:

- Removed `__launcher__` scheduler directories from the staging payload.
- Removed `.log` files from the staging payload.
- Removed `minute_*.json` progress snapshots from the staging payload.
- Replaced local machine paths, private remote paths, internal hostnames, private IPs, API-key pattern, and personal identifiers in text artifacts.
- Replaced non-English CJK text in copied text artifacts with English descriptions or neutral English placeholders.
- Renamed internal host path segments to `host_XX`.

Validation status:

- Path-name sensitive scan: passed.
- Text-content sensitive scan over staging payload: passed.
- SHA256 checksums generated in `dist/SHA256SUMS.txt`.

Generated checksums:

```text
499785ff5b96f13e7a85bf10c70916908d19616cced8e924df47a93ce394c613  EXTERNAL_ARTIFACT_FILE_MANIFEST.txt
9d34c4b8a41ead188d9b2bb333c0d4d9ae2b304686ee9b415319b00dcf590001  EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
ea54262e184ca88833d74856cba2b4a00272d0e1cb203332aa05be9360fcb3f0  SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
```

Recommended external upload set:

```text
SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst
EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst
SHA256SUMS.txt
README_external_artifact.md
EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv
external_artifact_croissant.TEMPLATE.json
```

The uncompressed `EXTERNAL_ARTIFACT_FILE_MANIFEST.txt` is kept locally for inspection. Upload the compressed `.zst` copy unless the hosting service requires plain text.
