# External Artifact Upload Checklist

## Before Upload

- Run `inventory` and confirm all required source directories exist.
- Run `prepare` to create an anonymized staging payload.
- Run `scan` and confirm no private paths, private IPs, internal hostnames, API keys, or personal emails remain.
- Run `package` to create the `tar.zst` archive.
- Run `checksum` after packaging.

## Files to Upload Externally

- `dist/SymbolicArena_NeurIPS26_ED_external_full_artifact.tar.zst`
- `dist/SHA256SUMS.txt`
- `dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt.zst`
- `README_external_artifact.md`
- `EXTERNAL_ARTIFACT_SOURCE_MANIFEST.csv`
- `external_artifact_croissant.TEMPLATE.json`

The plain `dist/EXTERNAL_ARTIFACT_FILE_MANIFEST.txt` is large because the payload contains more than one million files. Prefer uploading the compressed `.zst` copy.

## After Upload

- Replace every `TBD_EXTERNAL_ARTIFACT_URL` placeholder with the anonymous URL.
- Verify the URL works in an incognito browser session.
- Download the archive once and verify its SHA256 checksum.
- Do not reveal account names, usernames, hostnames, or institution-specific private paths in the hosting page.

## OpenReview Upload Package Linkage

The OpenReview package should not contain the 20GB artifact. It should contain only the lightweight upload package and reference this external URL.
