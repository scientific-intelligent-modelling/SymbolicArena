# Anonymous Code URL Payload

Use this directory as a seed for the anonymous code-hosting release.

Recommended hosts:

- Anonymous GitHub mirror through an anonymization service.
- Anonymous GitLab repository.
- Anonymous OSF project with a downloadable code archive.

## Included Files

```text
SymbolicArena_NeurIPS26_ED_code_snapshot_minimal.zip
README_code_artifact.md
environment.yml
toolbox_config.json
```

The snapshot is intentionally minimal. For the final Code URL, the preferred artifact is a cleaned runnable repository with:

- Installation instructions.
- Environment setup.
- Smoke tests.
- Benchmark runner entry points.
- Result schema and postprocessing scripts.
- No API keys, LLM provider configs, SSH configs, local paths, personal names, or private hostnames.

## OpenReview Field

Paste the anonymous repository URL into:

```text
Code URL
```

Do not leave the Code URL blank for this submission unless the paper is reframed as purely analytical. SymbolicArena is a reusable benchmark/evaluation infrastructure, so code access is important for review.
