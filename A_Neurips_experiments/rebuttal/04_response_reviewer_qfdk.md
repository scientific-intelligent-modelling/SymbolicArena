Long draft, not paste-ready. Character limit and final wording still need author compression.

# Draft response to Reviewer QfdK

Thank you for the detailed review. We agree with the core diagnosis that the current draft does not explain the construction pipeline and artifact entry points clearly enough. Below we respond point by point and separate what is already in the artifact from what we should state more directly in the paper.

## Concern summary

You raise six main issues: (1) the writing is obscure and key terms are deferred too far into the appendix; (2) the dataset collection and standardization pipeline is not clear enough; (3) the Core-50 construction procedure is too abstractly described; (4) the metric formulas and coefficients appear arbitrary; (5) the code / artifact documentation is insufficient, including the review-facing README; and (6) the Croissant metadata did not validate cleanly.

## 1. Clarity of the main paper

We agree. The pipeline has three conceptually different layers that the current draft does not separate cleanly enough in the main text:

1. reservoir standardization: heterogeneous source tasks are filtered and normalized into GT-Reservoir-664;
2. benchmark distillation: the 664-task reservoir is compressed into Core-50 through dual-probe mining, Candidate-200 calibration, Probe-4 selection, and constrained subset optimization;
3. leaderboard evaluation: once Core-50 is frozen, all methods are evaluated through the same execution and scoring contract.

Several terms that you flagged, such as execution health, panel fidelity, behavioral complementarity, and coverage, do have operational definitions in the appendix and code assets, but we agree that the main paper should surface those definitions earlier and more plainly. We also agree that variables appearing in equations should be reintroduced locally instead of assuming appendix context.

## 2. Dataset collection and standardization pipeline

This point is well taken. The intended story is:

- start from a larger heterogeneous SR candidate pool;
- retain only tasks with executable formulas, valid splits, consistent targets, and sufficient metadata;
- normalize each retained task into a shared format with train / validation / ID-test / OOD-test CSVs, metadata, formula identity, feature names, target name, and source-family tags;
- apply duplicate control and audit the resulting reservoir for source-family and structural coverage.

The resulting ground-truth reservoir contains 664 tasks. We should describe much more explicitly in the main paper which source families are included, how tasks are converted to the shared format, and which steps are mechanical versus manually reviewed. We already release manifests for the Core-50 formulas, source families, and reproducibility assets, but the paper should point readers to them directly instead of assuming the artifact will be explored independently.

## 3. Core-50 construction is too abstract

We agree that phrases like “high-disagreement tasks,” “one-sided evaluability cases,” and “mid-gap tasks under source and subgroup caps” are too compressed for the main paper. In plainer terms, the construction is:

- dual-probe mining over all 664 tasks to surface informative differences;
- Candidate-200 construction from 140 high-disagreement tasks, 20 one-sided evaluability tasks, and 40 mid-gap tasks;
- 12-method calibration on Candidate-200 to choose a compact non-LLM Probe-4 panel;
- 3-seed Probe4-Full evaluation on all 664 tasks;
- constrained optimization of a 50-task subset to preserve coverage, information, discriminability, difficulty balance, redundancy control, failure-mode balance, and stability.

We should move this simplified procedural summary into the main paper and leave the exact formulas for the appendix.

## 4. Metric formulas and coefficients

We agree that the current draft does not justify the coefficients well enough. Our intention was to use fixed absolute mappings so that adding new algorithms does not retroactively change previously reported scores, which we view as important for a dynamic benchmark. But that motivation does not eliminate the need to explain coefficient choices and to avoid over-interpreting small score gaps.

So the right rebuttal stance is narrow: the protocol is fixed, explicit, and auditable, but the current paper should do a better job separating robust directional findings from fragile fine-grained ranking differences.

## 5. Reproducibility and documentation

We agree that the review-facing documentation should have been stronger. The underlying artifact already contains:

- installation and upload maps;
- reproduction commands and artifact checklists;
- machine-readable hyperparameter manifests;
- benchmark metadata and formula manifests;
- Croissant metadata and dataset-metadata instructions.

However, the review-facing README was not a strong enough entry point, and readers should not have to infer the intended install / run / extend path by exploring the package manually. We will therefore frame this as a documentation shortcoming that needs correction rather than arguing that the current entry point is already ideal.

## 6. README placeholder and Croissant validation

We agree with both points.

First, the review-facing README should not look like a placeholder; even if the underlying code is substantial, the entry point shapes reproducibility perception.

Second, the Croissant issue is real. The metadata package needs to be explicit, validated, and complete for the E&D setting. Our current artifact tree already contains upload-ready metadata files and instructions, but the submission-side packaging did not make that sufficiently reliable or obvious. We should acknowledge that directly.

## Closing

Overall, we think your review identifies a presentation and packaging problem more than a missing infrastructure problem. The underlying contribution is a standardized, executable SR evaluation substrate with dataset normalization, benchmark distillation, unified wrappers, fixed result schemas, and released artifacts. But we agree that the paper must be reorganized so that readers can understand that contribution from the main text without depending on appendix archaeology or artifact guesswork.

## Author confirmation needed before posting

- Confirm whether we want to explicitly promise moving several appendix definitions into the main paper.
- Confirm whether we want to state that the Croissant issue has already been corrected in the artifact package, or more conservatively say that we identified the problem and will correct it.
- Confirm whether we want to mention the exact Candidate-200 composition numbers (`140 + 20 + 40`) in the rebuttal.
- Confirm whether we want to describe the current README issue as “too generic” or more directly as “not review-facing enough.”
