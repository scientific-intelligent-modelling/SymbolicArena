> Long-form draft for author editing. Not paste-ready.

# Reply to Reviewer QfdK

Thank you for the detailed review. We agree with the core diagnosis that the
current draft does not explain the construction pipeline and artifact entry
points clearly enough. Below we respond point by point and separate what is
already present in the artifact from what the paper should state much more
directly.

## Concern summary

You raise six main issues: (1) the writing is obscure and key terms are
deferred too far into the appendix; (2) the dataset collection and
standardization pipeline is not clear enough; (3) the Core-50 construction
procedure is described too abstractly; (4) the metric formulas and coefficients
appear arbitrary; (5) the code / artifact documentation is insufficient,
including the review-facing README; and (6) the Croissant metadata did not
validate cleanly.

## 1. Clarity of the main paper

We agree. The pipeline has three conceptually different layers that the current
draft does not separate cleanly enough in the main text:

1. reservoir standardization: heterogeneous source tasks are filtered and
   normalized into GT-Reservoir-664;
2. benchmark distillation: the 664-task reservoir is compressed into Core-50
   through dual-probe mining, Candidate-200 calibration, Probe-4 selection,
   and constrained subset optimization;
3. leaderboard evaluation: once Core-50 is frozen, all methods are evaluated
   through the same execution and scoring contract.

Several terms you flagged, such as execution health, panel fidelity,
behavioral complementarity, and coverage, do have operational definitions in
the appendix and artifact, but we agree that the main paper should surface
those definitions earlier and more plainly. We also agree that variables
appearing in equations should be reintroduced locally instead of assuming
appendix context.

## 2. Dataset collection and standardization pipeline

This point is well taken. The intended story is:

- start from a larger heterogeneous SR candidate pool;
- retain only tasks with executable formulas, valid splits, consistent targets,
  and sufficient metadata;
- normalize each retained task into a shared format with train / validation /
  ID-test / OOD-test CSVs, metadata, formula identity, feature names, target
  name, and source-family tags;
- apply duplicate control and audit the resulting reservoir for source-family
  and structural coverage.

The resulting ground-truth reservoir contains 664 tasks. We should describe much
more explicitly in the main paper which source families are included, how tasks
are converted to the shared format, and which steps are mechanical versus
manually reviewed.

## 3. Core-50 construction is too abstract

We agree that phrases like "high-disagreement tasks," "one-sided evaluability
cases," and "mid-gap tasks under source and subgroup caps" are too compressed
for the main paper. In plainer terms, the construction is:

- dual-probe mining over all 664 tasks to surface informative differences;
- Candidate-200 construction from 140 high-disagreement tasks, 20 one-sided
  evaluability tasks, and 40 mid-gap tasks;
- 12-method calibration on Candidate-200 to choose a compact non-LLM Probe-4
  panel;
- 3-seed Probe-4 Full-664 evaluation on all 664 tasks;
- constrained optimization of a 50-task subset to preserve coverage,
  information, discriminability, difficulty balance, redundancy control,
  failure-mode balance, and seed-level stability.

We agree that this simplified procedural summary should move into the main
paper, with the exact optimization formulas left in the appendix.

## 4. Metric formulas and coefficients

We agree that the current draft does not justify the coefficients well enough.
Our intention was to use fixed absolute mappings so that adding new algorithms
does not retroactively change previously reported scores, which we view as
important for a dynamic benchmark. But that motivation does not remove the need
to explain coefficient choices and to avoid over-interpreting close score gaps.

So the right rebuttal stance is narrow: the protocol is fixed, explicit, and
auditable, but the current paper should do a better job separating robust
directional findings from fragile fine-grained ranking differences.

## 5. Reproducibility and documentation

We agree that the review-facing documentation should have been stronger. The
underlying artifact already contains:

- installation and upload maps;
- reproduction commands and artifact checklists;
- machine-readable hyperparameter manifests;
- benchmark metadata and formula manifests; and
- metadata instructions and supplementary packaging files.

However, the review-facing README was not a strong enough entry point, and
readers should not have to infer the intended install / run / extend path by
exploring the package manually. We should frame this as a documentation
shortcoming rather than argue that the current entry point is already ideal.

## 6. README placeholder and Croissant validation

We agree with both points.

First, the review-facing README should not look like a placeholder; even if the
underlying code is substantial, the entry point strongly shapes reproducibility
perception.

Second, the Croissant issue is real. The metadata package needs to be explicit,
validated, and complete for the E&D setting. We should acknowledge that
directly rather than downplay it.

## Closing

Overall, we think your review identifies a presentation and packaging problem
more than an absence of infrastructure. The underlying contribution is a
standardized executable SR evaluation substrate with dataset normalization,
benchmark distillation, unified wrappers, fixed result schemas, and released
artifacts. But we agree that the paper must be reorganized so that readers can
understand that contribution from the main text without depending on appendix
archaeology or artifact guesswork.

## Author checks before posting

- Confirm whether to explicitly mention the Candidate-200 composition numbers
  (140 + 20 + 40) in the short rebuttal.
- Confirm whether to phrase the README issue as "too generic" or more directly
  as "not review-facing enough."
- Confirm whether to say only that the metadata packaging issue has been
  identified, or to claim that a corrected package will definitely be supplied.
