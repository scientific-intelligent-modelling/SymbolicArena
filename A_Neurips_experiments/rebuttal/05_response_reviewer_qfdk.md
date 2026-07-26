# Response to Reviewer QfdK

> Long draft, not paste-ready.

Thank you for identifying that the submission's organization and artifact
entry points made the framework much harder to understand and reproduce than
the underlying implementation warrants. We agree that an infrastructure paper
must be usable from the main text and repository landing page, not only
recoverable from appendices and internal files.

## 1. Undefined terms and variables

The reviewer is correct that the main text uses several compressed labels
before giving their quantitative meanings. The intended definitions are:

- **Execution health** \(H(P)\): the panel-average combination of finite-output
  rate and non-explosion rate.
- **Panel fidelity** \(F(P)\): the Spearman agreement between dataset
  informativeness induced by a candidate probe panel and by the full
  12-algorithm panel.
- **Behavioral complementarity** \(C(P)\): the mean pairwise Spearman distance
  between algorithms' ID rank, OOD rank, and ID-to-OOD-change vectors.
- **Dataset coverage** \(V(P)\): family and subgroup coverage among the 50 tasks
  ranked most informative by the candidate panel, subject to a 40% maximum
  family share.

The appendix defines these quantities, but that does not solve the reading
problem in the main paper. The response should add these one-sentence
definitions where the terms first appear and define every symbol immediately
after each displayed equation rather than sending the reader to a distant
appendix.

## 2. Main experimental figures

We agree. The current main text gives the leaderboard table and hexagonal
heatmap, while several figures needed to understand construction quality,
coverage, and axis relationships remain in the appendix. At minimum, the main
paper should visually show the 664-to-50 distillation path and the direct
Core-50 versus full-reservoir validation, with secondary diagnostic figures
left in the appendix.

`[AUTHOR CONFIRM BEFORE POSTING]` We will promote the direct
Core-50/full-reservoir validation figure and a compact construction diagram to
the main paper, replacing descriptive material rather than merely adding
pages.

## 3. Dataset sources and conversion pipeline

The reservoir is not a newly collected set of measurements. It is a normalized
union of formula-backed tasks from SRSD, LLM-SRBench, SRBench 1.0,
SRBench2025 first-principles tasks, and classical synthetic families. The
standardization pipeline retains source family and subgroup, and converts each
task to:

```text
train.csv
valid.csv
id_test.csv
ood_test.csv
metadata.yaml
formula.py
```

Mechanical steps include schema normalization, deterministic split generation
when official compatible splits are unavailable, formula replay, target and
feature checks, canonical expression generation, and semantic duplicate
audits. Fields that require semantic judgment, such as missing citations or
descriptions for black-box variables, are not inferred automatically; they are
emitted to a manual-gap report. These details exist in the infrastructure and
data-cleaning appendix, but the main paper should include a compact source
table and a mechanical-versus-human decision table so readers do not need to
reconstruct the pipeline.

## 4. Candidate-200 and Core-50 construction

The phrases flagged by the reviewer refer to explicit strata, not informal
manual choices:

- 140 **high-disagreement** tasks have large clipped log-NMSE disagreement
  between the PySR and LLM-SR discovery probes.
- 20 **one-sided evaluability** tasks expose cases where one probe yields an
  evaluable result and the other does not.
- 40 **mid-gap** tasks prevent the calibration pool from containing only
  extreme disagreements.
- Source, subgroup, normalized basename, and advantage-side caps prevent a
  single family, duplicate equation, or probe-favored side from dominating the
  200 tasks.

The selected non-LLM Probe-4 is then run on all 664 tasks with three seeds.
Core-50 is chosen from those full-reservoir responses using structural and
response coverage, discriminability, stability, difficulty balance,
failure-mode coverage, redundancy control, and hard quota constraints. We
agree that the submitted prose compresses this dependency chain too heavily.
The revision should present it as a numbered pipeline and place the quota
values next to the relevant stage.

## 5. Metric forms and constants

The coefficients are fixed protocol choices rather than values fitted to the
reported algorithm panel. The absolute NMSE mapping has fixed log-scale
anchors, OOD-G separates absolute OOD quality from retention, and SYM-F
prioritizes exact equivalence over partial structural overlap. Each axis is
reported separately, so a reader need not accept an opaque weighted total, and
adding a new algorithm cannot rescale existing scores.

Nevertheless, the reviewer is right that this rationale does not replace a
sensitivity analysis. The submitted version should not imply that close
algorithm differences are invariant to all coefficient choices.

`[AUTHOR CONFIRM BEFORE POSTING]` We will add a bounded perturbation analysis
for the construction and metric weights. If that analysis is not completed
before the response deadline, this sentence must instead state the limitation
and avoid promising a result.

## 6. Anonymous code documentation

We agree that the Code URL landing page is inadequate. The submitted payload
describes itself as a lightweight audit snapshot and lists what a future full
release should contain. That is an internal packaging note, not a user-facing
README, and it does not satisfy the installation/run/extension requirements of
an evaluation-infrastructure contribution.

The repository does contain the runner, standardized result schema, 12
wrappers, environment manifests, scoring scripts, and wrapper contract, but
their presence does not compensate for the missing reproducibility path.

`[AUTHOR CONFIRM BEFORE POSTING]` The anonymous landing page will be replaced
with tested instructions for environment creation, data download and checksum
verification, a one-dataset smoke test, Core-50 execution, leaderboard
regeneration, result-schema interpretation, and adding a new wrapper. This
claim must only be posted after those commands have been run from the anonymous
artifact.

## 7. Croissant conformance

The reviewer is correct. The current record uses generic keys such as
`rai:limitations` and `rai:useCases`, but the track validator expects
`rai:dataLimitations`, `rai:dataBiases`, `rai:dataUseCases`,
`rai:dataSocialImpact`, and `rai:hasSyntheticData`. The optional provenance
fields are also absent. This is a schema-conformance error, not an ambiguity in
the review.

The corrected metadata should state:

- `rai:hasSyntheticData`: true;
- `rai:dataLimitations`: Core-50 is conditioned on the current task pool and
  probes and does not validate arbitrary real scientific data;
- `rai:dataBiases`: source-family, equation-family, operator, dimensionality,
  and formula-availability biases;
- `rai:dataUseCases`: controlled SR benchmarking, wrapper validation, and
  diagnostic evaluation;
- `rai:dataSocialImpact`: no human-subject data, with the principal risk being
  overstatement of close rankings or transfer to real scientific discovery;
- `prov:wasDerivedFrom`: the named upstream benchmark sources;
- `prov:wasGeneratedBy`: the deterministic standardization and validation
  pipeline.

`[AUTHOR CONFIRM BEFORE POSTING]` We will validate the corrected JSON-LD with
the same conformance checker before saying that this issue is resolved.

## Closing

The core technical response is not that the missing explanations were
unnecessary; it is that the implementation and appendices contain an auditable
pipeline whose main-paper and repository presentation failed to expose it.
We appreciate this distinction and will narrow terminology, move essential
definitions and evidence forward, and treat installation and metadata
validation as release gates rather than optional documentation.

## Author checks before posting

- [ ] Confirm the OpenReview character limit and compress this long-form draft.
- [ ] Verify which paper or artifact revisions are permitted during rebuttal.
- [ ] Remove or approve every `[AUTHOR CONFIRM BEFORE POSTING]` commitment.
- [ ] Do not claim that README or Croissant issues are fixed until the public
  anonymous artifacts themselves pass the documented smoke and schema checks.
