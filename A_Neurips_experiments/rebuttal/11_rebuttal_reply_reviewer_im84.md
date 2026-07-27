> Long-form draft for author editing. Not paste-ready.
>
> Source: Reviewer iM84 official review on June 25, 2026, modified July 23, 2026.

# Reply to Reviewer iM84

Thank you for the careful and technically sharp review. We especially
appreciate that you separated the infrastructure contribution, the fixed-score
design, and the diagnostic interpretation of specific methods. We agree that
the strongest current weaknesses are the held-out validation story for Core-50,
the lack of a full sensitivity analysis for fixed constants, and the absence of
an explicit contamination control for LLM-based methods. Below we respond point
by point and narrow the claims where needed.

## Main concerns addressed

This draft follows the five highest-signal issues in the review:

1. held-out validation of Core-50 beyond the construction panel;
2. missing sensitivity analysis for fixed constants;
3. LLM contamination risk;
4. whether RAG-SR looks genuinely weak or mis-integrated; and
5. fairness of a shared one-hour budget across heterogeneous paradigms.

## 1. Distillation faithfulness beyond the construction panel

We agree that this is the most important concern. The original submission
showed that Core-50 preserves reservoir-level behavior better than random,
stratified, or other deterministic 50-task subsets, but that comparison is not
fully decisive if the evidence remains too close to the quantities used during
construction.

To probe exactly this issue, after submission we ran three additional
algorithms that were not involved anywhere in benchmark construction:
FePySR, JAXSR, and SymbolFit. These methods did not participate in:

1. PySR + LLM-SR dual-probe mining;
2. the 12-method Candidate-200 calibration;
3. Probe-4 selection; or
4. Core-50 construction itself.

We evaluated them on all 664 tasks with 3 seeds under the same clean one-hour
contract, yielding all 5976 expected runs with complete numerical metrics.

Using the same-seed comparison between the Core-50 slice and the full
664-task reservoir:

- across the full 7-algorithm panel, penalized OOD log-NMSE has Pearson
  0.989456061 and Spearman 0.892857143;
- for the 3 genuinely held-out methods alone, the OOD ordering is preserved
  exactly, with Pearson 0.815541684 and Spearman 1.0.

This is the strongest evidence we currently have that Core-50 preserves broad
numerical ranking behavior beyond the algorithms used during construction.

For symbolic fidelity, the picture is more mixed and we think the rebuttal
should say that directly. Across the 7 methods, Core-50 and Full-664 SYM-F
still correlate strongly (Pearson 0.971269484, Spearman 0.857142857). However,
among the 3 held-out methods alone, the symbolic ordering is unstable
(Spearman = -0.5). So the right claim is not that Core-50 preserves every
fine-grained symbolic comparison, but that it preserves broad panel-level
trends while near-tied symbolic gaps remain fragile.

## 2. Fixed constants and missing sensitivity analysis

We agree with this criticism. The current paper contains several fixed
constants: Probe-4 selection weights, Core-50 objective weights, the absolute
numerical mapping, and the subweights inside OOD-G, SYM-F, ROBU, and STAB. The
paper explains what these constants do, but it does not yet provide a complete
sensitivity analysis showing how rankings or Core-50 membership move under
reasonable perturbations.

What we can safely claim today is:

- the constants are fixed protocol anchors, not values fit to maximize the
  performance of the reported algorithm panel;
- the protocol reports per-axis values separately, so readers can inspect OOD-G,
  SYM-F, EFF, ROBU, and STAB without accepting a hidden total score;
- absolute mappings ensure that adding a new algorithm does not rescale
  previously reported scores.

What we should not claim without further evidence is that close ranks are
insensitive to all reasonable perturbations of these constants.

## 3. LLM contamination

We agree that the current paper does not control contamination strongly enough
to justify a claim about uncontaminated LLM reasoning. Withholding physical
descriptions reduces semantic assistance, but it does not prevent the
possibility that a model has already seen standard SR benchmark families or
their formulas.

The strongest accurate statement is therefore narrower:

- LLM-SR is used only in the initial dual-probe stage to identify informative
  tasks and disagreement patterns;
- the final Core-50 construction is based on full-664 responses of the non-LLM
  Probe-4 panel;
- nevertheless, the initial LLM probe can still indirectly influence which
  task patterns or downstream panels appear most informative.

So contamination remains a current threat to validity, not just future work. A
stronger claim would require dedicated controls such as procedurally novel
formula families or an explicit post-cutoff evaluation set.

## 4. RAG-SR collapse and whether this is an integration bug

We agree that the paper should not leave the impression that RAG-SR is simply
"near zero on every axis" without diagnosis. The diagnostic appendix already
suggests that this is not primarily a missing-output case.

On the clean Core-50 leaderboard, RAG-SR has:

- valid-output rate 1.000;
- metric-complete rate 0.964;
- 8 timed-out runs and 9 metric-incomplete runs out of 250 total runs;
- weak clean-track ID/OOD quality, exact-equivalence rate 1.6%, and EFF 3.2.

The failure-diagnostics summary further shows that 42 of 50 datasets have
five-seed median OOD log-NMSE above 0, and 27 of 50 are above 2. So under the
current wrapper and context protocol, the dominant problem looks like
numerical underfitting / retrieval mismatch / weak extrapolation rather than a
simple wrapper crash.

We therefore agree that the result needs to be explained more clearly, but the
current diagnostics do not indicate a simple integration bug.

## 5. Single one-hour budget and fairness across paradigms

We agree that a single one-hour wall-clock budget is not "fair" in every
possible sense. It is not equal-FLOP, equal-dollar, or equal-hyperparameter
search fairness. The intended claim is narrower: it is a shared, reproducible,
user-facing evaluation contract across heterogeneous paradigms.

This is also why the benchmark does not rely only on final-time quality. For
snapshot-capable methods, the runner stores minute-level best-so-far traces,
and the protocol reports an explicit EFF axis based on anytime behavior. But we
agree with your broader point: some rankings could change under a different
budget horizon. So the one-hour leaderboard should be presented as one
standardized operating point, not as a universal budget-free ordering of all SR
paradigms.

## Closing

Your review usefully separates what the current paper already establishes from
what it does not yet establish. We think the new held-out experiment materially
improves the Core-50 validation story for numerical ranking preservation, but
we also agree with three important boundaries: the current submission lacks a
full sensitivity analysis for protocol constants, lacks a contamination control
for LLM-based methods, and uses a wall-clock contract rather than a
compute-equivalent fairness notion. Stated with those boundaries, we believe
the benchmark still makes a meaningful contribution as a unified and auditable
SR evaluation substrate.

## Author checks before posting

- Confirm whether to quote the held-out Pearson / Spearman numbers directly in
  the short rebuttal.
- Confirm whether to explicitly name FePySR, JAXSR, and SymbolFit.
- Confirm whether to use the softer wording "the current diagnostics do not
  indicate a simple integration bug" for RAG-SR.
- Confirm whether to promise any bounded sensitivity analysis, or only narrow
  the current claim and mark full sensitivity as future work.
