Long draft, not paste-ready. Character limit and final wording still need author compression.

# Draft response to Reviewer iM84

Thank you for the careful and technically sharp review. We appreciate both the positive assessment of the engineering and the pointed concerns about validity. We agree that several claims in the current draft should be narrowed and better evidenced. Below we respond point by point.

## Concern summary

Your main concerns are: (1) Core-50 validation is too close to the same probe family used in construction; (2) the protocol uses many fixed coefficients without enough sensitivity analysis; (3) LLM contamination is not controlled; (4) the very weak RAG-SR result may reflect an integration artifact; and (5) a single one-hour budget may not be equally fair across paradigms.

## 1. Core-50 distillation validation and held-out algorithms

We agree that the original draft provides stronger evidence for “preserves the full 664-task probe conclusions” than for the stricter claim “generalizes to algorithms outside the construction family.” That is a fair weakness.

To probe exactly this issue, after submission we ran three additional algorithms that were not part of the original construction pipeline (FePySR, JAXSR, SymbolFit) on the same Full-664 clean protocol and merged them with four existing Stage-3 algorithms into a 7-algorithm comparison. This new held-out evidence materially strengthens the numerical representativeness claim:

- across the 7 algorithms, Core-50 vs Full-664 mean OOD ranking has Pearson `0.989456061` and Spearman `0.892857143`;
- for the three truly held-out algorithms alone, the OOD ordering is preserved exactly (Spearman `1.0`, Pearson `0.815541684`).

For symbolic fidelity, the evidence is encouraging but weaker:

- across the 7 algorithms, Core-50 vs Full-664 SYM-F has Pearson `0.971269484` and Spearman `0.857142857`;
- however, for the three held-out algorithms alone, the SYM-F ordering is not stable (Spearman `-0.5`), and the score gaps are small.

So the rebuttal-safe statement is: the new held-out results strongly support Core-50 as a faithful low-cost proxy for reservoir-level numerical conclusions, and they support group-level symbolic-fidelity preservation, but they do not justify a strong claim that fine-grained held-out symbolic ordering is fully solved.

## 2. Fixed scoring constants and coefficient sensitivity

We agree that the paper currently discloses the constants more clearly than it justifies them. Our design goal was to use fixed absolute mappings so that scores do not drift when new algorithms are added; this matters for a living leaderboard. But we agree that “drift-free” is not the same as “fully sensitivity-validated.”

Accordingly, we should narrow the claim in the rebuttal:

- the protocol is fixed, explicit, and auditable;
- many of the main qualitative conclusions are robust at the level of broad separations;
- but the current draft should not ask readers to over-interpret very small inter-method differences without a fuller sensitivity study.

Relatedly, the current leaderboard package already reports dataset-bootstrap 95% confidence intervals for the formal hexagon scores, which helps show that some small gaps are not worth over-reading. We will use that evidence to temper the claim, not to imply that bootstrap intervals replace a full coefficient-perturbation analysis.

## 3. LLM contamination

We agree with this concern. Withholding physical descriptions during the dual-probe mining stage reduces semantic-context leakage, but it does not rule out formula memorization on standard benchmark families such as Feynman-style tasks. So this should be treated as a current threat to validity, not only future work.

Our intended scope is therefore narrower than “LLM-assisted SR has been contamination-controlled.” The safe claim is that SymbolicArena exposes such methods to the same standardized execution and evaluation contract; it does not yet provide a dedicated contamination-proof protocol. We should make that limitation explicit.

## 4. RAG-SR collapse: integration bug or genuine weak performance?

We checked this concern carefully, and the current evidence points more toward genuine weak performance under the shared protocol than toward a simple wrapper failure.

In the clean Core-50 leaderboard, RAG-SR has:

- valid-output rate `1.000`,
- metric-complete rate `0.964`,
- only `9/250` metric-incomplete runs,
- `1` dataset with good OOD quality and `27` with bad OOD quality,
- exact equivalence `1.6%`,
- efficiency score `3.2`.

The corresponding diagnostic appendix therefore classifies RAG-SR mainly as numerical underfitting and weak efficiency, not as a missing-output or non-functional integration case. So we agree the result deserves explanation, but the available evidence does not support the simpler story that the method “just broke.”

## 5. One-hour budget fairness

We agree that a single wall-clock budget is not a perfect paradigm-neutral notion of compute fairness. Our intention was to define a standardized evaluation contract that can be applied uniformly across heterogeneous methods, not to claim that one hour equalizes search effort, model pretraining cost, or per-iteration efficiency.

We therefore think the right wording is:

- the one-hour budget is a shared deployment budget, useful for a common benchmark contract;
- the minute-level best-so-far traces were logged precisely so that anytime and budget-sweep analyses can be added later;
- the current paper should not overclaim that this budget is the unique or universally fairest way to compare all paradigms.

## Closing

Overall, we agree with the direction of your critique. The strongest rebuttal-supported update is the new held-out Full-664 evidence: it substantially strengthens the claim that Core-50 preserves reservoir-level numerical conclusions for algorithms outside the construction family. At the same time, we should narrow the paper where the evidence is weaker: coefficient sensitivity is not yet fully established, contamination is not controlled, and the one-hour budget should be framed as a standardized contract rather than a complete answer to compute fairness.

## Author confirmation needed before posting

- Confirm whether we want to quote the new held-out Pearson / Spearman numbers directly in the rebuttal.
- Confirm whether we want to explicitly name FePySR, JAXSR, and SymbolFit as post-submission held-out methods.
- Confirm whether we want to say “RAG-SR is not an integration bug” or the slightly softer “the current diagnostics do not indicate a simple integration bug.”
- Confirm whether we want to promise a future sensitivity sweep, or only say that we will narrow the current claim and mark it as future work.
