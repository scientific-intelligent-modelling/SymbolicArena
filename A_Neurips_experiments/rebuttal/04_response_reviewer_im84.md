# Response to Reviewer iM84

> Long draft, not paste-ready.

Thank you for the careful review. We especially appreciate that you separated
the infrastructure contribution, the fixed-absolute scoring design, and the
failure modes of specific algorithms rather than reducing the paper to a single
leaderboard table. We agree that the strongest current weaknesses are the
held-out validation story for Core-50, the lack of a full sensitivity analysis
for hand-set constants, and the absence of an explicit contamination control for
LLM-based methods. Below we respond point by point and narrow the claims where
needed.

## 1. Core-50 validation beyond the construction panel

We agree that this is the most important concern. The original submission
showed that Core-50 preserves reservoir-level behavior better than random,
stratified, or other deterministic 50-task subsets, but that comparison is not
fully convincing if all evidence stays too close to the quantities optimized
during construction.

We therefore ran a post-submission held-out test on **three algorithms that were
not involved anywhere in benchmark construction**: **FePySR, JAXSR, and
SymbolFit**. These methods were integrated only after the benchmark had already
been frozen and did **not** participate in:

1. the initial **PySR + LLM-SR dual-probe mining**;
2. the **12-method Candidate-200 calibration**;
3. the **Probe-4 selection**;
4. the **Core-50 construction** itself.

We evaluated these three methods on **all 664 tasks**, with **3 seeds**
(`520/521/522`), under the same **clean 1-hour** execution contract. This
yielded all **5976/5976** expected runs with complete numerical metrics.

Using the resulting same-seed comparison between the **Core-50 slice** and the
**full 664-task reservoir**:

- across the full **7-algorithm** panel (the original 4 Probe-4 methods plus
  these 3 held-out methods), penalized OOD log-NMSE has **Pearson
  0.989456061** and **Spearman 0.892857143**;
- for the **3 genuinely held-out methods alone**, the OOD ordering is preserved
  exactly, with **Pearson 0.815541684** and **Spearman 1.0**.

This is the strongest evidence we currently have that Core-50 preserves broad
**numerical** ranking behavior beyond the algorithms used during construction.

For **symbolic fidelity**, the picture is more mixed and we think the rebuttal
should say that explicitly rather than oversell it. Across the same 7 methods,
Core-50 and Full-664 SYM-F still correlate strongly (**Pearson 0.971269484**,
**Spearman 0.857142857**). However, among the 3 held-out methods alone, the
symbolic ordering is unstable (**Spearman = -0.5**). So the correct claim is
not that Core-50 preserves every fine-grained symbolic comparison, but that it
preserves broad panel-level trends while fine symbolic gaps between near-tied
methods remain fragile.

## 2. Hand-set constants and missing sensitivity analysis

We agree with this criticism. The current paper contains many fixed constants:
Probe-4 selection weights, Core-50 objective weights, the absolute numerical
mapping, and the subweights inside OOD-G, SYM-F, ROBU, and STAB. The paper
explains *what* these constants do, but it does **not** yet provide a complete
sensitivity analysis showing how rankings or Core-50 membership move under
reasonable perturbations.

What we can claim today, and should restrict ourselves to claiming, is:

- the constants are **fixed protocol anchors**, not values fit to maximize the
  performance of the reported algorithm panel;
- the protocol reports **per-axis values separately**, so the reader can inspect
  OOD-G, SYM-F, EFF, ROBU, and STAB without accepting a hidden total score;
- the use of **absolute mappings** means that adding a new algorithm does not
  rescale previously reported scores.

What we should **not** claim without further evidence is that close ranks are
insensitive to all reasonable perturbations of these constants.

`[AUTHOR CONFIRM BEFORE POSTING]` If a bounded sensitivity analysis is completed
before the rebuttal deadline, we should report only the exact results that are
actually run. Otherwise, the safe response is to acknowledge this as a current
limitation and position the constants as transparent protocol choices rather
than fully validated invariants.

## 3. LLM contamination

We agree that the current paper does **not** control contamination strongly
enough to justify a claim about uncontaminated LLM reasoning. Withholding
physical-background descriptions reduces semantic assistance, but it does not
prevent the possibility that a model has already seen standard SR benchmark
families or exact formulas.

The strongest accurate statement we can make is more limited:

- **LLM-SR** is used only in the **initial dual-probe stage**, where it helps
  identify informative tasks and disagreement patterns;
- the **final Core-50 construction** is performed from full-664 responses of the
  non-LLM **Probe-4** panel;
- nevertheless, the initial LLM probe can still **indirectly** influence which
  downstream panel or task pool looks informative.

So contamination remains a **current threat to validity**, not just future work.
The present results support only performance under the recorded prompt/context
contract on the released task families. They do **not** establish that the
observed LLM behavior reflects pure reasoning rather than some mixture of
reasoning and memorization.

A stronger claim would require a dedicated control, such as procedurally novel
formula families or an explicit post-cutoff evaluation set. We do not have that
control in the current submission, and the rebuttal should say so plainly.

## 4. RAG-SR collapse and whether this is an integration bug

We agree that the paper should not leave the impression that RAG-SR is simply
"zero on every axis" without diagnosis. The diagnostic appendix already shows
that this is **not** primarily a missing-output case.

On the clean Core-50 leaderboard, RAG-SR has **250 runs**, of which:

- **8** are timed out;
- **9** are metric-incomplete;
- but the dominant problem is **poor numerical quality**, not total failure to
  emit results.

At the dataset level, the diagnostic summary shows:

- **42/50** datasets have five-seed median OOD log-NMSE **above 0**;
- **27/50** datasets are **above 2**;
- **EFF = 3.2**, the weakest efficiency profile in the panel.

So under the **current wrapper and context protocol**, RAG-SR mostly exhibits
**numerical underfitting / weak extrapolation**, not blanket no-output failure.
That is exactly why the diagnostic appendix phrases the issue as retrieval or
generation mismatch under the released contract, rather than declaring the
method universally non-functional.

We agree the main paper should have summarized this diagnosis more clearly:
RAG-SR’s last-place ranking is a meaningful benchmark outcome under the current
integration and prompt contract, but it should not be rhetorically inflated
into a claim that the underlying method family is universally invalid.

## 5. Single 1-hour budget and fairness across paradigms

We agree that a single 1-hour wall-clock budget is not "fair" in every
possible sense. It is **not** equal-FLOP, equal-dollar, or equal-hyperparameter
search fairness. The intended claim is narrower: it is a shared, reproducible,
user-facing **evaluation contract** across heterogeneous paradigms.

The reason we think this contract is still informative is that the benchmark
does not rely only on final-time quality. For snapshot-capable methods, the
runner stores **minute-level best-so-far traces**, and the protocol reports an
explicit **EFF** axis based on anytime behavior. That means a method that finds
useful expressions quickly can be distinguished from one that needs nearly the
entire hour.

Still, we agree with your broader point: some rankings could change under a
different budget horizon. So the paper should present the 1-hour leaderboard as
one **standardized operating point**, not as a universal budget-free ordering of
all SR paradigms. A budget sweep or compute-normalized track would be a
valuable complement, but it is not part of the present submission.

## Closing

Your review usefully separates what the current paper already establishes from
what it does not yet establish. We think the new held-out experiment materially
improves the Core-50 validation story for **numerical** ranking preservation,
but we also agree with three important boundaries: the current submission lacks
a full sensitivity analysis for protocol constants, lacks a contamination
control for LLM-based methods, and uses a 1-hour wall-clock contract rather
than a compute-equivalent fairness notion. Stated with those boundaries,
we believe the benchmark still makes a meaningful contribution as a unified,
auditable, and extensible SR evaluation substrate.
