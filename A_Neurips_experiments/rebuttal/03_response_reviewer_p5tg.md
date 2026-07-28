# Response to Reviewer p5tG

> Long draft, not paste-ready.

Thank you for the constructive review and for recognizing the value of a unified
SR evaluation substrate. We agree that the original submission did not make the
Core-50/full-664 relationship, the leaderboard reading rule, and the intended
scope under noise as explicit as they should have been. Below we answer each of
your concerns directly and narrow the claims where the current evidence does not
support a stronger statement.

## 1. Core-50 representativeness and correlation to Full-664

You are right: MAE measures the size of the score error, but it does not show
whether Core-50 preserves the relationship among algorithms. We therefore made
a direct matched-run comparison. For each method, we computed the Core-50
result by restricting its completed Full-664 runs to the frozen 50 tasks. The
comparison uses the clean leaderboard's ranking metric, seed-median penalized
OOD log-NMSE, where lower is better.

| Algorithm | Full-664 OOD | Rank | Core-50 OOD | Rank |
|---|---:|---:|---:|---:|
| uDSR | -5.196 | 1 | -5.431 | 1 |
| iMCTS | -4.252 | 2 | -4.284 | 2 |
| DSO | -2.095 | 3 | -1.848 | 6 |
| FePySR | -1.794 | 4 | -2.187 | 3 |
| SymbolFit | -1.547 | 5 | -2.114 | 4 |
| JAXSR | -1.244 | 6 | -2.016 | 5 |
| PySR | -0.751 | 7 | -0.115 | 7 |
| LLM-SR | 0.797 | 8 | 2.201 | 9 |
| PyOperon | 0.854 | 9 | 0.820 | 8 |

The continuous OOD scores have **Pearson `r=0.967`** (exact permutation
`p=0.000154`), while the method ranks have **Spearman `rho=0.883`**
(`p=0.003075`). These two numbers answer different questions: Pearson measures
agreement of the actual OOD values, whereas Spearman measures agreement of the
ordering. In concrete terms, **32/36 pairwise method orderings (88.9%)** are
preserved. The top two methods remain first and second; the largest change is
DSO moving from rank 3 to rank 6, while the same four methods occupy ranks 3--6.

For the ID/OOD aggregate score mentioned in the question, the corresponding
correlations are **Pearson `r=0.975`** and **Spearman `rho=0.867`**. Thus the
correlation analysis supports the specific claim that Core-50 preserves broad
Full-664 numerical conclusions; it does not claim that every adjacent rank is
identical. Of the nine methods, FePySR, JAXSR, and SymbolFit are genuinely
post-submission held-out methods and keep the same internal OOD order on both
task sets. We treat that held-out result as supporting evidence rather than a
standalone significance test because it contains only three methods. PySR and
LLM-SR currently have one Full-664 seed, which we will state explicitly.

## 2. How to read the leaderboard and identify the best algorithm

We agree that the table becomes hard to read if the reader does not know which
column defines rank. In our protocol, the ranking key is **penalized mean OOD
log-NMSE** (lower is better). The remaining columns are intentionally *not*
collapsed into that rank, because they are meant to expose dimensions that do
not always align with numerical extrapolation quality.

Concretely, the clean Core-50 leaderboard in the submission already shows that:

- **uDSR** ranks first numerically by OOD log-NMSE;
- **PySR** is stronger on symbolic recovery, with the highest SYM-F and exact
  equivalence rate among the 12 methods;
- these are therefore not the same notion of “best.”

This is not a bug in the table; it is one of the paper’s main findings. A
symbolic regressor can achieve very low numerical error while still failing to
recover the ground-truth formula exactly. One concrete example from our
diagnostic appendix is the repeated **Nguyen-9** shortcut case: the target is
`sin(x0) + sin(x1^2)`, while several low-NMSE predictions collapse to
`sin(x0) + sin(x0^2)`. Numerically the error can remain extremely small, but
symbolically the expression is wrong. That is precisely why the protocol keeps
numerical quality and symbolic fidelity separate.

So if a reader wants a **single numerical winner**, the paper should tell them
explicitly to look at penalized OOD log-NMSE. If they want a **symbolic-recovery
winner**, they should look at SYM-F / exact equivalence. The present version
did not make that reading rule prominent enough, and we agree this should be
clarified in the main text and caption.

## 3. Whether small score differences are meaningful

We agree that not every small difference should be interpreted as meaningful.
The current submission does provide **dataset-bootstrap 95% confidence
intervals** for the hexagon scores, and the Core-50 validation also reports
aggregate preservation rather than just point estimates. However, it does **not**
present a complete pairwise significance analysis for every close leaderboard
gap, and it should not imply more certainty than the statistics support.
For example, the reported HexaScore intervals for iMCTS
(`48.7 [40.5, 56.1]`) and uDSR (`47.1 [40.2, 53.8]`) overlap substantially.
We therefore do not treat their 1.6-point difference as a resolved pairwise
ordering without a paired difference analysis.

Our revised interpretation is therefore:

- large and repeated separations, especially when they remain visible across
  both Core-50 and Full-664, are more trustworthy;
- small gaps between near-tied methods should be treated cautiously;
- the new held-out experiment supports the stability of the **broad numerical
  ordering**, but not every fine symbolic ordering.

This is also why the rebuttal evidence above is useful: it strengthens the
claim that Core-50 preserves reservoir-level conclusions at the level of
overall **ranking structure**, without claiming that every adjacent gap is
individually significant.

## 4. Why a 1-hour budget, and whether it is fair across paradigms

The 1-hour budget is a **reproducible wall-clock contract**, not a claim of
equal FLOPs, equal API cost, or equal optimization opportunity across GP, tree
search, neural, and LLM-assisted paradigms. The reason we chose it is practical:
it asks what a fixed, versioned method configuration returns within the same
user-facing evaluation horizon.

To reduce the risk that the leaderboard becomes a pure final-time snapshot, the
benchmark also records **minute-level best-so-far traces** and derives an
explicit **EFF** axis from the quality-time AUC. That means a method is not
judged only by its final 60-minute point; methods that find useful expressions
earlier receive separate credit.

We agree that a budget sweep would be informative, and the current
infrastructure is designed so that such a view is possible. But the present
paper’s fairness claim should be stated more narrowly: it is fair only in the
sense of a **shared wall-clock evaluation contract**, not in the sense of
compute-equivalent resource normalization.

## 5. Scope under noisy scientific data

We also agree that the scope under noise should be stated more carefully. The
benchmark is built from **formula-backed symbolic-regression tasks** with
standardized clean train/validation/ID/OOD splits, and the main clean
leaderboard is therefore best interpreted as a controlled evaluation of
symbolic-regression behavior under that contract.

The paper does include a **noisy-train / clean-test robustness extension** on
Core-50 at standardized noise levels `0.01`, `0.05`, and `0.10`, but that is
still a controlled protocol rather than a claim that the benchmark directly
captures all properties of real noisy scientific data. In particular, we do
**not** claim that SymbolicArena fully represents arbitrary measurement noise,
unobserved confounding, domain shift beyond the released families, or black-box
scientific pipelines.

So the proper claim is:

- **yes**, SymbolicArena can evaluate robustness under controlled noise within a
  standardized formula-backed benchmark;
- **no**, this does not by itself validate generalization to all real noisy
  scientific datasets.

## Closing

Your questions point to the right clarifications: we should make the ranking
rule explicit, report direct Core-50/Full-664 correlation rather than relying
mostly on MAE, and state more carefully what the 1-hour and noise conclusions
do and do not mean. The new held-out-algorithm experiment substantially
strengthens the representativeness case for **numerical** ranking preservation,
while also making us narrow the symbolic claim where the evidence is weaker.
