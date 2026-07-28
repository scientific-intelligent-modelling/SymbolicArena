> Long-form draft for author editing. Not paste-ready.
>
> Source: Reviewer p5tG official review on June 26, 2026, modified July 23, 2026.

# Reply to Reviewer p5tG

Thank you for the constructive review and for recognizing the value of a
unified SR evaluation substrate. We agree that the original submission did not
make the Core-50 / Full-664 relationship, the leaderboard reading rule, and
the intended scope under noise as explicit as they should have been. Below we
answer each concern directly and narrow the claims where the current evidence
does not support a stronger statement.

## Main concerns addressed

This draft answers the reviewer in the same order as the main questions:

1. Core-50 versus Full-664 correlation and representativeness;
2. how to read the leaderboard and decide what "best" means;
3. how to interpret numerical quality versus symbolic quality misalignment;
4. when small score gaps are or are not meaningful;
5. why a one-hour budget was chosen; and
6. what claims are justified for noisy scientific data.

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

The continuous OOD scores have Pearson `r=0.967` (exact permutation
`p=0.000154`), while the method ranks have Spearman `rho=0.883`
(`p=0.003075`). These two numbers answer different questions: Pearson measures
agreement of the actual OOD values, whereas Spearman measures agreement of the
ordering. In concrete terms, 32/36 pairwise method orderings (88.9%) are
preserved. The top two methods remain first and second; the largest change is
DSO moving from rank 3 to rank 6, while the same four methods occupy ranks 3--6.

For the ID/OOD aggregate score mentioned in the question, the corresponding
correlations are Pearson `r=0.975` and Spearman `rho=0.867`. Thus the
correlation analysis supports the specific claim that Core-50 preserves broad
Full-664 numerical conclusions; it does not claim that every adjacent rank is
identical. Of the nine methods, FePySR, JAXSR, and SymbolFit are genuinely
post-submission held-out methods and keep the same internal OOD order on both
task sets. We treat that held-out result as supporting evidence rather than a
standalone significance test because it contains only three methods. PySR and
LLM-SR currently have one Full-664 seed, which we will state explicitly.

## 2. How to read the leaderboard and identify the best algorithm

We agree that the table is hard to read if the reader does not know which
column defines rank. In our protocol, the ranking key is penalized mean OOD
log-NMSE, where lower is better. The remaining columns are intentionally not
collapsed into that rank, because they are meant to expose dimensions that do
not always align with numerical extrapolation quality.

Concretely, the clean Core-50 leaderboard already shows that:

- uDSR ranks first numerically by OOD log-NMSE;
- PySR is stronger on symbolic recovery, with the highest SYM-F and exact
  equivalence rate among the 12 methods;
- these are therefore not the same notion of "best."

This is not a flaw in the table; it is one of the paper's main findings. A
symbolic regressor can achieve very low numerical error while still failing to
recover the ground-truth formula exactly.

## 3. Interpreting numerical / symbolic misalignment

We agree that this point should be illustrated more explicitly. A concrete case
already present in our diagnostic appendix is Nguyen-9. The target formula is

`sin(x0) + sin(x1^2)`,

while several low-NMSE predictions collapse to

`sin(x0) + sin(x0^2)`.

Numerically, the error can remain extremely small on the benchmark split, but
symbolically the expression is wrong because the second variable has been
replaced by the first. That is exactly why the protocol keeps numerical quality
and symbolic fidelity separate instead of treating low NMSE as sufficient.

## 4. Whether small score differences are meaningful

We agree that not every small difference should be interpreted as meaningful.
The current leaderboard package already reports dataset-bootstrap 95%
confidence intervals for the formal hexagon scores, but it does not present a
complete pairwise significance analysis for every close gap. So the paper
should not imply more certainty than the current statistics support. For
example, the current clean HexaScore confidence intervals for uDSR and iMCTS
overlap materially, so their small gap should be treated as suggestive rather
than as a resolved pairwise ordering.

Our revised interpretation is therefore:

- large and repeated separations, especially when they remain visible across
  both Core-50 and Full-664, are more trustworthy;
- small gaps between near-tied methods should be treated cautiously;
- the new held-out experiment supports the stability of the broad numerical
  ordering, but not every fine symbolic ordering.

## 5. Why a one-hour budget, and whether it is fair across paradigms

The one-hour budget is a reproducible wall-clock contract, not a claim of equal
FLOPs, equal API cost, or equal optimization opportunity across GP, tree
search, neural, and LLM-assisted paradigms. The reason we chose it is
practical: it asks what a fixed, versioned method configuration returns within
the same user-facing evaluation horizon.

To reduce the risk that the leaderboard becomes only a final-time snapshot, the
benchmark also records minute-level best-so-far traces and derives an explicit
EFF axis from the quality-time AUC. That means a method is not judged only by
its final 60-minute point; methods that find useful expressions earlier receive
separate credit.

We agree that a budget sweep would be informative. The current infrastructure
is designed so that such a view is possible. But the present paper's fairness
claim should be stated more narrowly: it is fair only in the sense of a shared
wall-clock evaluation contract, not in the sense of fully compute-normalized
resource matching.

## 6. Scope under noisy scientific data

We also agree that the scope under noise should be stated more carefully. The
benchmark is built from formula-backed symbolic-regression tasks with
standardized clean train/validation/ID/OOD splits, so the main clean leaderboard
is best interpreted as a controlled evaluation of symbolic-regression behavior
under that contract.

The paper does include a noisy-train / clean-test robustness extension on
Core-50 at standardized noise levels 0.01, 0.05, and 0.10, but that is still a
controlled protocol rather than a claim that the benchmark directly captures
all properties of real noisy scientific data.

## Closing

Your questions point to the right clarifications: we should make the ranking
rule explicit, report direct Core-50 / Full-664 correlation rather than
relying mostly on MAE, and state more carefully what the one-hour and noise
conclusions do and do not mean. The new held-out-algorithm experiment
substantially strengthens the representativeness case for numerical ranking
preservation, while also making us narrow the symbolic claim where the evidence
is weaker.

## Author checks before posting

- Confirm whether to include the exact held-out correlation numbers in the
  rebuttal.
- Confirm whether to keep the Nguyen-9 example in the short OpenReview version
  or reserve it for a richer appendix-style draft.
- Confirm whether to say "shared wall-clock contract" or the slightly more
  reader-friendly "standardized one-hour evaluation budget."
