Long draft, not paste-ready. Character limit and final wording still need author compression.

# Draft response to Reviewer p5tG

Thank you for the positive assessment and for the concrete requests on interpretability, statistical meaning, and scope. We agree that the paper should make the benchmark’s representativeness and usage boundaries easier to read directly from the main text.

## Concern summary

Your main questions are: (1) whether Core-50 is truly representative of the full 664-task reservoir, ideally via correlation analysis; (2) how to interpret a multi-metric leaderboard and select a “best” method; (3) whether small score differences are statistically meaningful; (4) whether a one-hour budget is fair across method types; (5) how to interpret mismatches between numerical accuracy and symbolic recovery; and (6) how far conclusions should generalize to noisy scientific data.

## 1. Core-50 vs Full-664 correlation

We agree that aggregate-score MAE alone is not the most intuitive way to express representativeness. To address this directly, after submission we ran three additional algorithms that were not part of the original construction pipeline (FePySR, JAXSR, SymbolFit) on the Full-664 clean protocol and merged them with four existing Stage-3 algorithms into a 7-algorithm comparison.

This gives a direct Core-50 vs Full-664 correlation check:

- for mean OOD performance across the 7 algorithms, Pearson correlation is `0.989456061` and Spearman correlation is `0.892857143`;
- for the three held-out algorithms alone, the OOD ordering is preserved exactly (Spearman `1.0`, Pearson `0.815541684`);
- for SYM-F across the 7 algorithms, Pearson is `0.971269484` and Spearman is `0.857142857`.

At the same time, the three held-out algorithms do **not** show stable fine-grained SYM-F ordering by themselves (Spearman `-0.5`), and their Full-664 symbolic-fidelity scores are close. So we think the strongest supported claim is that Core-50 is a strong low-cost proxy for the reservoir-level numerical conclusions, with good group-level preservation of symbolic-fidelity trends. We would avoid claiming that every fine-grained symbolic ordering is preserved, especially when score gaps are small.

## 2. How to read the leaderboard and choose a “best” algorithm

This is a very helpful point. Our intended design is:

- **OOD log NMSE** is the primary ranking key in the clean leaderboard, because OOD generalization is the main numerical criterion;
- **SYM-F / Exact / TreeSim** are not secondary decorations, but explicit diagnostics showing whether strong numerical performance also reflects faithful recovery of the underlying formula;
- the six-axis protocol is meant to show that there is not always one universally best method across all dimensions.

So if a reader wants a single default answer, the clean leaderboard rank is determined by penalized OOD log NMSE. But if the reader wants a method that is also strong in symbolic recovery, efficiency, or stability, the rank alone is intentionally insufficient. We agree that the paper should explain this more directly and visually highlight the ranking key.

## 3. Are small differences meaningful?

We agree that the current draft should be more cautious here. The paper already reports dataset-bootstrap 95% confidence intervals for the formal hexagon scores, and these intervals help indicate when fine-grained score gaps should not be over-interpreted. But we agree that the current paper does not provide a full significance story for every pairwise method comparison.

So the rebuttal-safe position is:

- broad separations and repeated diagnostic patterns are meaningful;
- very small score differences should not be treated as decisive without additional statistical support;
- we should revise the prose so that the paper emphasizes robust trends rather than implying that every nearby ranking swap is important.

## 4. One-hour budget and fairness

We agree that different SR paradigms have very different search dynamics, and a one-hour wall-clock budget is not a perfect notion of compute fairness. Our intention was to define a shared execution contract under which all methods receive the same wall-clock budget, the same dataset contract, the same result schema, and the same minute-level logging.

So the strongest defensible claim is not “one hour is the uniquely fair budget,” but rather:

- one hour is a standardized and auditable benchmark contract;
- the minute-level traces preserve the information needed for future anytime or budget-sweep analyses;
- conclusions should be read under this fixed-budget protocol, not as an absolute statement about all possible budget settings.

## 5. Numerical accuracy vs symbolic fidelity mismatch

We agree that this deserves a concrete example. A representative case already exposed by the formal symbolic-fidelity audit is **Nguyen-9**. The ground-truth expression is `sin(x0) + sin(x1^2)`, while several low-error predictions simplify to `sin(x0) + sin(x0^2)`. These expressions can achieve near-machine-precision numerical error on the sampled splits, but they are not symbolically equivalent to the true formula and therefore receive much lower symbolic-fidelity credit.

This is exactly why we report symbolic fidelity separately from numerical error: a method can interpolate and even extrapolate well on the benchmark splits without recovering the correct underlying structure.

## 6. Scope with respect to noisy scientific data

We agree that the scope should be stated more narrowly. The main benchmark is a clean ground-truth symbolic-regression benchmark. Noisy training is treated separately through the ROBU extension, which uses standardized synthetic label noise. This does **not** mean that the current paper fully characterizes real noisy scientific discovery settings.

So the correct scope statement is:

- SymbolicArena supports clean ground-truth evaluation plus a standardized noisy-training extension;
- it provides a controlled benchmark substrate for studying robustness;
- but conclusions from the clean leaderboard should not be overstated as direct claims about all real noisy scientific data.

## Closing

We appreciate these questions because they point to places where the paper can become much easier to use. The new held-out Full-664 evidence gives us a clearer and more intuitive representativeness argument; the leaderboard explanation should make the primary ranking key and the purpose of the additional axes more explicit; and the paper should more clearly separate what is established under the current protocol from what remains a scope boundary.

## Author confirmation needed before posting

- Confirm whether we want to include the exact Pearson / Spearman numbers in the rebuttal.
- Confirm whether we want to explicitly mention Nguyen-9 by name as the symbolic-mismatch example.
- Confirm whether we want to say “OOD rank is the default ranking key” or the slightly stronger “OOD rank defines the leaderboard order.”
- Confirm whether we want to promise an expanded limitations paragraph, or more conservatively say that we will clarify scope and statistical interpretation in the revision.
