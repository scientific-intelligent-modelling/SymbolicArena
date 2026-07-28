# INTERNAL DRAFT: NOT SUBMISSION-READY WHILE THE BASELINE GATE FAILS

## Core-50 objective sensitivity under fixed Probe-4

We added an offline sensitivity analysis that reuses the completed Full-664
three-seed outputs of the fixed Probe-4 panel (DSO, PyOperon, iMCTS, and uDSR);
no new SR runs were launched. We reconstructed the stated constrained
one-for-one local search and first applied an exact baseline-reproduction gate.
The nominal reconstruction did not exactly reproduce the frozen Core-50, so we report this as a counterfactual reconstruction rather than evidence from the historical selector.

Under the local factorial perturbation, each of the three objective weights was
scaled by 0.8--1.2 and renormalized (121
configurations). The selected subsets retained at least
19/50 frozen tasks and had a minimum Jaccard
similarity of 0.754 to the reconstructed
nominal solution. Their Probe-4 rankings relative to Full-664 had minimum
Spearman rho 1.000, minimum Kendall tau
1.000, and minimum pairwise-order agreement
1.000. A broader 0.1-simplex stress
test (66 configurations) gave a minimum
reconstructed-reference Jaccard of
0.190 and minimum rank Spearman rho
1.000.

The rank result is an internal diagnostic over only the four construction
probes; it is not a held-out-algorithm validation and should not be presented as
one. Because the exact historical-membership reproduction gate currently
fails, this paragraph must not be submitted as a claim that the frozen Core-50
membership is weight-robust.

We also disclose two specification gaps rather than filling them post hoc:
`ood_type` is not materialized in the released dataset-level table, and the
paper does not give numerical limited-quota caps. We therefore exclude invented
values from the primary analysis and will clarify these implementation details
in the revision.
