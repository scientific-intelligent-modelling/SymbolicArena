# Core-50 vs full-664 rank correlation: 9 algorithms

## Primary protocol

- Metric: penalized OOD `log10(NMSE)`, clipped to `[-12, 12]`.
- Missing or invalid split metrics receive `+12`.
- Each `algorithm x dataset` is first aggregated by the median across seeds.
- Algorithm scores are then averaged across 664 datasets or the frozen Core-50.
- Lower scores are better.

This is a matched-run comparison: Core-50 is sliced directly from the same
full-664 runs. The AAAI 3-hour Core-50 results are not mixed with the 1-hour
full-664 results.

## Main result

- Pearson correlation of continuous OOD scores: `0.966685`
  (exact two-sided permutation `p=0.000154321`).
- Pearson correlation of rank vectors / Spearman rank correlation:
  `0.883333`
  (exact two-sided permutation `p=0.0030754`).
- Kendall tau-b: `0.777778`.
- Pairwise ordering agreement:
  `32/36`
  (`0.888889`).
- Full-664 OOD order: uDSR, iMCTS, DSO, FePySR, SymbolFit, JAXSR, PySR, LLM-SR, PyOperon.
- Core-50 OOD order: uDSR, iMCTS, FePySR, SymbolFit, JAXSR, DSO, PySR, PyOperon, LLM-SR.

For the aggregate numerical score mentioned by the reviewer, defined as the
equal-weight mean of penalized seed-median ID and OOD log NMSE:

- Pearson score correlation: `0.975443`.
- Spearman rank correlation: `0.866667`.

The three post-submission algorithms preserve the same internal OOD ordering
on full-664 and Core-50 (`rho=1.000000`), but `n=3` is
too small to present as a standalone definitive correlation test.

## Evidence boundary

- Probe-4 (`DSO`, `iMCTS`, `PyOperon`, `uDSR`) directly participated in the
  final construction panel.
- Probe-2 (`PySR`, `LLM-SR`) participated in earlier discovery and has only
  one seed (`1314`), while the other seven algorithms use seeds
  `520, 521, 522`.
- Only `FePySR`, `JAXSR`, and `SymbolFit` are post-submission held-out
  algorithms. Therefore the nine-algorithm result is an expanded
  representativeness check, not a fully independent nine-algorithm validation.

## Suggested rebuttal sentence

> We added a matched-run comparison on nine methods by slicing the frozen
> Core-50 directly from their full-664 results. Under the paper's
> seed-median penalized OOD log-NMSE protocol, Core-50 and full-664 have a
> Pearson score correlation of 0.967 and a Spearman rank
> correlation of 0.883, with
> 32/36 pairwise method
> orderings preserved. The three post-submission methods also retain the same
> internal ordering. We will explicitly distinguish these truly held-out
> methods from the construction probes and note that the two discovery probes
> currently have only one full-reservoir seed.
