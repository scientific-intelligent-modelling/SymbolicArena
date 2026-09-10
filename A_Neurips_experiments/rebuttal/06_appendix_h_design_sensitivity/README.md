# Appendix H design-parameter sensitivity

This directory contains fail-closed runners for the two benchmark-construction
experiments described in Appendix H.  They reuse frozen construction inputs and
do not launch symbolic-regression training, use GPUs, or call an LLM API.

## Formal contract

- Probe-4: draw 5,000 independent local perturbations around
  `(0.20, 0.40, 0.25, 0.15)`, renormalize each draw, and rescore every frozen
  feasible panel from its `H/F/C/V` components.
- Core50: draw 5,000 independent local perturbations around
  `(0.45, 0.35, 0.20)`, renormalize each draw, and replay the same constrained
  selector for every configuration.
- The unnormalized perturbation interval is `default +/- 0.05` for every
  weight.  Raw and normalized weights are both retained.
- The nominal configuration is an audit gate and is not counted among the
  5,000 random configurations.

The runners intentionally reject substitute inputs.  In particular,
`probe4_combo_scores_nmse_only.csv` does not contain the final Appendix H
`H/F/C/V` components and cannot be used as the Probe-4 input.

## Commands

Probe-4 requires a frozen candidate table with exactly these columns:

```text
panel,health,panel_fidelity,complementarity,coverage
```

Run it with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/run_probe4_sensitivity.py \
  --candidate-components /path/to/frozen_probe4_hfcv_components.csv \
  --output-dir A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/results/probe4
```

Run the Core50 gate and experiment with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/run_core50_sensitivity.py \
  --output A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/results/core50
```

Both commands default to 5,000 configurations.  Use a fresh output directory
for a new formal run.  Core50 supports checkpoint repair and resume within the
same contracted run.

After both experiments pass, validate the full row-level contract and render
Figure H.1 with:

```bash
python A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/plot_appendix_h.py \
  --probe4-results RESULTS/probe4/probe4_perturbation_results.csv \
  --core50-config-results RESULTS/core50/joint_config_results.csv \
  --core50-memberships RESULTS/core50/selected_memberships.csv \
  --core50-frequency RESULTS/core50/membership_frequency.csv \
  --output-dir RESULTS/figure_h1
```

The plotter requires all 5,000 Probe-4 rows, all 5,000 Core50 configuration
rows, all 250,000 Core50 membership rows, and the full 664-task frequency
table.  It refuses partial inputs.

## Reproducibility gates

Probe-4 must reproduce the historical nominal audit before writing results:

- strict best panel: `dso;pyoperon;qlattice;udsr`, reported score `0.6570`;
- selected panel: `dso;imcts;pyoperon;udsr`, rank 2, reported score `0.6512`.

Core50 must reproduce all 50 frozen task IDs at the nominal weights.  A mismatch
writes only `audit.json` and exits with status 2.  No partial sensitivity table
is presented as a formal result.

## Current status

The implementation and focused tests are complete.  The current repository
does not contain the frozen Probe-4 `H/F/C/V` component table.  The available
NMSE-only table is rejected by design.

The current Core50 reconstruction also fails the nominal gate: it overlaps the
frozen manifest on 23 of 50 tasks (Jaccard `0.2987012987`).  The exact historical
selector, `ood_type` materialization, limited-quota caps, and search/tie-break
contract must be recovered before the formal 5,000-configuration Core50 run can
be started.

The recorded failed-gate evidence is in
`runs/core50_strict_baseline_attempt_v1/audit.json`.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pytest -q A_Neurips_experiments/rebuttal/06_appendix_h_design_sensitivity/test_*.py
```
