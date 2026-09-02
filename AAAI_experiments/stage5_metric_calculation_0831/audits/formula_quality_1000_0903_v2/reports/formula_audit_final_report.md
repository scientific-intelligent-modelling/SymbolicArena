# Formula Audit Final Report

## Summary

- Final records: 1050
- Records with issues: 172
- Severity: `{"critical":12,"major":18,"minor":1,"pass":920,"undetermined":99}`
- Physical attempts: 1505 / 1575
- Priced cost (CNY): 54.763836
- Cache creation tokens (unpriced): 0

## Attempt Accounting

| Source | Attempts | Input | Output | Cache read | Cache creation | Cost CNY |
|---|---:|---:|---:|---:|---:|---:|
| external | 28 | 49338 | 20773 | 0 | 0 | 0.459609 |
| round1 | 1230 | 3659860 | 1929403 | 0 | 0 | 39.920625 |
| round2 | 195 | 1330007 | 458199 | 0 | 0 | 10.863006 |
| round2_retry | 52 | 462137 | 142279 | 0 | 0 | 3.520596 |

## Resolution Policy

Local deterministic counterexamples or strict symbolic proofs take precedence. Otherwise a successful round 2 retry takes precedence over the original round 2, which takes precedence over round 1. Without determinate evidence, the final decision is undetermined. Model-reported severity is never used.

## Dimension Overview

| Dimension | Value | Records | Issues |
|---|---|---:|---:|
| algorithm | QLattice | 62 | 1 |
| algorithm | drsr | 61 | 14 |
| algorithm | dso | 63 | 10 |
| algorithm | e2esr | 61 | 2 |
| algorithm | fepysr | 62 | 1 |
| algorithm | gplearn | 134 | 107 |
| algorithm | iMCTS | 61 | 8 |
| algorithm | jaxsr | 61 | 3 |
| algorithm | llmsr | 62 | 5 |
| algorithm | not_applicable | 50 | 1 |
| algorithm | pyoperon | 62 | 2 |
| algorithm | pysr | 65 | 4 |
| algorithm | ragsr | 61 | 1 |
| algorithm | symbolfit | 62 | 5 |
| algorithm | tpsr | 62 | 2 |
| algorithm | udsr | 61 | 6 |
| condition | clean | 386 | 77 |
| condition | noise001 | 334 | 50 |
| condition | noise005 | 330 | 45 |
| seed | 520 | 332 | 54 |
| seed | 521 | 336 | 62 |
| seed | 522 | 332 | 55 |
| seed | not_applicable | 50 | 1 |
| audit_scope | ground_truth_simplification | 50 | 1 |
| audit_scope | prediction_formula | 1000 | 171 |
