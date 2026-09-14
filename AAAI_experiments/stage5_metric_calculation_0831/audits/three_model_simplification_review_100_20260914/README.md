# Three-Model Review of 100 Opus5 Simplification Records

## Scope

- Frozen sample seed: `20260914`
- Sample size: 100 unique context-bound review items
- Models: `kimi-k3`, `glm-5.2`, `gpt-5.6-sol`
- Configured concurrency: 32 per model, at most 96 total
- Logical model-item tasks: 300
- Shared physical request cap: 450
- Physical requests used: 326
- Stream: false

The sample contains 70 changed prediction simplifications, 20 unchanged prediction controls, 5 changed Ground Truth records, and 5 unchanged Ground Truth controls. The 70 changed prediction records cover all 45 algorithm-condition cells. This is a risk-enriched smoke audit, not an unbiased estimate of the full release pass rate.

## Completion

| Model | Completed | Pass | Fail | Unavailable | Tasks retried |
|---|---:|---:|---:|---:|---:|
| kimi-k3 | 97 | 92 | 5 | 3 | 10 |
| glm-5.2 | 94 | 83 | 11 | 6 | 16 |
| gpt-5.6-sol | 100 | 86 | 14 | 0 | 0 |

Kimi's three terminal failures were timeouts. GLM's six terminal failures followed HTTP-200 responses that did not satisfy the strict response parser/schema after one retry. A retry is transport recovery only and never creates an additional vote.

## Consensus

| Consensus | Count |
|---|---:|
| accepted_unanimous | 75 |
| accepted_with_dissent | 15 |
| rejected | 9 |
| unresolved | 1 |

By sample group:

| Sample group | Items | Unanimous pass | Majority pass total | Rejected | Unresolved |
|---|---:|---:|---:|---:|---:|
| prediction_simplified | 70 | 46 | 60 | 9 | 1 |
| prediction_unchanged | 20 | 19 | 20 | 0 | 0 |
| ground_truth_simplified | 5 | 5 | 5 | 0 | 0 |
| ground_truth_unchanged | 5 | 5 | 5 | 0 | 0 |

The majority acceptance rate among the 70 changed predictions is `60/70 = 85.71%`; strict unanimous acceptance is `46/70 = 65.71%`. These are results for the frozen risk-enriched sample only.

All three votes were available for 91 items. Pairwise agreement was:

- Kimi-K3 vs GLM-5.2: `81/91 = 89.01%`
- Kimi-K3 vs GPT-5.6-sol: `91/97 = 93.81%`
- GLM-5.2 vs GPT-5.6-sol: `81/94 = 86.17%`
- Binary Fleiss' kappa on the 91 complete triples: `0.3835`

The relatively modest kappa despite high raw agreement is caused in part by the strong prevalence of pass judgments. It should be reported as inter-model agreement, not human inter-rater reliability or model accuracy.

## Usage

| Model | Physical attempts | Input tokens | Output tokens | Total tokens | Opus-tariff reference cost (CNY) |
|---|---:|---:|---:|---:|---:|
| kimi-k3 | 110 | 95,312 | 189,411 | 302,739 | 3.127101 |
| glm-5.2 | 116 | 93,498 | 32,264 | 125,762 | 0.764454 |
| gpt-5.6-sol | 100 | 121,095 | 37,830 | 158,925 | 0.930735 |
| Total | 326 | 309,905 | 259,505 | 587,426 | 4.822290 |

The cost column uses the previously supplied Opus5 tariff only as a conservative reference. It is not the actual invoice for these three models.

## Interpretation

This run demonstrates that all three endpoints can review real frozen formulas at the requested concurrency. GPT completed all tasks without retry; Kimi and GLM need stronger timeout/schema recovery before a full audit. The nine majority-rejected changed simplifications and the one unresolved item must be inspected individually before using this exercise as validation evidence for Opus5 simplification quality.

Primary artifacts are `sample.jsonl`, `consensus.csv`, `disagreements.csv`, `cost_ledger.csv`, `summary.json`, and the per-model task records. `SHA256SUMS` binds the full audit directory.
