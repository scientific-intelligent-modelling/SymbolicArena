# Three-Model Review of 100 Opus5 Simplification Records: Common-Domain Approximate Policy

## Scope

- Frozen sample source: strict-v1 `sample.jsonl`, SHA256 `eddc3b97bae27a79c038a50ff5bfa3dffd593696d613b084825dbb95435a0c7e`
- Sample size: 100 unique context-bound review items
- Models: `kimi-k3`, `glm-5.2`, `gpt-5.6-sol`
- Configured concurrency: 32 per model, at most 96 total
- Logical model-item tasks: 300
- Shared physical request cap: 450
- Physical requests used: 322
- Stream: false
- Prompt version: `three_model_simplification_review.common_domain_approx.v2`

The sample is identical to strict v1: 70 changed prediction simplifications, 20 unchanged prediction controls, 5 changed Ground Truth records, and 5 unchanged Ground Truth controls. The 70 changed prediction records cover all 45 algorithm-condition cells. This is a risk-enriched smoke audit, not an unbiased estimate of the full release pass rate.

## Review Policy

The reviewer compares expression values only on the common real-valued domain. Domain expansion or contraction alone is not a failure. Function values are treated as matching when

```text
abs(f(x) - g(x)) <= 1e-9 + 1e-6 * max(abs(f(x)), abs(g(x)))
```

Constants are not refitted. Variable-to-input-column identity and algorithm-native operator value semantics remain mandatory. Protected-operator rewrites fail when they change values on the common domain, even if ordinary algebra would permit the rewrite. A passing result must also be no more complex than the original.

## Completion

| Model | Completed | Pass | Fail | Unavailable | Tasks retried |
|---|---:|---:|---:|---:|---:|
| kimi-k3 | 99 | 95 | 4 | 1 | 5 |
| glm-5.2 | 92 | 88 | 4 | 8 | 17 |
| gpt-5.6-sol | 100 | 96 | 4 | 0 | 0 |

Kimi's terminal failure was a timeout after two attempts. GLM's eight terminal failures followed HTTP-200 responses that did not satisfy the response parser/schema after one retry. A retry is transport recovery only and never creates an additional vote.

## Consensus

| Consensus | Strict v1 | Relaxed v2 |
|---|---:|---:|
| accepted_unanimous | 75 | 84 |
| accepted_with_dissent | 15 | 11 |
| rejected | 9 | 4 |
| unresolved | 1 | 1 |

By sample group under relaxed v2:

| Sample group | Items | Unanimous pass | Majority pass total | Rejected | Unresolved |
|---|---:|---:|---:|---:|---:|
| prediction_simplified | 70 | 55 | 66 | 3 | 1 |
| prediction_unchanged | 20 | 19 | 19 | 1 | 0 |
| ground_truth_simplified | 5 | 5 | 5 | 0 | 0 |
| ground_truth_unchanged | 5 | 5 | 5 | 0 | 0 |

For the 70 changed predictions, majority acceptance rose from `60/70 = 85.71%` to `66/70 = 94.29%`; unanimous acceptance rose from `46/70 = 65.71%` to `55/70 = 78.57%`. These rates apply only to the frozen risk-enriched sample.

All three votes were available for 91 items. Pairwise agreement was:

- Kimi-K3 vs GLM-5.2: `85/91 = 93.41%`
- Kimi-K3 vs GPT-5.6-sol: `98/99 = 98.99%`
- GLM-5.2 vs GPT-5.6-sol: `84/92 = 91.30%`
- Binary Fleiss' kappa on the 91 complete triples: `0.3369`

The lower kappa alongside higher raw agreement is a prevalence effect: nearly all votes are passes. It must not be reported as human inter-rater reliability or as model accuracy.

## Remaining Negative Cases

The four majority-rejected items are:

- `sample_0005`: protected-division fallback behavior changed.
- `sample_0020`: `log(sqrt(x3)) -> log(x3)/2` crosses gplearn's protected-log threshold and changes values.
- `sample_0035`: multiple protected divisions were merged, changing fallback behavior.
- `sample_0075`: a nested subexpression was changed rather than merely simplified; Kimi and GLM detected it, while GPT missed it.

`sample_0065` remains unresolved because GPT found a protected-log counterexample, GLM passed it, and Kimi timed out twice. The counterexample is substantive, so this item should remain fail-closed pending another independent adjudication.

## Usage

| Model | Physical attempts | Input tokens | Output tokens | Total tokens | Opus-tariff reference cost (CNY) |
|---|---:|---:|---:|---:|---:|
| kimi-k3 | 105 | 107,402 | 223,521 | 366,522 | 3.675021 |
| glm-5.2 | 117 | 99,216 | 24,527 | 123,743 | 0.665553 |
| gpt-5.6-sol | 100 | 149,104 | 33,415 | 182,519 | 0.948537 |
| Total | 322 | 355,722 | 281,463 | 672,784 | 5.289111 |

The cost column applies the previously supplied Opus5 tariff, CNY 3 per million input tokens and CNY 15 per million output tokens, only as a reference. It is not the actual invoice for these three models.

## Artifacts

Primary artifacts are `sample.jsonl`, `consensus.csv`, `disagreements.csv`, `cost_ledger.csv`, `summary.json`, the per-model task records, and `strict_v1_to_common_domain_approx_v2.csv`. `SHA256SUMS` binds the audit directory.
