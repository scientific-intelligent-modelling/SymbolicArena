# Formula Quality Audit Protocol

## Scope

- Audit all 50 frozen Ground Truth simplifications.
- Audit 1000 unique final prediction formulas sampled from 6749 auditable runs.
- Record the single non-auditable `noise005` run in a separate exclusion manifest.
- Force all simplification fallbacks and prior `equivalent` or `undetermined`
  decisions into the sample, then deterministically fill by
  algorithm-condition-seed strata while balancing dataset coverage.
- Treat this as a risk-targeted quality audit. Raw sample rates are not unbiased
  population estimates.

## Blind Review

- Use direct, non-streaming Anthropic Messages API calls to `claude-opus-5` with
  `xhigh` effort. Do not invoke Claude Code.
- Send only formulas, variables, allowed functions, domain assumptions, review
  scope, and opaque integrity hashes.
- Keep algorithm, dataset, condition, seed, metrics, source paths, prior
  simplification outcomes, and prior equivalence decisions outside the model
  prompt.
- Validate every response against the frozen JSON schema and the independent
  Python contract before freezing it.

## Review Rounds And Budget

- Round 1 has exactly 1050 logical tasks: 50 Ground Truth and 1000 predictions.
- Each Round 1 task may make at most two physical attempts to recover a malformed
  or transiently failed API response.
- Trigger one independent Round 2 judgment for unavailable, uncertain,
  low-confidence, simplification-conflicting, or prior-decision-conflicting
  results.
- Round 2 receives neither the prior pipeline verdict nor the Round 1 answer.
- A Round 2 response rejected only by transport or strict output validation may
  receive one `retry1` attempt with the same mathematical request and a tighter
  formatting prompt. This is a failed-delivery retry, not a third judgment.
- Count every HTTP attempt, including failures, against the hard global limit of
  1575 physical requests. If all required Round 2 judgments do not fit, fail the
  audit rather than silently selecting a subset.

## Local Verification

- Recheck every Round 2 trigger with the repository's safe symbolic parser,
  bounded symbolic proof, and deterministic numerical probes.
- For simplification, a zero symbolic difference proves equality only on the
  common domain. It cannot override a model-supported domain mismatch; only
  artifact identity proves exact preservation of definedness locally.
- Keep local decisions and compact proof evidence separate from both model
  rounds.
- Recompute final severity offline. Model-reported severity is evidence only.
- Treat prior `unable` or `undetermined` outcomes as abstentions. A determinate
  audit result that resolves an abstention is coverage recovery, not a prior
  decision conflict.

## Persistent Artifacts

The formal audit root is:

`AAAI_experiments/stage5_metric_calculation_0831/audits/formula_quality_1000_0903_v2`

It contains immutable plans, sample and exclusion manifests, API attempt records,
frozen model results, local evidence, comparisons, and final reports. Input,
prompt, schema, request, response, and result hashes provide replayable bindings.
