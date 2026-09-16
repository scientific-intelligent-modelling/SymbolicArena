# SymbolFit clean terminal Opus5 retry status

Date: 2026-09-16. This is a partial post-processing attempt, not a formal six-axis release.

## Frozen inputs

- 150 valid minute-180 SymbolFit clean endpoints, including the passed strict rerun of `symbolfit_s521_clean_g0039`.
- All 150 terminal expressions have valid canonical artifacts. The strict rerun endpoint differs from its `result.json` final expression; the endpoint was selected.
- The 150 prediction-simplification tasks have zero exact evaluation-key or input-expression matches in the 2026-09-14 published SymbolFit clean cache.
- `manifest.json`, `terminal_inventory.csv`, `symbolfit_clean_terminal_freeze.jsonl.gz`, `pred_simplify_plan.jsonl`, and `prediction_cache_match.csv` contain the frozen preflight outputs.

## API attempts

- `~/.claude/settings.json` currently targets MiniMax-M3, not Opus5. No request was sent through it.
- Opus5 requests used the Routify channel configured in `~/.claude/settings-jyh.json`; yapi was not used.
- The first one-task Opus5 canary succeeded and was frozen in `pred_api/frozen/`; reported cost: CNY 0.034515.
- The subsequent batch encountered repeated Routify `AllModelsFailed` responses and a circuit breaker. It was stopped. `pred_api/full_report.json` records the failed run; the state database has 1 frozen task, 17 exhausted tasks, 130 pending tasks, and 2 stale running leases. These are database states, not live processes.
- A fresh, single-concurrency retry of a previously failed task ended in an API read timeout after about 319 seconds, with no model result or reported cost. Evidence is in `pred_api_retry_canary/`.
- A short direct Opus5 diagnostic request returned HTTP 401. This establishes that the channel is currently unusable from this machine; it does not, by itself, identify whether the user credential or a provider-side upstream credential is at fault.

The attempted plan was preserved as `pred_api/attempted_plan_before_provenance_fix.jsonl`. The current `pred_simplify_plan.jsonl` differs only in the strict rerun's source provenance: its batch is identified explicitly and its unknown host is not fabricated. Do not resume the old state database against the current plan without a plan/evaluation-key binding audit.

## Remaining work

1. Restore a working Opus5 Anthropic-compatible endpoint and verify it with a short single-request probe.
2. Reuse the one successful frozen response only after checking its evaluation key against the current plan; then run the remaining prediction simplifications with a new, auditable state database.
3. Build equivalence and three-seed structure tasks from the frozen current simplifications, using the fixed Ground Truth references.
4. Export per-run and per-pair evidence with input formulas, dependencies, model settings, and unresolved records. Do not mark `formal_ready` until all bindings close.

No algorithm experiment was rerun, no gplearn post-processing was changed, and no API worker remains active.
