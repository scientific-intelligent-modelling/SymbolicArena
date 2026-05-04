# Core50 DRSR LLM budget rerun, seeds 5/6/7

- Batch prefix: `core50_drsr_llmbudget_seed567_20260504`
- Purpose: rerun original clean Core-50 DRSR with the updated DRSR progress logging that records LLM token/time budget.
- Datasets: Core-50, 50 datasets.
- Algorithm: `drsr` only.
- Seeds: `5,6,7`.
- Budget: 3600 seconds per run.
- Semantics: `inject_prompt_semantics=true`, `canonical_prompt_variables=true`.
- Noise: disabled.
- Model split: stable half between DeepInfra base and turbo Llama-3.1-8B.
- Global LLM limits: `base <= 80`, `turbo <= 80`.

Task count: 50 datasets x 3 seeds x 1 algorithm = 150 tasks.
