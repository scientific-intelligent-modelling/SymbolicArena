# Core50 LLMSR/DRSR model-split rerun

- Batch: `core50_llmsr_drsr_modelsplit_rerun_20260503`
- Datasets: Core50, 50 datasets.
- Algorithms: `llmsr`, `drsr`.
- Seeds: `0,1,2,3,4`.
- Budget: 3600 seconds per run.
- Semantics: `inject_prompt_semantics=true`, `canonical_prompt_variables=true`.
- Model split: exactly half tasks use base model and half use turbo model.
- Host mapping: `iaaccn23=base`, `iaaccn24=turbo`.
- Concurrency: each host runs 10 launchers x 10 workers = 100 concurrent tasks.

Task count: 2 algorithms x 50 datasets x 5 seeds = 500 tasks; base=250, turbo=250.
