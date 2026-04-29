# semantic200_llm_physics_v2_4host

- seed: 1314
- workers_per_host: 50
- queue: llmsr on iaaccn23/iaaccn24/iaaccn25/iaaccn26, then drsr on the same four hosts
- prompt policy: x0/x1/.../y prompt variables with physical metadata semantics
- model split: iaaccn23 and iaaccn25 use Meta-Llama-3.1-8B-Instruct; iaaccn24 and iaaccn26 use Meta-Llama-3.1-8B-Instruct-Turbo
- launch guard: export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v2_4host only after explicit user confirmation

- iaaccn23: 50 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct`, config: `benchmark_llm_deepinfra_llama31_8b.config`
- iaaccn24: 50 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, config: `benchmark_llm_deepinfra_llama31_8b_turbo.config`
- iaaccn25: 50 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct`, config: `benchmark_llm_deepinfra_llama31_8b.config`
- iaaccn26: 50 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, config: `benchmark_llm_deepinfra_llama31_8b_turbo.config`
