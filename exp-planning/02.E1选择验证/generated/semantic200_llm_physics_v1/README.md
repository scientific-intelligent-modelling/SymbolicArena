# semantic200_llm_physics_v1

- seed: 1314
- workers_per_host: 50
- queue: llmsr on iaaccn22/iaaccn23, then drsr on iaaccn22/iaaccn23
- prompt policy: x0/x1/.../y prompt variables with physical metadata semantics
- model split: iaaccn22 uses Meta-Llama-3.1-8B-Instruct; iaaccn23 uses Meta-Llama-3.1-8B-Instruct-Turbo
- launch guard: export CONFIRM_SEMANTIC200_LLM_PHYSICS=semantic200_llm_physics_v1 only after explicit user confirmation

- iaaccn22: 100 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct`, config: `benchmark_llm_deepinfra_llama31_8b.config`
- iaaccn23: 100 datasets, model: `deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, config: `benchmark_llm_deepinfra_llama31_8b_turbo.config`
