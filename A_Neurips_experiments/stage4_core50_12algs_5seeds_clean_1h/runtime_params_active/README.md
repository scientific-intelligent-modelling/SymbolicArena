# Core-50 12 算法正式 1h 有效搜索参数

本目录只用于正式 benchmark/Core-50 路径，目标是让历史上提前结束的算法持续做有效搜索到接近 `timeout_in_seconds=3600`，不影响 `check/*`、smoke 和本地快速验证默认参数。

使用方式：正式队列调度器传入本目录作为参数根，例如：

```bash
python check/run_e1_candidate200_12alg_load_queue.py \
  --source-csv frozen-results/neurips26/final-experiments/03_core50_12alg_final/datasets_to_run.csv \
  --params-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_1h_effective \
  --queue-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_queue_1h_effective \
  --expected-rows 50
```

本目录包含完整 12 算法参数，便于直接作为 `--params-root` 使用：`pyoperon`、`qlattice`、`e2esr`、`tpsr`、`dso`、`udsr`、`imcts`、`ragsr` 使用 1h 有效搜索增强参数；`drsr`、`gplearn`、`llmsr`、`pysr` 沿用论文 artifact 中已基本跑满 1h 的原正式参数。

`runtime_budget_plan.csv` 记录每个算法的历史 median runtime、提前结束原因、改动和预期行为。
