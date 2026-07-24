# Core-50 Stage4 路径说明

生成时间：2026-07-25

本文件只描述 `A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h`
这一层的来源，不改写 Stage1-3。

## 本地整理目录

```text
/home/family/workplace/scientific-intelligent-modelling/A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h
```

## 上游冻结包

冻结来源目录：

```text
/home/family/workplace/scientific-intelligent-modelling/frozen-results/neurips26/final-experiments/03_core50_12alg_final
```

本次完整复制了其中这些真实目录或文件：

```text
README.md                     -> 已改写为当前整理包说明
core50_manifest/
collected_results/
formal_analysis/
hexagon/
noise_robustness/
paper_tables/
runtime_params_1h_effective/
runtime_params_24h_effective/
runtime_params_3h_effective/
runtime_params_active/
runtime_params_active.README.md
runtime_queue_4alg_budgetfix_3seed_20260523-005312/
runtime_queue_5alg_budgetfix_smoke_20260523/
runtime_queue_core50_12alg_5x3h_20260523-215546/
runtime_queue_tpsr_short4_20260522/
```

## 被刻意省略的软链

冻结包里原本存在：

```text
raw_results/core50_12alg_5seed_all_20260502-065700
  -> ../../../../../experiments/core50_12alg_5seed_all_20260502-065700
```

这个软链离开原目录后会失效，所以本次只在 `raw_results/` 中保留说明文件，不复制软链本体。

## 运行归档来源

`run_archive/` 来自：

```text
/home/family/workplace/scientific-intelligent-modelling/exp-planning/04.Core50正式全量评测/generated/core50_12alg_5seed_final_results
```

复制范围：

```text
analysis/
manifest.csv
summary.json
state/
results/
```

其中：

- `run_archive/summary.json`
  - `batch_name = core50_12alg_5seed_all_20260502-065700`
  - `expected_tasks = 3000`
  - `final_result_files = 3000`

- `run_archive/results/`
  - `12 x 5 x 50 = 3000` 个 JSON
  - 目录模式：`results/<tool>/seed<0-4>/*.json`

- `run_archive/manifest.csv`
  - 每行给出：
    - `task_id`
    - `tool`
    - `seed`
    - `host`
    - `result_json`
    - `raw_result_json`

## 原始实验树是否在包内

不在。

`raw_result_json` 列仍然指向原始实验树中的相对路径，例如：

```text
experiments/core50_12alg_5seed_all_20260502-065700/drsr/seed0/tasks/.../result.json
```

这意味着：

1. 本包保留了追溯字符串。
2. 本包不保证原始 `tasks/.../result.json` 目录树仍然随包分发。
3. 真正做 run-level 复盘时，优先使用 `run_archive/results/*.json`；只有需要回到原任务目录结构时，才额外去找外部 `experiments/` 树。

## Noise 鲁棒性来源

Noise 材料来自冻结包中的：

```text
frozen-results/neurips26/final-experiments/03_core50_12alg_final/noise_robustness/
```

协议与规模：

```text
noisy-training / clean-test
sigma = 0.01, 0.05, 0.10
12 algorithms x 50 datasets x 5 seeds x 3 sigmas = 9000 runs
```

当前包保留：

```text
noise_robustness/noise_final_runs.csv
noise_robustness/noise_completion_summary.csv
noise_robustness/robustness_components.csv
paper_tables/table13_noise_robustness_summary.csv
paper_tables/table14_noise_by_sigma.csv
```

当前冻结包没有给出 noisy runs 的逐任务 JSON 归档或逐行原始路径。因此
`9000` 条 noisy run-level 记录可用于复核论文汇总，但不能仅凭本包恢复各任务的
原始日志、checkpoint 或 `result.json` 文件。

## 使用建议

1. 看 clean 最终跑完多少，先查 `run_archive/summary.json`。
2. 看每条 run 的归档结果，先查 `run_archive/results/`。
3. 看最终数值榜单，优先查 `collected_results/`，必要时和 `run_archive/analysis/` 交叉核对。
4. 看 formal symbolic fidelity，查 `formal_analysis/`。
5. 看 noise robustness，查 `noise_robustness/` 以及论文 Table 13/14。

## provenance 风险提示

这里至少有两层混合来源需要显式记住：

1. `formal_analysis/`、`hexagon/`、`noise_robustness/`、`paper_tables/` 来自
   NeurIPS 冻结包的下游分析。
2. `run_archive/analysis/`、`manifest.csv`、`summary.json`、`state/`、`results/` 来自 `exp-planning` 的运行归档生成目录。

因此：

- 如果要拼一张“数值表现 + formal symbolic fidelity”的联合总表，
- 必须先声明它是 mixed-source 整理结果，
- 不要把它描述成同一个单次 postprocess 目录直接产物。
