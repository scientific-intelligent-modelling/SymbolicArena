# Stage 4: Core-50 12algs x 5seeds x clean x 1h

这个目录现在对应的是 NeurIPS 最终阶段，不再是原先 AAAI 的 SSR-50 15 算法包。

```text
12 algorithms x 50 datasets x 5 seeds x clean-only x 1h = 3000 runs
batch = core50_12alg_5seed_all_20260502-065700
seeds = 0, 1, 2, 3, 4
```

## 当前权威口径

优先看这几个文件：

- `run_archive/summary.json`
  - 当前最权威的运行完成度摘要
  - 关键口径：`expected_tasks=3000`，`final_result_files=3000`

- `run_archive/manifest.csv`
  - `3000` 行任务清单
  - 每行一条 `algorithm x dataset x seed` 运行，并带 `raw_result_json`

- `run_archive/results/`
  - `3000` 个归档 `json`
  - 这是本包内最完整的 run 级结果集合

- `collected_results/`
  - NeurIPS 冻结包内的 clean leaderboard、run-level 汇总和 Probe-4 对照表

- `formal_analysis/`
  - formal symbolic fidelity 指标

- `EXPERIMENT_PATHS.md`
  - 记录本目录使用的真实来源路径、缺失的原始实验树以及使用边界

- `SOURCE_MAP.tsv`
  - 本整理目录的来源映射

## 目录说明

- `core50_manifest/`
  - Core-50 冻结数据集清单和路径表

- `run_archive/`
  - 从 `exp-planning/04.Core50正式全量评测/generated/core50_12alg_5seed_final_results` 复制的运行归档
  - 包含 `analysis/`、`manifest.csv`、`summary.json`、`state/`、`results/`

- `collected_results/`
  - 来自 NeurIPS 冻结包的 clean 汇总表
  - 与 `run_archive/analysis/` 高度重合，但后者额外保留了 host 并发表和两个 JSON 摘要

- `formal_analysis/`、`hexagon/`、`paper_tables/`
  - NeurIPS 下游分析产物

- `noise_robustness/`
  - 也是冻结包内现成分析结果
  - 注意它不是本阶段 clean 1h 主运行归档本身

- `raw_results/`
  - 只保留 `README.md`，说明原始结果树未随包归档
  - 原冻结包这里是一个指向 `experiments/core50_12alg_5seed_all_20260502-065700` 的软链
  - 该软链在整理包内无效，所以本次明确不复制

## 当前完成度

`run_archive/summary.json` 当前记录：

- 总任务数：`3000`
- 最终归档结果文件：`3000`
- `ok=2982`
- `timed_out=9`
- `no_valid_output=9`
- 每个算法：`250`
- 每个 seed：`600`

## 重要边界

1. 本目录没有打包原始 `experiments/core50_12alg_5seed_all_20260502-065700` 实验树。
2. 但 `run_archive/results/` 的 `3000` 个归档 JSON 都已包含结果内容，`run_archive/manifest.csv` 也保留了 `raw_result_json` 路径字符串，足够继续做大多数 run 级追溯。
3. `formal_analysis/` 的符号表和 `collected_results/`、`run_archive/analysis/` 的数值表来自不同上游整理步骤；做跨表拼接时要先看 `SOURCE_MAP.tsv`，避免把 mixed provenance 当成单一流水线直接产物。

## 完整性校验

本阶段已生成：

- `FILE_INVENTORY.tsv`：记录阶段内文件的相对路径和字节数
- `CHECKSUMS.sha256`：记录阶段内文件的 SHA-256

从本目录执行：

```bash
sha256sum -c CHECKSUMS.sha256
```
