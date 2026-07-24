# Stage 3: 664dats × 4probes × 3seeds × 1h

这个目录整理的是第三阶段实验结果：

```text
664 datasets × 4 probes × 3 seeds × 1h = 7968 runs
probes = dso, imcts, pyoperon, udsr
目标 = 用 Probe-4 在全 664 数据集上三种子验证，并为 SSR-50/Core-50 选择提供依据
```

## 文件

- `probe4_postprocess_run_level.csv`
  - `7968` 行
  - 每行是一条 `dataset × algorithm × seed` 运行记录

- `probe4_current_run_level_raw_digest_7968.csv`
  - `7968` 行
  - 原始 raw digest，包含 `result_result_path`
  - 用于从远端 SSH 继续追每条原始 `result.json`

- `probe4_current_method_seed_coverage.csv`
  - `12` 行
  - `method × seed` 覆盖摘要

- `probe4_final_digest_summary.json`
  - raw digest 的总体摘要

- `probe4_postprocess_dataset_algorithm.csv`
  - `2656` 行
  - 每行是一个 `dataset × algorithm` 的跨 seed 汇总

- `probe4_postprocess_dataset_level.csv`
  - `664` 行
  - 每行是一个 dataset 的选择诊断指标

- `probe4_postprocess_summary.json`
  - 总体完成度摘要
  - 关键口径：`expected_total_runs=7968`，`finished_runs=7968`，`ready_for_core50_selection=true`

- `selection_constraints_report.md`
  - Core-50/SSR-50 选择前的约束检查报告

- `README_original.md`
  - 原 postprocess 目录里的说明文件

- `SOURCE_MAP.tsv`
  - 本整理目录的来源映射

- `FILE_INVENTORY.tsv`
  - 文件清单

- `CHECKSUMS.sha256`
  - 搬迁完整性校验

## 搬迁后校验

```bash
cd /path/to/stage3_664dats_4probes_3seeds_1h
sha256sum -c CHECKSUMS.sha256
```
