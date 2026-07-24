# 第二阶段运行追溯材料

这个目录补的是第二阶段 `Candidate-200 × 12 algorithms × 1h` 的运行侧材料。

它和 `03_e1_12alg_calibration/` 的区别是：

- `03_e1_12alg_calibration/`：冻结后的指标表、Probe-4 选择报告和选择方法。
- `07_stage2_run_provenance/`：原始聚合结果、运行索引、host/wave 摘要、启动脚本和参数快照。

## 关键文件

- `raw_clean_archive/result_index.csv`
  - `1400` 行
  - 每行是一条被归档的运行结果索引
  - 字段包括 `wave, tool, host, global_index, dataset_name, status, seconds, source_result_path`

- `raw_clean_archive/all_results.jsonl`
  - `1400` 行
  - 每行包含一条聚合后的原始 result 内容
  - 用它可以不依赖原始 `result.json` 森林做二次解析

- `raw_clean_archive/summary/e1_run_summary_by_host_20260424-041046.csv`
  - `31` 行
  - host / wave 粒度的完成度摘要

- `raw_clean_archive/summary/wave_manifest.csv`
  - `31` 行
  - 第二阶段 wave 分发清单

- `generated/launch/run_w*.sh`
  - 原始启动脚本快照

- `generated/slices/`
  - 原始 wave/algorithm/host 切片

- `generated/remote_jobs/`
  - 原始远端 job 脚本

- `generated/remaining5_full200_v1/`
  - `e2esr, iMCTS, QLattice, ragsr, udsr` 这 5 个算法的 full Candidate-200 分发材料
  - 用于解释最终 2400 表中这 5 个算法没有逐行 `source_result_path` 的来源口径

- `generated/params/*.json`
  - 第二阶段生成出来的 12 算法参数文件

- `hyperparams_snapshot_20260423/*.json`
  - 第二阶段运行前冻结的超参数快照

- `docs/e1_run_summary_20260424-041046.md`
  - 第二阶段运行摘要

- `docs/e1_repair_summary_20260426.md`
  - 第二阶段后处理/修复摘要

## 关于 1400 和 2400

这个目录里的 `1400` 行是当时 clean archive 里实际归档到的原始运行聚合结果。

而 `03_e1_12alg_calibration/e1_12_dataset_algorithm_nmse_table_2400.csv`
和 `06_merged_tables/candidate200_e1_nmse_by_dataset_wide_12alg.csv`
是后续冻结后的校准表，口径是：

```text
200 datasets × 12 algorithms = 2400 dataset-algorithm rows
```

所以：

- 想看最终第二阶段指标：看 `03_e1_12alg_calibration/` 或 `06_merged_tables/`
- 想追第二阶段怎么跑、哪些机器、哪些 wave、原始聚合 result：看本目录

## 没有放什么

这里没有放完整原始实验目录森林，也没有放远端机器上的日志目录和环境。

原因是这些内容体积大、路径依赖强，迁移价值低；当前包保留的是能复盘和二次统计的最小充分材料。
