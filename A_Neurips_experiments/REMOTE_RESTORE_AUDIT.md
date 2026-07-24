# NeurIPS Experiments 恢复审计

审计基准：

- Stage 1--3：沿用 2026-07-20 对 `iaaccn22~29` 的只读远端检查
- Stage 4：2026-07-25 从本地 NeurIPS 冻结材料、3000 个 clean 归档 JSON
  和 9000 条 noisy run-level 记录重建

本审计区分两个问题：

1. 是否可以恢复论文使用的结果、排行榜和符号指标？
2. 是否可以恢复当时完整的远端运行现场？

## 结论

论文结果和主要结论可以恢复。

完整远端运行现场不能保证 100% 恢复。NeurIPS Stage 4 已把 clean 批次
`3000/3000` 个最终 `result.json` 收入本地归档，不依赖失效的
`frozen-results/.../raw_results` 软链；三个 noisy 档位保留了完整的
`9000` 条 run-level 记录和汇总表，但没有逐任务原始 JSON。

## 四阶段口径

```text
stage1: 664 datasets × 2 probes × 1 seed × 1h = 1328
stage2: 200 datasets × 12 algorithms × 1 seed × 1h = 2400
stage3: 664 datasets × 4 probes × 3 seeds × 1h = 7968
stage4: 50 datasets × 12 algorithms × 5 seeds × 4 conditions × 1h = 12000
        conditions = clean, sigma 0.01, sigma 0.05, sigma 0.10
```

`REMOTE_RESULT_PATHS.tsv` 只登记具备逐行原始路径的项目：

```text
stage1: 1328 paths; 1162 yes, 166 no
stage2: 1400 paths; 1400 yes
stage3: 7968 paths; 7968 yes
stage4: 3000 clean paths; remote not rechecked, local clean archive 3000/3000
total: 13696 paths
```

Stage 4 的 `remote_exists=not_checked` 只针对具备路径字段的 clean 批次，
不代表结果缺失。对应结果已经位于：

```text
stage4_core50_12algs_5seeds_4noise_1h/run_archive/results/
```

三个 noisy 档位的 `9000` 条记录没有逐行原始路径，因此未计入
`REMOTE_RESULT_PATHS.tsv` 的 13696 条路径统计。

## Stage 1 缺口

Stage 1 有 166 条 `iaaccn22/pysr/timed_out` 原始远端路径在
2026-07-20 检查时不存在，详见：

```text
REMOTE_RESULT_PATHS_MISSING.tsv
```

这些任务的恢复后公式、指标和 timeout 信息仍保存在：

```text
stage1_664dats_2probes_1seed_1h/01_probe_run_results/
```

## Stage 2 缺口

Stage 2 的最终 v02 表有 2400 行，但其中 1000 行没有逐行
`source_result_path`，涉及 `e2esr/imcts/qlattice/ragsr/udsr` 各 200 行。

详见：

```text
STAGE2_SOURCE_PATH_GAPS.tsv
```

因此 Stage 2 的统计结论可恢复，但这 1000 行不能仅靠 v02 表直接定位原始
`result.json`。

## Stage 3

Stage 3 保存：

```text
probe4_current_run_level_raw_digest_7968.csv
probe4_current_method_seed_coverage.csv
probe4_final_digest_summary.json
```

其中 7968 条原始路径在 2026-07-20 的远端检查中全部存在。

## NeurIPS Stage 4

Clean 权威批次：

```text
core50_12alg_5seed_all_20260502-065700
```

完成情况：

```text
expected tasks: 3000
final result files: 3000
algorithms: 12
datasets: 50
seeds: 0, 1, 2, 3, 4
condition: clean
```

Noise 鲁棒性扩展：

```text
protocol: noisy-training / clean-test
noise sigma: 0.01, 0.05, 0.10
observed runs: 9000
valid_output=True: 8636
status: ok=8918, timed_out=31, no_valid_output=36, error=15
```

Stage 4 包含：

- 3000 个 clean 最终结果 JSON 和任务 manifest
- 9000 条 noisy run-level 记录、36 个算法/噪声单元的完成度摘要
- run-level、dataset-level 和算法级数值汇总
- formal SYM-F run-level 与算法级汇总
- Core-50 manifest、论文表、hexagon 和噪声鲁棒性表
- 1h/3h/24h 参数快照与后续预算审计材料

需要注意：论文 `table10_clean_core50_leaderboard.csv` 的数值列来自
2026-05-03 clean 排行榜；符号列来自后续 formal SYM-F 汇总，其中
LLM-SR/DRSR 使用了更新后的表达式链。该混合来源已在 Stage 4 README 和
`SOURCE_MAP.tsv` 中明确记录。Noise 表来自冻结包的 noisy-training 汇总，
当前包不包含 noisy runs 的逐任务 JSON 或原始路径清单。

## 可以恢复

- 四阶段的任务规模、输入选择和主要结论
- Candidate-200、Probe-4 和 Core-50 的选择链
- Stage 3 的 7968 条 Probe-4 结果摘要
- Stage 4 的 3000 个 clean 最终结果 JSON
- Stage 4 的 9000 条 noisy run-level 记录和三档 sigma 汇总
- NeurIPS clean 排行榜和 formal 符号指标
- NeurIPS noisy-training / clean-test 鲁棒性论文表
- 归档内所有文件的 SHA256 完整性

## 不能保证恢复

- Stage 1 的 166 个缺失远端原始结果
- Stage 2 的 1000 个逐行来源路径缺口
- Stage 4 原始实验树中的完整日志、checkpoint 和 minute snapshots
- Stage 4 noisy runs 的逐任务原始 JSON 和远端来源路径
- 当时的 conda / Julia / LLM API 运行环境
- 数据集实体 CSV 的完整快照

## 校验

```bash
cd A_Neurips_experiments/stage1_664dats_2probes_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage2_200dats_12algs_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage3_664dats_4probes_3seeds_1h
sha256sum -c CHECKSUMS.sha256

cd ../stage4_core50_12algs_5seeds_4noise_1h
sha256sum -c CHECKSUMS.sha256

cd ..
sha256sum -c CHECKSUMS.sha256
```
