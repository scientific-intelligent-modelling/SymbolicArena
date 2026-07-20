# AAAI Experiments 远端恢复审计

审计时间：2026-07-20

本审计回答一个问题：

```text
只知道 AAAI_experiments 这几个文件夹，并且能 SSH 到 iaaccn22~29，
是否可以还原实验结果、结论和原始实验现场？
```

## 结论

结果和结论：可以还原。

完整原始实验现场：还不能说 100%。

当前包内已经保存了四个阶段的关键表、结论表、选择表、路径说明和校验哈希。  
但“原始现场”还依赖远端机器上的 `result.json`、日志、中间进度文件、运行环境和数据集实体。

## 远端 result.json 覆盖

已生成统一远端路径表：

```text
REMOTE_RESULT_PATHS.tsv
```

并用 `ssh + test -f` 对 8 台机器做了只读检查。

汇总：

```text
total remote result paths: 17446
exists: 17280
missing: 166
```

按阶段：

```text
stage1: 1162 / 1328 exists, 166 missing
stage2: 1400 / 1400 exists
stage3: 7968 / 7968 exists
stage4: 6750 / 6750 exists
```

缺失清单：

```text
REMOTE_RESULT_PATHS_MISSING.tsv
```

缺失项全部集中在：

```text
stage1 / iaaccn22 / pysr / timed_out
```

这 166 条远端原始 `result.json` 当前按包内路径找不到。  
不过 stage1 本地整理表已经保留这些任务的恢复后公式、指标和 timeout 信息：

```text
stage1_664dats_2probes_1seed_1h/01_probe_run_results/one_seed_probe_task_results_1328.csv
stage1_664dats_2probes_1seed_1h/01_probe_run_results/one_seed_probe_formulas_1328.csv
```

所以 stage1 的结论可以还原，但这 166 条的远端原始文件现场不可视为 100% 可取回。

## Stage 2 特别说明

Stage2 最终 v02 大表是：

```text
stage2_200dats_12algs_1seed_1h/01_v02_selection_digest/probe4_run_level_big_table_2400.csv
```

它有 `2400 = 200 datasets × 12 algorithms` 行。

其中：

```text
source_result_path nonempty: 1400
source_result_path empty: 1000
```

缺少逐行 `source_result_path` 的算法是：

```text
e2esr: 200
imcts: 200
qlattice: 200
ragsr: 200
udsr: 200
```

缺口清单：

```text
STAGE2_SOURCE_PATH_GAPS.tsv
```

这些算法的最终指标和结论在 stage2 表里，但 v02 表本身没有逐行原始 `result.json` 路径。  
补充的运行侧材料已经放入：

```text
stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/generated/remaining5_full200_v1/
stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/generated/slices/
stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/generated/remote_jobs/
```

这能解释这 5 个算法如何分发和运行，但不能把 v02 表缺失的逐行 `source_result_path` 自动补成 2400/2400。

## Stage 3 补强

Stage3 原先只有 postprocess 表。现在已经补入 raw digest：

```text
stage3_664dats_4probes_3seeds_1h/probe4_current_run_level_raw_digest_7968.csv
stage3_664dats_4probes_3seeds_1h/probe4_current_method_seed_coverage.csv
stage3_664dats_4probes_3seeds_1h/probe4_final_digest_summary.json
```

其中 `probe4_current_run_level_raw_digest_7968.csv` 包含 `7968` 条 `result_result_path`，并且本次远端检查为：

```text
7968 / 7968 exists
```

## 什么可以恢复

可以从当前 `AAAI_experiments/` 恢复：

- 四阶段的主要实验规模和任务口径
- stage1 的 1328 条双探针结果和 Candidate-200 选择
- stage2 的 2400 条 12 算法校准结果、Probe-4 选择和语义覆盖
- stage3 的 7968 条 Probe-4 full664 后处理结果，并可追原始 result
- stage4 的 6750 条最终 SSR-50 打榜结果，并可追原始 result
- stage4 的 15 算法排行榜和横向宽表
- 每个整理包的完整性校验
- 绝大多数远端 result.json 的取回路径

## 不能保证恢复

当前不能保证 100% 恢复：

- stage1 `iaaccn22/pysr` 的 166 个远端原始 `result.json`
- stage2 v02 表里 5 个算法的逐行 `source_result_path`
- 远端完整日志目录
- 远端中间 checkpoint / progress / hall-of-fame 等非结果文件
- conda / Julia / LLM API 运行环境
- 数据集实体 CSV 的完整快照
- 当时完整代码环境和第三方依赖状态

## 最小使用方式

验证包内文件：

```bash
cd AAAI_experiments/stage1_664dats_2probes_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage2_200dats_12algs_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage3_664dats_4probes_3seeds_1h
sha256sum -c CHECKSUMS.sha256

cd ../stage4_ssr50_15algs_3seeds_3noise_3h
sha256sum -c CHECKSUMS.sha256
```

查看远端恢复状态：

```bash
cd AAAI_experiments
column -t -s $'\t' REMOTE_RESULT_PATHS_MISSING.tsv | less -S
cat REMOTE_RESTORE_SUMMARY.json
```

## 最终判断

当前状态适合：

```text
复查结果、写论文、生成表格、解释实验链路、追主要远端 result.json
```

当前状态不适合宣称：

```text
完整原始实验现场 100% 已归档
```

若要达到真正 100%，还需要另做一次完整归档：

```text
远端 result/log/progress/checkpoint 全量 rsync
数据集实体快照
运行环境快照
代码 commit/tag + 依赖锁定
```
