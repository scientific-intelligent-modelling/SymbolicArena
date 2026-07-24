# 横向合并大表

这里放的是“同一任务，不同算法并排比较”的宽表，方便直接用表格软件或 pandas 看。

## 文件说明

- `candidate200_v02_by_dataset_wide_12alg.csv`
  - 来源：`01_v02_selection_digest/probe4_run_level_big_table_2400.csv`
  - 粒度：每行一个 Candidate-200 数据集，共 `200` 行
  - 列形式：`<algorithm>__<metric>`
  - 用途：看最终 v0.2 口径下 12 个算法在同一数据集上的指标、状态、表达式摘要和来源路径

- `candidate200_e1_nmse_by_dataset_wide_12alg.csv`
  - 来源：`03_e1_12alg_calibration/e1_12_dataset_algorithm_nmse_table_2400.csv`
  - 粒度：每行一个 Candidate-200 数据集，共 `200` 行
  - 列形式：`<algorithm>__<metric>`
  - 用途：看 E1 12 算法校准时的 NMSE / log NMSE 对比

Candidate-200 两张宽表按 `dataset_id` 合并；基础数据集信息优先取
`02_candidate200_inputs/candidate200_unified_with_paths.csv`，同时保留
`observed_*_values` 列，记录长表里观察到的描述字段取值，避免因为
`hard` / `hard non-dummy` 这类描述差异把同一个任务拆成多行。

Probe-4 全 664 三种子结果已经迁移到：

```text
../stage3_664dats_4probes_3seeds_1h/
```

## 读列名的方法

例如：

```text
pysr__id_nmse
pysr__ood_nmse
pysr__combined_log_id_ood_nmse
llmsr__normalized_status_for_probe4
```

意思就是同一行对应的数据集上，`pysr` 或 `llmsr` 的对应指标。

## 重新生成

如果上游整理包里的 CSV 被替换，可以运行：

```bash
python 99_audit/build_merged_tables.py
```
