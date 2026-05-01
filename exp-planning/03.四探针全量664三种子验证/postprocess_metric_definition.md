# Probe4 Full664 后处理指标定义

本文件定义 `4 algorithms × 664 datasets × 3 seeds` 后处理的指标口径。目标是把原始 run-level 结果转换为后续 Core-50 selection 所需的干净输入。

本阶段只生成指标和报告，不选择 Core-50。

## 核心原则

`unfinished run` 不能被当成 `invalid run`。

`pending`、`running`、`queued`、`not_started` 只表示实验尚未完成，不表示算法失败；它们只进入 completion rate，不进入 invalid rate、timeout rate 或 failure mode。

## 输出层级

脚本输出三个主要 CSV：

- `probe4_postprocess_run_level.csv`: 每行对应一个 `dataset × method × seed` run。
- `probe4_postprocess_dataset_algorithm.csv`: 每行对应一个 `dataset × method` 的 3-seed 聚合。
- `probe4_postprocess_dataset_level.csv`: 每行对应一个 dataset 的整体选择诊断指标。

同时输出：

- `probe4_postprocess_summary.json`: 机器可读 summary。
- `selection_constraints_report.md`: 人读完成度与选择约束报告。

## Run-Level 口径

### 标准化字段

- `method_norm`: 标准化算法名，当前预期为 `dso`、`imcts`、`pyoperon`、`udsr`。
- `seed_norm`: 标准化 seed，当前预期为 `520`、`521`、`522`。
- `dataset_key`: 稳定 groupby key，优先使用 `dataset_id`。

### 完成与可评估

- `is_finished_run`: run 是否进入终态。
- `wrong_dataset_flag`: 结果是否来自错误数据集。
- `is_evaluable_run`: `is_finished_run and not wrong_dataset_flag`。
- `not_finished_flag`: `not is_finished_run`。

### NMSE 处理

所有 NMSE 都进入 clipped log 空间：

```text
log_nmse = clip(log10(max(raw_nmse, 1e-12)), -12, 12)
```

若输入 CSV 已存在 `*_log10_nmse_clip12`，优先使用原字段。

输出字段：

- `train_log_nmse_used`
- `valid_log_nmse_used`
- `id_log_nmse_used`
- `ood_log_nmse_used`

### Run Outcome

`run_outcome_class` 允许取值：

- `not_finished`: 实验未完成，不算失败。
- `wrong_dataset`: 数据集身份不匹配，属于实验污染。
- `valid_finite_result`: 合法输出，指标完整，误差非极端。
- `valid_extreme_error`: 合法输出，指标完整，但 clipped log NMSE 到达极端上界。
- `partial_output`: 有表达式或部分输出，但主性能指标不完整。
- `timeout_no_output`: 超时且没有有效输出。
- `invalid_output`: 表达式非法、预测 NaN/Inf 或指标不可评估。
- `no_output`: 终态但没有表达式。
- `system_error`: wrapper、环境或子进程错误。
- `unknown_failure`: 兜底失败类型。

`valid_extreme_error` 仍是有效结果，会参与 clipped log NMSE 聚合；它不是 invalid。

## Dataset × Algorithm 口径

每行聚合一个算法在一个数据集上的 3 个 seeds。

主要计数：

- `n_expected_seeds`: 固定为 3。
- `n_observed_rows`: 原始 CSV 中实际存在的 row 数。
- `n_finished_seeds`: 进入终态的 seed 数。
- `n_evaluable_seeds`: finished 且不是 wrong dataset 的 seed 数。
- `n_valid_total_seeds`: `valid_finite_result + valid_extreme_error`。
- `n_invalid_seeds`: `invalid_output/no_output/system_error/unknown_failure`。
- `n_timeout_seeds`: `timeout_no_output`。
- `n_not_finished_seeds`: `not_finished`。

主要比例：

- `completion_rate = n_finished_seeds / n_expected_seeds`
- `evaluable_rate = n_evaluable_seeds / n_expected_seeds`
- `valid_rate = n_valid_total_seeds / max(n_evaluable_seeds, 1)`
- `valid_rate_over_expected = n_valid_total_seeds / n_expected_seeds`
- `invalid_rate = n_invalid_seeds / max(n_evaluable_seeds, 1)`
- `timeout_rate = n_timeout_seeds / max(n_evaluable_seeds, 1)`
- `extreme_error_rate = n_valid_extreme_seeds / max(n_evaluable_seeds, 1)`

性能聚合只使用：

- `valid_finite_result`
- `valid_extreme_error`

输出包括：

- `median_log_train_nmse`
- `median_log_valid_nmse`
- `median_log_id_nmse`
- `median_log_ood_nmse`
- `iqr_log_id_nmse`
- `iqr_log_ood_nmse`
- `median_id_r2`
- `median_ood_r2`
- `median_runtime`
- `p95_runtime`
- `median_complexity`
- `iqr_complexity`
- `median_tree_depth`

## Dataset-Level 口径

每行聚合一个 dataset 的 4 个 probe methods。

完成度：

- `expected_probe_methods`: 固定为 4。
- `expected_total_runs`: 固定为 12。
- `observed_methods`: 原始 CSV 中出现过的 method 数。
- `completed_methods`: 3 seeds 全部 finished 的 method 数。
- `valid_methods`: 至少一个 seed 有有效结果的 method 数。
- `completion_rate = finished_runs / expected_total_runs`
- `dataset_valid_rate = valid_runs / max(finished_runs, 1)`
- `dataset_invalid_rate = invalid_runs / max(finished_runs, 1)`
- `dataset_timeout_rate = timeout_runs / max(finished_runs, 1)`
- `dataset_extreme_error_rate = valid_extreme_error_runs / max(finished_runs, 1)`

区分度：

- `id_probe_variance`: 4 个 probe 的 ID 表现方差。
- `ood_probe_variance`: 4 个 probe 的 OOD 表现方差。
- `ood_id_gap_variance`: OOD-ID gap 的 probe 间方差。
- `pairwise_gap_mean`: OOD 表现两两差异均值。
- `pairwise_gap_max`: OOD 表现最大两两差异。
- `valid_pattern_entropy`: valid/invalid 模式熵。
- `rank_entropy`: 基于 `softmax(-median_log_ood_nmse / T)` 的 winner pattern 熵，当前 `T=1.0`。

综合分数：

```text
discrimination_score =
    0.35 * norm(id_probe_variance)
  + 0.35 * norm(ood_probe_variance)
  + 0.15 * norm(ood_id_gap_variance)
  + 0.10 * norm(pairwise_gap_mean)
  + 0.05 * norm(valid_pattern_entropy)
```

稳定性：

```text
instability_raw =
    0.40 * mean_iqr_log_ood_nmse
  + 0.20 * mean_iqr_log_id_nmse
  + 0.20 * max_iqr_log_ood_nmse
  + 0.10 * dataset_invalid_rate
  + 0.10 * dataset_timeout_rate

stability_score = 1 - robust_norm(instability_raw)
```

信息量：

```text
info_score = discrimination_score * stability_score
```

## Difficulty 与 Failure Mode

`difficulty_score` 为各 probe 的 ID/OOD log NMSE 平均难度：

```text
difficulty_score = mean(0.5 * median_log_id_nmse + 0.5 * median_log_ood_nmse)
```

`difficulty_bin` 按分位数划分：

- `easy`: bottom 20%
- `medium`: 20%-60%
- `hard`: 60%-90%
- `extreme`: top 10%

若 `dataset_invalid_rate` 或 `dataset_timeout_rate` 很高，可直接进入 `extreme`。

`failure_mode` 按优先级判断：

1. `incomplete`
2. `invalid_prone`
3. `one_sided`
4. `unstable`
5. `ood_failure`
6. `all_struggle`
7. `one_method_wins`
8. `all_good`

## Eligible Class

`eligible_class` 是诊断标签，不是最终选择结果。

允许取值：

- `incomplete`: 实验未完成，不能用于正式选择。
- `eligible`: 正常候选。
- `limited_quota`: 有价值但后续 selector 必须限制数量。
- `excluded`: all-invalid、wrong-dataset 或严重实验污染。

## Normalization

所有 `norm(...)` 默认使用 robust min-max：

```text
robust_min = 5th percentile
robust_max = 95th percentile
norm(x) = clip((x - robust_min) / (robust_max - robust_min), 0, 1)
```

若 `robust_max == robust_min`，返回 0。

## 明确不做

本阶段不做以下事情：

- 不正式选择 Core-50。
- 不修改原始 CSV。
- 不调整算法结果。
- 不把 unfinished 当 invalid。
- 不删除 `valid_extreme_error`。
- 不写 SRSD-specific 特殊规则。
