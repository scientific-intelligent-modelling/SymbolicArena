# NeurIPS Stage 4 clean/noise 多种子均值整理

本目录根据 NeurIPS 最后阶段的冻结结果，生成与 `噪声.png` 结构一致的
clean/noise 对照表。这里只做本地后处理，**没有重新运行任何算法**。

## 数据口径

- 算法：12 个
- 数据集：Core-50
- 随机种子：`0,1,2,3,4`
- 条件：clean、noise `0.01`、`0.05`、`0.10`
- 总记录：`12 × 50 × 5 × 4 = 12000`

输入文件：

- clean：
  `paper/neurips26-upload/results-artifact/clean-core50/clean_final_runs_updated.csv`
- noise：
  `A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h/noise_robustness/noise_final_runs.csv`
- 正式 ROBU 对照：
  `A_Neurips_experiments/stage4_core50_12algs_5seeds_4noise_1h/paper_tables/table13_noise_robustness_summary.csv`

clean 输入采用论文结果工件中的更新版，以纳入 DRSR 和 LLM-SR 的最终修正结果。

## 图片五项指标

图片中的“指标 1–5”被落实为以下可由 Stage 4 run-level 数据直接计算的统计量：

1. `Valid rate (%)`：五个种子中具有有效且完整 ID/OOD 指标的比例。
2. `ID-Q mean (%)`：每个 run 的 ID NMSE 经正式 `phi` 映射后，对五种子取算术均值。
3. `OOD-Q mean (%)`：每个 run 的 OOD NMSE 经正式 `phi` 映射后，对五种子取算术均值。
4. `Joint-Q mean (%)`：`0.5 × ID-Q + 0.5 × OOD-Q`。
5. `Clean-relative retention (%)`：noisy Joint-Q 相对同一算法、同一任务 clean Joint-Q 的保持率。

最后一列为 `ROBU seed-mean (%)`：

```text
100 × mean_dataset,sigma(0.7 × Joint-Q + 0.3 × retention)
```

所有指标均为越高越好。失败、无有效输出、指标不完整或 NMSE 非有限的 run 不会被过滤，
而是沿用论文 failure semantics，将 ID/OOD NMSE 设为 `1e2`，对应质量分为零。

## 与论文正式 ROBU 的边界

用户指定“多个种子求均值”，因此本表先对每个种子的质量分取**算术均值**。
论文正式 ROBU 则先对五个种子的 NMSE 取**中位数**，再进行 `phi` 映射。
两者不是同一个估计量。本目录保留
`stage4_robu_seed_mean_vs_formal.csv`，用于量化二者差异；正式论文数值仍应引用
`table13_noise_robustness_summary.csv`。

本表的五项统计也不是把 ID-Q、OOD-G、SYM-F、EFF、STAB 五条正式 clean 轴
在噪声下重新计算。Stage 4 噪声轨道的正式第六轴只有 ROBU。

## 输出

- `stage4_noise_seed_mean_table.png/.pdf`：与参考图结构一致的完整表格。
- `stage4_noise_seed_mean_summary.md`：完整性、排名和正式口径对照摘要。
- `stage4_condition_metrics_seed_mean_wide.csv`：每算法五行的宽表。
- `stage4_condition_metrics_seed_mean_long.csv`：长表。
- `stage4_dataset_condition_seed_mean.csv`：算法 × 数据集 × 条件的五种子聚合。
- `stage4_condition_summary.csv`：算法 × 条件汇总。
- `stage4_condition_completeness.csv`：完成度和失败惩罚计数。
- `stage4_robu_seed_mean_vs_formal.csv`：seed-mean 与正式 seed-median ROBU 对照。
- `reproducibility.json`：输入哈希、Git commit、环境、定义和网格审计。

## 复现

从仓库根目录运行：

```bash
PYTHONPATH=. python \
  A_Neurips_experiments/rebuttal/05_stage4_noise_seed_mean_summary/analyze_stage4_noise_seed_mean.py
```

运行测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  pytest -q \
  A_Neurips_experiments/rebuttal/05_stage4_noise_seed_mean_summary/test_analyze_stage4_noise_seed_mean.py
```
