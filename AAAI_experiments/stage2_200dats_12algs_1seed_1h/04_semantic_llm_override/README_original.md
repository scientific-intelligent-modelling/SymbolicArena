# Semantic-200 LLM Comparison

- 新语义批次：`semantic200_llm_physics_v2_4host_seed1314_20260429-205250`
- 旧基线：`e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv`
- 比较对象：`llmsr`、`drsr`，均为 Candidate-200、seed=1314。
- 主要判据：`delta_log10_*_nmse_semantic_minus_nonsemantic`，负数表示加入语义后 NMSE 变小。
- delta 按 `NMSE` 裁剪到 `[1e-12, 1e12]` 后计算，避免少数爆炸值主导均值。

## Coverage

| algorithm | rows | semantic_metrics | nonsemantic_metrics |
| --- | --- | --- | --- |
| drsr | 200 | 200 | 200 |
| llmsr | 200 | 196 | 200 |

## Method Summary

| algorithm | n | median_log10_ood_nonsemantic | median_log10_ood_semantic | median_delta_log10_ood | ood_better_ge_2x | ood_similar_within_2x | ood_worse_ge_2x | median_log10_valid_nonsemantic | median_log10_valid_semantic | median_delta_log10_valid | valid_better_ge_2x | valid_similar_within_2x | valid_worse_ge_2x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drsr | 200 | -2.828e-06 | -0.479 | -2.65 | 130 | 58 | 12 | -0.2707 | -1.435 | -3.377 | 137 | 58 | 5 |
| llmsr | 200 | -8.98e-06 | -1.129 | -4.728 | 151 | 37 | 8 | -0.3073 | -2.535 | -4.81 | 163 | 36 | 1 |

## Output Files

- `semantic_results_raw.csv`：远端新语义批次 `result.json` 摘要。
- `semantic_vs_nonsemantic_run_table.csv`：逐数据集逐算法对比主表。
- `semantic_vs_nonsemantic_method_summary.csv`：按算法汇总。
- `semantic_vs_nonsemantic_family_summary.csv`：按算法和 family 汇总。
- `semantic_vs_nonsemantic_top_changes.csv`：OOD 指标改善/退化最大的样本。
