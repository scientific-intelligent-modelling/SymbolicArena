# Probe4 Full664 Postprocess Outputs

- 输入: `/home/family/workplace/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/probe4_full664_v1/final_digest_20260501-105508/probe4_current_run_level.csv`
- 目标: 生成 Core-50 selection 前置诊断指标，不选择 Core-50。

## 文件

- `probe4_postprocess_run_level.csv`: 每条 run 的完成/有效/失败分类与 log NMSE。
- `probe4_postprocess_dataset_algorithm.csv`: 每个 dataset × algorithm 的 3-seed 聚合。
- `probe4_postprocess_dataset_level.csv`: 每个 dataset 的区分度、稳定性、难度和候选资格指标。
- `probe4_postprocess_summary.json`: 机器可读 summary。
- `selection_constraints_report.md`: 人读完成度与选择约束报告。

## 重要口径

- `not_finished` 不计入 invalid、timeout 或算法失败。
- `valid_extreme_error` 仍是有效输出，会参与 clipped log NMSE 统计。
- `eligible_class` 是诊断标签，不等于最终 Core-50 选择结果。
