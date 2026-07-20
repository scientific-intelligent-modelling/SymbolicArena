# benchmark_formal200_20260417

这个目录汇总了三套实验结果：

1. 单种子探针结果（seed=1314，覆盖 664 数据集，方法为 PySR + LLM-SR）
2. 双种子正式结果（seeds=520/521，覆盖 200 候选数据集，方法为 PySR + LLM-SR）
3. 三种子正式结果（seeds=520/521/522，覆盖 200 候选数据集，方法为 PySR + LLM-SR，且 PySR 已对 `slice_01` 做 1h 统一口径纠偏）
4. 基于三种子正式结果筛出的 `Master-100 / Dev-50 / Core-50` 切分结果

## 文件说明

- `one_seed_probe_task_results.csv`：单种子探针任务级总表（1328 行）
- `one_seed_probe_dataset_compare.csv`：单种子探针按 dataset 对齐后的方法对照表（664 行）
- `one_seed_probe_formulas.csv`：单种子探针的独立公式表（1328 行，记录格式化后的公式）
- `one_seed_probe_summary.md`：单种子探针摘要
- `two_seed_formal_task_results.csv`：双种子正式任务级总表（800 行）
- `two_seed_formal_dataset_method_summary.csv`：双种子正式按 dataset×method 聚合后的摘要
- `two_seed_formal_dataset_compare.csv`：双种子正式按 dataset 对齐后的方法对照表（200 行）
- `two_seed_formal_formulas.csv`：双种子正式的独立公式表（800 行）
- `two_seed_formal_summary.md`：双种子正式摘要
- `three_seed_formal_task_results.csv`：三种子正式任务级总表（1200 行，统一 1h 口径）
- `three_seed_formal_dataset_method_summary.csv`：三种子正式按 dataset×method 聚合后的摘要
- `three_seed_formal_dataset_compare.csv`：三种子正式按 dataset 对齐后的方法对照表（200 行）
- `three_seed_formal_formulas.csv`：三种子正式的独立公式表（1200 行）
- `three_seed_formal_summary.md`：三种子正式摘要
- `dev_core_split_v1/master100_candidates.csv`：基于三种子正式结果筛出的 `Master-100`
- `dev_core_split_v1/benchmark_dev50.csv`：仅用于进化/调参的 `Dev-50`
- `dev_core_split_v1/benchmark_core50.csv`：冻结的最终评测集 `Core-50`
- `dev_core_split_v1/benchmark_dev_core_split_audit.json`：切分的机器可读审计信息
- `dev_core_split_v1/benchmark_dev_core_split_audit.md`：切分的人类可读审计报告

## 关键口径

- 所有正式结果都基于 `task_status.jsonl + experiment_dir/result.json` 汇总，不使用容易互相覆盖的外层 `result.json`。
- 单种子 probe 结果中额外补了 `is_formal200_candidate` 与候选池标签，便于和 200 候选集联动分析。
- `three_seed_formal_*` 这组结果已经把 `pysr` 的 `slice_01` 用 1h 纠偏重跑结果覆盖回去，因此三种子正式表可以视为统一 1h 口径的正式结果。
- 主结果表默认不带 `equation` 列；若需要查看格式化后的公式，请使用 `*_formulas.csv` 这三张独立公式表。
- `Master-100` 允许使用三种子正式结果筛选；`Dev/Core` 切分阶段只使用非结果信息（`family / subgroup / selection_mode / candidate_advantage_side / basename / 特征维度 / 样本量 / 静态公式复杂度`），以减少对最终测试集独立性的破坏。
- `basename <= 1` 在整个 `Master-100` 上成立，避免 `Dev-50` 与 `Core-50` 之间出现同 basename 变体泄漏。

## 复现命令

```bash
python check/generate_master100_dev_core_split.py \
  --compare-csv experiment-results/benchmark_formal200_20260417/three_seed_formal_dataset_compare.csv \
  --candidate-json /tmp/candidate_seeds_200_v3.json \
  --output-dir experiment-results/benchmark_formal200_20260417/dev_core_split_v1
```
