# Stage 1: 664dats × 2probes × 1seed × 1h

这个目录整理的是第一阶段实验材料：

```text
664 datasets × 2 probes × 1 seed × 1h
probes = pysr, llmsr
目标 = 从 664 个数据集中筛出 Candidate-200
```

## 先看哪里

1. `01_probe_run_results/`
   - 双探针一轮运行后的主结果
   - `one_seed_probe_task_results_1328.csv` 是 `664 × 2 = 1328` 条任务结果
   - `one_seed_probe_dataset_compare_664.csv` 是每个数据集一行的 PySR/LLMSR 对比
   - `one_seed_probe_formulas_1328.csv` 是公式结果表

2. `02_candidate200_selection/`
   - 从 664 选到 Candidate-200 的选择材料
   - `stage1_candidate200_flat_200.csv` 是最终 200 个候选数据集
   - `stage1_probe_family_wins.csv` 是按 family/probe 的胜出摘要

3. `03_launch_materials/`
   - 第一阶段启动和切片材料
   - 包括 `datasets_to_run_664.csv`、`launch_pysr_probe.py`、`launch_llmsr_probe.py` 和各 host slice

4. `04_iteration_analysis/`
   - LLMSR 迭代统计补充材料

5. `99_audit/`
   - `source_map.tsv` 记录来源
   - `file_inventory.tsv` 记录文件清单
   - `checksums.sha256` 用于搬迁校验

## 关键行数

- `one_seed_probe_task_results_1328.csv`: `1328` rows
- `one_seed_probe_dataset_compare_664.csv`: `664` rows
- `one_seed_probe_formulas_1328.csv`: `1328` rows
- `stage1_candidate200_flat_200.csv`: `200` rows

## 搬迁后校验

```bash
cd /path/to/stage1_664dats_2probes_1seed_1h
sha256sum -c 99_audit/checksums.sha256
```

全部显示 `OK` 即表示包内文件搬迁完整。
