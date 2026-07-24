# Core-50 数据集名单校验报告

校验时间：2026-05-02

## 结论

- 数据集条数：`50`
- 重复 dataset name：`0`
- 重复 dataset dir：`0`
- 缺失目录：`0`
- 缺失 `metadata.yaml`：`0`
- 缺失 `train.csv / valid.csv / id_test.csv / ood_test.csv`：`0`

这 50 个数据集可以作为后续 Core-50 正式全量评测的输入名单。

## Family 分布

- `llm-srbench`: 16
- `srsd`: 15
- `srbench1.0`: 9
- `nguyen`: 3
- `srbench2025`: 2
- `korns`: 2
- `keijzer`: 2
- `vladislavleva`: 1

## 文件说明

- `core50_dataset_paths.tsv`: 用户给定的原始名称与路径名单。
- `core50_datasets.csv`: 已补充 family、subgroup、target、feature_count 和 split row counts 的机器可读 manifest。

