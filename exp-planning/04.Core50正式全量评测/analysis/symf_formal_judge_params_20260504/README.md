# SYM-F formal judge 参数补齐审计

- Created at: `2026-05-04T23:38:37`
- Datasets: `50`
- Auto confirmed: `50`
- Needs review: `0`
- Formula parse ok: `50`
- Protected-operator datasets: `12`
- Probe samples per dataset: `4096`
- Probe random seed: `20260504`
- DeepSeek / LLM API needed: `false` for current audit.

## 输出文件

- `symf_formal_judge_parameters.csv`: 每个数据集的公式来源、变量映射、probe 范围和 judge policy。
- `symf_formal_judge_unresolved.csv`: 需要人工确认的问题项。
- `symf_formal_judge_config.json`: 机器可读摘要。

## 当前结论

- 所有 Core-50 数据集的必要参数均已自动确认，可以进入 formal judge 实现。