# SYM-F formal judge 参数补齐审计

- Created at: `2026-07-26T12:06:38`
- Datasets: `664`
- Auto confirmed: `663`
- Needs review: `1`
- Formula parse ok: `664`
- Protected-operator datasets: `214`
- Probe samples per dataset: `4096`
- Probe random seed: `20260504`
- DeepSeek / LLM API needed: `false` for current audit.

## 输出文件

- `symf_formal_judge_parameters.csv`: 公式来源、变量映射、probe 范围和 judge policy。
- `symf_formal_judge_unresolved.csv`: 需要人工确认的问题项。
- `symf_formal_judge_config.json`: 机器可读摘要。

## 当前结论

- 有部分数据集需要确认，优先查看 `symf_formal_judge_unresolved.csv`。

|   core50_index | gid   | dataset   | dataset_dir                                 | issue                                          | needs_user_confirmation   | recommendation                                        |
|---------------:|:------|:----------|:--------------------------------------------|:-----------------------------------------------|:--------------------------|:------------------------------------------------------|
|            217 | g0217 | PO27      | sim-datasets-data/llm-srbench/phys_osc/PO27 | formula_arg_mapped_by_remaining_position:F0->x | True                      | 确认公式语义、变量顺序或采样范围后再纳入 formal judge |