# Semantic-200 LLM Physics Spec Examples

这些文件由当前 runner 的 `build_runner_params()` 和统一 `spec_builder` 生成，用来检查 LLMSR/DRSR 有物理背景版本的真实 spec 口径。

| global_index | dataset | family | subgroup | n_features | note | files |
|---:|---|---|---|---:|---|---|
| 7 | `CRK22` | `llm-srbench` | `llm-srbench/chem_react` | 2 | llm-srbench chemistry; ordinary physical semantics | `g0007_CRK22_llmsr_spec.txt`, `g0007_CRK22_drsr_spec.txt` |
| 37 | `PO14` | `llm-srbench` | `llm-srbench/phys_osc` | 3 | llm-srbench oscillator; ordinary physical semantics | `g0037_PO14_llmsr_spec.txt`, `g0037_PO14_drsr_spec.txt` |
| 1 | `feynman-iii.15.12` | `srsd` | `srsd/srsd-feynman_medium_dummy` | 5 | SRSD dummy; hidden feature-role mapping with distractors | `g0001_feynman-iii.15.12_llmsr_spec.txt`, `g0001_feynman-iii.15.12_drsr_spec.txt` |
| 3 | `feynman-ii.11.20` | `srsd` | `srsd/srsd-feynman_hard` | 4 | SRSD non-dummy; direct physical feature semantics | `g0003_feynman-ii.11.20_llmsr_spec.txt`, `g0003_feynman-ii.11.20_drsr_spec.txt` |

关键检查点：

- LLMSR 与 DRSR 使用相同 background、变量名和变量描述。
- 二者主要差异在 evaluate block：LLMSR spec 内部会 BFGS 拟合参数，DRSR spec 使用固定 params 评估，后续 wrapper 再做参数恢复。
- SRSD dummy 不暴露 x_i 到物理角色的映射，只暴露语义角色多重集和 distractor 数量。
