# Probe-4 v0.2 人话版选择包

这个文件夹只放 Probe-4 v0.2 的可读结论和可计算大表，不再混放旧版 `nmse_only` 输出。

## 我们现在到底要选什么

我们已经有 `200 个数据集 x 12 个算法 = 2400 条实验结果`。下一步不是找 4 个平均分最高的算法，而是找 4 个最适合当“探针”的算法。

探针的作用是：后面用这 4 个算法去跑更大的 664 数据集池，尽量低成本复现完整算法面板会看到的数据集差异。

换句话说，Probe-4 要回答的是：

```text
如果只允许用 4 个算法观察数据集，哪 4 个算法最不容易看偏？
```

## 每个评分项说人话

- `stability`：这个算法是不是经常能给出 train/id/ood 三类有效数值。经常缺结果的算法不能当主探针。
- `discrimination_fidelity`：这 4 个算法认为“哪些数据集最有区分度”，是否接近完整算法面板的判断。这是最核心的项。
- `complementarity`：4 个算法是不是看问题的角度不同。如果两个算法在所有数据集上同涨同跌，它们同时入选的价值就低。
- `selected_dataset_coverage`：这 4 个算法挑出来的高信息量数据集，不能全堆在某个 family 或 subgroup。
- `gradient_diversity`：不同算法的退化路径是否不同。这里现在只看 `train -> id` 和 `id -> ood`，不再用 valid。
- `baseline_quality`：探针不能全是烂算法。它不要求最强，但至少要有基本可用的拟合能力。
- `taxonomy_penalty`：避免 4 个算法都来自同一类方法，例如全是 GP。
- `missing_penalty`：缺 train/id/ood 指标越多，扣分越多。
- `explosion_penalty`：只要 ID/OOD NMSE 超过 100，就认为这个 run 对实际评价有爆炸风险。爆炸多的算法不能因为方差大而被奖励。

## timeout 的语义

`timed_out` 不等于失败。这里的 timeout 只是说明算法跑满了一小时预算。如果它在预算结束时已经落盘了 train/ID/OOD 指标，我们在 Probe-4 选择里把它记为 `success_budget_exhausted`。

这正是本轮实验要看的问题：给算法一小时，它在这个预算内能交出多好的结果。

## 这次大表额外加了什么

除了原始 NMSE，大表还加入了三类信息：

- 数值健康：finite 标记、log NMSE、train 到 ID 的退化、ID 到 OOD 的退化、NMSE > 100 爆炸标记。
- 表达式结构：表达式长度、token 数、AST 节点数、树深度、用了几个变量、变量覆盖率、用了哪些算子类别。
- 工程健康：是否有表达式 artifact、artifact 是否有效、是否能被 sympy 解析、原始状态、是否预算用尽、Probe-4 口径下是否成功。

## LLM 算法结果口径

`llmsr` 和 `drsr` 现在使用带物理语义背景的新批次结果，替换掉旧 E1 里不带语义的结果。
这个替换只发生在这两个算法上，其它 10 个算法仍使用原 Candidate-200 E1 结果。

语义批次的 prompt 会告诉模型目标物理量、候选变量语义角色集合和 dummy 变量数量，但不会告诉它具体哪个 `x_i` 对应哪个物理角色。
大表里的 `prompt_semantics_mode = physics_semantic_hidden_mapping` 就表示该行来自这个新口径。

## 表达式 artifact 覆盖情况

- 表达式 artifact 完整覆盖的算法：`drsr, dso, gplearn, llmsr, pyoperon, pysr`。
- 表达式 artifact 部分覆盖的算法：`tpsr`。
- 当前 2400 行 digest 里存在、但 clean artifact 归档完全没有覆盖的算法：`e2esr, imcts, qlattice, ragsr, udsr`。
- 对 artifact 缺失的算法，大表不会伪造表达式复杂度，而是把 `expression_artifact_available` 标成 `0`。

## 当前最需要警惕的信号

按 `ID/OOD NMSE > 100` 的爆炸率看，风险最高的算法是：

| algorithm | explosion_gt_100_rate | artifact_available_rate | median_ast_nodes |
| --- | ---: | ---: | ---: |
| `ragsr` | 0.585 | 0.000 |  |
| `tpsr` | 0.250 | 0.990 | 29 |
| `pysr` | 0.205 | 1.000 | 15 |
| `e2esr` | 0.140 | 0.000 |  |
| `gplearn` | 0.140 | 1.000 | 15.5 |

## 文件说明

- `probe4_run_level_big_table.csv`：2400 行大表，每行是一个数据集和一个算法的结果，包含 NMSE、派生数值指标、表达式结构指标和工程健康指标。
- `probe4_algorithm_level_summary.csv`：按算法汇总后的表，用来快速看每个算法是否稳定、是否容易爆炸、表达式复杂度是否过高。
- `probe4_dataset_level_summary.csv`：按数据集汇总后的表，用来快速看哪些数据集在 12 个算法之间拉开了差异。
- `probe4_metric_dictionary_human.md`：每个字段的人话解释。

## 后续怎么用

先用 `probe4_algorithm_level_summary.csv` 排除明显不适合作主探针的算法，再用 `probe4_run_level_big_table.csv` 计算组合级分数。最终 shortlist 不能只看一个总分，还要检查 PySR/no-PySR、RAGSR/no-RAGSR、with/without DRSR 的敏感性。
