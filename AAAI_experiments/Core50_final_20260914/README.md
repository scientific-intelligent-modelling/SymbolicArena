# Core50 Final Release 20260914

本目录是 AAAI Core-50 当前唯一的最终交付目录。它只组合已经通过来源校验的当前结果；不包含中间实验，也不会用 EFF 补跑的 terminal result 覆盖原正式实验的最终公式。

## 实验范围

- 数据集：50 个 Core-50 任务，完整保留 `metadata.yaml`、`formula.py` 与四个数据切分。
- 算法：15 个，分别为 DRSR、DSO、E2ESR、FePySR、gplearn、iMCTS、JAXSR、LLM-SR、PyOperon、PySR、QLattice、RAG-SR、SymbolFit、TPSR、uDSR。
- 条件：`clean`、`noise001`、`noise005`。
- 随机种子：520、521、522。
- 最终运行：每个条件 2250 条，共 6750 条。
- EFF：每个条件 2250 条可审计轨迹、405000 条逐分钟记录、2700 条算法分钟曲线，三个条件均为 15/15 算法可用。

## 目录说明

- `datasets/`：50 个数据集与 Ground Truth 源文件。
- `ground_truth/`：当前 Ground Truth 参考式及 Opus5 化简证据。
- `results/<condition>/raw_results.jsonl.gz`：逐 algorithm-task-seed 原始最终结果。
- `results/<condition>/run_final.csv`：逐运行最终 ID、OOD、SYM、MIN 等计算输入与结果。
- `results/<condition>/six_axis.csv`：15 个算法的最终六轴汇总。
- `results/<condition>/task_stability.csv`：逐 algorithm-task 的跨种子 STAB 证据。
- `results/<condition>/opus5_prediction.jsonl`：预测公式化简证据。
- `results/<condition>/opus5_equivalence.jsonl`：预测式与 Ground Truth 的等价性证据。
- `results/<condition>/opus5_structure.jsonl`：跨种子结构一致性证据。
- `results/<condition>/run_trajectories_180min.csv.gz`：逐运行 180 分钟原生 incumbent 轨迹。
- `results/<condition>/id_ood_eff_minute.csv.gz`：逐 algorithm-task-seed-minute 的 ID/OOD 数值质量与 EFF。
- `results/<condition>/curves_2700.csv`：15 算法乘 180 分钟的聚合曲线。
- `results/<condition>/run_eff_status.csv`：逐运行 EFF 完整性与来源状态。
- `provenance/`：来源清单、SHA256、EFF v3 审计、原始远端补跑包，以及 gplearn 结构重裁的完整提示词、请求和响应。
- `code/`：生成、校验、聚合所用的关键代码快照。

## EFF 正式口径

EFF 使用 `algorithm-native internal best-so-far.v1`。每分钟只按算法原生 loss、reward、score 或模型选择规则确定 incumbent；统一评估器随后计算 ID/OOD，测试质量不参与候选选择。首次有效候选前记 0，更新之间及提前结束后向前延续；缺少证据则应标记 unavailable，禁止使用未来公式或 legacy/test-quality fallback。

为补齐这一口径，精确重跑了 474 条：clean 165 条、noise001 154 条、noise005 155 条。这些重跑全部冻结为 `eff_only`，原正式实验的最终公式和 ID/OOD/SYM/MIN/STAB 均保持不变。

## Opus5 状态与 gplearn 重裁

新增大模型请求全部使用 Routify 的 `claude-opus-5`、`xhigh`、非流式接口；未使用 yapi。连同本次 gplearn 重裁，当前账本记录 3115 次尝试，其中 2361 次通过严格校验、754 次失败，估算费用为 97.7543616 元。

`gplearn / strogatz_barmag2 / g0029 / s520-s521` 曾因普通显示表达式丢失 gplearn protected `log/sqrt` 语义而无法裁决。修正提示词并绑定原生 prefix tree 后，Opus5 在第一次请求即通过 Draft-07 严格校验，返回 `different_structure`，置信度 0.96。因此：

- clean 的该任务结构一致性为 0，任务 STAB 为 0；
- clean 的 gplearn STAB 为 23.577217230298906；
- clean、noise001、noise005 的六轴均为 15/15 正式就绪；
- 新裁决完整证据保存在 `provenance/gplearn_structure_resolution_20260914/`；此前六次失败尝试保留在 `provenance/opus_unresolved_clean_structure/` 作为历史审计证据。

本发布包含最终 SYM、MIN、STAB 和完整逐分钟 ID/OOD/EFF；它不宣称已经计算逐分钟 SYM、MIN、STAB。

## 完整性校验

在本目录执行：

```bash
sha256sum -c SHA256SUMS
```

总体状态见 `FINAL_STATUS.json`，发布器的逐项检查见 `PUBLICATION_REPORT.json`。
