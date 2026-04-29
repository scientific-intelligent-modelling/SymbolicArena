# Probe-4 选择方法说明

本文档说明如何基于当前 `200 datasets x 12 algorithms x 1 seed` 的实验结果，离线选择后续 `4 probes x 664 datasets x 3 seeds` 中使用的 4 个探针算法。

当前版本是 `NMSE-only` 方法说明：`20260429` 汇总表只包含 `train/valid/id/ood NMSE`，没有统一的 `runtime/status/failure_reason` 字段。因此本文档中的稳定性主要指“指标可有限评估的稳定性”，不是完整运行稳定性。完整版本应在 raw result 表补齐后，把 runtime、timeout、failure 类型纳入评分。

## 1. 目标

Probe-4 的目标不是选择 4 个 leaderboard 最强算法，而是选择 4 个能够用于数据集筛选的代表性探针。理想的 4 个探针应同时满足：

- **稳定性**：在 200 个 candidate 数据集上，大多数任务能产生有限的 ID/OOD 指标。
- **区分度**：在不同数据集上的表现差异足够大，能帮助发现“有信息量”的数据集。
- **互补性**：4 个探针之间失败模式不完全重合，避免只测同一种 inductive bias。
- **覆盖性**：在不同 family/subgroup 上都能给出可用结果，不被单一来源支配。
- **方法族多样性**：尽量覆盖 GP、RL、MCTS、预训练神经方法、hybrid 方法等不同 SR 路线。

因此，Probe-4 选择是一个 **multi-objective selection** 问题，不是按平均 NMSE top-4 直接排序。

## 2. 输入文件

主输入表：

```text
exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv
```

每行是一个 `(dataset_id, algorithm)` 的 1 seed 结果，核心列为：

```text
dataset_id
dataset_name
family
srsd_variant
algorithm
train_nmse
valid_nmse
id_nmse
ood_nmse
```

候选数据集 metadata：

```text
exp-planning/02.E1选择验证/generated/candidate200_unified.csv
```

该表用于补充：

```text
family
subgroup
basename
selection_mode
candidate_advantage_side
```

当前实现脚本：

```text
check/analyze_probe4_selection_metrics.py
```

当前输出目录：

```text
exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/probe4_selection_nmse_only/
```

## 3. 候选算法集合

12 个已汇总算法中，当前选择 Probe-4 时先排除：

```text
drsr
llmsr
```

原因：

- `llmsr` 已经是第一轮 `664 -> 200` 的原始 probe 之一，继续纳入 4-probe 会增加选择闭环偏置。
- `drsr` 与 LLM 配置、采样链路和 `llmsr` 关系较近，当前阶段先不作为非 LLM probe 候选。

剩余 10 个候选算法：

| algorithm | taxonomy |
| --- | --- |
| `dso` | `rl_policy` |
| `e2esr` | `pretrained_neural` |
| `gplearn` | `classic_gp` |
| `imcts` | `mcts` |
| `pyoperon` | `evolutionary_gp` |
| `pysr` | `evolutionary_gp` |
| `qlattice` | `graph_hybrid` |
| `ragsr` | `rag_hybrid` |
| `tpsr` | `pretrained_neural` |
| `udsr` | `rl_hybrid` |

## 4. 预处理规则

对每条 run：

1. 将空字符串、`None`、`nan`、`NaN`、非有限值统一视为缺失。
2. 负 NMSE 视为非法缺失。
3. 对 `dataset_id` 形如 `g0001` 的行，解析 `global_index = 1`，再从 `candidate200_unified.csv` 回填 metadata。
4. 当前不直接删除爆炸值。只要是有限数，就保留并通过 log clipping 降低其对评分的支配性。

## 5. NMSE 到 log score 的转换

设算法为 `a`，数据集为 `d`。

对任意 NMSE 值 `x`，先做裁剪 log 变换：

```text
log_nmse(x) = clip(log10(max(x, 1e-12)), -12, 12)
```

其中：

```text
LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12
LOG_CLIP_MAX = 12
```

解释：

- `1e-12` 防止接近 0 的 NMSE 让 log 无界变小。
- `[-12, 12]` 防止数值爆炸任务支配整个评分。
- log 分数越小，表示 NMSE 越好。

对每个 `(a, d)`，定义 ID/OOD 综合 log score：

```text
s(a,d) = 0.5 * log_nmse(id_nmse(a,d)) + 0.5 * log_nmse(ood_nmse(a,d))
```

若 `id_nmse` 或 `ood_nmse` 任一缺失，则 `s(a,d)` 缺失。

注意：Probe-4 选择不是直接选 `s(a,d)` 最小的算法，而是使用 `s(a,d)` 的分布来衡量稳定性、区分度、互补性和覆盖性。

## 6. 单算法指标

设 candidate 数据集总数为：

```text
N = 200
```

### 6.1 指标完整率

对每个算法 `a`：

```text
finite_train_rate(a) = count_d finite(train_nmse(a,d)) / N
finite_valid_rate(a) = count_d finite(valid_nmse(a,d)) / N
finite_id_rate(a)    = count_d finite(id_nmse(a,d)) / N
finite_ood_rate(a)   = count_d finite(ood_nmse(a,d)) / N
```

ID/OOD 同时可评估率：

```text
finite_id_ood_rate(a)
  = count_d [finite(id_nmse(a,d)) and finite(ood_nmse(a,d))] / N
```

核心三类 split 全部可评估率：

```text
train_id_ood_present_rate(a)
  = count_d [finite(train_nmse(a,d))
             and finite(id_nmse(a,d))
             and finite(ood_nmse(a,d))] / N
```

### 6.2 爆炸率

当前使用阈值：

```text
EXPLOSION_THRESHOLD = 100
```

ID/OOD 爆炸率：

```text
id_ood_explosion_rate_gt_100(a)
  = count_d [id_nmse(a,d) > 100 or ood_nmse(a,d) > 100] / N
```

train/ID/OOD 爆炸率：

```text
train_id_ood_explosion_rate_gt_100(a)
  = count_d [train_nmse(a,d) > 100
             or id_nmse(a,d) > 100
             or ood_nmse(a,d) > 100] / N
```

这里的爆炸率不直接作为硬过滤条件。原因是 Probe 的一部分价值就是识别“哪些数据集会让某类方法数值崩溃”。但爆炸率会在人工审计时作为风险信号。

### 6.3 误差分布统计

令 `S_a = {s(a,d) | s(a,d) finite}`。

```text
median_combined_log_id_ood_nmse(a) = median(S_a)
mean_combined_log_id_ood_nmse(a)   = mean(S_a)
std_combined_log_id_ood_nmse(a)    = std(S_a)
iqr_combined_log_id_ood_nmse(a)    = Q3(S_a) - Q1(S_a)
```

其中 `std` 和 `iqr` 用于衡量该算法在不同数据集上的表现波动。波动越大，通常说明它能更好地区分数据集难度和方法失效模式。

### 6.4 family/subgroup 级别差异

对每个 family `f`：

```text
mean_score(a,f) = mean_{d in f} s(a,d)
```

定义 family 级差异：

```text
family_mean_score_std(a) = std_f mean_score(a,f)
```

对每个 subgroup `g`：

```text
mean_score(a,g) = mean_{d in g} s(a,d)
```

定义 subgroup 级差异：

```text
subgroup_mean_score_std(a) = std_g mean_score(a,g)
```

这两个指标衡量算法是否能在不同来源和子类型之间拉开差异。

## 7. Coverage 指标

对算法 `a` 和 group `g`，group 可以是 family 或 subgroup。

```text
finite_group_rate(a,g)
  = count_{d in g} finite(s(a,d)) / count_{d in g} 1
```

如果某个 group 内至少 90% 数据集可评估，则认为算法覆盖了该 group：

```text
covered(a,g) = 1[finite_group_rate(a,g) >= 0.9]
```

family 覆盖率：

```text
family_coverage_rate_ge_0_9(a)
  = count_f covered(a,f) / count_f 1
```

subgroup 覆盖率：

```text
subgroup_coverage_rate_ge_0_9(a)
  = count_g covered(a,g) / count_g 1
```

综合 coverage score：

```text
coverage_score(a)
  = 0.60 * family_coverage_rate_ge_0_9(a)
  + 0.40 * subgroup_coverage_rate_ge_0_9(a)
```

family 权重更高，因为后续 664 全量筛选必须避免被某个 benchmark source 主导。

## 8. Stability 指标

当前 `NMSE-only` 版本中，稳定性定义为可评估指标完整性：

```text
operational_stability_score(a)
  = 0.60 * finite_id_ood_rate(a)
  + 0.40 * train_id_ood_present_rate(a)
```

解释：

- `finite_id_ood_rate` 权重更高，因为后续数据集筛选主要依赖 ID/OOD 泛化表现。
- `train_id_ood_present_rate` 保留 train 维度，但不要求 valid；Probe-4 选择主要依赖训练可回放、ID 泛化和 OOD 外推三类结果。

限制：

```text
operational_stability_score
```

当前还不包含：

```text
runtime
timeout_type
failure_reason
artifact_integrity
minute_snapshot_integrity
```

这些字段应在 raw result 表补齐后进入完整版本。

## 9. Discrimination 指标

首先对四个区分度分量做 max-normalization。

设所有候选算法集合为 `A`：

```text
dataset_discrimination_std_norm(a)
  = std_combined_log_id_ood_nmse(a)
    / max_{b in A} std_combined_log_id_ood_nmse(b)
```

```text
dataset_discrimination_iqr_norm(a)
  = iqr_combined_log_id_ood_nmse(a)
    / max_{b in A} iqr_combined_log_id_ood_nmse(b)
```

```text
family_discrimination_norm(a)
  = family_mean_score_std(a)
    / max_{b in A} family_mean_score_std(b)
```

```text
subgroup_discrimination_norm(a)
  = subgroup_mean_score_std(a)
    / max_{b in A} subgroup_mean_score_std(b)
```

综合区分度：

```text
discrimination_score(a)
  = 0.45 * dataset_discrimination_std_norm(a)
  + 0.25 * dataset_discrimination_iqr_norm(a)
  + 0.15 * family_discrimination_norm(a)
  + 0.15 * subgroup_discrimination_norm(a)
```

权重解释：

- 数据集级 `std` 权重最高，表示该算法能否在 200 个任务上产生足够分散的响应。
- `iqr` 用于降低少量极端爆炸值对 `std` 的单独支配。
- family/subgroup 差异用于保证这种区分不是只来自个别数据点，而是在来源结构上也有可解释差异。

## 10. Baseline Quality 指标

Probe 不是 leaderboard，但探针不能整体太差，否则只会测“谁都失败”。因此保留一个低权重的 baseline quality。

由于 log score 被裁剪到 `[-12, 12]`，定义：

```text
baseline_quality_score(a)
  = clip((12 - median_combined_log_id_ood_nmse(a)) / 24, 0, 1)
```

等价写法：

```text
baseline_quality_score(a)
  = clip((LOG_CLIP_MAX - median_score(a))
         / (LOG_CLIP_MAX - LOG_CLIP_MIN), 0, 1)
```

该指标越高，表示算法在 200 个数据集上的典型 ID/OOD NMSE 越低。

权重较低，避免 Probe-4 退化成“平均性能最强 top-4”。

## 11. 单算法综合分

单算法可用分：

```text
available_metric_score(a)
  = 0.15 * operational_stability_score(a)
  + 0.30 * discrimination_score(a)
  + 0.15 * coverage_score(a)
  + 0.05 * baseline_quality_score(a)
  - 0.10 * (1 - finite_id_ood_rate(a))
```

解释：

- `discrimination_score` 权重最高，因为探针核心用途是挑数据集，而不是争 leaderboard。
- `stability` 和 `coverage` 保证算法能在大多数数据和来源上工作。
- `baseline_quality` 只占 5%，避免选出完全无效但波动很大的算法。
- `finite_id_ood_rate` 缺失惩罚用于压低大量缺失结果的算法。

该分数用于单算法排序，但不直接决定最终 4 个探针。最终选择要看组合互补性和方法族覆盖。

## 12. 两两互补性

对两个算法 `a,b`，取共同有 `s(a,d)` 和 `s(b,d)` 的数据集集合：

```text
D_ab = {d | finite(s(a,d)) and finite(s(b,d))}
```

在 `D_ab` 上计算 Spearman rank correlation：

```text
rho(a,b)
  = Pearson(rank({s(a,d)}_{d in D_ab}),
            rank({s(b,d)}_{d in D_ab}))
```

互补性定义为：

```text
complementarity_score(a,b) = 1 - abs(rho(a,b))
```

解释：

- `rho` 接近 `1`：两个算法在数据集上的好坏排序高度一致，互补性低。
- `rho` 接近 `-1`：两个算法排序高度反向，说明两者仍强绑定在同一差异轴上，互补性也低。
- `rho` 接近 `0`：两个算法对数据集的响应弱相关，互补性最高。

这个定义的目的不是找“谁赢谁输相反”的算法，而是找失败模式不被同一个单轴解释的算法。

## 13. 四算法组合指标

对一个 4 算法组合：

```text
C = {a1, a2, a3, a4}
```

组合稳定性：

```text
stability(C) = mean_{a in C} operational_stability_score(a)
```

组合区分度：

```text
discrimination(C) = mean_{a in C} discrimination_score(a)
```

组合覆盖性：

```text
coverage(C) = mean_{a in C} coverage_score(a)
```

组合 baseline quality：

```text
quality(C) = mean_{a in C} baseline_quality_score(a)
```

组合 ID/OOD 有效率：

```text
finite(C) = mean_{a in C} finite_id_ood_rate(a)
```

组合互补性：

```text
complementarity(C)
  = mean_{a,b in C, a < b} complementarity_score(a,b)
```

方法族多样性：

```text
taxonomy_diversity_score(C)
  = count_unique_taxonomies(C) / 4
```

方法族冗余惩罚：

```text
taxonomy_redundancy_penalty(C)
  = max(0, max_taxonomy_count(C) - 2) * 0.10
```

当前没有 runtime/status，因此 practical cost 设为中性常数：

```text
practical_cost_score(C) = 0.5
```

## 14. 四算法组合总分

最终组合分数：

```text
combo_score(C)
  = 0.15 * stability(C)
  + 0.30 * discrimination(C)
  + 0.25 * complementarity(C)
  + 0.15 * coverage(C)
  + 0.10 * practical_cost_score(C)
  + 0.05 * quality(C)
  + 0.05 * taxonomy_diversity_score(C)
  - 0.10 * (1 - finite(C))
  - taxonomy_redundancy_penalty(C)
```

权重解释：

- `discrimination` 最高：Probe-4 的第一任务是挑出能区分算法的数据集。
- `complementarity` 第二高：4 个 probe 不能测同一种失败模式。
- `stability` 和 `coverage` 保证后续 `664 x 3 seeds` 可执行。
- `practical_cost` 当前没有真实数据，暂设中性；补齐 runtime 后应替换为真实成本分。
- `quality` 只占低权重，防止把 probe 选择变成 leaderboard top-4。
- `taxonomy_diversity` 和冗余惩罚保证方法族多样性。

## 15. 当前输出文件解释

当前脚本生成：

```text
probe4_algorithm_scores_nmse_only.csv
```

单算法指标表，包含稳定性、区分度、覆盖性、爆炸率、综合分。

```text
probe4_pairwise_complementarity_nmse_only.csv
```

两两算法互补性表，包含 Spearman 相关、共同数据集数量、互补性分数。

```text
probe4_combo_scores_nmse_only.csv
```

所有 `C(10,4)=210` 个四算法组合的综合评分。

```text
probe4_family_coverage_nmse_only.csv
```

算法在各 family 上的覆盖率、爆炸率和 log NMSE 统计。

```text
probe4_selection_report.md
```

自动生成的摘要报告，列出单算法 top-10 和四算法组合 top-10。

## 16. 当前 NMSE-only 结果摘要

单算法可用分前 10：

| rank | algorithm | taxonomy | score | finite_id_ood | explosion_gt_100 | discrimination | coverage |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `pysr` | `evolutionary_gp` | 0.6004 | 0.985 | 0.205 | 0.984 | 0.867 |
| 2 | `imcts` | `mcts` | 0.4812 | 0.990 | 0.055 | 0.574 | 0.891 |
| 3 | `udsr` | `rl_hybrid` | 0.4690 | 0.990 | 0.015 | 0.572 | 0.782 |
| 4 | `gplearn` | `classic_gp` | 0.4488 | 1.000 | 0.140 | 0.408 | 1.000 |
| 5 | `tpsr` | `pretrained_neural` | 0.4479 | 0.975 | 0.250 | 0.442 | 0.976 |
| 6 | `dso` | `rl_policy` | 0.4406 | 0.990 | 0.050 | 0.442 | 0.891 |
| 7 | `ragsr` | `rag_hybrid` | 0.4148 | 0.885 | 0.585 | 0.649 | 0.539 |
| 8 | `pyoperon` | `evolutionary_gp` | 0.3687 | 0.975 | 0.035 | 0.179 | 0.976 |
| 9 | `e2esr` | `pretrained_neural` | 0.3655 | 0.695 | 0.140 | 0.601 | 0.555 |
| 10 | `qlattice` | `graph_hybrid` | 0.3610 | 0.950 | 0.030 | 0.228 | 0.844 |

四算法组合前 10：

| rank | combo | score | stability | discrimination | complementarity | coverage | finite_id_ood |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `imcts;pysr;ragsr;udsr` | 0.7895 | 0.960 | 0.695 | 0.793 | 0.770 | 0.963 |
| 2 | `imcts;pysr;ragsr;tpsr` | 0.7835 | 0.956 | 0.662 | 0.790 | 0.818 | 0.959 |
| 3 | `pysr;ragsr;tpsr;udsr` | 0.7812 | 0.956 | 0.662 | 0.793 | 0.791 | 0.959 |
| 4 | `gplearn;imcts;pysr;ragsr` | 0.7732 | 0.962 | 0.654 | 0.748 | 0.824 | 0.965 |
| 5 | `gplearn;pysr;ragsr;udsr` | 0.7718 | 0.962 | 0.653 | 0.754 | 0.797 | 0.965 |
| 6 | `dso;pysr;ragsr;udsr` | 0.7710 | 0.960 | 0.662 | 0.759 | 0.770 | 0.963 |
| 7 | `e2esr;imcts;pysr;ragsr` | 0.7694 | 0.886 | 0.702 | 0.816 | 0.713 | 0.889 |
| 8 | `imcts;pysr;qlattice;ragsr` | 0.7684 | 0.950 | 0.609 | 0.816 | 0.785 | 0.953 |
| 9 | `imcts;pysr;tpsr;udsr` | 0.7678 | 0.982 | 0.643 | 0.673 | 0.879 | 0.985 |
| 10 | `dso;imcts;pysr;ragsr` | 0.7648 | 0.960 | 0.662 | 0.722 | 0.797 | 0.963 |

## 17. 如何使用当前结果选 Probe-4

当前结果应作为 Probe-4 的第一轮 shortlist，而不是最终无人工审计的自动决策。

推荐使用流程：

1. 先查看 `probe4_combo_scores_nmse_only.csv` 的 top 组合。
2. 对 top 组合中的每个算法检查单算法指标，重点看 `finite_id_ood_rate`、`id_ood_explosion_rate_gt_100`、`coverage_score`。
3. 检查组合是否包含足够多的方法族，避免 4 个 probe 过度集中在同一类 GP 或同一类神经方法。
4. 对明显高风险算法做人工审计。例如当前 `ragsr` 互补性和区分度高，但 `finite_id_ood_rate=0.885`、`coverage=0.539`，不能只因为进入 top combo 就直接冻结。
5. 在 raw result 表补齐 `runtime/status/failure_reason` 后，重新计算完整版本，把真实运行成本和失败类型纳入分数。
6. 最终选定 4 个 probe 后，进入 `4 x 664 x 3 seeds`，再用多 seed 结果筛选最终 Core-50。

当前 `NMSE-only` 结果中，排名最高的组合是：

```text
imcts;pysr;ragsr;udsr
```

但这个组合不应被直接视为最终答案。它的意义是：

- `pysr` 提供强区分度和经典强 baseline。
- `imcts` 提供 MCTS 路线。
- `udsr` 提供 RL-hybrid 路线，且当前 finite 率较好。
- `ragsr` 提供高互补性，但需要额外审计其覆盖和失败原因。

如果后续审计发现 `ragsr` 的缺失主要来自集成问题，应先修工具集再重算。如果缺失来自算法本身在当前数据分布下不稳定，则需要在 top 组合中比较 `tpsr`、`gplearn`、`dso` 等替代项。

## 18. 当前方法的已知限制

1. 当前没有 runtime/status，因此不能判断某个算法是“正常失败”“超时有结果”“超时报错”还是“集成层失败”。
2. 当前没有 symbolic fidelity、表达式复杂度、分钟级快照完整性，因此不能衡量探针输出是否适合后续精细分析。
3. 当前只基于 200 candidate 数据集。这个 200 本身是 contrastive pool，不是原始 664 的无偏随机样本。
4. 当前互补性基于 log NMSE 排名相关，不包含表达式结构互补性。
5. 当前 `valid_extreme_error` 一类有限爆炸结果被保留为 finite，会提高“可评估率”，但需要结合爆炸率审计。

因此，本文档给出的流程适合作为 Probe-4 的可复现选择框架；最终 Probe-4 冻结前仍需补充 raw result 层面的失败类型和运行成本指标。

## 19. 后续完整版本应补充的字段

完整 Probe-4 选择建议在 raw result 消化后补充：

```text
status
failure_reason
timeout_type
wall_time
budget_exhausted
valid_output
artifact_exists
minute_snapshot_count
final_expression
expression_complexity
```

对应新增指标：

```text
successful_artifact_rate
normal_timeout_recovery_rate
median_wall_time
cost_score
snapshot_integrity_score
complexity_sanity_score
```

补齐后，`practical_cost_score` 不再设为 `0.5`，而应由真实 runtime 和资源消耗计算。
