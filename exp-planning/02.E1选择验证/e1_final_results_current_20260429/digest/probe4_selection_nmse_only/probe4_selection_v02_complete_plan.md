# Probe-4 选择完整方案 v0.2

本文档给出一套完整、可复现、可审计的流程，用于从当前 `200 datasets x 12 algorithms x 1 seed` 实验结果中选择后续 `4 probes x 664 datasets x 3 seeds` 使用的 4 个探针算法。

结论先行：已有 `NMSE-only v0.1` 方案适合作为第一轮 shortlist，但不应直接 freeze 最终 4 个 probe。v0.2 的核心修改是：把“单算法平均区分度”升级为“4 个 probe 对完整算法面板的信息量保真”，并加入性能梯度、top-k 数据集覆盖、爆炸惩罚和 sensitivity audit。

## 1. 总体目标

Probe-4 不是选 4 个最强算法，也不是选 4 个平均 NMSE 最低的算法。它的目标是选出 4 个能代表更大算法面板观察数据集的探针，用于后续在 664 个数据集上低成本评估数据集的信息量。

最终 4 个探针应该满足：

- 稳定：大多数数据集上能产生有限、可回放、可比较的指标。
- 有区分度：能把不同难度和不同失败模式的数据集拉开。
- 保真：用这 4 个算法计算出来的数据集信息量，尽量接近完整 12 算法面板。
- 互补：4 个算法的失败模式和性能梯度不高度重复。
- 覆盖：高信息量数据集不能集中在单一 family/subgroup。
- 可解释：方法族覆盖 GP、RL、MCTS、神经/预训练、hybrid 等不同 SR 路线。
- 可执行：后续 `4 x 664 x 3` 的计算成本、失败率、恢复能力可控。

## 2. 当前 v0.1 的 review

已有 v0.1 做对了这些事：

- 没有把 Probe-4 当成 leaderboard top-4。
- 明确当前是 `NMSE-only`，没有假装已有 runtime/status/failure reason。
- 枚举了 `C(10,4)=210` 个组合，工程上简单可靠。
- 排除了 `llmsr` 和 `drsr`，降低 LLM 链路和第一阶段 probe 的闭环偏置。
- 引入了 stability、discrimination、complementarity、coverage、taxonomy diversity。

但 v0.1 不能直接 freeze，原因是：

- `discrimination(C)` 只是 4 个单算法 discrimination 的平均，不是组合对完整 panel 的保真。
- `coverage(C)` 只是算法 finite coverage，不是 4 个 probe 选出的高信息量数据集的 family/subgroup 覆盖。
- `complementarity = 1 - abs(rho)` 会把强反向失败模式也判成低互补，过于保守。
- `practical_cost_score = 0.5` 是常数，对排序没有作用，不应作为有效指标。
- 爆炸率只被报告，没有进入 penalty，容易奖励“靠爆炸产生高方差”的算法。
- 没有显式使用 `train -> id -> ood` 的性能梯度。
- 没有做 `allow/forbid PySR`、`allow/forbid RAGSR`、`with/without DRSR` 的 sensitivity audit。

v0.2 保留 v0.1 的可执行框架，但把核心评分改成组合级保真评分。

## 3. 输入数据

主输入表：

```text
exp-planning/02.E1选择验证/e1_final_results_current_20260429/digest/e1_12_dataset_algorithm_nmse_table.csv
```

每行表示一个 `(dataset_id, algorithm)` 的 1 seed 结果：

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

候选集 metadata：

```text
exp-planning/02.E1选择验证/generated/candidate200_unified.csv
```

用于补齐：

```text
family
subgroup
basename
selection_mode
candidate_advantage_side
```

当前 12 个算法：

```text
dso
drsr
e2esr
gplearn
imcts
llmsr
pyoperon
pysr
qlattice
ragsr
tpsr
udsr
```

默认 Probe-4 候选算法集合：

```text
P = all_algorithms - {llmsr, drsr}
```

即：

```text
dso
e2esr
gplearn
imcts
pyoperon
pysr
qlattice
ragsr
tpsr
udsr
```

## 4. 符号定义

设：

```text
D = 200 个 candidate 数据集
A = 12 个已跑算法
P = 默认 10 个 Probe-4 候选算法
C = 一个 4 算法组合，C subset P, |C| = 4
i = 数据集索引
a = 算法索引
```

原始 NMSE：

```text
train_nmse(i,a)
valid_nmse(i,a)
id_nmse(i,a)
ood_nmse(i,a)
```

有限性指示：

```text
m_split(i,a) = 1 if split_nmse(i,a) is finite else 0
```

ID/OOD 同时有效：

```text
m_id_ood(i,a) = m_id(i,a) * m_ood(i,a)
```

核心三类 split 全有效：

```text
m_train_id_ood(i,a) = m_train(i,a) * m_id(i,a) * m_ood(i,a)
```

## 5. 预处理

对所有 NMSE 字段：

- 空字符串、`None`、`nan`、`NaN`、`inf`、`-inf` 视为缺失。
- 负数 NMSE 视为非法缺失。
- 有限但极大的 NMSE 保留，不直接删除。
- 对 `dataset_id = g0001` 解析 `global_index = 1`，从 `candidate200_unified.csv` 回填 metadata。

爆炸阈值：

```text
EXPLOSION_THRESHOLD = 100
```

爆炸指示：

```text
e_id_ood(i,a) = 1[id_nmse(i,a) > 100 or ood_nmse(i,a) > 100]
e_train_id_ood(i,a) = 1[train_nmse(i,a) > 100
                        or id_nmse(i,a) > 100
                        or ood_nmse(i,a) > 100]
```

## 6. Log NMSE 变换

为避免 0 附近和极大爆炸值支配评分，所有 NMSE 先转成 clipped log：

```text
z(x) = clip(log10(max(x, 1e-12)), -12, 12)
```

其中：

```text
LOG_FLOOR = 1e-12
LOG_CLIP_MIN = -12
LOG_CLIP_MAX = 12
```

定义：

```text
z_train(i,a) = z(train_nmse(i,a))
z_id(i,a)    = z(id_nmse(i,a))
z_ood(i,a)   = z(ood_nmse(i,a))
```

如果原始 NMSE 缺失，则对应 `z_*` 缺失。

ID/OOD 综合 log score：

```text
s(i,a) = 0.5 * z_id(i,a) + 0.5 * z_ood(i,a)
```

`s(i,a)` 越小表示算法在该数据集上越好。但 Probe-4 不直接选择 `s` 最低的算法，而是使用 `s` 的分布、相对排序和失败模式来选择 probe。

## 7. 性能梯度

当前 Probe-4 口径只使用 `train/id/ood` 三个层次刻画性能退化路径；`valid_nmse` 保留为原始观测列，但不进入梯度特征。

定义两个梯度：

```text
delta_ti(i,a) = z_id(i,a)  - z_train(i,a)
delta_io(i,a) = z_ood(i,a) - z_id(i,a)
```

解释：

- `delta_ti`：训练到 ID test 的退化，反映从训练拟合到同分布测试的稳定性。
- `delta_io`：ID 到 OOD 的退化，反映外推和分布迁移能力。

组合梯度向量：

```text
g(i,a) = [delta_ti(i,a), delta_io(i,a)]
```

如果相关 split 缺失，则对应梯度缺失。

## 8. Teacher panel 设计

Probe-4 的核心是：4 个 probe 计算出的数据集信息量，要尽量接近更完整算法面板计算出的信息量。

因此需要定义 teacher panel。

默认同时计算三套 teacher：

```text
T_all12 = all 12 algorithms
T_no_llm = all algorithms - {llmsr, drsr}
T_no_stage1 = all algorithms - {llmsr, pysr}
```

三套 teacher 的用途：

- `T_all12`：完整实验结果视角。
- `T_no_llm`：排除 LLM 链路影响后的非 LLM/弱 LLM 视角。
- `T_no_stage1`：排除第一阶段 dual-probe 中的 `llmsr` 和 `pysr`，检查 circularity。

主评分建议使用：

```text
T_main = T_no_llm
```

同时报告 `T_all12` 和 `T_no_stage1` sensitivity。

## 9. 数据集信息量 I(i,S)

对任意算法集合 `S`，定义数据集 `i` 在 panel `S` 下的信息量。

### 9.1 算法间方差

ID 方差：

```text
V_id(i,S) = Var_{a in S, finite z_id(i,a)} z_id(i,a)
```

OOD 方差：

```text
V_ood(i,S) = Var_{a in S, finite z_ood(i,a)} z_ood(i,a)
```

ID/OOD gap 方差：

```text
gap(i,a) = z_ood(i,a) - z_id(i,a)
V_gap(i,S) = Var_{a in S, finite gap(i,a)} gap(i,a)
```

OOD 梯度方差：

```text
V_grad_io(i,S) = Var_{a in S, finite delta_io(i,a)} delta_io(i,a)
```

如果某个方差项的有效算法数少于 2，则该项记为 0。

### 9.2 有效性熵

对数据集 `i`，在 panel `S` 中统计 ID/OOD 有效率：

```text
p_valid(i,S) = mean_{a in S} m_id_ood(i,a)
```

二元熵：

```text
H_valid(i,S)
  = -p_valid(i,S) * log2(p_valid(i,S))
    -[1-p_valid(i,S)] * log2(1-p_valid(i,S))
```

当 `p_valid` 为 0 或 1 时，熵记为 0。

解释：

- 如果所有算法都有效或都无效，valid pattern 对区分算法帮助有限。
- 如果一部分算法有效、一部分算法无效，说明该数据集具有可评估性区分。
- 但该项权重必须低，避免把“失败多”当成主要信息量。

### 9.3 信息量总分

先对 `V_id/V_ood/V_gap/V_grad_io` 在 200 个数据集内做 min-max 归一化到 `[0,1]`：

```text
norm_X(i,S) = (X(i,S) - min_i X(i,S)) / (max_i X(i,S) - min_i X(i,S))
```

若分母为 0，则该项全记为 0。

数据集信息量：

```text
I(i,S)
  = 0.35 * norm_V_id(i,S)
  + 0.35 * norm_V_ood(i,S)
  + 0.15 * norm_V_gap(i,S)
  + 0.10 * norm_V_grad_io(i,S)
  + 0.05 * H_valid(i,S)
```

解释：

- `V_id` 和 `V_ood` 是主体，衡量算法面板是否对该数据集产生分歧。
- `V_gap` 衡量泛化差异。
- `V_grad_io` 衡量 OOD 退化模式差异。
- `H_valid` 只给 5%，因为可评估性差异有价值，但不能主导选择。

## 10. Discrimination Fidelity

v0.2 中，组合区分度不再是单算法 discrimination 的平均，而是：

```text
4 个 probe 对 teacher panel 数据集信息量排序的保真程度
```

对组合 `C` 和 teacher `T`：

```text
I_C(i) = I(i,C)
I_T(i) = I(i,T)
```

计算加权 Spearman 相关：

```text
rho_w(C,T) = WeightedSpearman(I_C, I_T; weights=w_i)
```

默认权重 `w_i` 使用 family-balanced 权重：

```text
w_i = 1 / count_family(family(i))
```

然后归一化到 `[0,1]`：

```text
rank_fidelity(C,T) = (rho_w(C,T) + 1) / 2
```

信息强度：

```text
info_strength(C) = mean_i I_C(i)
```

最终 discrimination fidelity：

```text
discrimination_fidelity(C,T)
  = 0.70 * rank_fidelity(C,T)
  + 0.30 * info_strength(C)
```

解释：

- `rank_fidelity` 回答：4 个 probe 认为哪些数据集有信息量，是否接近 teacher panel。
- `info_strength` 防止组合虽然相关性高，但自身信息量整体很低。
- 使用 family-balanced 权重，避免 `srsd` 或 `llm-srbench` 因数量多而支配相关性。

## 11. Algorithm Complementarity

v0.1 使用：

```text
1 - abs(rho)
```

问题是它把强反向失败模式也视为低互补。v0.2 改用混合行为距离。

对两个算法 `a,b`：

ID 排序距离：

```text
D_id(a,b) = [1 - Spearman(z_id(:,a), z_id(:,b))] / 2
```

OOD 排序距离：

```text
D_ood(a,b) = [1 - Spearman(z_ood(:,a), z_ood(:,b))] / 2
```

泛化 gap 距离：

```text
D_gap(a,b) = [1 - Spearman(gap(:,a), gap(:,b))] / 2
```

有效性模式距离：

```text
D_valid(a,b)
  = mean_i XOR(m_id_ood(i,a), m_id_ood(i,b))
```

梯度距离：

```text
D_gradient(a,b)
  = mean over k in {ti,io}
    [1 - Spearman(delta_k(:,a), delta_k(:,b))] / 2
```

如果某个 Spearman 的共同有效样本数少于阈值，例如 30，则该项记为 0.5 中性值，并在报告中标记低重叠。

两两互补性：

```text
D(a,b)
  = 0.35 * D_id(a,b)
  + 0.30 * D_ood(a,b)
  + 0.15 * D_gap(a,b)
  + 0.10 * D_valid(a,b)
  + 0.10 * D_gradient(a,b)
```

组合互补性：

```text
algorithm_complementarity(C)
  = mean_{a,b in C, a < b} D(a,b)
```

可选轴冗余惩罚：

```text
axis_redundancy_penalty(C)
  = mean_{a,b in C} max(0, abs(Spearman(s(:,a), s(:,b))) - 0.85) * 0.10
```

默认 v0.2 可以先不把 `axis_redundancy_penalty` 加进主分数，只在报告中审计；如果发现 top 组合中存在高度绑定的 pair，再启用。

## 12. Operational Stability

单算法稳定性：

```text
finite_id_ood_rate(a)
  = mean_i m_id_ood(i,a)
```

```text
train_id_ood_present_rate(a)
  = mean_i m_train_id_ood(i,a)
```

算法自身稳定性：

```text
stability(a)
  = 0.60 * finite_id_ood_rate(a)
  + 0.40 * train_id_ood_present_rate(a)
```

组合稳定性：

```text
operational_stability(C)
  = mean_{a in C} stability(a)
```

硬阈值建议：

```text
finite_id_ood_rate(a) >= 0.90
```

低于该阈值的算法仍可进入 shortlist，但必须标记为 `audit_required`，不能直接 freeze。

完整版本应补充：

```text
status_success_rate
timeout_recovered_rate
artifact_integrity_rate
minute_snapshot_integrity_rate
median_wall_time
```

当前 NMSE-only 版本暂不包含这些字段。

## 13. Coverage

Coverage 分两层：算法可评估覆盖和高信息数据集分布覆盖。

### 13.1 算法可评估覆盖

对算法 `a` 和 group `g`，group 可以是 family 或 subgroup：

```text
finite_group_rate(a,g)
  = count_{i in g} m_id_ood(i,a) / count_{i in g} 1
```

如果：

```text
finite_group_rate(a,g) >= 0.90
```

则认为算法覆盖该 group。

family 覆盖率：

```text
family_finite_coverage(a)
  = count_f covered(a,f) / count_f 1
```

subgroup 覆盖率：

```text
subgroup_finite_coverage(a)
  = count_g covered(a,g) / count_g 1
```

单算法 finite coverage：

```text
finite_coverage(a)
  = 0.60 * family_finite_coverage(a)
  + 0.40 * subgroup_finite_coverage(a)
```

组合 finite coverage：

```text
finite_coverage(C) = mean_{a in C} finite_coverage(a)
```

### 13.2 高信息数据集分布覆盖

用组合 `C` 计算每个数据集的信息量 `I_C(i)`，取 top-k：

```text
TopK(C,k) = top k datasets by I_C(i)
```

默认计算：

```text
k = 50
k = 100
```

令：

```text
q_family(C,k) = TopK(C,k) 中 family 分布
q_subgroup(C,k) = TopK(C,k) 中 subgroup 分布
```

目标分布可以用 Candidate-200 的分布：

```text
p_family = Candidate-200 family distribution
p_subgroup = Candidate-200 subgroup distribution
```

也可以用预设配额分布。当前建议先用 Candidate-200 分布，后续 Core-50 切分再用明确 family quota。

归一化 Jensen-Shannon 距离：

```text
JSD_norm(p,q) = JSD(p,q) / log(2)
```

分布覆盖：

```text
distribution_coverage(C,k)
  = 0.50 * [1 - JSD_norm(q_family(C,k), p_family)]
  + 0.50 * [1 - JSD_norm(q_subgroup(C,k), p_subgroup)]
```

组合 selected-dataset coverage：

```text
selected_dataset_coverage(C)
  = 0.40 * finite_coverage(C)
  + 0.30 * distribution_coverage(C,50)
  + 0.30 * distribution_coverage(C,100)
```

解释：

- `finite_coverage` 保证算法在各来源上能跑。
- `distribution_coverage` 保证 probe 认为高信息量的数据集不会集中在单一 family/subgroup。

## 14. Performance Gradient Diversity

组合中算法的性能退化路径应该不同。

两两梯度距离已经定义为：

```text
D_gradient(a,b)
```

组合梯度多样性：

```text
performance_gradient_diversity(C)
  = mean_{a,b in C, a < b} D_gradient(a,b)
```

该项权重不宜太高，因为它和 complementarity 有重叠。但它明确利用了 `train/id/ood` 三层信息，能识别：

- train 好但 ID 崩的算法。
- ID 好但 OOD 崩的算法。
- train/id/ood 都稳定的算法。

## 15. Baseline Quality

Probe 不应完全按性能选，但算法也不能整体太差。

对算法 `a`：

```text
median_score(a) = median_i s(i,a)
```

归一化质量分：

```text
quality(a)
  = clip((12 - median_score(a)) / 24, 0, 1)
```

组合质量：

```text
baseline_quality(C) = mean_{a in C} quality(a)
```

该项权重低，只用于防止选择整体无效但波动很大的算法。

## 16. Penalty

### 16.1 方法族冗余惩罚

每个算法有 taxonomy，例如：

```text
classic_gp
evolutionary_gp
mcts
rl_policy
rl_hybrid
pretrained_neural
rag_hybrid
graph_hybrid
```

组合中某个 taxonomy 出现次数过多时扣分：

```text
taxonomy_redundancy_penalty(C)
  = max(0, max_taxonomy_count(C) - 2) * 0.10
```

即同一 taxonomy 超过 2 个才开始扣分。

### 16.2 缺失惩罚

组合平均 ID/OOD 有效率：

```text
finite(C) = mean_{a in C} finite_id_ood_rate(a)
```

组合最弱算法有效率：

```text
finite_min(C) = min_{a in C} finite_id_ood_rate(a)
```

缺失惩罚：

```text
missing_invalid_penalty(C)
  = 0.10 * max(0, 0.95 - finite(C)) / 0.95
  + 0.10 * max(0, 0.90 - finite_min(C)) / 0.90
```

解释：

- 平均有效率低会扣分。
- 只要组合里有一个算法低于 0.90，也会额外扣分。

### 16.3 爆炸惩罚

单算法 ID/OOD 爆炸率：

```text
explosion_rate(a) = mean_i e_id_ood(i,a)
```

组合平均爆炸率：

```text
explosion_mean(C) = mean_{a in C} explosion_rate(a)
```

组合最大爆炸率：

```text
explosion_max(C) = max_{a in C} explosion_rate(a)
```

爆炸惩罚：

```text
explosion_penalty(C)
  = 0.10 * max(0, explosion_mean(C) - 0.10) / 0.90
  + 0.10 * max(0, explosion_max(C) - 0.15) / 0.85
```

解释：

- 平均爆炸率超过 10% 开始扣分。
- 任一算法爆炸率超过 15% 开始额外扣分。
- 这样不会直接删除 stress-sensitive 方法，但会阻止“靠爆炸产生高区分度”。

硬审计阈值：

```text
explosion_rate(a) <= 0.15
```

超过该阈值的算法可进入 shortlist，但必须标记 `explosion_audit_required`。

## 17. v0.2 组合总分

当前 NMSE-only v0.2 不使用 practical cost，因为没有 runtime/status。

主分数：

```text
combo_score(C,T)
  = 0.15 * operational_stability(C)
  + 0.30 * discrimination_fidelity(C,T)
  + 0.20 * algorithm_complementarity(C)
  + 0.15 * selected_dataset_coverage(C)
  + 0.05 * performance_gradient_diversity(C)
  + 0.05 * baseline_quality(C)
  - taxonomy_redundancy_penalty(C)
  - missing_invalid_penalty(C)
  - explosion_penalty(C)
```

默认主 teacher：

```text
T = T_no_llm
```

解释：

- `discrimination_fidelity` 是主项，保证 4 probes 能复现 teacher panel 对数据集信息量的判断。
- `algorithm_complementarity` 保证 probe 之间行为不同。
- `selected_dataset_coverage` 防止高信息 top-k 被某个 family/subgroup 吞掉。
- `operational_stability` 保证后续 664 全量跑得动。
- `performance_gradient_diversity` 使用 train/id/ood 三层变化。
- `baseline_quality` 防止选出整体过弱 probe。
- penalty 用来控制缺失、爆炸和方法族冗余。

## 18. Bootstrap Robust Score

为了避免 200 个 candidate 中某些 family 数量过大或少数极端数据集主导分数，需要做 family-balanced bootstrap。

每次 bootstrap：

1. 对每个 family 内部有放回采样，保持 family 数量不变。
2. 在采样后的 200 个样本上重新计算所有组合分。
3. 重复 `B` 次。

建议：

```text
B = 500
```

如果计算很快，可用：

```text
B = 1000
```

对每个组合：

```text
mean_score(C) = mean_b combo_score_b(C)
std_score(C)  = std_b combo_score_b(C)
```

稳健分数：

```text
robust_score(C)
  = mean_score(C) - 0.50 * std_score(C)
```

最终排序用：

```text
robust_score(C)
```

而不是单次 `combo_score(C)`。

## 19. Sensitivity Audit

最终 Probe-4 不能只看一个排序。必须至少做以下 sensitivity。

### 19.1 PySR circularity audit

原因：`pysr` 参与过第一阶段 `664 -> 200` dual-probe filtering。

计算两套：

```text
allow_pysr: P = all - {llmsr, drsr}
forbid_pysr: P = all - {llmsr, drsr, pysr}
```

比较：

```text
best_combo_score
top10_combo_overlap
top50_dataset_overlap_by_I_C
family/subgroup coverage
```

决策规则：

- 如果 `forbid_pysr` 最优组合 robust_score 只比 `allow_pysr` 低很少，例如 `< 0.02`，优先考虑 no-PySR 组合，因为 circularity 更小。
- 如果 `allow_pysr` 明显更好，则可以保留 PySR，但论文中必须说明 PySR 是通过行为保真被再次选中，并报告 no-PySR ablation。

### 19.2 RAGSR stability audit

原因：当前 v0.1 中 `ragsr` 互补性高，但 finite、coverage、explosion 风险也高。

计算：

```text
allow_ragsr
forbid_ragsr
```

决策规则：

- 如果 `ragsr` 所在组合只是因为 explosion/缺失带来高互补，应排除或先修工具链后重算。
- 如果 `ragsr` 在补充 penalty 后仍稳定 top-3，且缺失主要不是工程问题，可以保留作为 RAG/hybrid 代表。

### 19.3 DRSR inclusion audit

当前主候选排除 `drsr`。但应做一版：

```text
P = all - {llmsr}
```

也就是允许 `drsr` 进入候选。

目的：

- 判断排除 `drsr` 是否改变最终 Probe-4。
- 如果 `drsr` 稳定进入 top combo，需要明确它是 LLM-based、LLM-assisted、还是 DSR-family 方法，并决定是否允许它参与非 LLM probe。

### 19.4 Teacher panel sensitivity

同一个组合分别对三套 teacher 计算：

```text
T_all12
T_no_llm
T_no_stage1
```

输出：

```text
score_under_all12
score_under_no_llm
score_under_no_stage1
rank_under_all12
rank_under_no_llm
rank_under_no_stage1
```

稳定组合应该在不同 teacher 下 rank 不发生剧烈变化。

### 19.5 Hard-gate sensitivity

比较三种约束：

```text
soft: 只使用 penalty，不做硬过滤
medium: finite_id_ood >= 0.85, explosion <= 0.20
strict: finite_id_ood >= 0.90, explosion <= 0.15
```

如果最终组合只在 `soft` 下成立，在 `strict` 下消失，说明它风险过高。

## 20. 选择流程

完整流程如下。

### Step 1：生成矩阵

从 `e1_12_dataset_algorithm_nmse_table.csv` 构建：

```text
NMSE tensor: dataset x algorithm x split
valid mask: dataset x algorithm x split
metadata: dataset -> family/subgroup/srsd_variant
```

### Step 2：计算 log score 和梯度

计算：

```text
z_train, z_id, z_ood
s = 0.5*z_id + 0.5*z_ood
delta_ti, delta_io
gap = z_ood - z_id
```

### Step 3：计算 teacher 信息量

对：

```text
T_all12
T_no_llm
T_no_stage1
```

分别计算：

```text
I(i,T)
```

并输出 teacher 的 top-k 信息数据集分布。

### Step 4：枚举候选组合

默认候选集合：

```text
P = all - {llmsr, drsr}
```

枚举：

```text
all C subset P, |C| = 4
```

共：

```text
C(10,4) = 210
```

### Step 5：计算组合指标

对每个组合计算：

```text
operational_stability(C)
discrimination_fidelity(C,T_no_llm)
algorithm_complementarity(C)
selected_dataset_coverage(C)
performance_gradient_diversity(C)
baseline_quality(C)
taxonomy_redundancy_penalty(C)
missing_invalid_penalty(C)
explosion_penalty(C)
combo_score(C,T_no_llm)
```

### Step 6：Bootstrap 稳健排序

做 family-balanced bootstrap，输出：

```text
combo_score_mean
combo_score_std
robust_score
rank_by_robust_score
```

### Step 7：Sensitivity audit

至少运行：

```text
allow_pysr
forbid_pysr
allow_ragsr
forbid_ragsr
exclude_llmsr_only
teacher_all12
teacher_no_llm
teacher_no_stage1
soft_gate
medium_gate
strict_gate
```

### Step 8：形成 shortlist

输出前 5 到 10 个候选组合。

每个组合必须带：

```text
robust_score
rank under each sensitivity
finite/explosion risk
taxonomy composition
top50 family/subgroup distribution
top100 family/subgroup distribution
PySR circularity flag
RAGSR audit flag
DRSR inclusion sensitivity
```

### Step 9：人工审计

对 top shortlist 做人工审计：

- 是否过度依赖 PySR。
- 是否过度依赖 RAGSR 的缺失/爆炸。
- 是否有两个算法本质上同质。
- 是否覆盖至少 3 到 4 种方法族。
- 是否在主要 family 上都能运行。
- top-k 信息数据集是否被 SRSD 或 LLM-SRBench 主导。

### Step 10：冻结 Probe-4

满足以下条件才冻结：

- 主排序 robust_score top-3。
- 至少两套 teacher sensitivity 下 rank 仍然靠前。
- `strict` 或 `medium` hard-gate 下不崩。
- 如果包含 PySR，no-PySR ablation 必须报告。
- 如果包含 RAGSR，RAGSR 的缺失/爆炸原因必须解释。
- 组合 top50/top100 信息数据集 family/subgroup 覆盖可接受。

## 21. 输出文件设计

建议 v0.2 生成以下文件：

```text
probe4_v02_algorithm_metrics.csv
```

单算法指标：

```text
algorithm
taxonomy
finite_train_rate
finite_id_rate
finite_ood_rate
finite_id_ood_rate
train_id_ood_present_rate
id_ood_explosion_rate_gt_100
train_id_ood_explosion_rate_gt_100
median_combined_log_id_ood_nmse
baseline_quality
finite_coverage
stability
audit_flags
```

```text
probe4_v02_pairwise_behavior_distance.csv
```

两两行为距离：

```text
algorithm_a
algorithm_b
D_id
D_ood
D_gap
D_valid
D_gradient
D_total
overlap_id
overlap_ood
overlap_gradient
axis_redundancy
```

```text
probe4_v02_dataset_information.csv
```

每个 teacher/combination 的数据集信息量：

```text
dataset_id
family
subgroup
teacher_panel
I
V_id
V_ood
V_gap
V_grad_io
H_valid
rank
```

```text
probe4_v02_combo_scores.csv
```

主组合评分：

```text
combo
combo_score
robust_score
bootstrap_mean
bootstrap_std
operational_stability
discrimination_fidelity
algorithm_complementarity
selected_dataset_coverage
performance_gradient_diversity
baseline_quality
taxonomy_redundancy_penalty
missing_invalid_penalty
explosion_penalty
taxonomy
audit_flags
```

```text
probe4_v02_sensitivity_summary.csv
```

sensitivity 结果：

```text
combo
setting
rank
score
robust_score
top50_dataset_overlap_with_main
top100_dataset_overlap_with_main
family_jsd_top50
subgroup_jsd_top50
```

```text
probe4_v02_topk_distribution.csv
```

top-k 高信息数据集分布：

```text
combo
k
group_type
group_name
count
ratio
target_ratio
abs_diff
```

```text
probe4_v02_selection_report.md
```

人类可读报告：

- 最终推荐组合。
- top shortlist。
- 每个组合的风险。
- PySR/no-PySR 对比。
- RAGSR/no-RAGSR 对比。
- with/without DRSR 对比。
- teacher sensitivity。
- hard gate sensitivity。
- 最终是否可以 freeze 的判断。

## 22. 当前 v0.1 top combo 的处理方式

当前 v0.1 top-1：

```text
imcts;pysr;ragsr;udsr
```

它应该进入 v0.2 shortlist，但不能直接 freeze。

优点：

- `imcts` 提供 MCTS 路线。
- `pysr` 是强 evolutionary GP baseline，区分度高。
- `ragsr` 行为独特，可能提供 RAG/hybrid 失败模式。
- `udsr` 提供 RL-hybrid 路线，当前 finite 率较好。

风险：

- `pysr` 参与过第一阶段 `664 -> 200`，有 circularity 风险。
- `ragsr` 当前 finite 率、coverage、explosion 风险较高。
- 该组合在 v0.1 中受互补性强烈驱动，需要确认不是由缺失/爆炸制造的假互补。

v0.2 中必须和这些替代组合比较：

```text
imcts;pysr;tpsr;udsr
imcts;pysr;gplearn;udsr
imcts;pysr;dso;tpsr
imcts;pyoperon;tpsr;udsr
gplearn;imcts;pysr;udsr
dso;imcts;pysr;tpsr
```

如果 `imcts;pysr;ragsr;udsr` 在加入 v0.2 penalty 和 sensitivity 后仍稳定 top-3，才有资格成为最终 Probe-4。

## 23. 推荐决策规则

最终选择时按下面优先级决策。

### 23.1 主规则

选择：

```text
rank_by_robust_score <= 3
```

且：

```text
finite_mean >= 0.95
finite_min >= 0.90
explosion_mean <= 0.10
explosion_max <= 0.15
```

如果没有组合同时满足，则放宽为：

```text
finite_mean >= 0.93
finite_min >= 0.85
explosion_mean <= 0.15
explosion_max <= 0.20
```

但必须标记为 `conditional_freeze`。

### 23.2 PySR 规则

如果最佳组合包含 PySR：

- 必须报告 no-PySR 最佳组合。
- 如果 no-PySR robust_score 距离最佳组合小于 `0.02`，优先考虑 no-PySR。
- 如果 PySR 组合明显更优，则保留 PySR，但论文中报告 circularity audit。

### 23.3 RAGSR 规则

如果最佳组合包含 RAGSR：

- 必须检查 RAGSR 缺失和爆炸是否来自工具集问题。
- 如果是工具集问题，先修复再重跑 v0.2。
- 如果是算法本身问题，只有在 penalty 后仍 top-3 且 coverage 可接受时才保留。

### 23.4 DRSR 规则

如果 `exclude_llmsr_only` setting 中 DRSR 稳定进入 top-3：

- 需要单独判断 DRSR 是否属于本阶段允许的 probe 类型。
- 如果允许，应把 `exclude {llmsr, drsr}` 改成只排除 `llmsr` 后重跑主流程。
- 如果不允许，报告 DRSR sensitivity，说明排除理由。

## 24. 最终汇报模板

最终报告应包含：

```text
Selected Probe-4:
  algorithm_1
  algorithm_2
  algorithm_3
  algorithm_4

Main teacher:
  T_no_llm

Main robust_score:
  ...

Sensitivity:
  rank_all12 = ...
  rank_no_llm = ...
  rank_no_stage1 = ...
  rank_forbid_pysr = ...
  rank_forbid_ragsr = ...
  rank_include_drsr = ...

Risk:
  finite_mean = ...
  finite_min = ...
  explosion_mean = ...
  explosion_max = ...
  taxonomy = ...
  audit_flags = ...

Dataset coverage:
  top50 family JSD = ...
  top50 subgroup JSD = ...
  top100 family JSD = ...
  top100 subgroup JSD = ...
```

## 25. 最小实现闭环

建议按三步实现。

### Phase A：v0.2 NMSE-only scorer

实现：

```text
check/analyze_probe4_selection_v02.py
```

输入：

```text
e1_12_dataset_algorithm_nmse_table.csv
candidate200_unified.csv
```

输出：

```text
probe4_v02_algorithm_metrics.csv
probe4_v02_pairwise_behavior_distance.csv
probe4_v02_combo_scores.csv
probe4_v02_sensitivity_summary.csv
probe4_v02_selection_report.md
```

### Phase B：raw result 增强版

等 raw result 表补齐后，加入：

```text
runtime
status
failure_reason
timeout_type
artifact_integrity
minute_snapshot_integrity
```

恢复 practical cost：

```text
combo_score_with_cost
  = 0.10 * practical_cost
  + 其它项重新归一化
```

### Phase C：Probe-4 freeze report

生成最终 freeze 报告：

```text
probe4_freeze_report.md
```

报告中必须包含：

- v0.2 主排序。
- bootstrap 稳健性。
- PySR/no-PySR。
- RAGSR/no-RAGSR。
- with/without DRSR。
- teacher sensitivity。
- 最终 4 个 probe 的选择理由和风险。

## 26. 最终结论

完整 Probe-4 选择不应停留在“哪个组合分数最高”。正确问题是：

```text
哪 4 个算法最能低成本复现完整算法面板对数据集信息量的判断，
同时保持稳定、互补、覆盖和可解释？
```

因此，v0.2 的核心是：

- 用 `I(i,C)` 对 `I(i,T)` 的保真替代单算法平均区分度。
- 用 top-k 高信息数据集的 family/subgroup 分布替代单纯 finite coverage。
- 用 train/id/ood 梯度刻画性能退化路径。
- 删除当前无效的 constant practical cost。
- 加入 missing 和 explosion penalty。
- 用 bootstrap 和 sensitivity audit 决定是否 freeze。

当前 v0.1 top combo `imcts;pysr;ragsr;udsr` 是合理 shortlist，但最终 Probe-4 必须等 v0.2 scorer 和 sensitivity audit 跑完后再冻结。
