# 固定 Probe-4 的 Core-50 成员敏感性

本目录只复用已经完成的 Full-664 Probe-4 三种子结果，不启动任何新的
符号回归训练。固定算法为 `dso / imcts / pyoperon / udsr`。

## 结论边界

基线复现闸门：**未通过**。

- 冻结 Core-50 与论文公式重构基线重合 `23/50`，
  Jaccard 为 `0.2987`。
- 冻结清单目标值为 `0.760703`，重构局部最优为
  `0.821051`。
- 名义基线使用 `6` 个互异可行起点；每个
  权重配置使用重构基线加 `2`
  个辅助起点。
- 因此不能把后续扫描写成历史 Core-50 选择器的直接复现；它只能作为论文公式的可审计反事实重构。

这个边界很重要：仓库中没有论文所述“随机重启 + 一换一局部搜索”的历史
可执行选择器；冻结清单来自更早的 Master-100 到 Dev/Core 切分链。

## 可执行重构

目标函数为：

```text
J = w_cov * Coverage + w_info * MeanInfo + w_bal * Balance
baseline = (0.45, 0.35, 0.20)
```

已落实的硬约束：

1. 恰好 50 个任务。
2. semantic duplicate 与 basename 均至多出现一次。
3. family smoothing 为 `0.6`，下/上松弛为 `-1/+2`。
4. subgroup cap 为 `5`。
5. easy / medium / hard / extreme 均至少出现一次。

Coverage 和 Balance 使用论文列出的已物化分类字段；`dummy_variable_status`
由 SRSd metadata class 确定，`valid_output_pattern` 由四探针有效 seed 数编码。
论文声明的 `ood_type` 未出现在 Full-664 后处理表中，未伪造该字段。Balance
的结构/响应混合沿用 Coverage 的 `0.6/0.4`，因为论文没有给出另一组数值。
`limited-quota` 的具体 cap 同样未公开，所以不把事后猜测写进主约束。

## 扫描结果

局部合理扰动把三个主权重分别乘以
`{0.8, 0.9, 1.0, 1.1, 1.2}` 后重新归一化：

- 配置数：`121`
- 对冻结清单 overlap：最小 `19/50`，
  中位 `23.0/50`
- 对重构基线 Jaccard 最小值：
  `0.7544`
- Probe-4 对 Full-664 的 Spearman 最小值：
  `1.0000`
- Kendall 最小值：`1.0000`
- 成对排序一致率最小值：
  `1.0000`

这组结果的正确解释是：

1. **冻结成员资格尚不能由论文目标函数复现。** 在补齐历史选择器或修正文稿
   之前，不应声称冻结 Core-50 对这些权重已经通过成员稳定性验证。
2. **条件于重构选择器，局部权重扰动较稳定。** 最差仍保留
   `43/50` 个重构基线任务。
3. **Probe-4 排序稳定只是面板内诊断。** 这里只包含参与构造的 4 个算法，
   `rho=1` 不能替代 held-out algorithm 验证，也不能单独证明 Core-50
   的外部代表性。

宽范围压力测试使用步长 `0.1` 的三权重完整单纯形：

- 配置数：`66`
- 对冻结清单 overlap 最小值：
  `12/50`
- 对重构基线 Jaccard 最小值：
  `0.1905`
- Probe-4 排名顺序种类：`1`
- Spearman 最小值：`1.0000`

约束/内部常数压力测试：

- 配置数：`27`
- 成功构造并优化：`21`
- 冻结清单在其中不可行：`15`
- 成功配置对冻结 overlap 最小值：
  `19/50`

## 复现

```bash
OPENBLAS_NUM_THREADS=1 python check/analyze_core50_membership_sensitivity.py
```

关键输入 SHA-256：

```text
dataset_level     f88a594cd678ef7497f91301675a7c7f3e8169f14e6fd7dc9dbcf640821f7a13
dataset_algorithm 7a00dc7461e3a129f2973108b5f4ed1f1e2dce282f173d7be479689ec1fb3aa5
core_manifest     5a0824c10f57009b13f2ebc16059e1472baac3743473a47f170df2cdcc5ecf65
```

## 输出

- `input_audit.json`：输入、字段和约束定义审计。
- `baseline_reproduction.json`：基线复现闸门。
- `weight_sweep_local.csv`：主权重 ±20% 全因子扫描。
- `weight_sweep_simplex.csv`：宽范围单纯形压力测试。
- `constraint_stress.csv`：结构/响应混合、family smoothing、subgroup cap。
- `selected_memberships.csv`：每个权重配置的 50 个成员。
- `membership_frequency.csv`：任务入选频率。
- `summary.json`：机器可读汇总。
- `rebuttal_text.md`：内部草稿；基线闸门失败时禁止直接提交。
