# Response to Reviewer p5tG: statistical significance

## Rebuttal-ready English response

Thank you for raising this important point. We agree that small differences in
the leaderboard should not be interpreted as meaningful solely from point
estimates or marginal confidence-interval overlap. We therefore added a
task-paired uncertainty analysis over the 50 shared Core-50 tasks. For each of
the six axes, we report 95% confidence intervals from 20,000 synchronized
task-level bootstrap resamples. For every algorithm pair, we additionally
report the paired score difference and its bootstrap interval, together with a
two-sided task-level sign-flip permutation test (20,000 resamples) and Holm
correction over all 105 pairwise comparisons within that axis.

The new analysis confirms that many small leaderboard gaps are not
statistically distinguishable. Depending on the axis, only 51--65 of the 105
pairs remain significant after Holm correction (ID-Q: 65, OOD-G: 51, SYM-F:
57, EFF: 56, ROBU: 51, STAB: 56). For example, iMCTS and uDSR are not
significantly different on any axis after correction. Their paired
iMCTS-minus-uDSR differences (95% CIs) are: ID-Q -5.69
[-20.57, 9.83], OOD-G 4.13 [-7.20, 15.82], SYM-F 4.67
[-4.70, 14.33], EFF 5.71 [-8.28, 20.02], ROBU -2.65
[-9.50, 4.21], and STAB 2.40 [-4.66, 9.45]. Conversely, some differences are
clearly supported: JAXSR exceeds FePySR on STAB by 17.87 points
[11.41, 23.78], with Holm-adjusted p = 0.00525, while their other five axes
are not significantly different.

We will add the six-panel confidence-interval figure and the complete pairwise
inference table to the revision, and will explicitly avoid interpreting
non-significant small gaps as evidence of superiority. The 15-method
supplement combines the frozen NeurIPS results for the original 12 methods
(1 h, five seeds) with one-hour checkpoints from the completed AAAI
three-hour trajectories for FePySR, JAXSR, and SymbolFit (three seeds).
For ROBU, the original methods use noise levels 1%, 5%, and 10%, whereas the
three added methods use the available 1% and 5% runs; we state this scope
difference in the figure caption and do not use it to make fine-grained
cross-cohort robustness claims.

## 中文核对

- 置信区间单位是共享的 50 个任务，而不是把不同 seed 当作独立样本。
- 每次 bootstrap 对所有算法同步抽取同一批任务，因此算法间差值保持配对。
- 显著性结论以成对差值表为准，不能只根据两条边际置信区间是否重叠判断。
- 每个轴内部对 15 个算法的 105 组比较做 Holm 多重检验校正。
- 原 12 个算法沿用 NeurIPS 冻结结果；新增 3 个算法读取 AAAI 三小时轨迹的
  `minute_0060`。
- 新增算法 ROBU 只有 1%/5% 噪声，原算法为 1%/5%/10%，因此仅作补充展示。

## 提交前内部审计提醒

冻结的正式 judge 会对已经采用 `x0/x1/...` 表示的 canonical 表达式再次应用
`feature_to_x_map`。例如 `g0001`（`Keijzer-11`）中的 `x0*x1` 会被清洗成
`x0*x0`。当前补充材料为保持与 NeurIPS 原 12 个算法完全可比，复现了这一
冻结行为。提交前必须在以下两种口径中二选一：

1. 保留冻结口径，并仅称其为对已提交分数的配对不确定性分析。
2. 修复 judge，并对全部 15 个算法统一重算 SYM-F 及受其影响的 STAB。

不能把修正后的新 3 算法分数与冻结的原 12 算法分数混在同一张图中。

## Supporting artifacts

- Figure: `six_axis_uncertainty_15algs.png`
- Vector figure: `six_axis_uncertainty_15algs.pdf`
- Means and marginal intervals: `six_axis_means_ci_15algs.csv`
- Paired differences, permutation p-values, and Holm correction:
  `six_axis_pairwise_inference.csv`
- Task-level six-axis components: `six_axis_task_components_15algs.csv`
