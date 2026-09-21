# Response to Reviewer p5tG: statistical significance

## Rebuttal-ready English response

Thank you for pointing this out. We agree that point estimates and rankings
alone do not establish whether small score differences are meaningful. We
have therefore added uncertainty estimates based on 20,000
synchronized bootstrap resamples of the 50 Core tasks. The new six-panel
figure reports task-level mean scores and marginal 95% bootstrap intervals
for all six axes. These intervals quantify task-sampling uncertainty rather
than seed-resampling uncertainty.

To directly assess algorithm-to-algorithm differences, we additionally
compute paired bootstrap intervals for

`Delta(a,b) = S(a) - S(b),`

using the same resampled task set for both methods. We complement these
intervals with two-sided task-level sign-flip tests (20,000
resamples) and apply Holm correction to all 105 method pairs within each
axis. We call a pair statistically distinguishable only when the paired 95%
interval excludes zero and the Holm-adjusted p-value is below 0.05.
Accordingly, only 48--65 of 105 pairs
are distinguishable on any axis (ID-Q: 65, OOD-G:
51, SYM-F: 57, EFF: 56, ROBU:
51, STAB: 48). We therefore avoid interpreting
the remaining small score gaps as definitive orderings.

This conclusion is even clearer for neighboring ranks: only
4 of the 84 adjacent-ranking comparisons are
distinguishable after correction (ID-Q: 1, OOD-G:
1, SYM-F: 0, EFF:
1, ROBU: 1, STAB:
0), and none of the adjacent comparisons within the
Top-5 are distinguishable. This directly confirms that small neighboring
score gaps should not be interpreted as statistically resolved rankings.

For readability, the rebuttal shows a Top-5 paired-difference matrix for each
axis and reports all adjacent-ranking comparisons in a companion table. The
complete 630-pair result is also provided. As a fairness check, we repeated
the analysis with exactly three runs per method on the five axes that admit a
matched reconstruction. Rank correlations between the primary and matched
analyses are 0.975--0.996, and the largest absolute score
change is 4.05 points. This sensitivity result supports the main
ordering while making clear where uncertainty remains. ROBU in this check
uses the two common noise levels; STAB is not included because its matched
seed-level structural proxy is not part of the sensitivity artifact.

## Figure caption

**Six-axis score uncertainty across 15 symbolic regression methods.** Points
denote mean scores over Core-50 tasks, and horizontal bars denote marginal
95% intervals obtained from 20,000 synchronized task-level
bootstrap resamples. These bars visualize task-sampling uncertainty;
statistical comparisons are determined separately using paired bootstrap
intervals of algorithm-score differences and Holm-adjusted task-level
sign-flip tests.

## 中文核对

- 主图置信区间只表示任务采样不确定性，不声称覆盖 seed 重采样方差。
- 两算法比较使用同一组重采样任务，保持严格配对。
- “可区分”同时要求差值区间排除 0 且 Holm 校正后 p < 0.05。
- 正文展示每轴 Top-5 矩阵和相邻排名比较，完整 630 组结果保留在 CSV。
- matched-three-run sensitivity 对所有方法统一使用 3 次运行；ROBU 使用两个
  共同噪声水平；STAB 不使用不可严格复算的近似值。
