# SymbolicArena Figure Explanations

本文档是 `imgs/` 下论文配图的说明清单。每张图包含四类信息：

- `Files`：对应的图片文件。
- `Question`：这张图回答的审稿问题。
- `How to read`：读图方式和核心观察点。
- `Caption draft`：可直接改写进论文的英文 caption 草稿。

## Figure 1: Pipeline

- Files: `imgs/figure01_pipeline.png`, `imgs/figure01_pipeline.pdf`
- Question: SymbolicArena 如何从原始数据池逐步得到 Core-50，并形成最终 leaderboard?
- How to read: 这张图应放在实验或方法总览最前面。它强调流程不是一次性抽样，而是从 ground-truth reservoir、dual-probe scan、Candidate-200、Probe-4 全量验证、Core-50 到最终 12 算法 leaderboard 的逐级蒸馏。
- Caption draft: Overview of the SymbolicArena evaluation pipeline. A ground-truth reservoir is filtered and distilled through dual-probe screening, 12-method calibration, Probe-4 full-reservoir validation, and Core-50 selection before running the final multi-axis leaderboard.

## Figure 2: GT-Reservoir composition

- Files: `imgs/figure02_reservoir_composition.png`, `imgs/figure02_reservoir_composition.pdf`
- Question: GT-Reservoir-664 是否足够多样，而不是单一来源或单一公式族?
- How to read: 这张图现在采用一行四列的层次化布局，每个子图对应 GT-Reservoir-664 的一个特征维度：benchmark family、operator group、formula complexity、Probe4-derived difficulty。每个分类项都在 x 轴显式展示，并在柱顶标注数据集数量，用于支撑 “reservoir 是一个结构丰富的数据池” 的 claim。
- Caption draft: Composition of the GT-Reservoir-664 across four dataset characteristics: benchmark family, operator group, formula-complexity bin, and Probe4-derived difficulty. Each category is explicitly labeled on the x-axis with dataset counts annotated above the bars.

## Figure 3: PySR vs LLM-SR gap scatter

- Files: `imgs/figure03_dual_probe_scatter.png`, `imgs/figure03_dual_probe_scatter.pdf`
- Question: Candidate-200 是否来自有行为差异的数据集，而不是随机抽样?
- How to read: 每个点是一个数据集，横纵轴比较 PySR 和 LLM-SR 的 clipped log NMSE。偏离对角线越明显，说明两个 probe 的行为差异越大。颜色或标记用于区分是否进入 Candidate-200。
- Caption draft: Dual-probe behavior on the 664-task reservoir. Candidate-200 is enriched for datasets where PySR and LLM-SR exhibit informative disagreement under the same evaluation protocol.

## Figure 4: overall_gap_score distribution

- Files: `imgs/figure04_gap_score_distribution.png`, `imgs/figure04_gap_score_distribution.pdf`
- Question: Candidate-200 是否确实比未选数据集更有区分度?
- How to read: 比较 selected 与 non-selected 的 gap score 分布。Candidate-200 应整体右移，但不应只包含最极端样本。
- Caption draft: Distribution of dual-probe gap scores for selected and non-selected datasets. Candidate-200 increases informativeness while retaining non-extreme calibration cases.

## Figure 5: Candidate-200 composition

- Files: `imgs/figure05_candidate200_composition_modes.png`, `imgs/figure05_candidate200_composition_modes.pdf`, `imgs/figure05_candidate200_family_distribution.png`, `imgs/figure05_candidate200_family_distribution.pdf`
- Question: Candidate-200 的构成是否受控，是否被某个 family 或某种 selection mode 主导?
- How to read: 第一张看 high-discrimination、mid-gap、one-sided 等 selection mode；第二张看 family 配额。它证明 Candidate-200 同时考虑信息量和结构覆盖。
- Caption draft: Candidate-200 composition by selection mode and benchmark family. The candidate pool combines high-discrimination, mid-gap, and one-sided cases under family-level coverage constraints.

## Figure 6: 12 algorithm health

- Files: `imgs/figure06_algorithm_health.png`, `imgs/figure06_algorithm_health.pdf`
- Question: 12 个算法集成是否健康，哪些算法适合作为 Probe-4 候选?
- How to read: 看每个算法在 Candidate-200 上的 valid rate、finite metric rate 和 failure rate。低健康度算法不一定不能做 leaderboard，但不适合作为筛选 probe。
- Caption draft: Health statistics of the 12 integrated symbolic-regression algorithms on Candidate-200. These diagnostics separate leaderboard coverage from probe suitability.

## Figure 7: Algorithm behavior correlation

- Files: `imgs/figure07_algorithm_behavior_correlation.png`, `imgs/figure07_algorithm_behavior_correlation.pdf`
- Question: Probe-4 是否覆盖不同算法行为，而不是选了四个重复方法?
- How to read: 相关性越高代表两个算法在数据集上的响应模式越接近。Probe-4 应尽量覆盖低相关或互补区域。
- Caption draft: Behavioral correlation among the 12 algorithms based on dataset-level performance profiles. The selected Probe-4 covers complementary algorithmic responses rather than redundant solvers.

## Figure 8: Probe-4 selection score decomposition

- Files: `imgs/figure08_probe4_score_decomposition.png`, `imgs/figure08_probe4_score_decomposition.pdf`
- Question: 为什么选 DSO、PyOperon、iMCTS、uDSR 作为 Probe-4?
- How to read: 看 top probe combinations 的 health、panel fidelity、complementarity、coverage 分项。最终组合不是单一指标最高，而是整体 trade-off 最好并覆盖不同范式。
- Caption draft: Decomposition of Probe-4 selection scores across top candidate combinations. The selected panel balances health, full-panel fidelity, behavioral complementarity, and dataset coverage.

## Figure 9: Panel Fidelity scatter

- Files: `imgs/figure09_panel_fidelity_scatter.png`, `imgs/figure09_panel_fidelity_scatter.pdf`
- Question: Probe-4 对数据集信息量的判断是否接近完整 12 算法 panel?
- How to read: 每个点是 Candidate-200 中一个数据集。横轴是 12 算法 panel 的 informativeness，纵轴是 Probe-4 的 informativeness。点越贴近单调趋势，Probe-4 越可靠。
- Caption draft: Dataset informativeness estimated by the selected Probe-4 versus the full 12-method panel. High rank agreement supports using Probe-4 for full-reservoir distillation.

## Figure 10: Core-50 coverage map

- Files: `imgs/figure10_core50_coverage_map.png`, `imgs/figure10_core50_coverage_map.pdf`
- Question: Core-50 是否覆盖 GT-Reservoir-664 的主要结构和响应区域?
- How to read: 灰色点是 664 个数据集，突出点是 Core-50。若 Core-50 分布在多个区域，而不是聚成一团，说明它有较好的覆盖性。
- Caption draft: Coverage map of Core-50 within the 664-task reservoir using PCA over structural and Probe-4 response features. Core-50 covers multiple regions of the reservoir rather than a narrow cluster.

## Figure 11: Core-50 vs baselines quality radar

- Files: `imgs/figure11_core50_vs_baselines_radar.png`, `imgs/figure11_core50_vs_baselines_radar.pdf`
- Question: Core-50 相比 random、metadata-only、top-info 等 baseline 是否更均衡?
- How to read: 雷达图不是算法能力图，而是 benchmark 子集质量图。Core-50 不必每一轴最高，但应在 coverage、fidelity、stability、non-redundancy 等维度更均衡。
- Caption draft: Benchmark-quality comparison between Core-50 and subset-selection baselines. Core-50 provides a balanced trade-off among coverage, fidelity, informativeness, stability, and non-redundancy.

## Figure 12: Rank fidelity comparison

- Files: `imgs/figure12_rank_fidelity_comparison.png`, `imgs/figure12_rank_fidelity_comparison.pdf`
- Question: Core-50 是否能保留 full-reservoir 上 Probe-4 的算法排序?
- How to read: 比较 Spearman、Kendall、pairwise win agreement 和 aggregate score error。Core-50 应比 random 和 metadata-only baseline 更接近 Full-664。
- Caption draft: Rank-fidelity comparison of Core-50 against alternative 50-task subsets. Core-50 better preserves full-reservoir method ordering and aggregate scores.

## Figure 13: Distribution matching

- Files: `imgs/figure13_distribution_matching.png`, `imgs/figure13_distribution_matching.pdf`
- Question: Core-50 是否在 family、难度和 failure mode 上严重偏分布?
- How to read: 将 Full-664、Core-50 和 baseline 的分布并排比较。重点看 Core-50 是否既保持大类分布，又有意覆盖有信息的失败模式。
- Caption draft: Distribution matching between Full-664, Core-50, and subset baselines across family, difficulty, and failure-mode labels.

## Figure 14: K-scaling curve

- Files: `imgs/figure14_k_scaling_curve.png`, `imgs/figure14_k_scaling_curve.pdf`
- Question: 为什么 Core 的规模选 50，而不是 20、30、100?
- How to read: 横轴是 subset size K，纵轴是 coverage、rank fidelity、pairwise agreement 或综合质量。K=50 应位于收益拐点附近。
- Caption draft: Subset-size scaling analysis. Core-50 is selected as a practical elbow point where fidelity and coverage gains begin to saturate relative to evaluation cost.

## Figure 15: 12 algorithms x 6 axes heatmap

- Files: `imgs/figure15_hexagon_heatmap_formal.png`, `imgs/figure15_hexagon_heatmap_formal.pdf`
- Question: 12 个算法在六轴协议下整体表现如何?
- How to read: 行是算法，列是 ID-Q、OOD-G、SYM-F、EFF、ROBU、STAB。颜色越深代表分数越高。它是正式 leaderboard 主图之一。
- Caption draft: Six-axis Core-50 leaderboard over 12 symbolic-regression algorithms. Scores are fixed absolute 0--100 metrics and do not depend on the current set of compared algorithms.

## Figure 16: Top-4 radar

- Files: `imgs/figure16_radar_top4_formal.png`, `imgs/figure16_radar_top4_formal.pdf`
- Question: 代表性算法的能力形状有什么不同?
- How to read: 雷达图只适合少量算法。看每个算法在哪些轴突出、哪些轴短板明显。它用于说明 “没有单一方法统治所有维度”。
- Caption draft: Radar visualization of representative top algorithms under the six-axis Core-50 protocol, showing distinct capability profiles across numerical quality, generalization, fidelity, efficiency, robustness, and stability.

## Figure 17: ID-Q vs SYM-F

- Files: `imgs/figure17_idq_vs_symf.png`, `imgs/figure17_idq_vs_symf.pdf`
- Question: ID 数值拟合好是否等价于符号恢复好?
- How to read: 横轴 ID-Q，纵轴 SYM-F。右下区域代表 ID 拟合强但符号不忠实，是 SR benchmark 中最需要揭示的问题之一。
- Caption draft: Trade-off between in-distribution numerical quality and symbolic fidelity. Strong ID performance does not necessarily imply recovery of the ground-truth expression.

## Figure 18: OOD-G vs SYM-F scatter

- Files: `imgs/figure18_symf_vs_oodg.png`, `imgs/figure18_symf_vs_oodg.pdf`
- Question: 符号保真性是否和 OOD 泛化相关?
- How to read: 横轴 SYM-F，纵轴 OOD-G。若存在高 OOD 但低 SYM 或反之，说明数值泛化与符号等价仍需分开评估。
- Caption draft: Relationship between symbolic fidelity and out-of-distribution generalization. The two axes capture related but non-identical aspects of symbolic-regression quality.

## Figure 19: Low-NMSE non-equivalent cases

- Files: `imgs/figure19_low_nmse_non_equivalent_cases.png`, `imgs/figure19_low_nmse_non_equivalent_cases.pdf`
- Question: 是否存在低 NMSE 但公式不等价的典型反例?
- How to read: 表格列出低误差但未通过符号等价判断的 run。它用于支撑 “只看 NMSE 会误判 SR 算法”。
- Caption draft: Examples of low-NMSE but non-equivalent formulas. These cases demonstrate why numerical accuracy alone is insufficient for symbolic-regression evaluation.

## Figure 20: Equivalence rate by NMSE threshold

- Files: `imgs/figure20_equivalence_rate_by_nmse_threshold.png`, `imgs/figure20_equivalence_rate_by_nmse_threshold.pdf`
- Question: 随着 NMSE 阈值变严格，符号等价率如何变化?
- How to read: 横轴是 NMSE threshold，纵轴是等价率。不同算法曲线的差异表示 “数值好” 到 “公式真对” 的转化能力不同。
- Caption draft: Symbolic equivalence rate as a function of NMSE threshold. Different algorithms exhibit different relationships between numerical accuracy and exact symbolic recovery.

## Figure 21: Anytime performance curve

- Files: `imgs/figure21_anytime_performance_curve.png`, `imgs/figure21_anytime_performance_curve.pdf`
- Question: 各算法在 1 小时预算内的搜索过程是否不同?
- How to read: 横轴是分钟，纵轴是 best-so-far quality。早期高说明算法启动快，后期继续上升说明算法仍在有效搜索。
- Caption draft: Anytime best-so-far performance curves derived from minute-level snapshots. The curves reveal fast starters, slow improvers, and stagnating methods.
- Note: 当前版本是 `generated_proxy`，使用本地可用的 clean LLM 与低噪声 snapshot，正式论文中应标明数据范围或等完整 clean trace 汇总后重画。

## Figure 22: Time-to-threshold survival curve

- Files: `imgs/figure22_time_to_threshold_survival.png`, `imgs/figure22_time_to_threshold_survival.pdf`
- Question: 算法达到给定质量阈值的速度如何?
- How to read: 曲线下降越快，说明越多 run 更早达到阈值；尾部高说明大量任务一直未解。
- Caption draft: Time-to-threshold survival analysis from minute-level best-so-far traces, quantifying how quickly each method reaches a target quality level.
- Note: 当前版本是 `generated_proxy`，正式使用前建议用完整 clean Core-50 trace 重新生成。

## Figure 23: Complexity over time

- Files: `imgs/figure23_complexity_over_time.png`, `imgs/figure23_complexity_over_time.pdf`
- Question: 算法是否通过表达式复杂度膨胀换取数值提升?
- How to read: 横轴分钟，纵轴复杂度。若复杂度持续上升但质量停滞，可能说明搜索在堆复杂表达式。
- Caption draft: Evolution of best-so-far expression complexity over time. This trace helps identify methods that trade expression complexity for numerical gains.
- Note: 当前版本是 `generated_proxy`，需要在完整 minute snapshots 上最终确认。

## Figure 24: Early vs final performance

- Files: `imgs/figure24_early_vs_final_performance.png`, `imgs/figure24_early_vs_final_performance.pdf`
- Question: 早期表现能否预测最终表现，哪些算法慢热?
- How to read: 横轴早期质量，纵轴最终质量。左上是慢热型，右上是快速且最终强，右下是早期好但后续无改进。
- Caption draft: Early-versus-final performance comparison from best-so-far traces. The plot separates fast-and-strong methods from slow improvers and early stagnation.
- Note: 当前版本是 `generated_proxy`，正式论文中需说明当前 trace 覆盖范围。

## Figure 25: Noise robustness curve

- Files: `imgs/figure25_noise_robustness_curve.png`, `imgs/figure25_noise_robustness_curve.pdf`
- Question: 训练标签加噪后，各算法 clean-test 表现如何退化?
- How to read: 横轴噪声等级，纵轴 quality。曲线下降越慢，说明噪声鲁棒性越好。
- Caption draft: Noise-robustness curves under noisy-training and clean-test evaluation. Robust methods retain higher quality as label noise increases.

## Figure 26: ROBU heatmap

- Files: `imgs/figure26_noise_quality_heatmap.png`, `imgs/figure26_noise_quality_heatmap.pdf`
- Question: 各算法在 1%、5%、10% 噪声下的鲁棒性分布如何?
- How to read: 行是算法，列是噪声等级或 ROBU 汇总项。它比曲线图更适合对所有 12 个算法做横向比较。
- Caption draft: Heatmap of noisy-training performance across algorithms and noise levels, summarizing the ROBU extension track.

## Figure 27: STAB component chart

- Files: `imgs/figure27_stab_component_chart.png`, `imgs/figure27_stab_component_chart.pdf`
- Question: STAB 为什么不是简单的 seed 方差?
- How to read: 看 numerical stability、valid rate、structural consistency、performance correction 等分项。稳定失败的算法不应拿高分。
- Caption draft: Decomposition of effective stability into numerical consistency, valid-output rate, structural consistency, and performance correction.

## Figure 28: Score drift comparison

- Files: `imgs/figure28_score_drift_comparison.png`, `imgs/figure28_score_drift_comparison.pdf`
- Question: 固定绝对分数相比 rank-normalized 分数有什么优势?
- How to read: 绝对六轴分数在加入或移除算法时不漂移；相对排名或相对归一化分数会随 leaderboard 组成变化。
- Caption draft: Score-drift comparison under simulated leaderboard composition changes. Fixed absolute metrics remain stable, whereas rank-normalized scores drift when the algorithm set changes.

## Figure 29: Axis correlation matrix

- Files: `imgs/figure29_axis_correlation_matrix.png`, `imgs/figure29_axis_correlation_matrix.pdf`
- Question: 六个指标轴是否重复，还是刻画了不同能力?
- How to read: 高相关说明两个轴有共同信号，低相关说明它们提供互补信息。若所有轴高度相关，六轴设计就没有必要。
- Caption draft: Correlation matrix of the six Core-50 evaluation axes. The axes capture related but distinct aspects of symbolic-regression performance.

## Figure 30: Bootstrap confidence interval

- Files: `imgs/figure30_bootstrap_ci.png`, `imgs/figure30_bootstrap_ci.pdf`
- Question: 六轴分数在 Core-50 任务采样下是否稳定?
- How to read: 每个点是算法分数，误差条是 dataset bootstrap 置信区间。区间重叠提示排序差异需要谨慎解释。
- Caption draft: Bootstrap confidence intervals for six-axis leaderboard scores obtained by resampling Core-50 datasets.

## Figure 31: Ablation summary

- Files: `imgs/figure31_ablation_summary.png`, `imgs/figure31_ablation_summary.pdf`
- Question: pipeline 中的关键设计是否必要?
- How to read: 比较完整方案与 baseline / ablation 版本在 fidelity、coverage、stability 等指标上的差异。它用于回应 “是不是拍脑袋选出来的”。
- Caption draft: Summary of selection and metric ablations. The full SymbolicArena design improves the balance between fidelity, coverage, stability, and non-redundancy over simpler alternatives.
- Note: 当前版本使用已有 subset baselines；如果后续补齐 full objective ablation，可以直接替换本图。

## Usage notes

- 主文建议优先使用 Figure 1、3、8、10、12、14、15、17、21、25、30。
- Figure 21--24 当前标记为 `generated_proxy`，除非重新用完整 clean 12-algorithm minute snapshots 生成，否则论文中必须显式说明数据范围。
- Figure 5 对应两个文件，写论文时可以作为一个 multi-panel figure。
- Figure 15 和 Figure 16 都来自正式六轴结果，主文通常保留 heatmap，radar 放少数代表算法或 appendix。
- Figure 19 是 case-study 图，建议主文只放少量典型案例，完整表可放 appendix。

## Additional Figure: Hexagon small multiples

- Files: `imgs/fig_hexagon_small_multiples_neurips.png`, `imgs/fig_hexagon_small_multiples_neurips.pdf`
- Question: 12 个算法的六轴 profile 在同一坐标系下如何比较，且如何避免单张雷达图过度拥挤?
- How to read: 图为 3 行 4 列，每个 panel 对应一个算法，并按 HexaScore 从高到低排序。每个 panel 内只保留同一坐标系下的六边形网格与轮廓，不再额外放置顶部标题或右上角 axis-key 图。背景线透明度约为 9.5%，表示全部 12 个算法；当前 panel 对应算法用同色粗边和约 20% 不透明度的同色内部填充高亮。半径保持真实线性尺度 `r=score/100`。六个维度按固定顺序读取：从顶部开始顺时针依次为 ID-Q、OOD-G、SYM-F、EFF、ROBU、STAB。
- Caption draft: Small-multiple visualization of Core-50 six-axis profiles. The 12 panels are ordered by HexaScore and overlay all algorithms at low opacity, while the highlighted outline marks the current algorithm. Axis order is fixed clockwise from the top: ID-Q, OOD-G, SYM-F, EFF, ROBU, and STAB.
- Note: 这张图适合替代多算法重叠雷达图；如果主文空间紧张，可以放在 appendix 或作为 Figure 16 的扩展版本。
