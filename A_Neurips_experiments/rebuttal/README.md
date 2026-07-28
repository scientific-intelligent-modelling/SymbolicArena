# NeurIPS Rebuttal 实验

这里保存 NeurIPS 投稿之后新增的 rebuttal 实验。新增结果作为独立证据链归档，
不回写或覆盖原 Stage 1--4 的历史结果。

## 1. 新增三算法 Full-664

目录：`01_new3algs_full664_3seeds_clean_1h/`

```text
algorithms = fepysr, jaxsr, symbolfit
datasets = Stage 3 的 full-664
seeds = 520, 521, 522
condition = clean only
budget = 1h per task
tasks = 3 × 664 × 3 = 5976
```

参数从 AAAI Stage 4 实际运行的三份 clean 3h 参数继承，仅将任务超时预算从
`10800` 秒改为 `3600` 秒。正式实验前必须通过：

1. 本地参数、清单、任务数与 dry-run 校验。
2. `iaaccn22~29` 八台机器远端 preflight。
3. `2 datasets × 3 algorithms × 3 seeds = 18` 个 600 秒 smoke。
4. smoke collect、harvest 和 audit 全部通过。

正式实验现已完成：

- `5976/5976` 个任务全部 `done`，三个算法和三个 seed 各自网格完整。
- 最终 assigned-host 严格审计验证 `5976/5976`，8 台主机全部通过，
  `issue_count=0`。
- 正式 harvest、audit 和基础指标分析均已拉回；新三算法与 Stage 3 四算法
  合并为 `13944` 条 run-level 记录。
- 7 个 formal SYM-F 分片各 `1992` 条，合并后的 `13944` 条符号指标通过
  内容哈希、运行键、算法、数据集和 seed 完整性复验。
- 权威 7 算法榜单为
  `analysis/full664_7alg_leaderboard_with_symf.csv`。

实验运行和审计细节见该实验目录的 `README.md`。

## 2. Core-50 与 Full-664 排名相关性

目录：`02_core50_full664_rank_correlation_9algs/`

使用同时具有 Core-50 和 Full-664 结果的 9 个算法，直接对照逐算法 OOD
分数与排名。主报告用 Pearson 解释连续 OOD 分数的一致性，用 Spearman
解释算法排序的一致性；紧凑表、散点图、审计统计与证据边界见该目录的
`README.md`。

## 3. SYM-F 权重敏感性

目录：`03_symf_sensitivity_12algs/`

基于原始 12 算法在 Core-50 上的 `3000` 条逐运行符号指标，复算：

1. TreeSim 与 SOF1 权重的局部扰动和宽范围压力网格。
2. 非等价分数上限扫描。
3. TreeSim、变量 F1、算子 F1 的相关性、回归与消融。

本分析只检验 SYM-F 的聚合权重和非等价上限，不检验 CAS/数值等价检测器
及其阈值。完整结果、图表和复现命令见该目录的 `README.md`。

## 4. 固定 Probe-4 的 Core-50 成员敏感性

目录：`04_core50_membership_sensitivity_fixed_probe4/`

只复用已完成的 Full-664 Probe-4 三种子结果，不启动新的算法训练。分析先
检查论文目标函数重构能否精确复现冻结 Core-50，再扫描：

1. `Coverage / MeanInfo / Balance` 三个主权重的 ±20% 全因子扰动。
2. 步长 0.1 的完整权重单纯形压力测试。
3. 结构/响应混合、family smoothing 与 subgroup cap。

由于仓库中没有论文所述历史局部搜索选择器的可执行版本，所有结论都受
基线复现闸门约束；未复现时只报告为 paper-spec counterfactual，不冒充
历史选择过程。完整证据与复现命令见该目录的 `README.md`。

## 5. Stage-4 逐分钟多种子汇总

目录：`05_stage4_anytime_12algs_5seeds_clean_1h/`

从 NeurIPS Stage 4 的远端原始 clean 快照恢复 `3000 × 60 = 180000`
条 run-minute 记录，并按 5、10、30、60 分钟整理为图片所示的
`算法 × 六轴指标 × 时间点` 表。目录同时提供按五种子均值汇总的主表和
论文协议聚合对照表。

当前边界是：

1. Clean ID-Q、OOD-G 和 EFF 从当时的 best-so-far 重算。
2. 时间点 SYM-F 与 STAB 为明确标注的轻量 proxy，不冒充正式
   CAS/NED 或完整结构一致性指标。
3. Stage 4 未归档 noisy minute snapshots，因此 5/10/30 分钟 ROBU 为
   `NA`；60 分钟引用完整 noisy track 正式值，不从 clean 结果倒灌。

紧急补充的 uDSR、iMCTS、PySR `10/20/30/40/50/60min` 五种子均值表见
`05_stage4_anytime_12algs_5seeds_clean_1h/table_seed_mean_top3_10_20_30_40_50_60.md`。
