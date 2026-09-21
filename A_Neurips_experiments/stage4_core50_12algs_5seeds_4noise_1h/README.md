# Stage 4: Core-50 12algs x 5seeds x 4noise x 1h

这个目录对应 NeurIPS 最终阶段，不再是后续 AAAI 的 SSR-50 十五算法包。
这里沿用 AAAI 目录的计数方式，把 clean 也算作一个 noise condition：

```text
12 algorithms x 50 datasets x 5 seeds x 4 conditions x 1h = 12000 runs
conditions = clean, sigma 0.01, sigma 0.05, sigma 0.10
seeds = 0, 1, 2, 3, 4
```

其中：

- clean 主榜：`3000` runs
- 三档 noisy-training / clean-test 鲁棒性评测：`9000` runs

## 当前权威口径

### Clean 主榜

- `run_archive/summary.json`
  - clean 批次完成度摘要
  - `batch_name=core50_12alg_5seed_all_20260502-065700`
  - `expected_tasks=3000`，`final_result_files=3000`

- `run_archive/manifest.csv`
  - `3000` 行 clean 任务清单
  - 每行一条 `algorithm x dataset x seed` 运行，并带 `raw_result_json`

- `run_archive/results/`
  - `3000` 个 clean 归档 JSON
  - 这是本包内最完整的逐任务结果集合

- `collected_results/`
  - clean leaderboard、run-level 汇总和 Probe-4 对照表

- `formal_analysis/`
  - clean 结果对应的 formal symbolic fidelity 指标

### Noise 鲁棒性

- `noise_robustness/noise_final_runs.csv`
  - `9000` 条 noisy run-level 记录
  - 覆盖 `12 algorithms x 50 datasets x 5 seeds x 3 non-zero sigmas`

- `noise_robustness/noise_completion_summary.csv`
  - 36 个 `algorithm x sigma` 单元的完成度和 clean-test NMSE 摘要

- `noise_robustness/robustness_components.csv`
  - clean 与 noisy 表现的 retention/robustness 组成项

- `paper_tables/table13_noise_robustness_summary.csv`
  - 算法级 noise robustness 论文表

- `paper_tables/table14_noise_by_sigma.csv`
  - 按 `sigma=0.01/0.05/0.10` 展开的论文表

## 目录说明

- `core50_manifest/`
  - Core-50 冻结数据集清单和路径表

- `run_archive/`
  - 来自 `exp-planning/04.Core50正式全量评测/generated/core50_12alg_5seed_final_results`
  - 包含 clean `analysis/`、`manifest.csv`、`summary.json`、`state/`、`results/`

- `collected_results/`
  - 来自 NeurIPS 冻结包的 clean 汇总表

- `formal_analysis/`、`hexagon/`、`paper_tables/`
  - NeurIPS 下游分析和论文产物

### STAB 口径修正（2026-07-31）

`hexagon/` 三个 CSV 与 `paper_tables/table11_hexagon_scores_formal.csv` 的 `STAB`
已按论文附录 F.5 重算：性能修正项代入 formal SYM-F，而不是早先的 proxy 符号分。

```text
q_perf   = 0.4*q_id + 0.3*q_oodG + 0.3*q_sym_formal
PureStab = 0.4*N + 0.3*V + 0.3*C
STAB     = PureStab * sqrt(q_perf)
```

修正原因：`check/plot_core50_formal_figures.py` 早先只把 SYM-F 一列换成 formal，
STAB 整列从 v1 继承，而 v1 的 STAB 用的是 `q_sym_proxy`（旧的数值代入式等价判定
＋算子/变量 Jaccard 树相似度）。这与附录 F.5 引用 F.2 formal 定义的表述不符。

影响：STAB 轴首位由 iMCTS 变为 PySR（66 对比较中 9 对翻转，Spearman 0.874）；
iMCTS 48.26→42.79、QLattice 39.24→33.81，其余 10 个算法上升 0.9–5.3。
`HexaScore_formal_*` 前四名不变，QLattice 在含 ROB 口径下由第 5 降至第 7。
ID-Q / OOD-G / SYM-F / EFF / ROBU 逐位未变。

旧口径数值保留在 `STAB_proxy`、`STAB_proxy_component` 与
`HexaScore_formal_*_stabproxy` 列中，可直接对照。

仍未 formal 化的两项（论文附录需相应说明）：

- `EFF` 仍是 proxy（`final_quality × 时间折扣`）。附录 F.3 描述的分钟级
  best-so-far 质量-时间 AUC 需要逐分钟快照，该数据未落盘，无法从归档产物重算。
- `STAB` 的结构一致性 `C` 仍是 proxy（只比较 `pred_skeleton` 字符串相等），
  未实现 F.5 的四条判定规则。

- `noise_robustness/`
  - NeurIPS 冻结包中的 noisy-training / clean-test run 表和汇总

- `raw_results/`
  - 只保留 `README.md`，说明原始结果树未随包归档
  - 原冻结包这里是一个指向 clean 批次
    `experiments/core50_12alg_5seed_all_20260502-065700` 的软链
  - 该软链在整理包内无效，所以没有复制

## 当前完成度

Clean `run_archive/summary.json`：

- 总任务数：`3000`
- 最终归档结果文件：`3000`
- `ok=2982`
- `timed_out=9`
- `no_valid_output=9`

Noise `noise_robustness/noise_final_runs.csv`：

- 总记录数：`9000`
- 每个 sigma：`3000`
- 每个算法：`750`
- 每个 seed：`1800`
- `valid_output=True`：`8636`
- status：`ok=8918`、`timed_out=31`、`no_valid_output=36`、`error=15`

## 重要边界

1. Clean 部分保留了 `3000` 个逐任务归档 JSON 和带原始路径的 manifest。
2. Noise 部分保留了 `9000` 条 run-level 记录、完成度摘要和论文表，但当前
   整理包没有 noisy runs 的逐任务原始 JSON、日志、checkpoint 或远端路径清单。
3. `formal_analysis/`、`noise_robustness/`、`collected_results/` 和
   `run_archive/analysis/` 来自不同上游整理步骤。跨表拼接时必须参考
   `SOURCE_MAP.tsv`，不要把 mixed provenance 描述成单次 postprocess 的直接产物。

## 完整性校验

本阶段已生成：

- `FILE_INVENTORY.tsv`：记录阶段内文件的相对路径和字节数
- `CHECKSUMS.sha256`：记录阶段内文件的 SHA-256

从本目录执行：

```bash
sha256sum -c CHECKSUMS.sha256
```
