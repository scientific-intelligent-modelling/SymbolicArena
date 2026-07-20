# Stage 4: SSR-50 15algs × 3seeds × 3noise × 3h

这个目录整理的是最终打榜实验：

```text
15 algorithms × 50 datasets × 3 seeds × 3 noise levels × 3h = 6750 runs
noise = clean, noise001, noise005
seeds = 520, 521, 522
```

## 当前权威口径

优先看这几个文件：

- `algorithm_summary_with_fepysr_rerun.csv`
  - `15` 行，每行一个算法
  - 当前最终排行榜/效果摘要
  - `score = (mean_id_log + mean_ood_log) / 2`，越小越好

- `selected_runs_with_fepysr_rerun.csv`
  - `6750` 行，每行一个 `algorithm × dataset × seed × noise` 运行
  - 当前最终 run 级明细

- `selected_runs_by_task_wide_15alg.csv`
  - `450` 行，每行一个 `dataset × seed × noise`
  - 15 个算法横向展开，列名形如 `<algorithm>__ood_nmse`

- `remote_scan_with_fepysr_rerun.json`
  - 远端 8 台机器只读扫描快照
  - 已包含 LLMSR 恢复和 fepysr rerun overlay

- `EXPERIMENT_PATHS.md`
  - 最重要的路径说明
  - 记录真实远端实验来源、overlay 规则、LLMSR 回写路径、DRSR/LLMSR 独立批次

- `experiment_paths_manifest.json`
  - `EXPERIMENT_PATHS.md` 的机器可读版本

## 历史/对照文件

- `algorithm_summary.csv`
- `selected_runs.csv`

这两个是旧口径，没有 overlay `fepysr` rerun，不作为当前结论。

- `algorithm_summary_all_recovered.csv`
- `selected_runs_all_recovered.csv`

这两个和 `*_with_fepysr_rerun.csv` 当前 sha256 一致，保留用于追踪命名演进。

## 当前完成度

`selected_runs_with_fepysr_rerun.csv` 里：

- 总行数：`6750`
- 算法数：`15`
- 每个算法：`450/450`
- seeds：`520, 521, 522`
- noise：`clean, noise001, noise005`
- status：全部 `ok`

算法包括：

```text
QLattice, drsr, dso, e2esr, fepysr, gplearn, iMCTS, jaxsr,
llmsr, pyoperon, pysr, ragsr, symbolfit, tpsr, udsr
```

## 来源批次

核心来源记录在 `EXPERIMENT_PATHS.md` 和 `experiment_paths_manifest.json`。

简要口径：

- 13 个非 LLM 主批：`formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658`
- `fepysr` 补跑 overlay：`formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658_fepysr_rerun`
- `tpsr/ragsr` 补跑：`formal3h_tpsr_ragsr_rerun_20260624-115748`
- `llmsr` 独立 3h 批次：`llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442`
- `drsr` 独立 3h 批次：`drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355`

## 搬迁后校验

```bash
cd /path/to/stage4_ssr50_15algs_3seeds_3noise_3h
sha256sum -c CHECKSUMS.sha256
```

全部显示 `OK` 即表示包内文件搬迁完整。
