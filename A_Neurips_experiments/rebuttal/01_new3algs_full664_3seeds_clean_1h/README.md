# Rebuttal：新增算法全量 clean 复现实验（664 × 3 seeds × 3算法）

目录名称约定：`01_new3algs_full664_3seeds_clean_1h`

- **数据集**：full-664（`full664_unified.csv`）
- **算法**：`fepysr`、`jaxsr`、`symbolfit`
- **种子**：`520, 521, 522`
- **噪声**：`clean`（`noise_sigma=0.0`）
- **预算**：`3600s`（1h）
- **最小运行时**：`3300s`
- **任务量**：`664 × 3 × 3 = 5976`

## 数据与稳定身份

数据集清单来自：

`exp-planning/02.E1选择验证/generated/probe4_full664_v1/full664_unified.csv`

清单必须恰好有 664 行，`global_index` 唯一并连续覆盖 `1..664`，
`dataset_dir` 与 `dataset_rel` 各自唯一。稳定数据集身份使用
`g0001..g0664` 和 `dataset_rel`，不能使用会重名的 `dataset_name` 或
`basename`。

## 参数来源

权威来源是 AAAI Stage 4 实际批次：

`benchmark-runs/formal3h/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658/params/`

`provenance/aaai_params_3h/` 保存来源参数副本，`params/` 保存正式 1h 参数，
`smoke/params/` 保存 600 秒 smoke 参数。正式参数相对来源只允许修改
`timeout_in_seconds: 10800 -> 3600`；smoke 只允许修改为 `600`。

`min_runtime_seconds=3300` 是合规审计阈值，写在 manifest 中，不作为第三方
算法参数透传。算法自己的搜索超参数保持 AAAI 实际配置。

## 目录说明

- `manifest/`：5976 条正式任务及预算、数据集、算法和来源指纹。
- `params/`：正式 1h 参数。
- `provenance/`：AAAI 3h 参数来源快照。
- `queues/full664_source.csv`：正式调度数据源和正式队列状态。
- `smoke/`：18 条 smoke 任务、参数、队列和审计产物。
- `deploy/`：同步、preflight、smoke、正式调度和收集审计脚本。
- `analysis/`：新三算法 run-level 汇总与 Stage 3 合并后的 7 算法榜单。
- `BATCH_NAME.txt`：冻结的正式 batch ID。
- `SOURCE_MAP.tsv`：关键材料来源映射。

## 执行门禁

```text
prepare + tests + dry-run
  -> remote preflight: 8/8 hosts pass
  -> smoke dispatch: 18 tasks
  -> smoke collect + harvest + audit: 18/18 pass
  -> full dispatch: 5976 tasks
  -> collect + harvest + audit + rerun until complete
  -> metric aggregation and Stage 3 four-probe comparison
```

本地入口：

```bash
bash A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/deploy/00_validate_and_sync_to_iaaccn22.sh
```

后续脚本必须从 `iaaccn22` 的仓库根目录运行，顺序是
`01_preflight_from_iaaccn22.sh`、`02_smoke_dispatch_from_iaaccn22.sh`、
`03_full_dispatch_from_iaaccn22.sh`。正式任务完成后再运行
`04_collect_audit_from_iaaccn22.sh`。该脚本审计通过后会继续调用
`05_generate_symf_from_iaaccn22.sh`。

正式队列运行期间可重复执行：

```bash
bash A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/deploy/06_audit_completed_from_iaaccn22.sh
```

该脚本冻结一次 state 快照，按 assigned host 到 8 台机器逐条核对已经
标记为 `done` 的结果，要求运行时不少于 `3300s`，并检查状态、数据身份、
ID/OOD NMSE、canonical artifact 和方程。每次报告保存在
`monitoring/completed_audit/<timestamp>/`。

队列控制器按 `batch_name` 持有 `state/*.controller.lock` 单实例锁。
重复启动同一批次必须立即失败，不能同时使用 tmux 和 nohup 启动两个
控制器。`20260725-045816` 的双控制器事件、重复 session 清理清单、
加锁重启和首轮 `118/118` 结果审计证据保存在
`monitoring/controller_incident/20260725-045816/`。事件处理没有删除或
移动结果；harvest 仍按最终 state 的 `assigned_host` 选择规范结果。

完成波峰期间，控制器已将远端残留进程回收和精确 session 确认改成
无残留立即返回、每台主机一次 SSH 批量确认。`20260725-072427` 的
受控重启把新调度器加载到 tmux 单控制器中，未停止任何算法 worker。
恢复后盘点为一个控制器、零重复任务；`20260725-073556` 固定快照审计
对 `551/551` 个 done 任务全部验证通过，`issue_count=0`。恢复与时延
证据分别保存在 `monitoring/controller_recovery/20260725-072427/` 和
`monitoring/completed_audit/20260725-073556/`。

## 指标汇总

`04_collect_audit_from_iaaccn22.sh` 在 `5976/5976` 收集并通过 audit gate
后，会自动运行：

```bash
python check/analyze_neurips_rebuttal_full664.py \
  --batch-dir A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h
```

分析器保留 manifest 的完整期望网格。ID/OOD NMSE 先执行
`log10(max(NMSE, 1e-12))` 并截断到 `[-12, 12]`，各自缺失值在算法
均值中按 `+12` 计入。最终排名按 penalized mean OOD log NMSE 升序。

运行期间仅可用 `--allow-incomplete` 生成诊断预览；预览会标记为
`INCOMPLETE_PREVIEW`，不得用于论文结论。正式模式会要求结果完整、
数据身份无冲突且 audit gate 通过，否则直接拒绝生成最终榜单。

formal symbolic judge 使用 `formula.py` 作为 ground truth，按稳定 `gid`
关联 664 个数据集，并使用 Stage 3 raw digest 中的 normalized/instantiated
表达式。唯一保留的参数审计告警是 `g0217/PO27`：历史公式参数名 `F0`
与数据列名 `x` 不一致；两边各只有一个未匹配变量，因此采用
`F0 -> x` 的唯一剩余位置映射，并在参数审计中保留该记录。

`05_generate_symf_from_iaaccn22.sh` 会对 7 算法共 `13944` 条 run 计算
SYM-F、Exact、TreeSim，并严格按 `Algorithm key` 合入最终榜单。

## 最终产物

正式审计闭环后至少输出：

- 每算法 `Valid`、`Metric`、ID/OOD log NMSE。
- 运行时间和表达式复杂度。
- 可计算时输出 SYM-F、Exact 和 TreeSim。
- 与 Stage 3 的 `dso`、`imcts`、`pyoperon`、`udsr` 按相同
  full-664、seeds 520--522、clean、1h 口径合并成 7 算法比较表。
- `analysis/new3_run_level.csv` 与 `new3_algorithm_summary.csv`。
- `analysis/full664_7alg_run_level.csv` 与
  `full664_7alg_leaderboard.csv`。
- `symf/full664_7alg/symbolic_metrics_formal*.csv`。
- `analysis/full664_7alg_leaderboard_with_symf.csv`。

在 5976 条任务完成并通过审计之前，不得把该目录描述成最终结果。
