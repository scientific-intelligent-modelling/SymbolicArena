# all_15alg_fullcpu_v1（已中止，非权威来源）

该批次已于 2026-09-05 中止，仅保留作调度事故与资源审计证据。禁止恢复 controller、继续派发、冻结其 partial outputs，或将其用于正式六轴聚合。`commands/run_all_15alg_preflight.sh` 与 `commands/run_all_15alg_formal.sh` 已永久熔断并直接退出。

## 历史计划范围

- 逻辑母集：`15 x 50 x 3 x 3 = 6750`。
- 已由 `all_conditions_cpu_v2` 覆盖且从新队列排除：`143`。
- 历史新队列：`6607`。
- `143 + 6607 = 6750` 仅证明原计划分区，不表示执行完成，也不构成正式来源证明。
- 算法：gplearn, llmsr, pyoperon, drsr, pysr, dso, tpsr, e2esr, fepysr, jaxsr, qlattice, imcts, udsr, ragsr, symbolfit。
- 条件：`clean`、`noise001`、`noise005`；种子：520、521、522。

## 中止状态

- 批次：`all_15alg_fullcpu_v1_formal`。
- 已完成但隔离：`2118`；已取消：`4489`。
- `iaaccn22~29` 的匹配进程和 tmux 会话均已清零，controller 已停止。
- 隔离标记：`ABORTED_DO_NOT_USE.json`。
- 紧凑审计：`runtime/abort_fullcpu_20260905-1118/`。

没有删除任何结果文件。隔离的 `2118` 条 partial outputs 只能用于事故审计，不得补数、续跑或进入正式来源清单。

## 正式数据来源

正式六轴采用原始未受影响结果作为基底，仅按冻结的 replacement scope 覆盖已完成的 `368` 个针对性重跑。任何来源路径、batch 或 manifest 只要引用 `all_15alg_fullcpu_v1_formal`，冻结与聚合都必须失败。

这 `368` 个针对性任务包括：SymbolFit clean `150`、JAXSR clean `35`、iMCTS clean `40`，以及 noisy `143`（DRSR `1`、JAXSR `63`、iMCTS `79`）。

## 历史资产

三份 source CSV、参数、composite ledger、历史启动命令和运行时代码指纹继续保留，目的仅为复盘当时如何构建与调度队列。它们不是正式结果输入或权威 provenance。

`commands/run_all_15alg_dry_run.sh` 仍可用于本地审计，不连接远端。其余部署、preflight、formal 命令不得执行。

资产哈希见 `asset_manifest.json` 与 `asset_manifest.sha256`；中止后的状态与审计证据另由 `ABORTED_DO_NOT_USE.json` 和 `runtime/abort_fullcpu_20260905-1118/` 固化。
