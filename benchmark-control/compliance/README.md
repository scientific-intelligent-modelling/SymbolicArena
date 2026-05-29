# Benchmark Compliance Control

本目录保存 benchmark 合规闭环的控制面文件。这里的内容进入 Git，用于记录实验如何生成、如何分发、如何审计、如何修复和如何通过 `/goal` 被 Codex 接管。

真实实验产物不放在本目录，而是放在被 Git 忽略的：

```text
benchmark-runs/
```

## 子目录

- `goals/`：Codex `/goal` 接管说明。
- `manifests/`：算法、数据集、预算和任务清单模板或轻量快照。
- `templates/`：批次、队列、部署和重跑模板。
- `audit-schemas/`：审计输出字段和失败分类定义。
- `repair-notes/`：修复摘要和可提交记录。
- `launchers/`：远端分发和控制器脚本。
- `run-summaries/`：从 `benchmark-runs/` 抽取出的轻量批次摘要。

## 阶段一范围

```text
15 algorithms × SSR50 × seed520 × 1h
```

阶段一只验证预算合规和落盘合规。通过后再扩展到 `3 seeds × 24h × noise`。

## 本地准备命令

以下命令默认在仓库根目录执行。`prepare_batch.py` 和 `audit_batch.py` 都支持传入 repo-relative 路径；真实批次产物写入 `benchmark-runs/`，该目录已被 Git 忽略，不会纳入提交。

准备一个新的阶段一批次：

```bash
BATCH_ID="compliance_15alg_ssr50_seed520_1h_$(date +%Y%m%d-%H%M%S)"
mkdir -p "benchmark-runs/compliance/${BATCH_ID}"
python benchmark-control/compliance/launchers/prepare_batch.py \
  --toolbox-config scientific_intelligent_modelling/config/toolbox_config.json \
  --ssr50-root sim-datasets-data/ssr50 \
  --batch-dir "benchmark-runs/compliance/${BATCH_ID}"
ln -sfn "${BATCH_ID}" benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/write_remote_sync_commands.py \
  --batch-dir benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/write_stage1_queue_commands.py \
  --batch-dir benchmark-runs/compliance/latest

python benchmark-control/compliance/launchers/check_stage1_readiness.py \
  --batch-dir benchmark-runs/compliance/latest
```

这些命令只生成 `deploy/00_sync_code_and_batch_to_iaaccn22.sh`、`deploy/01_preflight_from_iaaccn22.sh`、`deploy/02_smoke_dispatch_from_iaaccn22.sh` 和 `deploy/03_full_dispatch_from_iaaccn22.sh`，不会连接远端，也不会启动实验。
`deploy/01_preflight_from_iaaccn22.sh` 会在预检完成后调用 `check_preflight_report.py`；只有全部 host 的代码、参数、数据和环境检查通过，才允许进入 smoke。
`deploy/02_smoke_dispatch_from_iaaccn22.sh` 会在 30 个 smoke 任务结束后收集 `${BATCH_ID}_smoke` 产物，harvest 到 `benchmark-runs/compliance/latest/smoke/`，再用 `check_audit_success.py --expected-total-tasks 30` 阻断不合格 smoke。
`deploy/03_full_dispatch_from_iaaccn22.sh` 会在 750 个 full 任务结束后收集 `${BATCH_ID}` 产物，harvest 到 `benchmark-runs/compliance/latest/`，再用 `check_audit_success.py --expected-total-tasks 750` 验证最终 gate。

审计当前批次并写入心跳与重跑队列：

```bash
BATCH_ID="$(basename "$(readlink -f benchmark-runs/compliance/latest)")"

python benchmark-control/compliance/launchers/collect_remote_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --batch-id "${BATCH_ID}" \
  --hosts iaaccn22 iaaccn23 iaaccn24 iaaccn25 iaaccn26 iaaccn27 iaaccn28 iaaccn29 \
  --controller-host iaaccn22 \
  --use-internal-ips

python benchmark-control/compliance/launchers/harvest_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn22 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn23 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn24 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn25 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn26 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn27 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn28 \
  --experiment-root benchmark-runs/compliance/latest/remote-experiments/iaaccn29

python benchmark-control/compliance/launchers/audit_batch.py \
  --batch-dir benchmark-runs/compliance/latest \
  --write-heartbeat \
  --write-rerun \
  --round-id 1
```

`collect_remote_batch.py` 只用过滤 `rsync` 拉取 `result.json`、`progress/`、launcher 状态等审计所需轻量产物；`harvest_batch.py` 只把收集到的调度结果映射到当前批次的 `runs/` 审计布局。上面的 prepare、collect、harvest 和 audit 命令不会启动远端实验，也不会触发远端 dispatch。
