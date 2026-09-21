# 全量重跑 13 算法 3h 实验 — 部署计划

## 目标

新建独立批次，全量重跑 13 alg × 50 datasets × 3 seeds × 3 noise = 5850 tasks，并发 70/host，跳过 smoke 直接 full。

## 步骤

### 1. 创建新批次目录并生成 manifest + params

```bash
BATCH_ID="formal3h_13alg_ssr50_seed520-522_noise0-001-005_$(date +%Y%m%d-%H%M%S)"
BATCH_DIR="benchmark-runs/formal3h/$BATCH_ID"

python benchmark-control/compliance/launchers/prepare_batch.py \
  --profile formal3h_13alg_3seed_3noise \
  --batch-dir "$BATCH_DIR"
```

产物：
- `manifest/tasks.csv` (5850 行)
- `manifest/datasets.csv` (50 行)
- `queues/ssr50_source.csv`, `queues/smoke_2datasets_source.csv`
- `params/<tool>__<noise>.json` (13×3 = 39 个)
- `params_smoke/<tool>__<noise>.json` (39 个)

### 2. 更新 latest 软链接

```bash
ln -sfn "$BATCH_ID" benchmark-runs/formal3h/latest
```

### 3. 生成部署脚本

```bash
python benchmark-control/compliance/launchers/write_full3h_queue_commands.py \
  --batch-dir "$BATCH_DIR"
```

### 4. 手动修改 full dispatch 脚本

在生成的 `04_full_dispatch_from_iaaccn22.sh` 中：
- **删除** `--task-id-allowlist-csv` 行（不走 missing-only，全量跑）
- **添加** `--max-jobs-per-host 70`
- **添加** `--max-new-jobs-per-host-per-poll 5`
- **删除** smoke gate check（跳过 smoke 直接 full）
- 修改 session-prefix 为新批次标识

### 5. 同步到 iaaccn22 并 fan-out 到 23~29

执行 `00_sync_code_and_batch_to_iaaccn22.sh`

### 6. Preflight 检查

在 iaaccn22 上执行 `01_preflight_from_iaaccn22.sh` 确认 8 台机器可达、环境可用

### 7. 启动 full dispatch

在 iaaccn22 上通过 tmux 执行修改后的 full dispatch 脚本

## 关键配置

| 参数 | 值 |
|------|-----|
| `--max-jobs-per-host` | 70 |
| `--max-new-jobs-per-host-per-poll` | 5 |
| `timeout_in_seconds` | 10800 |
| `min_runtime_seconds` | 10500 |
| `progress_snapshot_interval_seconds` | 60 |
| `--task-id-allowlist-csv` | 不传（全量） |

## 不做

- 不删除历史批次
- 不清理数据盘
- 不提交 benchmark-runs/
- 不跑 smoke（代码仅改了 1 行防护性 patch）
