# all_conditions_cpu_v2

这是 Stage5 `native_fidelity_v1` 噪声精确重跑的单一 CPU 感知队列资产，不启动远端任务。

## 任务范围

- 合并去重后的 allowlist：**143** 条，所有 `task_id` 唯一。
- 算法计数：JAXSR **63**、iMCTS **79**、DRSR **1**。
- 条件计数：`noise001` **79**、`noise005` **64**。
- 一个 controller、一个 allowlist、一个 queue；两个噪声条件通过同一条 `--noise-sigmas 0.01 0.05` 入队。
- 主机固定为 `iaaccn22` 到 `iaaccn29` 共 8 台，controller 为 `iaaccn22`。
- 每台最多 18 个任务，8 台首轮容量为 `8 x 18 = 144`，因此 143 条任务可以一次性铺开；这里的“打满”指把剩余任务并发提交，不代表物理占满 2048 核。每个任务仍是单 worker，并受 CPU 权重、load 和内存熔断约束。

五个输入 allowlist 原样冻结在 `frozen_inputs/native_fidelity_v1_allowlists/`，合并结果为 `manifests/all_conditions_noise_task_allowlist.csv`。formal 参数复制到 `params/`，每个参数均检查 `timeout_in_seconds=10800` 与 `progress_snapshot_interval_seconds=60`。`drsr.json` 是 DRSR 从参数推断模型桶所需的 base formal 参数；实际噪声任务使用 `drsr__noise005.json`。

## 调度约束

生成的三个入口都使用即将加入 scheduler 的 `--max-cpu-used-ratio 0.95`。`--max-jobs-per-host 18` 和每轮最多新增 `18` 个任务是高位保险与首轮铺开配置；load 和内存阈值均为 `0.95`，仍保留熔断。controller 日志持久化到 `logs/controller_*.log`，队列状态与事件日志在各自 `queues/*/state/` 下；本目录新增入口不使用 `/tmp`。

生成器会根据当前 checkout 的 scheduler CLI 自动选择 dry-run 验收方式：若已声明 `--max-cpu-used-ratio`，本地验收会带该参数；若尚未落地，则只在本地兼容验收时暂时省略它。正式生成的 `run_all_conditions_dry_run.sh`、`run_all_conditions_preflight.sh` 和 `run_all_conditions_formal.sh` 始终保留该参数。当前 scheduler 的旧 preflight/SSH 提交实现若仍创建瞬时 `/tmp` 文件，需要由 scheduler 本身另行收口；本资产不把 controller 日志放在 `/tmp`。

## 旧轨迹与重跑边界

既有 **4500 条旧 noise trajectory** 在回收后走后处理路径；它们不需要因为本次内部目标证据而全部重跑。本目录只生成冻结审计明确要求的 143 条补充 noise rerun，**无需全重跑** 4500 条旧 noise 轨迹，也无需重跑完整 6750 网格。

## 执行入口

依次使用：

1. `commands/run_all_conditions_preflight.sh`
2. `commands/run_all_conditions_dry_run.sh`
3. `commands/run_all_conditions_formal.sh`

dry-run 只构建本地 state，不连接远端；本次生成过程没有启动 preflight 或 formal。输入和输出文件的 SHA256 记录在 `asset_manifest.json`，其自身的哈希记录在 `asset_manifest.sha256`。
