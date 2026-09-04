# all_15alg_fullcpu_v1

这是 SSR-50 十五算法、三随机种子、三条件的全量 CPU 重跑队列资产。生成资产和 dry-run 不会连接远端；只有 `run_all_15alg_preflight.sh` 与 `run_all_15alg_formal.sh` 会访问服务器。

## 范围与排除

- 逻辑母集：`15 x 50 x 3 x 3 = 6750`。
- 已由 `all_conditions_cpu_v2` 覆盖且从新队列排除：`143`。
- 新队列：`6607`。
- `manifests/composite_ledger_6750.csv` 逐行记录母集归属，严格证明 `143 + 6607 = 6750`、集合无交集且并集等于母集。
- 算法：gplearn, llmsr, pyoperon, drsr, pysr, dso, tpsr, e2esr, fepysr, jaxsr, qlattice, imcts, udsr, ragsr, symbolfit。
- 条件：`clean`、`noise001`、`noise005`；种子：520、521、522。

## CPU 饱和口径

机器固定为 `iaaccn22~29`，每机实测 `256` 个逻辑 CPU、`128` 个物理核。首轮按 115 weight 派发后实测仍有 64%~78% CPU idle，因此每机 CPU weight 上限提高到 `230/256`，即 `0.8984375` 的逻辑 CPU 调度预算。

这里的“打满”是允许调度器一次补位到 230 CPU weight；并不强行把每个算法假定成持续满核，因为 LLM 等任务存在等待阶段。控制器仍会在真实 load 达到 0.90、内存达到保护线或 weight 用尽时停止追加。重启控制器会复用同一 state 和会话前缀，只补充空余配额，不停止或重复已启动任务。

资源保护：load 与内存使用率上限均为 0.90，每机至少保留 32 GB 可用内存，轮询间隔 30 秒。LLMSR 与 DRSR 共用 turbo 全局并发上限 30。

新旧 controller 的 `--session-prefix` 与 `--host-session-count-prefix` 都精确使用 `all_conditions_cpu_v2_`。因此新 controller 会把旧 143 个会话计入资源占用；同时 143 个 task_id 已从新 allowlist 排除，不会产生同名 session 冲突。

## 冻结输入

三份 source CSV 均为 50 行且 SHA256 完全一致。参数来自 formal3h 的 13 个算法、DRSR 和 LLMSR，共 `45` 个 condition 参数；每份均验证 `timeout_in_seconds=10800`、`progress_snapshot_interval_seconds=60` 及噪声字段。

`manifests/runtime_code_fingerprints.json` 固化 scheduler、launcher、runner、subprocess runner、artifact schema、normalizers、15 个 wrapper、iMCTS native regressor 与 toolbox config 的哈希。资产只复制算法参数，不复制 LLM config、API 密钥或其它凭证。

## 执行顺序

1. `commands/run_all_15alg_dry_run.sh`：仅本地构建/检查 6607 任务，不连接远端。
2. `commands/deploy_via_iaaccn22.sh`：先从本机同步到 iaaccn22，再由 iaaccn22 经内网同步到 23~29；本生成器不会执行它。
3. `commands/run_all_15alg_preflight.sh`：对八台 CPU 服务器执行正式前检查。
4. `commands/run_all_15alg_formal.sh`：单 controller 位于 iaaccn22，启用 `--force-rerun-existing` 并持续滚动补位。

生成器不会启动远端。资产哈希见 `asset_manifest.json` 与 `asset_manifest.sha256`。
