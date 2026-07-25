# 2026-07-25 09:16 重试上限恢复

## 原因

`symbolfit_s520_clean_g0205` 的第一次尝试在算法已经运行约 57 分钟并
写出可用快照后，因上游最终表达式数字格式化异常而失败。第二次尝试
启动于 `08:34:39`，早于异常快照恢复 runner 在远端的部署时间，因此
该进程仍然加载旧代码。

控制器当时使用 CLI 默认 `retry_limit=1`。按调度器语义，这只允许初次
运行后再重试一次；若第二次再次复现确定性异常，任务会直接标记失败，
无法启动能够加载新 runner 的第三次尝试。

## 修复

- 正式 dispatcher 显式增加 `--retry-limit 2`。
- 这是队列容错设置，不改变 FePySR、JAXSR 或 SymbolFit 的算法超参数、
  随机种子、运行预算、数据和噪声口径。
- 最大尝试次数变为初次运行加两次重试；只有真实失败才会消耗重试。
- 没有手工修改 task state，也没有停止或重启任何算法 worker。

## 恢复结果

- 原控制器 PID 为 `170080`，新控制器 PID 为 `225994`。
- tmux session 保持为 `neurips_rebuttal_new3_full_controller`。
- 新控制器命令行包含 `--retry-limit 2`，单实例锁指向新 PID。
- 8 台主机全部重新同步支持文件。
- 首轮轮询后队列为 `789 done / 236 running / 4951 pending`。
- 后续调度于 `09:27:45` 更新为
  `789 done / 248 running / 4939 pending`，仍无 failed/error 任务。
- 跨主机盘点为 `236` 个 session、`236` 个唯一任务、零重复任务，
  且只有一个控制器。
- `symbolfit_s520_clean_g0205` 仍保持第二次尝试运行；若它再次失败，
  控制器会自动启动加载新 runner 的第三次尝试。

## 证据

- `restart_request.json`：重启前 PID、控制器命令和 state 时间。
- `restart_result.json`：新 PID、tmux session 和单实例锁。
- `latest_before_recovery.json`：受控重启前的队列摘要。
- `latest_after_recovery.json`：新控制器恢复轮询并继续调度后的队列摘要。
- `controller_command_after_recovery.txt`：包含显式重试上限的实际命令行。
- `session_inventory_after_recovery.json`：跨 8 台主机的 session 唯一性盘点。
- `queue_health_after_recovery.json`：任务状态、尝试次数和错误分布。
- `symbolfit_g0205_state_after_recovery.json`：目标重试任务的权威 state 摘要。
