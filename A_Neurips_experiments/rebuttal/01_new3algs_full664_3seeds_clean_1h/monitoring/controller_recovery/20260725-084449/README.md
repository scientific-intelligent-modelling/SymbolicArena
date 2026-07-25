# 2026-07-25 08:44 受控恢复

## 目的

将以下两个运行时修复加载到正式队列，同时不停止现有算法 worker：

- 按主机批量回收同一轮已经完成的任务进程，避免完成波峰被逐任务 SSH
  串行回收拖慢。
- 对支持分钟快照的算法，在搜索已经产出可用结果、但最终格式化阶段异常时，
  仅从通过完整可用性检查的快照恢复最终结果。

## 恢复结果

- 原控制器 PID 为 `30910`，新控制器 PID 为 `170080`。
- tmux session 保持为 `neurips_rebuttal_new3_full_controller`。
- 新控制器在 8 台主机上全部完成支持文件同步，并在
  `08:49:49`、`08:52:05`、`08:54:43` 连续完成主机探测。
- 事件窗口内有 `26` 个完成任务，`26/26` 进程回收成功。
- `08:49:28` 的完成波峰中，同一主机一次批量回收最多 `4` 个任务。
- `08:52:49` 会话盘点为 `200` 个 session、`200` 个唯一任务、
  `0` 个重复任务，且只发现一个控制器。
- `08:55:06` 队列状态为 `741 done / 210 running / 5025 pending`；
  `08:57:14` 健康审计未发现 failed/error 任务。

## 证据

- `restart_request.json`：重启前 PID、命令和 state 时间。
- `restart_result.json`：新 PID、tmux session、单实例锁和恢复结果。
- `latest_before_recovery.json`：受控重启前的队列摘要。
- `latest_after_recovery.json`：新控制器完成轮询后的队列摘要。
- `session_inventory_after_recovery.json`：跨 8 台主机的 session 唯一性盘点。
- `queue_health_after_recovery.json`：任务状态、重试和错误分布。
- `events_after_recovery.jsonl`：恢复后的支持同步、回收、完成、探测和批量启动事件。
- `event_summary.json`：从上述事件提取的批量回收与轮询摘要。

本次操作没有修改队列 task state，也没有停止或重启算法 worker。
