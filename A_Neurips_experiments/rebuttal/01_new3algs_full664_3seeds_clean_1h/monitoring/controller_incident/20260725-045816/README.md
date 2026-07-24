# 双控制器事件记录

## 现象

- `2026-07-25 04:51` 前后，队列完成计数出现非单调读取。
- `events.jsonl` 显示同一个 `task_id` 在不同主机重复完成。
- 进程核验发现同一 batch 有两个控制器：
  - PID `84096`，启动于 `03:23:19`，由正式 tmux session 持有。
  - PID `112122`，启动于 `03:38:34`，由额外 nohup shell 持有。

## 处理

1. 在处理前冻结 state、latest、events 和控制器进程证据。
2. 提交 `46d49db`，为同一 `batch_name` 增加基于 `flock` 的单实例锁。
3. 终止较新的重复控制器 PID `112122`，保留正式 tmux 控制器。
4. 枚举八台主机的存活 session：
   - `134` 个 session；
   - `78` 个唯一任务；
   - `56` 个任务各有一个重复 session。
5. 按 state 中的 `assigned_host` 保留规范副本：
   - 停止 `51` 个仍存活的非规范 session；
   - 另有 `5` 个非规范 session 在执行前已自然结束；
   - 没有停止任何 state 指定的规范 session。
6. 受控重启正式控制器，使其加载单实例锁。

## 验证

- 重启后的控制器 PID 为 `5222`，锁文件记录于
  `controller.lock.after_restart.json`。
- 第二次同 batch 加锁测试返回码为 `1`，并报告已有控制器持锁。
- 清理后的存活 session 数与唯一任务数一致，重复任务数为 `0`。
- 没有删除或移动任何结果目录；最终 harvest 继续按 state 中的
  `assigned_host` 选择规范结果。

## 文件

- `state.before.json` / `latest.before.json` / `events.before.jsonl`
- `controllers.before.txt`
- `live_sessions.after_duplicate_controller_stop.tsv`
- `duplicate_sessions_to_stop.tsv`
- `stop_duplicate_sessions.sh`
- `stop_duplicate_sessions.log`
- `live_sessions.after_cleanup.tsv`
- `state.before_locked_restart.json`
- `controller.before_locked_restart.txt`
- `controller.lock.after_restart.json`
- `live_sessions.after_locked_restart.tsv`
- `duplicate_task_ids.after_locked_restart.txt`
