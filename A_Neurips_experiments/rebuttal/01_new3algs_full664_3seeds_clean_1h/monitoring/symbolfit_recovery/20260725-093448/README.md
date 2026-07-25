# SymbolFit g0205 恢复证据

任务：`symbolfit_s520_clean_g0205`

数据集：`sim-datasets-data/llm-srbench/phys_osc/PO16`

## 第一次尝试

- 主机：`iaaccn25`
- 运行时间：`3444.563s`
- 最终状态：`error`
- 终止原因：`error`
- 根因：SymbolFit 最终表达式数字格式化抛出
  `TypeError: Cannot convert complex to int`
- 最终 `result.json` 没有方程、artifact 或 split 指标。
- `minute_0057.json` 已在 `3420.354s` 写出有效方程和 canonical artifact；
  Valid、ID、OOD NMSE 都是有限值，但数值很大。

这证明第一次失败发生在搜索产出快照之后，而不是没有运行或没有候选。

## 第二次尝试

- 主机：`iaaccn23`
- 运行时间：`3604.961s`
- 最终状态：`ok`
- 终止原因：`budget_exhausted_with_output`
- 恢复路径：`recovered_from_timeout=true`
- canonical artifact：有效
- Valid NMSE：`0.1058993391092403`
- ID NMSE：`0.11244737442162067`
- OOD NMSE：`0.07905583153849755`
- `tool=symbolfit`、`seed=520`、`task_global_index=205` 和数据集身份均匹配。

第二次尝试没有再次走最终格式化异常，而是在 1 小时预算耗尽后由已有
timeout 快照恢复路径成功收口，因此不需要第三次尝试。正式 dispatcher
保留 `--retry-limit 2` 作为后续同类异常的容错保险。

控制器于 `09:39:14` 将该任务正式记录为 `done`，`attempts=2`、
`status_counts.ok=1`、`error=null`。

## 文件

- `first_attempt_task_status.json`
- `first_attempt_result.json`
- `first_attempt_minute_0057.json`
- `second_attempt_task_status.json`
- `second_attempt_result.json`
- `comparison_summary.json`
- `queue_state_after_recovery.json`
