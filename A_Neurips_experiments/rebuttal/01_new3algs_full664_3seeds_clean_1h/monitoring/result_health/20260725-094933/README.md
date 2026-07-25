# 热修复后正式结果健康扫描

## 口径

- State 更新时间：`2026-07-25T09:49:33`
- State SHA256：
  `0f042f4fd9a25d1da4a5b4ae742ed07dc91b49d8cfde88d05c4981c835f4d6c1`
- 截止时间：`2026-07-25T08:43:00`
- 只纳入 state 中标记为 `done`、位于 `assigned_host` 的规范外层
  `result.json`，且 `started_at` 或 `ended_at` 不早于截止时间。
- 不纳入 running/pending、非 assigned-host 副本或 SymbolFit 嵌套实验产物。

共扫描 `194/194` 条规范结果，8 台主机均成功返回。算法分布为
FePySR `60`、JAXSR `62`、SymbolFit `72`；其中 `28` 条在修复部署后
启动，另外 `166` 条在部署前启动、部署后完成。

## 结果

- `status != ok`：`0`
- `error` 非空：`0`
- `recovered_from_error=true`：`0`
- canonical artifact 无效：`0`
- Valid、ID 或 OOD NMSE 缺失、非数值或非有限：`0`
- 结果缺失、多份、不可读或非 JSON 对象：`0`
- `recovered_from_timeout=true`：`79`

这 `79` 条预算恢复结果全部满足 `status=ok`、`error=null`、
canonical artifact 有效且三个 split 的 NMSE 有限；运行时间范围为
`3604.885--3631.254s`。其中 SymbolFit `72` 条、JAXSR `7` 条、
FePySR `0` 条；`78` 条是首次尝试，`symbolfit_s520_clean_g0205`
是第二次尝试。

因此，当前发现的 timeout 恢复均为正常的 1 小时预算收口，不是失败。
本快照没有发现新的异常恢复、错误结果或指标空壳。

## 边界

部署前启动的 `166` 条结果不能用于证明其进程加载了新 runner；本报告
只证明这些结果本身在完成后满足健康检查。后续增量扫描仍需继续观察
`recovered_from_error`、非有限 NMSE 和无效 artifact。

## 文件

- `scan_context.json`：冻结口径、state 时间与哈希。
- `scan_summary.json`：分类计数及 `79` 条恢复结果的任务、主机和路径。
