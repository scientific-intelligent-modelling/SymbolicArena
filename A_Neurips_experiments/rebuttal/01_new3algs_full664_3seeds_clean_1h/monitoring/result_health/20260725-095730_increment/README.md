# 新增完成结果健康扫描

## 口径

- 基线 state 更新时间：`2026-07-25T09:49:33`
- 基线 state SHA256：
  `0f042f4fd9a25d1da4a5b4ae742ed07dc91b49d8cfde88d05c4981c835f4d6c1`
- 当前 state 更新时间：`2026-07-25T09:57:30`
- 当前 state SHA256：
  `6d0b6f691bf3dd9d5bf491558bdc9e24a89c62edbc909b1162f67b3d563fd23e`
- 只纳入当前 state 中为 `done`、但基线 state 中尚未为 `done` 的任务，
  并且只读取任务 `assigned_host` 上唯一的规范外层 `result.json`。
- 不纳入两个快照之外的新增完成任务、running/pending、非 assigned-host
  副本或 SymbolFit 嵌套实验产物。

基线为 `894` 个 done，当前快照为 `938` 个 done，因此新增 `44` 个
done。算法分布为 FePySR `14`、JAXSR `15`、SymbolFit `15`；全部属于
seed `520`，且全部是第一次尝试。主机分布为：
`iaaccn22=2`、`iaaccn23=5`、`iaaccn24=8`、`iaaccn25=7`、
`iaaccn26=7`、`iaaccn27=7`、`iaaccn28=4`、`iaaccn29=4`。

## 结果

- 唯一且可读取的规范结果：`44/44`
- `status=ok`：`44/44`
- `error` 为空：`44/44`
- canonical artifact 有效：`44/44`
- Valid、ID 和 OOD NMSE 均为有限值：`44/44`
- tool、seed、global index、dataset 和 dataset_rel 身份一致：`44/44`
- `recovered_from_error=true`：`0`
- `recovered_from_timeout=true`：`18`
- 任意失败信号：`0`

这 `18` 条 timeout 恢复结果同样通过 artifact、三个 split 指标和身份
检查，属于现有 1 小时预算恢复路径的健康结果。

## 边界

本报告只验证两个固定 queue state 之间新增完成结果的结构、指标和身份
健康性，不替代最终的运行时、clean 条件、超参数、预算和全网格完整性
审计。最终结论仍须等待 `5976/5976` 完成后运行正式 collect、harvest
和 audit gate。

## 文件

- `scan_context.json`：两个冻结 state、数据目录的哈希和选择边界。
- `scan_summary.json`：8 台主机汇总后的健康检查计数。
