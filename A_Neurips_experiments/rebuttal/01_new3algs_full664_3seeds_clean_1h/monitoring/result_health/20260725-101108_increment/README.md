# 第二轮新增完成结果健康扫描

## 口径

- 基线 state 更新时间：`2026-07-25T09:57:30`
- 基线 state SHA256：
  `6d0b6f691bf3dd9d5bf491558bdc9e24a89c62edbc909b1162f67b3d563fd23e`
- 当前 state 更新时间：`2026-07-25T10:11:08`
- 当前 state SHA256：
  `c453f59cf49bd956b04abf947352564fdb086f8859a24887c6abbee8028f4a2e`
- 只纳入当前 state 中为 `done`、但基线 state 中尚未为 `done` 的任务，
  并且只读取任务 `assigned_host` 上唯一的规范外层 `result.json`。
- 不纳入两个快照之外的新增完成任务、running/pending、非 assigned-host
  副本或 SymbolFit 嵌套实验产物。

基线为 `938` 个 done，当前快照为 `998` 个 done，因此新增 `60` 个
done。算法分布为 FePySR `20`、JAXSR `20`、SymbolFit `20`；全部属于
seed `520`，且全部是第一次尝试。主机分布为：
`iaaccn22=9`、`iaaccn23=6`、`iaaccn24=7`、`iaaccn25=10`、
`iaaccn26=6`、`iaaccn27=6`、`iaaccn28=8`、`iaaccn29=8`。

## 结果

- 唯一且可读取的规范结果：`60/60`
- `status=ok`：`60/60`
- `error` 为空：`60/60`
- canonical artifact 有效：`60/60`
- Valid、ID 和 OOD NMSE 均为有限值：`60/60`
- tool、seed、global index、dataset 和 dataset_rel 身份一致：`60/60`
- `recovered_from_error=true`：`0`
- `recovered_from_timeout=true`：`23`
- 任意失败信号：`0`

这 `23` 条 timeout 恢复结果同样通过 artifact、三个 split 指标和身份
检查，属于现有 1 小时预算恢复路径的健康结果。

## 边界

本报告只验证两个固定 queue state 之间新增完成结果的结构、指标和身份
健康性，不替代最终的运行时、clean 条件、超参数、预算和全网格完整性
审计。`20260725-101556` completed audit 已对稍后的 `1007` 个 done 结果
执行完整契约审计并通过；最终结论仍须等待 `5976/5976` 完成。

## 文件

- `scan_context.json`：两个冻结 state、数据目录的哈希和选择边界。
- `scan_summary.json`：8 台主机汇总后的健康检查计数。
