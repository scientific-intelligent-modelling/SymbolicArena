# E2ESR 多数据集复验

## 结论

这次验证支持“已检查的预测、导出、快照、训练 MSE 选模和恢复路径保持一致”，
不支持“所有数据集的拟合和外推问题都已解决”。SymbolFit 没有在本轮修复。

预先固定 10 个数据集，全部正常完成，原生预测、导出公式、快照回放一致。
补修变量编号边界后，对全部数据做公式回放及 Python 3.7 的真实恢复 API 检查，10/10 通过；
另外重新训练 Korns-1、BPG3，两者也通过。没有用 ID/OOD 挑选候选或决定保留哪一轮。

## 运行口径

- 算法：E2ESR；seed=520，CPU 单线程，两路并发。
- 每个真实数据任务：120 秒内部预算，180 秒外部进程树硬上限，包含初始化和评估。
- `max_number_bags=1`、`n_trees_to_refine=4`，属于短预算诊断，不是默认正式榜单参数。
- 传入完整 train；模型内部每个 bag 最多 200 点，选模损失计算于完整训练集。
- 10 次首轮 + 2 次修正后重跑 + 1 次常量列合成检查，共 13 次算法运行，无硬超时。
- 真实数据首轮整条诊断最大约 115.89 秒；两次重跑约 104 秒；合成检查约 36.30 秒。

## 精度结果

下表为原生模型 R2，越大越好。Korns-1、BPG3 采用修正后重跑结果，其余采用首轮结果；
这两次重跑的 ID/OOD 精度与首轮相同。分母始终是预选的全部 10 个数据集。

| 数据集 | 维数 | ID 样本数 | ID R2 | OOD R2 |
| --- | ---: | ---: | ---: | ---: |
| Nguyen-1 | 1 | 2 | 1.000000 | 1.000000 |
| Nguyen-5 | 1 | 2 | 0.999991 | 0.955065 |
| Nguyen-7 | 1 | 2 | 0.999884 | 0.981588 |
| Nguyen-12 | 2 | 2 | 0.998698 | 0.988053 |
| Keijzer-5 | 3 | 10000 | -77425.272820 | -0.285737 |
| Korns-1 | 5 | 100 | 1.000000 | 1.000000 |
| Vladislavleva-4 | 5 | 102 | 0.999493 | 0.999524 |
| Feynman I.6.2b | 3 | 8000 | 0.990254 | 0.946477 |
| Feynman I.9.18 | 9 | 8000 | 0.996043 | 0.942938 |
| BPG3 | 2 | 500 | 1.000000 | -262.459137 |

9/10 的 ID R2 >= 0.99，3/10 的 OOD R2 >= 0.99；这只是描述性阈值，不是修复验收标准。
CSV 保留未四舍五入的值、MSE、NMSE 和各 split 样本数。MSE=`mean((y-pred)^2)`；
NMSE=`MSE/mean(y^2)`，两者越小越好；R2=`1-MSE/var(y)`。没有对不同数据集分数求均值排名。

主要剩余风险：

1. Keijzer-5 的原生模型本身 ID 误差极大，导出和恢复忠实重现该误差，不能归咎于导出错位。
2. BPG3 的 ID 极好但 OOD 很差。OOD MSE 约 4.55e-11，但该 split 的目标变化也很小，
   NMSE 仍约 229.54，不能用很小的绝对 MSE 掩盖外推失败。
3. Nguyen 系列 ID 只有 2 个点，单独用 ID 判断泛化不可靠。
4. 单 seed、短预算、少量 bag 不能证明多 seed 稳定性或正式长预算表现。

## 补充修复

10 个实测最终公式未触发新发现的稀疏变量边界，但定向测试成功复现：
公式不含 x_0 时，旧 normalizer 会将 x_1/x_10 左移成 x0/x9；
还可能把越界变量错误地改成合法列。修复前新增导出相关测试为 4 failed / 1 passed。

- E2ESR 的公共 normalizer 明确保持零基编号，移除仅恢复路径使用的零系数锚点补丁。
- TPSR 按实际后端确定原生编号：E2E 零基、NeSymReS 一基；预测保留原生变量，
  对外方程、快照和导出统一转换为零基，不根据“缺少 x0”猜测编号。
- NeSymReS 的实际配置使用 x_1/x_2/x_3，因此不能简单把带下划线的变量都判为零基。
- 额外合成任务验证首列恒为 50、第二列变化时仍能拟合和恢复，ID/OOD R2 均为 1。
  它不是正式数据集，也不计入上述 10 个数据集的分母。

最终聚焦回归为 198 passed。NeSymReS 的一基转换、原生预测、快照和反序列化有
定向测试覆盖，但没有加载 NeSymReS 权重重训；缺少 `params.backbone_model` 的极旧
TPSR 状态仍按 E2E 默认解释。本轮没有批量迁移历史 TPSR 快照。

另补做一次 TPSR E2E 后端的 Korns-1 真训练：45 秒内部预算、150 秒 worker 硬上限，
实际 64.55 秒正常完成，原生与导出预测逐点一致；ID R2=0.9999990533，
OOD R2=0.9999988846。它单独保存在 `tpsr_korns1_smoke/`，不混入 E2ESR 的分母。
TPSR 内部软预算并非严格墙钟截止，因此仍需保留外部硬超时。

## SymbolFit

`ec490b85`、`cb75c3d7` 及本轮均未修复 SymbolFit 算法。
`test_run_benchmark_task_writes_final_progress_for_non_snapshot_tool` 仍失败：
预期 `final_best`，实际 `budget_end_internal_best`。完整快照测试为 36 passed / 1 failed。
这是一项已知终态记录语义问题，不能声称已修复，也没有为得到绿色测试而更改断言。

## 证据

- [逐数据集 CSV](evidence/e2esr_multidataset_20260921/summary.csv)
- [机器可读摘要](evidence/e2esr_multidataset_20260921/summary.json)
- [原始数据哈希、命令和首轮结果](evidence/e2esr_multidataset_20260921/initial/manifest.json)
- [修正后两次重跑](evidence/e2esr_multidataset_20260921/fixed/execution_summary.json)
- 每个 run 保留输入/预测数组、worker 日志、候选历史、最终快照、API 恢复及 post-fix audit。
- [已完成整分钟的证据](evidence/e2esr_multidataset_20260921/minute_evidence.jsonl)
- [未完成正式六轴的显式清单](evidence/e2esr_multidataset_20260921/unresolved.json)

本轮不调用付费裁决 API，没有把缺失 SYM/MIN/EFF/STAB 填零或包装成正式六轴发布；
保留完整候选时间线和数值证据，`formal_ready=false`。历史正式实验和数据子仓库没有修改。
