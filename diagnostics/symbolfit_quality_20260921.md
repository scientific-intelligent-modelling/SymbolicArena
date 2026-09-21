# SymbolFit 修复与验收范围

## 三算法覆盖澄清

“已修复明确的实现问题”和“效果问题全部解决”不同；不能把 E2ESR 的 10 数据集测试
算到 TPSR、RAG-SR 上。这里统计的是修复后的诊断记录，不包含旧的正式实验。

| 算法 | 上一轮指定的 10 数据集覆盖 | 已有额外实跑 | 是否可宣布全部修好 |
| --- | ---: | --- | --- |
| E2ESR | 10/10 | 两次复跑、合成边界任务 | 否，Keijzer-5 的 ID、BPG3 的 OOD 仍差 |
| TPSR | 1/10，Korns-1 | CRK0、合成仿射任务 | 否，尚未完成同组验收 |
| RAG-SR | 0/10 | CRK0、合成仿射任务 | 否，尚未完成同组验收 |

前一轮完整证据见 [E2ESR 多数据集复验](e2esr_multidataset_validation_20260921.md)。
本轮没有额外训练这三个算法，因此还缺 TPSR 的 9 项、RAG-SR 的 10 项同组检查。

## 本轮修复

1. **负参数替换改变运算含义**：原来直接把负数拼到字符串，`a1**2` 在 a1=-2 时
   变成 `-2.0**2`。现在用 SymPy 结构化替换，并拒绝非有限 best-fit 参数。
   复现中的预测从错误的 [-5, -6] 修复为 [3, 2]，同时覆盖 a1/a10 不混淆。
2. **无效候选参与选模**：NaN RMSE 可能被 `min()` 选中。现在排除 NaN/Inf/负 RMSE，
   所有候选无效时明确报错，非有限 R2 不作为更优的 tie-break。
3. **实际缩放与元数据兜底不一致**：常量列导致上游除零，现在仅对该次 fit 关闭其
   不支持的缩放；目标均值/最大值/范数为零或相对于数据幅值仅剩浮点残差时，关闭目标缩放。
   不修改用户原始参数，记录实际生效的坐标变换和原因。极小但正常的目标幅值仍允许缩放。
4. **无候选心跳冒充最终最佳**：runner 只有真正存在候选公式时才输出
   `budget_end_internal_best`；没有候选时保留 wrapper 已生成的最终公式和指标，输出 `final_best`。
   已有候选即使 OOD 回放失败，也保留其失败证据，不用测试集表现决定换公式。

原来失败的 `non_snapshot_tool` 测试错误地使用了已经支持快照的 SymbolFit。
现在分别覆盖真正的非快照工具、SymbolFit 有候选、无候选和候选评估失败场景，
不是直接把断言从 `final_best` 改成 `budget_end_internal_best`。

最终联合回归：278 passed，18.40 秒；其中完整快照测试文件 39 passed。
命令统一使用 `timeout 60s`，未做有训练副作用的全仓 pytest 收集。

保留已有评估约定：搜索快照优先 PySR 内部 loss，wrapper 最终选取 LMFIT refit RMSE 最优候选。
本轮没有把两个目标混用，也没有重写历史正式实验。

## 实跑证据

在 iaaccn22 的既有环境运行，使用 `/tmp/symbolfit_debug_20260921/` 隔离副本，
未部署覆盖远端正式仓库。版本：SymbolFit 0.2.5、PySR 1.5.9、LMFIT 1.3.4。
每个 worker 180 秒硬上限；共 9 次运行，最长 100.61 秒，均无硬超时。
其中 2 次旧版对照、1 次仅处理精确零但未处理浮点残差的中间版、6 个修正后的验收场景。

| 场景 | 旧版 | 修正后 |
| --- | --- | --- |
| 常量输入列 | `Input X contains NaN`，无法训练 | ID/OOD R2 均为 1 |
| 数学上零均值、浮点残差约 4e-17 | 学成常量，ID/OOD R2 均为 0 | ID/OOD R2 均为 1 |

其余短预算结果如下；全部通过原生预测、独立参数绑定回放、导出、序列化恢复的一致性检查。
原生模型预测正确地被导出，不等于它已经找到高质量公式。

| 场景 | ID R2 | OOD R2 |
| --- | ---: | ---: |
| 平移缩放仿射任务 | 0.884695 | -1.087268 |
| Nguyen-1 | 0.321218 | 0.817957 |
| Korns-1 | 1.000000 | 1.000000 |
| BPG3 | 0.999247 | -134.583961 |

这些诊断使用 seed=520、45 秒内部预算、3 次搜索迭代、maxsize/max_complexity=8、
仅 sin/cos 一元运算，`fill_timeout_budget=False`。此配置刻意缩小结构搜索空间，
甚至不足以表达部分目标公式，不能用它评价默认参数性能，更不能把剩余误差全归因于算法 bug。
Nguyen-1 的 ID 还只有 2 个点。三份真实数据的 CSV/metadata 哈希已与本地逐文件核对一致。

指标：MSE=`mean((y-pred)^2)`，NMSE=`MSE/mean(y^2)`，越小越好；
R2=`1-MSE/var(y)`，越大越好。未跨数据集平均或重新排名，ID/OOD 没有参与选模。

## 交付与限制

- [逐 run 摘要](evidence/symbolfit_quality_20260921/summary.json)
- [逐 run CSV](evidence/symbolfit_quality_20260921/summary.csv)
- [原始完整日志与证据归档](evidence/symbolfit_quality_20260921/raw_evidence.tar.gz)
- [可复用诊断脚本](symbolfit_quality_probe.py)
- [测试记录](evidence/symbolfit_quality_20260921/verification.json)
- 每个 run 保存输入、预测、refit 候选表、版本、源码哈希、原生快照及实际经过分钟的记录。
- 仅在接近零分母分支上继续修正过中间版；其他验收输入不触发该分支，报告保留各自真实源码版本。
- 没有付费裁决 API 调用；缺失正式 SYM/MIN/EFF/STAB 记录为 unresolved，`formal_ready=false`。

本轮证明上述实现问题得到修复，但不宣称 SymbolFit 在全部数据集、全部参数设置下效果良好。
