# E2ESR 最终选模与恢复验证

## 已完成

用户已授权统一 E2ESR 的快照、最终模型和超时恢复口径。本轮不再用解码概率选公式，
而是使用传给 fit 的完整训练集上的 MSE，方向为 min。ID/OOD 仅用于选模之后评估。
新版本标记：`selection_policy=e2esr_training_mse_v1`。

主要修复：

1. 每个 bag 生成后立即精化，遵守 bag 数量和时间上限，不把全部预算耗在 forward 上。
2. raw 和 BFGS 候选共同按训练 MSE 排名；后续更差的轮次不覆盖已有最佳结果。
3. BFGS 的预测 `(N,1)` 与目标 `(N,)` 统一为逐样本向量，消除错误的 `(N,N)` 广播损失。
   最小复现中，旧实现把真值斜率 2 优化到约 0；新实现恢复到 2。
4. 优化器从第一次评估就正确记录最佳损失，并复制参数，防止后续更差点或原地修改污染最佳值。
5. 修复变量重标的级联替换、逆缩放越界变量死循环、多位变量索引及常量列的非有限缩放。
   固定预训练词表不变，越界变量按原生预测语义投影为 0。
6. 只接受新口径且训练 loss 一致、有限的恢复快照；不再回退使用旧 likelihood 结果。
7. 已有有效快照时，直接 API 恢复无需加载 358MB 预训练权重，支持预测与轻量序列化。
   子进程预测/读取方程/导出也跳过无用的模型构造；其他 wrapper 默认保留原构造流程。

## 最终验收结果

所有算法运行均设置外部 180 秒硬上限。以下为本轮最终验收，非全量榜单重跑：

| 场景 | 墙钟秒数 | ID R2 | OOD R2 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 平移/缩放仿射任务，45 秒内部预算 | 37.16 | 1.000000 | 1.000000 | 正常完成，BFGS 有效 |
| CRK0，150 秒内部预算 | 142.62 | 0.999755 | -8.675527 | 正常完成，13 个完整轮次 |
| CRK0，正式 benchmark 链路，45 秒 fit 预算 | 120.44 | 0.999722 | -0.353128 | result.json 为 ok |
| CRK0，主动在 45 秒中断 | 45.06 | 0.999722 | -0.353128 | 从当时最佳快照恢复 |

硬中断记录的墙钟包含发出终止信号后约 0.06 秒的进程回收时间，训练预算没有延长。
正常完成任务的原生预测、导出公式、已选快照与恢复回放一致，最大误差仅为浮点舍入量级。
实际 Python 3.7 恢复验证不加载权重，恢复及预测/方程/导出分发约 0.154 秒（不含 Python 启动），
状态约 1.6KB。完整 benchmark 链路还包含模型序列化和多次跨环境调用，其墙钟见表格。

CRK0 的 150 秒结果比 45 秒结果具有更小的训练误差，但 OOD 更差。
这里保留训练口径选出的最终结果，没有按 OOD 分数挑选较早或其他轮次。
因此修复的是实现与选模一致性，不能宣称已经解决算法本身的外推误差。

## 交付

- [最终结果汇总](evidence/e2esr_final_selection_20260921/summary.json)
- [正式 benchmark result.json](evidence/e2esr_final_selection_20260921/benchmark45_final/runs/e2esr/CRK0/result.json)
- [完整数据、预测数组、快照及运行证据](evidence/e2esr_final_selection_20260921/)
- [SHA256 清单](evidence/e2esr_final_selection_20260921/SHA256.json)

复现当前诊断：

```bash
PYTHONPATH=. python diagnostics/three_algorithm_quality_probe.py \
  --algorithm e2esr --python /home/family/anaconda3/envs/sim_e2esr/bin/python \
  --dataset sim-datasets-data/llm-srbench/chem_react/CRK0 \
  --seconds 180 --fit-seconds 150 --output /tmp/e2esr_final_new
```

在上述命令中增加 `--hard-stop-seconds 45` 可验证强制中断后的快照恢复。
该模式退出码为 124，不伪装成自然完成，恢复公式和评估保存在 report.json 中。

正式 runner 的短预算验收命令：

```bash
PYTHONPATH=. python -m scientific_intelligent_modelling.cli \
  --algorithm e2esr --train-path sim-datasets-data/llm-srbench/chem_react/CRK0 \
  --seed 520 --timeout-in-seconds 45 --max-number-bags 1 --n-trees-to-refine 4 \
  --progress-snapshot-interval-seconds 60 --output-root /tmp/e2esr_benchmark_new
```

`timeout-in-seconds` 是 fit 子进程预算，不是整条 CLI 的总墙钟预算；本轮正式链路另外用
180 秒进程树监控器保证总运行上限，含跨 session 的子进程清理。

187 项聚焦回归通过。全量快照测试仍有 1 项之前已在未修改基线上复现的 SymbolFit
`final_best` / `budget_end_internal_best` 断言失败，与本轮 E2ESR 无关。
没有将调试过程中的旧失败记录作为新实验结果发布；历史实验目录没有参与此次改动。
