# E2ESR / TPSR / RAG-SR 短预算质量诊断

本文保留第一阶段的诊断记录。用户随后已批准修正 E2ESR 的选模与精化逻辑，
当前实现及最终验收见 [E2ESR 最终选模与恢复验证](e2esr_final_selection_20260921.md)。

## 范围与证据

基线为 `94a7202b`。共完成 12 次真实算法诊断，每次均有不超过 180 秒的外部硬超时；
最长一次约 91.14 秒，另一次主动在 30 秒硬停止以验证快照恢复场景。
E2ESR / TPSR 使用本机 CPU 与已有共享权重；RAG-SR 使用 iaaccn22 的已有
Evolutionary Forest 0.2.5 环境。远端修复版仅通过 `/tmp` 文件加载，未部署到正式仓库。
本轮无付费 API、无依赖安装，也未改写任何历史实验或重新计算正式六轴榜单。

- [逐次运行与比较汇总](evidence/three_algorithm_quality_20260921/summary.json)
- [原始报告、输入、预测数组、进度候选及日志](evidence/three_algorithm_quality_20260921/)
- [SHA256 清单](evidence/three_algorithm_quality_20260921/SHA256.json)

所有比较使用同一套 train/ID/OOD 数据，固定 seed=520。指标为 MSE、
`NMSE = MSE / mean(y^2)`（越小越好）和 `R2 = 1 - MSE / var(y)`（越大越好）。
这里没有跨数据集或种子聚合，不可当作正式排名。异常预测显式记录为无效，不静默填零。

## 已确认并修复

### E2ESR：进度公式缺少逆标准化

`fit()` 在局部特征坐标中标准化 X，但 forward/refinement 期间直接输出 scaled tree。
runner 的进度评估和超时恢复则以原始 X 回放它。自然完成的最终 tree 原先已经逆缩放，
因此会出现“最终原生模型很好，训练中/超时恢复结果很差”。

现在按数据集保留 scaler 和参数，只对待输出的树副本逆缩放，再恢复真实特征索引。
最终已逆缩放的树不重复转换；失败时不输出错误坐标公式。保留原有 likelihood 排名。

### TPSR：中间和最终路径不一致

训练中 `e2e_candidate` / `e2e_terminal` 返回标准化公式，只有 `e2e_final` 做了逆缩放。
现在三条路径共用原坐标表达式转换；树不原地修改，字符串兜底也必须完成同步变量替换。
缺少转换状态或无法解析时不再悄悄返回 scaled 表达式。搜索 reward、词表和参数未改。

### RAG-SR：序列化漏掉 X 变换

后端 `model()` 只逆变换 y，并使用 `x0/x1` 变量。原 wrapper 只识别 `ARG0/x_0`，
导致序列化后在原始 X 上执行本应接收归一化 X 的表达式。
现在单次匹配三种变量格式，按 MinMax 的 `scale_`、`min_` 嵌入变换，避免二次替换；
同时覆盖常量列、非默认 feature_range 和 StandardScaler，不重复逆变换 y。

## 数值结果

平移和缩放后的二维仿射任务，真值 `y = 2*x0 - 3*x1 + 0.7`：

| 路径 | 修复前 ID R2 | 修复后 ID R2 |
| --- | ---: | ---: |
| E2ESR 原生 likelihood incumbent 快照回放 | -120.510926 | 0.999926 |
| TPSR 训练中快照回放 | -120.510926 | 0.999926 |
| RAG-SR 序列化后预测 | -730.178554 | 0.996245 |

RAG-SR 修复前后原生模型预测完全相同；修复后 ID 原生/恢复预测最大差为 `1.42e-14`。
E2ESR / TPSR 最终原生与导出公式的 ID 最大预测差分别约 `7.11e-15`、`0`。
这些是坐标修复的证据，不是通过调整超参数得到的排名提升。

真实 CRK0 的短预算结果如下，三者均完成且原生/导出回放一致：

| 算法 | 总运行秒数 | 最终模型 ID R2 | 最终模型 OOD R2 |
| --- | ---: | ---: | ---: |
| E2ESR | 52.13 | 0.997282 | -20.502911 |
| TPSR | 59.95 | 0.636832 | 0.573448 |
| RAG-SR | 7.11 | 0.999999940 | 0.999998596 |

RAG-SR 的同参数修复前 CRK0 原生模型也达到上述精度，但恢复后 ID R2 为 `-2.307647`、
OOD R2 为 `-9392.016288`；修复后最大预测差小于 `1.14e-16`。
这是独立于仿射测试的真实任务复现。

## 尚未修改的原因

1. **E2ESR 选模口径不一致，已实证。** CRK0 最终模型 ID R2 为 `0.997282`，
   但坐标修复后的最高解码 likelihood 快照仍只有 `-2.683686`。
   解码概率不是训练误差；当前 BFGS 候选缺少 likelihood，因此不能替换该 incumbent。
   这会影响超时恢复和基于进度快照的统计。本轮没有擅自改动已有评测合同，已向用户询问。
2. **E2ESR 预算调度风险。** 开启进度和 timeout 后，内部 forward 会取消 max_bags 上限，
   一直采样到预算末尾才精化；外层计时又包含启动/加载，可能更早硬停止。
   30 秒诊断确实在 fit 中被终止并留下候选。不能把这种情况解释成“E2ESR 无法拟合”。
3. **仍有真实搜索/外推误差。** CRK0 的 E2ESR 最终模型 OOD 仍较差，TPSR 短预算 ID
   也未达高精度。这部分不是坐标修复能消除的，不能据两个任务宣称全量问题已经解决。
4. **非默认 RAG ensemble。** EF `ensemble_size > 1` 的 y-scaler 导出还有独立兼容性风险，
   本轮默认 `ensemble_size=1`，未扩展修改该第三方后端。

## 复现

```bash
PYTHONPATH=. python diagnostics/three_algorithm_quality_probe.py \
  --algorithm e2esr --python /home/family/anaconda3/envs/sim_e2esr/bin/python \
  --case shifted --seconds 180 --fit-seconds 45 \
  --output /tmp/e2esr_quality_new
```

可将 algorithm/python 对应换成 tpsr/sim_tpsr，或在已安装后端的环境中使用 ragsr/sim_ragsr。
用 `--dataset sim-datasets-data/llm-srbench/chem_react/CRK0` 替代合成任务。
输出目录必须不存在，避免覆盖证据。诊断专用参数刻意缩小规模，不代表正式 3h 预算。
最终诊断器以退出码 `0` 表示训练完成且原生/导出预测一致，`1` 表示 worker 或回放验证失败，
`124` 表示硬超时；不会因为报告已落盘就把超时当作成功。已存的 12 次原始报告保留当时格式。

回归测试覆盖：逆缩放、特征重标、最终输出不二次转换、候选不突变、三种变量别名、
MinMax 常量列与范围、原生/恢复预测一致，以及 180 秒上限与超时进程回收。
最终聚焦回归为 `91 passed`，所有单次 pytest 命令均有 60 秒上限。
扩大测试发现 SymbolFit `final_best` / `budget_end_internal_best` 的既有断言失败；
已在未修改 HEAD 的独立归档上复现，未为本次修复改动它。

下一步最小闭环：确认 E2ESR 的选模口径后，修复快照选取和 forward/refine 预算分配，
仍先跑不超过 180 秒的对照；正式重跑必须新建批次，保留旧结果及修复版本绑定。
