# Core-50 噪声鲁棒性评测

## 实验目标

在 Core-50 上评估 12 个符号回归算法对训练标签噪声的鲁棒性。

本实验只对训练标签加噪，评测集保持干净：

- clean valid
- clean ID test
- clean OOD test

## 噪声协议

噪声水平：

```text
sigma in {0.01, 0.05, 0.10}
```

训练标签：

```text
y_noisy = y + sigma * std(y) * epsilon
epsilon ~ N(0, 1)
```

实现入口参数：

```json
{
  "train_label_noise_enabled": true,
  "train_label_noise_sigma": 0.01
}
```

`runner` 会保证：

- 噪声只传入 `reg.fit(X_train, y_train_noisy)`。
- 原始 `train.csv` 不会被改写。
- `train/valid/id_test/ood_test` 指标仍按 clean label 计算。
- `result.json` 记录 `train_label_noise` 元信息，包含 `sigma`、`scale`、`rng_seed`。

## 任务规模

每个噪声水平：

```text
50 datasets x 12 algorithms x 5 seeds = 3000 runs
```

三个噪声水平总计：

```text
9000 runs
```

## 目录约定

```text
generated/
  本实验的分发资产、参数文件、切片、队列 state 和启动脚本。

results/
  汇总回本地的最终结果、分析 CSV、leaderboard 和压缩包。
  不放远端训练过程文件。
```

## LLM 并发约束

`llmsr` 和 `drsr` 需要遵守全局模型桶并发限制：

```text
base  <= 100 running tasks
turbo <= 100 running tasks
```

调度策略：

- 优先派发 `llmsr/drsr`。
- 如果对应 LLM 模型桶已满，则继续派发非 LLM 算法，避免机器空转。

## 远端原始结果

远端原始运行目录仍建议使用：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/<batch_name>/
```

本地最终汇总结果统一回收到本目录的：

```text
results/
```
