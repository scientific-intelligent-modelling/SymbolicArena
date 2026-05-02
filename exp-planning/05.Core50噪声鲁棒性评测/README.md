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
50 datasets x 12 algorithms x 3 seeds = 1800 runs
```

三个噪声水平总计：

```text
5400 runs
```

本实验 seed 固定为：

```text
0, 1, 2
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

## LLM base/turbo 分发规则

`base` 和 `turbo` 在本实验中视为同一个 Llama-3.1-8B 模型的两个 API 并发桶。
它们不作为独立算法变体进入 leaderboard。

分发目标：

- `llmsr` 在 `base/turbo` 之间均匀。
- `drsr` 在 `base/turbo` 之间均匀。
- 每个 `sigma` 内部均匀。
- 每个 `seed` 内部均匀。
- 数据集顺序上交错均匀，避免某个 family 或难度段集中落到同一个桶。

推荐使用调度器参数：

```bash
--llm-model-assignment stable-half
--llm-model-buckets base,turbo
--llm-model-bucket-limits base:100,turbo:100
--prioritize-llm
```

`stable-half` 会按如下任务身份做稳定哈希：

```text
tool | seed | global_index | dataset_dir
```

因此同一个 `tool x dataset x seed` 会稳定落到同一个桶；不同 seed 会自然交错。
在 Core-50 的 3 seeds 设置下，单个 `tool x sigma` 的 150 条 LLM 任务会近似分成：

```text
base  ~= 75
turbo ~= 75
```

两个 LLM 算法合计，每个 `sigma`：

```text
llmsr: 150 runs -> base/turbo 各约 75
drsr:  150 runs -> base/turbo 各约 75
LLM total: 300 runs -> base/turbo 各约 150
```

三个噪声水平合计：

```text
LLM total: 900 runs -> base/turbo 各约 450
```

注意：

- 汇总结果时不要把 `base/turbo` 当作两个模型。
- `result.json` 中可保留 `params.llm_model_assignment` 或参数文件名用于审计。
- 如果后续发现某个桶有 API 限流，只需要调低对应 `--llm-model-bucket-limits`，不需要改任务定义。

## 远端原始结果

远端原始运行目录仍建议使用：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/<batch_name>/
```

本地最终汇总结果统一回收到本目录的：

```text
results/
```
