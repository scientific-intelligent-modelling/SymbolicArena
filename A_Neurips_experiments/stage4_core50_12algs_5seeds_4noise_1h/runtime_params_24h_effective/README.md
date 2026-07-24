# runtime_params_24h_effective

24 小时预算参数集。用于 Core-50 / noise robustness 等长预算实验。

## 用法

调度脚本中把参数根目录指向本目录：

```bash
--params-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_24h_effective
```

或者使用开关脚本切换 `runtime_params_active`：

```bash
python check/select_core50_runtime_budget.py --budget 24h
```

然后调度时使用：

```bash
--params-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_active
```

## 预算策略

- 统一 `timeout_in_seconds=86400`。
- 对有独立时间参数的算法同步设置：`pyoperon.max_time=86400`，`ragsr.time_limit=86395`。
- 对非时间上限统一放大，避免先于 24h 自然耗尽。
- `dso/udsr/imcts/tpsr/e2esr` 依赖 wrapper 的预算循环；如果单个内部 chunk 自然完成，会自动重启 chunk 直到 wall-clock 预算耗尽。

见 `runtime_budget_plan.csv` 获取逐算法预算清单。
