# runtime_params_3h_effective

3 小时预算参数集，用于 Core-50 小规模长预算预检。

## 用法

```bash
--params-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_3h_effective
```

## 预算策略

- 统一 `timeout_in_seconds=10800`。
- `pyoperon.max_time=10800`，`ragsr.time_limit=10795`。
- 其它非时间迭代/采样上限沿用 24h 参数集的大预算，确保本次 3h smoke 主要由 wall-clock 收口。
- `dso/udsr/imcts/tpsr/e2esr` 依赖 wrapper budget loop，内部自然结束会重启 chunk 直到 3h 预算耗尽。

见 `runtime_budget_plan.csv` 获取逐算法预算清单。
