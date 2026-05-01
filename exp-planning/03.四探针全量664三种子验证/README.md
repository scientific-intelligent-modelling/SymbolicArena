# 03. 四探针全量 664 三种子验证

本目录用于整理 `4 algorithms × 664 datasets × 3 seeds` 这轮全量验证实验的规划、分发、过程审计与结果汇总材料。

当前 probe methods：

- `dso`
- `imcts`
- `pyoperon`
- `udsr`

当前 seeds：

- `520`
- `521`
- `522`

## 后处理

后处理脚本：

```bash
python check/postprocess_probe4_full664_metrics.py
```

默认输入为 `02.E1选择验证/generated/probe4_full664_v1/current_digest_latest.txt` 指向的 `probe4_current_run_level.csv`。

默认输出到：

```text
exp-planning/03.四探针全量664三种子验证/generated/postprocess_<timestamp>/
```

指标定义见：

```text
postprocess_metric_definition.md
```
