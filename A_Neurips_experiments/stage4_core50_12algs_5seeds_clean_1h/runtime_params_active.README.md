# runtime_params_active

这是运行时预算开关的目标路径。

请不要手工编辑本目录；使用：

```bash
python check/select_core50_runtime_budget.py --budget 1h
python check/select_core50_runtime_budget.py --budget 3h
python check/select_core50_runtime_budget.py --budget 24h
```

调度脚本可统一使用：

```bash
--params-root frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_active
```
