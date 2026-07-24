# NeurIPS Rebuttal 实验

这里保存 NeurIPS 投稿之后新增的 rebuttal 实验。新增结果作为独立证据链归档，
不回写或覆盖原 Stage 1--4 的历史结果。

## 1. 新增三算法 Full-664

目录：`01_new3algs_full664_3seeds_clean_1h/`

```text
algorithms = fepysr, jaxsr, symbolfit
datasets = Stage 3 的 full-664
seeds = 520, 521, 522
condition = clean only
budget = 1h per task
tasks = 3 × 664 × 3 = 5976
```

参数从 AAAI Stage 4 实际运行的三份 clean 3h 参数继承，仅将任务超时预算从
`10800` 秒改为 `3600` 秒。正式实验前必须通过：

1. 本地参数、清单、任务数与 dry-run 校验。
2. `iaaccn22~29` 八台机器远端 preflight。
3. `2 datasets × 3 algorithms × 3 seeds = 18` 个 600 秒 smoke。
4. smoke collect、harvest 和 audit 全部通过。

实验运行和审计细节见该实验目录的 `README.md`。
