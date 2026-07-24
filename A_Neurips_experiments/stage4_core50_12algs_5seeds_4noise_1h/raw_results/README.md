# Raw results boundary

原冻结包在这里保存的是一个软链：

```text
core50_12alg_5seed_all_20260502-065700
  -> ../../../../../experiments/core50_12alg_5seed_all_20260502-065700
```

该目标不属于当前整理包，复制后会成为失效软链，因此没有归档软链本体。

本阶段可直接使用的 run 级结果位于：

```text
../run_archive/results/
```

原始任务路径字符串仍保留在 `../run_archive/manifest.csv` 的
`raw_result_json` 列中。
