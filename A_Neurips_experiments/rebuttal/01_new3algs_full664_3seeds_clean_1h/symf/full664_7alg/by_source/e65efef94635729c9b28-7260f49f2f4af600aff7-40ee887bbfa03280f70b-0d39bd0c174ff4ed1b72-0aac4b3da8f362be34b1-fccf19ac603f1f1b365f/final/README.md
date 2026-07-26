# Core-50 SYM-F formal metrics

- Created at: `2026-07-26T13:11:34`
- Runs: `13944`
- Datasets: `664`
- Algorithms: `7`
- Valid-for-symbolic rate: `0.9748`
- Prediction parse rate: `0.9986`
- Exact equivalence rate: `0.1509`

## 评分口径

- `equiv_final = cas_equiv or numeric_equiv`。
- `numeric_equiv` 阈值：`NMSE <= 1e-10` 或 `max_rel <= 1e-08` 或 `max_abs <= 1e-10`。
- `numeric_equiv_reason` 记录每条数值等价通过或跳过的具体原因，便于审计。
- 等价公式 `sym_f_formal = 1.0`。
- 非等价但可解析公式 `sym_f_formal = 0.3 * tree_similarity + 0.2 * ((var_f1 + op_f1) / 2)`。
- invalid / metric incomplete / unparsable run 记 `0`。
- `llmsr/drsr` 优先使用 `result.json` 中的 `instantiated_expression`，避免用 `c0/c1` skeleton 做数值等价。

## Algorithm Summary

| algorithm   |   datasets |   SYM_F_formal |   exact_equiv_rate |   cas_equiv_rate |   numeric_equiv_rate |   pred_parse_rate |   mean_tree_similarity |   mean_var_f1 |   mean_op_f1 |
|:------------|-----------:|---------------:|-------------------:|-----------------:|---------------------:|------------------:|-----------------------:|--------------:|-------------:|
| udsr        |        664 |        39.0786 |             0.2636 |           0.0597 |               0.2636 |            1.0000 |                 0.0964 |        0.9128 |       0.7715 |
| imcts       |        664 |        38.5302 |             0.2550 |           0.0201 |               0.2550 |            1.0000 |                 0.1387 |        0.8458 |       0.7721 |
| dso         |        664 |        28.8911 |             0.1451 |           0.0868 |               0.1451 |            1.0000 |                 0.1273 |        0.8521 |       0.7153 |
| symbolfit   |        664 |        27.2220 |             0.1386 |           0.0000 |               0.1386 |            1.0000 |                 0.0072 |        0.9164 |       0.6283 |
| fepysr      |        664 |        27.1252 |             0.1466 |           0.0527 |               0.1466 |            1.0000 |                 0.1332 |        0.6496 |       0.5683 |
| jaxsr       |        664 |        25.6103 |             0.1039 |           0.0000 |               0.1039 |            1.0000 |                 0.0828 |        0.7793 |       0.6733 |
| pyoperon    |        664 |        13.6670 |             0.0035 |           0.0000 |               0.0035 |            0.9900 |                 0.0455 |        0.5973 |       0.6185 |
