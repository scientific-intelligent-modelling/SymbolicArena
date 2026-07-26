# Core-50 SYM-F formal metrics

- Created at: `2026-07-26T12:13:39`
- Runs: `1992`
- Datasets: `664`
- Algorithms: `1`
- Valid-for-symbolic rate: `0.9689`
- Prediction parse rate: `1.0000`
- Exact equivalence rate: `0.1451`

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
| dso         |        664 |        28.8911 |             0.1451 |           0.0868 |               0.1451 |            1.0000 |                 0.1273 |        0.8521 |       0.7153 |
