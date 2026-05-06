# Core-50 SYM-F formal metrics

- Created at: `2026-05-05T00:34:18`
- Runs: `3000`
- Datasets: `50`
- Algorithms: `12`
- Valid-for-symbolic rate: `0.9547`
- Prediction parse rate: `0.9947`
- Exact equivalence rate: `0.1220`

## Scoring Protocol

- `equiv_final = cas_equiv or numeric_equiv` 
- `numeric_equiv` Threshold `NMSE <= 1e-10` note `max_rel <= 1e-08` note `max_abs <= 1e-10` 
- `numeric_equiv_reason` records the concrete reason why each numeric-equivalence check passed or was skipped for auditing 
- Equivalent formulas `sym_f_formal = 1.0` 
- Non-equivalent but parseable formulas `sym_f_formal = 0.3 * tree_similarity + 0.2 * ((var_f1 + op_f1) / 2)` 
- invalid / metric-incomplete / unparsable runs are scored as `0` 
- `llmsr/drsr` prefer `result.json` note `instantiated_expression` avoid using `c0/c1` skeleton for numeric equivalence 

## Algorithm Summary

| algorithm   |   datasets |   SYM_F_formal |   exact_equiv_rate |   cas_equiv_rate |   numeric_equiv_rate |   pred_parse_rate |   mean_tree_similarity |   mean_var_f1 |   mean_op_f1 |
|:------------|-----------:|---------------:|-------------------:|-----------------:|---------------------:|------------------:|-----------------------:|--------------:|-------------:|
| pysr        |         50 |        43.6170 |             0.3400 |           0.1360 |               0.3400 |            1.0000 |                 0.2418 |        0.6830 |       0.7396 |
| imcts       |         50 |        41.5033 |             0.3080 |           0.0160 |               0.3080 |            1.0000 |                 0.1848 |        0.7130 |       0.7907 |
| udsr        |         50 |        36.8341 |             0.2400 |           0.0560 |               0.2400 |            1.0000 |                 0.1292 |        0.8521 |       0.7672 |
| dso         |         50 |        31.2551 |             0.1800 |           0.1520 |               0.1800 |            1.0000 |                 0.1829 |        0.7959 |       0.6938 |
| llmsr       |         50 |        27.2955 |             0.1280 |           0.0360 |               0.1280 |            1.0000 |                 0.1021 |        0.8299 |       0.7562 |
| drsr        |         50 |        26.8379 |             0.1200 |           0.0080 |               0.1200 |            0.9920 |                 0.0976 |        0.8177 |       0.7742 |
| gplearn     |         50 |        20.0162 |             0.0800 |           0.0600 |               0.0800 |            0.9800 |                 0.0851 |        0.6968 |       0.5272 |
| qlattice    |         50 |        17.8146 |             0.0480 |           0.0000 |               0.0480 |            0.9800 |                 0.0294 |        0.6595 |       0.6408 |
| ragsr       |         50 |        15.7203 |             0.0160 |           0.0000 |               0.0160 |            1.0000 |                 0.0000 |        0.9079 |       0.5792 |
| tpsr        |         50 |        15.2404 |             0.0040 |           0.0000 |               0.0040 |            1.0000 |                 0.0355 |        0.7356 |       0.6552 |
| e2esr       |         50 |        12.4395 |             0.0000 |           0.0000 |               0.0000 |            1.0000 |                 0.0008 |        0.8480 |       0.6812 |
| pyoperon    |         50 |        11.3046 |             0.0000 |           0.0000 |               0.0000 |            0.9840 |                 0.0256 |        0.5143 |       0.5795 |
