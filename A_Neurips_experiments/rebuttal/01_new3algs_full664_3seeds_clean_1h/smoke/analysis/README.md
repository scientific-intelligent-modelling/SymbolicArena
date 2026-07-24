# NeurIPS rebuttal Full664 分析

- 状态：`FINAL_READY`
- `Valid rate`：有表达式、canonical artifact 未明确无效，且 ID/OOD NMSE 完整。
- `Metric complete rate`：ID/OOD NMSE 均为有限非负数。
- `Mean ID/OOD log NMSE`：`log10(max(NMSE, 1e-12))` 截断到 `[-12, 12]`；
  对各自缺失值按 `+12` 计入均值。
- 排名：按 penalized mean OOD log NMSE 升序。
- Stage3 对比：本次未启用 Stage3 四算法合并。

`INCOMPLETE_PREVIEW` 只用于运行中巡检，不得作为论文最终结果引用。
