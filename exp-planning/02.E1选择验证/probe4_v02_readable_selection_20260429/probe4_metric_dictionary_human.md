# Probe-4 大表字段解释

这份字段解释故意不用公式堆砌，只说明每个指标在实际决策里有什么用。

## 数值健康

- `finite_train`：训练集 NMSE 是不是一个非负有限数。它不是“训练成功”的严格证明，只说明 train_nmse 能参与计算。
- `finite_id`：ID test NMSE 是否可计算。
- `finite_ood`：OOD test NMSE 是否可计算。
- `finite_id_ood`：ID 和 OOD 两个最终评价 split 是否都可计算。
- `finite_train_id_ood`：train、ID、OOD 三类是否都可计算。Probe-4 现在不要求 valid。
- `prompt_semantics_mode`：该行是否使用物理语义 prompt。`physics_semantic_hidden_mapping` 表示 `llmsr/drsr` 新语义批次；`none_or_original_e1` 表示原 E1 口径。
- `llm_model_assignment`：语义批次中实际使用的 LLM 模型分配。只用于审计，不作为 Probe-4 评分项。
- `semantic_background_preview`：语义 prompt 背景摘要预览。只用于审计，不作为 Probe-4 评分项。
- `combined_log_id_ood_nmse`：把 ID 和 OOD 的误差压到 log 尺度后取平均。它用于避免极端大数直接支配表格。
- `gap_log_ood_minus_id`：OOD 比 ID 坏多少。越大说明外推退化越明显。
- `delta_id_minus_train`：ID 比 train 坏多少。它反映从训练拟合到同分布测试是否稳定。
- `delta_ood_minus_id`：OOD 比 ID 坏多少。它反映分布外泛化是否崩。
- `id_ood_explosion_gt_100`：ID 或 OOD NMSE 是否超过 100。超过就算实际评价风险很高。
- `train_id_ood_explosion_gt_100`：train、ID、OOD 任一超过 100。

## 表达式结构

- `expression_artifact_available`：是否有可解析的表达式归档。没有归档就不能计算表达式复杂度。
- `artifact_valid`：工具集 canonicalizer 是否认为表达式 artifact 合法。
- `sympy_parse_ok`：表达式是否能被 sympy 解析。
- `raw_equation_char_count`：原始表达式字符长度。过长通常解释性差。
- `normalized_expression_char_count`：归一化表达式字符长度。
- `raw_equation_token_count`：原始表达式大概有多少 token。
- `normalized_expression_token_count`：归一化后表达式大概有多少 token。
- `ast_node_count`：表达式树节点数。比字符长度更接近结构复杂度。
- `tree_depth`：表达式树有多深。很深的表达式通常更难解释，也更容易数值不稳定。
- `expression_complexity_bucket`：把表达式粗分成 trivial/simple/medium/complex/very_complex，方便快速筛查。
- `used_variable_count`：表达式用了多少个输入变量。
- `variable_coverage_ratio`：表达式用到的变量数占数据集特征数的比例。
- `uses_all_features`：是否使用了全部输入变量。
- `uses_no_features_constant`：是否基本是常数表达式。
- `operator_set`：表达式里出现过哪些算子。
- `operator_category_count`：算子类别覆盖数。类别多不一定好，但能表示表达式结构更丰富。
- `uses_trig_ops`：是否用 sin/cos/tan 等周期函数。
- `uses_exp_log_ops`：是否用 exp/log。
- `uses_product_power_ops`：是否用乘除幂。
- `uses_root_abs_inv_ops`：是否用 sqrt/abs/inv。
- `uses_complex_symbols`：是否出现复数相关符号。这通常需要额外审计。

## 工程健康

- `raw_result_status`：原始 result 的状态，例如 ok 或 timed_out。它只记录运行器看到的原始状态，不直接等于 Probe-4 成功/失败。
- `budget_exhausted`：是否跑满预算。`1` 通常对应原始 `timed_out`。
- `probe4_success`：Probe-4 选择口径下是否成功。只要 train、ID、OOD 指标都能落盘并可计算，就算成功；即使原始状态是 timed_out 也算成功。
- `normalized_status_for_probe4`：把原始状态翻译成适合本实验的状态。例如 `success_budget_exhausted` 表示跑满预算但结果可用。
- `wall_time_seconds`：这条 run 大概用了多久。
- `expression_health_label`：把数值健康和 artifact 健康合成的人话标签，例如 `metric_ok_artifact_ok` 或 `finite_but_exploded`。

## 为什么这些对选 Probe-4 有用

Probe 不是只要误差低。一个好 probe 应该稳定、有不同失败模式、能产出可解释表达式，并且不会靠大量爆炸值制造虚假的区分度。这个大表就是为了把这些因素都显式摆出来。
