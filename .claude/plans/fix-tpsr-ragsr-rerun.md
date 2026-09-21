# 修复 TPSR + RAG-SR wrapper 并重跑三算法

## 诊断结论

| 算法 | 根因 | 可修复 |
|------|------|--------|
| TPSR | Wrapper bug: 缺少 StandardScaler + rescale_function | ✅ 修两处 |
| RAG-SR | Wrapper bug: model() 输出的公式期待 MinMax-normalized X，但 predict/snapshot 用 raw X | ✅ 修嵌入 x_scaler |
| E2ESR | 算法固有限制：单次前向推理，表达力不足 | ❌ 不修（非 bug） |

## 修复计划

### 1. TPSR — 两个 critical fix

**文件**: `scientific_intelligent_modelling/algorithms/tpsr_wrapper/wrapper.py`

**Fix 1**: 在构造 samples dict 前，用 `utils_wrapper.StandardScaler()` 标准化 `reward_X`
- 位置: ~line 1066-1076
- 添加: `scaler = utils_wrapper.StandardScaler(); scaled_X = scaler.fit_transform(reward_X); scale_params = scaler.get_params()`
- 将 `scaled_X` 传入 `samples['x_to_fit']` 和 `samples['x_to_pred']`
- 保存 `self._scaler` 和 `self._scale_params`

**Fix 2**: 在 `_fit_e2e()` 提取 refined tree 后，调用 `scaler.rescale_function()` 反变换
- 位置: ~line 694-710
- 对 `refine_for_sample` 返回的 tree，调用 `self._scaler.rescale_function(equation_env, tree, *self._scale_params)`
- 然后再提取 infix

### 2. RAG-SR — 在表达式中嵌入 x_scaler 变换

**文件**: `scientific_intelligent_modelling/algorithms/ragsr_wrapper/wrapper.py`

**Fix**: 在 `_extract_model_expression()` 中，获取 `self.model.x_scaler` 参数，将公式中的 `ARGi` 替换为 `((ARGi - data_min_i) / (data_max_i - data_min_i))`
- 位置: ~line 396-403
- 同时在 `serialize()` 中保存 x_scaler 参数，在 `deserialize()` 中恢复

### 3. E2ESR — 不修改

算法本身限制，不是 wrapper bug。保持现状。

### 4. 部署重跑

- 新建批次，只包含 tpsr + ragsr（e2esr 不重跑）
- 复用同样的 seeds(520,521,522) × noise(0,0.01,0.05) × 50 datasets = 450×2 = 900 tasks
- 并发 70/host
- 同步代码到 8 台机器
- 直接启动 full dispatch

## 不做

- 不修 E2ESR
- 不影响正在跑的 formal3h 主批次
- 不删除历史结果
