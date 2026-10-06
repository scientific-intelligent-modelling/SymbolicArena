# 接入接口要求

符号回归工具接入当前仓库时，需要同时提供包装器、注册、验收与 manifest。

## 1. 包装层

必须存在：

- `scientific_intelligent_modelling/algorithms/<tool>_wrapper/wrapper.py`

建议同时存在：

- `scientific_intelligent_modelling/algorithms/<tool>_wrapper/__init__.py`
- 内置第三方源码目录，例如：
  - `scientific_intelligent_modelling/algorithms/<tool>_wrapper/<vendor_repo>`

当前自动接入模块支持外部源码 checkout。manifest 保存作者仓库完整 commit、必要文件 SHA256 和运行时源码环境变量。源码有明确分发许可证时，可按项目约定保留许可证和来源；源码未提供许可证时采用外部引用。

包装器必须满足：

- 实现 `fit(X, y)`
- 实现 `predict(X)`
- 实现 `get_optimal_equation()`
- 实现 `get_total_equations()`

如果底层模型不能可靠 pickle：

- 改写 `serialize()/deserialize()`
- 明确保存最小可恢复状态

## 2. 注册层

必须更新：

- `scientific_intelligent_modelling/config/toolbox_config.json`
- `scientific_intelligent_modelling/config/envs_config.json`

其中：

- `toolbox_config.json` 负责 `tool_name -> env + regressor class`
- `envs_config.json` 负责 conda 环境定义与安装后命令

## 3. 验收层

必须存在：

- `check/check_<tool>.py`

运行时要求：

- 能构造 `SymbolicRegressor("<tool>")`
- 能跑一次离线 `fit`
- 能拿到最优方程
- 如果工具支持预测，能跑一次 `predict`
- 原生预测、canonical artifact 数值回放以及序列化恢复预测一致。
- 预算结束时恢复预算内原生最佳候选；分钟记录包含原始表达式、训练目标、时间、来源及 SHA256。

## 4. Manifest 层

必须存在：

- `tools/sr_onboarder/manifests/<tool>.json`

manifest 是接入声明，不是用户文档。它至少描述：

- `schema`：`symbolicarena-integration-v1`
- `tool_name`、`wrapper_class`、`environment`：须与两个注册文件一致。
- `source`：作者仓库 URL、40 位 commit、外部源码环境变量、文件 SHA256、许可证状态及修改文件路径。
- `parameters`：默认搜索参数与独立的 `smoke`、`budget`、`core50` 验收参数。
- `core50`：冻结名单路径、SHA256、50 个任务及明确 seed。

包装器写入 `symbolicarena-native-best-v1` 记录。注册项中的 `normalizer` 使用 `module:function` 形式，`progress_file` 和 `progress_history_file` 使用目录内文件名。runner 根据这些字段接入公式回放、预算恢复和分钟记录。

## 推荐目录结构

```text
.codex/skills/sr-tool-onboarder/
tools/sr_onboarder/manifests/
tools/sr_onboarder/patches/
scientific_intelligent_modelling/onboarding/
scientific_intelligent_modelling/algorithms/<tool>_wrapper/
check/check_<tool>.py
```
