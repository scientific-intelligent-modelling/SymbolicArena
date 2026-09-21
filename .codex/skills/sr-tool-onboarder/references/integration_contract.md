# 接入契约

新的符号回归工具接入当前仓库时，最少要落到下面 4 层。

## 1. 包装层

必须存在：

- `scientific_intelligent_modelling/algorithms/<tool>_wrapper/wrapper.py`

建议同时存在：

- `scientific_intelligent_modelling/algorithms/<tool>_wrapper/__init__.py`
- 内置第三方源码目录，例如：
  - `scientific_intelligent_modelling/algorithms/<tool>_wrapper/<vendor_repo>`

第三方算法源码必须由主仓库直接跟踪，不得保留嵌套 `.git` 或注册为算法子模块。
接入时应保留上游许可证，并在 `algorithms/VENDORED_SOURCES.md` 中记录来源与提交号。

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

最小要求：

- 能构造 `SymbolicRegressor("<tool>")`
- 能跑一次离线 `fit`
- 能拿到最优方程
- 如果工具支持预测，能跑一次 `predict`

## 4. Manifest 层

必须存在：

- `tools/sr_onboarder/manifests/<tool>.json`

manifest 是接入声明，不是用户文档。它至少描述：

- 工具名
- 包装模式
- 外部仓库路径
- 环境名与依赖
- 输入形状
- 是否支持 `predict`
- 包装器类名
- smoke check 默认参数

## 推荐目录结构

```text
.codex/skills/sr-tool-onboarder/
tools/sr_onboarder/
  manifests/
  scripts/
  templates/
scientific_intelligent_modelling/algorithms/<tool>_wrapper/
check/check_<tool>.py
```
