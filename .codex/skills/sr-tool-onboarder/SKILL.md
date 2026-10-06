---
name: sr-tool-onboarder
description: 让智能体将指定符号回归源码接入 SymbolicArena，并通过真实依赖完成验收。适用于固定源码版本、实现 wrapper 和原生公式导出、注册独立环境、验证预算与分钟记录，以及在冻结 Core50 上运行接入验收。
---

# SR Tool Onboarder

输入算法论文或源码地址，交付可运行的包装器、版本固定的 manifest 和实际验收报告。执行模块位于 `scientific_intelligent_modelling/onboarding/`，入口为 `sim-onboard`。

## 何时使用

- 用户要新增一个符号回归算法到当前框架。
- 用户要把外部仓库整理成 `scientific_intelligent_modelling/algorithms/<tool>_wrapper` 下的子工具。
- 用户要求智能体完成算法接入、环境配置、模型恢复和验收。

## 快速流程

1. 阅读接入接口要求：
   - `references/integration_contract.md`
2. 按实际训练入口选择包装模式：
   - Python API / 可编辑安装：读 `references/wrapper_patterns.md`
   - CLI-only：也读 `references/wrapper_patterns.md`
3. 固定作者仓库的完整 commit，调用 `sim-onboard prepare --source-url <url> --revision <commit> --output-root .agent/work/<task>/source`，阅读源码、依赖与输出的 `source-review.json`。准备阶段生成 `agent_task.md`，供智能体直接执行接入任务。
4. 使用文件编辑工具实现包装器，并新建 `tools/sr_onboarder/manifests/<tool>.json`。该文件须通过 `onboarding/manifest.py` 的 JSON Schema；`dgp.json` 是已运行的完整实例。注册 `normalizer`、`progress_file`、`progress_history_file` 后，runner 自动读取该算法的原生候选记录。
5. 通过 `sim-onboard validate --manifest <path> --output-root <work>/structure` 检查源码 SHA256、接口、注册与 Core50 版本。
6. 通过 `sim-onboard setup --manifest <path> --output-root <work>/environment` 创建独立环境并检查依赖。源码版本及修改内容由智能体使用 `apply_patch` 编辑；执行模块写入版本、任务与验收数据。
7. 按 [验收规则](references/acceptance_rules.md) 运行 `accept` 的 `smoke`、`budget`、`core50` 三个阶段。失败时查看实际日志和原生候选，修复后重新执行受影响阶段。
8. 复杂接入完成后，使用独立智能体读取当前技能、源码及 manifest 执行验收。评估者自行判定结果，并提供实际报告路径。远程运行使用独立 checkout、明确数据根目录和保存的输入哈希。

## 强约束

- 新工具优先使用小写 `tool_name`，避免继续扩散大小写混用的工具 ID。
- 每个新工具优先独立 `env`，除非你能明确证明可安全复用现有环境。
- 包装器必须实现当前 `BaseWrapper` 接口，并能被 `subprocess_runner.py` 动态导入。
- 离线 smoke check 必须先通过，再做在线或远程批量实验。
- API key 仅在运行时注入。付费裁决执行前报告请求量、成本和重试上限。
- 保留算法原生训练目标选中的完整精度表达式、原生算子语义和变量编号。恢复、导出及分钟快照使用相同候选。
- 源码未声明许可证时，使用固定版本的外部 checkout；manifest 记录 `license: not-provided`。原生修改文件的 SHA256 与可复现修改内容一并保留。
- 六轴正式结果要求逐分钟评分与来源证据齐全。未完成的评分记录在 `unresolved_axes`，`formal_six_axis_ready` 保持 false。

## 需要重点看的文件

- 包装器基类：`scientific_intelligent_modelling/algorithms/base_wrapper.py`
- 子进程动态加载：`scientific_intelligent_modelling/srkit/subprocess_runner.py`
- 工具注册：`scientific_intelligent_modelling/config/toolbox_config.json`
- 环境注册：`scientific_intelligent_modelling/config/envs_config.json`
- 现有验收脚本目录：`check/`

## 交付证据

- 源码 commit、必要文件 SHA256、环境依赖清单与 manifest 版本。
- `native_api.json` 中的原生、导出回放及恢复预测。
- `acceptance.json` 中各阶段判定及真实 `result.json` 路径。
- Core50 的 `input_manifest.json` 与全部任务运行记录。
- 当前分支、正式文件和报告路径记录到 `.agent/STATE.md`。

## 参考资料

- 接入接口要求：`references/integration_contract.md`
- 包装模式：`references/wrapper_patterns.md`
- 验收规则：`references/acceptance_rules.md`
