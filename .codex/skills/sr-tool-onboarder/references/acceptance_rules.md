# 验收规则

## 最低验收标准

接入验收使用实际源码与实际依赖。报告位于每次调用的 `--output-root`，未完成报告的 `passed` 保持 false。

1. 结构检查
   - manifest 可解析
   - `wrapper.py` 存在
   - 源码 commit 和全部必要文件 SHA256 符合 manifest
   - `toolbox_config.json` 与 `envs_config.json` 中有对应注册

2. 环境与导入检查
   - `scientific_intelligent_modelling.algorithms.<tool>_wrapper.wrapper` 可导入
   - manifest 声明的包装器类真实存在
   - 独立环境安装完成且 `pip check` 通过

3. 原生 API 与 benchmark 检查：`accept --stage smoke`
   - 能实例化 `SymbolicRegressor("<tool>")`
   - 能跑 `fit`
   - 能拿到非空最优方程
   - 若支持预测，能拿到正确形状的 `predict` 结果
   - 原生预测与 canonical 数值回放一致，完整精度常数与零基变量保留
   - 序列化后的预测逐值一致
   - 标准 runner 在真实数据集上生成公式、canonical artifact 及有限 ID/OOD 指标
   - 输入维度错误在 `fit` 入口拒绝

4. 预算与分钟检查：`accept --stage budget`
   - 运行预算至少 60 秒，时间上限来自 manifest
   - 预算结束后有可恢复的原生候选，终止时间在明确允许范围内
   - 分钟记录含原生训练目标、候选表达式和不可变来源文件 SHA256
   - 当前分钟没有新候选时保留 carry-forward 来源，缺失候选明确记录

5. 全量接口检查：`accept --stage core50`
   - 冻结名单恰好 50 个唯一任务，文件 SHA256 一致
   - 每个任务的 metadata、train、valid、ID、OOD 输入均已预先验证并记录 SHA256
   - 每个 seed 的全部 50 个任务有实际 `result.json`，公式和数值指标可读取
   - 此阶段使用 manifest 中明确保存的接入验收参数，结果用于判定接口可用

6. 正式六轴数据
   - 逐分钟 SYM/MIN 评分与证据、EFF 累计、跨 seed STAB 证据齐全后才允许 `formal_six_axis_ready=true`
   - 尚未取得的裁决保存到 `unresolved_axes`；所有原生候选和来源保留
   - 付费 API 执行前报告请求量、预计成本和重试上限

复杂接入由独立智能体读取技能、manifest、真实源码与验收报告复核。输入为实际任务描述，评估者自行选择必要检查并给出通过或失败及证据路径。

## 推荐扩展检查

- `python3 -m py_compile` 检查新脚本语法
- 若使用新环境，验证 `post_install_commands` 合理
- 若依赖外部仓库，确认 vendor 路径真实存在
- 若工具需要在线服务，在线 check 必须和离线 check 分开

## 不通过时优先排查

1. `tool_name` 与包装器目录不一致
2. `toolbox_config.json` 中的 regressor 类名不一致
3. `envs_config.json` 的 env 名与 toolbox 注册不一致
4. 输入形状没有适配
5. 底层模型无法序列化，导致 fit 后 predict 失效
6. 最优方程提取逻辑写在了 check 脚本里，而不是包装器里
