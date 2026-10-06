# 包装模式

## 模式一：Python API

适用场景：

- 外部仓库能通过 `import` 调到类或函数
- 训练接口是 `fit(X, y)` 或接近该形式
- 预测接口可直接调用或可从训练结果恢复

建议做法：

1. 在 `wrapper.py` 里延迟导入外部入口。
2. 先剥离框架元参数：
   - `exp_name`
   - `exp_path`
   - `problem_name`
   - `seed`
3. 适配输入形状：
   - 本框架默认 `X.shape == (n_samples, n_features)`
   - 有些仓库要求 `(n_features, n_samples)`，此时在包装层转置
4. 训练完成后统一抽取：
   - 最优方程
   - 候选方程列表
5. 若底层模型可 pickle，优先沿用基类序列化。
6. 若底层模型不可 pickle，再改成自定义 JSON 状态。

## 模式二：CLI-only

适用场景：

- 外部仓库只提供命令行入口
- 训练或预测依赖文件输入输出

建议做法：

1. 在包装层中准备临时输入文件。
2. 用 `subprocess.run()` 调工具 CLI。
3. 约定稳定的输出目录和结果文件。
4. 训练后把必要状态转为本框架可序列化的最小结果：
   - 最优方程文本
   - 候选方程列表
   - 如可行，再保存预测所需的中间产物路径或表达式

注意：

- CLI-only 工具不能直接照搬 Python API 脚手架。
- 智能体须根据真实 CLI 输出完成训练、状态保存和预测入口，并通过同一套运行时验收。

## 参数处理建议

- 若工具参数很脆弱，显式维护 allowlist。
- 若工具参数较稳，可先透传，再逐步收紧。
- 对外接口优先贴近当前框架已有命名，而不是把底层仓库原始参数名直接暴露出去。

## 表达式提取建议

优先尝试：

- `get_optimal_equation()`
- `get_equation()`
- `get_model_string()`
- `best_`
- `program_`
- `symbolic_model`

候选方程优先尝试：

- `get_total_equations()`
- `get_all_equations()`
- `hall_of_fame_`
- `models_`

如果这些都没有：

- 就在包装器里显式实现自己的提取逻辑，不要把“如何找表达式”留给上层猜。
