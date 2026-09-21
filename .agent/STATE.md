# 工作区状态

## 当前任务

- 编号：ENV-002；任务：逐机器、逐算法执行两个 180 秒训练测试；状态：15 台可连接机器已完成，54 不可达。
- 当前采用方案：22~29、48~53、55，每台 15 算法，使用 g0275（feynman-i.14.4，含干扰变量）和 g0596（strogatz_glider1），clean、seed=520。训练预算 180 秒，保留预算内最佳候选；结果恢复和评估另计时间。LLMSR/DRSR 使用 DeepInfra Meta-Llama-3.1-8B-Instruct-Turbo，3 轮各 4 个样本。
- 有效结果：[逐机器结果](work/ENV-002/results_by_machine.csv)、[逐测试明细](work/ENV-002/results.csv)；脚本、配置、输入及 SHA256 见 [ENV-002](work/ENV-002/) 内 `run_checks.py`、`params.json`、`inputs_manifest.json`。远端原始输出位于 `<根目录>/sim-runtime/checks/ENV-002/results/`。
- 已确认：450/450 测试结束，无遗留测试会话；449 条产生有限 ID R2，25 上 E2ESR/g0275 预算内无有效公式。产生指标不代表预测效果或评估逻辑通过。
- 未解决问题：49、51、52、53、55 的 LLMSR/DRSR 无成功 API 日志，已见 DeepInfra DNS 解析失败；已有基线指标不能作为 LLM 调用成功证据。PySR/PyOperon 表达式未使用首列时，normalizers.py 会错误移动变量索引；已复算 PySR 示例，原公式 ID R2=0.9999999999999963，报告为 -8.822646727794563e37，证据 `metric_mapping_check.json`。本轮尚未修改算法代码、系统 DNS 或原始结果。54 仍不可达，30 个测试未执行。
- 下一步：处理 DeepInfra DNS 和变量索引问题；54 恢复后补充部署和同口径测试。

## 环境入口：已确认

- CPU 项目位于 `~/workplace/scientific-intelligent-modelling` 和 `~/projects/scientific-intelligent-modelling`，两处已同步当前代码。
- GPU 48、50~53、55 使用 `/data1/zhangziwen`，49 使用 `/data3/zhangziwen`；代码为 `<根目录>/sim-runtime/code`，激活入口为 `<根目录>/sim-runtime/runtime.env`。复用环境仍存在部分第三方包版本差异。
- ENV-001 的文件校验与导入结果位于 [ENV-001](work/ENV-001/)，默认不重复读取。

## 有效入口：已确认

- 全局规则：[/home/family/.codex/AGENTS.md](/home/family/.codex/AGENTS.md)。
- 项目规则：[AGENTS.md](../AGENTS.md)；实验说明：[.codex/AGENTS.md](../.codex/AGENTS.md)。
- 代码与测试：[scientific_intelligent_modelling/](../scientific_intelligent_modelling/)、[tests/](../tests/)；项目入口：[cli.py](../scientific_intelligent_modelling/cli.py)。
- 配置：[toolbox_config.json](../scientific_intelligent_modelling/config/toolbox_config.json)、[envs_config.json](../scientific_intelligent_modelling/config/envs_config.json)。
- 数据目录：[sim-datasets-data/](../sim-datasets-data/)、[sim-datasets-py/](../sim-datasets-py/)；独立仓库保持原位置。
- 目录分类：`AAAI_experiments/`、`A_ICLR_experiments/`、`A_Neurips_experiments/` 为现有实验目录，按具体任务读取。
- ICLR stage4 目录：[stage4_core80_15algs_3seeds_3h/](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/)。

## 候选文件与问题：待验证

- CPU 节点的 `workplace` 与 `projects` 两处项目路径均被已有环境引用，本次已同步两处代码。GPU 使用上方独立目录。
- `.codex/AGENTS.md`、`.gitignore` 的已有未提交修改保持原样。

## 替代与归档

- ENV-001 当前采用最终 `gpu_verify_*.json`；初次安装和 Julia 初始化记录保留在源机器的同任务目录，不代表当前验证状态。
- WS-001 已完成；未产生独立过程记录，无需移动文件。
- WS-002 已完成；未产生独立过程记录，无需移动文件。
- SR-CLOUD-30 已关闭；本地诊断文件和关联测试已按用户确认删除。
- CL-001 已完成；`.agent/work/CL-001/` 已归档，默认不读取。
- DIR-001 已完成；未产生过程记录。
