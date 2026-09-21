# 工作区状态

## 当前任务

- 编号：ENV-003；任务：每台机器3数据集、15算法、每实验180秒快速测试；状态：15台各45项全部结束，共675项，54不可达。
- 当前方案：feynman-i.14.4（g0275，含干扰变量）、strogatz_glider1（g0596）、BPG3（g0005）；clean、seed520，每台并发45。LLMSR/DRSR 使用 DeepInfra Llama-3.1-8B-Instruct-Turbo，3轮每轮4候选、max_tokens1024，不额外重试。
- 有效文件：[逐实验结果](work/ENV-003/results.csv)、[算法汇总](work/ENV-003/summary.csv)；同目录保留 `run_checks.py`、`run.sh`、`control.py`、`node.py`、`prepare_inputs.py`、`export_results.py`、`params.json`、`inputs_manifest.json` 和 `results_*.json`。远端原始结果为各节点数据根目录的 `sim-runtime/checks/ENV-003/results/`。结果采用预算内原生最佳快照或终态，不按测试分数改选。
- 验证：675个组合完整，无剩余测试会话；673个有有限ID R²，420个ID R²>0.99；90个LLM实验均记录API成功响应，合计677次。单任务训练180秒，包含恢复和评估的最大总耗时219.3秒。
- 未解决问题：25上的E2ESR、SymbolFit在g0275未取得预算内可用指标；FePySR等算法部分数据集效果仍差。未启动额外修复或重跑；54恢复后尚需补45项。

## 共享恢复修复

- 编号：FIX-003；任务：只修复 LLMSR/DRSR 及最初4算法的共享恢复问题；状态：已完成。
- 已确认：LLMSR、DRSR、TPSR、RAG-SR、SymbolFit 存在根据不可用测试指标改选旧公式的问题，E2ESR 原有保护有效。共享恢复现对全部15算法统一保留原生候选，PySR/PyOperon 同一分支同时覆盖；不改变各算法搜索、公式导出或指标定义。
- 有效文件：`scientific_intelligent_modelling/benchmarks/runner.py`、`tests/test_native_budget_selection.py`。本地15算法的候选恢复和分钟快照恢复检查通过，另3个变量索引测试通过；15台同步后各4项测试通过，共60项，23处源码哈希一致，54不可达。证据：[remote_sync.json](work/FIX-003/remote_sync.json)；执行脚本为同目录 `run_sync.py`、`sync_remote.py`。
- 未解决问题与下一步：未调用 LLM、未重跑训练、未改写历史结果；54恢复后补同步。此项保证选模规则正确，不代表评测精度必然提高。

## 已完成检查

- 编号：FIX-002；任务：检查 QLattice、DSO、FePySR、gplearn、iMCTS、JAXSR、uDSR；状态：本轮检查、修复和同步已完成，排除 LLMSR/DRSR。
- 当前采用方案：7算法预算恢复保留原生选中公式，不按测试集有限性改选；gplearn 完整精度导出并保留跨代最佳 program；QLattice 数值导出使用17位有效数字；iMCTS 常数预测按样本数广播；JAXSR 从已验证 model_state 恢复。DSO/uDSR 明确拒绝尚无可靠导出的 protected=True，默认 False 不受影响。
- 已确认：FePySR、DSO、uDSR、iMCTS 稀疏变量回放一致；gplearn/QLattice/JAXSR 用真实依赖复现并验证修复。15台完成6个源码文件同步，138处目标文件哈希一致；每台13个针对性测试通过，共195个。54仍不可达。
- 有效文件：`scientific_intelligent_modelling/benchmarks/runner.py`、对应5个 wrapper；测试为 `tests/test_native_budget_selection.py`、`test_remaining_native_predictions.py`、`test_dso_protected_contract.py`、`test_gplearn_native_export.py`、`test_qlattice_native_export.py`、`test_jaxsr_native_restore.py`，另回归 `test_native_variable_indices.py`。同步与验证证据：[remote_sync.json](work/FIX-002/remote_sync.json)；执行脚本为同目录 `run_sync.py`、`sync_remote.py`，真实复现脚本为 `reproduce_*.py`。
- 未解决问题：未重跑正式实验，未改写旧结果。gplearn/QLattice 旧结果仅含舍入公式时，需原生模型才能无损恢复；不能直接据此声称全部旧记录可复用。已导入的 DSO/uDSR 各363条参数均为 protected=False。
- 下一步：正式实验采用当前代码；旧结果按原生模型是否完整单独判定复用，54恢复后补同步与验证。

## 已完成修复

- 编号：FIX-001；任务：修复 PySR/PyOperon 变量索引并核验旧结果复用；状态：代码修复与数值更正已完成。
- 当前采用方案：PySR 保持零基变量，PyOperon 只转换一次；PyOperon 数值回放使用原生 X 变量公式。历史回放从原始公式重建，保留训练目标选择与来源。
- 已确认：51个数据集的输入与历史冻结数据相符。726次旧训练记录可复用（PySR/PyOperon各363；合计clean306、noise001210、noise005210），无需因索引错误重新训练。708次有有效终态，18次按无有效输出保留。
- 更正范围：291份终态表达式、52747条分钟表达式；全部130680条分钟数值、ID/OOD/EFF及STAB数值部分已核验。300份符号评估和142组三种子结构评估待更新，其他旧裁决仍需绑定核验；未调用付费API，未标记六维正式就绪。
- 有效入口：[current_evaluations.csv](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/current_evaluations.csv)。更正数据在每个seed的 `corrected/`，`current.json` 指定有效文件；原始文件保留。正式脚本、输入核验、复用报告为同目录 `correct_history.py`、`input_audit.json`、`reuse_report.json`。
- 验证：3个针对性测试通过；15台可连接机器同步并全部通过相同测试，54仍不可达。远端记录：[remote_sync.json](work/FIX-001/remote_sync.json)；数据核验：[verification.json](work/FIX-001/verification.json)。
- 下一步：依据当前有效索引更新受影响的 Opus 后处理，再汇总正式六维指标；其余9个已导入算法未在本次做数值复用审核。

## 已有实验

- 编号：IMPORT-001；任务：按新 Core80 交集整理已有实验数据；状态：已完成。
- 当前采用方案：[2、experiments](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/) 使用 `{algorithm}/{dataset}/{condition}/{seed}`；排除 E2ESR、TPSR、RAG-SR、SymbolFit。
- 已确认：11算法、51数据集、3993次运行。旧 Core50 的35个交集任务贡献3465次，stage6新增30中的16个交集任务贡献528次。clean=1683，noise001=1155，noise005=1155。
- 原始导入索引：[manifest.csv](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/manifest.csv)、[manifest.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/manifest.json)；当前评估使用上方 `current_evaluations.csv`。整理脚本保留在 `.agent/work/IMPORT-001/import_experiments.py`。
- 验证：3993份结果原文与最新汇总的选定冻结哈希一致，718740条逐分钟指标齐全，快照来源哈希已核验。stage6终态与分钟快照的元信息版本分别按各自绑定保留；源文件未移动或改写。
- 下一步：该目录作为已有结果输入；尚未启动新 Core80 实验。

## 选集入口

- 编号：CORE-001；任务：整理并阅读 Core30~Core80 压缩包；状态：目录创建、复制、解压及阅读已完成，选集仍为候选。
- 文件位置：[1、build_core30_80](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、build_core30_80/)；压缩包复制到该目录，原始文件保留在 stage4 根目录。
- 阅读入口：[README.md](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、build_core30_80/core30_core80_all_sensitivity_package/core_nested_all_sensitivity/README.md)。程序入口为 `select_nested_cores.py`，名单位于包内 `results/core30.csv` 至 `core80.csv`。
- 已确认：包含664任务及2656条四探针聚合输入；六份默认名单按30、40、50、60、70、80严格嵌套；每个规模有5000条敏感性结果。敏感性固定各自父集合，不表示整链联合扰动；包不含实际训练和测试数据。
- 验证：249个文件哈希、压缩包复制一致性、名单数量及前缀嵌套、敏感性CSV行数均通过；未重新求解。证据：[inspection.json](work/CORE-001/inspection.json)。
- Core80 名单已用于 IMPORT-001 的路径交集筛选，未更改包内名单。

## 集群任务：待处理

- 编号：ENV-002；任务：逐机器训练测试及 DeepInfra 网络修复；状态：15 台完成测试，5 台 API 连接已修复，54 不可达。
- 当前采用方案：22~29、48~53、55，每台 15 算法，使用 g0275（feynman-i.14.4，含干扰变量）和 g0596（strogatz_glider1），clean、seed=520。训练预算 180 秒，保留预算内最佳候选；结果恢复和评估另计时间。LLMSR/DRSR 使用 DeepInfra Meta-Llama-3.1-8B-Instruct-Turbo，3 轮各 4 个样本。
- 有效结果：[逐机器结果](work/ENV-002/results_by_machine.csv)、[逐测试明细](work/ENV-002/results.csv)；脚本、配置、输入及 SHA256 见 [ENV-002](work/ENV-002/) 内 `run_checks.py`、`params.json`、`inputs_manifest.json`。远端原始输出位于 `<根目录>/sim-runtime/checks/ENV-002/results/`。
- 已确认：450/450 测试结束，无遗留测试会话；449 条产生有限 ID R2，25 上 E2ESR/g0275 预算内无有效公式。产生指标不代表预测效果或评估逻辑通过。
- 网络修复：49、51、52、53、55 的 `sim_llm` 通过 `.pth` 加载 `<根目录>/sim-runtime/network/sitecustomize.py`，仅将 `api.deepinfra.com` 连接地址设为 `38.101.151.13`，保留 URL、SNI 与 TLS 证书校验。5 台真实 Llama-3.1-8B-Instruct-Turbo 请求均回复 OK，证据 `dns_api_*.json`；原始 DNS/IP 连通检查为 `dns_49.json` 等。安装与复核脚本为 `install_dns.py`、`verify_deepinfra.py`。
- 未解决问题：历史20项 LLM 训练尚未重跑，原结果保留；系统DNS未修改。IP变化时通过 `SIM_DEEPINFRA_CONNECT_IP` 更新，空字符串可停用覆盖。变量索引问题已由 FIX-001 修复，历史3分钟测试输出未覆盖。54仍有30项测试未执行。
- 下一步：按后续任务重测历史20项 LLM；54恢复后补充部署和同口径测试。

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
- CORE-001 已完成；`.agent/work/CORE-001/` 原位归档，默认不读取。
- IMPORT-001 已完成；过程目录原位归档，整理脚本持续保留供复现。
- FIX-001 数值更正完成；过程目录原位归档，正式更正脚本和核验输入已保存在实验目录。
- FIX-002 已完成；过程目录原位归档，保留真实复现脚本与远端同步、验证记录。
- FIX-003 已完成；过程目录原位归档，保留同步脚本与验证记录。
- ENV-003 已完成；过程目录原位归档，保留运行脚本、输入、配置和结果。
