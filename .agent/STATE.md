# 工作区状态

## 当前任务

- 编号：EXP-001；任务：执行冻结的Core80正式全量实验；状态：已启动，训练已派发。
- 当前批次：`stage4_core80_all_20260922`；总任务10800，15台机器（22~29、48~53、55），54未纳入；每项10800秒，clean/noise001/noise005按顺序调度，seed520/521/522按顺序调度。LLM桶无数量限制，技术重试设置为高上限 `1000000`。
- 有效入口：[experiment_config.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/experiment_config.json)；任务源、参数切片和队列状态位于 `.agent/work/EXP-001/controller/`；调度日志：[controller.log](work/EXP-001/controller/queue/controller.log)；本地tmux会话：`core80_dispatch_20260922`。
- 当前验证：Dry-run已确认15算法×80数据集×3种子×3条件=10800项；LLM运行配置已部署到15台可用机器，配置哈希一致。15台机器支持文件已同步完成，GPU七台数据审计均为 `__MISSING__=0`；当前队列为 `2780 done / 1462 running / 6558 pending`，15个算法均已有完成结果，其中 FePySR、PySR、SymbolFit 已分别完成 `171、170、183` 项；已完成任务的远端 `result.json` 均为 `status=ok` 并包含 ID/OOD 指标；`clean` 已无待运行任务且有 `820` 项运行中，`noise001` 正在执行（`640` 项运行中、`2960` 项待运行），`noise005` 尚未开始；Opus后处理尚未启动。首次派发发现 `/home/anonymous` Julia 路径和GPU数据不完整问题，已修正启动环境、持久临时目录并补齐GPU数据；新启动的PySR任务已写出 `minute_0001.json` 和 `hall_of_fame.csv`。iaaccn25 当前 SSH 与任务状态查询无响应，调度器已重启并将主机不可达宽限调整为900秒，9项超预算任务已重新排队，随后继续派发任务。当前配置哈希为 `a0da7a60a112ff5ea1e9879140e8987e60bdfb8b6da4ba9186d8ef3231092629`，技术重试设置为高上限 `1000000`，付费服务按用户授权不设置费用上限。
- 下一步：持续轮询队列状态、主机状态和任务结果；按阶段向用户汇报已派发、运行、完成、失败和重试数量。完成训练后执行逐分钟六维Opus后处理。
- 待验证：FePySR正式任务曾因临时目录路径过长触发 `AF_UNIX path too long`；已将所有远端任务临时目录改为短持久路径，并从07:09起同步到新派发任务，等待新任务完成特征映射阶段后确认。

## 实验冻结配置

- 编号：FREEZE-001；任务：参考AAAI Stage6冻结新Core80统一实验配置；状态：已完成，已用于EXP-001。
- 当前采用方案：[experiment_config.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/experiment_config.json)为统一入口；保留15份 `formal_clean_params` 的算法参数与线程数，公共种子520/521/522、三条件0/0.01/0.05、每项10800秒、每60秒快照，共10800项。Core80名单固定80个唯一完整路径；噪声种子按固定数据集身份计算，运行时显式传入，避免主机路径影响噪声。
- 验证：15算法源JSON与最新wrapper/runner哈希已绑定；LLM正式配置的非敏感字段已在22核验（DeepInfra Llama3.1-8B Turbo，temperature0.6、max_tokens1024）。主机路径及密钥运行时绑定；TPSR模板seed23改由统一任务种子传入，符合当前runner行为。配置哈希 `a0da7a60a112ff5ea1e9879140e8987e60bdfb8b6da4ba9186d8ef3231092629`，同目录 `freeze_experiment_config.py --check` 通过。
- 下一步：正式结果完成后回收远端输出并启动逐分钟Opus后处理；`2、experiments`本地目录仍无文件，当前输出位于远端实验目录。

## 目录清理

- 编号：CLEAR-001；任务：清空 `stage4_core80_15algs_3seeds_3h/2、experiments` 文件并保留原目录结构；状态：已完成。
- 已确认：删除前161996个文件、10616个目录、约2.8 GiB；删除后文件数为0，目录数仍为10616，目录列表哈希保持 `1db4a0da500ed5908758c05d667e7739aca2fd4dc7d0a579baf6e65a40911722`。没有运行中的相关实验进程。
- 当前采用方案：清理后的目录保留11个算法及数据集、条件、seed、`corrected`、`progress`等目录，等待新的Core80正式实验写入；本次未处理 `1、build_core30_80` 或外部来源资料。
- 下一步：执行 Core80 × 15算法 × 3seed（520/521/522）× clean/noise001/noise005 的正式实验，共10800个任务，再进入Opus后处理。

## 最近任务

- 编号：FIX-004；任务：修复 FePySR 并逐机器重跑；状态：已完成15台各3个180秒实验，共45项，54不可达。
- 当前方案：核验原生配置和依赖后再写初始常数；常数基线不阻止短预算启动，先确定实际搜索次数再分配PySR时间；异常恢复保留指标但不标记训练成功。GPU七台配置从22的持久源码复制并通过Hydra验证。
- 有效文件：`algorithms/fepysr_wrapper/wrapper.py`、`benchmarks/runner.py`（均在 `scientific_intelligent_modelling/`）、`tests/test_fepysr_initialization.py`；执行脚本、输入、配置和逐节点结果位于 `.agent/work/FIX-004/`。配置核验 `config_verification.json`、代码同步 `remote_sync.json`；[前后对比及原式](work/FIX-004/results.csv)由 `export_results.py` 生成。3个输入与ENV-003相同，原结果不覆盖。
- 验证：15台代码同步且各2项针对性测试通过；45项重跑均有指标，无执行错误和剩余测试会话；44项选中真实搜索公式。ID R²中位数：g0275 -0.005105→0.338951，g0596 -0.044665→0.995413，g0005 -0.000111→0.999985；44项ID改善。
- 未解决问题：feynman-i.14.4在3分钟内效果仍不足，其中1项保留初始常数；54恢复后仍需补部署与3项重跑。28的初次独立配置探测连接中断，其后3项真实训练均完成，配置可正常使用。

## 上轮快速测试

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
- 历史更正入口曾位于 `2、experiments/current_evaluations.csv`，已由 CLEAR-001 按用户确认清除；本次未处理外部来源资料。
- 验证：3个针对性测试通过；15台可连接机器同步并全部通过相同测试，54仍不可达。远端记录：[remote_sync.json](work/FIX-001/remote_sync.json)；数据核验：[verification.json](work/FIX-001/verification.json)。
- 下一步：依据当前有效索引更新受影响的 Opus 后处理，再汇总正式六维指标；其余9个已导入算法未在本次做数值复用审核。

## 已有实验

- 编号：IMPORT-001；任务：按新 Core80 交集整理已有实验数据；状态：已完成，结果文件已由 CLEAR-001 清除。
- 原目录结构曾使用 `{algorithm}/{dataset}/{condition}/{seed}`；导入时为11算法、51数据集、3993次运行。原导入索引和更正结果文件位于目标目录中的文件已一并清除；整理脚本仍保留在 `.agent/work/IMPORT-001/import_experiments.py`。
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
- FIX-004 已完成；过程目录原位归档，保留部署与重跑脚本、配置来源和结果。
- CLEAR-001 已完成；过程目录无新增文件，删除范围与目录结构核验记录保留在本状态文件。
- FREEZE-001 已完成；核验过程原位归档，正式配置及生成/校验脚本保留在stage4根目录。
