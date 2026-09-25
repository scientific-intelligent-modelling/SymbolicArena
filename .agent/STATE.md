# 工作区状态

## 当前任务

- IDOOD-FULL664-001（2026-09-26，已完成）：生成`3、metrics/full664_core50_10m/id_mean.csv`与`ood_mean.csv`，各15行，字段`algorithm,full664_mean,core50_mean`。使用本轮clean/seed1314/600秒的9960项；Core50按当前选择ZIP的完整数据集路径从同批结果取子集。逐任务使用既有phi_nmse转换到0至100，无有效数值计0，分母固定664/50。脚本preflight `aggregate_full664_core50_id_ood.py`；逐任务输入与来源SHA为同目录`score_inputs.csv`，输入版本、名单、无效数量及输出SHA见`manifest.json`。9960份来源哈希、完整覆盖、均值复算及两表15算法一致性核验通过；原实验与三小时汇总未修改。
- FULL664-RECOVERY49（2026-09-26 05:12，已完成）：用户授权的236项已在13台机器重跑并收集，clean、seed1314、每项600秒及原算法参数保持不变；235项done、DrSR/g0378一项failed，候选池为空触发softmax错误，已保留原始结果和日志。全部9960项现已收集到对应seed1314目录，原9724份result SHA256逐项核验未改变，无同步错误或结果冲突。入口preflight `rerun_full664_missing.py`，任务、参数及原结果哈希位于`full664_10m/recovery49/`；完成核验`completion.json`，训练终态`terminal_queue.json`，数据来源选择`delivery_overrides.json`，逐文件SHA256证据`recovery49/bulk_manifests/`。本轮队列及采集进程均已结束，`.agent/work/FULL664-10M/recovery49/`原位归档，未删除旧输出；下一步等待指标任务。
- FULL664-10M（2026-09-26，训练记录已交付）：按用户紧急要求重跑664×15=9960项，clean、seed1314、每项600秒，保留60秒快照及原生训练目标选出的预算内最佳结果；未运行冒烟测试、未启动Opus。iaaccn22~29、48~53、55的数据文件存在、非空及LFS检查通过，共15台、每台664任务；54不可达。调度按现有CPU权重使用100%容量、内存使用92%停止新增、保留8GiB，不设置单进程内存上限；每台最多256项，每轮最多64项，LLM不另设全局任务数量上限。
- 有效入口：preflight `full664_10m.py`及`full664_10m/`中的664项清单、15份600秒参数、`experiment.json`、`host_data_audit.json`、`support_sync.json`；训练使用`run_core50_queue.py`，队列及日志位于`.agent/work/FULL664-10M/`。同步目标`2、experiments/clean/{algorithm}/{dataset}/1314/`，215个非Core50同名任务用g编号区分，已有Core50目录和旧seed保留。
- 训练状态：队列于04:13:40结束，done9956/failed4，无running或pending；失败为DrSR/g0378、JaxSR/g0338及g0383、RAG-SR/g0268，各已尝试2次。调度器已退出，未追加训练或Opus请求。
- 同步加速（已完成）：preflight `collect_full664_bulk.py`按机器gzip打包、4台并行传输、整包及逐文件SHA256核验，再只补充缺失文件。9724项来自原批次，49未取回的236项使用用户授权重跑结果；总计9960项均有本地记录，9764项有公式、9761项有有限ID/OOD指标，不能把完整采集解释为全部算法输出有效。
- 当前证据：preflight `full664_10m/collection.csv`、`collection_summary.json`；原批次源文件清单与SHA256为`bulk_manifests/{host}.json`，输入版本为`bulk_input_queue.json`及`bulk_input_collection.csv`，原始压缩包和传输日志保留在`.agent/work/FULL664-10M/bulk/`。重跑来源由`delivery_overrides.json`明确选择，证据独立保存在`recovery49/`。失联机器自动续传已停止；原批次队列和采集过程原位归档，默认不读取，未删除任何实验结果。
- REPORT-LLM-001（2026-09-26，已完成）：报告`3、metrics/opus_llama_protocol_report.md`说明Opus5对SYM/MIN/STAB的作用、Nguyen-9真实例子、LLM-SR/DrSR的Llama生成与BFGS评估、早期探针和Core50背景策略；附24个完整模板/spec/Schema区块。脚本preflight `build_llm_protocol_report.py`，输入来源及SHA256在报告末尾。已核验两算法900份参数、Stage2语义实验400条模型分组和iaaccn22当前非敏感API配置；原生buffer成功重新渲染两份真实spec的初始消息。历史逐请求报文未恢复，报告明确证据范围；未改训练、指标或Overleaf，未新增付费请求。无新增过程目录，下一步等待具体任务。
- CI-CORE50-001（2026-09-26，已完成）：clean六轴95% task bootstrap CI已重算。输入2250份运行及750个algorithm-task级STAB完整，固定seed20260926统一抽样50个task、有放回重复1000次，每个task保留3个seed；STAB按task聚合，使用2.5%/97.5%线性分位数。输出`3、metrics/clean_six_axis_task_bootstrap_ci.csv`共90行，字段`algorithm,axis,mean,ci_lower,ci_upper`，采用0至100尺度；mean与当前clean汇总一致。脚本preflight `compute_clean_task_bootstrap_ci.py`；2250行输入、750行STAB、1000×50抽样索引及输入/输出SHA256保存在`3、metrics/clean_task_bootstrap/`。完整覆盖、逐run抽样与task重复计数复算一致性检查通过；未修改实验、图表或Overleaf。
- TABLE-TEX-001（2026-09-25，已完成）：新增clean表格组件`3、metrics/tables/six_axis_clean_typeset.tex`及同名PDF预览；等宽指标列、统一行距、独立淡红/淡蓝高亮、局部分组样式，使用最新90个数值。预览入口preflight `preview_clean_typeset.tex`，按5.5英寸正文宽度通过XeLaTeX编译，一页、字体嵌入，无警告或溢出；PDF实际文本90个数值、列等距及横向位置核验通过。未修改原表或Overleaf；验证记录`.agent/work/TABLE-TEX-001/`已归档并原位保留。
- FORMULA-OVERRIDE-001（2026-09-25，已完成）：按用户明确要求，LLM-SR/g0436/noise005/seed522采用上一个可处理的原生候选sample196（第116分钟仍被选用），仅此一份采用该选式例外。原始result.json保持原SHA，所选候选、参数、训练目标及预算内快照逐项核验；新结果与证据位于`3、metrics/formula_override_g0436_s522/`，脚本preflight `use_core50_previous_formula.py`。4项Opus处理均首次验收，费用0.132294元；最终ID/OOD/SYM/MIN采用所选公式，该数据集STAB三个seed联合重算，EFF保持原生训练轨迹。
- LLM-SR/noise005最新六维均值为ID 33.35351、OOD 23.00112、SYM 31.26407、MIN 51.32484、EFF 80.96271、STAB 4.05176；各运行轴150/150、STAB 50/50，三个条件汇总表均无NA。已更新本地CSV、运行明细、Opus索引及PNG/PDF表格。仅目标运行的最终公式和同一数据集的STAB受影响，其他运行及所有EFF通过不变性核验。原始输入及前一汇总保留；Overleaf本轮未同步。过程`.agent/work/FORMULA-OVERRIDE-001/`已归档并原位保留。
- EFF-RECOVER-001（2026-09-25，已完成）：恢复154/158份缺失EFF，其中E2ESR148份、LLM-SR正常结束后延续1份、QLattice结束后延续1份、原生参数绑定恢复4份。DrSR、LLM-SR、E2ESR在clean/noise001/noise005均为150/150份，EFF均值依次为DrSR 83.92644/84.73846/83.42825，LLM-SR 80.54579/81.68784/80.96271，E2ESR 90.95185/89.44074/89.17778。JAXSR两种噪声仍有3/1份不可恢复，分别按147/149份求均值并标注数量；其余指标口径不变。
- 当前有效EFF输入为`3、metrics/eff_recovery/recovered_run_minutes.jsonl.gz`覆盖对应logical_key/minute，原始分钟记录保留在`inputs/minute_evidence_full/`。恢复前汇总保存在`eff_recovery/input_snapshot/`；脚本preflight `recover_core50_eff.py`及`--publish`，证据`eff_recovery/verification.json`，45组均值`eff_recovery/eff_means.csv`。已更新本地六维CSV、运行明细、缺失清单、SHA256及三张PNG/PDF表格；原生历史函数、参数和训练目标逐项绑定，其他五轴未改变。新增训练及API请求均为0。Overleaf本轮未修改，待明确同步要求；过程`.agent/work/EFF-RECOVER-001/`已归档并原位保留。
- EFF-DIAG-001（2026-09-25，已由EFF-RECOVER-001更正）：148份E2ESR轨迹因结束分钟取整条件被阻止延续。全部最终公式与末次快照一致，候选实际发现时间均早于末次记录对应分钟。原始诊断证据`.agent/work/EFF-DIAG-001/e2esr_tail_audit.json`及`inspect_e2esr.py`原位保留。
- OVERLEAF-002（2026-09-25，已更新）：仅修改项目`6a8fd70fc3fed21947d38f86`的`tables/six_axis_table_clean.tex`、`tables/six_axis_table_noise1.tex`、`tables/six_axis_table_noise5.tex`。使用本地六轴已验收结果更新270个单元格、第一第二名标记和NA说明，原布局、色系、分组及标签保留。每次提交前读取最新全文并精确替换，服务均返回`confirmed=true`；更新后逐字复核通过，版本分别38、15、14，各有2条待审修订，未接受修订、未编译、未操作其他远端文件。证据位于`.agent/work/OVERLEAF-002/`的`validation.json`、`edit_*.json`、`after_*.json`、`changes_*.json`；已提交LaTeX保存在`3、metrics/tables/six_axis_{clean,noise001,noise005}.tex`。过程目录已归档并原位保留，等待后续明确任务。
- OVERLEAF-001（2026-09-25，连接已验证）：项目`6a8fd70fc3fed21947d38f86`，名称`example`，XeLaTeX，主文档`main.tex`共220行；已通过HTTPS JSON-RPC完成initialize、initialized、tools/list及project_info/list_files/read_doc，读取第1至80行。文件227个、目录17个，主文档已有2条待审修订；未编译、未修改、未接受修订。独立原生服务`overleaf_6a8fd70fc3fed21947d38f86`已加入全局Codex配置，原有配置内容核验未变；下次启动Codex加载，当前会话使用直接HTTPS。凭据保存在工作区外的当前用户受限目录，通过http_headers_helper读取，禁止写入项目或输出。证据为`.agent/work/OVERLEAF-001/verification.json`及同目录脱敏工具响应；本次过程已归档并原位保留，等待具体项目任务。
- TABLE-CORE50（2026-09-25，已完成）：clean、noise001、noise005三张六轴表沿用六组范式及15算法顺序，270个单元格均显示两位小数，按完整精度标注第一和第二。JAXSR噪声EFF参与数量147/150、149/150以脚注说明。当前PNG/PDF/CSV已重新生成并核验数值、边界与重叠；输出`3、metrics/tables/six_axis_{condition}.{png,pdf,csv}`及`manifest.json`，脚本preflight `render_core50_six_axis_tables.py`。原始指标和样本数量保持不变，Overleaf本轮未修改。过程目录`.agent/work/TABLE-CORE50/`已归档并原位保留。
- GOAL-CORE50最终汇总（2026-09-25，已交付）：仅使用最终公式裁决计算SYM/MIN/STAB，EFF使用既有180分钟数值轨迹，新增API请求为0。6750份运行、2250组任务、45个算法条件组合覆盖核验通过；本地六维表复算、输出SHA256及全部已复制裁决与输入哈希核验通过。
- 有效文件：`A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/3、metrics/`中的`algorithm_six_axis.csv`（clean 15行）、`noise_supplement.csv`（30行）、`terminal_run_metrics.csv`、`terminal_run_metrics.jsonl.gz`、`unresolved.json`、`verification.json`。生成脚本为preflight `aggregate_core50_terminal_metrics.py`，本地采集入口`collect_core50_postprocess.py --terminal-only`；原始裁决保存在`opus/`，数值轨迹、训练结果冻结输入及50个数据集的输入版本保存在`inputs/`，来源映射见`storage_map.json`。
- 未解决：EFF剩余4份JAXSR轨迹缺失，均值注明可用分母，`formal_ready=false`。最终SYM/MIN/STAB已完整。当前证据以`3、metrics/verification.json`、EFF恢复清单及单公式选择例外清单为准；`.agent/work/GOAL-CORE50/`原始采集记录已归档并原位保留。

## 已替代执行快照

- GOAL-CORE50范围更新（2026-09-25，当前有效）：用户明确仅对最终公式进行Opus化简，逐分钟化简及逐分钟符号比较不再执行。iaaccn22逐分钟执行器`210454`和自动续行`183531`已终止，既有结果、原始数据及训练轨迹保留。远端`core50_minutes/stopped_by_user.json`记录停止范围，两个执行入口读取该文件并拒绝重新启动；采集器按该状态结束等待。最终公式处理批次已结束：终态等价性6737项、结构比较6732项已验收，见`core50_comparisons/terminal_continuation.json`。最终六维汇总仍待完成，应直接使用最终公式裁决和已有数值轨迹计算EFF，不再依赖逐分钟Opus。下方逐分钟API预算及自动续行计划已替代。
- GOAL-CORE50并发调整（2026-09-25，已确认）：数学响应校验50并发，公式及比较准备50个进程，API上限保持300。五份正式执行脚本已同步iaaccn22且SHA256一致，Python编译与Shell语法检查通过，提交`66365054`。旧执行器已完成提交中的请求，3059项化简验收记录保留；新执行器进度明确报告`semantic_validation_concurrency=50`，实时状态库确认300项运行中。准备阶段按50进程配置执行，当前仍处于API化简阶段。切换入口`.agent/work/GOAL-CORE50/restart_postprocess_50.sh`，远端`core50_minutes/concurrency50_restart.log`及`concurrency50_verification.json`。
- GOAL-CORE50续行（2026-09-25，执行中）：1432项补充化简及2项LLMSR参数恢复后的化简均已验收。剩余1441项中的6项PySR无公式、1项`np.linalg.norm`整批输入表达式保持原因记录。2项LLMSR在三个真实数据划分上通过预测一致性检查。终态已验收13443项比较，另外26项计算证据正在补齐；原2项空字段格式错误和2项序列化中断均通过已保存响应复核，未重新请求。未重跑训练，原始结果未覆盖。
- 后续执行入口：preflight `prepare_core50_remaining.py`、`recover_core50_llmsr_parameters.py`、`continue_core50_terminal.py`；远端`core50_comparisons/terminal_continuation.json`记录阶段，`continuation.log`记录新比较计划，执行器保持300并发、64k输出、每项首次加最多5次重试。新任务生成后优先处理未完成计划；已完成计划继续保留原哈希绑定。
- 全量分钟与数值处理已完成6750项、1215000个分钟记录的生成，生成器版本5：1187257个数值有效点、25327个明确无效点、2416个待核验点；真实来源检查未发现使用未来backfill的情况，证据`.agent/work/GOAL-CORE50/minute_causality_audit.json`。原生目标选式、canonical回放和早结束后的carry-forward均保留来源。输入位于`source_freezes_full/`、`source_trajectory_full/`，输出位于`minute_evidence_full/`。最终数值回放clean/noise001/noise005分别2218/2209/2207项有效、32/41/43项无效，见`terminal_numeric_full/`；数值无效不等于符号公式无效。
- 逐分钟Opus已经在iaaccn22执行：新增56220项化简、复用4719项；等价性至多60939项、结构比较至多96689项，总请求上限213848，每项最多5次重试。费用与重试额度已向用户报告，预算通知哈希见`.agent/work/GOAL-CORE50/minute_budget_notified.json`。总并发上限300，终态剩余比较未结束时逐分钟阶段使用274个名额，为终态保留26个名额，输出上限65536。
- 持续执行入口为preflight `continue_core50_minutes.py`、`run_core50_minute_opus.py`、`aggregate_core50_minute_metrics.py`；远端状态目录`/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_minutes/`包含`pipeline_status.json`、`progress.json`、`budget.json`和逐任务状态库。后续比较、六维汇总、核验依次自动执行。正式clean榜单使用最终表达式，EFF使用180分钟轨迹，噪声和逐分钟结果独立输出；最终验收尚未完成，`formal_ready=false`。
- 运行修复：准备队列按50个进程最多同时提交100项并保存SQLite检查点；API审计JSON使用紧凑编码；状态库读连接显式关闭。精确十进制复核在数值代入前保持表达式不求值，修复QLattice/PyOperon预计算触发密集多项式运算的问题；真实QLattice g0542/s521比较在0.425秒完成，原先超过180秒。证据为远端`thread_profile.json`、`comparison_format_revalidation.json`、`serialization_recovery.json`及preflight对应脚本。
- 本地`collect_core50_postprocess.py`持续等待远端核验结束，随后同步结果、Opus证据和输入版本到`A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/3、metrics/`；采集状态`.agent/work/GOAL-CORE50/postprocess_delivery.json`。当前没有完成交付，不得把启动或局部验收视为Goal完成。

## 本轮较早快照

- GOAL-CORE50（2026-09-24 19:54，执行中）：6750份训练结果已收集，完整输入SHA和任务身份核验通过；新增2025份训练中2016份有公式和数值指标，9份PySR g0270的HOF及备份仅含Loss=inf，保留不可用原因。输入与核验：`.agent/work/GOAL-CORE50/source_freezes_full/`、`sixaxis_input_audit_full.json`。
- Opus执行（已确认）：iaaccn22已采用300并发完成当前比较队列，`max_tokens=65536`；等价性5294项已验收、2项重试耗尽，结构4721项已验收，精确提示词重试81项已验收。远端进度时间戳1790257887.8188946，`registered=10098`、`in_flight=0`；执行器已退出。最后CPU 0.6%，可用内存约223 GiB。当前批次结束，全量后处理尚未完成。无物理内存或单进程内存限制；首次请求加最多5次重试。远端根目录`/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_comparisons/`，证据为`progress.json`、`api.log`、`execution/*/state.sqlite3`、`events.jsonl`。运行入口仍为preflight `start_core50_comparisons.sh`。
- 242项原失败已核查：159项symbolic artifact不一致、46项不等价、25项HTTP超时、10项JSON格式错误、2项JSON解析深度不足。隔离运行环境固定SymPy 1.13.1，159项原artifact哈希全部一致；重新验收161份历史响应，其中112份simplified、47份unchanged、2份unable。其余81项采用精确运算提示词并通过API重新请求，46份simplified、35份unchanged；2份unable继续保留未解决状态。旧请求和响应未改写。证据：远端`core50_new15_opus_v5/failure_audit/`、比较目录`retry_exact_manifest.json`及`execution/pred_simplify__*/`。
- 当前正式脚本：preflight中的`prepare_core50_opus_dependencies.py`、`run_core50_comparisons.py`、`revalidate_core50_opus.py`、`build_core50_exact_retry.py`、`start_core50_comparisons.sh`、`core50_opus_requirements.txt`；提示词：`AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt`。历史有效依赖1995份保留来源绑定，比较采用当前核验的数据探针；初始比较范围为5222项等价性和4622项结构任务，81项重试完成后自动补充可用比较。
- 验证：8项针对性测试通过，Python编译及Shell语法检查通过，远端实际API已返回等价性、结构及81份新提示词结果。尚未完成：另1441份新增训练输入的预测化简计划、完整逐分钟SYM/MIN/STAB及最终六维汇总。`formal_ready=false`。下一步持续完成已提交比较并补齐全部6750项后处理；过程目录`.agent/work/GOAL-CORE50/`保持活动。

## 已替代的任务快照

- 最新进度（2026-09-24 17:35）：6个iaaccn22直接HTTP API执行器均使用`max_tokens=65536`，主计划并发135，142项重试计划并发15，总并发150。已核验64k请求返回`end_turn`并通过结构验证。主状态库（clean/noise001/noise005）目前分别为`frozen/exhausted/running/pending`：`165/68/7/0`、`904/47/53/556`、`737/31/30/745`；GT另15项已完成。重试状态库覆盖原耗尽142项，目前`frozen=6, exhausted=7, running=15, pending=114`，每项新批次最多6次请求（首次加5次重试），新增上限852次。合并同一任务身份后，3358项为1827项已有有效冻结结果、11项耗尽、105项运行中、1415项待处理。64k批次报告累计费用¥24.020610；抽查主批次失败来自`validation_failed`和`structured_output_invalid`，重试批次已有7项再次耗尽。机器256核，load average为4.46/3.48/2.90，可用内存约229 GiB；6个API进程合计RSS约1.7 GiB，没有设置物理内存限制。旧尝试与失败记录均保留；重试计划清单为远端`/home/zhangziwen/sim-runtime/core50-opus-runtime/core50_new15_opus_v5/retry64k/retry_manifest.json`，生成脚本及执行入口为`.agent/work/GOAL-CORE50/build_core50_opus_retry_plans.py`、`run_core50_opus_retry64k.sh`，主执行入口为`.agent/work/GOAL-CORE50/run_core50_opus_v5.sh`。历史状态与费用见同一远端目录内各计划报告、SQLite状态库和逐次attempt JSON。
- 当前进度（2026-09-24 16:00）：DeepInfra和Opus API费用已获授权。新增训练队列中非LLM `1755/1755 done`，LLM `199/270 done、71 running、0 pending`；合计1954项终止、71项运行。采集器已导入1470项终态结果，其中1462项有公式和ID/OOD指标。8项PySR `g0270`任务（clean种子520/521/522，noise001种子520/521/522，noise005种子520/521）均在HOF和备份中只有`Loss=inf`，保留预算终止原因且不重跑；当前采集没有SHA冲突或同步错误。
- 复用范围未变：原Core50/Core80重叠35项的4725份运行已复制并核验，共859349个文件、8,147,741,253字节，保留849876个逐分钟progress文件。Opus输入SHA256绑定核验通过：1960项有效结果已复制，2762项缺少化简，3项无可用表达式；GT化简覆盖36个数据集。证据为preflight `reuse_summary.json`、`reuse_manifest.csv`、`pending_training.csv`、`pending_opus.csv`及`reuse_support/opus_bindings.json`。
- 六轴输入预审：4725份复用结果及新增结果的SHA256、任务身份和50个数据集输入哈希通过。v5 source freeze覆盖5309/6750项，剩余1441项；采集清单现为1470项，完整覆盖审计待刷新。报告`.agent/work/GOAL-CORE50/sixaxis_input_audit_collected_v5.json`，文件位于`.agent/work/GOAL-CORE50/source_freezes_collected_v5/`。
- 有效执行入口：`1、preflight/run_core50_queue.py`、`run_core50_nonllm_queue.sh`、`run_core50_llm_queue.sh`、`collect_core50_new15_results.py`、`build_core50_sixaxis_inputs.py`；持久队列状态位于`.agent/work/GOAL-CORE50/queue_{nonllm_v2,llm_v1}/`。
- 采集器已修正`iMCTS`、`QLattice`远端子目录大小写及本地规范目标路径，并对已有绑定重新核验result SHA和指标字段。旧代码产生的小写`imcts/qlattice`额外副本仍保留，未获用户确认前不删除；六轴输入按冻结清单中的规范路径读取。
- 三条件v5逐分钟来源各覆盖全部当前可用run，inventory/freeze绑定均`contract_ok=true`、`drift_count=0`。SHA绑定轨迹点数为clean 319271/320040、noise001 318680/319320、noise005 315549/316260；缺失分钟分别769、640、711，冲突和解析错误为0。4725项复用记录只有本地outer进度副本，584项新增结果还含inner目录；来源扫描核验inner副本可用情况，冻结文件按选中路径重新读取并核验SHA。inventory、freeze及绑定报告位于`.agent/work/GOAL-CORE50/source_trajectory_freeze_v5/`。
- v5逐分钟数值表按同一分钟的ID/OOD值生成，未用未来或最终公式填补缺失点。clean/noise001/noise005分别有315180/312849/311716个可用run-minute数值点，其余记录保持空值；明细及SHA见`.agent/work/GOAL-CORE50/dynamic_numeric_v5/`。
- v5逐分钟来源已扫描5309项运行，保留所选公式、ID/OOD原始指标、路径和SHA256。inventory/freeze绑定三条件均通过，`drift_count=0`；clean/noise001/noise005分别有769/640/711个缺失分钟，冲突和解析错误均为0。结果与报告位于`.agent/work/GOAL-CORE50/source_trajectory_freeze_v5/`。clean EFF确定性回放进程PID2733006仍运行，采用v5 source binding及无修复清单；缺失分钟按unresolved保留。
- 复用Opus核验：1960份预测化简缓存均通过原result SHA、公式、prompt/schema和历史响应验收；其中1327份来自Claude Code历史通道，633份来自直接API。所有缓存与当前evaluation_key不同，差异集中于探针和裁决证据字段，需用独立缓存重绑记录保留历史来源。36份GT缓存中35份与当前Ground Truth证据一致，`feynman-i.11.19`一项SHA不一致；目前需要15项GT直接API处理。审计报告`.agent/work/GOAL-CORE50/opus_cache_audit.json`，审计脚本`1、preflight/audit_core50_opus_reuse.py`。
- Stage5候选预测化简计划基于5309项输入：可调用clean 1777、noise001 1771、noise005 1755；6项无公式任务记为no-call。复用缓存1995项通过来源、表达式及prompt/schema核验；v5直接API待处理3358项（clean240、noise0011560、noise0051543、GT15），单任务重试上限16790、总尝试上限20148。该快照未包括后续采集结果，也未计等价、结构及逐分钟任务。计划与哈希见`.agent/work/GOAL-CORE50/opus_plan_merge_collected_v5/merge_report.json`。
- Stage5逐分钟选择已修正为使用同一分钟的算法原生incumbent，不按ID/OOD质量选择历史快照。动态噪声真实数据核验：DSO/Keijzer-11三个seed共540个run-minute点通过，seed522的`q(t)`从0.2551降至0.1871时仍使用minute 50公式及其SHA；证据`.agent/work/GOAL-CORE50/dynamic_real_smoke_v5b/`。定向测试4项通过；代码提交`daca57c1`。
- Opus直接API代码和密钥配置已在iaaccn22准备，未发送本Goal的API请求。物理内存264010832 kB，90%目标上限约237.9 GB；`systemd-run --user --scope --property=MemoryMax=237907890000`仍因委派权限失败，实际限制不可用。待管理员启用memory cgroup委派或提供系统级受限服务后再启动Opus；训练、采集和离线计划继续运行。
- clean EFF：首次计算因Stage5完整网格门要求2250行、v5当前仅有1778行而退出，保留`.agent/work/GOAL-CORE50/clean_eff_v5/run_trajectory_wide.csv`及失败原因。重算进程PID `2739592`显式使用`limit_runs=1778`，结果标记为部分输入，不能作为完整Core50正式结果。输入为v5 clean绑定、Stage5原生incumbent adapters及`trajectory_repairs_none_v5.json`。历史`trajectory_repairs.v1.json`的wrapper SHA与当前源码不符，已拒绝沿用；769个缺失分钟不修补，保留为unresolved。

- IMPORT-002（并入GOAL-CORE50，复制已完成）：新Stage4目录`2、experiments/{noise}/{algorithm}/{dataset_name}/{seed}`。35个重叠数据集的4725份运行已复制，15个新数据集对应2025项待训练；保存859349个文件及849876个逐分钟progress文件。4721项有可用数值指标，Opus SHA256绑定有效1960项、缺失2762项、无可用表达式3项；GT已覆盖36个数据集。全部统计、pending清单和来源哈希见preflight `reuse_summary.json`、`reuse_manifest.csv`、`pending_training.csv`、`pending_opus.csv`及`reuse_support/opus_bindings.json`。源实验未修改。

- COMPARE-001（2026-09-24，已确认）：用户新放入preflight的`core50_selection_outputs.zip`内`core50.csv`包含50个唯一任务，与`pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/experiment_config.json`冻结Core80重叠35项，Core50新增15项，Core80独有45项，并集95项。ID与完整数据集路径两种比较一致；脚本`.agent/work/COMPARE-001/compare.py`，文件哈希及差异名单见同目录`comparison.json`，过程原位归档。压缩包未修改或解压，原preflight文件未恢复。

- DIR-002（2026-09-24，已完成）：用户删除原`1、preflight/`后，按要求重新创建同名空目录，已核验目录为空，未恢复任何原文件。以下METHOD任务引用该目录的代码、名单和报告现已失效，仅保留历史记录。METHOD-007调整任务已被用户中止，未继续计算。

- METHOD-006（2026-09-24，已确认）：当前采用纯四probe W1选择及既有资格、family/subgroup和去重约束；`select_response_distribution.py`已移除Info下限参数及约束，Info保持诊断用途。当前有效候选为preflight1 `response_distribution_core50/core50.csv`，名单哈希未变，MAE=0.0130244276、Info=0.2157438292。帮助入口、真实保存解、变量边界、数量约束、W1和MAE复算通过，证据`.agent/work/METHOD-006/verification.json`，核验脚本同目录`verify_current.py`；过程原位归档。
- METHOD-006评估：该候选可用于固定Core50上的算法训练评测；四次留出均值0.201357、最大0.503287，尚不支持其稳定达到约0.15或可靠替代Full664。方法说明第13节记录使用范围；历史原生预测回放、新算法/三个小时/噪声/六维代表性尚未确认，formal_ready=false。METHOD-003候选已替代，原结果保留在`response_distribution_info050_core50/`，生成源码由提交`b773c44c`保留；本次未删除历史结果，未重新选集或启动训练。

- METHOD-005（2026-09-24，四次留出检查完成）：纯W1方法保持不变，每次只用其余三个probe重新选Core50，复用uDSR结果，其他三项并行、每项170秒求解，进程约176.5秒。DSO/iMCTS/PyOperon/uDSR的held-out绝对误差为0.5032872262/0.0627355933/0.0254151988/0.2139897890，平均0.2013569518；construction MAE平均0.0191444845，2/4次held-out误差≤0.15。
- METHOD-005有效入口：preflight1 `run_heldout_audit.py`及`heldout_loo_summary.json`；配置、输入/源码哈希、解和名单位于各`heldout_<probe>_core50/`。原始数据复算误差/W1及数量、资格、family/subgroup和去重检查全部通过，四次输入/选择器/评分/模型哈希一致。DSO/iMCTS/PyOperon/uDSR的求解gap约6.74%/3.43%/0.72%/1.90%，未证明全局最优。当前结果尚不支持各算法稳定达到约0.15；方法开发曾使用全部四probe，未执行外部验证或随机选集比较。方法说明第14.1节保留条件；`.agent/work/METHOD-005/`日志原位归档，原候选未替换，后续任务待确认。

- METHOD-004（2026-09-24，已完成单次留出检查）：固定uDSR为held-out，DSO/iMCTS/PyOperon构造纯W1 Core50，资格依据construction的9次运行重算，排除7项；选择阶段只读取身份和结构元信息，不使用四probe的Info/资格/响应标签。170秒求解，construction MAE=0.016728635；uDSR Full664均值5.6965008195、Core50均值5.9104906084，held-out绝对误差0.213989789，未达到约0.15。W1 gap约1.90%，未证明全局最优。
- METHOD-004入口为preflight1 `select_heldout_response.py`，配置、输入及源码哈希、冻结名单、完整解和报告位于`heldout_udsr_core50/`。名单保存后才评估held-out，原候选未替换。真实数据复算和约束检查通过；共享模型默认四probe路径复算原保存解通过，证据`.agent/work/METHOD-004/verification.json`，脚本同目录`verify_heldout.py`；过程原位归档。本次检验纯W1方法，未检验Info≥0.5候选。方法开发此前使用全部四probe，当前是回顾性单次检查，外部算法泛化尚未确认；后续任务待用户决定。

- METHOD-003（已替代，历史证据）：Info下限0.5实验结果MAE=0.271454、Info=0.500069，未达到约0.15目标。配置、输入哈希、名单和解保存在preflight1 `response_distribution_info050_core50/`，生成源码为提交`b773c44c`；核验证据`.agent/work/METHOD-003/verification.json`，过程原位归档，默认不读取。当前入口见METHOD-006。

- REVIEW-001（2026-09-24，已确认）：核验 `2、preflight2/core50_selection_package.zip`，原脚本真实运行0.61秒选出50项，独立重放名单一致，包内四份数据与Stage3逐字节相同。按已确认response口径从7968条原始记录复算MAE=2.23158742216141，DSO/iMCTS/PyOperon/uDSR分别1.291151/3.128115/1.274196/3.232888。当前输出未达到约0.15目标。
- REVIEW-001问题：程序只按固定个人评分排序并检查semantic duplicate和subgroup≤6；没有执行eligibility、family/difficulty/failure配额、basename约束或MILP，未调用MAE计算。实际选入有效比例1/3的g0619；LLM-SRBench仅4项、SRSD有23项，违反此前通用family配额。当前名单没有两类重复，subgroup上限通过。原压缩包及源码未修改，候选版本未替换。核验脚本 `.agent/work/REVIEW-001/audit_preflight2.py`，输入原包SHA256=a080317ecfed38c7ca7d9415c4856dc1633767ef444290144e1d175d440eb04d，完整命令、哈希、指标和约束证据 `audit.json`，输出 `replay/core50.csv`；该过程目录原位归档。后续修正待用户确认，历史原生预测回放未执行。

- METHOD-002（2026-09-24，已完成候选计算）：按用户要求重构选择规则，当前候选方法直接最小化四probe分别标准化response的一维W1均值，保留通用family/subgroup、资格及两类去重约束。Coverage和Info作为诊断项。入口 [select_response_distribution.py](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/select_response_distribution.py)，[方法与结果](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/response_distribution_method.md)，[候选名单](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/response_distribution_core50/core50.csv)。
- METHOD-002说明已补全：方法文档逐项定义输入字段、有效seed、response、总体标准化、经验CDF/W1、每项数量约束、整数CDF加强约束、MAE及Coverage/Info全部诊断分量，附真实数值例子、参数表和函数/输出映射。新增核验脚本 `.agent/work/METHOD-002/document_checks.py`，证据同目录documentation_checks.json；Disc/Stab/Info重算最大差小于3e-16，历史类别标签一致，实际W1、MAE、Coverage、配额和示例复算通过。说明明确区分历史NMSE输入与原生预测回放，以及当前目标和仅供诊断的字段。本次没有修改算法、参数或名单，也没有重新求解MILP。
- METHOD-002实测：180秒预算下Core50的四probe构造内MAE=0.013024427640043565；W1均值0.02783574193218572，求解下界0.02568934735860432，相对gap约7.71%，尚未证明全局最优。Coverage=0.961111，MeanInfo=0.215744，信息量低于用户固定50的0.518694。与暂存旧Core80重叠10，新增40；与用户50重叠7。约束、整数性、线性模型和独立W1计算通过，另由原始记录独立复算MAE一致；证据 `.agent/work/METHOD-001/response_distribution_verification.json`。模型未使用历史成员标签或MAE目标；该MAE用于方法开发，LOO与历史原生预测回放待做，formal_ready=false。
- METHOD-002保留的比较：仅替换Balance并保留Coverage/Info几何目标的方案，MAE=0.485717，记录位于preflight/response_balance_core50/。当前两个方案的脚本、配置、源码及输入哈希均保留；未启动训练或API。

- METHOD-001（2026-09-24，已确认调研）：用户固定50与SymbolicArenaCode/check/core50_selection/outputs/core50.csv为50/50一致；配套core50_selector_recovered.py使用历史成员标签训练ExtraTrees，属于名单兼容复现，不能据此确认原始文字方案独立生成了该名单。未修改公开仓库或选集程序。
- 两份50名单在暂存Core80方法的全部约束下均可行。按该方法重算，用户50的Coverage/Info/Balance为0.966667/0.518694/0.870042，暂存方法的Core50为0.994444/0.577630/0.904687；当前评分MAE分别0.151155/1.017740。后者三个目标分量均更高，因此改变这些分量的正权重无法使用户50在这两份名单间得分更高。
- 候选调整仅作双名单计算：Balance使用exp(-四probe分别标准化response的一维W1均值)，保留Coverage和Info。两个名单的响应分布距离分别0.256466/0.422651，对应几何目标0.729349/0.722035。尚未用该目标重新选集，不能承诺MAE改善。证据 `.agent/work/METHOD-001/selection_comparison.json`，脚本同目录inspect_selection.py；后续方法修改待确认。

- EVAL-002（2026-09-24，已完成）：以pre_exp_26.09.23暂存实验冻结的Core80名单，在同一Stage3 Full664数据及EVAL-001评分口径下计算MAE=0.9633741588372269。DSO/iMCTS/PyOperon/uDSR分别为0.550300648621008、0.7938809941935538、0.7749096449956594、1.7344053475386865。入口 [evaluate_core80.py](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/evaluate_core80.py)，结果 [core80_mae.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/core80_mae.json)；已核验80个唯一ID及完整路径、冻结参考CSV哈希、评分函数和Full664数据哈希与EVAL-001一致。结果是四probe历史运行诊断，未执行LOO或原生模型回放。

- EVAL-001（2026-09-24，已完成）：计算用户指定50个任务相对Full664的四probe平均response绝对差，使用已确认的probe_consensus_response_v1，MAE=0.15115479966356438。DSO/iMCTS/PyOperon/uDSR分别为0.06744972332041543、0.012748913307117249、0.07498742617122117、0.4494331358555037。入口 [evaluate_core50.py](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/evaluate_core50.py)，结果 [core50_mae.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/core50_mae.json)；保存50个ID、原始数据SHA256、脚本SHA256及评分参数。已核验7968条运行、每组3个seed、50个唯一任务及Full664覆盖。该结果基于历史原始指标，是固定名单诊断，未执行LOO或原生模型回放。

- TRACK-001（2026-09-24，已完成）：新Stage4的 `1、preflight/` 已开放Git跟踪，当前仅添加 `.gitkeep` 保留目录；未来 `2、experiments/` 继续忽略。其他ICLR实验目录保持原有忽略范围，已跟踪文件不受忽略规则影响。已核验preflight内文档及CSV路径可跟踪、实验数据路径被忽略。

- CLEAR-002（2026-09-24，已完成）：按用户明确要求清空 [1、preflight/](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/)，删除其中58个文件和4个子目录，保留preflight空目录。已核验没有相关选集进程，目录内容为空。
- CORE-002的选集任务书、计划、脚本及名单已清除，其过程目录 `.agent/work/CORE-002/` 保持原位置并标记已归档，默认不读取或执行。Stage3原始探针数据及pre_exp_26.09.23暂存实验未改动；当前固定名单评估见EVAL-001。

## 已暂存实验

- ARCHIVE-001（2026-09-24，已完成）：按用户确认，将ICLR Stage4完整迁移至 [pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/)，实验暂存，不再作为当前有效批次。
- EXP-001已停止并归档：Opus执行器PID2483768已结束，监控与准备服务均inactive，准备服务已取消注册；未删除实验文件。`.agent/work/EXP-001/` 保留原位置并标记已归档，默认不读取、不续跑。后续重新选择数据集尚未执行。
- 验证：同一文件系统目录重命名，Stage4目录inode70526846与实验目录inode70527320保持不变；冻结配置SHA256为c73295555568a46e00ea015a50df51caa01eeeaaf9f6ccfe4076cf09c9bf513a，支持源码ZIP为991d93f0c5996c3f05ad3cea241b575f68ed65def45848fac8f4c8f56da8bf8a，迁移前后一致。历史配置及证据原文保留，旧路径的前缀映射见 [暂存说明](../A_ICLR_experiments/pre_exp_26.09.23/README.md)。

## EXP-001历史记录（已归档，默认不读取）

- Opus API可用性（2026-09-24 01:38，已确认）：现有routify渠道真实请求1次、无重试，HTTP200，响应模型claude-opus-5，10.718秒返回OK，stop_reason=end_turn。脚本 `followup/probe_opus_api.py`，证据 `followup/probe_opus_api.jsonl`；使用与任务相同的adaptive thinking和xhigh配置。本次只检查接口，未修改或重启批处理执行器。

- 准备任务总内存上限更新（2026-09-24，已确认）：按用户要求调整为14GiB，仅覆盖准备服务及全部子进程；单进程不设置RLIMIT_AS，API仍为32并发。stage4 `core80-prepare.service` 已更新，运行时无重启应用；主PID2485101保持不变，systemd MemoryMax及该服务cgroup的memory.max均为15032385536。旧8GiB设置已替代。本机物理内存约15.4GiB，API及其他进程另占内存，仍存在系统级内存压力风险。

- API并发更新（已确认）：用户要求32并发，已停止原300并发执行器，并使用 `--workers 32 --recover-interrupted` 启动PID2483768；所有类型和条件共用32个请求线程及HTTP连接上限。入口为stage4 `run_core80_followup.sh`，tmux为 `core80_followup32`。已有选定结果、SQLite及计划保留，启动时恢复未完成请求；准备服务及8GiB总上限不变。已核验唯一API执行器启动参数为32，启动后的计划载入期间，旧进度文件仍可能显示上次参数；以下300并发记录已替代。

- 准备任务内存配置（2026-09-23，已确认）：按用户要求取消该准备任务工作进程和配对计算子进程的RLIMIT_AS限制，全部准备进程由 `core80-prepare.service` 统一使用8GiB总上限；旧单进程4GiB及进程组6GiB配置已替代。服务已重启，PID2483184，实际MemoryMax=8589934592，主进程地址空间限制为unlimited；配对助手接收memory_limit_bytes=None，其他调用者默认设置保持不变。真实BPG3/DRSR配对重算与既有完整证据一致，脚本及输出沿用 `followup/check_preparation_resume.py`、`pair_replay.json`。API PID2459526仍使用300并发，未重启。复杂输入的持续运行内存情况仍需监控；当前确认范围为配置生效和真实单项验证。

- 准备进程恢复（2026-09-23 23:01，EXP-001）：已确认原PID2461461所属进程组在21:31被systemd-oomd终止，占用7.5GiB，内存压力64.42%持续超过20秒；证据 `/var/log/syslog:25917`、`:26176`、`:26256`。原代码累计保留已完成Future的全部证据。现采用spawn独立工作进程、最多4个待处理Future、完成后释放结果引用，保留2个计算并发和单项180秒/4GiB限制；从已发布466批继续，已有批次和API结果保留。正式启动配置为stage4 `core80-prepare.service`，整组内存上限6GiB、禁用该服务交换空间、异常10秒后重启、每小时最多5次启动；PID2479901，取代准备进程tmux入口。每分钟监控新增准备进程状态及内存统计。API PID2459526仍为300并发，未重启。代码编译和差异检查通过；用BPG3/DRSR真实冻结输入重算配对证据，与原证据完整JSON一致，脚本 `followup/check_preparation_resume.py`、输出 `followup/pair_replay.json`。当前服务正在重新核验完整输入，恢复后的新批次生成仍待确认。

- 后续准备进程采用独立子进程执行每个配对证据，父进程执行180秒时限、每个子进程4GiB限制，2个计算并发；按表达式长度安排顺序，每25项发布计划。准备进程PID已更新为 `2461461`，tmux为 `core80_prepare`；API执行器PID `2459526` 保持运行，共享300并发上限。每个配对结束后由操作系统释放子进程资源。原准备进程的两个计算任务持续12分钟未返回，证据为 `followup/downstream_preparation.log` 和进程时间记录，不能将配置并发数当作实际HTTP并发数。已验证新批次被API执行器接收，结构判定真实请求已产生通过记录，证据 `followup/base/events.jsonl`。分钟记录采集已完成iaaccn23的132057份；25、26连接超时，已安排当前采集结束后只重试失败机器、每台最多5次，入口 `followup/retry_progress.sh`。最新数量以实时文件为准。

- 后续处理已启动（2026-09-23 20:38，EXP-001）：修正iMCTS/QLattice同步路径大小写并优先保留JAXSR恢复结果，本地结果已核验10800份。新增前置计划1523项（80个GT、1443个补齐/恢复预测）；GT 80项已全部处理，补充预测已处理1421项。等价性判定已开始，首批100项已处理81项；后续等价性/结构任务由准备进程持续生成，共享同一个300并发API执行器。实时状态为 `.agent/work/EXP-001/followup/progress.json`，分类型计数持续更新；批次仍在生成，注册任务数不代表完整覆盖。
- 当前入口：stage4下 `prepare_core80_followup.py`、`run_core80_followup.py`、`prepare_core80_downstream.py`；API执行器PID `2459526`，准备进程PID `2459451`，tmux为 `core80_downstream`（沿用持久socket）。输入计划、依赖来源数据库和文件哈希位于 `followup/base_plan.jsonl`、`dependencies.json`、`downstream_plans/`；原生支持脚本已原样保存至stage4 `runtime_support_sources.zip`，SHA256 `991d93f0c5996c3f05ad3cea241b575f68ed65def45848fac8f4c8f56da8bf8a`。API继续使用Opus5单轮请求，本机化简校验1并发、4GiB；独立数值和结构证据由现有Stage5模块生成，已有通过结果不重复请求。
- 逐分钟记录同步已启动：stage4 `collect_core80_progress.py`，PID `2459197`，tmux `core80_progress_collection`；进度为 `followup/progress_collection.json` 和 `.log`，后续需据真实完整轨迹生成逐分钟SYM/MIN/EFF/STAB，禁止提前使用最终公式填充旧分钟。当前仍有66个非空/缺失表达式的解析或大小问题待核验，明细为 `followup/unresolved.jsonl`；新生成配对中的待完成依赖、缺失种子及数值指标问题也保留为unresolved。整体汇总与完整动态覆盖尚未完成，不能将局部API队列完成视为全部实验验收。

- 当前化简队列已完成（2026-09-23 20:04，已确认）：`oversample/opus/progress.json` 为6291/6291，其中clean重试101/101、noise001 3094/3094、noise005 3096/3096。最后3项UDSR均已通过；clean和noise seed521由更新后的API请求完成，noise seed522使用原始API回复重新执行完整输入绑定、HTTP模型/schema和数学验证后，由 `TaskStateStore.promote_failed_attempt` 接受，原始失败记录保留。API执行器已停止，每分钟监控将完整队列显示为complete。此处范围仅限现有化简计划，恢复的3项JAXSR补入、全量覆盖及后续等价性/结构判定仍未完成。
- 校验修复与放宽（已确认）：精确十进制复核从原始文本重建表达式，使用带subs的高精度evalf处理大数相消；真实数值反例在未取得符号证明时也进行高精度复核。实数域对数恒等式在通分前展开，精确差分复用限时计算。单校验地址空间4GiB、时间180秒、全局1并发；prompt为 `exact_expression.v3`。5项真实回复回放测试通过（4项等价回复接受、1项舍入导致不等价的回复拒绝），证据为 `.agent/work/EXP-001/oversample/recheck_*.json`。最后一项的正式重新验收脚本为stage4 `revalidate_opus_attempt.py`，冻结文件为 `oversample/opus/noise005/replica_3/frozen/748e671aa3296fb20228412255741ebaaaa715f38f1eaf88dddbc0ab8d961d06.json`，证明为 `symbolic_difference_zero`，来源和新校验代码哈希保存在其revalidation字段。

- 剩余失败任务超发（2026-09-23 19:33，已确认）：141项未选定任务各安排5份独立API请求，总并发300；执行器PID `2451331`，启动参数增加 `--replicas 5 --max-tokens 65536`。问题已定位为部分回复舍入常数造成严格符号差异非零，以及16384个token全部用于thinking导致 `stop_reason=max_tokens`；另有504和本机数学校验内存限制错误。采用 `exact_expression.v2` system prompt，明确以原始expression为准、保留精确常数算式、严格输出JSON；输出上限65536，数学验收标准不变。真实API测试HTTP 200、模型 `claude-opus-5`，新回复通过 `symbolic_difference_zero`，已新增2项选定结果、剩139项。证据：`oversample/opus/noise005/replica_4/attempts/8450914149c1020b85d3a16a579acd0942b763f68349347a6199d59fed006362.a01.json`；配置和system prompt哈希保留于 `run_configurations.jsonl` 与逐请求记录。已通过项不重复派发，每分钟监控和2GiB校验内存限制继续使用。

- 最新并发（2026-09-23 17:33，已确认）：用户要求提高至300，API执行器已使用 `--workers 300 --include-noise --recover-interrupted` 恢复，PID `2435653`；clean与两个noise条件共用300个请求线程。本机公式校验继续为全局2并发、每进程2GiB，每分钟监控服务保持运行。启动时已选clean76项、noise001 1063项、noise005 1093项，结果均保留；实时计数以 `oversample/opus/progress.json` 为准。以下100并发配置已替代。

- 当前Opus调度（2026-09-23 17:07，已确认）：按用户要求总API并发提高到100，clean剩余任务与noise001、noise005交替派发，共用100个线程和HTTP连接池；本机数学校验仍为全局2并发、每进程2GiB。执行器PID `2431371`，启动参数 `--workers 100 --include-noise --recover-interrupted`。clean重试批次101项已选72项；现有noise001计划3094项已选18项，noise005计划3096项已选13项，均已收到真实API结果并通过校验。两个噪声计划的6190项来源结果哈希全部核验一致。噪声任务首次各请求一次，失败后逐轮重试；clean保留3份独立请求。各条件使用独立SQLite数据库，输入哈希与运行配置保存在 `oversample/opus/{noise001,noise005}/request_manifest.json` 和 `run_configurations.jsonl`，已选结果仍统一索引于 `selected.json`。每分钟监控增加分条件统计，脚本为stage4 `monitor_opus.py`。待补充：新恢复的3项JAXSR结果尚未进入旧噪声化简计划，完整noise任务覆盖及后续裁决阶段仍待核验。以下32并发记录已替代。

- 当前Opus传输（2026-09-23 16:56，已确认）：从本次开始仅使用项目现有 `AnthropicApiRunner` 直接请求 Messages API，停止使用CC客户端；执行器PID `2430154`，HTTP并发32。已验证真实HTTP 200、响应模型 `claude-opus-5`、完整JSON和数学语义校验链。原有67项选定结果及3016项原始冻结结果保留，当前仍有34项待完成。有效脚本为stage4 `opus_remaining_replicas.py`；渠道凭据仅从既有用户设置读取，不写入结果。每分钟监控仍运行。
- 本机公式校验采用全局2并发，每个独立进程地址空间上限2GiB，入口为stage4 `limited_semantic_worker.py`，超限保留错误并拒绝接受结果。已确认LLMSR g0649/seed520的真实API回复在SymPy `simplify -> trigsimp -> factor`处理期间触及内存上限，主执行器保持运行；证据为 `oversample/opus/replica_3/attempts/1732d0bc3aa2d94ad53c1182deb2271e2a9f0ca5f3843c7689455cdaa5964ae7.a04.json`。16:55监控主进程约377MiB、可用内存约13.7GiB。HTTP请求、usage、校验结果和传输版本保存在各 `replica_*/attempts/` 中，当前运行配置保存在 `run_configurations.jsonl`。以下CC运行状态均已替代。

- 每分钟监控（2026-09-23，已启动）：`core80-opus-monitor.service` 独立运行，每5秒采样进程与内存，每60秒记录完成数、剩余数、进程存活和内存峰值。脚本为stage4下 `monitor_opus.py`，状态为 `.agent/work/EXP-001/oversample/opus/monitor_latest.json`，记录为同目录 `monitor.jsonl`。最新异常：16:43:22 Opus的32并发任务组再次被systemd-oomd终止（33个进程），已选67项、剩余34项；以下16:41运行状态已替代。监控服务只观察，不自动重启任务。

- JAXSR校验阈值（2026-09-23，已确认）：按用户要求将 `jaxsr_wrapper/wrapper.py` 的 `_FIDELITY_RTOL` 调整为 `3e-6`，沿用 `1e-14 + rtol * native_abs_scale` 判断方式；iaaccn48真实依赖下原生模型恢复测试2项通过（2.37秒）。正在运行的训练进程仍使用启动时载入的阈值，已有失败快照未更改，后续恢复须以原生模型重新核验。

- JAXSR三个原始任务恢复（2026-09-23 16:36，已确认）：采用 iaaccn29的seed520/noise001、iaaccn28的seed521/noise005、iaaccn52的seed522/noise005原生模型，均通过3e-6重新校验和序列化预测一致性验证。正式结果为 `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/jaxsr/strogatz_barmag2/{noise001/520,noise005/521,noise005/522}/result.json`；对应 `recovered_rtol3e6/` 保存原生状态、输入哈希、原始错误报告及恢复证据。179个已有分钟记录逐项核验相同原生模型哈希后补充指标，第180分钟明确继承第179分钟。ID R2依次为0.886840、0.885528、0.884512；OOD R2依次为-2.610194、-2.844296、-2.720641。恢复脚本 `recover_original_jaxsr.py`、发布脚本 `publish_jaxsr_recovery.py` 位于stage4目录；未重新训练或按测试分数选模。
- JAXSR超发任务已结束：iaaccn48、50、51的9份副本已停止，文件保留；原队列剩余3项已按恢复证据更新为done，训练调度器已停止，自动重新派发的重复任务一并终止。恢复前队列为 `.agent/work/EXP-001/oversample/queue_before_jaxsr_recovery.json`，追加任务索引为同目录 `jaxsr_launch.json`。当前训练队列10800项done；本地全量结果完整性仍待核验。当前有效结果由 `recovery_provenance.json` 和队列 `recovered_result_path` 标记，后续同步必须优先使用该恢复文件，不能用旧失败结果覆盖。
- 当前 Opus 超发（2026-09-23 16:41，已确认）：按用户最新要求仅运行Opus，恢复32并发；101项已选定67项，剩余34项继续每项3份独立请求，原有3016份冻结结果保留。脚本为 `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/opus_remaining_replicas.py --workers 32 --recover-interrupted`，执行器PID `2425602`。已核验训练调度器不存在，iaaccn48、50、51、53、26、29相关训练进程均已停止。结果索引为 `.agent/work/EXP-001/oversample/opus/selected.json`，实时进度为同目录 `progress.json`，输入为 `request_manifest.json`，运行配置为 `run_configurations.jsonl`，回复与校验为 `replica_*/attempts/`。tmux socket 为 `.agent/work/EXP-001/oversample/tmux/server.sock`，会话为 `core80_opus_replicas`。本机16:03和16:15分别在32和4并发时发生systemd-oomd终止，证据为用户journal；远程训练停止不会直接释放本机内存。16:41本机available约10GiB，内存风险仍需监控。待验证：剩余34项完成、正式汇总绑定、后续噪声条件和已停止的本机同步任务。
- 以下14:26及更早 Opus 运行状态已替代，仅供追溯；当前采用以上超发方案。

- 编号：EXP-001；任务：执行冻结的Core80正式全量实验；状态：已启动，训练已派发。
- 当前批次：`stage4_core80_all_20260922`；总任务10800，15台机器（22~29、48~53、55），54未纳入；每项10800秒，clean/noise001/noise005按顺序调度，seed520/521/522按顺序调度。LLM桶无数量限制，技术重试设置为高上限 `1000000`。
- 本机Opus后处理：已启动，使用 `.agent/work/EXP-001/opus_postprocess/`，并发32；当前运行预测公式 `simplify.v1`，单任务尝试上限已从6提高到1000000000，并已重新开放228项可重试任务；当前状态为冻结3016项、重试等待51项、数据库运行48项、耗尽2项。`run_report.json`最后于13:19写入，记录此前一次执行器因 circuit breaker 停止；当前计划执行器和 Claude worker 仍在运行，数据库在14:26继续更新重试任务，但自13:19以来尚未新增冻结结果。因超大 gplearn 表达式导致的本机内存占用已处理，计划生成器已将超过1000个 AST 节点的预测结果记入 unresolved；守护进程已改为计划运行期间跳过重复计划生成，避免再次占用大量内存。证据为 `plans/simplify_core80.report.json`、`state/opus_pred_simplify.sqlite3`、`state/run_report.json` 与 `daemon.log`。远端结果同步与本机处理同时进行。
- 最新核验（2026-09-23 14:06）：Core80训练队列为 `10797 done / 3 running`；clean `3600 done`，noise001 `3599 done / 1 running`，noise005 `3598 done / 2 running`。三个 JAXSR `g0595` 的远端 tmux 会话和运行进程均保持存活，iaaccn29 已写出 `minute_0096.json`，iaaccn52 已写出 `minute_0065.json`；训练调度器PID `1978557`仍在运行。Opus继续使用32并发，详见下一条最新 Opus 核验。
- 最新 Opus 核验（2026-09-23 14:26）：冻结结果仍为 `3016` 项；数据库为 `running 48 / retry_wait 51 / exhausted 2`，最近更新任务主要记录 `validation_failed` 与 `timeout`，当前计划执行器PID `2064853`及其 Claude worker仍存活。`run_report.json`的 circuit breaker 状态属于13:19的旧报告，不能作为当前进程已结束的依据；当前重试活动证据为数据库 `updated_at`、执行器进程和 `daemon.log`。
- 归档配置：[experiment_config.json](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/experiment_config.json)；任务源、参数切片和队列状态位于 `.agent/work/EXP-001/controller/`；调度日志：[controller.log](work/EXP-001/controller/queue/controller.log)；原训练调度器PID：`1978557`。
- 当前验证（2026-09-22 17:27）：Dry-run已确认15算法×80数据集×3种子×3条件=10800项；LLM运行配置已部署到15台可用机器，配置哈希一致。15台机器支持文件已同步完成，GPU七台数据审计均为 `__MISSING__=0`；当前队列为 `4234 done / 1473 running / 5122 pending`，15个算法均已有完成结果，其中 FePySR、PySR、SymbolFit 已分别完成 `268、269、283` 项，LLMSR、DRSR 已分别完成 `320、320` 项；已完成任务的远端 `result.json` 均为 `status=ok` 并包含 ID/OOD 指标；`clean` 已完成 `3598` 项、`2` 项运行中，`noise001` 正在执行（`636` 项已完成、`1471` 项运行中、`1522` 项待运行），`noise005` 尚未开始；Opus后处理尚未启动。首次派发发现 `/home/anonymous` Julia 路径和GPU数据不完整问题，已修正启动环境、持久临时目录并补齐GPU数据；新启动的PySR任务已写出 `minute_0001.json` 和 `hall_of_fame.csv`。iaaccn25 仍有间歇性状态查询无响应，17:23:34 的批量启动失败后继续按技术重试设置处理；iaaccn53 于17:27:47继续批量启动20余项 seed521 的 `noise001` 任务；clean 任务已基本完成，当前继续在多台机器派发任务。调度器已重启并将主机不可达宽限调整为900秒，随后继续派发任务。当前配置哈希为 `a0da7a60a112ff5ea1e9879140e8987e60bdfb8b6da4ba9186d8ef3231092629`，技术重试设置为高上限 `1000000`，付费服务按用户授权不设置费用上限。
- 下一步：持续轮询队列状态、主机状态、任务结果和本机Opus状态；训练结果同步完成后继续补充 simplify 任务，再建立 equivalence 与 structure 任务。
- Opus证据：预测运行状态库为 `.agent/work/EXP-001/opus_postprocess/state/opus_pred_simplify.sqlite3`，运行报告为 `.agent/work/EXP-001/opus_postprocess/state/run_report.json`，条件计划位于 `.agent/work/EXP-001/opus_postprocess/plans/simplify_core80_{clean,noise001,noise005}.jsonl`，同步报告为 `.agent/work/EXP-001/opus_postprocess/sync_report.json`。Ground Truth 另有1个公式静态抽取失败，证据为 `.agent/work/EXP-001/opus_postprocess/evidence/ground_truth.report.json`；12个算法结果暂不能建立 simplify 任务，清单为 `.agent/work/EXP-001/opus_postprocess/plans/simplify_core80.unresolved.jsonl`。初次全量计划的条件门错误已记录在运行报告，当前改用 clean、noise001、noise005 顺序处理。
- 待验证：FePySR正式任务曾因临时目录路径过长触发 `AF_UNIX path too long`；已将所有远端任务临时目录改为短持久路径，并从07:09起同步到新派发任务，等待新任务完成特征映射阶段后确认。

## 实验冻结配置

- 编号：FREEZE-001；任务：参考AAAI Stage6冻结新Core80统一实验配置；状态：已完成，已用于EXP-001。
- 已归档方案：[experiment_config.json](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/experiment_config.json)；保留15份 `formal_clean_params` 的算法参数与线程数，公共种子520/521/522、三条件0/0.01/0.05、每项10800秒、每60秒快照，共10800项。Core80名单固定80个唯一完整路径；噪声种子按固定数据集身份计算，运行时显式传入，避免主机路径影响噪声。
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
- 文件位置（已暂存）：[1、build_core30_80](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/1、build_core30_80/)；压缩包及原始文件随Stage4保留。
- 阅读入口：[README.md](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/1、build_core30_80/core30_core80_all_sensitivity_package/core_nested_all_sensitivity/README.md)。程序入口为 `select_nested_cores.py`，名单位于包内 `results/core30.csv` 至 `core80.csv`。
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
- 新ICLR Stage4入口：[1、preflight/](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/)，当前包含.gitkeep及EVAL-001/002的固定Core50、暂存Core80评估脚本与结果。
- 旧ICLR Stage4已暂存：[stage4_core80_15algs_3seeds_3h/](../A_ICLR_experiments/pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/)，实验已停止。

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
