# 工作区状态

## 当前任务

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
- 有效入口：[experiment_config.json](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/experiment_config.json)；任务源、参数切片和队列状态位于 `.agent/work/EXP-001/controller/`；调度日志：[controller.log](work/EXP-001/controller/queue/controller.log)；训练调度器独立会话PID：`1978557`。
- 当前验证（2026-09-22 17:27）：Dry-run已确认15算法×80数据集×3种子×3条件=10800项；LLM运行配置已部署到15台可用机器，配置哈希一致。15台机器支持文件已同步完成，GPU七台数据审计均为 `__MISSING__=0`；当前队列为 `4234 done / 1473 running / 5122 pending`，15个算法均已有完成结果，其中 FePySR、PySR、SymbolFit 已分别完成 `268、269、283` 项，LLMSR、DRSR 已分别完成 `320、320` 项；已完成任务的远端 `result.json` 均为 `status=ok` 并包含 ID/OOD 指标；`clean` 已完成 `3598` 项、`2` 项运行中，`noise001` 正在执行（`636` 项已完成、`1471` 项运行中、`1522` 项待运行），`noise005` 尚未开始；Opus后处理尚未启动。首次派发发现 `/home/anonymous` Julia 路径和GPU数据不完整问题，已修正启动环境、持久临时目录并补齐GPU数据；新启动的PySR任务已写出 `minute_0001.json` 和 `hall_of_fame.csv`。iaaccn25 仍有间歇性状态查询无响应，17:23:34 的批量启动失败后继续按技术重试设置处理；iaaccn53 于17:27:47继续批量启动20余项 seed521 的 `noise001` 任务；clean 任务已基本完成，当前继续在多台机器派发任务。调度器已重启并将主机不可达宽限调整为900秒，随后继续派发任务。当前配置哈希为 `a0da7a60a112ff5ea1e9879140e8987e60bdfb8b6da4ba9186d8ef3231092629`，技术重试设置为高上限 `1000000`，付费服务按用户授权不设置费用上限。
- 下一步：持续轮询队列状态、主机状态、任务结果和本机Opus状态；训练结果同步完成后继续补充 simplify 任务，再建立 equivalence 与 structure 任务。
- Opus证据：预测运行状态库为 `.agent/work/EXP-001/opus_postprocess/state/opus_pred_simplify.sqlite3`，运行报告为 `.agent/work/EXP-001/opus_postprocess/state/run_report.json`，条件计划位于 `.agent/work/EXP-001/opus_postprocess/plans/simplify_core80_{clean,noise001,noise005}.jsonl`，同步报告为 `.agent/work/EXP-001/opus_postprocess/sync_report.json`。Ground Truth 另有1个公式静态抽取失败，证据为 `.agent/work/EXP-001/opus_postprocess/evidence/ground_truth.report.json`；12个算法结果暂不能建立 simplify 任务，清单为 `.agent/work/EXP-001/opus_postprocess/plans/simplify_core80.unresolved.jsonl`。初次全量计划的条件门错误已记录在运行报告，当前改用 clean、noise001、noise005 顺序处理。
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
