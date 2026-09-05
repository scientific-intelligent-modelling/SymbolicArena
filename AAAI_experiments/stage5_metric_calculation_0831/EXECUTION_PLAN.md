# SymbolicArena 新六轴指标执行方案

## 1. 目标与边界

本方案用于执行新的六轴正式评测：

\[
\boxed{\mathrm{ID,\ OOD,\ SYM,\ MIN,\ EFF,\ STAB}}
\]

指标定义以 `SymbolicArena_SixAxis_Revised.md` 为唯一口径。本轮不得复用旧六轴的聚合逻辑，包括：

- seed 原始 NMSE 先取中位数；
- 带 ID 保持率的旧 OOD；
- 基于运行时间或最终性能的 EFF 代理；
- 带 ID/OOD/SYM 性能修正的旧 STAB；
- 将 ROB 或 ROBU 放入正式六轴。

本文件定义 Goal 的执行路径；正式大模型任务通过双渠道 Anthropic Messages API 直接执行，不再经过 Claude Code CLI。

### 1.0 当前权威输入（2026-09-05 修订）

`6750` 条逻辑评测网格保持不变，但正式输入不再来自全量重跑。权威来源采用 hybrid provenance：

- 原始未受影响结果作为基底；
- 已完成的 `368` 个针对性重跑按冻结的 replacement scope 覆盖对应 final 和/或 EFF 输入；
- clean replacement 共 `225` 个：SymbolFit `150`、JAXSR `35`、iMCTS `40`；
- noisy replacement 共 `143` 个：DRSR `1`、JAXSR `63`、iMCTS `79`；
- 每个 logical key 只能绑定一个明确来源，并冻结源路径、SHA-256、replacement scope 和替换理由；
- 发布前必须证明完整逻辑网格、来源哈希、重放路径、失败重试和最终覆盖率均闭合。

`all_15alg_fullcpu_v1_formal` 已于 2026-09-05 中止。该批次的 state、result、progress 和 partial outputs 全部隔离，禁止补数、续跑或进入正式冻结与聚合。批次标记与停机审计见 `reruns/all_15alg_fullcpu_v1/ABORTED_DO_NOT_USE.json` 和 `reruns/all_15alg_fullcpu_v1/runtime/abort_fullcpu_20260905-1118/`。

### 1.1 EFF 正式口径更新

本轮 EFF 的正式目标固定为：

> **算法内部搜索的真实进展**

具体约束如下：

- 每个算法的历史最优候选只能由该算法自身的内部目标决定，例如内部 loss、reward 或 score；
- ID/OOD 质量只用于重放并评价已经选出的候选，不得参与候选选择；
- 分钟轨迹必须单调保留预算内的内部历史最优，后续较差候选不得覆盖先前最优；
- 不使用“最终可交付公式出现时间”替代内部搜索进度；
- SymbolFit 所需的 `150` 个 clean runs 已包含在完成的针对性 replacement 中；
- `noise001` 和 `noise005` 仍只进入补充噪声诊断，不计算正式 EFF；只有上述 `143` 个 noisy replacement 可以覆盖原始输入，未列入 replacement manifest 的 noisy 结果沿用原始权威来源。

## 2. 冻结的实验范围

预期源数据网格为：

- 15 个算法；
- 50 个 SSR-50 任务；
- 3 个随机种子：`520`、`521`、`522`；
- 3 个条件：`clean`、`noise001`、`noise005`；
- 6750 条最终运行；
- 2250 条 clean 运行进入正式六轴；
- 每条 clean 运行最多 180 个分钟级检查点，用于 EFF。

Stage 4 原始基底索引为：

```text
../stage4_ssr50_15algs_3seeds_3noise_3h/selected_runs_with_fepysr_rerun.csv
```

本地预检确认该表当前包含 6750 个唯一 run，且全部标记为 `ok`，ID/OOD NMSE 均为有限值。但该表只是原始基底，不是最终唯一权威来源。最终权威索引必须由该基底与 `368` 个针对性 replacement 按 logical key 合成，显式记录 `selected_source`、`replacement_scope` 和输入 SHA-256；它仍需冻结 `result.json` 才能形成完整评分输入。

SSR-50 Ground Truth 清单为：

```text
../../exp-planning/04.Core50正式全量评测/core50_datasets.csv
```

其中 50 组 `metadata.yaml`、`formula.py` 和数据切分文件都需要冻结哈希。Ground Truth 公式必须结合 metadata 和 Python AST 提取，不能假设所有 `formula.py` 的目标函数都叫同一个名字。

## 3. Claude 逻辑任务与硬上限

在所有最终公式和 clean seed pair 都有效时，逻辑任务上限为：

| 任务类型 | 最大逻辑任务数 |
|---|---:|
| Ground Truth 化简 | 50 |
| 最终公式化简 | 6750 |
| 最终公式等价裁决 | 6750 |
| clean seed pair 结构一致性裁决 | 2250 |
| **合计** | **15800** |

结构一致性任务数为：

\[
15\times 50\times \binom{3}{2}=2250.
\]

用户允许总调用尝试比逻辑任务上限多 50%。因此物理尝试硬上限为：

\[
B_{\mathrm{attempt}}
=
\left\lfloor 1.5\times 15800\right\rfloor
=
23700.
\]

固定规则：

- 逻辑任务上限为 `15800`；
- 独立重试池最多为 `7900` 次；
- 物理尝试硬上限为 `23700`；
- 每个逻辑任务默认最多 3 次尝试，即首次加最多两次重试；
- 若冻结审计证明失败属于 API 瞬时错误、输出截断或可重试结构化输出错误，可先备份状态库、逐条校验不可变 attempt 证据，再审计化上调该批任务的单任务尝试上限；
- 任何单任务上限扩展都不得改变逻辑任务身份、prompt、模型或全局物理尝试硬上限，并必须留下旧上限、新上限、原因、候选 attempt SHA-256 和状态库备份 SHA-256；
- 全局硬上限优先于单任务重试额度；
- worker 必须在启动 Claude 进程前原子预占一次预算；
- 只要进程已发起，即使随后失败，也保守计为一次尝试；
- 预算预占数达到 `23700` 后，不得再启动任何新尝试；
- 源公式缺失或 seed pair 无效时，写入显式 no-call 终态，不伪造 Claude 响应。

## 4. 依赖关系

```text
阶段 0：冻结清单、来源和哈希
  |
  +--> 阶段 1：按 hybrid 权威清单冻结并核验 6750 个逻辑 final 输入
  |      |
  |      +--> ID/OOD 数值输入
  |      +--> 最终公式和确定性证据
  |
  +--> 阶段 2：采集并核验 2250 条 clean 轨迹
  |      |
  |      +--> EFF 数值输入
  |
  +--> 阶段 3：实现并测试指标核心与 Claude 控制面
         |
         +--> GT 化简（50）
         |      |
         |      +----------------------------+
         |                                   |
         +--> 最终公式化简 ------------------+--> 等价裁决
                         |
                         +--> clean seed pair 结构裁决

冻结的数值输入 + 冻结的 Claude 输出
  |
  +--> run 级指标
  +--> task 级 STAB
  +--> 15 行正式六轴表
  +--> 噪声补充诊断
  +--> 覆盖率、冲突、重试、成本和来源审计
```

任何下游任务只有在输入哈希和依赖结果冻结后才能进入队列。

## 5. 阶段 0：来源预检与冻结

### 5.1 校验笛卡尔积

付费调用前必须断言：

- 总行数恰好为 6750；
- 算法、任务、seed、noise 集合分别恰好为 15、50、3、3；
- `(algorithm, dataset_id, seed, noise_tag)` 全局唯一；
- 每个算法恰好 450 条；
- 每个条件恰好 2250 条；
- 不存在意外算法、任务、seed 或条件；
- 每条路径都能映射到 Stage 4 原始来源或冻结的针对性 replacement；
- 来源集合不含 `all_15alg_fullcpu_v1_formal` 的任何路径、batch、result 或 progress；
- `status=ok` 时，索引内 ID/OOD NMSE 为有限值。

任一断言失败都必须在第一次 Claude 调用前停止。

### 5.2 冻结 6750 个 hybrid final 输入

按 hybrid authoritative manifest，只读采集每个 logical key 最终选中的 `result.json` 及其引用的 canonical artifact，写入不可变的本地 source snapshot。任何指向 `all_15alg_fullcpu_v1_formal` 的记录必须立即拒绝。

采集器必须：

1. 将每条记录归一化为 `(algorithm, dataset_id, seed, noise_tag)`，同时保留源 `task_id`；
2. 保留原始主机和远端绝对路径；
3. 在字段提取前计算 SHA-256；
4. 提取最终公式、canonical artifact、validity、ID NMSE 和 OOD NMSE；
5. 在固定容差内比对原始结果和 Stage 4 索引的 NMSE；
6. 对选择不一致或指标不一致立即停止，禁止静默覆盖；
7. 对公式缺失、为空、无效或不可解析分别记录明确状态。

后续计算只能读取冻结的本地快照，不能一边评分一边访问可变远端文件。

远程采集遵循仓库规则：连接必须带超时；本地优先到 `iaaccn22`，再按需通过 `10.10.100.23~29` 访问其他机器；复杂逻辑先发脚本再执行，不内联长命令。采集行为不得改写远端实验目录。

### 5.3 冻结 2250 条 clean 轨迹

只有 clean 运行需要分钟级轨迹。采集器要索引 `t=1,...,180` 的预期检查点，保留每个源文件哈希，并按下面的逻辑键处理重复副本：

```text
(algorithm, dataset_id, seed, checkpoint_index)
```

只有归一化内容完全相同时，outer/inner 副本才能折叠。若同一逻辑键内容冲突，必须标记为来源冲突并阻断该 run 的 EFF，不能按路径顺序或 mtime 猜一个。

轨迹覆盖以原始冻结轨迹与 `225` 个 clean 针对性 replacement 的合成清单为准。缺失遥测不得换成旧 runtime proxy，不得直接解释为“该分钟还没有找到公式”，也不得用已中止 fullcpu 批次的 partial outputs 填补。

## 6. 阶段 1：新指标确定性核心

新公式独立实现。旧脚本最多复用经过测试的底层解析工具，不复用任何旧聚合函数。

### 6.1 ID 与 OOD

每条 clean task-seed 独立执行固定 `phi` 映射，然后对有界质量值做经验平均。禁止先对 seed 的原始 NMSE 取中位数。无效数值输出按指标文档记为质量 0。

应输出：

- 2250 行 clean run 级 ID/OOD 质量；
- 15 行算法级 ID/OOD 分数。

### 6.2 EFF

EFF 衡量“算法内部搜索的真实进展”，使用固定 180 点网格，并保证同一分钟的 ID/OOD 来自同一个候选公式：

1. 将时间戳或 checkpoint index 规范为 `t=1,...,180`；
2. 先完成副本去重和冲突检查；
3. 按算法自身的内部 loss/reward/score 选择并维护预算内历史最优候选；
4. 禁止使用 ID、OOD、二者平均质量或 Ground Truth 信息选择候选；
5. 对选定候选统一执行 canonical replay，再计算 `q(t)=(q_ID(t)+q_OOD(t))/2`；
6. 只有当轨迹格式能证明上一内部 best-so-far 状态仍有效时才允许向前填充；
7. 只有源记录明确表示尚无有效候选时，首个候选之前才填 0；
8. 最终公式只有在能够证明其是预算内内部历史最优时，才可作为 `t=180` 的新候选；
9. 计算 `q*=max_t q(t)` 和文档定义的相对进度积分；
10. 只有确认整次运行从未产生有效公式时，EFF 才记 0；
11. 因产物丢失而无法重建时，标记 unavailable，不伪造分数。

每个分钟快照还必须绑定任务身份：`task_label`、`task_global_index`、
`expected_dataset_rel`、`expected_dataset_dir` 和 `dataset_identity_check`。新产生的快照
必须由 runner 直接写入这些字段。对身份字段部署前已经启动的运行，只允许从冻结的调度
state 和唯一任务目录补全，并逐条验证快照中的 `dataset_dir` 与期望数据集完全一致；无法唯一
绑定时按 unavailable 处理，禁止仅凭文件名猜测。

SymbolFit 必须额外满足：快照记录 PySR 在缩放搜索空间中的在线历史最优内部 loss，同时保存把候选确定性逆变换回原始变量和目标尺度所需的元数据。不得用事后 LMFIT 结果回填早期分钟，也不得用 canonical replay 的 ID/OOD 质量反向选择 PySR 候选。所需 `150` 条 clean replacement 已完成，只接纳其冻结清单声明的证据。

QLattice、DSO 和 gplearn 的 EFF 优先从原始冻结证据按各自内部目标重建 best-so-far；确定性的 adapter/replay 修复不等于重跑训练。只有独立 replacement manifest 明确列出的任务才可覆盖原始输入。已中止 fullcpu 批次产生的 QLattice 或其它算法文件即使看似完整，也不得进入 freeze 或 EFF。

每条被接纳的针对性 replacement 快照必须包含可验证的内部目标，并与 `dataset_dir`、seed、condition、任务索引和 replacement scope 一致；180 个点未被本轮完整覆盖时不得混入 fullcpu 或其它实验代际的旧点补齐。

不同算法的轨迹格式可能需要独立 adapter。每个 adapter 都必须通过相同的 180 点契约测试。任一 clean EFF 仍不可重建时，不发布正式六轴榜单。

### 6.3 符号确定性证据

Claude 调用前，为每条公式生成并冻结：

- parser 状态和规范化表达式；
- 变量集合与算子集合；
- canonical expression tree 和节点数；
- SymPy 可判定时的等价结果；
- 固定随机种子、定义域保护和样本哈希的独立数值探针；
- 除法、对数、根号、幂等保护算子的语义假设；
- tree distance、变量 F1、算子 F1 的输入证据。

这些证据必须交给 Opus5，但不能替代任何文档规定的强制大模型调用。

## 7. 阶段 2：双渠道 Opus5 API 控制面

### 7.1 固定调用契约

每次调用都是独立、无状态、非流式的 Messages API 请求：

```text
model: claude-opus-5
effort: xhigh
stream: false
thinking: adaptive
transport: Anthropic Messages API
channels: routify, yapi
```

请求参数必须与指标文档一致。每个请求只接收准备好的表达式、确定性证据和版本化输出 Schema；不启用工具或会话。

API token 不得写入 manifest、日志、prompt、命令输出、Git 文件或最终报告。正式批处理前应轮换此前在对话中暴露过的 token。

控制器同时记录请求模型和实际响应模型。只要出现模型、effort、响应类型或传输契约不符，就打开全局熔断器；这不是普通单任务重试。首次请求按稳定哈希分配渠道，可重试请求轮换到另一渠道。

### 7.2 稳定任务 ID 与指纹

任务 ID 示例：

```text
gt_simplify::g0048
pred_simplify::llmsr::g0048::s521::clean
equivalence::llmsr::g0048::s521::clean
stab_structure::llmsr::g0048::s520-s521
```

任务指纹为：

```text
evaluation_key = sha256(
  task_type
  + logical_id
  + prompt_version
  + schema_version
  + requested_model
  + effort
  + normalized_input
  + deterministic_evidence_hash
)

attempt_id = evaluation_key + ".a01"
```

prompt、schema、表达式、证据、模型或 effort 任一变化都会产生新版本，不能伪装成旧任务的 retry。

### 7.3 严格 JSON Schema

使用三个版本化 Schema，均设置 `additionalProperties: false`，只要求简短理由，不请求思维链。

`simplify.v1`：

- `outcome`: `simplified`、`unchanged` 或 `unable`；
- `simplified_expression`；
- `equivalence_assessment`；
- `assumptions`；
- `confidence`；
- `brief_reason`。

`equivalence.v1`：

- `decision`: `equivalent`、`not_equivalent` 或 `undetermined`；
- `evidence_basis`；
- `assumptions`；
- `confidence`；
- `brief_reason`。

`structure.v1`：

- `decision`: `mathematically_equivalent`、`same_canonical_structure`、`different_structure` 或 `undetermined`；
- `confidence`；
- `brief_reason`。

正式布尔值由程序根据枚举派生，避免模型同时返回互相冲突的 enum 和 boolean。合法的 `undetermined` 是有效终态，保守映射为 `Eq=0` 或结构不一致，不能因为答案不理想而重试。

### 7.4 结果验收与冻结

“首个有效结果”必须同时满足：

- Claude 进程和外层 JSON 成功；
- `structured_output` 存在；
- 严格 Schema 通过；
- model、effort、tools、turn、session 契约通过；
- 返回公式可解析，且只使用允许的变量和函数；
- claimed simplification 没有确定性反例；
- 没有额外字段、提示词泄漏或截断。

第一个 contract-valid 终态立即冻结。重启、重复调度或人工再次运行都不得重新调用。禁止对多个有效回答挑选最有利结果。

## 8. 失败重试、断点续跑与熔断

### 8.1 可重试失败

- 进程超时；
- HTTP 408/429 或临时 5xx；
- HTTP 传输错误，但不包括认证失败或确定性的请求契约错误；
- 非流式响应为空或被截断；
- 外层 JSON 无效；
- 缺失 `structured_output`；
- 严格 Schema 不通过；
- 返回的化简公式不可解析；
- 返回公式使用未声明变量或不支持函数；
- 确定性反例表明 claimed simplification 改变了原公式。

重试使用指数退避和 jitter。正式请求当前单次硬超时为 300 秒；首次失败后最多再重试两次，并轮换 API 渠道。

### 8.2 不重试状态

- 源公式缺失；
- 依赖任务缺失或已经 exhausted；
- Schema 合法的 `unable` 或 `undetermined`；
- `prompt_too_long`；相同传输请求不得原样重试，必须先修订并审计 prompt-only 投影；
- 任务已经有 frozen 结果；
- 全局物理尝试预算耗尽；
- 契约不一致导致全局熔断，等待修复后再恢复。

judge 基础设施失败不能当作算法失败记 0。只要正式 clean 任务仍有 evaluator failure，六轴榜单就不能发布；噪声任务未完成则在补充报告中明确标为不完整。

### 8.3 重试优先级

共享重试池按以下优先级调度：

1. Ground Truth 化简；
2. clean 最终公式化简；
3. clean 等价裁决；
4. clean STAB 结构裁决；
5. noisy 最终公式化简；
6. noisy 等价裁决。

优先级只影响调度，不改变输入、判断或分数。

### 8.4 持久化状态

采用 SQLite WAL 作为控制面、不可变 JSON 文件作为审计面：

- `tasks`：逻辑 ID、依赖、优先级、输入哈希和状态；
- `attempts`：预算预占、lease、起止时间、错误分类、usage、延迟和成本；
- `frozen_results`：每个 `evaluation_key` 唯一；
- `events`：只追加状态转换。

worker 在事务中领取任务。进程崩溃后，只有 lease 过期的 `running` 任务可以恢复。启动时只恢复 `pending`、`retry_wait` 和 expired `running`；`frozen`、`non_applicable`、`exhausted` 永不自动重跑。

每次调用结束立刻落盘原始 request、stdout、stderr、structured output、验证结果和元数据，不能等一个大批次结束后才 checkpoint。

## 9. Canary 与并发爬坡

全量队列前依次执行：

1. 无网络的 unit/mock 测试；
2. 生成完整 dry-run manifest 和依赖图；
3. 按任务家族、算法、表达式长度、保护算子和三种判断类型抽取真实 canary；
4. 初始并发固定为 4；
5. 校验 Schema 成功率、实际模型、`xhigh`、零工具调用、延迟、token 和成本；
6. 429/5xx 与 p95/p99 稳定后，再逐级升到 8、16；
7. 持续限流时自动降并发、退避或熔断；
8. canary 中首个有效结果直接冻结并复用，不浪费调用。

canary 报告未通过前，不启动付费全量任务。

此前一次短 prompt smoke 的观测为约 55.893 秒、USD 0.0294025。它只能粗略占位：

- 15800 次约 USD 464.56、串行 245.3 小时；
- 23700 次约 USD 696.84、串行 368.0 小时；
- 理想并发 8 约 30.7 至 46.0 小时，不含限流退避；
- 理想并发 16 约 15.3 至 23.0 小时，不含限流退避。

全量前必须用分任务类型的 canary 统计替换该估算；以上数字不是费用或工期承诺。

## 10. 符号指标组装

### 10.1 Ground Truth

先完成并冻结全部 50 条 GT 化简。GT 化简若与原公式存在确定性反例，按同一 `evaluation_key` 重试。若重试耗尽，阻断该 dataset 的所有下游符号任务，不能静默切换到另一个 reference。

### 10.2 SYM 与 MIN

6750 个逻辑运行中每个可用最终公式都接受化简和等价裁决。4500 条 noisy 结果只形成补充产物；只有 2250 条 clean 结果进入正式 SYM/MIN。未替换 final 的运行复用原始冻结公式和既有依赖结果；`final_and_eff` replacement 才重建 final 下游链，`eff_only` replacement 不触发新的公式化简。

clean 组装规则：

- 精确等价信用来自冻结的 Claude decision；
- `undetermined` 保守映射为不等价；
- partial SYM 使用冻结的确定性 tree、variable、operator 证据；
- MIN 复用同一 Claude 化简结果和唯一 canonical node-count 实现；
- 算法本身未给出有效公式时按指标文档记分；
- evaluator 基础设施失败不得伪装成算法失败。

### 10.3 STAB

输出 750 行 `(algorithm, dataset_id)` 级结果。Numerical Consistency 使用三个 clean seed 的质量，Validity 使用有效 seed 比例，Structural Consistency 使用三个 seed pair。

两个 seed 都有有效最终公式时，执行一次强制结构裁决。任一 seed 无效时，该 pair 明确记为不一致且不伪造 Claude 调用。最终严格使用 `(NVC)^(1/3)`，不施加旧版性能修正。

## 11. 产物目录

后续实现按以下结构落盘：

```text
stage5_metric_calculation_0831/
  SymbolicArena_SixAxis_Revised.md
  EXECUTION_PLAN.md
  config/
    metric_contract.json
    prompts/
    schemas/
  manifests/
    source_runs.csv
    ground_truth.csv
    llm_tasks.jsonl
  source_snapshot/
    results/
    trajectories/
    result_index.csv
    progress_index.csv
    checksums.sha256
  control/
    state.sqlite3
  llm/
    attempts/
    frozen/
  derived/
    run_metrics_all_conditions.csv
    clean_run_metrics.csv
    task_stability.csv
    algorithm_six_axis.csv
    noise_supplement.csv
  reports/
    source_preflight.json
    trajectory_coverage.json
    canary.json
    retry_budget.json
    llm_coverage.json
    conflict_audit.json
    final_audit.json
  figures/
```

大体积原始源文件和 attempt 产物可以不进 Git，但其 manifest、checksum、prompt、schema、代码、测试和紧凑报告必须版本化。

## 12. 测试与硬验收门

### 12.1 单元与性质测试

- `phi` 端点、clip、NaN/Inf 和 invalid；
- seed-level 平均与被禁止的 median-first 逻辑；
- parser 规范化与保护算子语义；
- canonical node count、tree distance 和空集合 F1；
- EFF checkpoint 归一化、去重、填充、`q*=0` 和遥测缺失；
- STAB 的三个 seed pair、invalid seed 和几何平均；
- 三个严格 Schema 和后置校验器；
- 重试分类、预算原子预占、lease 恢复、幂等与预算耗尽；
- 任一 contract 输入变化后，任务指纹必须失效。

### 12.2 发布前硬门

- 6750 条 hybrid 来源网格完整，`368` 个 replacement 精确覆盖且无重复，所有输入 checksum 已冻结；
- 来源集合中不存在 `all_15alg_fullcpu_v1_formal` 的任何 state、result、progress 或 partial output；
- 50 条 GT reference 全部通过；
- 每条 clean final result 存在，或有源层面的算法无效记录；
- 2250 条 clean EFF 轨迹均能从原始冻结轨迹与 `225` 个 clean replacement 的闭合合成按 adapter 契约重建；
- 每条已接纳快照都有可验证的任务身份；QLattice 不含重置前遗留分钟点；
- 所有必需 clean Claude 任务 accepted 或合法 non-applicable；
- 物理尝试总数不超过 23700；
- accepted 调用不存在 model、effort、stream、响应类型或渠道身份违规；
- 新正式表不含 ROB、旧 OOD retention、median-first、EFF proxy 或旧 STAB 修正；
- 最终表恰好 15 个算法和 6 个正式轴；
- 每个分数单元都可反查来源 checksum、确定性证据和 Opus5 API task ID；
- 从冻结产物离线重跑聚合，输出 checksum 完全一致；
- clean 正式评测中 unresolved evaluator failure 为 0。

除非另有独立指标定义，不擅自发明六轴综合分或总排名。

## 13. Goal 完成标准

后续 Goal 只有满足以下条件才能标记完成：

1. 冻结并审计全部必需输入；
2. 独立实现新六轴及测试，不使用旧指标代理；
3. 按固定非流式单轮 API 契约执行全部强制 Opus5 任务；
4. 物理尝试始终不超过 23700，并保留每次 attempt；
5. 使用 clean 结果计算 ID、OOD、SYM、MIN、EFF、STAB；
6. 将 noisy 产物单独作为补充诊断；
7. 生成可复现表格、覆盖率、重试、成本、冲突和来源报告；
8. 通过全部硬验收门；
9. 按仓库规范提交紧凑代码和报告。

## 14. 可直接使用的 Goal Objective

```text
在 AAAI_experiments/stage5_metric_calculation_0831 内，严格依据
SymbolicArena_SixAxis_Revised.md 和 EXECUTION_PLAN.md，以 AAAI Stage 4 原始未受影响结果
为基底，叠加已完成的 368 个针对性重跑，冻结并审计 6750 个逻辑 final 输入、50 条
Ground Truth 以及 2250 条 clean 运行的 180 分钟轨迹；明确排除并禁止消费已中止的
all_15alg_fullcpu_v1_formal 批次。
实现并执行新的 ID/OOD/SYM/MIN/EFF/STAB 计算管线。所有文档规定的大模型化简与
裁决都必须使用 `claude-opus-5`、`xhigh`、`stream=false` 的双渠道 Anthropic Messages API
独立单轮请求；逻辑任务最多 15800，物理尝试硬上限 23700，每任务首次加最多两次
失败重试，并支持幂等断点续跑和完整审计。不得复用旧六轴代理定义，不得用缺失轨迹
伪造 EFF。最终产出可复现的 15 算法六轴表、任务和运行级明细、补充噪声诊断、覆盖率、
重试、成本、冲突、来源校验报告、必要图表与自动化测试，并按仓库约定提交。
```
