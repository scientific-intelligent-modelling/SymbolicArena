# GPT-5.6-Sol 全量复核 Opus5 化简结果

## 范围与口径

- 来源：`Core50_final_20260914`
- 总对象：6800 条，其中 Ground Truth 50 条、三个条件的预测公式各 2250 条
- 实际送审：6700 条；另外 100 条 Opus5 `outcome=unable` 没有候选化简式，记为 `non_applicable`
- 模型：`gpt-5.6-sol`
- 接口：OpenAI Responses，`stream=false`、`store=false`
- 并发：32
- 提示词版本：`gpt56_full_opus_review.common_domain_approx.native_evidence.v1`
- 数值口径：共同有效实数域，`abs_tol=1e-9`、`rel_tol=1e-6`
- 通过条件：函数值近似一致、变量映射与算法算子语义不变，且候选式不比原式更复杂

91 条发生文本改写的 gplearn 记录额外绑定了原生 prefix、特征名、`Xk` 映射和 protected `div/log/sqrt` 语义。DSO 和 uDSR 明确按 `protected=false` 审查。

## 总结果

| 项目 | 数量 | 占全部 6800 条比例 |
|---|---:|---:|
| 通过 | 6600 | 97.0588% |
| GPT 判定失败 | 100 | 1.4706% |
| 无 Opus 候选，未送审 | 100 | 1.4706% |
| 合计未通过 | 200 | 2.9412% |

因此，本报告的主通过率是 `6600/6800 = 97.0588%`。若只考察具有候选化简式的 6700 条记录，条件通过率为 `6600/6700 = 98.5075%`；后者只作为补充，不再作为主结果。

3396 对逐字节相同的控制公式全部通过，未出现长表达式 identity 幻觉。3304 对实际发生文本变化，其中 3204 条通过、100 条失败，改写公式通过率为 96.9734%。

按 Opus outcome：

| Opus outcome | 审查数 | 通过 | 失败 |
|---|---:|---:|---:|
| simplified | 2852 | 2755 | 97 |
| unchanged | 3848 | 3845 | 3 |
| unable | 100 | 不适用 | 不适用 |

## 失败归因

100 条 GPT 失败票分为四类：

| 类别 | 数量 | 含义 |
|---|---:|---|
| `protected_operator_semantics` | 41 | gplearn 普通代数改写没有保持原生 protected operator 值语义 |
| `clear_algebraic_mismatch` | 2 | 非 gplearn 的明确代数变换错误，有具体反例 |
| `extreme_scale_decimal_amplification` | 7 | 极小十进制残差仅在极端尺度或极点附近被放大 |
| `equivalent_but_more_complex` | 50 | 函数等价，但所谓化简式的操作数更多 |

因此，100 条失败不能全部解释成“Opus 改错了函数”：严格按本次合同，真正被 GPT 判为数值/语义不等价的是 50 条，另外 50 条只是不满足 minimality。若采用用户提出的 benchmark-support 实用口径，7 条只依赖极端尺度放大的小数差应单列，不应与明确代数错误混合。

## gplearn

gplearn 共有 450 个对象，其中 92 条 Opus `unable`，358 条可审。真正发生文本改写且附原生 prefix 的是 91 条：

- 50 条通过；
- 41 条因 protected operator 语义失败；
- 失败率 `41/91 = 45.05%`。

此前发现的四条 `Korns-2 / feynman-ii.34.2a / II.34.2_1_0` 案例全部包含在这 41 条中。这个比例只描述“gplearn 被 Opus 改写的公式”，不能外推到 gplearn 的全部原始运行，更不能解释为 ID/OOD 错误；ID/OOD 使用原生 prefix replay，不使用 Opus 化简式。

## 算法概览

失败数较高的算法是：gplearn 41、DSO 15、E2ESR 14、TPSR 12。DSO 的 15 条全部属于等价但更复杂；E2ESR 的 14 条中 12 条只是更复杂、2 条是语义错误；TPSR 的 12 条中 7 条更复杂、5 条语义错误。

FePySR、JAXSR、SymbolFit 的所有可审记录均通过；50 条 Ground Truth 也全部通过。完整算法和条件分组见 `algorithm_summary.csv` 与 `condition_summary.csv`。

## 请求与费用

- 物理请求：6704 / 10050
- 成功裁决：6700
- 失败尝试：4；其中两次网络错误随后恢复，一条 HTTP-200 schema 失败在第三次显式恢复后完成
- 记录到的输入 token：9,197,919
- 记录到的输出 token：1,295,675
- 记录到的总 token：10,493,594
- 按输入 3 元/百万、输出 15 元/百万计算的参考费用：47.028882 元

四次失败尝试没有返回可记录的 token usage，因此真实账单可能略高。67 个 checkpoint 从 100 到 6700 连续齐全。

## 解释限制

这是一轮单模型全量复核，结论表示 GPT-5.6-Sol 在冻结提示词和证据下的判断，不等同于形式化证明，也不会自动修改六轴结果。下一步若要修订正式指标，应先对 `failure_analysis.csv` 中 50 条语义失败做算法原生 evaluator 的完整 benchmark-support 数值复放，再决定哪些 Opus 化简式应回退或重做。50 条“等价但更复杂”只影响 MIN 口径，不应当作预测正确性错误。

## 主要文件

- `FINAL_AUDIT_REPORT.json`：机器可读总报告
- `failure_analysis.csv`：100 条失败及逐条理由、反例和分类
- `gplearn_native_changed_review.csv`：91 条 gplearn 原生语义改写复核
- `algorithm_summary.csv`、`condition_summary.csv`：分组汇总
- `review_index.csv`：6800 条完整 inventory 状态
- `per_model/gpt-5.6-sol/tasks/`：6700 条逐请求裁决
- `checkpoints/`：每 100 条增量汇总
- `cost_ledger.csv`、`request_budget.json`：请求与 token 账本
- `SHA256SUMS`：完整文件校验
