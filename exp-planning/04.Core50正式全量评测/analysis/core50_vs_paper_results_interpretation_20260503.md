# Core50 结果与相关论文口径差异解读

生成时间：2026-05-03

## 结论

当前 Core50 结果和原论文结果明显不同是预期现象，不应直接解释为“我们的实现一定错了”或“论文结果不可靠”。主要原因是评测对象、数据选择、指标、预算、输入语义和实现口径都不一致。

当前 Core50 更接近一个 **contrastive OOD stress benchmark**：

- 数据集不是论文原始全集，而是从候选池中筛出的高区分度 Core50。
- 指标主口径是 `OOD log10 NMSE`，缺失/爆炸按 `12` 惩罚。
- 每个算法跑 `50 datasets × 5 seeds`，统一 wall-clock 预算约 `3600s`。
- LLM 相关算法需要物理语义输入，但当前仍要区分旧结果、语义结果和正在重跑的 base/turbo 结果。
- Core50 中 `llm-srbench + srsd = 31/50`，明显偏向反记忆、OOD、dummy-variable 和高难度任务。

因此，这批结果更适合支撑论文中的一句话：

> Core50 不是复现各算法论文 leaderboard，而是在统一工具链、统一 OOD 指标、统一预算下刻画不同 SR 方法的失败模式和泛化差异。

## 当前 Core50 主要结果

来源：

- `generated/core50_12alg_5seed_final_results/analysis/core50_12alg_leaderboard_ood_log_nmse_20260503.md`
- `generated/core50_12alg_5seed_final_results/analysis/core50_12alg_dataset_algorithm_median_log_nmse_20260503.csv`
- `generated/core50_12alg_5seed_final_results/analysis/core50_probe4_previous_vs_current_ranking_20260503.md`

当前总榜，按 dataset×algorithm 的 5 seed OOD log10 NMSE 中位数，再跨 50 个数据集取均值，越低越好：

| rank | algorithm | OOD mean | OOD median | metric complete | wins |
|---:|---|---:|---:|---:|---:|
| 1 | `udsr` | -5.646 | -4.737 | 1.000 | 21 |
| 2 | `imcts` | -3.960 | -7.812 | 0.968 | 5 |
| 3 | `pysr` | -3.055 | -2.678 | 0.960 | 9 |
| 4 | `dso` | -2.458 | -0.632 | 0.972 | 3 |
| 5 | `drsr` | -2.087 | -0.706 | 1.000 | 7 |
| 6 | `llmsr` | -1.420 | -0.842 | 0.992 | 2 |
| 7 | `gplearn` | -1.207 | -0.260 | 1.000 | 0 |
| 8 | `qlattice` | -0.780 | -1.038 | 0.972 | 0 |
| 9 | `tpsr` | 0.896 | -0.000 | 0.996 | 1 |
| 10 | `pyoperon` | 1.079 | -0.075 | 0.840 | 1 |
| 11 | `e2esr` | 2.891 | 0.136 | 0.800 | 1 |
| 12 | `ragsr` | 3.788 | 2.852 | 0.964 | 0 |

关键观察：

- `udsr / imcts / dso / pyoperon` 在上一轮 664 三种子切到 Core50 和当前 5 seed 重跑中排序完全一致：`udsr > imcts > dso > pyoperon`，Spearman = 1.0。
- `llmsr` 和 `drsr` 目前的旧 Core50 结果只能作为临时结果，因为用户已要求用物理语义、base/turbo 模型分流重跑。
- `pyoperon` 当前 median runtime 约 10s，且 generation 字段几乎全为 1，需要单独审计 wrapper 是否真的按预期用满预算。
- `ragsr` 当前表现很差，和 RAG-SR 原论文口径差异极大。它必须单独审计是否为官方算法语义，而不是简化版或错误配置。

## Core50 数据构成

Core50 family 分布：

| family | count |
|---|---:|
| `llm-srbench` | 16 |
| `srsd` | 15 |
| `srbench1.0` | 9 |
| `nguyen` | 3 |
| `srbench2025` | 2 |
| `korns` | 2 |
| `keijzer` | 2 |
| `vladislavleva` | 1 |

解释：

- 这不是 SRBench 原始 252 问题的均匀子集。
- 这也不是 LLM-SRBench 原始 239 问题的完整评测。
- 这是从 200 candidate 中进一步筛出的 50 个高信息量任务。
- `llm-srbench` 和 `srsd` 占比高，天然会压低依赖训练分布或记忆先验的模型表现。

## 10 篇核心论文口径梳理

### 1. SRBench

论文/项目：

- La Cava et al., 2021, `Contemporary Symbolic Regression Methods and their Relative Performance`
- SRBench 官方页面：https://cavalab.org/srbench/

论文/项目口径：

- SRBench 是 living benchmark，目标是统一比较现代 SR 方法。
- 官方页面说明当前 benchmark 包括多种 SR 方法，并进行独立重复运行。
- 数据分为 black-box problems 和 ground-truth problems。
- SRBench 结果页强调的指标包括 test R2、模型复杂度、训练时间、accuracy-complexity-time tradeoff、symbolic equivalence、以及 `test R2 > 0.999` 的 accuracy solution rate。

与当前结果的差异：

- 我们当前主指标是 OOD log10 NMSE，不是 test R2 或 symbolic equivalence。
- 我们的 Core50 是筛选后的 stress subset，不是 SRBench 原始 252 个问题。
- SRBench 报告通常跨大量数据集和多次重复聚合；我们当前是 50 个任务、5 seeds、强 OOD 惩罚。

对当前结果的解释：

- 不能用 SRBench 的“某算法在 SRBench 排名靠前”直接推断它在 Core50 OOD log NMSE 上也靠前。
- `uDSR` 和 `iMCTS` 在我们 Core50 上强，和它们在 SRBench 类问题上强并不冲突。
- `PyOperon` 在当前结果偏弱，和 SRBench/Operon 的强势可能冲突，需要优先审计参数和 wrapper，而不是直接写成算法能力差。

### 2. SRSD

论文：

- Matsubara et al., `Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery`
- OpenReview PDF：https://openreview.net/pdf?id=oKwyEqClqkb

论文口径：

- 论文指出原 Feynman/FSRD 类数据会高估方法能力，因此构造更难的 SRSD。
- SRSD 结果显示 common baselines 在 SRSD 上 solution rate 明显下降。
- 论文表中给出的 SRSD solution rate 例子：`gplearn` 约 1.67%，`AI Feynman` 约 9.17%，`DSR` 约 15.0%。
- 论文还说明每个 dataset×method job 可以跑到 24 小时，并且每个 pair 可跑最多 100 个不同超参数训练 session。

与当前结果的差异：

- 我们对每个 run 用 3600s，不做 100 组超参数 session 搜索。
- 我们用 OOD log NMSE，不用 solution rate 作为主指标。
- 我们 Core50 中包含 15 个 SRSD，而且含 dummy / hard / medium 组合。

对当前结果的解释：

- `gplearn`、`E2ESR`、`TPSR` 在 SRSD 上弱是合理的，不一定是实现问题。
- `iMCTS` 和 `uDSR` 在 SRSD 上强，可能说明它们更适合高难度搜索和组合式结构。
- 如果要和 SRSD 论文对齐，必须额外输出 SRSD-style 指标：easy/medium/hard 分组的 `solution rate` 和 `R2 > 0.999`。

### 3. LLM-SRBench

论文：

- Shojaee et al., 2025, `LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models`
- arXiv：https://arxiv.org/abs/2504.10415

论文口径：

- LLM-SRBench 包含 239 个问题，跨 4 个科学领域。
- 它专门针对 LLM equation discovery 设计，用 LSR-Transform 和 LSR-Synth 降低 trivial memorization。
- 论文报告最佳系统 symbolic accuracy 也只有 31.5%。

与当前结果的差异：

- 我们只用了其中 16 个进入 Core50，而且这些是经筛选后的高区分度样本。
- 我们评估的是数值 OOD NMSE，不是 symbolic accuracy。
- 当前 LLM-SR/DrSR 旧结果还涉及是否注入物理语义的问题，正在重跑。

对当前结果的解释：

- `LLM-SRBench` 的设计目标就是让 LLM 难以靠记忆取胜，所以 LLM 类算法在这部分不一定占优。
- `uDSR` 在 `llm-srbench` family 上表现强，可能说明这一批任务更偏数据驱动结构搜索，而不是自然语言先验。
- LLM 方法弱不能直接说明 LLM-SR/DrSR 无效，需要等有物理语义的重跑结果完成后再判断。

### 4. LLM-SR

论文：

- Shojaee et al., 2025 ICLR Oral, `LLM-SR: Scientific Equation Discovery via Programming with Large Language Models`
- arXiv：https://arxiv.org/abs/2404.18400

论文口径：

- LLM-SR 把方程视为程序，利用 LLM 科学先验提出 equation skeleton，再优化参数。
- 论文强调使用 domain knowledge，并在少量跨领域 benchmark 上展示 OOD 优势。
- 论文评测对象是 carefully designed scientific discovery problems，不是我们这种高区分度 Core50 混合池。

与当前结果的差异：

- 我们的 Core50 包含大量变换、dummy、SRSD 和非自然语言友好的任务。
- 如果 prompt 没有给物理语义或变量含义，LLM-SR 的核心优势会被削弱。
- 我们使用的是 DeepInfra Llama 3.1 8B 系列，不一定等同论文设置。

对当前结果的解释：

- LLM-SR 在当前旧 Core50 总榜居中偏后，不足以否定论文结论。
- 更合理的判断方式是比较三组：无语义旧结果、有语义结果、base/turbo 重跑结果。
- 如果有语义重跑后仍然弱，说明 Core50 主要在测数据驱动泛化和反记忆，而不是 LLM scientific prior。

### 5. DrSR

论文：

- Wang et al., 2025, `DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience`
- arXiv：https://arxiv.org/abs/2506.04282

论文口径：

- DrSR 试图弥补 LLM-SR 过度依赖 internal prior 的问题。
- 它加入 data-driven insight，比如单调性、非线性、相关性分析。
- 它还加入 performance feedback 和 reflective loop。
- 论文声称在 physics、chemistry、biology、materials 等数据上提升 valid equation rate、accuracy、generalization 和 search efficiency。

与当前结果的差异：

- 当前旧 Core50 中 DrSR proposal iteration 中位数约 7，平均约 11.45，明显低于 LLM-SR 约 91 轮。
- 这可能是 DrSR 搜索设计差异，也可能是 wrapper/early-stop/配置差异。
- DrSR 论文强调 dual reasoning，但我们的 prompt 和 artifact 是否完整包含数据分析与经验反馈，需要单独核查。

对当前结果的解释：

- DrSR 目前排名略高于 LLM-SR，但未表现出论文里那种明显统治性。
- 在新重跑完成前，不应把 DrSR 的旧结果作为最终结论。
- 需要审计 DrSR 的每轮反馈、经验 buffer、prompt 结构和参数优化是否与论文一致。

### 6. DSO / DSR

论文：

- Petersen et al., 2021 ICLR, `Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients`
- arXiv：https://arxiv.org/abs/1912.04871

论文口径：

- DSR 用 RNN controller 生成表达式分布。
- 使用 risk-seeking policy gradient 优化 best-case 表现，而不是普通期望回报。
- 论文在 Nguyen 等 benchmark 上强调 exact expression recovery。

与当前结果的差异：

- 我们当前对 DSO/DSR 用 OOD log NMSE，而非 exact recovery。
- DSO 在当前 Core50 中间偏强，符合“训练型搜索方法能稳定探索”的预期。
- 但 DSO 对 OOD extrapolation 未必天然强，尤其面对 dummy、变换和高复杂度表达式。

对当前结果的解释：

- DSO 排第 4 是合理信号，不明显异常。
- 如果要和论文对齐，应额外报告 Nguyen subset 的 exact recovery 或 `R2 > 0.999`。

### 7. uDSR

论文：

- Landajuela et al., 2022 NeurIPS, `A Unified Framework for Deep Symbolic Regression`
- NeurIPS：https://papers.neurips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html

论文口径：

- uDSR 组合五类策略：recursive simplification、neural-guided search、large-scale pre-training、genetic programming、linear models。
- 论文称其在 SRBench 上对 symbolic recovery、accuracy 和 accuracy-complexity tradeoff 达到 SOTA。
- 该方法不是单一 DSO，而是混合框架。

与当前结果的差异：

- 我们当前 wrapper 名为 `udsr`，需要确认是不是 full uDSR 论文级实现，还是 DSO + LINEAR/poly + GP-meld 的工程近似版。
- 即使是近似版，它在 664 probe4 和 Core50 5 seed 上排序稳定为第 1，说明当前工具链下确实很强。

对当前结果的解释：

- `uDSR` 当前排第 1 并不违背论文直觉。
- 但 paper 中如果写“uDSR 复现论文 full method”，必须非常谨慎，需要列明我们实现了哪些模块，没实现哪些模块。

### 8. iMCTS / Deep Generative SR with MCTS

论文：

- Kamienny et al., 2023 ICML, `Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search`
- arXiv：https://arxiv.org/abs/2302.11223

论文口径：

- 该方法结合神经生成模型和 MCTS。
- 论文动机是纯 neural SR 在 OOD datasets 上缺少搜索能力，而 MCTS 能补足 search。
- 论文声称在 SRBench 上达到强性能。

与当前结果的差异：

- 当前 Core50 的主指标正是 OOD log NMSE，这正好有利于“neural + search”的设计。
- iMCTS 在我们结果中 OOD median 很强，但 OOD mean 低于 uDSR，说明它可能在部分任务上极强，同时仍有若干失败长尾。

对当前结果的解释：

- iMCTS 排第 2 是合理的。
- 需要进一步看 per-family：它在 SRSD、Nguyen、Keijzer 等 family 很强，但在 LLM-SRBench 上不是稳定第一。

### 9. TPSR

论文：

- Shojaee et al., 2023 NeurIPS, `Transformer-based Planning for Symbolic Regression`
- arXiv：https://arxiv.org/abs/2303.06833

论文口径：

- TPSR 在 transformer decoding 中加入 MCTS/planning。
- 论文强调能够把 fitting accuracy、complexity 等非可微反馈注入生成过程。
- TPSR 的强项来自预训练 transformer 与 planning 的结合。

与当前结果的差异：

- 我们当前集成曾出现固定变量词表和非法变量过滤问题，说明 wrapper 口径对它影响很大。
- TPSR/E2ESR 这类预训练模型高度依赖训练分布、变量维度、词表、算子和输入采样。
- Core50 中大量 LLM-SRBench transform、SRSD dummy 和 OOD split，可能远离预训练分布。

对当前结果的解释：

- TPSR 当前排名第 9，不一定说明论文错，更可能说明当前 Core50 是强 OOD stress。
- 如果要公平引用 TPSR，需要另做 TPSR-paper-style sanity check：使用论文原始数据、官方模型、官方采样和官方指标。

### 10. E2E Transformer SR

论文：

- Kamienny et al., 2022 NeurIPS, `End-to-end Symbolic Regression with Transformers`
- NeurIPS：https://proceedings.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html

论文口径：

- E2E-SR 直接预测完整表达式，包括常数。
- 论文指出它在 SRBench 上接近 state-of-the-art GP，同时推理速度快几个数量级。
- 它的强项是训练分布内的快速预测，不是无限制搜索。

与当前结果的差异：

- 我们当前用的是高难度 Core50，包含超出训练分布的 OOD 和变换任务。
- E2ESR 当前 metric complete 只有 0.800，说明有明显集成或适配问题。
- E2ESR 在 SRSD family 上 OOD mean 很差，和预训练模型 OOD 脆弱性一致。

对当前结果的解释：

- E2ESR 当前差是合理风险信号，但要区分“模型 OOD 失败”和“wrapper 没完全按官方口径跑”。
- 这类方法需要按 paper dataset 做单独 sanity reproduction。

## 附加相关论文/方法口径

### PySR

论文：

- Cranmer, 2023, `Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl`
- arXiv：https://arxiv.org/abs/2305.01582

PySR 使用多种群 evolutionary search，包含 evolve-simplify-optimize loop 和常数优化。当前 Core50 上 PySR 排第 3，整体合理。它在 `srbench1.0` 和 Nguyen 类问题上很强，但在 `llm-srbench` family 上均值被少数失败任务拉高。

### RAG-SR

论文：

- Zhang et al., 2025 ICLR, `RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression`
- ICLR：https://proceedings.iclr.cc/paper_files/paper/2025/hash/19a8e70828c01059631f913442ae31e6-Abstract-Conference.html

RAG-SR 原论文不是简单“外部 LLM + 文档 RAG”。它强调在线监督学习的 language model、evolutionary feature construction、retrieval searched symbolic expressions 和 scale-invariant augmentation。当前工具集中的 RAG-SR 如果只是轻量近似，结果不能直接代表论文方法。

### PyOperon / Operon

论文/实现：

- Burlacu et al., 2020, `Operon C++: an efficient genetic programming framework for symbolic regression`
- PyPI：https://pypi.org/project/pyoperon/

Operon 在 SRBench 系列里通常是强基线。当前 PyOperon 结果偏弱，同时 median runtime 约 10s、generation 字段几乎全为 1，优先怀疑运行口径或日志字段，而不是先否定算法。

### gplearn

gplearn 是 scikit-learn 风格 GP 实现，常作为基础 baseline。SRSD 论文中 gplearn 在 SRSD 上 solution rate 很低，所以当前 gplearn 在 Core50 中偏中下是合理的。

## 为什么当前结果和论文结果不一样

### 原因 1：数据集不是同一个分布

论文通常在自己的 benchmark 全集或特定任务上报告：

- SRBench：ground-truth + black-box，总体分布宽。
- SRSD：按 easy/medium/hard 设计的 scientific discovery 数据集。
- LLM-SRBench：239 个反记忆任务。
- LLM-SR：少量 carefully designed interdisciplinary problems。

我们当前：

- 从 800+ 进入 664，再经 probe 和 candidate 筛选得到 Core50。
- Core50 不是代表“所有任务平均水平”，而是代表“高信息量、高区分度、强 OOD 压力”。

这会天然改变算法排序。

### 原因 2：指标不同

论文常见指标：

- exact symbolic recovery
- symbolic accuracy
- test R2
- `R2 > 0.999` solution rate
- complexity / Pareto frontier

当前主指标：

- OOD log10 NMSE
- 缺失、NaN、Inf、不可评估统一惩罚到 12
- 先 seed median，再 dataset mean

这会惩罚 OOD 爆炸和不稳定表达式，也会让“偶尔精确恢复但长尾失败”的方法排名下降。

### 原因 3：预算和超参搜索不同

SRSD 论文中每个 dataset×method 可运行到 24h，并且最多跑 100 个超参数 session。我们是每 run 3600s，且固定 benchmark 默认参数。

所以当前实验更像“统一预算下的实用评测”，不是“每个算法论文的最佳调参结果”。

### 原因 4：LLM 方法强依赖语义输入和模型版本

LLM-SR 和 DrSR 的论文优势来自：

- domain prior
- problem description
- variable semantics
- LLM 代码生成能力
- feedback/reflection loop

如果变量匿名、dummy 顺序隐藏、prompt 语义不足，或者 LLM 模型不是论文所用模型，性能会显著变化。当前用户已经要求重跑 `llmsr/drsr`，使用 base/turbo 各半且打开物理语义，这是必要修正。

### 原因 5：预训练模型对分布外数据敏感

E2ESR、TPSR 这类模型强依赖合成训练分布。Core50 中的 LLM-SRBench transform、SRSD dummy、OOD split、真实科学小样本，很可能超出其训练分布。

这解释了它们在当前 Core50 上不如论文展示结果。

### 原因 6：部分 wrapper 可能还不是论文级复现

当前需要重点审计：

- `pyoperon`：是否真的使用了预期预算、代数、种群和优化参数。
- `ragsr`：是否复现官方 RAG-SR 的 online LM + retrieval + feature construction，而不是简化近似。
- `e2esr/tpsr`：预训练权重、变量词表、输入点采样、常数 refinement 是否和官方一致。
- `udsr`：是否 full uDSR，还是工程近似版。

## 推荐的下一步验证

### A. 不要直接重写 Core50 结论

当前结果可以先这样表述：

> Under a unified 3600s OOD stress setting on Core50, hybrid search-based methods dominate, while LLM- and pretraining-heavy methods are more sensitive to semantic prompt fidelity and distribution shift.

不要表述成：

> 我们证明某算法比论文里说的弱。

### B. 增加 paper-style sanity reproduction

为每类论文做最小对齐复现：

- SRBench-style：输出 exact symbolic equivalence、`R2 > 0.999`、complexity、runtime。
- SRSD-style：按 easy/medium/hard 报 solution rate。
- LLM-SRBench-style：按 symbolic accuracy 或公式等价判定。
- LLM-SR/DrSR-style：只在有清晰物理语义的任务上比较无语义 vs 有语义。

### C. 当前 Core50 paper 应该报告双口径

主口径：

- OOD log10 NMSE
- valid/complete rate
- runtime
- complexity
- train/valid/id/ood 分解

辅助口径：

- `R2 > 0.999`
- symbolic equivalence / formula recovery
- per-family breakdown
- leave-one-family-out
- 去掉 SRSD / 去掉 LLM-SRBench 后排序是否稳定

### D. 优先审计四个异常点

1. `pyoperon`：median runtime 约 10s，不像完整 3600s 搜索。
2. `ragsr`：当前结果不能代表 ICLR 2025 RAG-SR，需核查官方语义。
3. `llmsr/drsr`：等待有物理语义 base/turbo 重跑结果。
4. `e2esr/tpsr`：核查官方预训练权重、输入点数、变量维度、常数 refinement。

## 当前可写入论文的方法学解释

可以写：

> Prior SR benchmarks report performance under heterogeneous objectives such as symbolic recovery, high-R2 solution rate, and accuracy-complexity trade-offs. In contrast, our Core50 is constructed as an OOD-focused, contrastive benchmark under a unified wall-clock budget and artifact protocol. Therefore, its leaderboard should not be interpreted as a reproduction of any individual paper's benchmark table. Instead, it measures whether methods retain robust, finite, and generalizable equations under a deliberately selected set of high-information tasks.

对应中文解释：

> 这不是复现各论文 leaderboard 的实验，而是统一工程约束下的 Core50 OOD 压力测试。结果不同是合理的，但需要通过 paper-style sanity reproduction 和 per-family breakdown 来证明差异来自 benchmark 口径，而不是工具集错误。

