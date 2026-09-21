# 内置算法源码清单

本目录下的第三方算法源码由主仓库直接跟踪，不再使用 Git 子模块。
更新某个上游时，应在独立临时克隆中检出目标提交，只同步该提交跟踪的文件，
保留许可证，并更新下表。不得同步上游 `.git/`、缓存、构建产物或本地实验结果。

| 算法 | 内置目录 | 上游仓库 | 冻结提交 |
| --- | --- | --- | --- |
| DrSR | `drsr_wrapper/drsr` | `https://github.com/scientific-intelligent-modelling/drsr.git` | `0eb4b0b824e2069616c78004c190627022fbc92e` |
| DSO | `dso_wrapper/dso` | `https://github.com/scientific-intelligent-modelling/deep-symbolic-optimization.git` | `2ff3ece4ce0494ae5b55c1fc44ef27d46f03a24a` |
| E2ESR | `e2esr_wrapper/e2esr` | `https://github.com/scientific-intelligent-modelling/e2e-symbolic-regression.git` | `7d370b0f779397d74ff9ef3be994738805fe2ecf` |
| iMCTS | `iMCTS_wrapper/MCTS-4-SR` | `https://github.com/scientific-intelligent-modelling/MCTS-4-SR.git` | `054929943c764c635a95d68da9d115f3a41bc366` |
| LLM-SR | `llmsr_wrapper/llmsr` | `https://github.com/scientific-intelligent-modelling/LLM-SR.git` | `8c0c41658f1dc6f99c62e3b8565309f0c1979829` |
| TPSR | `tpsr_wrapper/tpsr` | `https://github.com/scientific-intelligent-modelling/TPSR.git` | `8c67185b8fc58f471948c385a721ada6a11e7aa6` |

除 DrSR 上游快照未提供许可证文件外，其余内置源码均保留了上游仓库中的
`LICENSE`，DSO 还保留了 `NOTICE`。在确认 DrSR 的授权条款前，不应对外再分发其源码。
