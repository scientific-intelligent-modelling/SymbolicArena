# Core50 vs Candidate200 observed iteration comparison

说明：这里统计的是归档过程文件中真实可恢复的 observed counter，不是配置里的上限。不同算法单位不同，不能跨算法直接比较。

| algorithm | unit | Core50 n/with | Core50 mean | Core50 median | Candidate200 n/with | Candidate200 mean | Candidate200 median | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gplearn | generation | 250/250 | 1,917.6 | 2,119.5 | 148/148 | 1,383.8 | 1,179.0 |  |
| pyoperon | generation | 250/246 | 1.00 | 1.00 | 148/148 | 1.00 | 1.00 | generation 字段全部为1，疑似 wrapper 只暴露最终代编号；可横向看覆盖，但信息量弱。 |
| pysr | N/A | 250/0 | N/A | N/A | 148/0 | N/A | N/A | 当前归档没有真实 niterations/迭代计数字段；hall_of_fame 不能还原实际迭代数。 |
| dso | iteration | 250/250 | 1,644.2 | 2,000.0 | 167/164 | 366.14 | 364.00 |  |
| tpsr | N/A | 250/0 | N/A | N/A | 148/0 | N/A | N/A | 当前 wrapper 只记录最终 source/stage，不记录搜索轮次。 |
| llmsr | llm_proposal_iteration | 250/244 | 91.90 | 91.00 | 200/196 | 96.31 | 95.50 |  |
| drsr | llm_proposal_iteration | 250/250 | 11.45 | 7.00 | 200/200 | 22.70 | 9.00 |  |
| e2esr | N/A | 250/0 | N/A | N/A | 200/0 | N/A | N/A | 当前 wrapper 只记录最终 stage/refinement_type，不记录搜索轮次。 |
| imcts | evaluations | 250/249 | 123,824.2 | 67097 | 200/198 | 211,120.7 | 206,881.0 |  |
| qlattice | epoch | 250/245 | 100.00 | 100 | 200/196 | 100.00 | 100.00 |  |
| ragsr | generation_iteration | 250/250 | 83.54 | 100.00 | 200/10 | N/A | N/A | Candidate200 的旧 RAGSR wrapper 基本未记录 iteration；仅10条为0，不作为有效平均。 |
| udsr | iteration | 250/250 | 194.64 | 190.00 | 200/200 | 55.31 | 18.50 |  |

已知限制：host_22 不可达，Candidate200 旧 7 算法的过程计数不是完整 200；LLMSR/DRSR 使用 semantic200 physics v2 的轮次 summary；PySR/E2ESR/TPSR 当前没有可恢复 observed iteration。
