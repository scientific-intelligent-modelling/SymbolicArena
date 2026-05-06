# Core50 vs Candidate200 observed iteration comparison

Note: this table reports the observed counters recoverable from archived process artifacts, not configured upper bounds. Counter units differ across algorithms and should not be compared directly across methods.

| algorithm | unit | Core50 n/with | Core50 mean | Core50 median | Candidate200 n/with | Candidate200 mean | Candidate200 median | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| gplearn | generation | 250/250 | 1,917.6 | 2,119.5 | 148/148 | 1,383.8 | 1,179.0 |  |
| pyoperon | generation | 250/246 | 1.00 | 1.00 | 148/148 | 1.00 | 1.00 | generation is always 1; the wrapper likely exposes only the final generation index. This is useful for coverage checks but weak as an iteration signal. |
| pysr | N/A | 250/0 | N/A | N/A | 148/0 | N/A | N/A | The archived artifact does not contain an actual niterations/iteration counter; hall_of_fame cannot recover the true iteration count. |
| dso | iteration | 250/250 | 1,644.2 | 2,000.0 | 167/164 | 366.14 | 364.00 |  |
| tpsr | N/A | 250/0 | N/A | N/A | 148/0 | N/A | N/A | The current wrapper records only the final source/stage, not search iterations. |
| llmsr | llm_proposal_iteration | 250/244 | 91.90 | 91.00 | 200/196 | 96.31 | 95.50 |  |
| drsr | llm_proposal_iteration | 250/250 | 11.45 | 7.00 | 200/200 | 22.70 | 9.00 |  |
| e2esr | N/A | 250/0 | N/A | N/A | 200/0 | N/A | N/A | current wrapper note stage/refinement_type note  |
| imcts | evaluations | 250/249 | 123,824.2 | 67097 | 200/198 | 211,120.7 | 206,881.0 |  |
| qlattice | epoch | 250/245 | 100.00 | 100 | 200/196 | 100.00 | 100.00 |  |
| ragsr | generation_iteration | 250/250 | 83.54 | 100.00 | 200/10 | N/A | N/A | The legacy Candidate-200 RAGSR wrapper mostly did not record iterations; only 10 zero-valued rows exist and are not treated as a valid average. |
| udsr | iteration | 250/250 | 194.64 | 190.00 | 200/200 | 55.31 | 18.50 |  |

Known limitations: host_22 was unavailable; the legacy Candidate-200 seven-algorithm process counters are not complete for all 200 tasks; LLMSR/DRSR use the semantic-200 physics-v2 iteration summary; PySR/E2ESR/TPSR currently have no recoverable observed-iteration field.
