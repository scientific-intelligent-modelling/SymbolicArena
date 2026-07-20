# E1 Repair Summary 2026-04-26

## Result

- Valid outputs after repair: `1385 / 1400`.
- `wrong_dataset_collision`: `0` remaining.
- Remaining invalid rows are exact dataset matches with empty expressions, non-finite predictions, or otherwise non-evaluable outputs under the 3600s E1 budget.

## Method Valid Outputs

| method | valid | total |
| --- | ---: | ---: |
| `llmsr` | 200 | 200 |
| `gplearn` | 200 | 200 |
| `drsr` | 200 | 200 |
| `tpsr` | 195 | 200 |
| `dso` | 198 | 200 |
| `pysr` | 197 | 200 |
| `pyoperon` | 195 | 200 |

## Remaining Nonvalid Rows

| method | dataset_id | dataset_name | timeout_type | reason |
| --- | --- | --- | --- | --- |
| `pysr` | `g0007` | `CRK22` | `partial_output` | TimeoutError("算法 'pysr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `pyoperon` | `g0012` | `feynman-i.26.2` | `not_timeout` | finite ID/OOD metrics missing after rerun/backfill |
| `tpsr` | `g0022` | `feynman-ii.11.20` | `partial_output` | TimeoutError("算法 'tpsr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `tpsr` | `g0025` | `feynman-bonus.4` | `no_valid_output` | TimeoutError("算法 'tpsr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `tpsr` | `g0039` | `feynman-ii.11.3` | `no_valid_output` | TimeoutError("算法 'tpsr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `pyoperon` | `g0078` | `III.21.20_2_0` | `not_timeout` | finite ID/OOD metrics missing after rerun/backfill |
| `dso` | `g0130` | `Keijzer-10` | `partial_output` | TimeoutError("算法 'dso' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `pysr` | `g0130` | `Keijzer-10` | `partial_output` | TimeoutError("算法 'pysr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `dso` | `g0141` | `Keijzer-15` | `partial_output` | TimeoutError("算法 'dso' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `pysr` | `g0143` | `MatSci8` | `partial_output` | TimeoutError("算法 'pysr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `tpsr` | `g0174` | `feynman-ii.35.18` | `partial_output` | TimeoutError("算法 'tpsr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `tpsr` | `g0175` | `feynman-bonus.2` | `partial_output` | TimeoutError("算法 'tpsr' 的子进程执行超时：action=fit, timeout_in_seconds=3600") |
| `pyoperon` | `g0184` | `III.15.14_1_0` | `not_timeout` | finite ID/OOD metrics missing after rerun/backfill |
| `pyoperon` | `g0193` | `II.21.32_3_0` | `not_timeout` | finite ID/OOD metrics missing after rerun/backfill |
| `pyoperon` | `g0194` | `II.21.32_2_0` | `not_timeout` | finite ID/OOD metrics missing after rerun/backfill |
