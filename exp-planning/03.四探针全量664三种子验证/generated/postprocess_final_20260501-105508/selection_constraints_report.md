# Probe4 Full664 后处理完成度与选择约束报告

本报告只判断当前后处理指标是否可用于后续 Core-50 selection，不执行 Core-50 选择。

## 总体完成度

- expected_total_runs: `7968`
- observed_runs: `7968`
- finished_runs: `7968`
- valid_runs: `7616`
- not_finished_runs: `0`
- ready_for_core50_selection: `true`

## Method Completion

| method | observed_runs | finished_runs | valid_runs | completion_rate | valid_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| dso | 1992 | 1992 | 1930 | 1.0000 | 0.9689 |
| imcts | 1992 | 1992 | 1963 | 1.0000 | 0.9854 |
| pyoperon | 1992 | 1992 | 1746 | 1.0000 | 0.8765 |
| udsr | 1992 | 1992 | 1977 | 1.0000 | 0.9925 |

## Seed Completion

| seed | observed_runs | finished_runs | valid_runs | completion_rate |
| --- | ---: | ---: | ---: | ---: |
| 520 | 2656 | 2656 | 2550 | 1.0000 |
| 521 | 2656 | 2656 | 2526 | 1.0000 |
| 522 | 2656 | 2656 | 2540 | 1.0000 |

## Run Outcome Distribution

- `partial_output`: 332
- `timeout_no_output`: 20
- `valid_extreme_error`: 113
- `valid_finite_result`: 7503

## Eligible Class Distribution

- `eligible`: 540
- `limited_quota`: 124

## Difficulty Distribution

- `easy`: 133
- `extreme`: 67
- `hard`: 199
- `medium`: 265

## Failure Mode Distribution

- `all_good`: 327
- `all_struggle`: 33
- `one_method_wins`: 83
- `one_sided`: 3
- `ood_failure`: 151
- `unstable`: 67

## 字段缺失

- missing_required_columns: `[]`
- missing_optional_columns: `[]`
