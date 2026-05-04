# Probe4 Full664 后处理完成度与选择约束报告

本报告只判断当前后处理指标是否可用于后续 Core-50 selection，不执行 Core-50 选择。

## 总体完成度

- expected_total_runs: `7968`
- observed_runs: `7968`
- finished_runs: `4182`
- valid_runs: `3970`
- not_finished_runs: `3786`
- ready_for_core50_selection: `false`

## Method Completion

| method | observed_runs | finished_runs | valid_runs | completion_rate | valid_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| dso | 1992 | 862 | 832 | 0.4327 | 0.9652 |
| imcts | 1992 | 1328 | 1308 | 0.6667 | 0.9849 |
| pyoperon | 1992 | 1328 | 1171 | 0.6667 | 0.8818 |
| udsr | 1992 | 664 | 659 | 0.3333 | 0.9925 |

## Seed Completion

| seed | observed_runs | finished_runs | valid_runs | completion_rate |
| --- | ---: | ---: | ---: | ---: |
| 520 | 2656 | 2656 | 2550 | 1.0000 |
| 521 | 2656 | 1526 | 1420 | 0.5745 |
| 522 | 2656 | 0 | 0 | 0.0000 |

## Run Outcome Distribution

- `not_finished`: 3786
- `partial_output`: 208
- `timeout_no_output`: 4
- `valid_extreme_error`: 65
- `valid_finite_result`: 3905

## Eligible Class Distribution

- `incomplete`: 664

## Difficulty Distribution

- `easy`: 133
- `extreme`: 67
- `hard`: 199
- `medium`: 265

## Failure Mode Distribution

- `incomplete`: 664

## 字段缺失

- missing_required_columns: `[]`
- missing_optional_columns: `[]`
