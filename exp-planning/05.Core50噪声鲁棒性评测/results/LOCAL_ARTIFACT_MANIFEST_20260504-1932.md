# Core50 本地实验产物索引

- Created at: `2026-05-04T22:47:58`

## Used artifact directories

- `remote_artifacts_20260504-183740`: `exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_20260504-183740`
  - exists: `True`, size: `0.692 GiB`
  - counts: `result_json=260`, `report_json=130`, `progress_json=132`, `minute_json=16600`, `best_history_json=1114`, `sample_top_json=1287`, `task_status_jsonl=130`, `state_json=7`, `latest_json=6`, `events_jsonl=6`
- `remote_artifacts_clean_retry_slim_20260504-193115`: `exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_clean_retry_slim_20260504-193115`
  - exists: `True`, size: `0.035 GiB`
  - counts: `result_json=60`, `report_json=30`, `progress_json=34`, `minute_json=4236`, `best_history_json=282`, `sample_top_json=323`, `task_status_jsonl=30`, `state_json=0`, `latest_json=0`, `events_jsonl=0`
- `remote_artifacts_clean_modelsplit_slim_20260504-224415`: `exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_clean_modelsplit_slim_20260504-224415`
  - exists: `True`, size: `0.55 GiB`
  - counts: `result_json=1000`, `report_json=500`, `progress_json=497`, `minute_json=60376`, `best_history_json=3926`, `sample_top_json=4959`, `task_status_jsonl=20`, `state_json=0`, `latest_json=0`, `events_jsonl=0`
- `remote_artifacts_noise_cpu_slim_20260504-185818`: `exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_noise_cpu_slim_20260504-185818`
  - exists: `True`, size: `3.299 GiB`
  - counts: `result_json=10323`, `report_json=5157`, `progress_json=1337`, `minute_json=425142`, `best_history_json=9838`, `sample_top_json=13022`, `task_status_jsonl=5157`, `state_json=0`, `latest_json=0`, `events_jsonl=0`
- `remote_artifacts_noise_gpu_slim_20260504-191614`: `exp-planning/05.Core50噪声鲁棒性评测/results/remote_artifacts_noise_gpu_slim_20260504-191614`
  - exists: `True`, size: `2.27 GiB`
  - counts: `result_json=8248`, `report_json=4119`, `progress_json=357`, `minute_json=288379`, `best_history_json=2534`, `sample_top_json=3416`, `task_status_jsonl=4119`, `state_json=0`, `latest_json=0`, `events_jsonl=0`

## Ignored partial or superseded directories

- `remote_artifacts_20260504-182437`
- `remote_artifacts_20260504-183733`
- `remote_artifacts_noise_20260504-184143`
- `remote_artifacts_noise_cpu_20260504-184611`

## Metric usage

- Task logs under __launcher__/logs/*.log are intentionally excluded from slim collections because they are not needed for metric computation and can be hundreds of MB per run.
- remote_artifacts_clean_modelsplit_slim_20260504-224415 contains the clean LLMSR/DRSR model-split rerun on iaaccn23/24: DRSR 250/250 ok and LLMSR 250/250 ok.
- remote_artifacts_clean_retry_slim_20260504-193115 supplements hosts that timed out during the first clean DRSR seed567 pull; iaaccn54 has no clean retry batch directory on remote and is treated as absent rather than failed.
- Use result_json/report_json for final metrics, minute_json/progress_json for EFF anytime metrics, best_history/sample_top for LLM iteration and candidate audits, task_status_jsonl for completion audits.
