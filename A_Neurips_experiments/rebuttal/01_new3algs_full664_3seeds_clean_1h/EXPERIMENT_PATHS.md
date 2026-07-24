# 路径清单（本实验）

```text
实验根目录:
  A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/

输入与清单:
  manifest/algorithms.json
  manifest/budget.json
  manifest/datasets.csv
  manifest/tasks.csv
  manifest/noise_levels.csv
  manifest/preparation_summary.json
  queues/full664_source.csv
  queues/smoke_2datasets_source.csv

参数:
  params/fepysr__clean.json
  params/jaxsr__clean.json
  params/symbolfit__clean.json
  params_smoke/fepysr__clean.json
  params_smoke/jaxsr__clean.json
  params_smoke/symbolfit__clean.json

烟雾:
  smoke/manifest/
  smoke/queues/smoke_2datasets_source.csv
  smoke/params/fepysr__clean.json
  smoke/params/jaxsr__clean.json
  smoke/params/symbolfit__clean.json

部署与执行:
  deploy/00_validate_and_sync_to_iaaccn22.sh
  deploy/00_fanout_from_iaaccn22.sh
  deploy/01_preflight_from_iaaccn22.sh
  deploy/02_smoke_dispatch_from_iaaccn22.sh
  deploy/03_full_dispatch_from_iaaccn22.sh
  deploy/04_collect_audit_from_iaaccn22.sh

正式运行状态:
  queues/load_queue_full/state/
  experiments/<BATCH_NAME>/  # 各远端机器
  remote-experiments/        # collect 后
  runs/                       # harvest 后
  audit/                      # audit 后

Smoke 运行状态:
  smoke/queues/load_queue_full/state/
  smoke/remote-experiments/
  smoke/runs/
  smoke/audit/
```
