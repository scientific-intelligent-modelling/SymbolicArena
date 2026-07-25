# 路径清单（本实验）

```text
实验根目录:
  A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/

输入与清单:
  BATCH_NAME.txt
  SOURCE_MAP.tsv
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
  deploy/05_generate_symf_from_iaaccn22.sh
  deploy/06_audit_completed_from_iaaccn22.sh

正式运行状态:
  queues/load_queue_full/state/
  /home/zhangziwen/workplace/scientific-intelligent-modelling/
    experiments/<BATCH_NAME>/  # 各远端机器上的规范结果
  remote-experiments/        # collect 后
  runs/                       # harvest 后
  audit/                      # audit 后

运行期审计:
  monitoring/completed_audit/<timestamp>/
  monitoring/result_health/<timestamp>/
  monitoring/controller_recovery/<timestamp>/
  monitoring/state_atomicity/<timestamp>/
  monitoring/progress_snapshot_audit/<timestamp>/
  monitoring/symf_capacity/<timestamp>/

分析与符号指标:
  analysis/                              # 运行中是 INCOMPLETE_PREVIEW
  analysis/audit_gate_summary.json       # 正式收口后生成
  analysis/full664_7alg_run_level.csv
  analysis/full664_7alg_leaderboard.csv
  analysis/full664_7alg_leaderboard_with_symf.csv
  symf/params/
  symf/source_snapshots/<snapshot_id>/
  symf/full664_7alg/by_source/<snapshot_id>/shards/<algorithm>/
  symf/full664_7alg/by_source/<snapshot_id>/final/
  symf/full664_7alg/by_source/<snapshot_id>/leaderboard/

来源与最终归档:
  provenance/
  MANIFEST.tsv             # 5976 完成并拉回正式产物后生成
  CHECKSUMS.sha256         # 由最终 MANIFEST.tsv 覆盖范围生成并校验

Smoke 运行状态:
  smoke/queues/load_queue_full/state/
  smoke/remote-experiments/
  smoke/runs/
  smoke/audit/
```
