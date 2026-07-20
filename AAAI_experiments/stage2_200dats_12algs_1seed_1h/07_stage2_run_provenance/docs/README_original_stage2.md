# E1 正式切片与启动资产

本目录由 `check/generate_e1_formal_assets.py` 生成。

## 输入

- Candidate-200 源表：
  - `experiment-results/benchmark_selection_dossier_20260422/tables/stage1_candidate200_flat.csv`

## 输出

- 统一任务表：
  - `generated/candidate200_unified.csv`
- 工具参数：
  - `generated/params/*.json`
- 各 wave / 各机器切片：
  - `generated/slices/<wave>/<tool>/<host>.csv`
- 逐主机远端 job 脚本：
  - `generated/remote_jobs/<wave>/<tool>_<host>.sh`
- 逐 wave 本地启动脚本：
  - `generated/launch/run_w1.sh` ~ `run_w5.sh`
- manifest：
  - `generated/wave_manifest.csv`

## 约定

- 远端代码根：
  - `/home/zhangziwen/workplace/scientific-intelligent-modelling`
- 远端数据根：
  - `/home/zhangziwen/sim-datasets-data`
- 固定 seed：
  - `1314`
