# 15 算法效果统计路径清单

生成时间：2026-06-27

最近更新：2026-07-19，已将 `llmsr_s521_clean_g0048` 从中间产物恢复并回写为 `ok`。

本文件记录这版 15 算法统计使用的真实实验来源、overlay 规则和本地产出路径。下次继续分析时优先从本目录开始，不要重新按 `latest` 猜口径。

## 本地分析目录

```text
/home/family/workplace/scientific-intelligent-modelling/benchmark-runs/analysis_15alg_effects_20260627
```

关键文件：

```text
algorithm_summary_with_fepysr_rerun.csv   # 修正版 15 算法汇总表
selected_runs_with_fepysr_rerun.csv       # 修正版 6750 条 run 级明细，path 列是远端 result.json 绝对路径
remote_scan_with_fepysr_rerun.json        # 远端 8 台机器只读扫描快照，已包含 LLMSR 恢复
remote_scan_after_llmsr_recovery.json     # LLMSR 回写后的远端扫描快照
scan_remote_effects.py                    # 当时使用的远端扫描脚本
experiment_paths_manifest.json            # 本文件的机器可读版本
algorithm_summary_all_recovered.csv       # 6750/6750 ok 后的汇总表
selected_runs_all_recovered.csv           # 6750/6750 ok 后的 run 级明细
```

旧版文件 `algorithm_summary.csv` 和 `selected_runs.csv` 没有 overlay `fepysr` rerun，只保留作对照，不再作为当前结论口径。

## 远端根目录

远端仓库根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling
```

远端结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments
```

扫描机器：

```text
iaaccn22
iaaccn23 / 10.10.100.23
iaaccn24 / 10.10.100.24
iaaccn25 / 10.10.100.25
iaaccn26 / 10.10.100.26
iaaccn27 / 10.10.100.27
iaaccn28 / 10.10.100.28
iaaccn29 / 10.10.100.29
```

## 当前统计口径

总规模：

```text
15 algorithms * 450 tasks = 6750 selected runs
```

选择规则：

1. `formal3h` 主批提供大部分 13 算法结果。
2. `fepysr` 主批原先 `406/450 ok`，用 `fepysr_rerun` 的 44 个任务 overlay 后变成 `450/450 ok`。
3. `tpsr` 和 `ragsr` 使用单独 rerun 批次覆盖主批结果。
4. `llmsr` 使用独立 3h LLM 批次；`llmsr_s521_clean_g0048` 已回写恢复，当前 `450/450 ok`。
5. `drsr` 使用独立 3h LLM 批次，当前 `450/450 ok`。

## 远端实验来源

### formal3h 主批

用于：

```text
QLattice
dso
e2esr
gplearn
iMCTS
jaxsr
pyoperon
pysr
symbolfit
udsr
fepysr 主批 406 个 ok 结果
```

远端实验结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658
```

远端控制面目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/benchmark-runs/formal3h/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658
```

### fepysr rerun overlay

用于：

```text
fepysr 44 个失败任务的 rerun 覆盖
```

远端实验结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658_fepysr_rerun
```

远端队列状态：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/benchmark-runs/formal3h/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658/queues/load_queue_fepysr_rerun/state/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658_fepysr_rerun.latest.json
```

关键状态：

```text
rerun queue: done=43, failed=1
selected overlay: fepysr 450/450 ok
```

说明：队列里的 `failed=1` 不等于最终 overlay 后仍失败；本次 run 级扫描里 `fepysr_rerun` 有 44 个可用 task-level 结果，overlay 后 `fepysr` 为 `450/450 ok`。

### tpsr/ragsr rerun

用于：

```text
tpsr
ragsr
```

远端实验结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/formal3h_tpsr_ragsr_rerun_20260624-115748
```

远端控制面目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/benchmark-runs/formal3h/formal3h_tpsr_ragsr_rerun_20260624-115748
```

### llmsr 3h 批次

用于：

```text
llmsr
```

远端实验结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442
```

本地控制面目录：

```text
/home/family/workplace/scientific-intelligent-modelling/benchmark-runs/llmsr3h/llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442
```

关键状态：

```text
llmsr selected: 450/450 ok
recovered task: llmsr_s521_clean_g0048 / feynman-i.11.19
recovery equation: c0*x0*x1 + c1*x2*x3
id_test nmse: 0.3246691804777049
ood_test nmse: 0.3288034818846632
```

回写备份：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442/llmsr/seed521/tasks/llmsr_s521_clean_g0048/iaaccn26/llmsr/g0048_feynman-i.11.19/result.json.timeout_backup_20260719-214512
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/llmsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260614-220442/llmsr/seed521/tasks/llmsr_s521_clean_g0048/iaaccn26/llmsr/g0048_feynman-i.11.19/experiments/g0048_feynman-i.11.19_llmsr_seed521_20260615-162520/result.json.timeout_backup_20260719-214512
```

### drsr 3h 批次

用于：

```text
drsr
```

远端实验结果根目录：

```text
/home/zhangziwen/workplace/scientific-intelligent-modelling/experiments/drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355
```

本地控制面目录：

```text
/home/family/workplace/scientific-intelligent-modelling/benchmark-runs/drsr3h/drsr3h_ssr50_seed520-522_noise0-001-005_turbo_20260618-162355
```

关键状态：

```text
drsr selected: 450/450 ok
```

## 常用入口

查看修正版算法汇总：

```bash
column -s, -t < benchmark-runs/analysis_15alg_effects_20260627/algorithm_summary_with_fepysr_rerun.csv
```

查某个算法的所有远端 result 路径：

```bash
python - <<'PY'
import csv
with open('benchmark-runs/analysis_15alg_effects_20260627/selected_runs_with_fepysr_rerun.csv') as f:
    for row in csv.DictReader(f):
        if row['algorithm'] == 'fepysr':
            print(row['task_id'], row['status'], row['path'])
PY
```

复用远端扫描脚本：

```bash
scp benchmark-runs/analysis_15alg_effects_20260627/scan_remote_effects.py iaaccn22:/tmp/sim_scan_effects.py
ssh iaaccn22 'python3 /tmp/sim_scan_effects.py > /tmp/sim_effects_15alg.json'
```
