# Probe4 Full-664 三种子分发资产

- source: `exp-planning/01.双探针实验/datasets_to_run.csv`
- datasets: `664`
- tools: `udsr, dso, imcts, pyoperon`
- seeds: `520, 521, 522`
- hosts: `iaaccn23, iaaccn24, iaaccn25, iaaccn26, iaaccn27, iaaccn28, iaaccn29`
- 注意：`iaaccn22` 当前不可达，因此本批次不分配任务到 22。

## 运行方式

单个 tool + seed 启动一轮，脚本会把该轮 664 个数据集切到 7 台机器上：

```bash
bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_tool_seed.sh dso 520
```

建议不要一次性启动 12 轮；每台机器同时跑多个 CPU-heavy worker 池会互相抢核。

## 默认 workers

| tool | env | workers/host | tasks/host |
|---|---|---:|---:|
| `udsr` | `sim_dso` | `15` | `95` |
| `dso` | `sim_dso` | `15` | `95` |
| `imcts` | `sim_iMCTS` | `24` | `95` |
| `pyoperon` | `sim_base` | `24` | `95` |

## 输出路径

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/<BATCH_NAME>/<tool>/seed<seed>/<host>/
```

每轮状态文件：

```text
__launcher__/task_status.jsonl
```
