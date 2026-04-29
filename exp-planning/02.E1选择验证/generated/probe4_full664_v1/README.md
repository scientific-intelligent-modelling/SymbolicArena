# Probe4 Full-664 三种子分发资产

- source: `exp-planning/01.双探针实验/datasets_to_run.csv`
- datasets: `664`
- tools: `udsr, dso, imcts, pyoperon`
- seeds: `520, 521, 522`
- hosts: `iaaccn23, iaaccn24, iaaccn25, iaaccn26, iaaccn27, iaaccn28, iaaccn29`
- 注意：`iaaccn22` 当前不可达，因此本批次不分配任务到 22。

## 负载队列入口

推荐使用负载感知队列，让 `iaaccn23` 做中心调度，`23~29` 空闲后自动领取 chunk：

```bash
tmux new-session -d -s probe4_full664_queue \
  bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_load_queue.sh
```

调度器默认每台机器同时只跑一个 chunk，并按 load、可用内存和已有 `probe4` session 数派发任务。

默认 chunk 策略：

| tool | env | workers/chunk | datasets/chunk |
|---|---|---:|---:|
| `pyoperon` | `sim_base` | `24` | `24` |
| `imcts` | `sim_iMCTS` | `24` | `24` |
| `dso` | `sim_dso` | `8` | `8` |
| `udsr` | `sim_dso` | `8` | `8` |

状态文件：

```text
exp-planning/02.E1选择验证/generated/probe4_full664_v1/load_queue/state/<BATCH_NAME>.latest.json
```

## 固定切片入口

保留固定切片入口用于回退。单个 tool + seed 启动一轮，脚本会把该轮 664 个数据集切到 7 台机器上：

```bash
bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_tool_seed.sh dso 520
```

固定切片方式不建议一次性启动 12 轮；每台机器同时跑多个 CPU-heavy worker 池会互相抢核。

## 固定切片默认 workers

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
