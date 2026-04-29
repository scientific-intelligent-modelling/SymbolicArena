# Probe4 Full-664 启动前准备报告

生成时间：2026-04-30 02:45 Asia/Shanghai

## 结论

Probe4 full-664 的启动前准备已经完成，但实验尚未启动。

本轮只完成了：

- SSH 路由确认
- 资产同步
- 远端数据校验
- 远端 conda 环境补齐
- 全机预检
- 未启动状态确认

没有执行任何 `run_tool_seed.sh`，也没有创建 `probe4_full664` 实验 tmux 会话。

## 实验范围

- 算法：`udsr`, `dso`, `imcts`, `pyoperon`
- 数据集：`664`
- 种子：`520`, `521`, `522`
- 总任务数：`4 * 664 * 3 = 7968`
- 当前可用机器：`iaaccn23~29`
- 当前排除机器：`iaaccn22`

`iaaccn22` 仍然无法稳定 SSH 连接，因此当前 full-664 资产按 `iaaccn23~29` 七台机器切片。

## 远端资产状态

资产目录：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/exp-planning/02.E1选择验证/generated/probe4_full664_v1
```

预检结果：

- `full664_unified.csv`: `664` 行
- `probe4_full664_manifest.csv`: `84` 行
- 缺失切片文件：`0`
- 切片行数异常：`0`
- 缺失远端 job 脚本：`0`

## 远端数据状态

数据根目录：

```text
/home/zhangziwen/sim-datasets-data
```

预检结果：

- `iaaccn23~29` 均可访问全部 `664` 个数据集目录
- 缺失数据集目录：`0`

## 环境状态

已补齐的环境：

- 从 `iaaccn25` 复制 `sim_iMCTS` 到 `iaaccn23/24/26/27/28/29`
- 从 `iaaccn25` 复制 `sim_dso` 到 `iaaccn29`

最终预检结果：

- `iaaccn23~29`: `sim_base` 可用
- `iaaccn23~29`: `sim_dso` 可用
- `iaaccn23~29`: `sim_iMCTS` 可用
- `iaaccn23~29`: benchmark runner import 通过
- `iaaccn23~29`: DSO/uDSR tool import 通过
- `iaaccn23~29`: iMCTS tool import 通过

远端预检 JSON 保存在：

```text
/home/zhangziwen/projects/scientific-intelligent-modelling/experiments/probe4_full664_preflight_20260430/preflight_<host>.json
```

## 未启动确认

已检查 `iaaccn23~29`：

- 未发现 `probe4_full664` tmux session
- 未发现 `launch_e1_benchmark.py run` 正式任务进程

## 后续启动入口

只有在明确收到启动指令后，才执行下面的命令。

启动单个算法与种子：

```bash
bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_tool_seed.sh <tool> <seed> <batch_name>
```

示例：

```bash
bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/launch/run_tool_seed.sh dso 520 probe4_full664_v1_20260430
```

合法取值：

- `<tool>`: `udsr`, `dso`, `imcts`, `pyoperon`
- `<seed>`: `520`, `521`, `522`

## 本地提交

相关最新提交：

- `651f66c [fix] 修正Probe4预检导入口径`
- `9d3f5ad [update] 增加Probe4环境准备脚本`
