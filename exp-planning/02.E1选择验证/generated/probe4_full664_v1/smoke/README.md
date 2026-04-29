# Probe4 Full-664 Smoke Test 资产

- 目的：正式 full-664 启动前，每台机器跑 4 个算法各 1 个真实数据集任务。
- 范围：`iaaccn23~26`。
- 算法：`udsr`, `dso`, `imcts`, `pyoperon`。
- seed: `990`。
- 预算：每任务 `timeout_in_seconds=180`，仅用于集成 smoke，不作为正式实验结果。

| host | dataset | global_index |
|---|---|---|
| `iaaccn23` | `Nguyen-1` | `633` |
| `iaaccn24` | `Nguyen-2` | `637` |
| `iaaccn25` | `Nguyen-3` | `638` |
| `iaaccn26` | `Nguyen-5` | `640` |

远端启动脚本：

```bash
bash exp-planning/02.E1选择验证/generated/probe4_full664_v1/smoke/remote_jobs/run_host_smoke.sh <host-label> <batch-name>
```
