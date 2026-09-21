# 工作区状态

## 当前任务

- 编号：WS-001；任务：建立工作区与任务状态管理；状态：已确认。
- 当前采用方案：保留项目结构，以本文件索引当前任务和有效文件；过程文件按需写入 `.agent/work/WS-001/`，同一任务复用目录。
- 验证情况：入口路径已核对，过程目录忽略规则生效，STATE.md 可纳入 Git；变更格式检查通过。
- 未解决问题：下列候选文件和规则冲突需要按对应任务核验。
- 下一步：按用户指定的任务更新本索引；清理操作等待具体清单确认。

## 有效入口：已确认

- 全局规则：[/home/family/.codex/AGENTS.md](/home/family/.codex/AGENTS.md)。
- 项目规则：[AGENTS.md](../AGENTS.md)；实验说明：[.codex/AGENTS.md](../.codex/AGENTS.md)。
- 代码与测试：[scientific_intelligent_modelling/](../scientific_intelligent_modelling/)、[tests/](../tests/)；项目入口：[cli.py](../scientific_intelligent_modelling/cli.py)。
- 配置：[toolbox_config.json](../scientific_intelligent_modelling/config/toolbox_config.json)、[envs_config.json](../scientific_intelligent_modelling/config/envs_config.json)。
- 数据目录：[sim-datasets-data/](../sim-datasets-data/)、[sim-datasets-py/](../sim-datasets-py/)；独立仓库保持原位置。
- 目录分类：`diagnostics/` 保存诊断脚本和结果；`AAAI_experiments/`、`A_ICLR_experiments/`、`A_Neurips_experiments/` 为现有实验目录，按具体任务读取。

## 候选文件与问题：待验证

- 云端三算法任务编号：SR-CLOUD-30；脚本：[cloud_ten_dataset_probe.py](../diagnostics/cloud_ten_dataset_probe.py)、[replay_cloud_ten_dataset.py](../diagnostics/replay_cloud_ten_dataset.py)、[summarize_cloud_ten_dataset.py](../diagnostics/summarize_cloud_ten_dataset.py)，目前未提交。
- 对应实验目录：`diagnostics/evidence/cloud_thirty_20260921/`；输入：`inputs/`；配置和输入版本：`manifest.json`；输出：`runs/`、`accuracy.csv`。本任务仅确认路径，未指定正式有效结果版本。
- 对应 `tests/test_cloud_ten_dataset_*.py` 为未提交候选；预算结果显示规则与这些脚本、测试和 CSV 的一致性待核验。恢复该任务需用户要求，过程目录使用 `.agent/work/SR-CLOUD-30/`。
- 已确认的未完成项：`tests/test_cloud_ten_dataset_replay.py` 引用了 `budget_hof_candidate`，对应脚本尚无该定义；本任务保留现状。
- 根目录 AGENTS.md 与 `.codex/AGENTS.md` 中的远程项目路径存在不同记录；远程执行前按当前任务核验。根目录规则中的临时路径示例使用当前全局过程目录约定。
- 本任务开始时 `.codex/AGENTS.md`、`.gitignore`、`AGENTS.md` 已有未提交修改；保留这些内容。

## 替代与归档

- 已替代：本次未指定文件替代关系。
- WS-001 已完成；未产生独立过程记录，无需移动文件。
- 拟移动清单：空。现有目录保持原位置，归档后默认不读取过程记录。
