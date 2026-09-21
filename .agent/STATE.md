# 工作区状态

## 当前任务

- 编号：DIR-001；任务：创建 ICLR stage4 实验目录；状态：已确认。
- 当前采用方案：使用 `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/`。
- 验证情况：目录创建成功，路径已核对。
- 未解决问题：规则中的远程项目路径记录仍待对应任务核验。
- 下一步：根据用户指定的新任务更新索引。

## 有效入口：已确认

- 全局规则：[/home/family/.codex/AGENTS.md](/home/family/.codex/AGENTS.md)。
- 项目规则：[AGENTS.md](../AGENTS.md)；实验说明：[.codex/AGENTS.md](../.codex/AGENTS.md)。
- 代码与测试：[scientific_intelligent_modelling/](../scientific_intelligent_modelling/)、[tests/](../tests/)；项目入口：[cli.py](../scientific_intelligent_modelling/cli.py)。
- 配置：[toolbox_config.json](../scientific_intelligent_modelling/config/toolbox_config.json)、[envs_config.json](../scientific_intelligent_modelling/config/envs_config.json)。
- 数据目录：[sim-datasets-data/](../sim-datasets-data/)、[sim-datasets-py/](../sim-datasets-py/)；独立仓库保持原位置。
- 目录分类：`AAAI_experiments/`、`A_ICLR_experiments/`、`A_Neurips_experiments/` 为现有实验目录，按具体任务读取。
- ICLR stage4 目录：[stage4_core80_15algs_3seeds_3h/](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/)。

## 候选文件与问题：待验证

- 根目录 AGENTS.md 与 `.codex/AGENTS.md` 中的远程项目路径存在不同记录；远程执行前按当前任务核验。根目录规则中的临时路径示例使用当前全局过程目录约定。
- `.codex/AGENTS.md`、`.gitignore` 的已有未提交修改保持原样。

## 替代与归档

- 已替代：本次未指定文件替代关系。
- WS-001 已完成；未产生独立过程记录，无需移动文件。
- WS-002 已完成；未产生独立过程记录，无需移动文件。
- SR-CLOUD-30 已关闭；本地诊断文件和关联测试已按用户确认删除。
- CL-001 已完成；`.agent/work/CL-001/` 已归档，默认不读取。
- DIR-001 已完成；未产生过程记录。
