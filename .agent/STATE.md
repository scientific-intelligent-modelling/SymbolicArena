# 工作区状态

## 当前任务

- 编号：ENV-001；任务：更新 iaaccn22~29 算法代码，再同步 iaaccn48~55 实验环境；状态：部分完成，54 待恢复连接。
- 当前采用方案：当前代码已同步到 CPU 节点的 `~/workplace/scientific-intelligent-modelling` 和 `~/projects/scientific-intelligent-modelling`。依赖取自已有 CPU 环境；GPU 复用已有 8 个环境，补充 FePySR、JAXSR、SymbolFit 和缺失依赖。
- GPU 路径：48、50~53、55 使用 `/data1/zhangziwen`，49 经用户确认使用 `/data3/zhangziwen`；代码为 `<根目录>/sim-runtime/code`，环境为 `<根目录>/anaconda3/envs`，使用前加载 `<根目录>/sim-runtime/runtime.env`。
- 有效证据：[ENV-001](work/ENV-001/)；CPU 文件校验 `cpu_sync_*.json`，CPU 导入检查 `cpu_prepare_*.json`，GPU 检查 `gpu_verify_*.json`。源机器同目录保留部署脚本、环境包和 `code_manifest.json`。
- 已确认：22~29 两处代码目录文件校验一致，TPSR、E2ESR、RAG-SR、SymbolFit、FePySR 导入通过。48~53、55 共 7 台机器的 1596 个同步文件与本地 SHA256 一致，77/77 环境导入检查通过，77/77 环境在不设置 PYTHONPATH 时均指向新代码目录。55 最后核验剩余 106.15 GiB。
- 未解决问题：54 无法通过 SSH 连接。复用环境存在部分第三方包版本差异；本次未执行正式数据集实验或 LLM API 调用。
- 下一步：54 恢复连接后复用同一任务目录补充部署。

## 有效入口：已确认

- 全局规则：[/home/family/.codex/AGENTS.md](/home/family/.codex/AGENTS.md)。
- 项目规则：[AGENTS.md](../AGENTS.md)；实验说明：[.codex/AGENTS.md](../.codex/AGENTS.md)。
- 代码与测试：[scientific_intelligent_modelling/](../scientific_intelligent_modelling/)、[tests/](../tests/)；项目入口：[cli.py](../scientific_intelligent_modelling/cli.py)。
- 配置：[toolbox_config.json](../scientific_intelligent_modelling/config/toolbox_config.json)、[envs_config.json](../scientific_intelligent_modelling/config/envs_config.json)。
- 数据目录：[sim-datasets-data/](../sim-datasets-data/)、[sim-datasets-py/](../sim-datasets-py/)；独立仓库保持原位置。
- 目录分类：`AAAI_experiments/`、`A_ICLR_experiments/`、`A_Neurips_experiments/` 为现有实验目录，按具体任务读取。
- ICLR stage4 目录：[stage4_core80_15algs_3seeds_3h/](../A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/)。

## 候选文件与问题：待验证

- CPU 节点的 `workplace` 与 `projects` 两处项目路径均被已有环境引用，本次已同步两处代码。GPU 使用上方独立目录。
- `.codex/AGENTS.md`、`.gitignore` 的已有未提交修改保持原样。

## 替代与归档

- ENV-001 当前采用最终 `gpu_verify_*.json`；初次安装和 Julia 初始化记录保留在源机器的同任务目录，不代表当前验证状态。
- WS-001 已完成；未产生独立过程记录，无需移动文件。
- WS-002 已完成；未产生独立过程记录，无需移动文件。
- SR-CLOUD-30 已关闭；本地诊断文件和关联测试已按用户确认删除。
- CL-001 已完成；`.agent/work/CL-001/` 已归档，默认不读取。
- DIR-001 已完成；未产生过程记录。
