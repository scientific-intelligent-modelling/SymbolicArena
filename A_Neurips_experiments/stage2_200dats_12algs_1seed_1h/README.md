# Probe-4 Candidate-200 选择整理包

这个目录是给后续换机器/换 session 用的整理版材料包，不是把原实验目录整棵复制进来。

它覆盖的是这一段证据链：

```text
Candidate-200
  -> E1 12算法校准
  -> llmsr/drsr 语义覆盖
  -> Probe-4 选择
```

## 先看哪里

1. `01_v02_selection_digest/`
   - 最新可读选择包，来自 `probe4_v02_readable_selection_20260429`
   - 主表是 `probe4_run_level_big_table_2400.csv`
   - 规模是 `200 datasets × 12 algorithms = 2400 rows`

2. `03_e1_12alg_calibration/`
   - 冻结版 E1 12算法校准结果
   - 用来追 Probe-4 为什么会被选出来

3. `04_semantic_llm_override/`
   - `llmsr` / `drsr` 的语义物理覆盖结果
   - 对应 `400` 行覆盖记录

4. `05_downstream_freeze_probe4_full664/`
   - 这里只保留迁移说明
   - Probe-4 全 664 三种子验证已经归入 `../stage3_664dats_4probes_3seeds_1h/`

5. `06_merged_tables/`
   - 同一任务下不同算法横向展开的大表
   - stage2 内只保留 Candidate-200 / E1 相关宽表

6. `07_stage2_run_provenance/`
   - 第二阶段运行追溯材料
   - 包括原始聚合结果、运行索引、host/wave 摘要、启动脚本和参数快照

7. `99_audit/`
   - `source_map.tsv` 记录每块材料来自哪里
   - `file_inventory.tsv` 记录包内文件清单
   - `checksums.sha256` 用于搬迁后校验

## 为什么没有直接复制原目录

原目录里有大量中间文件、重复文件和路径依赖。这个包只保留复盘和迁移时需要的材料：

- 选择主表
- 算法级/数据集级摘要
- 选择报告和方法说明
- 语义覆盖证据
- 算法横向合并大表
- 第二阶段运行索引与参数快照
- 指向 stage3 的下游验证说明
- 来源映射和哈希校验

不包含：

- 原始 `result.json` 森林
- 数据集实体 CSV
- conda / Julia / LLM API 环境
- 完整 `眼不见为净` 工作目录
- 被冻结版重复覆盖的旧 `e1_final_results_current_20260429` 整目录

其中 `e1_final_results_current_20260429` 没有单独整目录收入，是因为核心文件已经在冻结版 `02_candidate200_12alg_calibration` 中保留；之前用 sha256 校验过关键 CSV 和计划文档是同内容副本。

## 搬迁后校验

在新机器上复制完这个目录后，可以执行：

```bash
cd /path/to/stage2_200dats_12algs_1seed_1h
sha256sum -c 99_audit/checksums.sha256
```

如果全部显示 `OK`，说明整理包内文件没有在搬迁时损坏。

## 重要口径

- 这个包回答的是“Candidate-200 上如何选 Probe-4”。
- Probe-4 后续全 664 三种子验证已经迁移到 `../stage3_664dats_4probes_3seeds_1h/`。
- 它不是最终 SSR-50 打榜包。
- 如果要追 NeurIPS 最终 `12 algorithms × 50 datasets × 5 seeds × 4 conditions × 1h`，
  应查看 `../stage4_core50_12algs_5seeds_4noise_1h/`。四个条件是 clean 和
  `sigma=0.01/0.05/0.10`。后续 AAAI 的
  `15 algorithms × 50 datasets × 3 seeds × 3 noise × 3h` 不属于本归档。
