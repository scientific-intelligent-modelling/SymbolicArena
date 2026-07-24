# NeurIPS Experiments 整理入口

这个目录整理 NeurIPS 版本使用的四阶段实验链。Stage 1--3 与后续 AAAI
归档共享同一批上游筛选和 Probe-4 实验；Stage 4 保留 NeurIPS 论文实际使用的
Core-50 十二算法五种子 clean 批次，不使用 AAAI 后续的 SSR-50 十五算法实验。

## 目录

- `stage1_664dats_2probes_1seed_1h/`
  - 阶段一：`664 datasets × 2 probes × 1 seed × 1h`
  - 两个探针是 `pysr` 和 `llmsr`
  - 目标是从 664 个候选数据集中筛出 Candidate-200

- `stage2_200dats_12algs_1seed_1h/`
  - 阶段二：`200 datasets × 12 algorithms × 1 seed × 1h`
  - 目标是用 Candidate-200 校准 12 个算法，并支撑 Probe-4 选择

- `stage3_664dats_4probes_3seeds_1h/`
  - 阶段三：`664 datasets × 4 probes × 3 seeds × 1h`
  - 目标是用 Probe-4 在全 664 数据集上三种子验证，并支撑 Core-50 选择

- `stage4_core50_12algs_5seeds_clean_1h/`
  - 阶段四：`50 datasets × 12 algorithms × 5 seeds × clean × 1h`
  - 原始批次：`core50_12alg_5seed_all_20260502-065700`
  - 目标是生成 NeurIPS Core-50 clean 排行榜和正式符号保真指标
  - 包内保存 `3000/3000` 条归档 `result.json`

## 阶段关系

```text
Stage 1: GT-Reservoir-664 双探针
  -> Candidate-200
Stage 2: Candidate-200 十二算法校准
  -> Probe-4
Stage 3: Probe-4 Full-664 三种子验证
  -> Core-50
Stage 4: Core-50 十二算法五种子 clean 正式评测
```

NeurIPS Stage 4 与 AAAI Stage 4 不是同一批实验：

- NeurIPS：`12 algorithms × 50 datasets × 5 seeds × clean × 1h = 3000`
- AAAI：`15 algorithms × 50 datasets × 3 seeds × 3 noise × 3h = 6750`

## 归档边界

这里保存的是可迁移、可审计、可二次统计的整理包，不是完整运行环境备份。

已包含：

- 四阶段主要结果表和选择材料
- Stage 3 的 7968 条 run-level raw digest
- Stage 4 的 3000 个归档 `result.json`
- Stage 4 的数值排行榜、formal SYM-F、论文表和运行参数快照
- 来源映射、文件清单和 SHA256 校验

不包含：

- 数据集实体 CSV 的完整副本
- conda / Julia 环境
- LLM API key
- Stage 4 原始实验目录中的完整日志、checkpoint 和 minute snapshots

## 恢复审计

根目录恢复材料：

```text
REMOTE_RESTORE_AUDIT.md
REMOTE_RESTORE_SUMMARY.json
REMOTE_RESULT_PATHS.tsv
REMOTE_RESULT_PATHS_MISSING.tsv
STAGE2_SOURCE_PATH_GAPS.tsv
```

Stage 1--3 的远端状态沿用 2026-07-20 的只读审计。NeurIPS Stage 4 的
3000 条原始远端路径尚未重新 SSH 核验，但对应的 3000 个结果 JSON 已经复制到
Stage 4 的 `run_archive/results/`，因此主要结果和表格不依赖远端在线状态。

## 完整性校验

```bash
cd A_Neurips_experiments/stage1_664dats_2probes_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage2_200dats_12algs_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage3_664dats_4probes_3seeds_1h
sha256sum -c CHECKSUMS.sha256

cd ../stage4_core50_12algs_5seeds_clean_1h
sha256sum -c CHECKSUMS.sha256

cd ..
sha256sum -c CHECKSUMS.sha256
```
