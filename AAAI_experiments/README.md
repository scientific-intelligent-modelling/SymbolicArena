# AAAI Experiments 整理入口

这个目录把 AAAI 版本里四个阶段的实验材料从原工程散落位置整理到了一个统一入口。

## 目录

- `stage1_664dats_2probes_1seed_1h/`
  - 阶段一：`664 datasets × 2 probes × 1 seed × 1h`
  - 两个探针是 `pysr` 和 `llmsr`
  - 目标是从 664 个候选数据集中筛出 Candidate-200

- `stage2_200dats_12algs_1seed_1h/`
  - 阶段二：`200 datasets × 12 algorithms × 1 seed × 1h`
  - 目标是用 Candidate-200 校准 12 个算法，并支撑 Probe-4 选择
  - 这个目录来自此前整理好的 `probe4_candidate200_selection_package_20260720`

- `stage3_664dats_4probes_3seeds_1h/`
  - 阶段三：`664 datasets × 4 probes × 3 seeds × 1h`
  - 目标是用 Probe-4 在全 664 数据集上三种子验证，并支撑 SSR-50/Core-50 选择
  - 这个目录保持扁平，只放 postprocess 核心文件和顶层校验文件

- `stage4_ssr50_15algs_3seeds_3noise_3h/`
  - 阶段四：`50 datasets × 15 algorithms × 3 seeds × 3 noise × 3h`
  - 目标是最终 SSR-50 打榜
  - 这个目录保持扁平，当前权威口径是 `*_with_fepysr_rerun`

## 口径

这里放的是“可迁移、可复盘、可二次统计”的整理包，不是完整运行环境备份。

不包含：

- 数据集实体 CSV
- conda / Julia 环境
- LLM API key
- 远端机器完整日志目录
- 每个任务的完整原始结果文件森林

每个阶段目录下都有自己的 `README.md` 和校验文件：

- stage1 / stage2 使用 `99_audit/checksums.sha256`
- stage3 / stage4 使用顶层 `CHECKSUMS.sha256`
- `AAAI_experiments/` 根目录也有一份总 `CHECKSUMS.sha256`

远端恢复审计见：

```text
REMOTE_RESTORE_AUDIT.md
REMOTE_RESTORE_SUMMARY.json
REMOTE_RESULT_PATHS.tsv
REMOTE_RESULT_PATHS_MISSING.tsv
STAGE2_SOURCE_PATH_GAPS.tsv
```

搬迁后可以分别执行：

```bash
cd stage1_664dats_2probes_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage2_200dats_12algs_1seed_1h
sha256sum -c 99_audit/checksums.sha256

cd ../stage3_664dats_4probes_3seeds_1h
sha256sum -c CHECKSUMS.sha256

cd ../stage4_ssr50_15algs_3seeds_3noise_3h
sha256sum -c CHECKSUMS.sha256

cd ..
sha256sum -c CHECKSUMS.sha256
```
