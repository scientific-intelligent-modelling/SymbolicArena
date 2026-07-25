# SYM-F 正式收口容量与恢复性预检

- 本地与 `iaaccn22` 使用同一组 40 条 Stage 3 分层样本。
- 分层覆盖缺表达式、无效、简单、中等和复杂表达式，各 8 条。
- `iaaccn22` 的 664 数据集 probe cache 全部成功，构建耗时约 `0.63s`。
- 对 13944 条正式运行的远端经验投影：均值约 `4593s`，p90 约
  `11076s`。
- 40 条远端样本中出现 1 次受控 CAS 两秒超时，未出现未捕获异常。
- 分片合并器同步到 `iaaccn22` 后，以 2 算法、2 数据集、3 seeds 的
  12 条独立分片做真实 CLI 合并，运行网格与正式输出校验全部通过。

原实现只在 13944 条全部完成后写出结果，进程中断会丢失全部进度。
正式链路因此改为 7 个算法分片，每片 1992 条；每片经过完整
`algorithm × gid × seed` 网格校验后原子落盘，最终再严格合并并校验
13944 个唯一运行键。

原始测量文件：

- `local_summary.json`
- `local_sample_timings.csv`
- `iaaccn22_summary.json`
- `iaaccn22_sample_timings.csv`
- `remote_shard_merge_smoke.json`
