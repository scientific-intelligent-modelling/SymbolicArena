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

后续代码复审继续补齐了 source provenance、完整评分公式和派生文件
哈希门禁。params、run-level、performance、generator、SYM-F merger
和 leaderboard merger 均按内容寻址冻结；params 还嵌入 664 个
`formula.py` 的 source 与 SHA-256。分片、正式 SYM-F 和合并榜单都按
`snapshot_id` 分版本，避免输入或代码变化后误复用旧产物。

`2026-07-25T11:36:42+08:00` 在 `iaaccn22` 再次执行冻结来源 smoke：
one-row 正式生成、分片合并、输出哈希复验和榜单合并全部通过；
`664/664` 份内嵌公式均通过 base64、SHA-256 和 probe cache 验证，
错误数为 `0`。正式生成器现要求 `--require-frozen-formula-source`，
不再允许本批次回退到 live `formula.py`。

原始测量文件：

- `local_summary.json`
- `local_sample_timings.csv`
- `iaaccn22_summary.json`
- `iaaccn22_sample_timings.csv`
- `remote_shard_merge_smoke.json`
- `remote_frozen_provenance_smoke.json`
