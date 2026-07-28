# NeurIPS Stage-4 anytime 六轴整理

- 数据：NeurIPS Stage 4 clean，`12 algorithms × 50 datasets × 5 seeds`。
- 时间点：5、10、30、60 分钟；每分钟使用当时可用的 best-so-far。
- `table_seed_mean.*`：按用户要求，先在种子层面取均值，再对 50 个任务取均值。
- `table_protocol.*`：论文聚合规则对照；ID-Q/OOD-G/EFF 在任务内对种子取
  中位数，SYM-F proxy 对种子取均值，STAB proxy 由五种子共同定义。
- 时间点 SYM-F 使用轻量 proxy：`ID/OOD NMSE <= 1e-10` 的有效公式记为近等价，
  否则使用变量/算子 F1 与变量-算子集合 Jaccard。它用于搜索动态诊断，
  不能替代 Stage 4 已归档的 60 分钟 formal CAS+dense-probe+NED SYM-F。
- 时间点 STAB 使用数值 IQR、有效 seed rate、结构签名一致性和性能校正，
  因结构一致性是轻量签名比较，同样标为 proxy。
- `ROBU`：5/10/30 分钟为 `NA`。Stage 4 归档没有 noisy minute snapshots；
  60 分钟直接引用完整 noisy track 的正式 ROBU，不从 clean 结果估计。
- 失败或当时尚无可评估候选的 run 按协议记 `NMSE=1e2`、质量为 0。
- `final_carry_forward` 仅用于在该时间点之前真实完成的运行，不使用未来快照。

## 产物

- `table_seed_mean.md/csv/tex`：与示意图同结构的主表。
- `table_protocol.md/csv/tex`：论文种子聚合规则对照表；符号和结构部分为
  明确标注的 proxy。
- `checkpoint_scores_*.csv`：算法 × 时间点宽指标。
- `checkpoint_symbolic_metrics_proxy.csv`：时间点 SYM-F proxy 的运行级明细。
- `formal_60min_symf_reference.csv`：Stage 4 已归档的正式 60 分钟
  CAS+dense-probe+NED SYM-F 对照，不与时间点 proxy 混算。
- `dataset_axis_components.csv`：算法 × 任务 × 时间点分项。
- `clean_minute_run_level.csv.gz`：180,000 条 run-minute 记录。
- `checkpoint_coverage.csv`、`snapshot_source_audit.csv`：覆盖率与来源审计。

## 本地中间文件

- `source_manifests/`、`remote_extracts/`：远端只读提取的输入和压缩回传。
- `checkpoint_runs_for_symf.csv`：SYM-F 后处理输入。
- `symf_shards/`：曾尝试的 formal CAS/NED 分片；因复杂表达式成本过高而
  中止，分片不完整且不参与本目录任何汇总。正式表只读取
  `checkpoint_symbolic_metrics_proxy.csv`。

## 复算

先用 `extract_neurips_stage4_anytime_remote.py` 在各 manifest 指定主机上
生成 `remote_extracts/*.jsonl.gz`，再从仓库根目录运行：

```bash
python check/analyze_neurips_stage4_anytime.py prepare-symf
python check/analyze_neurips_stage4_anytime.py compute-symf-proxy
python check/analyze_neurips_stage4_anytime.py aggregate
pytest -q tests/test_neurips_stage4_anytime.py
```
