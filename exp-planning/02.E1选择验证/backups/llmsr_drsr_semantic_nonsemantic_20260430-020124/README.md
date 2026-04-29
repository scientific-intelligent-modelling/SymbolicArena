# LLM Semantic / Nonsemantic Result Backup

这份备份保存 Candidate-200 上 `llmsr` 与 `drsr` 的两套结果：

- `semantic_with_physics/`：带物理语义背景的新批次聚合结果与对比表。
- `nonsemantic_baseline/`：旧的不带语义 E1 基线结果，包括 12 算法总表和 `llmsr/drsr` 子表。
- `scripts/`：生成语义/非语义对比的可复跑脚本。

安全说明：未包含真实 `llm.config`、DeepInfra key 或其它 API key。
