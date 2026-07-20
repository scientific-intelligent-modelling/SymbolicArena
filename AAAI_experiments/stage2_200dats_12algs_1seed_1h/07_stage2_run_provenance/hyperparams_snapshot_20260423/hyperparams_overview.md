# 12 个算法超参数总览表

这张表汇总当前 benchmark 计划中 **12 个算法** 的默认实验口径，覆盖：

- `E1` 七算法选择验证
- `E3` 的多算法轻量验证
- `E6` 的最终 leaderboard

全局默认：

- `timeout_in_seconds = 3600`
- `progress_snapshot_interval_seconds = 60`

| 算法 | 使用阶段 | 方法家族 | 核心预算 / 搜索规模 | 表达空间 / 函数集 | 关键约束 / 结构控制 | 执行口径 |
|---|---|---|---|---|---|---|
| `gplearn` | `E1 (W1)` | 经典 GP | `population_size=1000`, `generations=1000000`, `tournament_size=20` | `add,sub,mul,div,sqrt,log,sin,cos` | `parsimony=0.001`, `init_depth=[2,6]`, `metric=MAE`, `max_samples=1.0` | `n_jobs=1`, `low_memory=false`, `warm_start=false` |
| `pysr` | `E1 (W3)` | 现代演化式 SR | `niterations=10000000`, `population_size=64`, `populations=8`, `ncycles_per_iteration=500` | 二元：`+,-,*,/`；一元：`square,cube,exp,log,sin,cos` | `maxsize=30`, `maxdepth=10`, `parsimony=0.001`, 显式 `constraints / nested_constraints / operator complexity` | `precision=32`, `deterministic=true`, `parallelism=serial`, `procs=1` |
| `pyoperon` | `E1 (W2)` | 现代 EA / tree-search | `population_size=500`, `pool_size=500`, `max_evaluations=500000`, `tournament_size=5` | `add,sub,mul,div,aq,exp,log,sin,cos,tanh,sqrt,square,constant,variable` | `max_length=50`, `max_depth=10`, `offspring_generator=basic`, `reinserter=keep-best` | `optimizer=lm`, `local_search_probability=1.0`, `n_threads=1` |
| `llmsr` | `E1 (W1)` | 纯 LLM-based SR | `niterations=100000`, `samples_per_iteration=4`, `max_params=10` | 统一中性背景；固定 `benchmark_llm.config` | `inject_prompt_semantics=false`, `persist_all_samples=false` | `DeepInfra Llama-3.1-8B-Instruct`, `max_tokens=1024`, `temperature=0.6`, `top_p=0.3`, `top_k=30` |
| `drsr` | `E1 (W2)` | LLM-hybrid / DSR | `niterations=100000`, `samples_per_iteration=4`, `max_params=10` | 统一中性背景；固定 `benchmark_llm.config` | `persist_all_samples=false` | 与 `llmsr` 共用同一份 LLM 配置；`top_k=30` 已打通透传链 |
| `dso` | `E1 (W4)` | 纯 RL-based SR | `training.n_samples=2000000`, `batch_size=1000`, `epsilon=0.05` | `add,sub,mul,div,sin,cos,exp,log` | `metric=inv_nrmse`, `threshold=1e-12`, `policy_optimizer(lr=0.0005, entropy_weight=0.03, entropy_gamma=0.7)`, 官方 regression `prior` | `n_cores_batch=1`, 接近官方 `config_regression.json` |
| `tpsr` | `E1 (W5)` | Transformer + planning | `beam_size=10`, `width=3`, `num_beams=1`, `rollout=3`, `horizon=200`, `lam=0.1` | `backbone_model=e2e` | `max_input_points=200`, `max_number_bags=10`, `stop_refinement_after=1`, `n_trees_to_refine=10`, `rescale=true` | `cpu=true`, `no_seq_cache=false`, `no_prefix_cache=true`, `reward_sample_limit=2048` |
| `e2esr` | `E3 / E6` | E2E Transformer-based SR | `max_input_points=200`, `n_trees_to_refine=10` | 预训练 E2E 模型 | `max_number_bags=10`, `stop_refinement_after=1`, `rescale=true` | `force_cpu=true`, `torch_num_threads=1`，对齐官方 `evaluate.py` black-box 评测口径并避免多 worker 线程爆炸 |
| `QLattice` | `E3 / E6` | 图结构搜索 | `n_epochs=100` | `kind=regression` | `criterion=bic`, `signif=4` | `threads=1` |
| `iMCTS` | `E3 / E6` | MCTS-based SR | `max_expressions=2000000`, `K=500`, `c=4.0`, `gamma=0.5` | `+,-,*,/,sin,cos,exp,log` | `max_depth=6`, `gp_rate=0.2`, `mutation_rate=0.1`, `exploration_rate=0.2`, `max_constants=10` | `optimization_method=LN_NELDERMEAD`, `verbose=false` |
| `udsr` | `E3 / E6` | uDSR-trunk / DSO + LINEAR/poly + GP-meld | `training.n_samples=2000000`, `batch_size=1000`, `gp_meld.generations=20` | `add,sub,mul,div,sin,cos,exp,log,sqrt,1.0,const,poly` | `poly_optimizer.degree=3`, `policy_optimizer_type=pg`, `epsilon=0.05`, `baseline=R_e`, `gp_meld.run_gp_meld=true` | 保持 `udsr_wrapper` 与 `tool_name=udsr`；组件开关 `aif=false, dsr=true, lspt=false, gp_meld=true, linear_poly=true`；不是论文 full uDSR |
| `ragsr` | `E3 / E6` | RAG-SR / EvolutionaryForest | `n_gen=100`, `n_pop=200`, `max_trees=10000`, `gene_num=10` | `Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg` | `select=AutomaticLexicase`, `cross_pb=0.9`, `mutation_pb=0.1`, `max_height=10`, `categorical_encoding=Target` | wrapper 对齐官方 Target encoding 默认值；`time_limit` 由 `timeout_in_seconds` 自动派生；`cpu_num_threads=4`；每分钟写 best-so-far 快照 |

## 补充说明

- `llmsr / drsr` 当前共享同一份：
  - `exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config`
- 真实 `benchmark_llm.config` 不进入 Git；仓库只提交：
  - `benchmark_llm.config.example`
- `tpsr` 与 `e2esr` 的默认口径都带有 **CPU 环境稳定性保护项**，不等同于“完全裸官方默认”。
- `E3 / E6` 的 `e2esr / qlattice / imcts / udsr / ragsr` 目前已经补进同一快照目录，但还未纳入 `generate_e1_formal_assets.py` 的 `E1` wave 分发。
