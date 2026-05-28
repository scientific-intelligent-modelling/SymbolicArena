  # 让未跑满 1 小时的算法使用完整有效搜索预算

  ## Summary

  - 目标：只影响正式评测参数/正式运行路径，不拖慢 check/* 和 smoke。
  - 口径：不靠空等补足时间，而是让算法持续做有效搜索，直到接近 timeout_in_seconds=3600。
  - 当前提前结束明显的算法：
      - 全部/大多数提前：pyoperon、qlattice、e2esr、tpsr、dso
      - 部分提前：imcts、udsr、ragsr
      - 基本已跑满：drsr、gplearn、llmsr、pysr
  - 新增一份正式 1h 参数 profile，并让 Core-50/正式队列显式使用；不改普通默认 check 速度。

  ## Key Changes

  - 新增正式参数目录，例如：
      - frozen-results/neurips26/final-experiments/03_core50_12alg_final/runtime_params_1h_effective/
      - 每个算法一个 JSON：pyoperon.json、qlattice.json、e2esr.json、tpsr.json、dso.json、udsr.json、imcts.json、ragsr.json 等。
  - 生成一份说明表：
      - runtime_budget_plan.csv
      - 字段：algorithm,current_median_seconds,problem_type,planned_change,expected_behavior
  - 正式运行时通过现有队列入口的 --params-root 指向这份新参数目录；不改 smoke/check 的默认参数。

  ## Algorithm-Specific Plan

  - pyoperon
      - 问题：max_time=3600 存在，但 wrapper 当前默认 generations=1，导致一代后自然结束。
      - 改法：正式参数设置 generations 为超大值，max_evaluations 提高到不会先耗尽；wrapper 若检测到 max_time 且未显式传 generations，
        自动进入“按时间跑多代”的模式。
      - 目标：由 ~10s 提升到接近 3600s。
  - qlattice
      - 问题：n_epochs=100 很快跑完，未使用 timeout_in_seconds 控制训练时长。
      - 改法：正式参数设置 n_epochs 为超大值；wrapper 显式读取 timeout_in_seconds，按单 epoch 循环直到 timeout - guard_seconds。
      - 目标：持续调用 auto_run(n_epochs=1)，保留每轮最佳模型。
  - e2esr
      - 问题：预训练模型推理/refine 是有限 pass，stop_refinement_after=1、max_number_bags=10 太小。
      - 改法：正式参数提高 max_number_bags、n_trees_to_refine，取消或显著放宽 stop_refinement_after；wrapper 增加时间预算循环，持续采
        样/精修候选并保留最优。
      - 目标：不是单次 decoder 结束就返回，而是在 1h 内持续生成/精修候选。
  - tpsr
      - 问题：搜索循环有固定 horizon=200 和固定 MCTS 步数，很多任务自然结束。
      - 改法：正式参数提高 horizon、rollout、beam_size、n_trees_to_refine；wrapper 把固定步数循环改成预算感知循环，直到接近
        timeout_in_seconds。
      - 目标：持续 TPSR planning/refinement，而不是固定 200 步后返回。
  - dso
      - 问题：n_samples=2_000_000 在部分任务上不足以占满 1h，且 wrapper 当前丢弃 timeout_in_seconds。
      - 改法：正式参数提高 training.n_samples；wrapper 保留 _timeout_in_seconds，用 train_one_step() 循环按时间优雅收口，并继续
        写 .dso_current_best.json。
      - 目标：让 DSO 在有效时间内持续采样训练，同时避免被外层硬 kill 后丢结果。
  - udsr
      - 问题：继承 DSO 路径，部分任务也提前完成。
      - 改法：沿用 DSO 的时间预算训练循环；正式参数同步提高 training.n_samples，并保留 uDSR 的 GP-meld/poly 配置。
      - 目标：保持 uDSR 特性，同时用满 1h 搜索预算。
  - imcts
      - 问题：K=500、max_expressions=2_000_000 对部分任务不足，底层 fit() 完成后直接返回。
      - 改法：正式参数提高 K 和 max_expressions；若底层不支持 warm-start，wrapper 用 seed offset 做多轮 chunk 搜索并保留最优表达式。
      - 目标：让 iMCTS 在 1h 内持续探索，而不是固定 K 次后结束。
  - ragsr
      - 问题：已有 time_limit=timeout-5 映射，但 n_gen=100 可能先跑完。
      - 改法：正式参数提高 n_gen 到大值，保留 time_limit=3595；必要时 wrapper 在有 timeout_in_seconds 时自动把默认 n_gen 提升。
      - 目标：让 EvolutionaryForest 优先由 time_limit 控制结束。

  ## Runtime Policy

  - 统一使用：
      - timeout_in_seconds=3600
      - progress_snapshot_interval_seconds=60
        -# 让未跑满 1 小时的算法在正式评测中尽量用满有效搜索预算

  ## Summary

  - 目标：只影响正式 benchmark/Core-50 运行参数，不拖慢 check_*、smoke 和本地快速验证。
  - 口径：优先让算法做更多有效搜索；不采用单纯 sleep 补满墙钟。
  - 依据：最终 Core-50 runtime 汇总显示 pyoperon/qlattice/e2esr/tpsr/dso/imcts/udsr/ragsr 存在明显提前结束，其中 pyoperon/qlattice/
    e2esr/tpsr 最严重。
  - 兜底：所有正式参数仍保留 timeout_in_seconds=3600，外层 runner 负责 1 小时终止和结果回收。

  ## Key Changes

  - 新增一套正式长预算参数目录，例如 frozen-results/neurips26/final-experiments/params_1h_effective_search/，不要改 wrapper 默认值。
  - 更新正式运行脚本/调度器使用该参数目录，通过现有 --params-json 注入；check/launch_e1_benchmark.py 已支持逐工具参数 JSON。
  - 对提前结束算法调大有效搜索预算：
      - pyoperon：大幅提高 max_evaluations，保留 max_time=3600，避免 500k evaluations 十几秒结束。
      - qlattice：大幅提高 n_epochs，让外层 3600s timeout 截断，而不是 100 epoch 自然结束。
      - e2esr：提高候选/refinement 预算，尤其 n_trees_to_refine，并取消或放宽 stop_refinement_after=1。
      - tpsr：提高 horizon、rollout、beam_size/num_beams、n_trees_to_refine，让 MCTS/decoder 搜索持续更久。
      - dso/udsr：提高 training.n_samples，必要时同步提高 batch/采样轮次；继续保留 progress snapshot。
      - imcts：提高 K 和 max_expressions，避免部分数据集十几秒到几分钟自然结束。
      - ragsr：提高 n_gen/n_pop，保留 time_limit=timeout_in_seconds-5 作为内部时间兜底。
  - 保持已基本跑满的算法参数不变：
      - drsr
      - gplearn
      - llmsr
      - pysr

  ## Implementation Details

  - 生成 core50_algorithm_runtime_summary.csv 作为基线证据，保留在 03_core50_12alg_final/collected_results/。
  - 从现有 core50_algorithm_hyperparameters_full.json 派生正式 1h 参数 JSON，每个工具一个文件：
      - pyoperon.json
      - qlattice.json
      - e2esr.json
      - tpsr.json
      - dso.json
      - udsr.json
      - imcts.json
      - ragsr.json
  - 推荐初始参数倍率：
      - pyoperon.max_evaluations: 500000 -> 200000000
      - qlattice.n_epochs: 100 -> 100000
      - e2esr.n_trees_to_refine: 10 -> 500，移除或设置 stop_refinement_after=null
      - tpsr.horizon: 200 -> 2000，rollout: 3 -> 10，beam_size: 10 -> 64，num_beams: 1 -> 8，n_trees_to_refine: 10 -> 500
      - dso.training.n_samples: 2000000 -> 10000000
      - udsr.training.n_samples: 2000000 -> 10000000，保留 uDSR 的 GP-meld/poly 配置
      - imcts.K: 500 -> 5000，max_expressions: 2000000 -> 50000000
      - ragsr.n_gen: 100 -> 100000，n_pop: 200 -> 500，time_limit: 3595
  - 若某些算法仍提前结束，再进行第二轮倍率调整；不在第一轮加入 sleep。

  ## Test Plan

  - 先做小样本 smoke，不跑全量：
      - 每个需调整算法选 2 个 Core-50 数据集 × 1 seed。
      - 验证 result.json 正常落盘、status 为 ok/timed_out、有 equation 或合理 no-valid-output。
  - 统计 smoke runtime：
      - 目标：多数任务 seconds >= 3300 或被外层 timeout 截断。
      - 如果任务仍 <1800s，说明该工具还有内部自然停止条件未解除，需要继续调该工具参数。
  - 再跑正式小批：
      - 每个需调整算法选 5 个 Core-50 数据集 × 1 seed。
      - 对比 seconds、有效输出率、是否出现新异常。
  - 最后才重新跑完整 Core-50 对应算法。

  ## Assumptions

  - 不改 wrapper 默认值；正式长预算只通过参数 JSON 生效。
  - 不用 sleep 补满一小时，除非后续明确要求墙钟公平而非有效搜索。
  - 当前 experiments/ 原始 raw 目录已不在根目录，因此本计划基于 frozen-results 中已固化的最终 CSV 和现有 wrapper 代码制定。
  - timeout_in_seconds=3600 继续作为硬预算；个别 LLM/DRSR 当前实际约 3900s 属于外层/远程收尾开销，不在本次调整范围内。

