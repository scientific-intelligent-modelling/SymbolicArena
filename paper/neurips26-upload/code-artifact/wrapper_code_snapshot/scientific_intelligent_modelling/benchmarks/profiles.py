"""统一 benchmark profile。

这些 profile 尽量忠实保留原始 benchmark 的“原生指标”，
同时映射到当前仓库可消费的字段命名。
"""

from __future__ import annotations

BENCHMARK_PROFILES = {
    "srbench": {
        "source": {
            "paper": "La Cava et al., 2021",
            "url": "https://cavalab.org/assets/papers/La%20Cava%20et%20al.%20-%202021%20-%20Contemporary%20Symbolic%20Regression%20Methods%20and%20their.pdf",
        },
        "task_groups": {
            "black_box_regression": {
                "selection_protocol": "5-fold CV for tuning; 75/25 train/test; 10 repeated trials",
                "metrics": [
                    {
                        "name": "r2_test",
                        "direction": "maximize",
                        "toolkit_field": "r2",
                        "note": "原论文黑盒回归主准确率指标",
                    },
                    {
                        "name": "model_size_raw",
                        "direction": "minimize",
                        "toolkit_field": "complexity_raw",
                        "note": "operators + features + constants",
                    },
                    {
                        "name": "model_size_simplified",
                        "direction": "minimize",
                        "toolkit_field": "complexity_simplified",
                        "note": "对表达式做 sympy simplify 后再计数",
                    },
                    {
                        "name": "training_time_seconds",
                        "direction": "minimize",
                        "toolkit_field": "train_time_seconds",
                        "note": "论文结果图包含训练时长",
                    },
                ],
            },
            "ground_truth_regression": {
                "selection_protocol": "multiple noise levels; 10 repeated trials",
                "metrics": [
                    {
                        "name": "symbolic_solution",
                        "direction": "maximize",
                        "toolkit_field": "symbolic_solution",
                        "note": "difference is constant OR ratio is non-zero constant",
                    },
                    {
                        "name": "solution_rate",
                        "direction": "maximize",
                        "toolkit_field": "solution_rate",
                        "note": "symbolic_solution 的比例统计",
                    },
                ],
            },
        },
    },
    "srsd": {
        "source": {
            "paper": "Matsubara et al., 2024",
            "url": "https://openreview.net/pdf?id=qrUdrXsiXX",
        },
        "task_groups": {
            "scientific_discovery": {
                "selection_protocol": "use validation geometric / relative error to choose best model, then compute symbolic metrics",
                "metrics": [
                    {
                        "name": "legacy_accuracy_proxy",
                        "direction": "maximize",
                        "toolkit_field": "legacy_accuracy_proxy",
                        "note": "用于兼容旧式 error / binary 视角；SRSD 真正强调的是 NED",
                    },
                    {
                        "name": "solution_rate",
                        "direction": "maximize",
                        "toolkit_field": "solution_rate",
                        "note": "沿用 SRBench 的 binary symbolic solution",
                    },
                    {
                        "name": "normalized_edit_distance",
                        "short_name": "ned",
                        "direction": "minimize",
                        "toolkit_field": "ned",
                        "note": "对简化后的 equation tree 做 edit distance，并按真值树大小归一化",
                    },
                ],
            },
        },
    },
    "llm_srbench": {
        "source": {
            "paper": "Shojaee et al., 2025",
            "url": "https://proceedings.mlr.press/v267/shojaee25a.html",
        },
        "task_groups": {
            "scientific_equation_discovery": {
                "selection_protocol": "report symbolic correctness together with ID/OOD numerical generalization",
                "metrics": [
                    {
                        "name": "symbolic_accuracy",
                        "short_name": "sa",
                        "direction": "maximize",
                        "toolkit_field": "symbolic_accuracy",
                        "note": "论文主表核心指标",
                    },
                    {
                        "name": "acc_0_1_id",
                        "direction": "maximize",
                        "toolkit_field": "id_test.acc_0_1",
                        "note": "Acc_tau with tau=0.1 on ID split",
                    },
                    {
                        "name": "acc_0_1_ood",
                        "direction": "maximize",
                        "toolkit_field": "ood_test.acc_0_1",
                        "note": "Acc_tau with tau=0.1 on OOD split",
                    },
                    {
                        "name": "nmse_id",
                        "direction": "minimize",
                        "toolkit_field": "id_test.nmse",
                        "note": "ID split NMSE",
                    },
                    {
                        "name": "nmse_ood",
                        "direction": "minimize",
                        "toolkit_field": "ood_test.nmse",
                        "note": "OOD split NMSE; 论文特别强调其重要性",
                    },
                ],
            },
        },
    },
}
