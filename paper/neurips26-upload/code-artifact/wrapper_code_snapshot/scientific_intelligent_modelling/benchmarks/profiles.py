"""note benchmark profile 

note profile note benchmark note note  
notecurrent notefieldnote 
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
                        "note": "note",
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
                        "note": "note sympy simplify note",
                    },
                    {
                        "name": "training_time_seconds",
                        "direction": "minimize",
                        "toolkit_field": "train_time_seconds",
                        "note": "note",
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
                        "note": "symbolic_solution note",
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
                        "note": "note error / binary note SRSD note NED",
                    },
                    {
                        "name": "solution_rate",
                        "direction": "maximize",
                        "toolkit_field": "solution_rate",
                        "note": "note SRBench note binary symbolic solution",
                    },
                    {
                        "name": "normalized_edit_distance",
                        "short_name": "ned",
                        "direction": "minimize",
                        "toolkit_field": "ned",
                        "note": "note equation tree note edit distance note",
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
                        "note": "note",
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
                        "note": "OOD split NMSE; note",
                    },
                ],
            },
        },
    },
}
