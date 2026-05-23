from __future__ import annotations

import pytest


def test_gplearn_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.gplearn_wrapper.wrapper import GPLearnRegressor

    reg = GPLearnRegressor()
    assert reg.params["population_size"] == 1000
    assert reg.params["generations"] == 1000000
    assert reg.params["function_set"] == ("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos")
    assert reg.params["metric"] == "mean absolute error"
    assert reg.params["n_jobs"] == 4


def test_pysr_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.pysr_wrapper.wrapper import PySRRegressor

    reg = PySRRegressor()
    assert reg.params["timeout_in_seconds"] == 3600
    assert reg.params["niterations"] == 10000000
    assert reg.params["population_size"] == 64
    assert reg.params["populations"] == 8
    assert reg.params["parallelism"] == "serial"
    assert reg.params["procs"] == 1


def test_fepysr_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.fepysr_wrapper.wrapper import FePySRRegressor

    reg = FePySRRegressor()
    assert reg.params["num_workers"] == 4
    assert reg.params["num_experiments"] == 8
    assert reg.params["fmn_epochs"] == 30
    assert reg.params["niterations"] == 40
    assert reg.params["maxsize"] == 20
    assert reg.params["binary_operators"] == ["+", "-", "*", "/"]


def test_jaxsr_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.jaxsr_wrapper.wrapper import JAXSRRegressor

    reg = JAXSRRegressor()
    assert reg.params["max_terms"] == 5
    assert reg.params["strategy"] == "greedy_forward"
    assert reg.params["information_criterion"] == "bic"
    assert reg.params["max_polynomial_degree"] == 3
    assert reg.params["include_interactions"] is True
    assert reg.params["include_transcendental"] is False


def test_symbolfit_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.symbolfit_wrapper.wrapper import SymbolFitRegressor

    reg = SymbolFitRegressor()
    assert reg.params["niterations"] == 40
    assert reg.params["maxsize"] == 25
    assert reg.params["max_complexity"] == 25
    assert reg.params["fit_y_unc"] is False
    assert reg.params["procs"] == 1
    assert reg.params["parallelism"] == "serial"


def test_pyoperon_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.pyoperon_wrapper.wrapper import OperonRegressor

    reg = OperonRegressor()
    assert reg.params["max_time"] == 3600
    assert reg.params["population_size"] == 500
    assert reg.params["pool_size"] == 500
    assert reg.params["allowed_symbols"] == "add,mul,aq,exp,log,sin,tanh,constant,variable"
    assert reg.params["max_evaluations"] == 500000
    assert reg.params["n_threads"] == 4


def test_llmsr_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.llmsr_wrapper.wrapper import LLMSRRegressor

    reg = LLMSRRegressor()
    assert reg.params["timeout_in_seconds"] == 3600
    assert reg.params["max_params"] == 10
    assert reg.params["niterations"] == 100000
    assert reg.params["samples_per_iteration"] == 4
    assert reg.params["persist_all_samples"] is False
    assert reg.params["inject_prompt_semantics"] is False
    assert reg.params["llm_config_path"].endswith("exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config")


def test_drsr_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor

    reg = DRSRRegressor()
    assert reg.params["timeout_in_seconds"] == 3600
    assert reg.params["max_params"] == 10
    assert reg.params["niterations"] == 100000
    assert reg.params["samples_per_iteration"] == 4
    assert reg.params["persist_all_samples"] is False
    assert reg.params["llm_config_path"].endswith("exp-planning/02.E1选择验证/llm_configs/benchmark_llm.config")


def test_dso_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.dso_wrapper.wrapper import DSORegressor

    reg = DSORegressor()
    assert reg.params["task"]["function_set"] == ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]
    assert reg.params["training"]["batch_size"] == 1000
    assert reg.params["training"]["n_samples"] == 2000000
    assert reg.params["training"]["n_cores_batch"] == 4
    assert reg.params["policy_optimizer"]["learning_rate"] == 0.0005
    assert reg.params["prior"]["soft_length"]["loc"] == 10


def test_tpsr_defaults_are_aligned():
    pytest.importorskip("torch")
    from scientific_intelligent_modelling.algorithms.tpsr_wrapper.wrapper import TPSRRegressor

    reg = TPSRRegressor()
    assert reg.params["cpu"] is True
    assert reg.params["max_input_points"] == 200
    assert reg.params["max_number_bags"] == 10
    assert reg.params["rollout"] == 3
    assert reg.params["reward_sample_limit"] == 2048
    assert reg.params["cpu_num_threads"] == 4


def test_e2esr_defaults_are_aligned(monkeypatch):
    pytest.importorskip("torch")
    from scientific_intelligent_modelling.algorithms.e2esr_wrapper.wrapper import E2ESRRegressor

    monkeypatch.setattr(E2ESRRegressor, "_load_model", lambda self: None)
    reg = E2ESRRegressor()
    assert reg.params["max_input_points"] == 200
    assert reg.params["max_number_bags"] == 10
    assert reg.params["stop_refinement_after"] == 1
    assert reg.params["n_trees_to_refine"] == 10
    assert reg.params["force_cpu"] is True
    assert reg.params["torch_num_threads"] == 1


def test_qlattice_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.QLattice_wrapper.wrapper import QLatticeRegressor

    reg = QLatticeRegressor()
    assert reg.params["n_epochs"] == 100
    assert reg.params["kind"] == "regression"
    assert reg.params["criterion"] == "bic"
    assert reg.params["threads"] == 4


def test_imcts_defaults_are_aligned():
    from scientific_intelligent_modelling.algorithms.iMCTS_wrapper.wrapper import iMCTSRegressor

    reg = iMCTSRegressor()
    assert reg.params["ops"] == ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
    assert reg.params["max_depth"] == 6
    assert reg.params["K"] == 500
    assert reg.params["max_expressions"] == 2000000
    assert reg.params["optimization_method"] == "LN_NELDERMEAD"


def test_ragsr_defaults_are_thread_capped():
    from scientific_intelligent_modelling.algorithms.ragsr_wrapper.wrapper import RAGSRRegressor

    reg = RAGSRRegressor(n_features=2, feature_names=["x0", "x1"], target_name="y")
    assert reg._cpu_num_threads == 4
    assert reg.params["n_pop"] == 200
    assert reg.params["n_gen"] == 100
    assert "cpu_num_threads" not in reg.params
