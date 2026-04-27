import json

import numpy as np

from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_drsr_budget_maps_to_llmsr_style_parameters():
    reg = DRSRRegressor(
        niterations=50,
        samples_per_iteration=4,
    )
    niterations, samples_per_iteration, max_samples = reg._resolve_search_budget()

    assert niterations == 50
    assert samples_per_iteration == 4
    assert max_samples == 200


def test_drsr_exp_layout_prefers_exp_path_and_exp_name(tmp_path):
    reg = DRSRRegressor(
        exp_path=str(tmp_path / "experiments"),
        exp_name="demo_run",
        workdir=str(tmp_path / "legacy_workdir"),
    )

    experiments_root, exp_name, workdir = reg._resolve_experiment_layout()

    assert exp_name == "demo_run"
    assert workdir == str(tmp_path / "experiments" / "demo_run")
    assert experiments_root == str(tmp_path / "experiments")


def test_drsr_default_exp_layout_uses_experiments_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reg = DRSRRegressor(problem_name="demo_problem")

    experiments_root, exp_name, workdir = reg._resolve_experiment_layout()

    assert experiments_root == str(tmp_path / "experiments")
    assert workdir == str(tmp_path / "experiments" / exp_name)
    assert exp_name.startswith("drsr_demo_problem_")


def test_drsr_anonymize_renames_features():
    """anonymize=True 只改变 prompt 变量名，真实数据契约不改名。"""
    reg = DRSRRegressor(
        feature_names=["mu", "Nn"],
        target_name="output",
        anonymize=True,
    )
    assert reg._feature_names == ["mu", "Nn"]
    assert reg._target_name == "output"
    assert reg._prompt_feature_names == ["x1", "x2"]
    assert reg._prompt_target_name == "y"
    names, descs, tdesc = reg._resolve_prompt_semantics(2)
    assert names == ["x1", "x2"]
    assert descs is None
    assert tdesc is None


def test_drsr_anonymize_fallback_no_feature_names():
    """anonymize=True 且未传入 feature_names 时按 n_features 生成 x1..xN。"""
    reg = DRSRRegressor(
        n_features=3,
        anonymize=True,
    )
    assert reg._feature_names is None
    assert reg._target_name is None
    assert reg._prompt_feature_names == ["x1", "x2", "x3"]
    assert reg._prompt_target_name == "y"
    names, descs, tdesc = reg._resolve_prompt_semantics(3)
    assert names == ["x1", "x2", "x3"]


def test_drsr_anonymize_default_false():
    """默认 anonymize=False，保持原始变量名。"""
    reg = DRSRRegressor(
        feature_names=["mu", "Nn"],
        target_name="output",
    )
    assert reg._feature_names == ["mu", "Nn"]
    assert reg._target_name == "output"


def test_drsr_anonymize_string_false_is_false():
    """配置文件传入字符串 false 时不应被 bool("false") 误判为 True。"""
    reg = DRSRRegressor(
        feature_names=["mu", "Nn"],
        target_name="output",
        anonymize="false",
    )
    assert reg._anonymize is False
    assert reg._feature_names == ["mu", "Nn"]
    assert reg._target_name == "output"


def test_drsr_compile_one_based_anonymized_variables():
    """DRSR 回放时支持 x1..xN 匿名变量，避免最后一个变量越界。"""
    func = DRSRRegressor._compile_equation("return x1 + x2\n", 2)
    pred = func(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]), np.ones(10))
    np.testing.assert_allclose(pred, np.asarray([4.0, 6.0]))


def test_drsr_serialize_preserves_anonymize_contract():
    reg = DRSRRegressor(
        n_features=2,
        feature_names=["mu", "Nn"],
        target_name="output",
        anonymize=True,
    )
    payload = reg.serialize()
    raw = json.loads(payload)
    assert raw["anonymize"] is True
    restored = DRSRRegressor.deserialize(payload)
    assert restored._anonymize is True
    assert restored._feature_names == ["mu", "Nn"]
    assert restored._target_name == "output"
    assert restored._prompt_feature_names == ["x1", "x2"]
    assert restored._prompt_target_name == "y"
