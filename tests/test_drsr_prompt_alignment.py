from pathlib import Path
import json

import numpy as np
import yaml

from scientific_intelligent_modelling.algorithms.drsr_wrapper.drsr.drsr_420.prompt_config import PromptContext
from scientific_intelligent_modelling.algorithms.drsr_wrapper.wrapper import DRSRRegressor


def test_prompt_context_uses_x_variables_and_metadata_semantics():
    ctx = PromptContext(
        n_features=2,
        feature_names=["x0", "x1"],
        dependent_name="y",
        problem_name="MatSci0",
        background="Calculate Stress given Strain and Temperature",
        feature_descriptions=["Strain", "Temperature"],
        target_description="Stress",
    )

    rendered = "\n".join(
        [
            ctx.render_head(),
            ctx.render_instruction(),
            ctx.render_residual_block_title(),
            ctx.render_residual_analysis_prompt("{}", "[[1, 2, 3, 4]]", "return params[0]"),
        ]
    )

    assert "with driving force" not in rendered
    assert "col0" not in rendered
    assert "col1" not in rendered
    assert "MatSci0" not in rendered
    assert "x0 (Strain)" in rendered
    assert "x1 (Temperature)" in rendered
    assert "y (Stress)" in rendered


def test_drsr_wrapper_prompt_semantics_follow_metadata(tmp_path: Path):
    metadata = {
        "dataset": {
            "features": [
                {"name": "epsilon", "description": "Strain"},
                {"name": "T", "description": "Temperature"},
            ],
            "target": {"name": "sigma", "description": "Stress"},
        }
    }
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, allow_unicode=True), encoding="utf-8")

    reg = DRSRRegressor(
        problem_name="MatSci0",
        background="Calculate Stress given Strain and Temperature",
        metadata_path=str(metadata_path),
    )

    feature_names, feature_descriptions, target_description = reg._resolve_prompt_semantics(2)
    assert feature_names == ["x0", "x1"]
    assert feature_descriptions == ["Strain", "Temperature"]
    assert target_description == "Stress"

    spec = reg._build_spec_from_background(np.zeros((8, 2)), np.zeros(8), "Calculate Stress given Strain and Temperature")
    assert "  - x0: Strain" in spec
    assert "  - x1: Temperature" in spec
    assert "  - y: Stress" in spec


def test_drsr_restores_best_history_full_function_with_feature_names(tmp_path: Path):
    exp_dir = tmp_path / "drsr_exp"
    best_dir = exp_dir / "best_history"
    best_dir.mkdir(parents=True)
    best_sample = {
        "iteration": 3,
        "sample_order": 10,
        "score": -0.001,
        "mse": 0.001,
        "nmse": 0.01,
        "params": [1.5, -2.0, 0.25],
        "function": '''
def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Candidate saved by DRSR best_history."""
    return params[0] * strain + params[1] * temp + params[2]
''',
    }
    (best_dir / "best_sample_10.json").write_text(
        json.dumps(best_sample, ensure_ascii=False),
        encoding="utf-8",
    )

    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float)
    y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.25
    reg = DRSRRegressor(
        existing_exp_dir=str(exp_dir),
        niterations=1,
        samples_per_iteration=1,
        n_features=2,
        feature_names=["strain", "temp"],
        target_name="stress",
    )

    reg.fit(X, y)

    eq = reg.get_optimal_equation()
    pred = reg.predict(X)
    assert "params[0] * col0" in eq
    assert "params[1] * col1" in eq
    assert "strain" not in eq
    assert "temp" not in eq
    assert "params[0] * x + params[1] * v" not in eq
    assert np.allclose(pred, y)
