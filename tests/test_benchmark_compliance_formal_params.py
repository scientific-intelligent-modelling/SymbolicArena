from __future__ import annotations

import json
from pathlib import Path


PARAMS_ROOT = Path(__file__).resolve().parents[1] / "exp-planning/02.E1选择验证/generated/params"


def test_stage1_qlattice_params_use_timeout_sized_epoch_budget() -> None:
    payload = json.loads((PARAMS_ROOT / "qlattice.json").read_text(encoding="utf-8"))

    assert payload["timeout_in_seconds"] == 3600
    assert payload["n_epochs"] >= 100000


def test_stage1_ragsr_params_use_timeout_sized_generation_budget() -> None:
    payload = json.loads((PARAMS_ROOT / "ragsr.json").read_text(encoding="utf-8"))

    assert payload["timeout_in_seconds"] == 3600
    assert payload["n_gen"] >= 100000
