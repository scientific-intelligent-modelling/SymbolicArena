from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from models import noise_tag_for_sigma


def generate_noise_params(
    *,
    source_params_root: Path,
    output_params_root: Path,
    tools: tuple[str, ...],
    noise_sigmas: tuple[float, ...],
    timeout_in_seconds: int,
    progress_snapshot_interval_seconds: int,
) -> dict[str, int]:
    output_params_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for tool in tools:
        source_path = source_params_root / f"{tool}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"参数文件不存在: {source_path}")
        base = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError(f"参数文件必须是 JSON object: {source_path}")
        for sigma in noise_sigmas:
            tag = noise_tag_for_sigma(float(sigma))
            payload: dict[str, Any] = dict(base)
            _apply_tool_budget_overrides(
                tool=tool,
                payload=payload,
                timeout_in_seconds=int(timeout_in_seconds),
            )
            payload["timeout_in_seconds"] = int(timeout_in_seconds)
            payload["progress_snapshot_interval_seconds"] = int(progress_snapshot_interval_seconds)
            payload["train_label_noise_sigma"] = float(sigma)
            payload["train_label_noise_enabled"] = bool(float(sigma) > 0.0)
            (output_params_root / f"{tool}__{tag}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            count += 1
    return {"tools": len(tools), "noise_levels": len(noise_sigmas), "params_files": count}


def _apply_tool_budget_overrides(*, tool: str, payload: dict[str, Any], timeout_in_seconds: int) -> None:
    if timeout_in_seconds <= 3600:
        return
    if tool == "gplearn":
        payload["n_jobs"] = 1
        payload["low_memory"] = True
        return
    if tool != "pyoperon":
        return
    try:
        base_evaluations = int(float(payload.get("max_evaluations", 0)))
    except (TypeError, ValueError):
        return
    if base_evaluations <= 0:
        return
    scaled_evaluations = math.ceil(base_evaluations * timeout_in_seconds / 3600 * 96)
    payload["max_evaluations"] = max(base_evaluations, scaled_evaluations)
