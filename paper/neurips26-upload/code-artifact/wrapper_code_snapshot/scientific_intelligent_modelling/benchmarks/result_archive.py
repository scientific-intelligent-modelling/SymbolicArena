"""实验结果归档辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_result_payload(
    payload: dict[str, Any],
    *,
    primary_path: str | Path,
    experiment_dir: str | Path | None = None,
    experiment_filename: str = "result.json",
) -> list[Path]:
    """将结果同时写到外层结果路径和实验目录内。

    返回实际写入的唯一文件路径列表。
    """
    paths: list[Path] = [Path(primary_path).resolve()]
    if experiment_dir:
        exp_path = Path(experiment_dir).resolve() / experiment_filename
        paths.append(exp_path)

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for path in unique_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return unique_paths
