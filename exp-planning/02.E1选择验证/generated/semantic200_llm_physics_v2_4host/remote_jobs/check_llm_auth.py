#!/usr/bin/env python3
import json
import os
from pathlib import Path

root = Path("/home/zhangziwen/projects/scientific-intelligent-modelling")
config_dir = root / "exp-planning/02.E1选择验证/llm_configs"
config_names = [
    "benchmark_llm_deepinfra_llama31_8b.config",
    "benchmark_llm_deepinfra_llama31_8b_turbo.config",
]

missing = []
has_env = bool(os.environ.get("DEEPINFRA_API_KEY"))
for name in config_names:
    path = config_dir / name
    if not path.exists():
        missing.append(f"missing:{name}")
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        missing.append(f"invalid:{name}:{type(exc).__name__}")
        continue
    if not has_env and not payload.get("api_key"):
        missing.append(f"no_api_key:{name}")

if missing:
    print("LLM_AUTH_FAIL", ",".join(missing))
    raise SystemExit(2)
print("LLM_AUTH_OK")
