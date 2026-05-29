from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_for_test(name: str) -> ModuleType:
    lib_dir = Path("benchmark-control/compliance/lib")
    sys.path.insert(0, str(lib_dir))
    path = lib_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_compliance_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
