#!/usr/bin/env python3
"""切换 Core-50 runtime params 的 1h / 24h 预算开关。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "frozen-results/neurips26/final-experiments/03_core50_12alg_final"
BUDGET_DIRS = {
    "1h": RUNTIME_ROOT / "runtime_params_1h_effective",
    "3h": RUNTIME_ROOT / "runtime_params_3h_effective",
    "24h": RUNTIME_ROOT / "runtime_params_24h_effective",
}
ACTIVE_DIR = RUNTIME_ROOT / "runtime_params_active"


def _copytree_atomic(source: Path, target: Path) -> None:
    tmp_target = target.with_name(f".{target.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    shutil.copytree(source, tmp_target)
    if target.exists():
        shutil.rmtree(target)
    tmp_target.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="切换 Core-50 运行预算参数集")
    parser.add_argument("--budget", choices=sorted(BUDGET_DIRS), required=True)
    parser.add_argument(
        "--print-params-root",
        action="store_true",
        help="切换后只输出可传给调度脚本的 --params-root 路径。",
    )
    args = parser.parse_args()

    source = BUDGET_DIRS[args.budget]
    if not source.is_dir():
        raise FileNotFoundError(f"预算参数目录不存在: {source}")

    _copytree_atomic(source, ACTIVE_DIR)
    rel_active = ACTIVE_DIR.relative_to(REPO_ROOT)
    if args.print_params_root:
        print(rel_active)
        return

    print(f"已切换 Core-50 runtime budget: {args.budget}")
    print(f"--params-root {rel_active}")


if __name__ == "__main__":
    main()
