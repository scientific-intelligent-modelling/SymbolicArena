from pathlib import Path


STAGE = Path(__file__).resolve().parent.parent
REPO = STAGE.parents[1]
LEGACY_ROOT = STAGE / "2、experiments"
FULL_ROOT = STAGE / "2.1 full set experimets"
CORE_ROOT = STAGE / "2.2 core50 experiments"
CONDITIONS = {"clean", "noise001", "noise005"}


def experiment_root(seed):
    seed = int(seed)
    if seed == 1314:
        return FULL_ROOT
    if seed in {520, 521, 522}:
        return CORE_ROOT
    raise ValueError(f"未声明的实验seed: {seed}")


def run_directory(algorithm, dataset, seed, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"未声明的实验条件: {condition}")
    return experiment_root(seed) / condition / algorithm / dataset / str(seed)


def relocated_path(value):
    path = Path(value)
    absolute = path if path.is_absolute() else REPO / path
    if not absolute.is_relative_to(LEGACY_ROOT):
        return path
    parts = absolute.relative_to(LEGACY_ROOT).parts
    if len(parts) < 4 or parts[0] not in CONDITIONS or parts[3] not in {"520", "521", "522", "1314"}:
        return path
    return experiment_root(parts[3]).joinpath(*parts)
