#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/examples/stressstrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/acceptance_runs}"
SEED="${SEED:-1314}"
NITERATIONS="${NITERATIONS:-5}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
POPULATIONS="${POPULATIONS:-4}"
MAXSIZE="${MAXSIZE:-20}"
PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT:-${HOME}/pyjuliapkg_pysr_acceptance}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

RUNNER_PY="$(mktemp "${TMPDIR:-/tmp}/sim_acceptance_pysr.XXXXXX.py")"
trap 'rm -f "${RUNNER_PY}"' EXIT

cat > "${RUNNER_PY}" <<'PY'
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scientific_intelligent_modelling.benchmarks.metrics import regression_metrics
from scientific_intelligent_modelling.benchmarks.result_artifacts import safe_export_canonical_artifact
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "metadata.yaml", "r", encoding="utf-8") as f:
        meta_root = yaml.safe_load(f)
    meta = meta_root.get("dataset", meta_root)
    feature_names = [item["name"] for item in meta["features"]]
    target_name = meta["target"]["name"]

    def split(name):
        df = pd.read_csv(dataset_dir / name)
        return df[feature_names].to_numpy(float), df[target_name].to_numpy(float)

    return {
        "dataset_dir": dataset_dir,
        "feature_names": feature_names,
        "target_name": target_name,
        "train": split("train.csv"),
        "valid": split("valid.csv"),
        "id_test": split("id_test.csv"),
        "ood_test": split("ood_test.csv"),
    }


def evaluate(reg, pair):
    X, y = pair
    pred = np.asarray(reg.predict(X), dtype=float).reshape(-1)
    metrics = regression_metrics(y, pred, acc_threshold=0.1)
    return {k: safe_float(v) for k, v in {
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "nmse": metrics["nmse"],
        "acc_0_1": metrics["acc_tau"],
    }.items()}


dataset_dir, output_root, seed, niterations, population_size, populations, maxsize = sys.argv[1:]
seed = int(seed)
output_dir = Path(output_root) / "pysr_stressstrain"
output_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset(dataset_dir)
X_train, y_train = ds["train"]

params = {
    "exp_path": str(output_dir / "experiments"),
    "niterations": int(niterations),
    "population_size": int(population_size),
    "populations": int(populations),
    "maxsize": int(maxsize),
    "procs": 1,
    "progress": False,
    "verbosity": 0,
    "random_state": seed,
    "n_features": len(ds["feature_names"]),
    "feature_names": ds["feature_names"],
    "target_name": ds["target_name"],
}

started = time.time()
result = {"tool": "pysr", "dataset": "stressstrain", "status": "ok", "params": params}
try:
    reg = SymbolicRegressor("pysr", problem_name="stressstrain", seed=seed, **params)
    reg.fit(X_train, y_train)
    result["equation"] = reg.get_optimal_equation()
    result["valid"] = evaluate(reg, ds["valid"])
    result["id_test"] = evaluate(reg, ds["id_test"])
    result["ood_test"] = evaluate(reg, ds["ood_test"])
    artifact, artifact_error = safe_export_canonical_artifact(reg)
    result["canonical_artifact"] = artifact
    result["canonical_artifact_error"] = artifact_error
except Exception as exc:
    result["status"] = "error"
    result["error"] = repr(exc)
finally:
    result["seconds"] = round(time.time() - started, 3)
    result["dataset_dir"] = str(ds["dataset_dir"].resolve())
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "ok":
        raise SystemExit(1)
PY

PYTHONPATH=. PYTHON_JULIAPKG_PROJECT="${PYTHON_JULIAPKG_PROJECT}" \
conda run -n sim_base python "${RUNNER_PY}" "$DATASET_DIR" "$OUTPUT_ROOT" "$SEED" "$NITERATIONS" "$POPULATION_SIZE" "$POPULATIONS" "$MAXSIZE"
