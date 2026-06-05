#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/examples/stressstrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/acceptance_runs}"
SEED="${SEED:-1314}"
NITERATIONS="${NITERATIONS:-2}"
SAMPLES_PER_ITERATION="${SAMPLES_PER_ITERATION:-1}"
LLM_CONFIG="${1:-${SIM_LLM_CONFIG:-}}"

if [[ -z "${LLM_CONFIG}" ]]; then
  printf '请提供 LLM 配置：SIM_LLM_CONFIG=/path/to/llm.config bash acceptance/测试llmsr.sh\n' >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

RUNNER_PY="$(mktemp "${TMPDIR:-/tmp}/sim_acceptance_llmsr.XXXXXX.py")"
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
    feature_descriptions = [item.get("description") or item["name"] for item in meta["features"]]
    target_description = meta["target"].get("description") or target_name

    def split(name):
        df = pd.read_csv(dataset_dir / name)
        return df[feature_names].to_numpy(float), df[target_name].to_numpy(float)

    return {
        "dataset_dir": dataset_dir,
        "metadata_path": str((dataset_dir / "metadata.yaml").resolve()),
        "feature_names": feature_names,
        "target_name": target_name,
        "feature_descriptions": feature_descriptions,
        "target_description": target_description,
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


dataset_dir, output_root, seed, niterations, samples_per_iteration, llm_config = sys.argv[1:]
seed = int(seed)
output_dir = Path(output_root) / "llmsr_stressstrain"
output_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset(dataset_dir)
X_train, y_train = ds["train"]

params = {
    "exp_path": str(output_dir / "experiments"),
    "exp_name": "llmsr_stressstrain_acceptance",
    "llm_config_path": llm_config,
    "metadata_path": ds["metadata_path"],
    "background": "Calculate Stress from Strain and Temperature using a compact symbolic regression formula.",
    "niterations": int(niterations),
    "samples_per_iteration": int(samples_per_iteration),
    "timeout_in_seconds": 1800,
    "max_params": 10,
    "persist_all_samples": True,
    "seed": seed,
    "n_features": len(ds["feature_names"]),
    "feature_names": ds["feature_names"],
    "target_name": ds["target_name"],
    "feature_descriptions": ds["feature_descriptions"],
    "target_description": ds["target_description"],
}

started = time.time()
result = {"tool": "llmsr", "dataset": "stressstrain", "status": "ok", "params": params}
try:
    reg = SymbolicRegressor("llmsr", problem_name="stressstrain", seed=seed, **params)
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

PYTHONPATH=. conda run -n sim_llm python "${RUNNER_PY}" "$DATASET_DIR" "$OUTPUT_ROOT" "$SEED" "$NITERATIONS" "$SAMPLES_PER_ITERATION" "$LLM_CONFIG"
