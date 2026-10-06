from __future__ import annotations

import ast
import csv
import importlib
import json
import math
import platform
import shlex
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.benchmarks.runner import load_canonical_dataset, run_benchmark_task, _predict_from_canonical_artifact
from scientific_intelligent_modelling.srkit.config_manager import config_manager
from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

from .manifest import load_manifest, repository_root, sha256_file, write_json
from .source import verify_source


def validate_integration(manifest_path, source_path=None):
    manifest = load_manifest(manifest_path)
    root = repository_root()
    source = verify_source(manifest, source_path)
    tool = manifest["tool_name"]
    registration = config_manager.get_config("toolbox_config")["tool_mapping"][tool]
    if registration["regressor"] != manifest["wrapper_class"] or registration["env"] != manifest["environment"]["name"]:
        raise ValueError("Manifest and algorithm registry disagree")
    environment = config_manager.get_config("envs_config")["env_list"][registration["env"]]
    if environment["python_version"] != manifest["environment"]["python_version"]:
        raise ValueError("Manifest and environment registry disagree")
    wrapper_path = root / "scientific_intelligent_modelling/algorithms" / (tool + "_wrapper") / "wrapper.py"
    tree = ast.parse(wrapper_path.read_text(encoding="utf-8"))
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == manifest["wrapper_class"])
    methods = {node.name for node in wrapper.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {"fit", "predict", "get_optimal_equation", "get_total_equations", "export_canonical_symbolic_program", "serialize", "deserialize"}
    if not required.issubset(methods):
        raise ValueError("The wrapper does not implement all required methods")
    if not registration.get("normalizer") or not registration.get("progress_file") or not registration.get("progress_history_file"):
        raise ValueError("The integration must register its normalizer and native progress files")
    selection = root / manifest["core50"]["selection"]
    if sha256_file(selection) != manifest["core50"]["sha256"]:
        raise ValueError("The frozen Core50 selection checksum differs")
    code_paths = list(wrapper_path.parent.glob("*.py")) + list((root / "scientific_intelligent_modelling/onboarding").glob("*.py"))
    code_paths.extend(root / "scientific_intelligent_modelling/config" / name for name in ("toolbox_config.json", "envs_config.json"))
    return {
        "gate": "structure",
        "passed": True,
        "wrapper_sha256": sha256_file(wrapper_path),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "source": str(source),
        "source_revision": manifest["source"]["revision"],
        "source_files": manifest["source"]["files"],
        "source_license": manifest["source"]["license"],
        "source_distribution": manifest["source"]["mode"],
        "selection_sha256": manifest["core50"]["sha256"],
        "integration_files": {str(path.relative_to(root)): sha256_file(path) for path in code_paths},
    }


def finite_metrics(result):
    if result.get("status") not in ("ok", "timed_out") or not result.get("equation") or not result.get("canonical_artifact"):
        raise ValueError("Benchmark did not deliver a usable native model: " + str(result.get("error")))
    for key in ("valid", "id_test", "ood_test"):
        values = result.get(key)
        if values is None or not isinstance(values.get("nmse"), (int, float)) or not math.isfinite(values["nmse"]):
            raise ValueError("Benchmark numerical metrics are unavailable: " + key)


def accept_api(manifest, dataset_dir, output_root):
    dataset = load_canonical_dataset(dataset_dir)
    module = importlib.import_module("scientific_intelligent_modelling.algorithms." + manifest["tool_name"] + "_wrapper.wrapper")
    wrapper_class = getattr(module, manifest["wrapper_class"])
    output = Path(output_root)
    parameters = dict(manifest["parameters"]["smoke"])
    parameters.update({
        "seed": manifest["core50"]["seeds"][0],
        "n_features": len(dataset.feature_names),
        "feature_names": dataset.feature_names,
        "target_name": dataset.target_name,
        "exp_path": str(output),
        "exp_name": "direct",
    })
    wrapper = wrapper_class(**parameters).fit(dataset.train.X, dataset.train.y)
    sample = dataset.id_test.X[:16]
    native = np.asarray(wrapper.predict(sample), dtype=float)
    if native.shape != (len(sample),) or not np.isfinite(native).all():
        raise ValueError("Native prediction is not a finite sample-length vector")
    artifact = wrapper.export_canonical_symbolic_program()
    replayed = _predict_from_canonical_artifact(artifact, sample)
    np.testing.assert_allclose(replayed, native, rtol=1e-10, atol=1e-10)
    restored = wrapper_class.deserialize(wrapper.serialize())
    np.testing.assert_allclose(restored.predict(sample), native, rtol=0, atol=0)
    regressor = SymbolicRegressor(
        manifest["tool_name"], seed=parameters["seed"],
        exp_path=str(output), exp_name="subprocess",
        **manifest["parameters"]["smoke"],
    )
    regressor.fit(dataset.train.X, dataset.train.y)
    predicted = np.asarray(regressor.predict(sample), dtype=float)
    if predicted.shape != native.shape or not np.isfinite(predicted).all() or not regressor.get_optimal_equation():
        raise ValueError("The subprocess API did not deliver valid predictions and an equation")
    report = {
        "gate": "native_api",
        "passed": True,
        "dataset_dir": str(dataset.dataset_dir),
        "parameters": parameters,
        "raw_equation": wrapper.get_optimal_equation(),
        "artifact": artifact,
        "native_prediction": native.tolist(),
        "replayed_prediction": np.asarray(replayed).tolist(),
        "subprocess_equation": regressor.get_optimal_equation(),
        "subprocess_prediction": predicted.tolist(),
    }
    write_json(output / "native_api.json", report)
    return report


def core50_inputs(manifest, datasets_root, output_root):
    root = repository_root()
    selection = root / manifest["core50"]["selection"]
    if sha256_file(selection) != manifest["core50"]["sha256"]:
        raise ValueError("Core50 selection changed")
    with selection.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 50 or len({row["dataset_id"] for row in rows}) != 50:
        raise ValueError("Core50 must contain 50 unique task IDs")
    datasets_root = Path(datasets_root).resolve()
    inputs = []
    for row in rows:
        relative = Path(row["dataset_rel"])
        if relative.parts[0] != "sim-datasets-data" or ".." in relative.parts:
            raise ValueError("Unexpected frozen dataset path")
        path = datasets_root.joinpath(*relative.parts[1:])
        files = {name: sha256_file(path / name) for name in ("metadata.yaml", "train.csv", "valid.csv", "id_test.csv", "ood_test.csv")}
        dataset = load_canonical_dataset(path)
        inputs.append({"dataset_id": row["dataset_id"], "dataset_dir": str(path), "n_features": len(dataset.feature_names), "files": files})
    write_json(Path(output_root) / "input_manifest.json", {"selection_sha256": manifest["core50"]["sha256"], "datasets": inputs})
    return inputs


def accept_benchmark(manifest, dataset_dir, output_root, parameters, seed):
    result_path = run_benchmark_task(
        tool_name=manifest["tool_name"], dataset_dir=dataset_dir,
        output_root=output_root, seed=seed, params_override=parameters,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    finite_metrics(result)
    return {"dataset_dir": str(dataset_dir), "seed": seed, "passed": True, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}


def accept_tmux_benchmark(manifest_path, dataset_dir, output_root, parameters, seed):
    manifest = load_manifest(manifest_path)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    write_json(request_path, {"manifest": str(Path(manifest_path).resolve()), "dataset_dir": str(dataset_dir), "output_root": str(output / "results"), "parameters": parameters, "seed": seed})
    session = "onboard_" + manifest["tool_name"] + "_" + uuid.uuid4().hex[:12]
    source = verify_source(manifest)
    command = [
        "env", manifest["source"]["environment_variable"] + "=" + str(source),
        "TMPDIR=" + str(output), "PYTHONPATH=" + str(repository_root()),
        "bash", str(repository_root() / "tools/sr_onboarder/run_task.sh"),
        sys.executable, str(request_path), str(output / "task.log"),
    ]
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(command)], check=True)
    started = time.monotonic()
    allowance = float(parameters["timeout_in_seconds"]) + 180
    while subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        if time.monotonic() - started > allowance:
            subprocess.run(["tmux", "kill-session", "-t", session], check=True)
            raise TimeoutError("Integration task exceeded its allowance: " + str(output / "task.log"))
        time.sleep(0.5)
    completion_path = output / "completion.json"
    if not completion_path.is_file():
        raise RuntimeError("Integration task failed; inspect " + str(output / "task.log"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    result_path = Path(completion["result_path"])
    if sha256_file(result_path) != completion["result_sha256"]:
        raise ValueError("The task result changed after completion")
    finite_metrics(json.loads(result_path.read_text(encoding="utf-8")))
    return {"dataset_dir": str(dataset_dir), "seed": seed, "passed": True, "session": session, "task_log": str(output / "task.log"), **completion}


def run_acceptance(manifest_path, output_root, stage, dataset_dir=None, datasets_root=None, workers=1, use_tmux=False):
    manifest = load_manifest(manifest_path)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "symbolicarena-integration-acceptance-v1",
        "tool_name": manifest["tool_name"],
        "stage": stage,
        "started_at_unix": time.time(),
        "python": platform.python_version(),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "passed": False,
        "gates": [],
        "formal_six_axis_ready": False,
        "unresolved_axes": ["SYM", "MIN", "EFF", "STAB"],
    }
    report["gates"].append(validate_integration(manifest_path))
    write_json(output / "acceptance.json", report)
    if stage in ("smoke", "all"):
        if dataset_dir is None:
            raise ValueError("The smoke stage requires --dataset-dir")
        api = accept_api(manifest, dataset_dir, output / "api")
        report["gates"].append({"gate": api["gate"], "passed": True, "evidence": str(output / "api/native_api.json")})
        benchmark = accept_benchmark(manifest, dataset_dir, output / "benchmark", manifest["parameters"]["smoke"], manifest["core50"]["seeds"][0])
        report["gates"].append({"gate": "benchmark", **benchmark})
        write_json(output / "acceptance.json", report)
    if stage in ("budget", "all"):
        if dataset_dir is None:
            raise ValueError("Budget acceptance requires --dataset-dir")
        parameters = manifest["parameters"]["budget"]
        budget = float(parameters["timeout_in_seconds"])
        if budget < 60:
            raise ValueError("Minute-level acceptance needs a budget of at least 60 seconds")
        budget_run = accept_benchmark(manifest, dataset_dir, output / "budget", parameters, manifest["core50"]["seeds"][0])
        result_path = Path(budget_run["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("budget_exhausted"):
            raise ValueError("Budget acceptance did not reach its configured time limit")
        if result["seconds"] > budget + 30:
            raise ValueError("The algorithm exceeded the budget termination allowance")
        snapshots = sorted((result_path.parent / "progress").glob("minute_*.json"))
        snapshots = [path for path in snapshots if json.loads(path.read_text()).get("candidate_available")]
        if not snapshots:
            raise ValueError("No minute-level candidate snapshot was delivered")
        for path in snapshots:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            evidence = snapshot["native_evidence"]
            if sha256_file(Path(evidence["source_path"])) != evidence["source_sha256"]:
                raise ValueError("Minute snapshot native evidence changed")
            if snapshot["internal_objective_value"] is None:
                raise ValueError("Minute snapshot native training objective is missing")
        report["gates"].append({"gate": "budget_and_minutes", "passed": True, "result": budget_run, "snapshots": [str(path) for path in snapshots]})
        write_json(output / "acceptance.json", report)
    if stage in ("core50", "all"):
        if datasets_root is None:
            raise ValueError("Core50 requires --datasets-root")
        inputs = core50_inputs(manifest, datasets_root, output)
        report["core50_runs"] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_function = accept_tmux_benchmark if use_tmux else accept_benchmark
            task_manifest = manifest_path if use_tmux else manifest
            futures = {
                executor.submit(
                    task_function, task_manifest, item["dataset_dir"],
                    output / "core50" / item["dataset_id"] / ("seed" + str(seed)),
                    manifest["parameters"]["core50"], seed,
                ): (item, seed)
                for item in inputs for seed in manifest["core50"]["seeds"]
            }
            for future in as_completed(futures):
                item, seed = futures[future]
                result = future.result()
                result["dataset_id"] = item["dataset_id"]
                report["core50_runs"].append(result)
                write_json(output / "acceptance.json", report)
        expected = len(inputs) * len(manifest["core50"]["seeds"])
        if len(report["core50_runs"]) != expected:
            raise ValueError("The complete Core50 run set is missing")
        report["gates"].append({"gate": "core50", "passed": True, "runs": expected, "input_manifest": str(output / "input_manifest.json")})
    report["passed"] = True
    report["completed_at_unix"] = time.time()
    write_json(output / "acceptance.json", report)
    return report
