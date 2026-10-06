from __future__ import annotations

import importlib
import json
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from scientific_intelligent_modelling.algorithms.base_wrapper import BaseWrapper
from scientific_intelligent_modelling.onboarding.manifest import load_manifest, repository_root, sha256_file, write_json
from scientific_intelligent_modelling.onboarding.source import verify_source
from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError

from .artifacts import normalize_dgp_artifact


MANIFEST_PATH = repository_root() / "tools/sr_onboarder/manifests/dgp.json"
BEST_FILENAME = ".symbolicarena_current_best.json"
HISTORY_FILENAME = ".symbolicarena_history.jsonl"


@lru_cache(maxsize=1)
def native_modules():
    source = verify_source(load_manifest(MANIFEST_PATH))
    sys.path.insert(0, str(source))
    names = ("const_inference_utils", "const_parallelgp", "const_model", "const_operations")
    modules = tuple(importlib.import_module(name) for name in names)
    for module in modules:
        if Path(module.__file__).resolve().parent != source:
            raise ValueError("DGP module was imported from a different source directory")
    return modules


def primitive_set(n_features):
    from deap import gp

    _, native_gp, _, _ = native_modules()
    pset = gp.PrimitiveSet("MAIN", n_features, prefix="x")
    for name, function, arity in (
        ("add", np.add, 2), ("sub", np.subtract, 2), ("mul", np.multiply, 2),
        ("div", native_gp.protectedDiv, 2), ("sin", np.sin, 1), ("cos", np.cos, 1),
        ("exp", native_gp.protectedExp, 1), ("log", native_gp.protectedLog, 1),
    ):
        pset.addPrimitive(function, arity, name=name)
    return pset


class DGPRegressor(BaseWrapper):
    _DESERIALIZE_WITHOUT_INIT = True
    DEFAULTS = {"niterations": 10, "epochs": 1000, "initial_expressions": 20, "samples_first": 500, "samples_next": 400, "diversification_steps": 20, "n_jobs": 1}
    META = {"exp_path", "exp_name", "problem_name", "seed", "timeout_in_seconds", "n_features", "feature_names", "target_name", "existing_exp_dir", "exp_dir"}

    def __init__(self, **kwargs):
        unknown = set(kwargs) - self.META - set(self.DEFAULTS)
        if unknown:
            raise ValueError("Unsupported DGP parameters: " + ", ".join(sorted(unknown)))
        self.params = {name: int(kwargs.get(name, value)) for name, value in self.DEFAULTS.items()}
        if any(value < 1 for value in self.params.values()):
            raise ValueError("DGP search parameters must be positive")
        self.seed = int(kwargs.get("seed", 1314))
        self.timeout = kwargs.get("timeout_in_seconds")
        if self.timeout is not None and float(self.timeout) <= 0:
            raise ValueError("DGP time budget must be positive")
        self.contract = {name: kwargs.get(name) for name in ("n_features", "feature_names", "target_name")}
        self.directory = None
        if kwargs.get("exp_path") and kwargs.get("exp_name"):
            self.directory = Path(kwargs["exp_path"]) / kwargs["exp_name"]
        existing = kwargs.get("existing_exp_dir") or kwargs.get("exp_dir")
        self.best = None
        self.n_features = None
        self.termination_reason = None
        if existing:
            self.directory = Path(existing)
            current = self.directory / BEST_FILENAME
            if current.is_file():
                self.best = json.loads(current.read_text(encoding="utf-8"))
                self.n_features = int(self.best["n_features"])

    def _generate_expression(self, depth=0):
        if depth >= 2:
            return "x" + str(random.randrange(self.n_features))
        operation = random.choice(("sin", "cos", "log", "exp", "+", "-", "*", "/"))
        left = self._generate_expression(depth + 1)
        if operation in ("+", "-", "*", "/"):
            return "(" + left + " " + operation + " " + self._generate_expression(depth + 1) + ")"
        return operation + "(" + left + ")"

    def _record_best(self, hall, gp_model, X, y, started, deadline, generation, step):
        from sklearn.metrics import r2_score

        if deadline is not None and time.monotonic() >= deadline:
            return
        selected = None
        for individual in hall:
            function = gp_model.toolbox.compile(expr=individual)
            predictions = np.broadcast_to(np.asarray(function(*X.T), dtype=float), (len(y),))
            score = float(r2_score(y, predictions)) if np.isfinite(predictions).all() else -999.0
            if selected is None or score > selected[0]:
                selected = (score, individual)
        if selected is None or (self.best is not None and selected[0] <= self.best["internal_objective_value"]):
            return
        if deadline is not None and time.monotonic() >= deadline:
            return
        score, individual = selected
        manifest = load_manifest(MANIFEST_PATH)
        self.best = {
            "schema": "symbolicarena-native-best-v1",
            "equation": str(individual),
            "n_features": self.n_features,
            "candidate_available": True,
            "algorithm_native_incumbent": True,
            "internal_objective": "training_r2",
            "internal_objective_direction": "max",
            "internal_objective_value": score,
            "native_fitness": list(individual.fitness.values),
            "selection_policy": "dgp_native_training_r2_v1",
            "generation": generation,
            "diversification_step": step,
            "elapsed_seconds": time.monotonic() - started,
            "created_at_unix": time.time(),
            "source": "dgp_native_hall_of_fame",
            "source_revision": manifest["source"]["revision"],
            "source_files": manifest["source"]["files"],
            "variable_index_base": 0,
        }
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            import hashlib

            digest = hashlib.sha256(json.dumps(self.best, sort_keys=True, allow_nan=False).encode()).hexdigest()
            immutable = self.directory / ".symbolicarena_candidates" / (digest + ".json")
            write_json(immutable, self.best)
            self.best["native_candidate_path"] = str(immutable.resolve())
            self.best["native_candidate_sha256"] = sha256_file(immutable)
            with (self.directory / HISTORY_FILENAME).open("a", encoding="utf-8") as output:
                output.write(json.dumps(self.best, allow_nan=False) + "\n")
            write_json(self.directory / BEST_FILENAME, self.best)

    def fit(self, X, y):
        import torch

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features = self._validate_explicit_dataset_contract(X, **self.contract, context="DGPRegressor")
        if X.ndim != 2 or len(y) != len(X) or len(y) < 2 or self.n_features < 1:
            raise ValueError("DGP requires a two-dimensional feature matrix and matching labels")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("DGP inputs must be finite")
        inference, native_gp, _, _ = native_modules()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.params["n_jobs"])
        started = time.monotonic()
        deadline = started + float(self.timeout) if self.timeout is not None else None
        primitives = ["+", "-", "*", "/", "sin", "cos", "exp", "log"] + ["x" + str(index) for index in range(self.n_features)]
        current = set(self._generate_expression() for _ in range(self.params["initial_expressions"]))
        existing = set(current)
        generation = 0
        model = native_gp.GP(
            self.n_features,
            n_jobs=self.params["n_jobs"],
            diversification_steps=self.params["diversification_steps"],
            on_step=lambda hall, step: self._record_best(hall, model, X, y, started, deadline, generation, step),
            deadline=deadline,
        )
        try:
            for generation in range(self.params["niterations"]):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if generation > 0:
                    current.update(self._generate_expression() for _ in range(2))
                samples = self.params["samples_first"] if generation == 0 else self.params["samples_next"]
                expressions = inference.optimize_with_gradient(
                    tuple(sorted(current)), self.params["epochs"], X, y, self.n_features, primitives.copy(), samples, deadline=deadline,
                )
                if not expressions or (deadline is not None and time.monotonic() >= deadline):
                    break
                _, scored, _ = inference.diversification(model, expressions, X, y, primitives, generation)
                scored.sort(key=lambda item: item[1], reverse=True)
                normalized = {inference.re.sub(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])", "rand101", expression) for expression in existing}
                next_expression = next((expression for expression, _score in scored if inference.re.sub(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])", "rand101", expression) not in normalized), None)
                if next_expression is None:
                    self.termination_reason = "expression_space_exhausted"
                    break
                current = {next_expression}
                existing.add(next_expression)
        finally:
            if model.pool is not None:
                model.pool.close()
                model.pool.join()
        if self.termination_reason is None:
            self.termination_reason = "budget_exhausted" if deadline is not None and time.monotonic() >= deadline else "search_completed"
        if self.best is None:
            raise NoValidOutputError("DGP produced no discrete candidate within the time budget")
        return self

    def predict(self, X):
        from deap import gp

        if self.best is None:
            raise ValueError("DGP has no fitted model")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("DGP prediction feature count differs from training")
        pset = primitive_set(self.n_features)
        individual = gp.PrimitiveTree.from_string(self.best["equation"], pset)
        function = gp.compile(individual, pset)
        return np.broadcast_to(np.asarray(function(*X.T), dtype=float), (len(X),)).copy()

    def get_optimal_equation(self):
        if self.best is None:
            raise ValueError("DGP has no fitted model")
        return self.best["equation"]

    def get_total_equations(self):
        return [self.get_optimal_equation()]

    def mark_budget_exhausted(self):
        self.termination_reason = "budget_exhausted"

    def export_canonical_symbolic_program(self):
        artifact = normalize_dgp_artifact(self.get_optimal_equation(), expected_n_features=self.n_features)
        artifact["native_selection"] = self.best
        artifact["native_termination_reason"] = self.termination_reason
        artifact["native_budget_exhausted"] = self.termination_reason == "budget_exhausted"
        return artifact

    def serialize(self):
        return json.dumps({"version": 1, "params": self.params, "seed": self.seed, "n_features": self.n_features, "best": self.best, "termination_reason": self.termination_reason}, allow_nan=False)

    @classmethod
    def deserialize(cls, serialized):
        value = json.loads(serialized)
        if value["version"] != 1:
            raise ValueError("Unsupported DGP model version")
        model = cls(**value["params"], seed=value["seed"])
        model.n_features = value["n_features"]
        model.best = value["best"]
        model.termination_reason = value["termination_reason"]
        return model
