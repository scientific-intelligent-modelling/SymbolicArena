"""Wrapper for TPSR (Transformer-based Planning for Symbolic Regression)."""

from __future__ import annotations

import base64
import os
import pickle
import re
import tempfile
import sys
import json
import time
import torch
import numpy as np
import shutil
from typing import Optional

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_tpsr_artifact


class TPSRRegressor(BaseWrapper):
    _PROGRESS_STATE_FILENAME = ".tpsr_current_best.json"

    def __init__(self, **kwargs):
        # note noteenvironment note
        self.params = kwargs
        self._contract_n_features = self.params.get("n_features")
        self._contract_feature_names = self.params.get("feature_names")
        self._contract_target_name = self.params.get("target_name")
        self.model = None
        self.best_tree = None
        self.all_trees = []
        self._predict_fn = None
        self._backend_params = {}
        self._predict_variable_names = []
        self._n_features = None
        self._exp_path = self.params.get("exp_path")
        self._exp_name = self.params.get("exp_name")
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)

        # note default note
        self.params.setdefault("backbone_model", "e2e")
        # note vendored note TPSR README / run.sh note 
        # default note bagging note note bag 200 note note 10 note bag 
        self.params.setdefault("max_input_points", 200)
        self.params.setdefault("max_number_bags", 10)
        self.params.setdefault("stop_refinement_after", 1)
        self.params.setdefault("n_trees_to_refine", 10)
        self.params.setdefault("rescale", True)
        self.params.setdefault("beam_size", 10)
        self.params.setdefault("beam_type", "sampling")
        self.params.setdefault("no_seq_cache", False)
        self.params.setdefault("no_prefix_cache", True)
        self.params.setdefault("width", 3)  # Top-k in TPSR's expansion step
        self.params.setdefault("num_beams", 1)  # Beam size in TPSR's evaluation
        self.params.setdefault("rollout", 3)  # Number of rollouts in TPSR
        self.params.setdefault("horizon", 200)  # Horizon of lookahead planning
        self.params.setdefault("seed", 23)
        self.params.setdefault("cpu", True)
        self.params.setdefault("cpu_num_threads", 4)
        self.params.setdefault("cpu_interop_threads", 1)
        self.params.setdefault("train_value", False)
        self.params.setdefault("lam", 0.1)
        # note benchmark note note README note 
        # note bag note note wrapper note reward/refinement
        # note note CPU note OOM kill 
        self.params.setdefault("reward_sample_limit", 2048)

        # NeSymReS note
        self.params.setdefault("nesymres_eq_setting_path", os.path.join("nesymres", "jupyter", "100M", "eq_setting.json"))
        self.params.setdefault("nesymres_cfg_path", os.path.join("nesymres", "jupyter", "100M", "config.yaml"))
        self.params.setdefault("nesymres_model_path", None)
        self.params.setdefault("symbolicregression_model_path", self._shared_symbolic_model_path())
        self.params.setdefault("symbolicregression_model_url", "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt")

    @classmethod
    def _resolve_progress_state_path(cls, exp_path, exp_name):
        if not isinstance(exp_path, str) or not exp_path.strip():
            return None
        if not isinstance(exp_name, str) or not exp_name.strip():
            return None
        return os.path.join(
            os.path.abspath(exp_path.strip()),
            exp_name.strip(),
            cls._PROGRESS_STATE_FILENAME,
        )

    @staticmethod
    def _shared_symbolic_model_path():
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(root_dir, "..", "e2esr_wrapper", "model.pt"))

    def _runtime_import_paths(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        tpsr_dir = os.path.join(root_dir, "tpsr")
        return [
            tpsr_dir,
            os.path.join(tpsr_dir, "nesymres", "src"),
        ]

    @staticmethod
    def _downsample_reward_arrays(X, y, limit: Optional[int]):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if y_arr.ndim > 1:
            y_arr = y_arr.reshape(-1)

        try:
            limit_val = int(limit) if limit is not None else None
        except Exception:
            limit_val = None

        if not limit_val or limit_val <= 0 or X_arr.shape[0] <= limit_val:
            return X_arr, y_arr

        # note note 
        indices = np.linspace(0, X_arr.shape[0] - 1, num=limit_val, dtype=int)
        indices = np.unique(indices)
        return X_arr[indices], y_arr[indices]

    def _set_parser_attr(self, args, name: str, value):
        if value is None:
            return
        try:
            setattr(args, name, value)
        except Exception:
            # notecurrent parser note note
            pass

    def _resolve_tpsr_path(self, rel_path: str):
        if not rel_path:
            return None

        if os.path.isabs(rel_path):
            return rel_path

        root_dir = os.path.dirname(os.path.abspath(__file__))
        tpsr_dir = os.path.join(root_dir, "tpsr")
        candidate = os.path.join(tpsr_dir, rel_path)
        if os.path.isfile(candidate):
            return candidate

        return os.path.abspath(rel_path)

    def _download_if_absent(self, url: str, target_path: str):
        if not url:
            return False
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            import urllib.request

            with urllib.request.urlopen(url) as response, open(target_path, "wb") as output:
                shutil.copyfileobj(response, output)
            return True
        except Exception as e:
            raise RuntimeError(f"TPSR note: {url}, reason: {e}")

    def _ensure_e2e_model(self):
        requested_path = self._resolve_tpsr_path(self.params.get("symbolicregression_model_path"))
        shared_path = self._shared_symbolic_model_path()
        target_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tpsr",
            "symbolicregression",
            "weights",
            "model.pt",
        )
        for candidate in (requested_path, shared_path, target_path):
            if candidate and os.path.isfile(candidate):
                resolved = os.path.abspath(candidate)
                os.environ["SIM_SYMBOLICREGRESSION_MODEL_PATH"] = resolved
                self.params["symbolicregression_model_path"] = resolved
                return resolved

        download_url = self.params.get("symbolicregression_model_url")
        if not os.path.isfile(shared_path):
            if not download_url:
                raise FileNotFoundError(
                    "note e2e note note symbolicregression_model_url"
                )
            self._download_if_absent(download_url, shared_path)
        resolved = os.path.abspath(shared_path)
        os.environ["SIM_SYMBOLICREGRESSION_MODEL_PATH"] = resolved
        self.params["symbolicregression_model_path"] = resolved
        return resolved

    def _resolve_nesymres_weights(self, cfg):
        model_path = self.params.get("nesymres_model_path")
        if not model_path:
            model_path = getattr(cfg, "model_path", None)

        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend(
            [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr", "nesymres", "weights", "10MCompleted.ckpt"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr", "nesymres", "weights", "10M.ckpt"),
            ]
        )

        for candidate in candidates:
            if not candidate:
                continue
            candidate = self._resolve_tpsr_path(candidate)
            if candidate and os.path.isfile(candidate):
                return candidate

        return None

    def _normalize_equation(self, expr):
        if expr is None:
            return ""

        text = str(expr)
        # note token note note Symbolic SR note 
        replacements = {
            "add": "+",
            "mul": "*",
            "sub": "-",
            "pow": "**",
            "inv": "1/",
        }
        for op, op_target in replacements.items():
            text = re.sub(rf"\b{op}\b", op_target, text)
        # note pow2/pow3 note
        text = re.sub(r"\bpow2\b", "**2", text)
        text = re.sub(r"\bpow3\b", "**3", text)
        text = re.sub(r"\bpow4\b", "**4", text)
        return text

    def _normalize_equation_list(self, trees):
        equations = []
        for tree in trees:
            if tree is None:
                continue
            equations.append(self._normalize_equation(tree))
        return equations

    def _write_progress_state(self, payload):
        if not self._progress_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._progress_state_path), exist_ok=True)
            with open(self._progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _sequence_length(sequence):
        if sequence is None:
            return None
        try:
            return int(len(sequence))
        except Exception:
            return None

    @staticmethod
    def _resolve_feature_dimension(X) -> int:
        X_arr = np.asarray(X)
        if X_arr.ndim <= 1:
            return 1
        return int(X_arr.shape[1])

    @staticmethod
    def _extract_variable_indices(equation: str) -> set[int]:
        text = "" if equation is None else str(equation)
        matches = set()
        for match in re.finditer(r"\bx_(\d+)\b|\bx(\d+)\b", text):
            idx_text = match.group(1) or match.group(2)
            try:
                matches.add(int(idx_text))
            except Exception:
                continue
        # note one-based note note x0/x_0 
        # note x1/x_1, x2/x_2 ... note 
        has_zero_based = bool(re.search(r"\bx_?0\b", text))
        if matches and not has_zero_based and min(matches) >= 1:
            matches = {idx - 1 for idx in matches}
        return matches

    @classmethod
    def _equation_within_feature_budget(cls, equation: str, n_features: Optional[int]) -> bool:
        if n_features is None:
            return True
        if n_features <= 0:
            return False
        indices = cls._extract_variable_indices(equation)
        return all(idx < int(n_features) for idx in indices)

    @classmethod
    def _project_equation_to_feature_budget(cls, equation: str, n_features: Optional[int]) -> str:
        """notecurrent note 0 

        TPSR note note `x_9` note
        current note note token note runner 
        note wrapper note note 0 
        """
        text = "" if equation is None else str(equation)
        if n_features is None:
            return text
        try:
            budget = int(n_features)
        except Exception:
            return text
        if budget <= 0:
            return text

        has_zero_based = bool(re.search(r"\bx_?0\b", text))

        def _replace(match):
            original = match.group(0)
            idx_text = match.group(1) or match.group(2)
            try:
                idx = int(idx_text)
            except Exception:
                return original

            if has_zero_based:
                return original if idx < budget else "0"
            return original if idx <= budget else "0"

        return re.sub(r"\bx_(\d+)\b|\bx(\d+)\b", _replace, text)

    def _capture_runtime_feature_context(self, X) -> int:
        """notecurrent note 

        note TPSR noteenvironment note noteenvironment note
        default 10 notecurrent note noteConsistency 
        note note 
        """
        n_features = self._resolve_feature_dimension(X)
        self._n_features = n_features
        self._predict_variable_names = [f"x_{i}" for i in range(n_features)]
        return n_features

    def _is_current_task_equation_valid(self, equation: str) -> bool:
        return self._equation_within_feature_budget(equation, self._n_features)

    def _emit_progress_equation(self, *, equation, score=None, complexity=None, source=None):
        if not isinstance(equation, str) or not equation.strip():
            return
        projected = self._project_equation_to_feature_budget(equation, self._n_features)
        normalized = self._normalize_equation(projected)
        if not self._is_current_task_equation_valid(normalized):
            return
        payload = {
            "equation": normalized,
            "score": float(score) if isinstance(score, (int, float, np.floating)) else None,
            "complexity": int(complexity) if isinstance(complexity, (int, float, np.integer, np.floating)) else None,
            "source": source,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._write_progress_state(payload)

    def _extract_best_new_candidate(self, candidate_programs, reward_fn, *, checked_count, best_reward):
        best_seq = None
        new_checked = len(candidate_programs)
        for seq in candidate_programs[checked_count:]:
            try:
                reward = reward_fn(seq)
            except Exception:
                continue
            if reward is None:
                continue
            try:
                reward = float(reward)
            except Exception:
                continue
            if reward > best_reward:
                best_reward = reward
                best_seq = seq
        return new_checked, best_reward, best_seq

    def _sequence_to_e2e_expression(self, args, model, equation_env, sequence, samples):
        from symbolicregression.e2e_model import pred_for_sample_no_refine, refine_for_sample

        refined_expr = None
        raw_expr = None
        try:
            _, refined_expr, _ = refine_for_sample(
                args,
                model,
                equation_env,
                sequence,
                samples["x_to_fit"],
                samples["y_to_fit"],
            )
        except Exception:
            refined_expr = None
        try:
            _, raw_expr, _ = pred_for_sample_no_refine(
                model,
                equation_env,
                sequence,
                samples["x_to_fit"],
            )
        except Exception:
            raw_expr = None

        for expr in (refined_expr, raw_expr):
            if isinstance(expr, str) and expr.strip():
                return self._normalize_equation(expr)
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        state["_predict_fn"] = None
        state["_backend_params"] = {
            "backend": getattr(self._backend_params, "get", lambda k, d=None: None)("backend"),
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None
        self._backend_params = dict(
            (k, v) for k, v in (self._backend_params or {}).items() if k == "backend"
        )
        if not self.all_trees:
            self.all_trees = []
        self.all_trees = self._normalize_equation_list(self.all_trees)
        if self.best_tree is None:
            self.best_tree = self.all_trees[0] if self.all_trees else None
        if self.best_tree is not None:
            self.best_tree = self._normalize_equation(self.best_tree)
        self.all_trees = list(self.all_trees or [])
        n_features = getattr(self, "_n_features", None)
        var_names = list(getattr(self, "_predict_variable_names", []) or [])
        if not var_names and n_features:
            var_names = [f"x_{i}" for i in range(int(n_features))]
        self._predict_fn = None
        if self.best_tree and var_names:
            self._predict_fn = self._build_predictor_from_expression(self.best_tree, var_names)

    def _build_predictor_from_expression(self, expr_text: str, variable_names):
        if not expr_text:
            return None

        expr_text = self._project_equation_to_feature_budget(expr_text, len(variable_names))
        expr_text = self._normalize_equation(expr_text)
        if not self._equation_within_feature_budget(expr_text, len(variable_names)):
            print("TPSR note note")
            return None
        try:
            import sympy as sp

            expr = sp.sympify(expr_text)
            symbols = [sp.Symbol(v) for v in variable_names]
            fn = sp.lambdify(symbols, expr, modules="numpy")
            self._backend_params["predict_symbols"] = symbols
            self._backend_params["predict_expr"] = expr

            def _predict(X):
                X_arr = np.asarray(X)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(1, -1)
                feature_count = X_arr.shape[1]
                available_args = []
                for idx, sym in enumerate(symbols):
                    if idx < feature_count:
                        available_args.append(X_arr[:, idx])
                    else:
                        # note 0 note NeSymReS note
                        available_args.append(np.zeros(X_arr.shape[0], dtype=float))
                outputs = fn(*available_args)
                outputs_arr = np.asarray(outputs, dtype=float)
                if outputs_arr.ndim == 0:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr))
                elif outputs_arr.ndim >= 1 and outputs_arr.size == 1:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr.reshape(-1)[0]))
                outputs_arr = outputs_arr.reshape(-1)
                if outputs_arr.shape[0] != X_arr.shape[0]:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr[0]))
                return outputs_arr

            return _predict

        except Exception as e:
            print(f"TPSR note note: {str(e)}")
            return None

    def _fit_e2e(self, X, y, equation_env, args, samples):
        self._ensure_e2e_model()

        from symbolicregression.e2e_model import Transformer
        from symbolicregression.e2e_model import pred_for_sample_no_refine, refine_for_sample
        from dyna_gym.agents.uct import UCT
        from dyna_gym.agents.mcts import update_root
        from rl_env import RLEnv
        from default_pi import E2EHeuristic

        # note TPSR note
        model = Transformer(params=args, env=equation_env, samples=samples)
        model.to(args.device)

        # note RL environment
        rl_env = RLEnv(
            params=args,
            samples=samples,
            equation_env=equation_env,
            model=model,
        )

        # note TPSR planner
        dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=args.width,
            num_beams=args.num_beams,
            horizon=args.horizon,
            device=args.device,
            use_seq_cache=not args.no_seq_cache,
            use_prefix_cache=not args.no_prefix_cache,
            length_penalty=args.beam_length_penalty if hasattr(args, "beam_length_penalty") else 1.0,
            train_value_mode=args.train_value if hasattr(args, "train_value") else False,
            debug=args.debug,
        )

        # note UCT note
        agent = UCT(
            action_space=[],
            gamma=1.0,
            ucb_constant=args.ucb_constant if hasattr(args, "ucb_constant") else 1.0,
            horizon=args.horizon,
            rollouts=args.rollout,
            dp=dp,
            width=args.width,
            reuse_tree=True,
            alg=args.uct_alg if hasattr(args, "uct_alg") else "uct",
            ucb_base=args.ucb_base if hasattr(args, "ucb_base") else 4,
        )

        # note
        done = False
        s = rl_env.state
        checked_candidate_count = 0
        best_progress_reward = float("-inf")
        for _ in range(args.horizon):
            if done or len(s) >= args.horizon:
                break
            act = agent.act(rl_env, done)
            s, _, done, _ = rl_env.step(act)
            update_root(agent, act, s)
            dp.update_cache(s)

            checked_candidate_count, best_progress_reward, best_seq = self._extract_best_new_candidate(
                dp.candidate_programs,
                rl_env.get_reward,
                checked_count=checked_candidate_count,
                best_reward=best_progress_reward,
            )
            if best_seq is not None:
                progress_expr = self._sequence_to_e2e_expression(args, model, equation_env, best_seq, samples)
                self._emit_progress_equation(
                    equation=progress_expr,
                    score=best_progress_reward,
                    complexity=self._sequence_length(best_seq),
                    source="e2e_candidate",
                )

            if done and s is not None:
                try:
                    terminal_reward = float(rl_env.get_reward(s))
                except Exception:
                    terminal_reward = None
                if terminal_reward is not None and terminal_reward > best_progress_reward:
                    best_progress_reward = terminal_reward
                    progress_expr = self._sequence_to_e2e_expression(args, model, equation_env, s, samples)
                    self._emit_progress_equation(
                        equation=progress_expr,
                        score=best_progress_reward,
                        complexity=self._sequence_length(s),
                        source="e2e_terminal",
                    )

        # `s` notecurrent horizon note note 
        # note default policy note note note `s` 
        candidate_sequences = []
        for seq in getattr(dp, "candidate_programs", []) or []:
            if seq is None:
                continue
            candidate_sequences.append(seq)
        if done and s is not None:
            candidate_sequences.append(s)

        if not candidate_sequences:
            raise RuntimeError("TPSR note")

        def _reward_of(seq):
            try:
                return float(rl_env.get_reward(seq, mode="test"))
            except Exception:
                return float("-inf")

        best_sequence = max(candidate_sequences, key=_reward_of)

        self.all_trees = []

        # TPSR note note E2E note 
        # note  MCTS + refinement  note note refinement note note no-ref note 
        try:
            _, refined_expr, refined_trees = refine_for_sample(
                args,
                model,
                equation_env,
                best_sequence,
                samples["x_to_fit"],
                samples["y_to_fit"],
            )
        except Exception:
            refined_expr, refined_trees = None, []

        try:
            _, raw_expr, raw_trees = pred_for_sample_no_refine(
                model,
                equation_env,
                best_sequence,
                samples["x_to_fit"],
            )
        except Exception:
            raw_expr, raw_trees = None, []

        candidate_exprs = []
        for expr in (refined_expr, raw_expr):
            if isinstance(expr, str) and expr.strip():
                candidate_exprs.append(expr)

        for tree_group in (refined_trees, raw_trees):
            if not isinstance(tree_group, list):
                continue
            for tree in tree_group:
                if tree is None:
                    continue
                text = str(tree.infix() if hasattr(tree, "infix") else tree).strip()
                if text:
                    candidate_exprs.append(text)

        # note
        seen = set()
        self.all_trees = []
        for expr in candidate_exprs:
            projected = self._project_equation_to_feature_budget(expr, self._n_features)
            normalized = self._normalize_equation(projected)
            if not self._is_current_task_equation_valid(normalized):
                continue
            if normalized not in seen:
                seen.add(normalized)
                self.all_trees.append(normalized)

        if not self.all_trees:
            raise RuntimeError("TPSR notecurrent note")

        self.best_tree = self.all_trees[0]
        self.model = model
        self._n_features = X.shape[1]
        self._predict_variable_names = [f"x_{i}" for i in range(self._n_features)]
        self._backend_params["backend"] = "e2e"
        self._backend_params["search_state"] = best_sequence
        self._predict_fn = self._build_predictor_from_expression(
            self.best_tree, self._predict_variable_names
        )
        self._emit_progress_equation(
            equation=self.best_tree,
            score=best_progress_reward if best_progress_reward != float("-inf") else None,
            complexity=self._sequence_length(best_sequence),
            source="e2e_final",
        )

    def _resolve_nesymres_resources(self, cfg):
        eq_setting_path = self.params.get("nesymres_eq_setting_path", "")
        cfg_path = self.params.get("nesymres_cfg_path", "")
        if not eq_setting_path:
            raise FileNotFoundError("NeSymReS note nesymres_eq_setting_path")
        if not cfg_path:
            raise FileNotFoundError("NeSymReS note nesymres_cfg_path")

        eq_setting_path = self._resolve_tpsr_path(eq_setting_path)
        cfg_path = self._resolve_tpsr_path(cfg_path)

        if not os.path.isfile(eq_setting_path):
            raise FileNotFoundError(f"note NeSymReS note: {eq_setting_path}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"note NeSymReS note: {cfg_path}")

        model_path = self._resolve_nesymres_weights(cfg)
        if not model_path:
            raise FileNotFoundError(
                "note NeSymReS note note nesymres_model_path note `nesymres/weights/` note ckpt note"
            )

        return eq_setting_path, cfg_path, model_path

    def _fit_nesymres(self, X, y, args, samples):
        import json
        from functools import partial
        from reward import compute_reward_nesymres
        from nesymres.src.nesymres.architectures.model import Model
        from nesymres.dclasses import FitParams, BFGSParams
        from dyna_gym.agents.uct import UCT
        from dyna_gym.agents.mcts import update_root
        from rl_env import RLEnv
        from default_pi import NesymresHeuristic
        import omegaconf

        cfg_path = self._resolve_tpsr_path(self.params.get("nesymres_cfg_path"))
        if not cfg_path or not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"NeSymReS note: {cfg_path}")
        cfg = omegaconf.OmegaConf.load(cfg_path)
        eq_setting_path = self._resolve_tpsr_path(self.params.get("nesymres_eq_setting_path"))
        eq_setting_path, _, weight_path = self._resolve_nesymres_resources(cfg)
        with open(eq_setting_path, "r", encoding="utf-8") as f:
            eq_setting = json.load(f)

        cfg_obj = cfg
        try:
            if getattr(cfg, "model", None) is not None:
                cfg_obj = cfg.model
        except Exception:
            cfg_obj = cfg
        self._set_parser_attr(cfg_obj, "cfg", cfg)
        bfgs_cfg = BFGSParams(
            activated=cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

        fit_params = FitParams(
            word2id=eq_setting["word2id"],
            id2word={int(k): v for k, v in eq_setting["id2word"].items()},
            una_ops=eq_setting["una_ops"],
            bin_ops=eq_setting["bin_ops"],
            total_variables=list(eq_setting["total_variables"]),
            total_coefficients=list(eq_setting["total_coefficients"]),
            rewrite_functions=list(eq_setting["rewrite_functions"]),
            bfgs=bfgs_cfg,
            beam_size=cfg.inference.beam_size if hasattr(cfg, "inference") else 5,
        )

        cfg.model_path = weight_path
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            import torch.serialization

            torch.serialization.add_safe_globals([ModelCheckpoint])
        except Exception:
            pass
        model = Model.load_from_checkpoint(weight_path, cfg=cfg.architecture)
        model.eval()
        if torch.cuda.is_available() and not args.cpu:
            model.cuda()

        fitfunc = partial(model.fitfunc, cfg_params=fit_params)
        y_flat = np.asarray(y).reshape(-1)
        baseline_output = fitfunc(X, y_flat)
        all_preds = baseline_output.get("all_bfgs_preds", [])
        best_preds = baseline_output.get("best_bfgs_preds", [])
        baseline_expr = best_preds[0] if best_preds else (all_preds[0] if all_preds else None)

        rl_env = RLEnv(
            params=args,
            samples=samples,
            model=model,
            cfg_params=fit_params,
        )
        model.to_encode(X, y_flat, cfg_params=fit_params)

        dp = NesymresHeuristic(
            rl_env=rl_env,
            model=model,
            k=args.width,
            num_beams=args.num_beams,
            horizon=args.horizon,
            device=args.device,
            use_seq_cache=not args.no_seq_cache,
            use_prefix_cache=not args.no_prefix_cache,
            length_penalty=args.beam_length_penalty if hasattr(args, "beam_length_penalty") else 1.0,
            cfg_params=fit_params,
            train_value_mode=args.train_value if hasattr(args, "train_value") else False,
            debug=args.debug,
        )

        agent = UCT(
            action_space=[],
            gamma=1.0,
            ucb_constant=args.ucb_constant if hasattr(args, "ucb_constant") else 1.0,
            horizon=args.horizon,
            rollouts=args.rollout,
            dp=dp,
            width=args.width,
            reuse_tree=True,
            alg=args.uct_alg if hasattr(args, "uct_alg") else "uct",
            ucb_base=args.ucb_base if hasattr(args, "ucb_base") else 4,
        )

        done = False
        s = rl_env.state
        checked_candidate_count = 0
        best_progress_reward = float("-inf")
        for _ in range(200 if not getattr(args, "sample_only", False) else 1):
            if done or len(s) >= args.horizon:
                break
            act = agent.act(rl_env, done)
            s, _, done, _ = rl_env.step(act)
            update_root(agent, act, s)
            dp.update_cache(s)

            def _reward_and_expr(seq):
                _, reward_val, pred_expr = compute_reward_nesymres(model.X, model.y, seq, fit_params)
                return reward_val, pred_expr

            for seq in dp.candidate_programs[checked_candidate_count:]:
                try:
                    reward_val, pred_expr = _reward_and_expr(seq)
                    reward_val = float(reward_val) if reward_val is not None else None
                except Exception:
                    continue
                if reward_val is None or reward_val <= best_progress_reward:
                    continue
                best_progress_reward = reward_val
                self._emit_progress_equation(
                    equation=pred_expr,
                    score=best_progress_reward,
                    complexity=self._sequence_length(seq),
                    source="nesymres_candidate",
                )
            checked_candidate_count = len(dp.candidate_programs)

            if done and s is not None:
                try:
                    _, reward_val, pred_expr = _reward_and_expr(s)
                    reward_val = float(reward_val) if reward_val is not None else None
                except Exception:
                    reward_val, pred_expr = None, None
                if reward_val is not None and reward_val > best_progress_reward:
                    best_progress_reward = reward_val
                    self._emit_progress_equation(
                        equation=pred_expr,
                        score=best_progress_reward,
                        complexity=self._sequence_length(s),
                        source="nesymres_terminal",
                    )

        _, reward_mcts, pred_str = compute_reward_nesymres(model.X, model.y, s, fit_params)
        mcts_expr = pred_str or baseline_expr
        if reward_mcts is None:
            print("NeSymReS note note")

        self.best_tree = mcts_expr
        self.all_trees = []
        for expr in self._normalize_equation_list(all_preds):
            projected = self._project_equation_to_feature_budget(expr, self._n_features)
            normalized = self._normalize_equation(projected)
            if self._is_current_task_equation_valid(normalized):
                self.all_trees.append(normalized)
        if baseline_expr and baseline_expr not in self.all_trees:
            normalized_baseline = self._normalize_equation(
                self._project_equation_to_feature_budget(baseline_expr, self._n_features)
            )
            if self._is_current_task_equation_valid(normalized_baseline):
                self.all_trees.append(normalized_baseline)

        if mcts_expr is None and self.all_trees:
            mcts_expr = self.all_trees[0]

        if isinstance(mcts_expr, str):
            normalized_mcts = self._normalize_equation(
                self._project_equation_to_feature_budget(mcts_expr, self._n_features)
            )
            if self._is_current_task_equation_valid(normalized_mcts):
                mcts_expr = normalized_mcts
            else:
                mcts_expr = self.all_trees[0] if self.all_trees else None

        self.model = model
        self._n_features = X.shape[1]
        self._predict_variable_names = list(eq_setting["total_variables"])[: self._n_features]
        self._predict_fn = self._build_predictor_from_expression(
            mcts_expr, self._predict_variable_names
        )
        self._backend_params["backend"] = "nesymres"
        self._backend_params["search_state"] = s
        self._backend_params["best_expr"] = mcts_expr
        self._backend_params["eq_variables"] = eq_setting["total_variables"]
        self._emit_progress_equation(
            equation=mcts_expr,
            score=best_progress_reward if best_progress_reward != float("-inf") else reward_mcts,
            complexity=self._sequence_length(s),
            source="nesymres_final",
        )

    def fit(self, X, y):
        """
        note TPSR note 

        note:
            X: note note (n_samples, n_features)
            y: note note (n_samples,) note (n_samples, 1)
        """
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="TPSRRegressor.fit",
        )
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        temp_dir = None
        original_cwd = None
        tpsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr")
        inserted_paths = []

        # note
        try:
            for import_path in self._runtime_import_paths():
                if os.path.isdir(import_path) and import_path not in sys.path:
                    sys.path.insert(0, import_path)
                    inserted_paths.append(import_path)
            original_cwd = os.getcwd()
            os.chdir(tpsr_dir)

            temp_dir = tempfile.mkdtemp(prefix="tpsr_")

            from symbolicregression.envs import build_env
            from parsers import get_parser

            parser = get_parser()
            args = parser.parse_args([])

            # note note parser note 
            self._set_parser_attr(args, "backbone_model", self.params.get("backbone_model", "e2e"))
            self._set_parser_attr(args, "beam_size", int(self.params.get("beam_size", 10)))
            self._set_parser_attr(args, "beam_type", self.params.get("beam_type", "sampling"))
            self._set_parser_attr(args, "no_seq_cache", bool(self.params.get("no_seq_cache", False)))
            self._set_parser_attr(args, "no_prefix_cache", bool(self.params.get("no_prefix_cache", True)))
            self._set_parser_attr(args, "width", int(self.params.get("width", 3)))
            self._set_parser_attr(args, "num_beams", int(self.params.get("num_beams", 1)))
            self._set_parser_attr(args, "rollout", int(self.params.get("rollout", 3)))
            self._set_parser_attr(args, "horizon", int(self.params.get("horizon", 200)))
            self._set_parser_attr(args, "n_trees_to_refine", int(self.params.get("n_trees_to_refine", 10)))
            self._set_parser_attr(args, "seed", int(self.params.get("seed", 23)))
            self._set_parser_attr(args, "debug", bool(self.params.get("debug", False)))
            self._set_parser_attr(args, "beam_length_penalty", float(self.params.get("beam_length_penalty", 1.0)))
            self._set_parser_attr(args, "train_value", bool(self.params.get("train_value", False)))
            self._set_parser_attr(args, "ucb_constant", float(self.params.get("ucb_constant", 1.0)))
            self._set_parser_attr(args, "uct_alg", self.params.get("uct_alg", "uct"))
            self._set_parser_attr(args, "ucb_base", float(self.params.get("ucb_base", 4.0)))
            self._set_parser_attr(args, "cpu", bool(self.params.get("cpu", False)))
            self._set_parser_attr(args, "lam", float(self.params.get("lam", 0.1)))
            self._set_parser_attr(args, "max_input_points", int(self.params.get("max_input_points", 200)))
            self._set_parser_attr(args, "max_number_bags", int(self.params.get("max_number_bags", 10)))
            self._set_parser_attr(args, "rescale", bool(self.params.get("rescale", True)))
            self._set_parser_attr(args, "sample_only", bool(self.params.get("sample_only", False)))
            self._capture_runtime_feature_context(X)

            args.cpu = bool(self.params.get("cpu", False))
            args.device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.cpu:
                try:
                    torch.set_num_threads(int(self.params.get("cpu_num_threads", 4)))
                    torch.set_num_interop_threads(int(self.params.get("cpu_interop_threads", 1)))
                except Exception:
                    pass

            equation_env = build_env(args)

            reward_X, reward_y = self._downsample_reward_arrays(
                X,
                y,
                self.params.get("reward_sample_limit"),
            )
            samples = {
                "x_to_fit": [np.asarray(reward_X)],
                "y_to_fit": [np.asarray(reward_y)],
                "x_to_pred": [np.asarray(reward_X)],
                "y_to_pred": [np.asarray(reward_y)],
            }

            backbone_model = self.params.get("backbone_model", "e2e").lower()
            if backbone_model == "e2e":
                self._fit_e2e(X, y, equation_env, args, samples)
            elif backbone_model == "nesymres":
                self._fit_nesymres(np.asarray(X), np.asarray(y), args, samples)
            else:
                raise ValueError(f"unsupported note backbone_model: {backbone_model}")

            # note note SymbolicTree note 
            self.all_trees = self._normalize_equation_list(self.all_trees)
            if self.best_tree is not None:
                self.best_tree = self._normalize_equation(self.best_tree)
            return self

        except Exception:
            import traceback

            traceback.print_exc()
            raise
        finally:
            if original_cwd is not None:
                os.chdir(original_cwd)
            for import_path in reversed(inserted_paths):
                if import_path in sys.path:
                    try:
                        sys.path.remove(import_path)
                    except ValueError:
                        pass
            if temp_dir is not None:
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

    def predict(self, X):
        """note"""
        if self._predict_fn is not None:
            return self._predict_fn(X)

        if hasattr(self.model, "predict"):
            return self.model.predict(X, refinement_type="BFGS")

        raise ValueError("current noteunsupported predict note fit note")

    def get_optimal_equation(self):
        """note"""
        if self.best_tree is None:
            raise ValueError("note")
        return self._normalize_equation(self.best_tree)

    def get_total_equations(self):
        """note"""
        if self.best_tree is None and not self.all_trees:
            raise ValueError("note notefitnote")
        return list(self.all_trees or [])

    def export_canonical_symbolic_program(self):
        if self.best_tree is None:
            raise ValueError("note")
        return normalize_tpsr_artifact(
            self.get_optimal_equation(),
            expected_n_features=self._n_features,
        )


if __name__ == "__main__":
    import numpy as np

    X = np.random.rand(80, 2)
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.02 * np.random.randn(80)

    reg = TPSRRegressor(
        backbone_model="e2e",
        beam_size=6,
        width=3,
        num_beams=1,
        rollout=2,
    )
    reg.fit(X, y)
    print("eq:", reg.get_optimal_equation())
    print("pred:", reg.predict(X[:3]))
