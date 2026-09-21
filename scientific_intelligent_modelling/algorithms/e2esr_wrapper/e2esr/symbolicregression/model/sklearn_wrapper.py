# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import math, time, copy
import os
import numpy as np
import torch
import sympy as sp
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
import traceback
from sklearn import feature_selection 

def corr(X, y, epsilon=1e-10):
    """
    X : shape n*d
    y : shape n
    """
    cov = (y @ X)/len(y) - y.mean()*X.mean(axis=0)
    corr = cov / (epsilon + X.std(axis=0) * y.std())
    return corr

def get_top_k_features(X, y, k=10):
    if y.ndim==2:
        y=y[:,0]
    if X.shape[1]<=k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        #scores = corr(X, y)
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])

def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for (old, new) in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree

class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                model=None,
                max_input_points=200,
                max_number_bags=10,
                stop_refinement_after=1,
                n_trees_to_refine=10,
                rescale=True,
                progress_state_path=None,
                progress_callback=None,
                timeout_in_seconds=None,
                timeout_guard_seconds=5,
                ):

        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale
        self.progress_state_path = progress_state_path
        self.progress_callback = progress_callback
        self.timeout_in_seconds = self._as_positive_float(timeout_in_seconds)
        self.timeout_guard_seconds = self._as_positive_float(timeout_guard_seconds) or 5.0

    @staticmethod
    def _as_positive_float(value):
        try:
            value = float(value)
        except Exception:
            return None
        return value if value > 0 else None

    def _time_budget_exhausted(self):
        if self.timeout_in_seconds is None:
            return False
        elapsed = time.time() - getattr(self, "start_fit", time.time())
        return elapsed >= max(0.0, self.timeout_in_seconds - self.timeout_guard_seconds)

    def _build_random_bag(self, scaled_X, Y, dataset_idx):
        x = np.asarray(scaled_X[dataset_idx])
        y = np.asarray(Y[dataset_idx])
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        if len(x) == 0:
            return []
        bag_size = min(int(self.max_input_points), len(x))
        indices = np.random.choice(len(x), size=bag_size, replace=len(x) < bag_size)
        return [[x[idx], y[idx]] for idx in indices]

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def _safe_tree_metric(self, tree, X, y, metric):
        try:
            value = self.evaluate_tree(tree, X, y, metric)
        except Exception:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def _candidate_complexity(self, tree):
        try:
            return int(len(tree.prefix().split(",")))
        except Exception:
            return None

    def _relabeled_tree(self, tree, dataset_idx):
        if tree is None:
            return None
        if not hasattr(self, "top_k_features"):
            return tree
        if dataset_idx >= len(self.top_k_features):
            return tree
        exchanges = {}
        for i, feature in enumerate(self.top_k_features[dataset_idx] or []):
            exchanges["x_{}".format(i)] = "x_{}".format(feature)
        if not exchanges:
            return tree
        try:
            return exchange_node_values(tree, exchanges)
        except Exception:
            return tree

    def _tree_to_expression(self, tree, dataset_idx):
        relabeled = self._relabeled_tree(tree, dataset_idx)
        if relabeled is None:
            return None
        try:
            model_str = relabeled.infix()
        except Exception:
            return None
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
        for op, replace_op in replace_ops.items():
            model_str = model_str.replace(op, replace_op)
        try:
            return str(sp.parse_expr(model_str))
        except Exception:
            return str(model_str)

    def _write_progress_state(self, payload):
        if not self.progress_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.progress_state_path), exist_ok=True)
            with open(self.progress_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _emit_progress_candidate(self, dataset_idx, candidate, X, y, *, stage, compute_metrics=True):
        if not self.progress_state_path or not isinstance(candidate, dict):
            return
        tree = candidate.get("predicted_tree")
        if tree is None:
            return
        equation = self._tree_to_expression(tree, dataset_idx)
        if not isinstance(equation, str) or not equation.strip():
            return
        native_score = candidate.get("native_model_score")
        try:
            native_score = float(native_score)
        except (TypeError, ValueError, OverflowError):
            native_score = None
        if native_score is not None and not np.isfinite(native_score):
            native_score = None
        payload = {
            "equation": equation,
            "native_model_score": native_score,
            "score": native_score,
            "internal_objective": "decoder_length_normalized_log_likelihood",
            "objective_direction": "max",
            "complexity": self._candidate_complexity(tree),
            "refinement_type": candidate.get("refinement_type"),
            "bag_index": candidate.get("bag_index"),
            "candidate_rank": candidate.get("candidate_rank"),
            "generation_source": candidate.get("generation_source"),
            "stage": stage,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if callable(self.progress_callback):
            self.progress_callback(
                equation=equation,
                native_model_score=native_score,
                bag_index=candidate.get("bag_index", 0),
                candidate_rank=candidate.get("candidate_rank", 0),
                generation_source=candidate.get("generation_source", "unknown"),
            )
        else:
            self._write_progress_state(payload)

    def fit(
        self,
        X,
        Y,
        verbose=False
    ):
        self.start_fit = getattr(self, "_external_start_fit", None) or time.time()

        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)

        self.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            self.top_k_features[i] = get_top_k_features(X[i], Y[i], k=self.model.env.params.max_input_dimension)
            X[i] = X[i][:, self.top_k_features[i]]
    
        scaler = utils_wrapper.StandardScaler() if self.rescale else None
        scale_params = {}
        if scaler is not None:
            scaled_X = []
            for i, x in enumerate(X):
                scaled_X.append(scaler.fit_transform(x))
                scale_params[i]=scaler.get_params()
        else:
            scaled_X = X

        inputs, inputs_ids = [], []
        for seq_id in range(len(scaled_X)):
            for seq_l in range(len(scaled_X[seq_id])):
                y_seq = Y[seq_id]
                if len(y_seq.shape)==1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l%self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                inputs[-1].append([scaled_X[seq_id][seq_l], y_seq[seq_l]])

        if self.max_number_bags>0:
            inputs = inputs[:self.max_number_bags]
            inputs_ids = inputs_ids[:self.max_number_bags]

        candidates = defaultdict(list)
        progressive_forward = bool(self.progress_state_path)
        if progressive_forward:
            forward_time = time.time()
            max_bags = int(self.max_number_bags) if self.max_number_bags and self.max_number_bags > 0 else None
            if self.timeout_in_seconds is not None:
                max_bags = None
            bag_index = 0
            while True:
                if bag_index < len(inputs):
                    bag = inputs[bag_index]
                    input_id = inputs_ids[bag_index]
                elif self.timeout_in_seconds is not None and (max_bags is None or bag_index < max_bags):
                    input_id = bag_index % n_datasets
                    bag = self._build_random_bag(scaled_X, Y, input_id)
                    if not bag:
                        break
                else:
                    break
                if self._time_budget_exhausted() and candidates:
                    break
                bag_outputs = self.model([bag])  ## 按 bag 增量前向，确保长任务能尽早产出中间态
                assert len(bag_outputs) == 1, "Problem with incremental inputs and outputs"
                candidate = bag_outputs[0]
                candidates[input_id].extend(candidate)
                metadata_groups = getattr(self.model, "last_generation_metadata", [])
                metadata = metadata_groups[0] if metadata_groups else []
                for candidate_rank, tree in enumerate(candidate):
                    native = metadata[candidate_rank] if candidate_rank < len(metadata) else {}
                    self._emit_progress_candidate(
                        input_id,
                        {
                            "refinement_type": "ForwardRaw",
                            "predicted_tree": tree,
                            "time": time.time() - self.start_fit,
                            "native_model_score": native.get("native_model_score"),
                            "candidate_rank": native.get("candidate_rank", candidate_rank),
                            "generation_source": native.get("generation_source", "unknown"),
                            "bag_index": bag_index,
                        },
                        scaled_X[input_id],
                        Y[input_id],
                        stage="forward_partial",
                        compute_metrics=False,
                    )
                bag_index += 1
            if verbose: print("Finished forward in {} secs".format(time.time()-forward_time))
        else:
            forward_time=time.time()
            outputs = self.model(inputs)  ##Forward transformer: returns predicted functions
            if verbose: print("Finished forward in {} secs".format(time.time()-forward_time))

            assert len(inputs) == len(outputs), "Problem with inputs and outputs"
            for i in range(len(inputs)):
                input_id = inputs_ids[i]
                candidate = outputs[i]
                candidates[input_id].extend(candidate)
        assert len(candidates.keys())==n_datasets
            
        self.tree = {}
        for input_id, candidates_id in candidates.items():
            if len(candidates_id)==0: 
                self.tree[input_id]=None
                continue
        
            refined_candidates = self.refine(input_id, scaled_X[input_id], Y[input_id], candidates_id, verbose=verbose)
            for i,candidate in enumerate(refined_candidates):
                if scaler is not None:
                    refined_candidates[i]["predicted_tree"]=scaler.rescale_function(self.model.env, candidate["predicted_tree"], *scale_params[input_id])
                else: 
                    refined_candidates[i]["predicted_tree"]=candidate["predicted_tree"]
            self.tree[input_id] = refined_candidates
            if refined_candidates:
                self._emit_progress_candidate(input_id, refined_candidates[0], X[input_id], Y[input_id], stage="fit_final")

    @torch.no_grad()
    def evaluate_tree(self, tree, X, y, metric):
        numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree)
        y_tilde = numexpr_fn(X)[:,0]
        metrics = compute_metrics({"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]}, metrics=metric)
        return metrics[metric][0]

    def order_candidates(self, X, y, candidates, metric="_mse", verbose=False):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self._safe_tree_metric(candidate["predicted_tree"], X, y, metric)
            else:
                score = candidate[metric]
            try:
                score = float(score)
            except Exception:
                score = None
            if score is None or math.isnan(score) or math.isinf(score):
                score = np.inf if metric.startswith("_") else -np.inf
            scores.append(score)
        ordered_idx = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=not metric.startswith("_"),
        )
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def refine(self, dataset_idx, X, y, candidates, verbose):
        refined_candidates = []
        
        ## For skeleton model
        for i, candidate in enumerate(candidates):
            candidate_skeleton, candidate_constants =  self.model.env.generator.function_to_skeleton(candidate, constants_with_idx=True)
            if "CONSTANT" in candidate_constants:
                candidates[i] = self.model.env.wrap_equation_floats(candidate_skeleton, np.random.randn(len(candidate_constants)))

        candidates = [{"refinement_type": "NoRef", "predicted_tree": candidate, "time": time.time()-self.start_fit} for candidate in candidates]
        candidates = self.order_candidates(X, y, candidates, metric="_mse", verbose=verbose)
        best_r2 = None
        if candidates:
            candidates[0]["_mse"] = self._safe_tree_metric(candidates[0]["predicted_tree"], X, y, "_mse")
            best_r2 = self._safe_tree_metric(candidates[0]["predicted_tree"], X, y, "r2")
            if best_r2 is not None:
                candidates[0]["r2"] = best_r2
            self._emit_progress_candidate(dataset_idx, candidates[0], X, y, stage="noref_best")

        ## REMOVE SKELETON DUPLICATAS
        skeleton_candidates, candidates_to_remove = {}, []
        for i, candidate in enumerate(candidates):
            skeleton_candidate, _ = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=False)
            if skeleton_candidate.infix() in skeleton_candidates:
                candidates_to_remove.append(i)
            else:
                skeleton_candidates[skeleton_candidate.infix()]=1
        if verbose: print("Removed {}/{} skeleton duplicata".format(len(candidates_to_remove), len(candidates)))

        candidates = [candidates[i] for i in range(len(candidates)) if i not in candidates_to_remove]
        if self.n_trees_to_refine>0:
            candidates_to_refine = candidates[:self.n_trees_to_refine]
        else:
            candidates_to_refine = copy.deepcopy(candidates)

        for candidate in candidates_to_refine:
            if self._time_budget_exhausted() and refined_candidates:
                break
            refinement_strategy = utils_wrapper.BFGSRefinement()
            candidate_skeleton, candidate_constants = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=True)
            try:
                refined_candidate = refinement_strategy.go(env=self.model.env, 
                                                        tree=candidate_skeleton, 
                                                        coeffs0=candidate_constants,
                                                        X=X,
                                                        y=y,
                                                        downsample=1024,
                                                        stop_after=self.stop_refinement_after)

            except Exception as e:
                if verbose: 
                    print(e)
                    #traceback.format_exc()
                continue
            
            if refined_candidate is not None:
                refined_entry = { 
                        "refinement_type": "BFGS",
                        "predicted_tree": refined_candidate,
                        }
                refined_entry["_mse"] = self._safe_tree_metric(refined_candidate, X, y, "_mse")
                refined_entry["r2"] = self._safe_tree_metric(refined_candidate, X, y, "r2")
                refined_candidates.append(refined_entry)
                candidate_r2 = refined_entry.get("r2")
                if candidate_r2 is not None and (best_r2 is None or candidate_r2 > best_r2):
                    best_r2 = candidate_r2
                    self._emit_progress_candidate(dataset_idx, refined_entry, X, y, stage="bfgs_best")
        candidates.extend(refined_candidates)  
        candidates = self.order_candidates(X, y, candidates, metric="r2")
        if candidates:
            self._emit_progress_candidate(dataset_idx, candidates[0], X, y, stage="refine_final")

        for candidate in candidates:
            if "time" not in candidate:
                candidate["time"]=time.time()-self.start_fit
        return candidates

    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.tree)):
                for gen in self.tree[tree_idx]:
                    print(gen)
        return "Transformer"

    def retrieve_refinements_types(self):
        return ["BFGS", "NoRef"]

    def exchange_tree_features(self):
        top_k_features = self.top_k_features
        for dataset_id, candidates in self.tree.items():
            exchanges = {}
            for i, feature in enumerate(top_k_features[dataset_id]):
                exchanges["x_{}".format(i)]="x_{}".format(feature)
            for candidate in candidates:
                candidate["relabed_predicted_tree"] = exchange_node_values(candidate["predicted_tree"], exchanges)

    def retrieve_tree(self, refinement_type=None, dataset_idx=0, all_trees=False, with_infos=False):
        self.exchange_tree_features()
        if dataset_idx == -1: idxs = [_ for _ in range(len(self.tree))] 
        else: idxs = [dataset_idx]
        best_trees = []
        for idx in idxs:
            best_tree = copy.deepcopy(self.tree[idx])
            if best_tree and refinement_type is not None:
                best_tree = list(filter(lambda gen: gen["refinement_type"]==refinement_type, best_tree))
            if not best_tree:
                if with_infos:
                    best_trees.append({"predicted_tree": None, "refinement_type": None, "time": None})
                else:
                    best_trees.append(None)
            else:
                if with_infos:
                    if all_trees:
                        best_trees.append(best_tree)
                    else:
                        best_trees.append(best_tree[0])
                else:
                    if all_trees:
                        best_trees.append([best_tree[i]["predicted_tree"] for i in range(len(best_tree))])
                    else:
                        best_trees.append(best_tree[0]["predicted_tree"])
        if dataset_idx != -1: 
            return best_trees[0]
        else: return best_trees


    def predict(self, X, refinement_type=None, tree_idx=0, batch=False):        

        if not isinstance(X, list):
            X = [X]
        for i in range(len(X)):
            X[i]=X[i][:,self.top_k_features[i]]

        res = []
        if batch:
            tree = self.retrieve_tree(refinement_type=refinement_type, dataset_idx=-1)
            for tree_idx in range(len(tree)):
                X_idx = X[tree_idx]
                if tree[tree_idx] is None: 
                    res.append(None)
                else:   
                    numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree[tree_idx])
                    y = numexpr_fn(X_idx)[:,0]
                    res.append(y)
            return res
        else:
            X_idx = X[tree_idx]
            tree = self.retrieve_tree(refinement_type=refinement_type, dataset_idx=tree_idx)
            if tree is not None:
                numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree)
                y = numexpr_fn(X_idx)[:,0]
                return y
            else:
                return None
