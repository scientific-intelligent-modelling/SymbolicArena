import pickle
import base64
import numpy as np
import os
import sys
import requests
import sympy as sp
import torch

from ..base_wrapper import BaseWrapper 
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_e2esr_artifact

class E2ESRRegressor(BaseWrapper):
    _PROGRESS_STATE_FILENAME = ".e2esr_current_best.json"

    @staticmethod
    def _resolve_default_model_path(current_dir):
        shared_path = os.environ.get("SIM_SYMBOLICREGRESSION_MODEL_PATH")
        candidates = [
            shared_path,
            os.path.join(current_dir, "model.pt"),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return os.path.join(current_dir, "model.pt")

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

    def __init__(self, 
                 model_path=None, 
                 model_url="https://dl.fbaipublicfiles.com/symbolicregression/model1.pt",
                 max_input_points=200, 
                 max_number_bags=10,
                 stop_refinement_after=1,
                 n_trees_to_refine=10, 
                 rescale=True,
                 force_cpu=True,
                 torch_num_threads=1,
                 **kwargs):
        """
        noteE2ESRnote
        
        note 
        model_path: note noteNone note default note
        model_url: noteURL noteURLnote
        max_input_points: note
        max_number_bags: note bag note default note evaluate.py note black-box note
        stop_refinement_after: note
        n_trees_to_refine: note
        rescale: note
        torch_num_threads: CPU note PyTorch note
        """
        # note
        self._exp_path = kwargs.get("exp_path")
        self._exp_name = kwargs.get("exp_name")
        self.params = {
            'max_input_points': max_input_points,
            'max_number_bags': max_number_bags,
            'stop_refinement_after': stop_refinement_after,
            'n_trees_to_refine': n_trees_to_refine,
            'rescale': rescale,
            'force_cpu': force_cpu,
            'torch_num_threads': torch_num_threads,
            **kwargs
        }
        self._contract_n_features = self.params.pop("n_features", None)
        self._contract_feature_names = self.params.pop("feature_names", None)
        self._contract_target_name = self.params.pop("target_name", None)
        self.model = None
        self.regressor = None
        self.best_tree = None
        self.n_features_ = None
        self._progress_state_path = self._resolve_progress_state_path(self._exp_path, self._exp_name)
        
        # notee2esrnote note
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.e2esr_path = os.path.join(current_dir, "e2esr")
        if self.e2esr_path not in sys.path:
            sys.path.insert(0, self.e2esr_path)
        
        # note note note default note
        if model_path is None:
            # prefernote model.pt note e2esr/model1.pt
            model_path = self._resolve_default_model_path(current_dir)
        
        self.model_path = model_path
        self.model_url = model_url
        
        # note
        self._load_model()
    
    def _load_model(self):
        """note noteURLnote"""
        # note note
        if self.model is not None:
            return
            
        try:
            shared_model_path = os.environ.get("SIM_SYMBOLICREGRESSION_MODEL_PATH")
            if not os.path.isfile(self.model_path) and shared_model_path and os.path.isfile(shared_model_path):
                self.model_path = shared_model_path
            # note noteURLnote
            if not os.path.isfile(self.model_path):
                print(f"note {self.model_url} note...")
                r = requests.get(self.model_url, allow_redirects=True)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    f.write(r.content)
                print(f"note {self.model_path}")
            
            # note
            force_cpu = bool(self.params.get('force_cpu', True))
            if force_cpu:
                self._configure_cpu_threads(self.params.get('torch_num_threads', 1))

            # default note CPU note CUDA note note/note 
            if force_cpu:
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                self.model = self.model.cpu()
            else:
                # note note CUDA note note CPU
                try:
                    if not torch.cuda.is_available():
                        self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                    else:
                        self.model = torch.load(self.model_path, map_location=torch.device('cuda'))
                        self.model = self.model.cuda()
                except Exception:
                    # noteenvironment note CUDA note CPU note 
                    self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                    self.model = self.model.cpu()
            
            print(f"note!note: {self.model.device if hasattr(self.model, 'device') else 'cpu'}")
            
        except Exception as e:
            print(f"note: {str(e)}")
            # note notefitnote
            self.model = None

    @staticmethod
    def _configure_cpu_threads(torch_num_threads):
        """note CPU note note worker note """
        try:
            threads = int(torch_num_threads)
        except Exception:
            threads = 1
        threads = max(1, threads)
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(threads)
        except Exception:
            # PyTorch note interop note 
            pass
    
    def fit(self, X, y):
        """
        note note note
        note
        """
        try:
            self._validate_explicit_dataset_contract(
                X,
                n_features=self._contract_n_features,
                feature_names=self._contract_feature_names,
                target_name=self._contract_target_name,
                context="E2ESRRegressor.fit",
            )
            self.n_features_ = int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else 1
            # noteE2ESRnote
            from symbolicregression.model import SymbolicTransformerRegressor
            
            # note note
            if self.model is None:
                raise ValueError("note note")

            allowed_regressor_params = {
                "max_input_points",
                "max_number_bags",
                "stop_refinement_after",
                "n_trees_to_refine",
                "rescale",
            }
            regressor_kwargs = {
                k: v
                for k, v in self.params.items()
                if k in allowed_regressor_params
            }
            wrapper_only_params = {"force_cpu", "torch_num_threads"}
            unknown_params = [k for k in self.params if k not in allowed_regressor_params and k not in wrapper_only_params]
            if unknown_params:
                # note SymbolicRegressor note note __init__ note
                pass
            
            self.regressor = SymbolicTransformerRegressor(
                model=self.model,
                progress_state_path=self._progress_state_path,
                **regressor_kwargs
            )
            
            # note
            self.regressor.fit(X, y)
            
            # note
            self.best_tree = self.regressor.retrieve_tree(with_infos=True)
            
            return self
            
        except Exception as e:
            print(f"note: {str(e)}")
            raise
    
    def predict(self, X):
        """note"""
        if self.regressor is None:
            raise ValueError("note notefitnote")
        return self.regressor.predict(X)
    
    def get_optimal_equation(self):
        """note"""
        if self.best_tree is None:
            raise ValueError("note notefitnote")
        
        # note note
        if hasattr(self, '_cached_optimal_equation'):
            return self._cached_optimal_equation
            
        try:
            # note
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            model_str = self.best_tree["relabed_predicted_tree"].infix()
            for op, replace_op in replace_ops.items():
                model_str = model_str.replace(op, replace_op)
            
            # notesympynote
            expr = sp.parse_expr(model_str)
            self._cached_optimal_equation = str(expr)
            return self._cached_optimal_equation
        except Exception as e:
            print(f"note: {str(e)}")
            # note note
            self._cached_optimal_equation = str(self.best_tree["relabed_predicted_tree"].infix())
            return self._cached_optimal_equation

    def get_total_equations(self):
        """note"""
        if self.best_tree is None:
            raise ValueError("note notefitnote")
        
        # note note
        if hasattr(self, '_cached_total_equations'):
            return self._cached_total_equations
            
        # E2ESRnote note
        self._cached_total_equations = [self.get_optimal_equation()]
        return self._cached_total_equations

    def export_canonical_symbolic_program(self):
        if self.best_tree is None:
            raise ValueError("note notefitnote")
        return normalize_e2esr_artifact(
            self.get_optimal_equation(),
            expected_n_features=getattr(self, "n_features_", None),
        )


if __name__ == "__main__":
    # note
    import numpy as np
    # note
    X = np.random.randn(100, 2)
    y = np.cos(2*np.pi*X[:, 0]) + X[:, 1]**2

    # note note
    model = E2ESRRegressor()
    
    # note
    model.fit(X, y)

    # note
    equation = model.get_optimal_equation()
    print(f"note: {equation}")

    # note
    y_pred = model.predict(X)
    print(f"note5note: {y_pred[:5]}")
