# srkit/regressor.py
import os
import re
import json
import tempfile
import subprocess
import signal
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import numpy as np

from .config_manager import config_manager
from .conda_env_manager import env_manager
from .exceptions import NoValidOutputError

class SymbolicRegressor:
    def __init__(self, tool_name, problem_name: Optional[str] = None, experiments_dir: Optional[str] = None, seed: int = 1314, **kwargs):
        """
        note symbolic regressionnote
        
        note:
            tool_name: note (note 'gplearn', 'pysr')
            problem_name: note/note note
            experiments_dir: note default notecurrent note './experiments' 
            seed: note default 1314  note
            **kwargs: note
        """
        self.tool_name = tool_name
        self.params = kwargs
        self.serialized_model = None
        
        # note
        self.problem_name = problem_name or "problem"
        self.seed = int(seed) if seed is not None else 1314
        # default experiments note notecurrent note 
        # note exp_path note note 
        explicit_exp_path = self.params.get("exp_path")
        explicit_exp_name = self.params.get("exp_name")
        if isinstance(explicit_exp_path, str) and explicit_exp_path.strip():
            self.experiments_root = os.path.abspath(explicit_exp_path.strip())
        else:
            self.experiments_root = experiments_dir or os.path.join(os.getcwd(), "experiments")

        # note {problem}_{tool}_seed{seed}_YYYYMMDD-HHMMSS
        def _slugify(text: str) -> str:
            try:
                return re.sub(r"[^A-Za-z0-9_\-]+", "-", str(text)).strip("-") or "item"
            except Exception:
                return "item"

        def _has_timestamp_suffix(text: str) -> bool:
            return bool(re.search(r"_\d{8}-\d{6}$", str(text).strip()))

        ts = time.strftime('%Y%m%d-%H%M%S')
        generated_exp_name = f"{_slugify(self.problem_name)}_{_slugify(self.tool_name)}_seed{self.seed}_{ts}"
        use_explicit_layout = (
            isinstance(explicit_exp_path, str)
            and explicit_exp_path.strip()
            and isinstance(explicit_exp_name, str)
            and explicit_exp_name.strip()
        )
        if use_explicit_layout:
            requested_exp_name = explicit_exp_name.strip()
            exp_name = requested_exp_name if _has_timestamp_suffix(requested_exp_name) else f"{requested_exp_name}_{ts}"
        else:
            exp_name = generated_exp_name
        try:
            os.makedirs(self.experiments_root, exist_ok=True)
            self.experiment_dir = os.path.join(self.experiments_root, exp_name)
            os.makedirs(self.experiment_dir, exist_ok=True)
        except Exception:
            # note note note
            tmp_root = tempfile.gettempdir()
            self.experiment_dir = os.path.join(tmp_root, exp_name)
            os.makedirs(self.experiment_dir, exist_ok=True)

        # note note  
        # - exp_path: note
        # - exp_name: current note
        # - problem_name / seed: note
        # note note  note manifest note 
        try:
            self.params["exp_path"] = self.experiments_root
            self.params["exp_name"] = os.path.basename(self.experiment_dir)
            self.params.setdefault("problem_name", self.problem_name)
            self.params.setdefault("seed", self.seed)
        except Exception:
            # note
            pass

        # note manifest.json  created note + note
        try:
            self._write_initial_manifest()
        except Exception:
            # note
            pass
        
        # noteconfig_managernoteenvironment note
        self.env_name = config_manager.get_env_name_by_tool(tool_name)
        if not self.env_name:
            raise ValueError(f"note '{tool_name}' noteenvironment note")
        
        # note note Python note noteenvironment note/note
        # note conda note note pip freeze python --version note 
        python_path = env_manager.get_env_python(self.env_name)
        if not python_path:
            # note python note/note
            exists, reason = env_manager.check_environment(self.env_name)
            if not exists:
                print(f"environment '{self.env_name}' note note...")
                success = env_manager.create_environment(self.env_name)
                if not success:
                    raise RuntimeError(f"noteenvironment '{self.env_name}' {reason}")
    
    def fit(self, X, y):
        """
        note
        
        note:
            X: note
            y: note
        
        note:
            self: note
        """
        # note numpy note note 
        recovery_X = np.asarray(X)
        recovery_y = np.asarray(y).reshape(-1)
        if recovery_X.ndim == 1:
            recovery_X = recovery_X.reshape(-1, 1)

        # note
        X_payload = recovery_X.tolist()
        y_payload = recovery_y.tolist()
        
        # note
        command = {
            'action': 'fit',
            'data': {'X': X_payload, 'y': y_payload},
            'params': self.params,
            'tool_name': self.tool_name,
            'serialized_model': self.serialized_model  # note
        }
        
        # note running
        try:
            self._update_manifest(status="running")
        except Exception:
            pass

        # note
        try:
            result = self._execute_subprocess(command)
        except TimeoutError:
            recovered = self._recover_from_timeout(recovery_X, recovery_y, command)
            if recovered:
                self.serialized_model = recovered
                try:
                    self._update_manifest(
                        status="success",
                        timeout_in_seconds=self._resolve_fit_timeout_seconds(command),
                        recovered_from_timeout=True,
                    )
                except Exception:
                    pass
                return self
            try:
                self._update_manifest(
                    status="timed_out",
                    timeout_in_seconds=self._resolve_fit_timeout_seconds(command),
                )
            except Exception:
                pass
            raise
        except Exception:
            # note
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise
        
        # note
        if result.get('no_valid_output'):
            try:
                self._update_manifest(status="no_valid_output")
            except Exception:
                pass
            raise NoValidOutputError(result.get('message') or "note")

        if 'error' in result:
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        
        # note
        self.serialized_model = result.get('serialized_model', {})
        # note
        try:
            self._update_manifest(status="success")
        except Exception:
            pass
        return self

    def _recover_from_timeout(self, X: np.ndarray, y: np.ndarray, fit_command: dict) -> Optional[str]:
        """note """
        exp_dir = self.experiment_dir
        if not exp_dir or not os.path.isdir(exp_dir):
            return None

        recovery_timeout = self._resolve_timeout_recovery_seconds(
            self._resolve_fit_timeout_seconds(fit_command)
        )
        recovery_command = {
            "action": "recover_from_timeout",
            "data": {
                "X": np.asarray(X).tolist(),
                "y": np.asarray(y).reshape(-1).tolist(),
            },
            "params": dict(self.params or {}),
            "tool_name": self.tool_name,
            "experiment_dir": exp_dir,
            "timeout_in_seconds": recovery_timeout,
        }
        try:
            result = self._execute_subprocess(recovery_command)
        except Exception:
            return None

        if not isinstance(result, dict) or result.get("error"):
            return None

        serialized_model = result.get("serialized_model")
        if not isinstance(serialized_model, str) or not serialized_model.strip():
            return None
        return serialized_model
    
    def predict(self, X):
        """
        note
        
        note:
            X: note
        
        note:
            predictions: note
        """
        # note
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        
        # note
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # note
        command = {
            'action': 'predict',
            'data': {'X': X},
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # note
        result = self._execute_subprocess(command)
        
        # note
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        
        # note
        predictions = result.get('predictions', [])
        return np.array(predictions)
    
    def get_optimal_equation(self):
        """
        note
        
        note:
            equation: note
        """
        # note
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        
        # note
        command = {
            'action': 'get_optimal_equation',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # note
        result = self._execute_subprocess(command)
        
        # note
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        
        # note
        return result.get('equation', '')
    

    def get_total_equations(self, n=None):
        """
        note
        
        note:
            equations: note
        """
        # note
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        
        # note
        command = {
            'action': 'get_total_equations',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # note
        result = self._execute_subprocess(command)
        
        # note
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        
        # note
        return result.get('equations', [])


    def __str__(self):
        """
        note
        
        note:
            model_str: note
        """
        # note
        model_str = f"SymbolicRegressor(tool='{self.tool_name}'"
        
        for key, value in self.params.items():
            model_str += f", {key}={value}"
        model_str += ")"

        # note note
        if self.serialized_model is not None:
            try:
                equation = self.get_optimal_equation()
                if equation:
                    model_str += f"\nnote:\n{equation}"
            except Exception as e:
                model_str += f"\nnote note: {str(e)}"
            try:
                params = self.get_fitted_params()
                if params is not None:
                    model_str += f"\nnote: {params}"
            except Exception:
                pass
        else:
            model_str += "\nnote"
        return model_str

    def get_fitted_params(self):
        """note note  """
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        command = {
            'action': 'get_fitted_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        return result.get('params')

    def get_total_equations_with_params(self, n=None):
        """note noteTop-N note note  """
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        command = {
            'action': 'get_total_equations_with_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name,
        }
        if n is not None:
            command['n'] = int(n)
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        return result.get('items', [])

    def export_canonical_symbolic_program(self):
        """note 

        current note Phase 1 note CanonicalSymbolicProgram note benchmark
        runner note normalizer note 
        """
        if self.serialized_model is None:
            raise ValueError("note notefitnote")
        command = {
            'action': 'export_canonical_symbolic_program',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name,
        }
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"note: {result['message']}\n{result.get('traceback', '')}")
        return result.get('artifact', {})


    def _execute_subprocess(self, command):
        """note"""
        # notePythonnote
        python_path = env_manager.get_env_python(self.env_name)
        if not python_path:
            raise RuntimeError(f"noteenvironment '{self.env_name}' notePythonnote")
        
        # note
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as cmd_file:
            cmd_path = cmd_file.name
            json.dump(command, cmd_file)
        
        # note
        result_path = cmd_path + '.result'
        
        # note
        runner_script = os.path.join(os.path.dirname(__file__), 'subprocess_runner.py')
        
        try:
            # note note  note stdout/stderr notecurrent note
            env = os.environ.copy()
            # note julia/pip noteenvironmentCONDA_PREFIX noteenvironment
            # note CONDA_PREFIX notecurrent noteenvironment note noteenvironmentConsistency
            try:
                py_path = env_manager.get_env_python(self.env_name)
                if py_path:
                    env["CONDA_PREFIX"] = str(Path(py_path).resolve().parent.parent)
            except Exception:
                pass
            # note julia/pythoncall noteenvironment note conda environment note
            env.setdefault(
                "PYTHON_JULIAPKG_PROJECT",
                str(Path(tempfile.gettempdir()) / f"pyjuliapkg_{self.env_name}")
            )
            # note julia note
            env.setdefault('PYTHONUNBUFFERED', '1')
            timeout_seconds = self._resolve_subprocess_timeout_seconds(command)
            proc = subprocess.Popen(
                [python_path, '-u', runner_script, '--input', cmd_path, '--output', result_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=(os.name != "nt"),
            )
            # note stdout/stderr note 
            assert proc.stdout is not None and proc.stderr is not None
            stdout_thread = threading.Thread(
                target=self._forward_subprocess_stream,
                args=(proc.stdout,),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._forward_subprocess_stream,
                args=(proc.stderr,),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            try:
                ret = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                self._terminate_subprocess_tree(proc)
                stdout_thread.join(timeout=1.0)
                stderr_thread.join(timeout=1.0)
                raise TimeoutError(
                    f"note '{self.tool_name}' note "
                    f"action={command.get('action')}, timeout_in_seconds={timeout_seconds}"
                ) from exc
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            if ret != 0:
                raise subprocess.CalledProcessError(ret, proc.args)

            # note
            with open(result_path, 'r') as f:
                result = json.load(f)

            return result
        except subprocess.CalledProcessError as e:
            result = self._safe_load_result_file(result_path)
            raise RuntimeError(f"note: {e}\n{result.get('traceback', '')}")
        except Exception as e:
            result = self._safe_load_result_file(result_path)
            if isinstance(e, TimeoutError):
                raise
            raise RuntimeError(f"note: {e}\n{result.get('traceback', '')}")
        finally:
            for path in (cmd_path, result_path):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

    def _resolve_fit_timeout_seconds(self, command: dict) -> Optional[int]:
        """note fit note """
        if command.get("action") != "fit":
            return None
        raw = command.get("timeout_in_seconds")
        if raw is None:
            raw = (command.get("params") or {}).get("timeout_in_seconds")
        try:
            raw = int(raw)
        except Exception:
            return None
        return raw if raw > 0 else None

    @staticmethod
    def _resolve_timeout_recovery_seconds(fit_timeout_seconds: Optional[int]) -> int:
        """note note """
        if isinstance(fit_timeout_seconds, int) and fit_timeout_seconds > 0:
            return max(60, min(300, fit_timeout_seconds))
        return 300

    def _resolve_subprocess_timeout_seconds(self, command: dict) -> Optional[int]:
        """notecurrent note """
        raw = command.get("timeout_in_seconds")
        try:
            raw = int(raw)
        except Exception:
            raw = None
        if raw is not None and raw > 0:
            return raw

        requested = self._resolve_fit_timeout_seconds(command)
        if requested is not None:
            return requested

        toolbox_config = config_manager.get_config("toolbox_config") or {}
        fallback = toolbox_config.get("subprocess_timeout")
        try:
            fallback = int(fallback)
        except Exception:
            return None
        return fallback if fallback > 0 else None

    @staticmethod
    def _forward_subprocess_stream(stream):
        try:
            for line in stream:
                print(line, end="")
        except Exception:
            pass

    @staticmethod
    def _terminate_subprocess_tree(proc):
        if proc.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=3)
            return
        except Exception:
            pass

        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass

    @staticmethod
    def _safe_load_result_file(result_path: str) -> dict:
        if not result_path or not os.path.exists(result_path):
            return {}
        try:
            with open(result_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    # ========= note manifest note =========
    def _sanitize_config(self) -> dict:
        """note/note note manifest note config field """
        hidden = {"api_key", "apikey", "token", "password", "secret"}
        cfg = {k: v for k, v in (self.params or {}).items() if str(k).lower() not in hidden}
        cfg.setdefault("tool_name", self.tool_name)
        cfg.setdefault("problem_name", self.problem_name)
        cfg.setdefault("seed", self.seed)
        return cfg

    def _manifest_path(self) -> str:
        return os.path.join(self.experiment_dir, "manifest.json")

    def _write_initial_manifest(self):
        now_local = datetime.now().astimezone()
        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        manifest = {
            "experiment_id": os.path.basename(self.experiment_dir),
            "problem_name": self.problem_name,
            "algorithm": self.tool_name,
            "seed": self.seed,
            "created_at_local": now_local.isoformat(),
            "created_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "status": "created",
            # note note
            "iterations": None,
            "config": self._sanitize_config(),
        }
        path = self._manifest_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _update_manifest(self, **fields):
        path = self._manifest_path()
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data.update(fields or {})
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update_iterations(self, iterations):
        """note current note  """
        try:
            it = int(iterations)
        except Exception:
            return
        try:
            self._update_manifest(iterations=it)
        except Exception:
            pass
