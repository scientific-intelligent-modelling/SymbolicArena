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
        初始化符号回归器
        
        参数:
            tool_name: 要使用的工具名称 (例如 'gplearn', 'pysr')
            problem_name: 问题/数据集名称，用于实验命名与目录组织
            experiments_dir: 实验目录根路径（默认在当前工作目录下的 './experiments'）
            seed: 随机种子（默认 1314），也用于实验目录命名
            **kwargs: 传递给实际工具的参数
        """
        self.tool_name = tool_name
        self.params = kwargs
        self.serialized_model = None
        
        # 基本实验信息
        self.problem_name = problem_name or "problem"
        self.seed = int(seed) if seed is not None else 1314
        # 默认 experiments 根目录：相对于调用者当前工作目录。
        # 若显式传入 exp_path，则优先以其作为根目录，避免外层与算法内层落到不同目录。
        explicit_exp_path = self.params.get("exp_path")
        explicit_exp_name = self.params.get("exp_name")
        if isinstance(explicit_exp_path, str) and explicit_exp_path.strip():
            self.experiments_root = os.path.abspath(explicit_exp_path.strip())
        else:
            self.experiments_root = experiments_dir or os.path.join(os.getcwd(), "experiments")

        # 创建实验目录：{problem}_{tool}_seed{seed}_YYYYMMDD-HHMMSS
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
            # 若目录创建失败，回退到临时目录，但不影响后续运行
            tmp_root = tempfile.gettempdir()
            self.experiment_dir = os.path.join(tmp_root, exp_name)
            os.makedirs(self.experiment_dir, exist_ok=True)

        # 将实验目录信息下传给具体算法包装器（若其选择使用）：
        # - exp_path: 统一的实验根目录
        # - exp_name: 当前实验子目录名
        # - problem_name / seed: 便于算法内部复用
        # 这里写入“最终解析后的目录名”，保证外层 manifest 与算法内层真实工作目录一致。
        try:
            self.params["exp_path"] = self.experiments_root
            self.params["exp_name"] = os.path.basename(self.experiment_dir)
            self.params.setdefault("problem_name", self.problem_name)
            self.params.setdefault("seed", self.seed)
        except Exception:
            # 下传失败不影响主流程
            pass

        # 写入最小元信息（manifest.json）：created 状态 + 基本配置
        try:
            self._write_initial_manifest()
        except Exception:
            # 元信息写入失败不影响主流程
            pass
        
        # 使用config_manager获取环境名称
        self.env_name = config_manager.get_env_name_by_tool(tool_name)
        if not self.env_name:
            raise ValueError(f"未找到工具 '{tool_name}' 的环境配置")
        
        # 快速路径：仅在无法定位 Python 可执行文件时，才进行较重的环境检查/创建
        # 这样可以避免每次实例化都调用昂贵的 conda 检查（如 pip freeze、python --version 等）
        python_path = env_manager.get_env_python(self.env_name)
        if not python_path:
            # 无法直接定位到 python，退回到完整检查/创建逻辑
            exists, reason = env_manager.check_environment(self.env_name)
            if not exists:
                print(f"环境 '{self.env_name}' 不存在或未就绪，正在创建...")
                success = env_manager.create_environment(self.env_name)
                if not success:
                    raise RuntimeError(f"无法创建环境 '{self.env_name}'：{reason}")
    
    def fit(self, X, y):
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
        
        返回:
            self: 支持链式调用
        """
        # 保留 numpy 版本，供超时后的恢复流程复用。
        recovery_X = np.asarray(X)
        recovery_y = np.asarray(y).reshape(-1)
        if recovery_X.ndim == 1:
            recovery_X = recovery_X.reshape(-1, 1)

        # 准备可序列化数据
        X_payload = recovery_X.tolist()
        y_payload = recovery_y.tolist()
        
        # 创建命令
        command = {
            'action': 'fit',
            'data': {'X': X_payload, 'y': y_payload},
            'params': self.params,
            'tool_name': self.tool_name,
            'serialized_model': self.serialized_model  # 传递现有模型状态以支持继续训练
        }
        
        # 标记实验进入 running
        try:
            self._update_manifest(status="running")
        except Exception:
            pass

        # 执行命令并获取结果
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
            # 失败状态落盘后再抛出
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise
        
        # 检查结果
        if result.get('no_valid_output'):
            try:
                self._update_manifest(status="no_valid_output")
            except Exception:
                pass
            raise NoValidOutputError(result.get('message') or "算法未产生可评估符号表达式")

        if 'error' in result:
            try:
                self._update_manifest(status="failed")
            except Exception:
                pass
            raise RuntimeError(f"训练失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 保存模型状态
        self.serialized_model = result.get('serialized_model', {})
        # 成功状态
        try:
            self._update_manifest(status="success")
        except Exception:
            pass
        return self

    def _recover_from_timeout(self, X: np.ndarray, y: np.ndarray, fit_command: dict) -> Optional[str]:
        """超时后尝试从实验目录恢复可用模型。"""
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
        使用模型进行预测
        
        参数:
            X: 特征矩阵
        
        返回:
            predictions: 预测结果
        """
        # 检查模型是否已训练
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 准备数据
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # 创建命令
        command = {
            'action': 'predict',
            'data': {'X': X},
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"预测失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回预测结果
        predictions = result.get('predictions', [])
        return np.array(predictions)
    
    def get_optimal_equation(self):
        """
        获取模型学习到的最优符号方程
        
        返回:
            equation: 符号方程的字符串表示
        """
        # 检查模型是否已训练
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 创建命令
        command = {
            'action': 'get_optimal_equation',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"获取方程失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回方程
        return result.get('equation', '')
    

    def get_total_equations(self, n=None):
        """
        获取模型学习到的所有符号方程
        
        返回:
            equations: 符号方程的字符串表示列表
        """
        # 检查模型是否已训练
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 创建命令
        command = {
            'action': 'get_total_equations',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        
        # 执行命令并获取结果
        result = self._execute_subprocess(command)
        
        # 检查结果
        if 'error' in result:
            raise RuntimeError(f"获取所有方程失败: {result['message']}\n{result.get('traceback', '')}")
        
        # 返回方程列表
        return result.get('equations', [])


    def __str__(self):
        """
        返回模型的字符串表示
        
        返回:
            model_str: 模型的字符串表示
        """
        # 基础信息
        model_str = f"SymbolicRegressor(tool='{self.tool_name}'"
        
        for key, value in self.params.items():
            model_str += f", {key}={value}"
        model_str += ")"

        # 若已训练，尝试通过子进程获取最佳方程与参数
        if self.serialized_model is not None:
            try:
                equation = self.get_optimal_equation()
                if equation:
                    model_str += f"\n最佳方程:\n{equation}"
            except Exception as e:
                model_str += f"\n模型已训练，但无法获取方程: {str(e)}"
            try:
                params = self.get_fitted_params()
                if params is not None:
                    model_str += f"\n最佳参数: {params}"
            except Exception:
                pass
        else:
            model_str += "\n模型尚未训练"
        return model_str

    def get_fitted_params(self):
        """获取最佳方程的训练期拟合参数（若算法支持）。"""
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        command = {
            'action': 'get_fitted_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name
        }
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"获取参数失败: {result['message']}\n{result.get('traceback', '')}")
        return result.get('params')

    def get_total_equations_with_params(self, n=None):
        """获取所有（或Top-N）候选的方程与参数（若算法支持）。"""
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        command = {
            'action': 'get_total_equations_with_params',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name,
        }
        if n is not None:
            command['n'] = int(n)
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"获取方程与参数失败: {result['message']}\n{result.get('traceback', '')}")
        return result.get('items', [])

    def export_canonical_symbolic_program(self):
        """导出统一符号工件。

        当前返回 Phase 1 的最小 CanonicalSymbolicProgram，供后续 benchmark
        runner 与 normalizer 继续加工。
        """
        if self.serialized_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        command = {
            'action': 'export_canonical_symbolic_program',
            'serialized_model': self.serialized_model,
            'tool_name': self.tool_name,
        }
        result = self._execute_subprocess(command)
        if 'error' in result:
            raise RuntimeError(f"导出统一符号工件失败: {result['message']}\n{result.get('traceback', '')}")
        return result.get('artifact', {})


    def _execute_subprocess(self, command):
        """执行子进程命令"""
        # 获取Python解释器路径
        python_path = env_manager.get_env_python(self.env_name)
        if not python_path:
            raise RuntimeError(f"无法获取环境 '{self.env_name}' 的Python路径")
        
        # 创建临时文件存储命令
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as cmd_file:
            cmd_path = cmd_file.name
            json.dump(command, cmd_file)
        
        # 创建临时文件存储结果
        result_path = cmd_path + '.result'
        
        # 构建子进程命令
        runner_script = os.path.join(os.path.dirname(__file__), 'subprocess_runner.py')
        
        try:
            # 执行子进程（无缓冲），并实时转发其 stdout/stderr 到当前进程
            env = os.environ.copy()
            # 防止 julia/pip 等库在子进程内错误读取主环境CONDA_PREFIX，导致写权限到基环境
            # 统一将 CONDA_PREFIX 指向当前工具环境的真实目录，提升跨环境一致性
            try:
                py_path = env_manager.get_env_python(self.env_name)
                if py_path:
                    env["CONDA_PREFIX"] = str(Path(py_path).resolve().parent.parent)
            except Exception:
                pass
            # 避免 julia/pythoncall 在受限环境尝试在只读 conda 环境中创建目录
            pyjuliapkg_project = Path(tempfile.gettempdir()) / f"pyjuliapkg_{self.env_name}"
            if self.env_name in {"sim_fepysr", "sim_symbolfit"}:
                pyjuliapkg_project = Path.home() / "pyjuliapkg_symbolfit"
            env.setdefault("PYTHON_JULIAPKG_PROJECT", str(pyjuliapkg_project))
            if "PYTHON_JULIAPKG_EXE" not in env and "PYTHON_JULIACALL_BINDIR" not in env:
                julia_candidates = [
                    Path(env["PYTHON_JULIAPKG_PROJECT"]) / "pyjuliapkg" / "install" / "bin" / "julia",
                    Path.home() / "pyjuliapkg_pysr" / "pyjuliapkg" / "install" / "bin" / "julia",
                ]
                for julia_candidate in julia_candidates:
                    if julia_candidate.exists():
                        env["PYTHON_JULIAPKG_EXE"] = str(julia_candidate)
                        env["PYTHON_JULIACALL_BINDIR"] = str(julia_candidate.parent)
                        break
            # 如果工具不依赖 julia，此注入不会产生副作用
            env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
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
            # 使用后台线程实时转发 stdout/stderr，避免管道阻塞。
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
                    f"算法 '{self.tool_name}' 的子进程执行超时："
                    f"action={command.get('action')}, timeout_in_seconds={timeout_seconds}"
                ) from exc
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            if ret != 0:
                raise subprocess.CalledProcessError(ret, proc.args)

            # 读取结果
            with open(result_path, 'r') as f:
                result = json.load(f)

            return result
        except subprocess.CalledProcessError as e:
            result = self._safe_load_result_file(result_path)
            raise RuntimeError(f"子进程执行失败: {e}\n{result.get('traceback', '')}")
        except Exception as e:
            result = self._safe_load_result_file(result_path)
            if isinstance(e, TimeoutError):
                raise
            raise RuntimeError(f"执行命令时发生错误: {e}\n{result.get('traceback', '')}")
        finally:
            for path in (cmd_path, result_path):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

    def _resolve_fit_timeout_seconds(self, command: dict) -> Optional[int]:
        """解析 fit 阶段的总时长上限。"""
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
        """恢复阶段使用独立短超时，避免再次长时间挂起。"""
        if isinstance(fit_timeout_seconds, int) and fit_timeout_seconds > 0:
            return max(60, min(300, fit_timeout_seconds))
        return 300

    def _resolve_subprocess_timeout_seconds(self, command: dict) -> Optional[int]:
        """解析当前子进程命令的超时时间。"""
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

    # ========= 实验清单（manifest）最小实现 =========
    def _sanitize_config(self) -> dict:
        """脱敏/规整后的配置，用于写入 manifest 的 config 字段。"""
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
            # 迭代次数占位，暂不写入具体数值
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
        """更新迭代次数占位接口（当前不主动调用）。"""
        try:
            it = int(iterations)
        except Exception:
            return
        try:
            self._update_manifest(iterations=it)
        except Exception:
            pass
