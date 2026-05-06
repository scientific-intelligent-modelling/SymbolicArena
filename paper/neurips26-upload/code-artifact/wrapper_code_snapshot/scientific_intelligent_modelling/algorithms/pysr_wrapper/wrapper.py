# tools/pysr_wrapper/wrapper.py
import os
import json
import numpy as np
from copy import deepcopy

from ..base_wrapper import BaseWrapper 
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_pysr_artifact

class PySRRegressor(BaseWrapper):
    _DEFAULT_PARAMS = {
        "timeout_in_seconds": 3600,
        "niterations": 10000000,
        "population_size": 64,
        "populations": 8,
        "ncycles_per_iteration": 500,
        "maxsize": 30,
        "maxdepth": 10,
        "parsimony": 0.001,
        "constraints": {
            "/": [-1, 9],
            "square": 9,
            "cube": 9,
            "exp": 7,
            "log": 7,
            "sin": 9,
            "cos": 9,
        },
        "nested_constraints": {
            "exp": {"exp": 0, "log": 1},
            "log": {"exp": 0, "log": 0},
            "square": {"square": 1, "cube": 1, "exp": 0, "log": 0},
            "cube": {"square": 1, "cube": 1, "exp": 0, "log": 0},
        },
        "complexity_of_operators": {
            "/": 2,
            "square": 2,
            "cube": 3,
            "sin": 2,
            "cos": 2,
            "exp": 3,
            "log": 3,
        },
        "complexity_of_constants": 2,
        "complexity_of_variables": 1,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["square", "cube", "exp", "log", "sin", "cos"],
        "precision": 32,
        "deterministic": True,
        "parallelism": "serial",
        "model_selection": "best",
        "progress": True,
        "verbosity": 1,
        "procs": 1,
    }
    _META_PARAMS = {"exp_name", "exp_path", "problem_name", "seed", "n_features", "feature_names", "target_name"}
    _THREAD_ENV_VARS = (
        "PYTHON_JULIACALL_THREADS",
        "JULIA_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    _FIXED_THREAD_COUNT = "4"
    _ALLOWED_PARAMS = {
        "niterations",
        "timeout_in_seconds",
        "max_evals",
        "early_stop_condition",
        "population_size",
        "populations",
        "ncycles_per_iteration",
        "maxsize",
        "maxdepth",
        "parsimony",
        "constraints",
        "nested_constraints",
        "complexity_of_operators",
        "complexity_of_constants",
        "complexity_of_variables",
        "binary_operators",
        "unary_operators",
        "elementwise_loss",
        "model_selection",
        "loss_function",
        "warm_start",
        "random_state",
        "precision",
        "deterministic",
        "parallelism",
        "procs",
        "n_jobs",
        "run_id",
        "output_directory",
        "verbosity",
        "progress",
    }
    
    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        self._contract_n_features = kwargs.get("n_features")
        self._contract_feature_names = kwargs.get("feature_names")
        self._contract_target_name = kwargs.get("target_name")
        existing_exp_dir = kwargs.pop("existing_exp_dir", None)
        self.params = self._validate_and_normalize_params(kwargs)
        self.model = None
        self._exp_dir = self._normalize_exp_dir(existing_exp_dir) or self._resolve_run_directory_from_params(self.params)

    @classmethod
    def _validate_and_normalize_params(cls, raw_params):
        params = {}
        raw_params = dict(raw_params)
        # 先保存元参数，便于后续复现性映射
        seed = raw_params.get("seed")
        exp_name = raw_params.get("exp_name")
        exp_path = raw_params.get("exp_path")
        # 剥离系统参数
        for key in cls._META_PARAMS:
            raw_params.pop(key, None)

        # 复现性参数透传映射
        if "random_state" not in raw_params and seed is not None:
            params["random_state"] = int(seed)
        # 将框架统一实验目录映射为 PySR 原生输出目录。
        if "run_id" not in raw_params and isinstance(exp_name, str) and exp_name.strip():
            params["run_id"] = exp_name.strip()
        if "output_directory" not in raw_params and isinstance(exp_path, str) and exp_path.strip():
            params["output_directory"] = os.path.abspath(exp_path.strip())

        for key, value in cls._DEFAULT_PARAMS.items():
            raw_params.setdefault(key, deepcopy(value))

        unknown = sorted(set(raw_params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "PySR 参数不受支持: {}。当前允许的参数有: {}。".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        for key, value in raw_params.items():
            if key == "n_jobs":
                # 与 pysr 主参数兼容：统一使用 procs
                params.setdefault("procs", int(value))
                continue
            if key == "seed":
                continue
            params[key] = value

        # 默认打开进度与基础日志，便于远程实验观测。
        params.setdefault("progress", True)
        params.setdefault("verbosity", 1)

        return params

    @classmethod
    def _apply_fixed_thread_env(cls):
        """固定 PySR/Julia 相关线程数，避免远程环境并行层叠失控。"""
        for key in cls._THREAD_ENV_VARS:
            os.environ[key] = cls._FIXED_THREAD_COUNT

    @staticmethod
    def _normalize_exp_dir(path):
        if not isinstance(path, str) or not path.strip():
            return None
        return os.path.abspath(path.strip())

    @staticmethod
    def _resolve_run_directory_from_params(params):
        output_directory = params.get("output_directory")
        run_id = params.get("run_id")
        if not isinstance(output_directory, str) or not output_directory.strip():
            return None
        if not isinstance(run_id, str) or not run_id.strip():
            return None
        return os.path.abspath(os.path.join(output_directory.strip(), run_id.strip()))

    def _resolve_run_directory_from_model(self):
        if self.model is None:
            return None

        for attr in ("run_directory_", "run_directory"):
            value = getattr(self.model, attr, None)
            if isinstance(value, str) and value.strip():
                return os.path.abspath(value.strip())

        output_directory = (
            getattr(self.model, "output_directory_", None)
            or getattr(self.model, "output_directory", None)
        )
        run_id = getattr(self.model, "run_id_", None) or getattr(self.model, "run_id", None)
        if isinstance(output_directory, str) and output_directory.strip() and isinstance(run_id, str) and run_id.strip():
            return os.path.abspath(os.path.join(output_directory.strip(), run_id.strip()))
        return None

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        if not self._exp_dir:
            raise ValueError("模型尚未训练，请先调用fit方法")

        self._apply_fixed_thread_env()
        from pysr import PySRRegressor as CorePySR

        self.model = CorePySR.from_file(run_directory=str(self._exp_dir))
        return self.model
    
    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="PySRRegressor.fit",
        )
        # 必须在导入 pysr/juliacall 前固定线程环境，否则远程多核环境容易过度订阅。
        self._apply_fixed_thread_env()
        # 仅在需要时导入
        from pysr import PySRRegressor as CorePySR
        
        # 创建并训练模型
        self.model = CorePySR(**self.params)
        self.model.fit(X, y)
        self._exp_dir = self._resolve_run_directory_from_model() or self._resolve_run_directory_from_params(self.params)

        return self

    def serialize(self):
        exp_dir = self._exp_dir or self._resolve_run_directory_from_model() or self._resolve_run_directory_from_params(self.params)
        if exp_dir:
            state = {
                "mode": "exp_dir",
                "params": self.params,
                "exp_dir": exp_dir,
            }
            return json.dumps(state, ensure_ascii=False)
        return super().serialize()

    @classmethod
    def deserialize(cls, payload):
        try:
            obj = json.loads(payload)
        except Exception:
            return BaseWrapper.deserialize(payload)

        if not isinstance(obj, dict) or obj.get("mode") != "exp_dir":
            return BaseWrapper.deserialize(payload)

        inst = cls(existing_exp_dir=obj.get("exp_dir"), **dict(obj.get("params") or {}))
        inst._exp_dir = inst._normalize_exp_dir(obj.get("exp_dir")) or inst._resolve_run_directory_from_params(inst.params)
        return inst
    
    def predict(self, X):
        model = self._ensure_model()
        return model.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的数学方程"""
        model = self._ensure_model()
        
        # 返回模型的字符串表示，这就是拟合的方程
        # self.model.best()
        return str(model.sympy())
    
    def get_total_equations(self):
        """
            获取模型学习到的所有符号方程
        """
        model = self._ensure_model()
        equations = model.equations_
        if hasattr(equations, "to_dict"):
            # pandas.DataFrame 常见于 PySR：优先提取可读表达式列并保序
            if hasattr(equations, "columns"):
                for col in ("sympy_format", "equation", "expr", "expression"):
                    if col in equations.columns:
                        return [str(item) for item in equations[col].dropna().tolist()]
                # 最后兜底：每行转为字符串字典
                return [
                    {k: str(v) for k, v in row.items()}
                    for row in equations.to_dict(orient="records")
                ]
        if isinstance(equations, list):
            return [str(eq) for eq in equations]
        return [str(equations)]

    def export_canonical_symbolic_program(self):
        model = self._ensure_model()
        return normalize_pysr_artifact(
            self.get_optimal_equation(),
            expected_n_features=getattr(model, "n_features_in_", None),
        )
  


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 生成示例数据
    X = np.random.rand(100, 2)
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.1*np.random.randn(100)

    # 创建并训练模型
    model = PySRRegressor(niterations=5, population_size=1000)
    model.fit(X, y)

    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    equations = model.get_total_equations()
    print(f"所有方程: {equations}")
