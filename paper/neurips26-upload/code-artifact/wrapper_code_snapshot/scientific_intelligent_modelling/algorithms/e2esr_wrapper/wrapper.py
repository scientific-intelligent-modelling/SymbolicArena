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
        初始化E2ESR回归器
        
        参数：
        model_path: 模型文件路径，如果为None，则使用默认路径
        model_url: 模型下载URL，如果本地没有模型文件则从此URL下载
        max_input_points: 最大输入点数
        max_number_bags: 最大 bag 数，默认对齐官方 evaluate.py 的 black-box 评测口径
        stop_refinement_after: 精炼停止条件
        n_trees_to_refine: 要优化的树的数量
        rescale: 是否重新缩放数据
        torch_num_threads: CPU 推理时每个进程允许使用的 PyTorch 线程数
        """
        # 保存参数
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
        
        # 获取e2esr模块的路径，添加到系统路径中
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.e2esr_path = os.path.join(current_dir, "e2esr")
        if self.e2esr_path not in sys.path:
            sys.path.insert(0, self.e2esr_path)
        
        # 如果提供了模型路径，则使用该路径，否则使用默认路径
        if model_path is None:
            # 优先使用包装器目录下的 model.pt，其次兼容历史的 e2esr/model1.pt
            model_path = self._resolve_default_model_path(current_dir)
        
        self.model_path = model_path
        self.model_url = model_url
        
        # 立即加载模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型，如果本地不存在则从URL下载"""
        # 如果模型已经加载，直接返回
        if self.model is not None:
            return
            
        try:
            shared_model_path = os.environ.get("SIM_SYMBOLICREGRESSION_MODEL_PATH")
            if not os.path.isfile(self.model_path) and shared_model_path and os.path.isfile(shared_model_path):
                self.model_path = shared_model_path
            # 检查模型文件是否存在，不存在则从URL下载
            if not os.path.isfile(self.model_path):
                print(f"从 {self.model_url} 下载模型...")
                r = requests.get(self.model_url, allow_redirects=True)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    f.write(r.content)
                print(f"模型已保存到 {self.model_path}")
            
            # 加载模型
            force_cpu = bool(self.params.get('force_cpu', True))
            if force_cpu:
                self._configure_cpu_threads(self.params.get('torch_num_threads', 1))

            # 默认优先 CPU，避免不同机器 CUDA 兼容性问题（例如驱动/算力不一致导致加载失败）
            if force_cpu:
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                self.model = self.model.cpu()
            else:
                # 保留旧行为：有 CUDA 时优先上卡，未命中则自动回退 CPU
                try:
                    if not torch.cuda.is_available():
                        self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                    else:
                        self.model = torch.load(self.model_path, map_location=torch.device('cuda'))
                        self.model = self.model.cuda()
                except Exception:
                    # 兼容某些环境的 CUDA 无法匹配时回退 CPU（例如显卡能力不匹配）
                    self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                    self.model = self.model.cpu()
            
            print(f"模型已成功加载！设备: {self.model.device if hasattr(self.model, 'device') else 'cpu'}")
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            # 如果加载失败，模型将在fit时创建
            self.model = None

    @staticmethod
    def _configure_cpu_threads(torch_num_threads):
        """限制单进程 CPU 线程，避免多 worker 批量评测时线程爆炸。"""
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
            # PyTorch 可能在并行运行时初始化后禁止再次设置 interop 线程。
            pass
    
    def fit(self, X, y):
        """
        训练模型，如果已经加载了预训练模型，则使用该模型
        否则创建一个新模型
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
            # 导入E2ESR相关模块
            from symbolicregression.model import SymbolicTransformerRegressor
            
            # 如果模型尚未加载，报错
            if self.model is None:
                raise ValueError("模型未加载成功，请检查模型路径或网络连接")

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
                # 剔除 SymbolicRegressor 注入的元参数，避免 __init__ 透传失败
                pass
            
            self.regressor = SymbolicTransformerRegressor(
                model=self.model,
                progress_state_path=self._progress_state_path,
                **regressor_kwargs
            )
            
            # 训练回归器
            self.regressor.fit(X, y)
            
            # 获取并保存最佳树
            self.best_tree = self.regressor.retrieve_tree(with_infos=True)
            
            return self
            
        except Exception as e:
            print(f"训练模型时出错: {str(e)}")
            raise
    
    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.regressor is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.regressor.predict(X)
    
    def get_optimal_equation(self):
        """返回模型拟合的最优数学方程"""
        if self.best_tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果已经计算过方程，直接返回缓存的结果
        if hasattr(self, '_cached_optimal_equation'):
            return self._cached_optimal_equation
            
        try:
            # 获取表达式并格式化
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            model_str = self.best_tree["relabed_predicted_tree"].infix()
            for op, replace_op in replace_ops.items():
                model_str = model_str.replace(op, replace_op)
            
            # 解析为sympy表达式并转为字符串
            expr = sp.parse_expr(model_str)
            self._cached_optimal_equation = str(expr)
            return self._cached_optimal_equation
        except Exception as e:
            print(f"获取方程时出错: {str(e)}")
            # 如果解析失败，返回原始的中缀表达式
            self._cached_optimal_equation = str(self.best_tree["relabed_predicted_tree"].infix())
            return self._cached_optimal_equation

    def get_total_equations(self):
        """获取模型学习到的所有符号方程"""
        if self.best_tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果已经计算过所有方程，直接返回缓存的结果
        if hasattr(self, '_cached_total_equations'):
            return self._cached_total_equations
            
        # E2ESR可能没有提供获取多个方程的方法，这里返回最优方程作为单元素列表
        self._cached_total_equations = [self.get_optimal_equation()]
        return self._cached_total_equations

    def export_canonical_symbolic_program(self):
        if self.best_tree is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return normalize_e2esr_artifact(
            self.get_optimal_equation(),
            expected_n_features=getattr(self, "n_features_", None),
        )


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 生成示例数据
    X = np.random.randn(100, 2)
    y = np.cos(2*np.pi*X[:, 0]) + X[:, 1]**2

    # 创建模型，指定预训练模型路径
    model = E2ESRRegressor()
    
    # 训练模型
    model.fit(X, y)

    # 获取最优方程
    equation = model.get_optimal_equation()
    print(f"最优方程: {equation}")

    # 进行预测
    y_pred = model.predict(X)
    print(f"预测值前5个: {y_pred[:5]}")
