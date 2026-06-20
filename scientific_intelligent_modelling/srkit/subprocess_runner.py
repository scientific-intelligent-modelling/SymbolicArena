# srkit/subprocess_runner.py
import argparse
import json
import importlib
import sys
import traceback
import os
try:
    from .config_manager import config_manager
except ImportError:
    from config_manager import config_manager

try:
    from scientific_intelligent_modelling.srkit.exceptions import NoValidOutputError
except ImportError:
    try:
        from .exceptions import NoValidOutputError
    except ImportError:
        from exceptions import NoValidOutputError
    from exceptions import NoValidOutputError

# print(f"当前工作目录: {os.getcwd()}")
# print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")

def main():
    """子进程执行入口点"""
    # 强制标准输出/错误按行刷新，避免缓冲堆积
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='符号回归子进程执行器')
    parser.add_argument('--input', required=True, help='输入命令文件路径')
    parser.add_argument('--output', required=True, help='输出结果文件路径')
    args = parser.parse_args()
    
    try:
        # 从文件读取命令
        with open(args.input, 'r') as f:
            command = json.load(f)
        
        # 获取工具名称
        tool_name = command.get('tool_name')
        if not tool_name:
            raise ValueError("命令中缺少工具名称")
        
        # 导入工具包装器
        wrapper_module = importlib.import_module(f"scientific_intelligent_modelling.algorithms.{tool_name}_wrapper.wrapper")
        
        # 执行命令
        result = execute_command(wrapper_module, command)
        
        # 写入结果
        with open(args.output, 'w') as f:
            json.dump(result, f)
            
    except NoValidOutputError as e:
        no_valid_result = {
            'success': False,
            'no_valid_output': True,
            'message': str(e),
            'traceback': traceback.format_exc(),
        }
        with open(args.output, 'w') as f:
            json.dump(no_valid_result, f)
    except Exception as e:
        # 错误处理
        error_result = {
            'error': True,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)

def execute_command(module, command):
    """根据命令类型执行相应操作"""
    action = command['action']
    tool_name = command['tool_name']
    
    # 获取正确的回归器类
    regressor_class = get_regressor_class(module, tool_name)
    
    if action == 'fit':
        return handle_fit(regressor_class, command)
    elif action == 'recover_from_timeout':
        return handle_recover_from_timeout(regressor_class, command)
    elif action == 'predict':
        return handle_predict(regressor_class, command)
    elif action == 'get_optimal_equation':
        return handle_get_optimal_equation(regressor_class, command)
    elif action == 'get_total_equations':
        return handle_get_total_equations(regressor_class, command)
    elif action == 'get_fitted_params':
        return handle_get_fitted_params(regressor_class, command)
    elif action == 'get_total_equations_with_params':
        return handle_get_total_equations_with_params(regressor_class, command)
    elif action == 'export_canonical_symbolic_program':
        return handle_export_canonical_symbolic_program(regressor_class, command)
    else:
        raise ValueError(f"未知操作: {action}")

def handle_get_optimal_equation(regressor_class, command):
    """处理get_optimal_equation操作，获取最优方程"""
    # 提取模型状态
    serialized_model = command['serialized_model']
    
    # 重建回归器
    regressor = regressor_class()
    
    # 如果有deserialize方法，用它加载状态
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # 获取方程
    equation = None
    
    # 不同工具可能有不同的获取方程的方法，尝试几种常见方法
    if hasattr(regressor, 'get_optimal_equation'):
        equation = regressor.get_optimal_equation()
    elif hasattr(regressor, 'get_equation'):
        equation = regressor.get_equation()
    elif hasattr(regressor, 'get_model_string'):
        equation = regressor.get_model_string()
    elif hasattr(regressor, 'symbolic_model'):
        equation = str(regressor.symbolic_model)
    elif hasattr(regressor, 'best_'):
        equation = str(regressor.best_)
    elif hasattr(regressor, 'program_'):
        equation = str(regressor.program_)
    else:
        raise ValueError("无法从模型中提取最优方程表达式")
    
    return {
        'success': True,
        'equation': equation
    }

def handle_get_total_equations(regressor_class, command):
    """处理get_total_equations操作，获取所有方程，支持 n 限制数量"""
    # 提取模型状态
    serialized_model = command['serialized_model']
    
    # 重建回归器
    regressor = regressor_class()
    
    # 如果有deserialize方法，用它加载状态
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # 获取所有方程
    equations = None
    
    # 从命令中读取可选参数 n
    n = command.get('n', None)

    # 尝试不同的方法获取所有方程
    if hasattr(regressor, 'get_total_equations'):
        try:
            equations = regressor.get_total_equations(n) if n is not None else regressor.get_total_equations()
        except TypeError:
            equations = regressor.get_total_equations()
    elif hasattr(regressor, 'get_all_equations'):
        equations = regressor.get_all_equations()
    elif hasattr(regressor, 'hall_of_fame_'):
        equations = [str(program) for program in regressor.hall_of_fame_]
    elif hasattr(regressor, 'models_'):
        equations = [str(model) for model in regressor.models_]
    else:
        raise ValueError("无法从模型中提取所有方程表达式")
    
    return {
        'success': True,
        'equations': equations
    }

def handle_get_fitted_params(regressor_class, command):
    """获取最佳方程的训练期拟合参数（若包装器实现）。"""
    serialized_model = command['serialized_model']
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    params = None
    if hasattr(regressor, 'get_fitted_params'):
        params = regressor.get_fitted_params()
    elif hasattr(regressor, '_best_params'):
        p = getattr(regressor, '_best_params')
        try:
            import numpy as _np
            params = _np.asarray(p).tolist() if p is not None else None
        except Exception:
            params = None
    else:
        raise ValueError("包装器未实现 get_fitted_params")
    return {
        'success': True,
        'params': params
    }

def handle_get_total_equations_with_params(regressor_class, command):
    """获取（或Top-N）候选方程与参数（若包装器实现）。"""
    serialized_model = command['serialized_model']
    n = command.get('n')
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    if hasattr(regressor, 'get_total_equations_with_params'):
        try:
            items = regressor.get_total_equations_with_params(n) if n is not None else regressor.get_total_equations_with_params()
        except TypeError:
            items = regressor.get_total_equations_with_params()
    else:
        raise ValueError("包装器未实现 get_total_equations_with_params")
    return {
        'success': True,
        'items': items
    }


def handle_export_canonical_symbolic_program(regressor_class, command):
    """导出统一符号工件。"""
    from scientific_intelligent_modelling.benchmarks.artifact_schema import (
        validate_canonical_symbolic_program,
    )

    serialized_model = command['serialized_model']
    regressor = regressor_class()
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)

    if not hasattr(regressor, 'export_canonical_symbolic_program'):
        raise ValueError("包装器未实现 export_canonical_symbolic_program")

    artifact = regressor.export_canonical_symbolic_program()
    artifact = validate_canonical_symbolic_program(artifact)
    return {
        'success': True,
        'artifact': artifact,
    }

# 从配置管理器获取工具映射
def get_class_mapping_from_config():
    # 获取toolbox配置
    toolbox_config = config_manager.get_config("toolbox_config")
    tool_mapping = toolbox_config.get("tool_mapping", {})
    
    # 构建工具名到类名的映射
    class_mapping = {}
    for tool_name, tool_info in tool_mapping.items():
        class_mapping[tool_name] = tool_info.get("regressor")
    
    # 如果需要添加配置中没有的映射，可以在这里手动添加
    if "srbench" not in class_mapping:
        class_mapping["srbench"] = "SRBenchRegressor"
        
    return class_mapping

def get_regressor_class(module, tool_name):
    """根据工具名获取对应的回归器类"""
    # 工具名到类名的映射
    class_mapping = get_class_mapping_from_config()
    
    class_name = class_mapping.get(tool_name)
    if class_name and hasattr(module, class_name):
        return getattr(module, class_name)
    
    # # 后备方案：尝试查找任何以Regressor结尾的类
    # for attr_name in dir(module):
    #     if attr_name.endswith('Regressor'):
    #         return getattr(module, attr_name)
    
    raise ValueError(f"在模块 {module.__name__} 中未找到合适的回归器类")

def handle_fit(regressor_class, command):
    """处理fit操作"""
    import numpy as np
    
    # 提取数据和参数
    data = command['data']
    params = dict(command.get('params', {}) or {})
    timeout_in_seconds = params.pop("timeout_in_seconds", None)
    if timeout_in_seconds is None:
        timeout_in_seconds = command.get("timeout_in_seconds")
    if timeout_in_seconds is not None and str(getattr(regressor_class, "__module__", "")).startswith(
        "scientific_intelligent_modelling.algorithms."
    ):
        params["timeout_in_seconds"] = timeout_in_seconds
    
    # 转换为numpy数组
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # 创建回归器并训练
    regressor = None
    serialized_model_from_command = command.get('serialized_model')

    if serialized_model_from_command and hasattr(regressor_class, 'deserialize'):
        # 尝试反序列化以从之前的状态恢复
        regressor = regressor_class.deserialize(serialized_model_from_command)
        # 注意：这里的 `params` 参数是本次 fit 调用的，而不是初始化的。
        # 如果具体的算法包装器需要在反序列化后更新参数（例如增加迭代次数），
        # 则需要在包装器内部的 `fit` 方法中处理。
        # 我们这里假设反序列化会恢复完整的模型状态，并且 `fit` 方法能基于此状态继续训练。
    else:
        # 第一次拟合，或没有提供序列化模型，初始化一个新的回归器
        regressor = regressor_class(**params)
    regressor.fit(X, y)
    
    # 序列化模型状态
    serialized_model = None
    if hasattr(regressor, 'serialize'):
        serialized_model = regressor.serialize()
    else:
        raise ValueError(f"回归器 {regressor_class.__name__} 未实现serialize方法")
    
    return {
        'success': True,
        'serialized_model': serialized_model
    }


def handle_recover_from_timeout(regressor_class, command):
    """处理超时后的实验目录恢复。"""
    import numpy as np

    data = command.get('data') or {}
    params = dict(command.get('params', {}) or {})
    X = np.array(data.get('X', []))
    y = np.array(data.get('y', []))
    if X.ndim == 1 and X.size > 0:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)

    experiment_dir = command.get('experiment_dir')
    if isinstance(experiment_dir, str) and experiment_dir.strip():
        params.setdefault('existing_exp_dir', experiment_dir.strip())
        params.setdefault('exp_dir', experiment_dir.strip())

    regressor = regressor_class(**params)

    # 部分 wrapper 仅在 fit 中实现“从已有实验目录恢复”的逻辑。
    if command.get('tool_name') == 'drsr':
        regressor.fit(X, y)

    equation = ""
    if hasattr(regressor, 'get_optimal_equation'):
        equation = str(regressor.get_optimal_equation() or "")
    if not equation.strip():
        raise ValueError("超时恢复失败：未找到可用最优方程")

    if X.size > 0:
        sample_rows = min(8, X.shape[0])
        sample_X = X[:sample_rows]
        predictions = np.asarray(regressor.predict(sample_X)).reshape(-1)
        if predictions.shape != (sample_rows,):
            raise ValueError(
                f"超时恢复失败：预测形状异常，期望 {(sample_rows,)}, 实际 {predictions.shape}"
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError("超时恢复失败：预测包含非有限值")

    if hasattr(regressor, 'serialize'):
        serialized_model = regressor.serialize()
    else:
        raise ValueError(f"回归器 {regressor_class.__name__} 未实现 serialize 方法")

    return {
        'success': True,
        'serialized_model': serialized_model,
        'equation': equation,
        'recovered_from_timeout': True,
    }

def handle_predict(regressor_class, command):
    """处理predict操作"""
    import numpy as np
    
    # 提取数据和模型状态
    data = command['data']
    serialized_model = command['serialized_model']
    
    # 转换为numpy数组
    X = np.array(data['X'])
    
    # 重建回归器并预测
    regressor = regressor_class()  # 使用动态获取的类
    
    # 如果有deserialize方法，用它加载状态
    if hasattr(regressor, 'deserialize'):
        regressor = regressor.deserialize(serialized_model)
    
    # 执行预测
    predictions = regressor.predict(X)
    
    # 确保预测结果可序列化
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    
    return {
        'success': True,
        'predictions': predictions
    }

if __name__ == "__main__":
    main()
