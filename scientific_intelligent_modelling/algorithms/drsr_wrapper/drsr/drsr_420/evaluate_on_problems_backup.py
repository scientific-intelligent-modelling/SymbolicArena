import numpy as np
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS
from scipy.optimize import minimize
from parallel_bfgs import parallel_multi_start_bfgs

# 全局变量定义保留小数点的位数
DECIMAL_PLACES = 3

def evaluate(data: dict , equation) -> float:
        """ 从大模型的输出program中直接获取的"""
        print('我运行了!')

        inputs, outputs = data['inputs'], data['outputs']
        X = inputs
        X_rounded = np.round(X, DECIMAL_PLACES)
        # Optimize parameters based on data
        
        def loss(params):
            y_pred = equation(*X.T, params)
            return np.mean((y_pred - outputs) ** 2)

        loss_partial = lambda params: loss(params)
        result = parallel_multi_start_bfgs(loss_partial, n_params=MAX_NPARAMS)

        # Return evaluation score
        optimized_params = result.x
        loss = result.fun
        if np.isnan(loss) or np.isinf(loss):
            return None
        else:
            # 计算并输出优化后的方程在输入数据上的预测结果
            optimized_predictions = equation(*X.T, optimized_params)
            # 计算残差（实际值 - 预测值）
            res = outputs - optimized_predictions
            
            # 使用全局变量保留预测结果和残差的小数位数
            res_rounded = np.round(res, DECIMAL_PLACES)
            outputs_rounded = np.round(outputs, DECIMAL_PLACES)  # 计算输出的四舍五入值
            # 直接返回数据矩阵，使用X_rounded作为输入数据
            result_data = np.column_stack((X_rounded, outputs_rounded, res_rounded))
            result_data = np.round(result_data, DECIMAL_PLACES)  # 保留小数点位数
            return -loss, result_data
