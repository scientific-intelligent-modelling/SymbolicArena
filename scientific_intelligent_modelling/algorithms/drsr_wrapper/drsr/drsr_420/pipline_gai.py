# 从sampler_real模块导入DataAnalyzer类
from data_analyse_real import DataAnalyzer
import json
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd


# 创建DataAnalyzer实例
analyzer = DataAnalyzer(timeout=600)  # 可以自定义参数

# 分析指定的CSV文件
result = analyzer.analyze(
    csv_file_path="/data/home/zdhs0036/DrSR/data/oscillator1/train.csv",
    # max_rows=1000,  # 可选：限制行数
    verbose=True    # 可选：显示详细信息
)

# 打印分析结果
print("\n===== 分析结果 =====")
print(result)
