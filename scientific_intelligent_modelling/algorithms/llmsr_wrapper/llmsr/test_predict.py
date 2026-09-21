from __future__ import annotations

"""使用 LLMSRRegressor 的 fit + predict 做一次端到端测试。"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from llmsr_regressor import LLMSRRegressor


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--problem_name", type=str, default="oscillator1")
    parser.add_argument("--exp_path", type=str, default="./experiments", help="实验根目录（默认 ./experiments）")
    parser.add_argument("--data_csv", type=str, default=None, help="训练使用的 CSV 路径（默认 ./data/<problem_name>/train.csv，仅用于初始化 Regressor）")
    parser.add_argument("--test_csv", type=str, default=None, help="预测数据的 CSV 路径（默认 ./data/<problem_name>/test_id.csv）")
    parser.add_argument("--llm_config", type=str, default="llm.config", help="LLM 配置文件路径（predict 只用到路径记录，不会调用大模型）")
    parser.add_argument("--background", type=str, default="这是一个抽象的数学回归问题。", help="问题背景描述，将写入动态规格")
    parser.add_argument("--max_params", type=int, default=10, help="规格中可优化参数个数（MAX_NPARAMS）")
    parser.add_argument("--niterations", type=int, default=3, help="搜索轮数（默认 3，用于快速测试）")
    parser.add_argument("--samples_per_iteration", type=int, default=4, help="每轮生成的候选数量（默认 4）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（仅作用于本地 NumPy/random/子进程）")
    parser.add_argument(
        "--anonymize",
        type=str,
        default="False",
        help="是否对变量名进行匿名化（仅当值为 True/true 时生效）",
    )
    args = parser.parse_args()

    problem_name = args.problem_name
    exp_path = args.exp_path

    # 默认的训练 / 测试 CSV 路径
    train_csv = args.data_csv or os.path.join("data", problem_name, "train.csv")
    test_csv = args.test_csv or os.path.join("data", problem_name, "test_id.csv")

    print(f"训练数据: {train_csv}")
    print(f"预测数据: {test_csv}")
    anonymize_flag = str(args.anonymize).lower() == "true"

    # 构造回归器实例，并直接在本脚本中执行一次 fit
    reg = LLMSRRegressor(
        problem_name=problem_name,
        data_csv=train_csv,
        llm_config_path=args.llm_config,
        background=args.background,
        exp_path=exp_path,
        exp_name=None,
        max_params=args.max_params,
        niterations=args.niterations,
        samples_per_iteration=args.samples_per_iteration,
        seed=args.seed,
        existing_exp_dir=None,
         anonymize=anonymize_flag,
    )

    print("开始执行 fit() ...")
    reg.fit()
    print("fit() 完成，开始预测 ...")

    df_test = pd.read_csv(test_csv)

    # 若使用匿名化，测试集列名也改为 x1,x2,...,y 以与训练阶段保持一致
    if anonymize_flag:
        cols = list(df_test.columns)
        if len(cols) < 2:
            raise ValueError("测试集 CSV 至少需要 2 列（前 n-1 列为特征，最后 1 列为目标）")
        n = len(cols)
        new_cols = [f"x{i+1}" for i in range(n - 1)] + ["y"]
        df_test.columns = new_cols

    # 直接传入完整的 DataFrame，Regressor 会根据训练时的 feature_names 自动选取列
    y_pred = reg.predict(df_test)

    print("预测结果示例（前 10 条）：")
    print(y_pred[:10])

    # 计算 MSE 与 NMSE（相对于真实 y 的方差进行归一化）
    cols = list(df_test.columns)
    if len(cols) < 2:
        raise ValueError("测试集 CSV 至少需要 2 列（前 n-1 列为特征，最后 1 列为目标）")
    y_true = df_test[cols[-1]].to_numpy()

    mse = float(np.mean((y_pred - y_true) ** 2))
    var_y = float(np.var(y_true))
    nmse = float(mse / var_y) if var_y > 0 else float("nan")

    print(f"\nMSE  = {mse:.6g}")
    print(f"NMSE = {nmse:.6g}  （MSE / Var(y_true)）")
