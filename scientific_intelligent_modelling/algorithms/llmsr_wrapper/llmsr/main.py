from argparse import ArgumentParser

from llmsr_regressor import LLMSRRegressor


parser = ArgumentParser()
parser.add_argument('--llm_config', type=str, required=True, help='LLM 配置文件路径（llm.config）')
parser.add_argument('--exp_path', type=str, default="./experiments", help='实验根目录（默认 ./experiments）')
parser.add_argument('--exp_name', type=str, default=None, help='实验名称（默认 问题名_时间戳）')
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--data_csv', type=str, required=True, help='CSV 路径：首行表头，前 n-1 列为特征，最后一列为目标')
parser.add_argument('--background', type=str, default='', help='问题背景描述，将写入动态规格')
parser.add_argument('--max_params', type=int, default=10, help='规格中可优化参数个数（MAX_NPARAMS）')
parser.add_argument('--niterations', type=int, default=2500, help='搜索轮数（默认 2500 轮）')
parser.add_argument('--samples_per_iteration', type=int, default=4, help='每轮生成的候选数量（默认 4）')
parser.add_argument('--timeout_in_seconds', type=int, default=None, help='总超时时间（秒）；达到即停止搜索')
parser.add_argument('--seed', type=int, default=None, help='随机种子（仅作用于本地 NumPy/random/子进程）')
parser.add_argument(
    '--anonymize',
    type=str,
    default="False",
    help='是否对变量名进行匿名化（仅当值为 True/true 时生效）',
)
args = parser.parse_args()


if __name__ == '__main__':
    anonymize_flag = str(args.anonymize).lower() == "true"
    reg = LLMSRRegressor(
        problem_name=args.problem_name,
        data_csv=args.data_csv,
        llm_config_path=args.llm_config,
        background=args.background,
        exp_path=args.exp_path,
        exp_name=args.exp_name,
        max_params=args.max_params,
        niterations=args.niterations,
        samples_per_iteration=args.samples_per_iteration,
        timeout_in_seconds=args.timeout_in_seconds,
        seed=args.seed,
        anonymize=anonymize_flag,
    )
    reg.fit()
