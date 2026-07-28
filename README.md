# Scientific Intelligent Modelling（科学智能建模）

![Scientific Intelligent Modelling](./images/cover.png)

Scientific Intelligent Modelling 是一个统一接入、运行和评测符号回归算法的工具箱。项目通过 `SymbolicRegressor` 提供一致的 Python API，并使用独立 Conda 环境和子进程隔离不同算法的依赖。

## 功能概览

- **统一接口**：不同算法共享 `fit`、`predict`、`get_optimal_equation` 和 `get_total_equations`。
- **环境隔离**：每类算法映射到独立或可复用的 Conda 环境，减少依赖冲突。
- **命令行入口**：通过 `sim-cli` 直接运行 CSV、NumPy 文件或标准数据集目录。
- **标准评测**：标准数据集目录会自动进入统一 benchmark runner，产出方程、指标和运行记录。
- **可扩展接入**：提供 manifest、脚手架生成器、结构校验和离线 smoke check。

当前已注册的算法包括：

| 算法 ID | Conda 环境 | 说明 |
| --- | --- | --- |
| `gplearn`、`pysr`、`pyoperon` | `sim_base` | 经典符号回归算法 |
| `fepysr` | `sim_fepysr` | 神经特征提取与 PySR |
| `jaxsr` | `sim_jaxsr` | 基于 JAX 的符号回归 |
| `symbolfit` | `sim_symbolfit` | 方程搜索与参数重优化 |
| `dso`、`udsr` | `sim_dso` | Deep Symbolic Optimization 系列 |
| `llmsr`、`drsr` | `sim_llm` | 大模型驱动的符号回归 |
| `tpsr` | `sim_tpsr` | Transformer 规划式符号回归 |
| `e2esr` | `sim_e2esr` | 端到端符号回归 |
| `ragsr` | `sim_ragsr` | 检索增强符号回归 |
| `QLattice` | `sim_qLattice` | QLattice 符号建模 |
| `iMCTS` | `sim_iMCTS` | 蒙特卡洛树搜索符号回归 |

算法 ID 区分大小写，应以
[`toolbox_config.json`](./scientific_intelligent_modelling/config/toolbox_config.json)
中的注册值为准。

## 安装

### 1. 前置条件

- Linux 或 macOS
- Git
- Conda（推荐 Miniconda 或 Miniforge）
- 能够访问所选算法需要的软件源、模型权重或 API

主控环境使用 Python 3.10。部分算法环境使用其他 Python 版本，具体配置见
[`envs_config.json`](./scientific_intelligent_modelling/config/envs_config.json)。

### 2. 克隆仓库及子模块

```bash
git clone \
  https://github.com/scientific-intelligent-modelling/scientific-intelligent-modelling.git
cd scientific-intelligent-modelling

# 拉取算法源码和数据集 Python 包，不要求访问论文私有子模块。
git submodule update --init --recursive -- \
  scientific_intelligent_modelling/algorithms \
  sim-datasets-py
```

若你有全部子模块的访问权限，也可以直接使用
`git clone --recursive`，或在已有仓库中执行
`git submodule update --init --recursive`。

### 3. 安装主控环境

```bash
conda env create -f environment.yml
conda activate sim
python -m pip install -e .
```

安装完成后可确认命令行入口：

```bash
sim-cli --help
```

### 4. 创建算法环境

首次实例化某个算法时，框架会根据配置尝试自动创建对应环境。为了让安装过程更可控，也可以提前创建需要的环境：

```python
from scientific_intelligent_modelling.srkit.conda_env_manager import env_manager

# 按实际需要选择，不必一次安装所有算法。
for env_name in ["sim_base", "sim_dso"]:
    if not env_manager.create_environment(env_name):
        raise RuntimeError(f"环境创建失败: {env_name}")
```

将上面代码保存为脚本后，在仓库根目录和 `sim` 环境中运行。也可以进入交互式环境管理器：

```bash
python -m scientific_intelligent_modelling.srkit.conda_env_manager
```

> 部分环境体积较大，且可能需要下载模型。建议只创建当前任务需要的环境。

### 5. 准备数据集（可选）

仓库约定：

- `sim-datasets-py/`：数据集 Python 包源码；
- `sim-datasets-data/`：标准数据集文件目录。

数据文件通常不随主仓库 Git 历史完整分发。需要完整 benchmark 数据时，可按
[`sim-datasets-data/README.md`](./sim-datasets-data/README.md)
从 Hugging Face 或 ModelScope 获取，并安装 Git LFS。

若 CLI 使用相对于数据集根目录的路径，设置：

```bash
export SIM_DATASETS_PATH="$(pwd)/sim-datasets-data"
```

## 快速使用

### Python API

下面以安装较轻量的 `gplearn` 为例：

```python
import numpy as np

from scientific_intelligent_modelling.srkit.regressor import SymbolicRegressor

rng = np.random.RandomState(0)
X = rng.rand(100, 2)
y = X[:, 0] ** 2 + X[:, 1] + 0.01 * rng.randn(100)

regressor = SymbolicRegressor(
    "gplearn",
    problem_name="quickstart",
    seed=42,
    population_size=500,
    generations=10,
)
regressor.fit(X, y)

print("最优方程:", regressor.get_optimal_equation())
print("候选方程:", regressor.get_total_equations()[:3])
print("预测结果:", regressor.predict(X[:5]))
```

不同算法的参数通过 `SymbolicRegressor(..., **kwargs)` 传入包装器。已有的参数和依赖说明可查阅
[`docs/`](./docs/) 下对应的 `工具_<算法>.md`，其余算法以包装器实现和配置文件为准。

### 命令行运行单个文件

CSV 默认第一行为表头、最后一列为目标列：

```bash
sim-cli \
  --algorithm gplearn \
  --train-path /path/to/train.csv \
  --dataset-name demo \
  --seed 42 \
  --population-size 500 \
  --generations 10
```

除通用参数外，其余 `--key value` 或 `--key=value` 参数会转成下划线命名并传给算法包装器。例如 `--population-size` 会作为 `population_size` 传入。

CLI 还支持：

- 带表头或不带表头的 CSV/文本数据；
- `.npy`；
- 包含 `arr_0` 的 `.npz`；
- 标准数据集目录。

### 运行标准数据集

标准数据集目录至少包含：

```text
dataset_name/
├── metadata.yaml
├── train.csv
├── valid.csv
├── id_test.csv
└── ood_test.csv
```

将目录直接传给 `--train-path`，CLI 会自动切换到统一 benchmark runner：

```bash
sim-cli \
  --algorithm pysr \
  --train-path sim-datasets-data/srbench1.0/black-box/dataset_name \
  --seed 42 \
  --timeout-in-seconds 600 \
  --output-root bench_results/quickstart
```

标准 runner 会读取 `metadata.yaml` 中的特征名和目标列，并显式向包装器注入 `n_features`、`feature_names` 和 `target_name` 数据契约。

### 运行算法自检

先运行目标算法的 check 脚本，不建议一开始就执行所有算法：

```bash
python check/check_gplearn.py
python check/check_pysr.py
python check/check_dso.py
```

其他算法使用同名脚本，例如 `check/check_llmsr.py`、`check/check_tpsr.py`。涉及大模型、外部服务或权重的算法，还需按对应工具文档配置 API key、网关或模型路径。

## 输出位置

直接使用 `SymbolicRegressor` 时，默认输出到：

```text
experiments/<problem>_<tool>_seed<seed>_<timestamp>/
```

使用标准数据集目录时，结果写入 `--output-root`；未指定时默认为：

```text
bench_results/sim_cli/
```

不同算法还可能在实验目录中保存日志、进度快照、候选方程或检查点。排查失败时，应先查看该任务的实验目录和单任务日志，不要只根据上层异常判断根因。

## 扩展新算法

推荐使用仓库内置的 `sr-tool-onboarder` 流程，而不是手工复制旧包装器。一次完整接入包含四层：

1. **包装层**：实现统一 API，并适配底层工具的输入、输出和序列化。
2. **注册层**：注册算法 ID、包装器类和 Conda 环境。
3. **验收层**：提供可离线运行的 `check/check_<tool>.py`。
4. **Manifest 层**：用机器可读文件记录接入方式、依赖和 smoke 参数。

### 1. 验证上游算法

先在独立环境中确认上游仓库能够完成最小训练和预测，再接入工具箱。若上游依赖与现有环境冲突，为新算法创建独立环境，不要直接污染 `sim` 主控环境。

外部源码可以作为子模块放在包装器目录下：

```bash
mkdir -p scientific_intelligent_modelling/algorithms/mytool_wrapper
git submodule add <上游仓库地址> \
  scientific_intelligent_modelling/algorithms/mytool_wrapper/mytool
```

如果上游包可稳定从 PyPI 安装，也可以仅在环境配置中声明依赖。

### 2. 创建 Manifest

复制示例：

```bash
cp tools/sr_onboarder/manifests/example_external_sr.json \
  tools/sr_onboarder/manifests/mytool.json
```

至少修改以下字段：

- `tool_name`：小写、稳定的算法 ID；
- `wrapper_class_name`：包装器类名；
- `integration_mode`：`python_api` 或 `cli_only`；
- `vendor_repo_relpath`：上游源码在仓库中的相对路径；
- `entrypoint`：上游 Python 模块和入口对象；
- `env`：环境名、Python 版本、依赖和安装命令；
- `adapter`：输入形状、预测能力、序列化方式和参数白名单；
- `smoke_test`：离线验收参数。

API key 只能在运行时通过环境变量注入，不能写入 manifest 或仓库。

### 3. 生成接入骨架

```bash
python3 tools/sr_onboarder/scripts/create_sr_tool.py \
  --manifest tools/sr_onboarder/manifests/mytool.json
```

生成器会创建或更新：

```text
scientific_intelligent_modelling/algorithms/mytool_wrapper/
├── __init__.py
└── wrapper.py
check/check_mytool.py
scientific_intelligent_modelling/config/toolbox_config.json
scientific_intelligent_modelling/config/envs_config.json
```

生成器不会猜测上游的真实训练入口，也不会自动解决依赖冲突；`wrapper.py` 中的 TODO 需要手工完成。

### 4. 实现包装器契约

包装器继承
[`BaseWrapper`](./scientific_intelligent_modelling/algorithms/base_wrapper.py)，
必须实现：

```python
from scientific_intelligent_modelling.algorithms.base_wrapper import BaseWrapper


class MyToolRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self.params.pop("n_features", None),
            feature_names=self.params.pop("feature_names", None),
            target_name=self.params.pop("target_name", None),
            context=self.__class__.__name__,
        )
        # 延迟导入上游工具，构造模型并训练。
        return self

    def predict(self, X):
        raise NotImplementedError

    def get_optimal_equation(self):
        raise NotImplementedError

    def get_total_equations(self):
        raise NotImplementedError
```

实现时还要处理：

- 框架元参数与上游模型参数的分离；
- `X.shape == (n_samples, n_features)` 的输入约定；
- 最优方程和候选方程的统一提取；
- 模型状态跨子进程序列化；
- 上游不支持 `predict` 或仅提供 CLI 时的明确降级行为；
- 标准 runner 注入的数据契约，避免元参数误传给第三方库。

如果底层模型不能可靠 pickle，应覆盖 `serialize()` 和 `deserialize()`，只保存恢复预测和方程所需的最小状态。

### 5. 注册与验收

确认两个配置文件中的名称完全一致：

- `toolbox_config.json`：`tool_name -> env + regressor`；
- `envs_config.json`：环境版本、依赖和安装后命令。

然后依次运行：

```bash
# 结构与导入检查
python3 tools/sr_onboarder/scripts/validate_sr_tool.py \
  --manifest tools/sr_onboarder/manifests/mytool.json

# 创建或准备好算法环境后，运行真实 smoke check
python3 tools/sr_onboarder/scripts/validate_sr_tool.py \
  --manifest tools/sr_onboarder/manifests/mytool.json \
  --runtime-check

# 直接执行算法自检
python check/check_mytool.py
```

最低验收标准：

- manifest 可解析，目录、配置和类名相互一致；
- 包装器模块能在目标环境中导入；
- `SymbolicRegressor("mytool")` 能实例化；
- 离线小数据能完成 `fit`；
- 最优方程非空；
- 若声明支持预测，`predict` 返回行数正确且数值有效；
- 新增测试通过后，再进行在线、GPU 或远程批量实验。

DSO 的历史接入过程可参考
[`docs/如何集成dso.md`](./docs/如何集成dso.md)，
当前统一接入契约和脚手架流程以
[`sr-tool-onboarder`](./.codex/skills/sr-tool-onboarder/SKILL.md)
为准。

## 常见问题

### 子模块目录为空

```bash
git submodule update --init --recursive -- \
  scientific_intelligent_modelling/algorithms \
  sim-datasets-py
```

### 找不到算法环境

在 `sim` 环境和仓库根目录中运行环境管理器，创建 `toolbox_config.json` 为该算法映射的环境。

### TPSR/E2ESR 缺少权重

按对应工具文档设置模型路径或执行环境配置中的下载步骤：

- [`docs/工具_tpsr.md`](./docs/工具_tpsr.md)
- [`docs/工具_e2esr.md`](./docs/工具_e2esr.md)

### LLM 算法调用失败

确认模型名、provider、API key 和网关地址都与运行配置一致。密钥应通过环境变量或本地配置注入，不要提交到 Git。

### PySR 首次运行很慢

首次启动可能需要准备 Julia 环境和编译依赖。重复运行前先确认当前任务仍在初始化，而不是直接中断。

### 标准数据集被识别为普通 CSV

确保传入的是目录本身，并至少存在 `metadata.yaml` 和 `train.csv`。完整评测还需要 `valid.csv`、`id_test.csv` 和 `ood_test.csv`。

## 进一步阅读

- [快速使用教程](./docs/使用教程.md)
- [工具参数目录](./docs/tool_parameters_catalog.md)
- [统一 benchmark 指标](./docs/benchmark_metrics.md)
- [DSO 使用说明](./docs/工具_dso.md)
- [DSO 历史接入记录](./docs/如何集成dso.md)
- [数据集 Python 包](./sim-datasets-py/README.md)

## 许可证

本项目采用 [GPL-3.0-or-later](./LICENSE) 许可证。
