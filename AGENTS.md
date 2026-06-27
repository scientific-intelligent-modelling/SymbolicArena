# 项目补充约定

## PySR / Julia 缓存

- 本机已准备好的 `juliapkg` 缓存目录：
  - `/home/family/pyjuliapkg_pysr`
```bash
PYTHON_JULIAPKG_PROJECT=/home/family/pyjuliapkg_pysr PYTHONPATH=. python /tmp/sim_runner_smoke/run_one.py pysr ...
```

## 远程服务器相关
远程服务器有 iaaccn22~29,这里是纯CPU机子
另外，还有iaaccn48~55，这里每台机子都有6~8卡的3090环境，如果没有必要，我会

## 远程执行与同步注意事项

### 1. 远程执行 Python / Shell 复杂逻辑时，优先发脚本，不要内联长命令

- 对远程机器执行多行 Python、包含引号、f-string、here-doc、列表/字典字面量时：
  - **优先做法**：先把脚本写到本地临时文件，再用 `scp` 发到远端 `/tmp/*.py`，最后执行：

```bash
scp /tmp/do_remote_check.py iaaccn22:/tmp/do_remote_check.py
ssh iaaccn22 'python /tmp/do_remote_check.py'
```

- 不要优先使用这类高风险形式：

```bash
ssh iaaccn22 'python - <<\"PY\" ... PY'
ssh iaaccn22 'python -c \"...很长的 Python 代码...\"'
```

- 原因：
  - 容易出现引号转义错误
  - 容易出现 here-doc 边界错误
  - 容易在嵌套 `ssh -> ssh -> python -c` 时被 shell 提前解释
  - 一旦出错，远端日志很难读

### 2. 多跳远程时，本地先上 `iaaccn22`，再由 `iaaccn22` 内网分发

- 当本地到 `iaaccn23~29` 的链路不稳定时：
  - **优先做法**：
    1. 本地先同步到 `iaaccn22`
    2. 再由 `iaaccn22` 通过内网 `10.10.100.23~29` 分发

- 典型模式：

```bash
rsync ... iaaccn22:/home/zhangziwen/...
ssh iaaccn22 'rsync ... 10.10.100.23:/home/zhangziwen/...'
```

- 不要在本地同时直接 fan-out 到 `iaaccn22~29`，链路不稳时很容易出现：
  - 部分机器成功
  - 部分机器卡死
  - 部分机器超时
  - 难以判断到底是同步失败还是代理链失败

### 3. 对远程连通性探测要加超时

- 所有远程探测、抽样校验、批量执行，默认加：
  - `timeout`
  - `ssh -o ConnectTimeout=...`
  - `ssh -o BatchMode=yes`

- 推荐形式：

```bash
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 iaaccn22 'hostname'
```

- 原因：
  - 避免卡住本地会话
  - 避免大量僵尸 `ssh -W` 代理进程残留

### 4. 批量同步数据文件时，不要误以为 Git 会带过去

- `sim-datasets-data/` 下的数据文件通常**不在主仓库 Git 跟踪内**。
- 因此：
  - 本地修复了 CSV
  - **不等于**远端仓库自动有这些修复

- 对数据文件修复必须显式同步：
  - 用 `rsync/scp`
  - 同步到远端真实数据目录

### 5. 远端真实数据目录与仓库目录不是同一个位置

- 远端仓库根目录：
  - `/home/zhangziwen/workplace/scientific-intelligent-modelling`

- 远端真实数据目录：
  - `/home/zhangziwen/sim-datasets-data`

- 不要默认远端数据就在：
  - `repo/sim-datasets-data/...`

- 真实运行、白名单核验、launcher 读取时，优先按：
  - `/home/zhangziwen/sim-datasets-data/...`
 处理。

### 6. 对 `datasets_to_run.csv` 做远端校验时，要做路径映射

- `datasets_to_run.csv` 里的 `dataset_dir` 可能是：
  - 相对路径，例如 `sim-datasets-data/...`
  - 本地路径风格

- 在远端做全量校验时，必须先映射到：
  - `/home/zhangziwen/sim-datasets-data/...`

- 否则会出现假阴性：
  - 文件其实存在
  - 但脚本误报“数据集目录不存在”

### 7. 远端 launcher 必须在正确环境里启动

- `launch_pysr_probe.py` 必须从 `sim_base` 启动
- `launch_llmsr_probe.py` 必须从 `sim_llm` 启动

- 原因：
  - launcher 内部子任务会继承 `sys.executable`
  - 如果启动环境错了，后续所有 worker 都会错环境

### 8. 遇到 Git LFS pointer，先判断是数据没落下来，不要直接误判为列名错误

- 若 CSV 首行像这样：

```text
version https://git-lfs.github.com/spec/v1
```

- 这说明文件还是 LFS pointer，不是真实 CSV。
- 这时不能直接下结论说：
  - `metadata.target.name` 和 CSV 列名不一致

- 正确顺序：
  1. 先确认是不是 LFS pointer
  2. 如果是，先补真实数据
  3. 再做目标列/表头一致性判断

### 9. 当前这批双探针数据修复的真实规则

- 不要误记为“改了 metadata”。
- 实际修复规则是：
  1. 批量修了 `343` 个数据集的 `ood_test.csv` 表头
     - 把最后一列 `target`
     - 改成 `metadata.yaml` 里的真实 `target.name`
  2. 通过 `ModelScope + git-lfs` 补齐了最后 `3` 个 `srbench1.0/feynman` 数据集的真实 CSV

- 后续如需在新机器复刻修复，按这两个规则操作，不要直接批量改 metadata。

### 10. 用 `tmux` 起远程任务时，优先直接执行脚本文件，不要把长命令内联给 `tmux`

- **优先做法**：

```bash
tmux new-session -d -s smoke_xxx /bin/bash /tmp/run_xxx.sh
```

- 不要优先使用：

```bash
tmux new-session -d -s smoke_xxx "bash -lc '很长的一串命令 ...'"
```

- 原因：
  - 多层 shell 时，命令字符串非常容易被提前吃掉
  - 表面上 `tmux` 会话已经创建，但 pane 里只剩一个空 `bash`
  - 这类错误很隐蔽，往往没有日志、没有报告文件，容易误判为算法失败

### 11. 通过 `/tmp/*.py` 调仓库代码时，要显式设置 `PYTHONPATH=.`

- 如果远程临时脚本在：
  - `/tmp/run_probe_smoke_one.py`
- 但脚本内部要 `import scientific_intelligent_modelling...`
- 必须在仓库根目录下执行，并显式带：

```bash
cd /home/zhangziwen/workplace/scientific-intelligent-modelling
PYTHONPATH=. conda run -n sim_base python /tmp/run_probe_smoke_one.py ...
```

- 否则容易出现：

```text
ModuleNotFoundError: No module named 'scientific_intelligent_modelling...'
```

- 不要假设：
  - `cd repo` 之后 Python 会自动把当前仓库加入模块搜索路径

### 12. 临时 smoke / one-shot 任务不要依赖 launcher 内部的路径兜底，优先把 `dataset_dir` 写成远端绝对路径

- launcher 内部已经做过相对路径到远端真实数据目录的映射。
- 但临时 one-shot 脚本、独立 smoke 脚本、调试脚本未必复用了那层逻辑。

- 因此在远端 smoke CSV 里，优先直接写：

```text
/home/zhangziwen/sim-datasets-data/...
```

- 不要依赖：

```text
sim-datasets-data/...
```

- 原因：
  - 临时脚本常常比正式 launcher 少一层路径映射
  - 很容易出现 `FileNotFoundError`
  - 这种错误会伪装成“算法没跑起来”

### 13. 从 `iaaccn22` 再去访问 `23~29` 时，优先用内网 IP，不要继续用主机别名

- 在本地可以用：
  - `iaaccn22`
  - `iaaccn23`
  - ...

- 但在 `iaaccn22` 内部再次访问其它机器时，优先用：
  - `10.10.100.23`
  - `10.10.100.24`
  - ...

- 不要优先继续用：
  - `iaaccn23`
  - `iaaccn24`
  - ...

- 原因：
  - 远端机器自己的 `~/.ssh/config` 可能带跳板、代理或私钥路径
  - 别名解析后可能再次走外层 hub
  - 容易出现：
    - 缺失私钥
    - 权限拒绝
    - banner 超时
    - 明明内网可达却连不上

### 14. 远端批量重试脚本不要直接 `set -e` 串到底

- 对 8 台机器批量起任务、同步脚本、同步切片时：
  - 单台机器临时连不上是常见情况

- 因此批量重试脚本里：
  - 单机失败要记录并继续
  - 不要让第一台失败直接中断整轮分发

- 推荐形式：

```bash
ssh host '...' || { echo PREP_FAIL host; continue; }
rsync ... || { echo SYNC_FAIL host; continue; }
tmux new-session ... && echo STARTED host || echo START_FAIL host
```

- 适用场景：
  - 网络抖动
  - 某几台机器偶发超时
  - 需要 22 上本地控制器持续重试把剩余机器逐步拉起来

### 15. `task_status.jsonl` 里的 `FileNotFoundError` 不一定是数据路径错了，要先看单任务日志

- 在 `pysr` / `subprocess_runner` 这条链路里，如果底层子进程异常退出、没有成功写出 `.result` 文件：
  - 上层很可能只看到：

```text
FileNotFoundError(2, 'No such file or directory')
```

- 这时不要立刻下结论说：
  - `dataset_dir` 路径错了
  - 远端缺少数据集目录

- 正确顺序：
  1. 先看 `__launcher__/logs/*.log`
  2. 再判断是：
     - 真正的路径缺失
     - 还是底层运行时崩了但没写 `.result`

- 这次真实案例里：
  - `task_status` 表面是 `FileNotFoundError`
  - 但单任务日志里真正根因是：

```text
ERROR: ArgumentError: Package Pkg not found in current path.
```

### 16. 如果远端 `pyjuliapkg_pysr` 连 `import Pkg` 都失败，不要只修 `Project.toml`，直接整目录替换

- 诊断命令：

```bash
/home/zhangziwen/pyjuliapkg_pysr/pyjuliapkg/install/bin/julia --startup-file=no -e 'import Pkg; println(\"pkg_ok\")'
```

- 如果这条都失败，说明问题已经不是：
  - `PySR` 参数
  - `datasets_to_run.csv`
  - `Project.toml` 某一项依赖

- 而是：
  - 远端 Julia 安装/stdlib/共享缓存本身坏了

- 最稳修法：
  1. 停掉正在跑的 `pysr` 任务
  2. 备份旧的 `~/pyjuliapkg_pysr`
  3. 用一份已验证可用的完整缓存整目录覆盖
  4. 再逐台执行一次：

```bash
PYTHON_JULIAPKG_PROJECT=/home/zhangziwen/pyjuliapkg_pysr \
PYTHONPATH=. conda run -n sim_base python -c 'from pysr import PySRRegressor; print("pysr_import_ok")'
```

  5. 预热成功后再重启全量 probe

- 不要优先做：
  - 只改 `Project.toml`
  - 只删 `lock.pid`
  - 只重试全量任务

- 因为这类坏缓存会让整片任务系统性报错。

### 17. 如果远端仓库内存在 `repo/sim-datasets-data` 残缺镜像，launcher 的 `dataset_dir` 解析必须优先命中 `~/sim-datasets-data`

- 当 `dataset_dir` 形如：

```text
sim-datasets-data/...
```

- 远端通常同时存在两份候选路径：
  1. `repo/sim-datasets-data/...`
  2. `~/sim-datasets-data/...`

- 如果仓库内那份只是之前同步留下的**局部镜像/残缺目录**，而 launcher 又优先选了它，
  就会出现：
  - 目录本身存在
  - 但 `metadata.yaml` / 某些 split 文件不存在
  - 上层表现成大量 `FileNotFoundError`

- 正确规则：
  - 只要 `dataset_dir` 以 `sim-datasets-data/` 开头，
  - **优先尝试 `Path.home() / dataset_dir`**
  - 再尝试 `Path.cwd() / dataset_dir`

- 不要默认：
  - `cwd/path` 在远端永远比 `home/path` 更可信

- 这次真实案例里：
  - `task_status` 大面积出现 `FileNotFoundError`
  - 表面像是数据集路径全错
  - 实际是 launcher 命中了仓库内不完整的 `repo/sim-datasets-data/...`

### 18. `pysr` 超时后若只写出 `timed_out` 壳结果，先检查恢复链，不要先下结论为“`hall_of_fame` 没了”

- 典型现象：
  - `task_status.jsonl` / `*.report.json` 全是 `timed_out`
  - `result.json` 里只有：
    - `status = timed_out`
    - `equation = null`
    - `valid / id_test / ood_test = null`

- 正确排查顺序：
  1. 先进实验目录检查：
     - `hall_of_fame.csv`
     - `hall_of_fame.csv.bak`
     - `checkpoint.pkl`
  2. 再判断是：
     - 文件真的缺失
     - 还是“超时当下自动恢复失败，但事后仍可恢复”

- 这次真实案例里：
  - `hall_of_fame.csv` / `.bak` / `checkpoint.pkl` 都在
  - 手动用 `PySRRegressor(existing_exp_dir=...)` 仍然能恢复出方程
  - 真正的问题不是“文件丢了”，而是“自动恢复时机过早，没把现成 best result 补写进 `result.json`”

- 修法优先级：
  1. 优先从 `hall_of_fame.csv`，兜底 `hall_of_fame.csv.bak` 反向恢复 best equation
  2. 再补写：
     - `canonical_artifact`
     - `valid / id_test / ood_test`
  3. 不要先把这类结果判成“彻底不可恢复”

### 19. 一整轮 probe 突然“都不在跑了”，先区分是异常停掉，还是整片任务正常超时收口

- 典型误判：
  - 看到 `tmux` 没了、`run-task` 也没了，就以为整轮崩了

- 正确排查顺序：
  1. 先看 `__launcher__/task_status.jsonl`
  2. 再统计：
     - `timed_out`
     - `error`
     - `ok`
  3. 最后再看 `controller.log`

- 这次真实案例里：
  - `iaaccn22~25` 上的 `pysr` 任务全都停了
  - 真实原因不是 controller 崩溃
  - 而是 `664 / 664` 全部正常跑到 `timeout_in_seconds=3600` 后收口退出

- 结论：
  - “整轮不在跑了” 不等于 “异常中断”
  - 先看状态分布，再判断是否真是事故

### 20. `pysr` 超时恢复后如果“有方程但指标还是空”，优先沿 `hall_of_fame` 找第一条数值稳定候选

- 典型现象：
  - `result.json` 里：
    - `equation` 已经有了
    - `canonical_artifact` 也有了
    - 但 `valid / id_test / ood_test` 仍然是 `null`

- 常见原因：
  - 当前恢复出的 loss 最优公式在某些 split 上数值发散，例如：
    - `log(0)`
    - 除零
    - `nan / inf`

- 正确修法：
  1. 按 `Loss` 升序遍历 `hall_of_fame.csv`
  2. 若不存在则遍历 `hall_of_fame.csv.bak`
  3. 对每条候选尝试：
     - 构造 `canonical_artifact`
     - 在 `valid / id_test / ood_test` 上做轻量预测
  4. 选第一条能给出有限预测值的候选作为稳定恢复结果

- 不要默认：
  - “loss 最低那条一定最适合回填指标”

- 这次真实抽样里：
  - 多个“有方程但没指标”的数据集
  - 都能在 `hall_of_fame` 的更低排名候选里找到稳定可评估的公式

### 21. 正式批次巡检前，先确认 `--output-root` 的真实目录布局，不要默认沿用上一轮 probe 的路径模式

- 典型误判：
  - 看到某台机器的 `task_status.jsonl` 没找到，就以为该机没跑起来

- 这次真实案例里：
  - probe 批次常见布局是类似：

```text
.../<host>/<tool>/...
```

  - 但正式 `formal200_v1_seed520_521_20260417-051058` 这轮，启动脚本传的是：

```text
--output-root .../pysr/seed520/iaaccn25
--output-root .../llmsr/seed521/iaaccn29
```

  - 所以实际目录布局变成了：

```text
.../<tool>/<seed>/<host>/__launcher__/task_status.jsonl
```

- 正确排查顺序：
  1. 先看远端 `/tmp/*_<tool>_seed*.sh` 或 controller 启动脚本里的 `--output-root`
  2. 再按真实目录模式去找：
     - `controller.log`
     - `__launcher__/task_status.jsonl`
     - `result.json`
  3. 不要先凭经验假设：
     - `.../<host>/<tool>/...`
     - 或 `.../<tool>/<host>/...`

- 结论：
  - **巡检统计必须以本轮实际 `--output-root` 为准**
  - 不同批次的目录布局可能不同，不能把上一轮的路径模式直接套到下一轮

### 21.1. 阅读“最近实验”时，先确认用户关注的实验族，不要只按全仓 mtime 下结论

- 当用户说“最近两次实验”但上下文指向特定算法或实验族时：
  - 先列出候选口径，例如：
    - 全仓 `benchmark-runs` 最新目录
    - 当前 `latest` 软链
    - 指定算法族目录，例如 `drsr3h/`、`llmsr3h/`、`formal3h/`
  - 再按用户语境选择主口径。

- 不要只因为 `formal3h` 目录 mtime 更新，就漏掉用户实际关注的：
  - `drsr3h`
  - `llmsr3h`
  - 其它指定算法族批次

- 汇报时必须明确：
  - “最近”的判定口径
  - 被纳入与被排除的候选批次
  - 当前结论是否仅基于本地文件，还是已经做过远端核验

### 22. 对固定预训练词表的算法，不要直接按当前数据集维度去缩小环境词表，优先在集成层做非法变量过滤或投影

- 典型场景：
  - `TPSR` / `E2ESR` 这类带预训练解码器的算法
  - 模型训练时绑定了固定变量词表，例如默认 `x_0 ~ x_9`

- 错误做法：
  - 看到当前数据集只有 `4` 维，就直接把：

```text
max_input_dimension = 4
```

  - 强行注入给环境 / 词表

- 风险：
  - 会破坏预训练模型和环境词表的一致性
  - 可能直接触发 decoder 断言失败或生成异常

- 正确做法：
  1. 保留模型原始固定词表
  2. 在 wrapper 中记录当前任务真实特征维度
  3. 在中间候选写出链和最终 best 候选链里：
     - 对超出当前特征维度的变量 token 做过滤
     - 或按 wrapper 既有预测语义投影为 `0`
  4. 再交给 benchmark runner / normalizer 做严格校验

- 结论：
  - 这类问题优先判定为**工具集集成约束没接好**
  - 不要第一反应去改预训练模型的环境词表大小

### 22. `DSO` 的分钟级快照不要只依赖 `*_hof.csv`，训练中应优先读 `.dso_current_best.json`

- `DSO` 上游默认会在训练结束时一次性写：
  - `*_hof.csv`
  - `*_pf.csv`
- 因此如果 benchmark runner 的周期快照只读 `*_hof.csv`：
  - 最终结果会有
  - 但训练中 `minute_0001.json` 往往拿不到候选

- 正确做法：
  1. 在 `dso_wrapper` 训练循环里，每轮 `train_one_step()` 后把当前 `trainer.p_r_best` 写成：

```text
.dso_current_best.json
```

  2. `runner._extract_dso_periodic_candidate(...)` 优先读这个状态文件
  3. 只有在状态文件不存在时，才退回去读 `*_hof.csv`

- 额外注意：
  - `DSO` 若要保留日志文件，不能直接复用上游 `DeepSymbolicRegressor.fit()`，因为它会把 `experiment.logdir=None`
  - benchmark 场景应走 `DeepSymbolicOptimizer.train_one_step()` 循环，并显式保留 `logdir`

### 22. 验证 `drsr` 落盘时，要把“分钟级中间快照”和“结束结果落盘”分成两个检查点

- 这次在 `iaaccn22` 上的 `drsr` smoke 证明：
  - 传入 `progress_snapshot_interval_seconds=60` 后，
  - 外层 `progress/minute_0001.json` 与实验目录内 `progress/minute_0001.json`
  - 可以正常按分钟落盘
  - 同时 `drsr` 自身也会持续更新：
    - `progress.json`
    - `best_history/`

- 但同一条 smoke 里，`timeout_in_seconds=190` 并**不保证**任务会在约 190 秒时结束并写出 `result.json`。
  - 真实现象是：
    - 分钟级快照持续写出
    - `progress.json` 继续更新
    - 但超过设定 wall time 后，进程仍可能继续运行，外层 `result.json` 未及时出现

- 因此对 `drsr` 的远程健康检查，至少要拆成两条：
  1. **长一点的 smoke**：验证 `minute_0001.json / progress.json / best_history` 是否正常出现
  2. **小预算自然结束 smoke**：单独验证任务能否自然结束并写出最终 `result.json`

- 不要因为“1 分钟快照存在”就默认：
  - `timeout_in_seconds` 一定已经被正确执行
  - 或 `result.json` 一定会按时落盘

### 23. runner 必须显式注入数据契约，wrapper 必须在 `fit()` 入口校验契约与真实输入一致

- 对所有 benchmark 任务：
  - `runner.build_runner_params()` 必须统一显式注入：
    - `n_features`
    - `feature_names`
    - `target_name`

- 不要只依赖：
  - wrapper 从 `X.shape[1]` 隐式猜维度

- 对所有 wrapper：
  - 若接收到上述字段，必须在 `fit(X, y)` 入口先校验：
    - `n_features == X.shape[1]`
    - `len(feature_names) == X.shape[1]`
    - `target_name` 为非空字符串

- 对严格白名单或会透传到底层库的 wrapper：
  - 必须先显式吸收这些元参数
  - 不要让它们作为未知参数透传给第三方库


## commit 规则
在commit的时候加上 Co-Authored-By: lilmortyj <781113402@qq.com>
