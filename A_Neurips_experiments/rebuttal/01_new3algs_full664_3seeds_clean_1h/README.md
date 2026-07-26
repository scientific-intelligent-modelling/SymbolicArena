# Rebuttal：新增算法全量 clean 复现实验（664 × 3 seeds × 3算法）

目录名称约定：`01_new3algs_full664_3seeds_clean_1h`

- **数据集**：full-664（`full664_unified.csv`）
- **算法**：`fepysr`、`jaxsr`、`symbolfit`
- **种子**：`520, 521, 522`
- **噪声**：`clean`（`noise_sigma=0.0`）
- **预算**：`3600s`（1h）
- **最小运行时**：`3300s`
- **任务量**：`664 × 3 × 3 = 5976`

## 数据与稳定身份

数据集清单来自：

`exp-planning/02.E1选择验证/generated/probe4_full664_v1/full664_unified.csv`

清单必须恰好有 664 行，`global_index` 唯一并连续覆盖 `1..664`，
`dataset_dir` 与 `dataset_rel` 各自唯一。稳定数据集身份使用
`g0001..g0664` 和 `dataset_rel`，不能使用会重名的 `dataset_name` 或
`basename`。

## 参数来源

权威来源是 AAAI Stage 4 实际批次：

`benchmark-runs/formal3h/formal3h_13alg_ssr50_seed520-522_noise0-001-005_20260622-014658/params/`

`provenance/aaai_params_3h/` 保存来源参数副本，`params/` 保存正式 1h 参数，
`smoke/params/` 保存 600 秒 smoke 参数。正式参数相对来源只允许修改
`timeout_in_seconds: 10800 -> 3600`；smoke 只允许修改为 `600`。

`manifest/source_fingerprints.json` 使用
`path_base=repository_root` 的仓库相对路径；准备时的
`git_revision=70c4da7...`、数据源 SHA-256 和三份 AAAI 参数 SHA-256
保持不变。最终 `analysis_summary.json` 也使用相同路径基准，避免归档
绑定本地或远端 home。run-level 内的 `result_path` 仍保留为原始结果
来源指针，不参与正式 SYM-F 解引用。

`min_runtime_seconds=3300` 是合规审计阈值，写在 manifest 中，不作为第三方
算法参数透传。算法自己的搜索超参数保持 AAAI 实际配置。

## 目录说明

- `manifest/`：5976 条正式任务及预算、数据集、算法和来源指纹。
- `params/`：正式 1h 参数。
- `provenance/`：AAAI 3h 参数来源快照。
- `queues/full664_source.csv`：正式调度数据源和正式队列状态。
- `smoke/`：18 条 smoke 任务、参数、队列和审计产物。
- `deploy/`：同步、preflight、smoke、正式调度和收集审计脚本。
- `analysis/`：新三算法 run-level 汇总与 Stage 3 合并后的 7 算法榜单。
- `BATCH_NAME.txt`：冻结的正式 batch ID。
- `SOURCE_MAP.tsv`：关键材料来源映射。

## 执行门禁

```text
prepare + tests + dry-run
  -> remote preflight: 8/8 hosts pass
  -> smoke dispatch: 18 tasks
  -> smoke collect + harvest + audit: 18/18 pass
  -> full dispatch: 5976 tasks
  -> collect + harvest + audit + rerun until complete
  -> metric aggregation and Stage 3 four-probe comparison
```

本地入口：

```bash
bash A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/deploy/00_validate_and_sync_to_iaaccn22.sh
```

后续脚本必须从 `iaaccn22` 的仓库根目录运行，顺序是
`01_preflight_from_iaaccn22.sh`、`02_smoke_dispatch_from_iaaccn22.sh`、
`03_full_dispatch_from_iaaccn22.sh`。正式任务完成后再运行
`04_collect_audit_from_iaaccn22.sh`。该脚本审计通过后会继续调用
`05_generate_symf_from_iaaccn22.sh`。

正式队列运行期间可重复执行：

```bash
bash A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h/deploy/06_audit_completed_from_iaaccn22.sh
```

该脚本冻结一次 state 快照，按 assigned host 到 8 台机器逐条核对已经
标记为 `done` 的结果，要求运行时不少于 `3300s`，并检查状态、数据身份、
ID/OOD NMSE、canonical artifact 和方程。每次报告保存在
`monitoring/completed_audit/<timestamp>/`。

队列控制器按 `batch_name` 持有 `state/*.controller.lock` 单实例锁。
重复启动同一批次必须立即失败，不能同时使用 tmux 和 nohup 启动两个
控制器。`20260725-045816` 的双控制器事件、重复 session 清理清单、
加锁重启和首轮 `118/118` 结果审计证据保存在
`monitoring/controller_incident/20260725-045816/`。事件处理没有删除或
移动结果；harvest 仍按最终 state 的 `assigned_host` 选择规范结果。

`20260725-121632` 的只读巡检捕获到一次 state 覆盖写入期间的短暂
半截 JSON。调度器现改为同目录临时文件加 `fsync` 后原子替换正式
state，并已同步到 `iaaccn22`。为避免干扰当前正常 worker，本轮没有
重启控制器；修复会在下次必要恢复或自然启动时加载。现有
`06_audit_completed_from_iaaccn22.sh` 继续通过有效 JSON 重试冻结
审计快照。根因、测试、同步哈希和无任务影响证据保存在
`monitoring/state_atomicity/20260725-121632/`。

完成波峰期间，控制器已将远端残留进程回收和精确 session 确认改成
无残留立即返回、每台主机一次 SSH 批量确认。`20260725-072427` 的
受控重启把新调度器加载到 tmux 单控制器中，未停止任何算法 worker。
恢复后盘点为一个控制器、零重复任务；`20260725-073556` 固定快照审计
对 `551/551` 个 done 任务全部验证通过，`issue_count=0`。恢复与时延
证据分别保存在 `monitoring/controller_recovery/20260725-072427/` 和
`monitoring/completed_audit/20260725-073556/`。

`20260725-084449` 的第二次受控重启进一步加载了按主机批量回收完成
任务进程的实现，仍未停止任何算法 worker。8 台主机均重新同步成功，
恢复后连续完成三轮主机探测；事件窗口内 `26/26` 个完成任务回收成功，
同一主机单次最多批量回收 `4` 个任务。跨主机盘点为一个控制器、
`200` 个 session、`200` 个唯一任务和零重复任务；`08:55:06` 队列为
`741 done / 210 running / 5025 pending`，且没有 failed/error 任务。
重启、事件和盘点证据保存在
`monitoring/controller_recovery/20260725-084449/`。

随后审计器增加了正式结果的 clean 契约检查，要求
`train_label_noise.enabled=false`、`requested=false`、`sigma=0`、
`scale=0`。增强后的 `20260725-075212` 固定快照再次对 `551/551`
个 done 任务全部验证通过，8 台主机均通过且 `issue_count=0`；证据
保存在 `monitoring/completed_audit/20260725-075212/`。

审计器还会将每个 `result.json.params` 与 `params/<tool>__clean.json`
逐项对照。只排除已由 runner 消费的快照/噪声控制字段，以及
`exp_path`、特征名、目标名等数据集动态字段；其余静态算法参数必须
完全一致。`20260725-075703` 对当前 `551/551` 个 done 任务再次通过
该参数审计，证据保存在
`monitoring/completed_audit/20260725-075703/`。

结果身份还会直接比较调度 state 的 `tool`、`seed`、`task_index` 与
`result.json` 的 `tool`、`seed`、`task_global_index`，避免目录落位正确但
结果内容串任务。`20260725-080212` 对当前 `551/551` 个 done 任务通过
身份、预算、clean、参数和指标的联合审计，8 台主机均通过且
`issue_count=0`；证据保存在
`monitoring/completed_audit/20260725-080212/`。

`20260725-090921` 冻结了更新后的正式 state，并对 `763/763` 个 done
任务再次执行上述完整审计。8 台主机全部通过，验证结果数为 `763`，
`issue_count=0`；state 快照哈希和逐主机报告保存在
`monitoring/completed_audit/20260725-090921/`。

`20260725-094400` 在 `symbolfit_s520_clean_g0205` 成功恢复并进入
`done` 后再次冻结 state；`835/835` 个结果全部通过，8 台主机均通过且
`issue_count=0`。逐主机报告明确包含该任务在 `iaaccn23` 的规范
`result.json`，证据保存在
`monitoring/completed_audit/20260725-094400/`。

`20260725-101556` 在队列跨过一千条完成项后再次冻结正式 state。
`1007/1007` 个 assigned-host 规范结果通过运行时、预算、clean、参数、
身份、artifact 和指标联合审计；8 台主机全部通过，`issue_count=0`，
证据保存在 `monitoring/completed_audit/20260725-101556/`。

`20260725-110024` 在最终 SYM-F 分片链路同步并通过远端 smoke 后再次
冻结正式 state。`1134/1134` 个 assigned-host 规范结果通过完整联合
审计，8 台主机全部通过且 `issue_count=0`；运行时范围为
`3602.706--3795.947s`，证据保存在
`monitoring/completed_audit/20260725-110024/`。

`20260725-120714` 在完成数跨过下一审计阈值后重新冻结正式 state。
`1349/1349` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`c8c24dd142f2dbc5566fafb98a1e6ed2f70b9807acebd8d84b64f1d41424dacb`，
证据保存在 `monitoring/completed_audit/20260725-120714/`。

`20260725-131044` 在完成数达到 `1584` 后再次冻结正式 state。
`1584/1584` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`0a11933ee4e840cee4fe23e6292e2fcab4c02968a36b8dc2093a5e548bf6e13f`，
证据保存在 `monitoring/completed_audit/20260725-131044/`。

`20260725-140054` 在完成数达到 `1805` 后再次冻结正式 state。
`1805/1805` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`4ea5999887a7e21de6685c0f1769e0443181ce2abf2f4081f6deecc5ed12c873`，
证据保存在 `monitoring/completed_audit/20260725-140054/`。

`20260725-154218` 冻结了 seed 520 完整收口后的正式 state。该快照中
seed 520 的 `1992/1992` 个任务全部为 `done`，三个算法各 `664`
个；同时已完成的 9 个 seed 521 任务也纳入审计。总计 `2001/2001`
个 assigned-host 规范结果通过完整联合审计，8 台主机全部通过且
`issue_count=0`；运行时范围为 `3358.965--3795.947s`，state 快照
SHA-256 为
`a31c8be2d3bbbb1ad54df9f20472f49cd8e2e39c632127bd033577d2389f5b12`，
证据保存在 `monitoring/completed_audit/20260725-154218/`。

`20260725-165839` 在 seed 521 继续推进后再次冻结正式 state。
`2215/2215` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `223` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`902bce55e267004ac2f63b31decb0502c5b4f05863f850968585bd71afb4830e`，
证据保存在 `monitoring/completed_audit/20260725-165839/`。

`20260725-174238` 在下一完成波峰冻结正式 state。
`2459/2459` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `467` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`d07c8c4933878607eb62a03f350fe6beb421059c0138e31a235400aa16612f41`，
证据保存在 `monitoring/completed_audit/20260725-174238/`。

`20260725-185145` 在 seed 521 完成数继续增长后冻结正式 state。
`2659/2659` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `667` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`d82dae4135d8f0d4ef4a0ca95e3035c881d98ef36c2f9a0f4a92a6a40ac41ef5`，
证据保存在 `monitoring/completed_audit/20260725-185145/`。

`20260725-214117` 在 seed 521 继续推进后冻结正式 state。
`3226/3226` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `1234` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`98b3c8e1d216c8e01f4a6b4f87a4301911b1c830a46e3517874ef7aaa0662dd5`，
证据保存在 `monitoring/completed_audit/20260725-214117/`。

`20260725-224628` 在 seed 521 下一完成波峰冻结正式 state。
`3437/3437` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `1445` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`23ac10316d47695c9f45d6834f5e360b09cd7a2a5c1a56f0d8812ad1ae55fc45`，
证据保存在 `monitoring/completed_audit/20260725-224628/`。

`20260725-233120` 在 seed 521 进入后段后冻结正式 state。
`3657/3657` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `1665` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`b8947196c9d1058a7f4876fde306ce7fd43e866f0290436da09cbf8003fc1004`，
证据保存在 `monitoring/completed_audit/20260725-233120/`。

`20260726-004014` 在 seed 521 接近全部派发时冻结正式 state。
`3861/3861` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；快照中 seed 520 保持 `1992/1992`
完整，seed 521 已完成 `1869` 个、仅余 `3` 个 pending。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`82b3f349601b845a80fc61edda59955ba312a4ceac8b8b0bfbb0fb0c19925859`，
证据保存在 `monitoring/completed_audit/20260726-004014/`。

`20260726-014939` 冻结了 seed 521 完整收口后的正式 state。该快照中
seed 520 和 seed 521 均为 `1992/1992` 个 `done`，每个 seed 下三个
算法各 `664` 个；同时 seed 522 已完成 `15` 个。总计 `3999/3999`
个 assigned-host 规范结果通过完整联合审计，8 台主机全部通过且
`issue_count=0`；运行时范围为 `3358.965--3795.947s`，state 快照
SHA-256 为
`14e8733c22cc25650a305f8ca26372fe91720833771609714cbf5c8f4ae48704`，
证据保存在 `monitoring/completed_audit/20260726-014939/`。

`20260726-030006` 在 seed 522 首个完成波峰冻结正式 state。
`4204/4204` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `220` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`3e8b4d4663635863f3883627093e8d9c34c3652db62d08648c621f4a496bbc76`，
证据保存在 `monitoring/completed_audit/20260726-030006/`。

`20260726-034525` 在 seed 522 继续推进后冻结正式 state。
`4409/4409` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `425` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`22729b5ae0f3d295b4875c203303f9cebf5ad20dba71b6cd468707eb439e75a3`，
证据保存在 `monitoring/completed_audit/20260726-034525/`。

`20260726-044926` 在 seed 522 下一完成波峰冻结正式 state。
`4617/4617` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `633` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`51461a63666e6d9cad707c113f9dcb0608edddf4e1fd6311b5175247216fa6b0`，
证据保存在 `monitoring/completed_audit/20260726-044926/`。

`20260726-055341` 在 seed 522 继续推进后冻结正式 state。
`4822/4822` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `838` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`582d4f0e3e8dd09ac86dcae0c6c9fbea06fc38d781d71b2f9755010e5061d9e6`，
证据保存在 `monitoring/completed_audit/20260726-055341/`。

`20260726-065306` 在总完成数跨过五千后冻结正式 state。
`5033/5033` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `1049` 个。运行时范围为
`3358.965--3795.947s`，state 快照 SHA-256 为
`cac52cf3347e633581fe44736075d17380d21191f985d3e476577021d1b94767`，
证据保存在 `monitoring/completed_audit/20260726-065306/`。

`20260726-075738` 在 seed 522 下一完成波峰冻结正式 state。
`5249/5249` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `1265` 个。运行时范围为
`3358.965--3809.110s`，state 快照 SHA-256 为
`231f2155ae60f037b1c240dd5423b4a747a2b5e2eed4b470fa00ed379db5d763`，
证据保存在 `monitoring/completed_audit/20260726-075738/`。

`20260726-090231` 在 seed 522 进入后段后冻结正式 state。
`5461/5461` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `1477` 个。运行时范围为
`3358.965--3809.110s`，state 快照 SHA-256 为
`810ff6fd4d6b61f04e3ae70760d56f5942fa1391932795e8de409ccfbba5eb34`，
证据保存在 `monitoring/completed_audit/20260726-090231/`。

`20260726-095213` 在 seed 522 临近尾段时冻结正式 state。
`5680/5680` 个 assigned-host 规范结果通过完整联合审计，8 台主机
全部通过且 `issue_count=0`；seed 520 和 seed 521 均保持
`1992/1992` 完整，seed 522 已完成 `1696` 个。运行时范围为
`3358.965--3809.110s`，state 快照 SHA-256 为
`021b939e37fcaedc305fd09078c77a3579608fa723f85298a111df21ccef11bf`，
证据保存在 `monitoring/completed_audit/20260726-095213/`。

`20260726-120437` 冻结了正式队列自然收口后的最终 state。
`5976/5976` 个任务全部为 `done`，三个算法各 `1992` 个、三个 seed
各 `1992` 个，全部是 clean 条件；其中 `5975` 个任务一次完成，
`1` 个任务经过一次受控重试。8 台主机的 assigned-host 规范结果全部
通过运行时、预算、clean、参数、身份、artifact 和指标联合审计，
`validated_results=5976`、`issue_count=0`。运行时范围为
`3358.965--3809.110s`，state 快照 SHA-256 为
`ad86e073a31d09bfcd2a625e1e7b2f6e706a4c589bc57f9a094d791f6afb20b5`，
证据保存在 `monitoring/completed_audit/20260726-120437/`。

正式运行中的 60 秒快照另做了独立抽样：`20260725-074327` 对
`fepysr`、`jaxsr`、`symbolfit` 各抽两个任务，共 `6/6` 通过。抽样时
任务已运行约 21 分钟，三算法均持续写到 `minute_0021.json`，且快照
JSON 可解析并包含数据身份、方程、ID/OOD 评估字段。证据保存在
`monitoring/progress_snapshot_audit/20260725-074327/summary.json`。

对 `SymbolFit` 的一次真实结束异常复核表明，算法已经运行约 57 分钟
并持续写出可评估快照，但上游最终表达式数字格式化在处理复数时抛出
异常。runner 现在会在支持快照的算法异常退出后尝试恢复，但只有快照
能够重新生成 canonical artifact，并通过完整 split 的有限预测与指标
检查时才写为 `ok`；同时保留 `recovered_from_error`、
`raw_execution_error` 和 `recovered_after_error` 作为追溯字段。
不满足这些条件的异常仍按原路径报错并进入队列重试，不会用空壳快照
掩盖算法失败或指标质量。

由于 `symbolfit_s520_clean_g0205` 的第二次尝试早于上述 runner 修复部署，
该进程仍会使用旧异常分支。正式 dispatcher 因此显式设置
`--retry-limit 2`，允许它在第二次真实失败后启动加载新 runner 的第三次
尝试。该值只控制队列容错，不改变算法超参数、预算、随机种子、数据或
噪声口径。`20260725-091637` 受控重启后，实际控制器命令包含该参数，
首轮轮询正常推进且跨主机 session 盘点无重复；证据保存在
`monitoring/controller_recovery/20260725-091637/`。

该任务的第二次尝试最终没有再次触发格式化异常，而是在 `3604.961s`
耗尽预算后由原有 timeout 快照恢复路径成功收口：`status=ok`、
`recovered_from_timeout=true`、canonical artifact 有效，Valid、ID 和
OOD NMSE 均为有限值。控制器于 `09:39:14` 将任务记为 `done`，
`attempts=2`、`status_counts.ok=1`；第一次错误、分钟快照、第二次结果
及最终 queue state 的对照证据保存在
`monitoring/symbolfit_recovery/20260725-093448/`。

`20260725-094933` 进一步按 assigned host 扫描了修复部署时点之后启动
或完成的 `194/194` 条规范结果。所有结果均为 `status=ok`，没有非空
error、异常恢复、无效 artifact 或非有限 split NMSE；其中 `79` 条
`recovered_from_timeout` 均在 `3604.885--3631.254s` 正常收口并通过
上述检查。扫描口径、state 哈希和完整任务路径保存在
`monitoring/result_health/20260725-094933/`。

`20260725-095730_increment` 固定比较了 `09:49:33` 与 `09:57:30`
两个 queue state，对期间新增完成的 `44/44` 条 assigned-host 规范结果
做了增量健康检查。结果全部为 `status=ok`，canonical artifact、三个
split NMSE 和五项任务身份均通过；没有异常恢复或失败信号，其中 `18`
条为健康的 timeout 恢复。两个 state 的精确哈希、排除边界和逐主机
计数保存在 `monitoring/result_health/20260725-095730_increment/`。

`20260725-101108_increment` 继续比较 `09:57:30` 与 `10:11:08`
两个冻结 state，对新增完成的 `60/60` 条结果执行相同健康检查。三种
算法各 `20` 条，全部状态、artifact、三个 split 指标和任务身份通过，
没有异常恢复或失败信号，其中 `23` 条为健康的 timeout 恢复。证据
保存在 `monitoring/result_health/20260725-101108_increment/`。

`20260725-103600` 提前执行了最终收口链路预检，并修复四类确定性问题：
controller 缺失 Stage 3 输入、SYM-F 脚本未映射远端 home 数据根、
formal summary 错按非唯一显示名计数，以及初始同步清单未覆盖最终输入。
`04_collect...` 还增加了收集前 8 机严格 completed audit 门禁。修复后
远端真实预检得到 `664` 份可解析参数、`13944` 个唯一 7 算法运行键，
formal summary 占位检查为 `runs=13944`、`datasets=664`、
`algorithms=7`；完整证据保存在
`monitoring/finalization_preflight/20260725-103600/`。

`20260725-104501` 又对正式 SYM-F 收口做了本地与 `iaaccn22` 容量预检。
远端 13944 条经验投影均值约 `4593s`、p90 约 `11076s`；原实现没有
中间落盘，任意中断都要全量重算。正式链路因此改为 7 个算法分片、
逐片网格校验和原子提交，证据保存在
`monitoring/symf_capacity/20260725-104501/`。

## 指标汇总

`04_collect_audit_from_iaaccn22.sh` 在 `5976/5976` 收集并通过 audit gate
后，会自动运行：

```bash
python check/analyze_neurips_rebuttal_full664.py \
  --batch-dir A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h
```

分析器保留 manifest 的完整期望网格。ID/OOD NMSE 先执行
`log10(max(NMSE, 1e-12))` 并截断到 `[-12, 12]`，各自缺失值在算法
均值中按 `+12` 计入。最终排名按 penalized mean OOD log NMSE 升序。

运行期间仅可用 `--allow-incomplete` 生成诊断预览；预览会标记为
`INCOMPLETE_PREVIEW`，不得用于论文结论。正式模式会要求结果完整、
数据身份无冲突且 audit gate 通过，否则直接拒绝生成最终榜单。

formal symbolic judge 使用 `formula.py` 作为 ground truth，按稳定 `gid`
关联 664 个数据集，并使用 Stage 3 raw digest 中的 normalized/instantiated
表达式。唯一保留的参数审计告警是 `g0217/PO27`：历史公式参数名 `F0`
与数据列名 `x` 不一致；两边各只有一个未匹配变量，因此采用
`F0 -> x` 的唯一剩余位置映射，并在参数审计中保留该记录。

`05_generate_symf_from_iaaccn22.sh` 会对 7 算法共 `13944` 条 run 计算
SYM-F、Exact、TreeSim。计算按算法拆成 7 个独立的 `1992` 行分片，
每个分片通过完整 `algorithm × gid × seed` 网格校验后才原子落盘；
中断后只重算尚未完成的算法。正式启动时先冻结 params、run-level、
performance、generator、SYM-F merger 和 leaderboard merger；params 内
还嵌入每个 `formula.py` 的 base64 source 与 SHA-256，numeric judge 不再
读取运行中的外部数据树。每个分片的 provenance 必须绑定上述 source
fingerprint、生成器版本和冻结表达式策略；任一输入或代码变更都会
进入新的 `by_source/<snapshot_id>/`，不会覆盖或误复用旧结果。

合并器还会验证 seed 为整数、分数有限且位于 `[0,1]`、完整正式评分
公式、布尔关系和 `gid↔dataset` 映射。全部分片完成后再次要求 `13944`
个运行键唯一且完整，并用内容哈希绑定 7 个分片和所有派生汇总，再
原子提交正式目录并按 `Algorithm key` 合入榜单。formal summary 必须
包含 `13944` 条 run、`664` 个数据集、`7` 个算法和 `664` 份公式参数，
最终榜单算法集合必须精确等于
`dso/fepysr/imcts/jaxsr/pyoperon/symbolfit/udsr`。带 SYM-F 的权威榜单
保存在对应版本的 `by_source/<snapshot_id>/leaderboard/`；`analysis/`
中的同名文件仅是验证通过后原子更新的 latest 镜像。

## 最终产物

正式审计闭环后至少输出：

- 每算法 `Valid`、`Metric`、ID/OOD log NMSE。
- 运行时间和表达式复杂度。
- 可计算时输出 SYM-F、Exact 和 TreeSim。
- 与 Stage 3 的 `dso`、`imcts`、`pyoperon`、`udsr` 按相同
  full-664、seeds 520--522、clean、1h 口径合并成 7 算法比较表。
- `analysis/new3_run_level.csv` 与 `new3_algorithm_summary.csv`。
- `analysis/full664_7alg_run_level.csv` 与
  `full664_7alg_leaderboard.csv`。
- `symf/source_snapshots/<snapshot_id>/`。
- `symf/full664_7alg/by_source/<snapshot_id>/shards/<algorithm>/`。
- `symf/full664_7alg/by_source/<snapshot_id>/final/`。
- `symf/full664_7alg/by_source/<snapshot_id>/leaderboard/`。
- `analysis/full664_7alg_leaderboard_with_symf.csv`。

在 5976 条任务完成并通过审计之前，不得把该目录描述成最终结果。

## 最终归档完整性

最终归档的内容范围由 `check/build_neurips_rebuttal_archive.py` 固定。
它覆盖输入清单、参数与来源、部署材料、preflight、队列状态、完整
`runs/`、audit、analysis、SYM-F、运行期监控和 smoke 闭环。下列内容
不属于权威归档范围：

- `remote-experiments/`：controller 收集时的主机暂存副本，权威 harvest
  结果在 `runs/`。
- `runtime_queue/`：部署期复制和 preflight 请求缓存。
- `.lock`、`.pid`、`.tmp.*`、`__pycache__` 和 `.pyc`：运行期易变文件。
- 软链接或其它特殊文件：生成器直接拒绝，不跟随。

`MANIFEST.tsv` 以实验根为路径基准，按 UTF-8 路径字节序记录
`relative_path` 和 `size_bytes`。`CHECKSUMS.sha256` 使用 coreutils
标准格式，覆盖 `MANIFEST.tsv` 本身及 manifest 中的所有文件，但不
递归覆盖自身。归档必须在远端 finalization 成功、最终产物全部拉回
并且文档冻结后生成：

```bash
BATCH=A_Neurips_experiments/rebuttal/01_new3algs_full664_3seeds_clean_1h

python check/build_neurips_rebuttal_archive.py \
  --batch-dir "$BATCH" --write
python check/build_neurips_rebuttal_archive.py \
  --batch-dir "$BATCH" --verify --require-exact-scope
(cd "$BATCH" && sha256sum -c CHECKSUMS.sha256)
```
