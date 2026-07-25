# 最终收口链路预检

## 目的

在正式队列完成前，只读验证 collect、harvest、Stage 3 合并和 SYM-F
链路，并修复会导致 `5976/5976` 后才失败的确定性阻断。没有运行正式
`04_collect_audit_from_iaaccn22.sh` 或
`05_generate_symf_from_iaaccn22.sh`，也没有生成可用于论文的预览指标。

## 发现与修复

1. `iaaccn22` 缺少 Stage 3 的 main/raw 两份 `7968` 行输入。两文件已
   定点同步，且本地、远端 SHA256 完全一致。
2. SYM-F 参数准备和 numeric probe 原先把相对 `sim-datasets-data/...`
   固定解析到仓库目录；远端真实数据位于
   `/home/zhangziwen/sim-datasets-data`。两个脚本现均优先解析 home
   数据根，并保留仓库路径作为回退。
3. formal summary 原先按非唯一的 `dataset` 显示名计数。Full664 的
   `664` 个 gid 只有 `549` 个唯一显示名，导致 `.datasets == 664`
   必然失败。现改为按稳定 `gid` 计数。
4. 初始同步脚本未包含 Stage 3 输入和最终分析脚本。现已增加仅同步到
   controller 的收口输入，避免向计算节点无谓分发大文件。
5. `04_collect...` 现在会在远端收集前先执行 8 机严格 completed audit，
   以运行时、参数、dataset_rel、artifact 和三个 split 的有限指标作为
   额外发布门禁。

## 数据一致性

对本地和 `iaaccn22` 的 `664` 份 `formula.py` 与 `metadata.yaml`
执行了换行归一化内容指纹对账。首次仅发现
`g0217/PO27/formula.py` 不一致：

- 远端旧文件 SHA256：
  `b1ec75bf978922a9e0297c4974049da835a08a00eb790bfb7bb5390ec392ae18`
- 修复后 SHA256：
  `70e1161efccfb35fc01010fcfa8c9e9e74672309f833b903aa29b4bf2247aa53`

旧文件把 `alpha`、`beta` 错当成额外函数参数；本地修正版将其恢复为
公式内部常数。只定点修复了 `iaaccn22` 上这一份文件。修复后
`664/664` 份 formula 和 metadata 的归一化哈希均与本地一致，
差异数为 `0`。

## 验证

- 相关回归测试：`39 passed`
- shell 语法检查：通过
- Stage 3 main：`7968` 行、`7968` 唯一键
- Stage 3 raw digest：`7968` 行、`7968` 唯一键
- 远端参数生成：`664` 数据集、`664` 公式可解析、`1` 个既有
  `F0 -> x` 人工确认项
- 远端 home 数据路径：`664/664` 目录和 `formula.py` 可解析
- 实际 numeric probe 抽样：`6/6` 无错误
- 7 算法预期网格：`13944` 行、`13944` 唯一
  `algorithm/gid/seed` 键
- formal summary 实际 Full664 占位写出检查：
  `runs=13944`、`datasets=664`、`algorithms=7`
- 最终脚本和 Stage 3 输入的本地、`iaaccn22` 内容哈希一致

代码修复提交为 `7edd220`、`8509b40` 和 `496eaa6`。

## 边界

本预检证明最终输入、路径解析、稳定身份计数和门禁链路可执行，不证明
新三算法结果已经完整，也不包含正式 SYM-F 数值。正式收口仍必须等待
queue 达到 `done=5976`、`running=0`、`pending=0`，随后由
`04_collect...` 触发严格审计、收集、harvest、标准 audit、分析和
`05_generate_symf...`。
