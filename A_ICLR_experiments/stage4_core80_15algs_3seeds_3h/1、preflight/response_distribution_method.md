# 四probe响应分布选集：定义、计算与复算

本说明对应select_response_distribution.py和response_distribution_core50/中的实际实现与结果。公式显示值可能经过舍入，程序使用CSV和JSON中的完整数值精度。本次补充说明不改变选择算法、参数或已保存名单。

## 1. 问题、输入和符号

任务是从固定的Full664中选出K个不同数据集，使这K个任务在四个probe上的响应分布接近Full664。当前已执行K=50，每次调用只选择一个规模。

### 1.1 符号表

| 符号 | 定义 |
| --- | --- |
| D、N | Full664任务集合及其数量，N=664 |
| i | 数据集身份，以dataset_id及dataset_rel绑定 |
| P、p | probe集合及其中一个算法；顺序固定为DSO、iMCTS、PyOperon、uDSR，数量为4 |
| T、s | 训练seed集合及其中一个seed，T={520,521,522}，数量为3 |
| b | 测试划分，取ID或OOD |
| S、K | 被选任务集合及其数量，K=50为当前试算 |
| V_{i,p} | 数据集i、probe p的有效seed集合 |
| v_{i,p} | 有效seed比例，等于有效seed数除以3 |
| ell、m | 单seed的截断log NMSE、跨有效seed的算术平均 |
| r_{i,p} | 数据集i对probe p的response，取值[-12,12]，越大表示表现越好 |
| mu_p、sigma_p | probe p在完整664任务上的response均值、总体标准差 |
| z_{i,p} | 标准化response |
| F_{p,S}、F_{p,D} | probe p在S和D上的标准化response经验分布 |
| W_p(S) | 上述两个一维分布之间的Wasserstein-1距离 |
| D_resp(S) | 四个W_p的等权平均，当前唯一优化目标，越小越好 |
| x_i | 二元选择变量；选中i时为1，否则为0 |
| 1[条件] | 指示函数，条件成立取1，否则取0 |

四个probe都参与当前正式候选的构造。训练seed用于重复运行算法，与选择器随机初始化无关。

### 1.2 三份输入表

文件位于[Stage3目录](../../stage3_664dats_4probes_3seeds_1h/)。

| 文件 | 行单位 | 本次用途 |
| --- | --- | --- |
| probe4_current_run_level_raw_digest_7968.csv | dataset×probe×seed，共7968行 | 从原始NMSE和有效输出标记重算r |
| probe4_postprocess_dataset_level.csv | dataset，共664行 | family、subgroup、重复组、资格与诊断字段 |
| probe4_postprocess_dataset_algorithm.csv | dataset×probe，共2656行 | 完整性检查及历史诊断的中位数、IQR来源 |

每个dataset×probe必须有seed 520、521、522三条记录。元信息要求每个数据集有12条已完成记录，completion_rate=1；跨seed表的n_expected_seeds、n_observed_rows、n_finished_seeds均为3。输入缺失或键不一致时计算停止。

原始run表会与预置SHA256校验；其余表记录实际SHA256并验证结构。当前版本：

| 输入 | SHA256 |
| --- | --- |
| 原始run表 | 1a588501a0b31b5b392592b05c3105489aee18ab19e3cffb190102e6bee5ddcb |
| dataset元信息 | f88a594cd678ef7497f91301675a7c7f3e8169f14e6fd7dc9dbcf640821f7a13 |
| dataset×probe汇总 | 7a00dc7461e3a129f2973108b5f4ed1f1e2dce282f173d7be479689ec1fb3aa5 |

本次使用历史记录标记和数值。historical_native_replay_verified=false表示尚未重新加载历史原生模型来验证这些预测和指标。

## 2. 从一次运行得到log NMSE

### 2.1 ID、OOD和NMSE字段

ID指任务的分布内测试集，OOD指任务提供的分布外测试集。选择器读取：

- result_id_test_nmse：ID测试集的历史NMSE。
- result_ood_test_nmse：OOD测试集的历史NMSE。
- result_valid_output：历史有效输出标记，1或0。
- state：运行完成状态，本轮要求为done或failed。
- result_dataset_identity_match：结果与预期数据集身份是否一致，本轮要求为1。

框架通用[regression_metrics](../../../scientific_intelligent_modelling/benchmarks/metrics.py)的有限数值NMSE定义为：

$$
MSE=\frac1n\sum_{j=1}^{n}(\hat y_j-y_j)^2,\qquad
NMSE=\frac{MSE}{\frac1n\sum_{j=1}^{n}y_j^2}.
$$

这里y_j为真实目标，hat y_j为模型预测，n为该测试集样本数量。框架实现还包含非有限残差和数值上限处理，精确行为见该函数。

当前选集直接读取上述历史NMSE列，不运行预测，也不重新计算NMSE。完整重现选择结果以这些固定输入值为准；历史运行是否全部符合当前指标实现，属于原生模型回放核验范围。文档中的NMSE说明不替代这一审核。

### 2.2 有效seed集合

$$
V_{i,p}=\{s\in T:\ result\_valid\_output_{i,p,s}=1\}.
$$

对V中的每条记录，程序要求ID、OOD两项NMSE都有限且非负。若标记为1而指标不满足要求，程序报错，不自动减少V。标记为0的运行不参加均值，即使它保留了一部分指标。

$$
n^{valid}_{i,p}=|V_{i,p}|,\qquad v_{i,p}=n^{valid}_{i,p}/3.
$$

分母固定为3。历史运行的终止原因不单独决定有效性；result_valid_output及指标一致性是当前读取依据。

### 2.3 每个seed单独变换

定义截断：

$$
\operatorname{clip}(x,a,b)=\min(b,\max(a,x)).
$$

对每个有效seed及每个split：

$$
\ell^b_{i,p,s}=
\operatorname{clip}\left[
\log_{10}\left(\max(NMSE^b_{i,p,s},10^{-12})\right),
-12,12
\right].
$$

计算顺序为：NMSE下界处理、取以10为底的对数、截断对数。

- NMSE≤10^-12时，ell=-12；NMSE=0也适用。
- NMSE=1时，ell=0。
- NMSE≥10^12且仍为有限值时，ell=12。
- 负数、NaN和无穷值不通过有效输入检查。

NMSE和ell越小表示预测误差越小。对数用于控制数量级差异，截断用于限制极端有限数值的影响。

## 3. 从三个seed得到任务response

### 3.1 有效seed的算术平均

当n_valid>0时：

$$
m^b_{i,p}=\frac1{n^{valid}_{i,p}}
\sum_{s\in V_{i,p}}\ell^b_{i,p,s},
\qquad b\in\{ID,OOD\}.
$$

ID和OOD使用同一个V。每个有效seed权重相同；先完成逐seed的log与截断，再计算均值。

### 3.2 合并两个split与失败惩罚

$$
a_{i,p}=\frac{m^{ID}_{i,p}+m^{OOD}_{i,p}}2,\qquad
penalty_{i,p}=2(1-v_{i,p}).
$$

a为ID/OOD等权平均误差。penalty分别为：

| 有效seed数 | v | penalty |
| ---: | ---: | ---: |
| 3 | 1 | 0 |
| 2 | 2/3 | 2/3 |
| 1 | 1/3 | 4/3 |

### 3.3 response及下界

$$
r_{i,p}=
\begin{cases}
-12, & n^{valid}_{i,p}=0,\\
\max(-12,-a_{i,p}-penalty_{i,p}), & n^{valid}_{i,p}>0.
\end{cases}
$$

负号将方向变为越大越好。下界保证部分有效运行受到惩罚后，r仍不小于-12；上界12由ell的下界和非负penalty共同保证。三个历史有效输出标记均为0时直接赋值-12，不计算空集合均值。

此处“全失败”按照历史有效输出标记判定，未包含本次新增的原生模型回放验证。原始记录的收集错误或评估错误需要另外核验。

### 3.4 真实例子：Keijzer-11、DSO

当前名单中的g0620，原始数据如下：

| seed | 有效标记 | ID NMSE | OOD NMSE | ID ell | OOD ell |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 520 | 1 | 0.1048399893 | 0.0283322854 | -0.9794730322 | -1.5477183917 |
| 521 | 1 | 0.0570894009 | 0.0202328259 | -1.2434445142 | -1.6939434564 |
| 522 | 1 | 0.0509453290 | 0.0166231944 | -1.2928956289 | -1.7792855159 |

三条记录都有效，得到：

$$
m^{ID}\approx-1.1719377251,\quad
m^{OOD}\approx-1.6736491213,\quad v=1,
$$

$$
r=-\frac{-1.1719377251-1.6736491213}{2}
\approx1.4227934232.
$$

另一个真实例子是名单中的g0348、PyOperon：只有seed 521有效，ID/OOD NMSE均为1，因此两个ell和均值均为0，v=1/3，r=-4/3，约-1.3333333333。

## 4. 对每个probe分别标准化

对完整664任务的r计算：

$$
\mu_p=\frac1N\sum_{i\in D}r_{i,p},\qquad
\sigma_p=\sqrt{\frac1N\sum_{i\in D}(r_{i,p}-\mu_p)^2},
$$

$$
z_{i,p}=\frac{r_{i,p}-\mu_p}{\sigma_p}.
$$

使用总体标准差，代码为std(ddof=0)。每个probe独立计算mu和sigma；若r存在非有限值或sigma=0，程序停止。

这里的总体始终是Full664，包括无法进入候选集的任务。选出的CoreK不重新计算自己的mu或sigma。数值也不再次截断。

本次参数按PROBES固定顺序保存：

| probe | mu | sigma |
| --- | ---: | ---: |
| DSO | 2.474613366405 | 4.251781837394 |
| iMCTS | 4.550114233502 | 5.587098983039 |
| PyOperon | -0.202863236584 | 3.916186951412 |
| uDSR | 5.696500819487 | 4.613400588510 |

z=0表示达到该probe在Full664上的平均response；z>0表示高于其平均值；z<0表示低于其平均值。一个任务保留四个z值，后续分别计算四个距离。

上例g0620、DSO：

$$
z=\frac{1.4227934232-2.4746133664}{4.2517818374}
\approx-0.2473833286.
$$

## 5. 经验分布、CDF和W1

### 5.1 经验分布的质量

固定一个probe p：

- F_{p,D}给Full664中的每个任务质量1/664。
- F_{p,S}给选中的每个任务质量1/K。
- 相同z值的任务分别贡献质量；聚合支持点时保留其计数。

经验CDF表示数值不超过t的累计比例：

$$
F_{p,D}(t)=\frac1N\sum_{i\in D}1[z_{i,p}\le t],\qquad
F_{p,S}(t)=\frac1K\sum_{i\in S}1[z_{i,p}\le t].
$$

### 5.2 W1定义

$$
W_p(S)=\int_{-\infty}^{+\infty}
\left|F_{p,S}(t)-F_{p,D}(t)\right|\,dt.
$$

它计算两条CDF之间的面积，比较一个probe的整个response分布。距离单位为该probe的总体标准差单位，W1越小，两个分布越接近。

### 5.3 程序实际使用的精确离散公式

将probe p在Full664上的不同z值排序为：

$$
t_{p,1}<\cdots<t_{p,M_p}.
$$

对相邻支持点之间的区间定义：

$$
\Delta_{p,l}=t_{p,l+1}-t_{p,l},\qquad
q_{p,l}=\frac1N\#\{i\in D:z_{i,p}\le t_{p,l}\},
$$

$$
c_{p,l}(x)=\sum_{i:z_{i,p}\le t_{p,l}}x_i.
$$

Delta为区间长度，q为Full664累计比例，c为选中任务的累计数量。CDF在每个区间上恒定，所以：

$$
W_p(S)=\sum_{l=1}^{M_p-1}
\Delta_{p,l}\left|\frac{c_{p,l}(x)}K-q_{p,l}\right|.
$$

该式直接处理两个集合样本数不同的情况。选择集合取自Full664，两端区间之外的CDF差为零。支持点只有一个时，距离为零。

## 6. 唯一优化目标

$$
D_{\mathrm{resp}}(S)=\frac14\sum_{p\in P}W_p(S),
$$

$$
\min_{\substack{S\subseteq D\\|S|=K\\S\text{满足第7节约束}}}
D_{\mathrm{resp}}(S).
$$

四个距离权重均为1/4。标准化控制probe之间的数值尺度差异；每个probe的CDF距离单独计算后再平均。

这是四个一维边际分布距离的平均，不包含四维联合分布或probe间相关性矩阵的匹配。一个任务是否入选由同一个x_i控制，因此它同时影响四个probe的分布。

Coverage、MeanInfo及MAE的目标系数均为零。历史Core50的成员身份不进入模型；MAE已用于本次方法开发的事后评价，因此这些结果属于construction诊断。

当前仅计算Core50。此入口一次处理一个K，没有跨规模嵌套约束，也没有对旧Core80重叠数的下限要求。

## 7. 必须满足的选择约束

### 7.1 规模和二元变量

$$
x_i\in\{0,1\},\qquad \sum_{i\in D}x_i=K.
$$

每个任务最多入选一次。当前K=50。

### 7.2 数据集资格

dataset_valid_rate是历史元信息中的数据集有效运行比例，原定义为：

$$
v_i^{meta}=\frac{valid\_runs_i}{\max(finished\_runs_i,1)}.
$$

本次finished_runs均为12，且已核验该字段与原始result_valid_output标记之和除以12一致。

实际允许条件为：

$$
e_i=1[v_i^{meta}\ge0.5]\,
1[wrong\_dataset\_runs_i=0]\,
1[eligible\_class_i\in\{eligible,limited\_quota\}].
$$

$$
x_i\le e_i.
$$

completion_rate=1作为整个输入的前置检查；存在未完成数据集时停止计算。当前663项符合资格，g0619的有效率为1/3，未进入候选集。

eligible_class的来源见第10节。limited_quota在当前程序中是允许的资格标签，本模型没有为该标签另设数量上限。

### 7.3 两类去重

semantic_duplicate_group是输入元信息提供的公式身份组；basename是输入记录中的任务基础名称。选择器按字段值完全相等分组，不重新做符号等价判断。

对每个语义组G和每个同名组H：

$$
\sum_{i\in G}x_i\le1,\qquad
\sum_{i\in H}x_i\le1.
$$

这两类约束同时生效。当前664条元信息的两个字段都没有空值。历史语义组由formula_identity优先确定；缺少该身份时，上游使用family、subgroup和basename组成的标识。

### 7.4 family配额

family表示数据集所属来源族。令F为Full664中不同family数量，n_f为family f在完整664中的任务数。当前F=8：

$$
w_f=0.6\frac{n_f}{N}+0.4\frac1F,\qquad q_f=K w_f.
$$

0.6部分按总体频率分配，0.4部分按family等额分配。q_f是期望数量，可以为小数。

$$
L_f=\max\left(1,\left\lfloor q_f-\frac K{50}\right\rfloor\right),
\qquad
U_f=\left\lceil q_f+\frac{2K}{50}\right\rceil,
$$

$$
L_f\le\sum_{i:family_i=f}x_i\le U_f.
$$

floor向下取整，ceil向上取整。K/50和2K/50是随规模变化的允许数量余量；最低1个任务适用于所有family。n_f使用Full664数量，未按eligible候选池重新计算。

当前K=50的实际约束和结果：

| family | Full664数量 | q_f | L_f | U_f | 选中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| keijzer | 15 | 3.177711 | 2 | 6 | 2 |
| korns | 12 | 3.042169 | 2 | 6 | 2 |
| llm-srbench | 240 | 13.343373 | 12 | 16 | 14 |
| nguyen | 12 | 3.042169 | 2 | 6 | 4 |
| srbench1.0 | 133 | 8.509036 | 7 | 11 | 9 |
| srbench2025/firstprinciples | 12 | 3.042169 | 2 | 6 | 3 |
| srsd | 232 | 12.981928 | 11 | 15 | 15 |
| vladislavleva | 8 | 2.861446 | 1 | 5 | 1 |

### 7.5 subgroup上限

subgroup表示输入元信息中的子类别，按该字段字符串分组：

$$
\sum_{i:subgroup_i=g}x_i\le
\left\lceil 8\frac K{50}\right\rceil.
$$

K=50时每个subgroup最多8项，下限为0。当前选中了18个subgroup中的17个。具体计数保存在[constraints.json](response_distribution_core50/constraints.json)。

family、subgroup、两个重复字段是当前实际加入MILP的四类分组约束。difficulty_bin、failure_mode、winner_probe用于诊断类别统计，本模型没有为它们设置独立数量限额。

## 8. MILP如何计算距离和求解

### 8.1 绝对值的线性表示

定义b_{p,l}=Kq_{p,l}，引入连续变量e_{p,l}：

$$
e_{p,l}\ge c_{p,l}-b_{p,l},\qquad
e_{p,l}\ge b_{p,l}-c_{p,l},\qquad 0\le e_{p,l}\le K.
$$

e的目标系数为正，在最优赋值中等于|c-b|。因此目标等价于：

$$
\min\frac1{4K}\sum_p\sum_l\Delta_{p,l}e_{p,l}.
$$

### 8.2 整数CDF加强约束

累计数量c_{p,l}为整数。令a=floor(b)、f=b-a，增加：

$$
e\ge(1-2f)c+f-(1-2f)a.
$$

这条直线连接c=a和c=a+1时的两个绝对值函数值，对其他整数c也不超过|c-b|。它提高连续松弛的下界，保持原有整数选择集合不变。程序对每个区间的c=0至K逐项验证这一不等式的有效性。

### 8.3 软件参数与目标缩放

- scipy.optimize.milp，SciPy 1.11.4，使用其HiGHS求解器。
- time_limit=180秒：本次求解器调用的预算。
- mip_rel_gap=0：请求继续搜索直到证明最优或达到时间预算。
- presolve=True：启用预处理。
- 所有目标系数乘10^6，改善数值尺度；保存结果时除回10^6。

整体乘同一正数不会改变候选的目标排序。求解预算不包含加载输入、构建矩阵和结果导出的全部时间；summary.seconds测量milp调用及返回开销。

### 8.4 辅助诊断变量

模型共用函数还保存类别是否出现的辅助变量，以及Coverage、MeanInfo的线性表达。当前入口将两项log代理变量固定为0，并将其目标系数设为0；Coverage、MeanInfo的范围为[0,1]，没有额外的质量下限约束。真正被最小化的数值是D_resp。

### 8.5 检查和求解状态

返回结果后依次验证：

1. 二元变量距最近整数不超过1e-7。
2. 恰好K项，全部满足资格和各组数量约束。
3. 所有线性约束违反量不超过1e-6。
4. 使用SciPy wasserstein_distance从最终名单独立重算四个距离。
5. 独立重算D_resp与求解器目标除以10^6的差不超过1e-9。

对于最小化问题，当前可行解的目标U是最优值上界，求解器给出的L是下界：

$$
L\le D_{\mathrm{resp}}(S^\star)\le U.
$$

当前U=0.02783574193218572，L=0.02568934735860432。相对gap为：

$$
\frac{U-L}{|U|}=0.07710929993625723\approx7.71\%.
$$

gap描述优化目标的上下界差异，适用范围为D_resp。MAE的定义见第9节。当前solver_status=1，表示达到时间预算；该名单是已验证的可行解，尚未证明全局最优。

certified_within_tolerance只在solver_status=0且目标与最优界差不超过1e-9时为true。

## 9. MAE的完整定义

MAE在原始response分数空间计算：

$$
A_p(D)=\frac1{664}\sum_{i\in D}r_{i,p},\qquad
A_p(S)=\frac1K\sum_{i\in S}r_{i,p},
$$

$$
e_p=A_p(S)-A_p(D),\quad
AE_p=|e_p|,\quad
MAE=\frac14\sum_p AE_p.
$$

每个集合内部对任务等权，最终对四个probe等权。这里MAE的单位是response分数，衡量两个集合平均表现的差异；模型预测误差通过NMSE进入response。

当前结果：

| probe | A_p(D) | A_p(S) | signed_error | absolute_error |
| --- | ---: | ---: | ---: | ---: |
| DSO | 2.4746133664 | 2.4989799369 | 0.0243665705 | 0.0243665705 |
| iMCTS | 4.5501142335 | 4.5484951620 | -0.0016190715 | 0.0016190715 |
| PyOperon | -0.2028632366 | -0.2008552724 | 0.0020079641 | 0.0020079641 |
| uDSR | 5.6965008195 | 5.7206049239 | 0.0241041044 | 0.0241041044 |

$$
MAE=0.013024427640043565.
$$

W1与这个MAE存在下列界：

$$
|A_p(S)-A_p(D)|\le\sigma_p W_p(S),\qquad
MAE\le\frac14\sum_p\sigma_p W_p(S).
$$

理由是两个分布的均值差不超过一维W1，标准化再换回response单位时乘sigma_p。本次右侧约0.1267940890。这个关系解释了匹配四个response分布为何能控制四probe的均值差；它不提供其他算法的误差保证。

## 10. Coverage、Info与元信息标签

这些量用于描述名单。优化目标仍为第6节的D_resp。

### 10.1 Coverage

对每个属性a，记C_a(D)为Full664中出现过的类别集合，C_a(S)为子集中出现过的类别集合：

$$
Coverage_a(S)=\frac{|C_a(S)|}{|C_a(D)|},\qquad
Coverage(S)=\frac1{10}\sum_{a=1}^{10}Coverage_a(S).
$$

同一类别出现1次或多次，对这一属性的覆盖数贡献都为1。十个属性等权，每个属性内部的类别也等权。

| 属性 | 定义 | 全量类别数 | 当前覆盖数 |
| --- | --- | ---: | ---: |
| family | 数据来源族 | 8 | 8 |
| subgroup | 元信息定义的子类别 | 18 | 17 |
| operator_group | 公式算子类别标签 | 6 | 5 |
| feature_count_bin | 特征数分组 | 3 | 3 |
| sample_count_bin | 样本数分组 | 3 | 3 |
| complexity_bin | Ground Truth算子数分组 | 3 | 3 |
| difficulty_bin | 历史probe难度分组 | 4 | 4 |
| failure_mode | 历史失败及响应模式标签 | 6 | 5 |
| winner_probe | 历史最佳probe身份 | 4 | 4 |
| eligible_class | 元信息资格类别 | 2 | 2 |

因此：

$$
Coverage=\frac{1+17/18+5/6+1+1+1+1+5/6+1+1}{10}
=0.9611111111111112.
$$

### 10.2 Info的来源和作用

当前选择器读取Stage3元信息中的discrimination_score、stability_score、info_score，并核验：

$$
Info_i=Disc_i\times Stab_i,\qquad
MeanInfo(S)=\frac1K\sum_{i\in S}Info_i.
$$

乘积核验容差为1e-10，info_score必须处于[0,1]的数值范围。本次MeanInfo=0.21574382919481022。

这组诊断字段使用历史跨seed中位数和IQR；当前response使用第3节的算术平均。两条计算链在输入、字段名称和作用上分别记录。

已找到其[历史后处理实现](../../../眼不见为净/check/postprocess_probe4_full664_metrics.py)，以下定义来自该实现。本次从冻结表重新计算Disc、Stab、Info，最大数值差均小于3e-16。

### 10.3 历史诊断的中位数、IQR和归一化

历史dataset×probe汇总使用run_outcome_class为valid_finite_result或valid_extreme_error的记录。其log值优先读取历史保存的clip12字段；没有已保存值时，上游才从原始NMSE计算。当前选择器直接读取这份冻结的汇总表。

- 中位数：排序后的中间值；偶数个数时取两个中间值的平均。
- Q_q：将m个有限值从小到大排列，使用位置h=(m-1)q进行相邻值线性插值。
- IQR：Q_0.75-Q_0.25；一个有效值时IQR=0。

对Full664上的某个诊断字段x定义：

$$
\mathcal N(x_i)=
\operatorname{clip}\left(\frac{x_i-Q_{0.05}}{Q_{0.95}-Q_{0.05}},0,1\right).
$$

分位数仅由该字段的有限值计算。单项缺失时其归一化值为0；没有有限值，或两个分位点在math.isclose默认容差下相等时，整列归一化值为0。

### 10.4 Discrimination的每个分量

对任务i，P_i^valid表示历史汇总中至少有一个有效seed的probe。记a_{i,p}和b_{i,p}为对应历史ID、OOD log NMSE中位数。

| 分量 | 精确定义 |
| --- | --- |
| id_probe_variance | 有限a_{i,p}的总体方差，分母为参与probe数；少于两个值时为0 |
| ood_probe_variance | 有限b_{i,p}的总体方差；少于两个值时为0 |
| ood_id_gap_variance | 有限b_{i,p}-a_{i,p}的总体方差；少于两个值时为0 |
| pairwise_gap_mean | 有限OOD中位数的所有无序probe对之绝对差的均值；没有有效对时缺失 |
| valid_pattern_entropy | 令u=有效probe数/4，计算-u log2(u)-(1-u)log2(1-u)；u为0或1时取0 |

每个分量分别经过上述Full664归一化：

$$
Disc_i=
0.35\mathcal N(Var^{ID}_i)
+0.35\mathcal N(Var^{OOD}_i)
+0.15\mathcal N(Var^{gap}_i)
+0.10\mathcal N(PairGap_i)
+0.05\mathcal N(H_i).
$$

这些系数用于生成冻结诊断字段；当前W1优化目标中，每个probe的权重为1/4。

### 10.5 Stability的每个分量

对P_i^valid中的probe，分别取历史seed IQR的平均值和最大值：

$$
u_i=
0.40\,MeanIQR^{OOD}_i+
0.20\,MeanIQR^{ID}_i+
0.20\,MaxIQR^{OOD}_i+
0.10\,InvalidRate_i+
0.10\,TimeoutRate_i,
$$

$$
Stab_i=1-\mathcal N(u_i).
$$

- MeanIQR只对有限值求平均；这些IQR来自每个probe的有效seed。
- MaxIQR为有效probe的有限OOD IQR最大值，没有值时为0。
- 上式中缺失的MeanIQR按历史实现取0。
- InvalidRate：run_outcome_class属于invalid_output、no_output、system_error、unknown_failure的次数，除以max(finished_runs,1)。
- TimeoutRate：run_outcome_class为timeout_no_output的次数，除以相同分母。
- partial_output不计入上述InvalidRate定义，因此InvalidRate不能直接写成1-dataset_valid_rate。

### 10.6 诊断类别标签如何生成

当前选集读取已冻结标签。以下规则来自历史后处理脚本，本次已核验与输入表一致：

1. feature_count_bin：对Full664有限feature_count取1/3和2/3分位点，按≤第一分位点、≤第二分位点、其余分为low、medium、high。
2. sample_count_bin：total_samples为train_samples、valid_samples、id_test_samples、ood_test_samples之和，缺失计0；按同样三分位规则标记small、medium、large。
3. complexity_bin：使用Ground Truth的formula_operator_count，按三分位规则标记simple、moderate、complex。以上缺失数值或分位点缺失时标记unknown。
4. operator_group：依据formula_return_expr的小写文本。含sin/cos/tan且无exp/log为trigonometric；含exp/log且无三角函数为exponential_log；两类同时存在为mixed_elementary；其余含除法符号为rational；其余含幂、sqrt或加减乘为polynomial；其他情况为unknown。该字段是文本规则生成的标签。
5. difficulty_score：在同时具有有限ID/OOD中位数的有效probe上，平均(a_{i,p}+b_{i,p})/2。使用Full664的20%、60%、90%分位点划分easy、medium、hard、extreme；InvalidRate≥0.5或TimeoutRate≥0.5时优先标记extreme；缺失时标记unknown。
6. winner_probe：有效probe按OOD中位数升序排序，依次用ID中位数升序、valid_rate降序、median_runtime升序、method_norm字典序处理相同值。排名第一者为winner；第二名OOD中位数减第一名得到winner_margin。

failure_mode按以下顺序匹配，命中后停止：

| 顺序 | 标签 | 条件 |
| ---: | --- | --- |
| 1 | incomplete | completion_rate<1 |
| 2 | invalid_prone | InvalidRate≥0.5或TimeoutRate≥0.5 |
| 3 | one_sided | 有效probe数≤2 |
| 4 | unstable | MaxIQR^OOD≥其Full664的90%分位点，且该分位点>0 |
| 5 | ood_failure | 跨有效probe的平均OOD中位数减平均ID中位数≥该差值的75%分位点，且分位点>0 |
| 6 | all_struggle | difficulty_bin为extreme |
| 7 | one_method_wins | winner_margin≥其Full664的75%分位点，且分位点>0 |
| 8 | all_good | 其余情况 |

eligible_class按顺序生成：

1. completion_rate<1：incomplete。
2. wrong_dataset_runs>0、有效probe数为0或dataset_valid_rate为0：excluded。
3. failure_mode属于invalid_prone、one_sided、unstable、all_struggle，或difficulty_bin为extreme：limited_quota。
4. 其余情况：eligible。

以上标签不等价于当前选集的数量配额。实际数量限制以第7节和constraints.json为准。

## 11. 结果字段、数值及边界

### 11.1 当前四个距离

| probe | 标准化W1 |
| --- | ---: |
| DSO | 0.03557465682277712 |
| iMCTS | 0.024178706621341618 |
| PyOperon | 0.024628981261548152 |
| uDSR | 0.02696062302307601 |

四项平均为0.02783574193218572，与summary.objective一致。

### 11.2 summary字段解释

| 字段 | 含义及作用 |
| --- | --- |
| objective、terms.mean_standardized_w1 | 当前唯一目标D_resp |
| dual_bound、absolute_gap、mip_gap | 最优值下界、目标绝对界差、相对界差 |
| solver_status、solver_message | 求解器结束状态 |
| certified_within_tolerance | 是否达到代码规定的最优性核验条件，本次false |
| terms.coverage、terms.mean_info | 第10节的诊断指标 |
| terms.response_balance | exp(-D_resp)，本次0.9725481025541267，作为诊断值 |
| terms.log_objective | 共享诊断函数计算的ln(Coverage)+ln(MeanInfo)-D_resp |
| terms.geometric_objective | 上一项除以3后取exp，等于(Coverage×MeanInfo×exp(-D_resp))^(1/3) |
| probe_results | 四个probe的Full664均值、子集均值、带符号差及绝对差 |
| mae | 第9节的四probe平均绝对差 |
| constraints_passed | 本次最终名单通过资格、数量、去重及线性约束核验 |
| historical_members_used_in_optimization | 历史固定50成员是否进入优化模型，本次false |
| mae_used_in_optimization | MAE是否作为数学求解目标，本次false |
| historical_native_replay_verified | 是否重放历史原生模型验证输入，本次false |
| loo_executed | 是否已完成leave-one-probe-out，本次false |
| formal_ready | 是否已经完成正式结果所需的输入和验证审核，本次false |

terms中的log_objective和geometric_objective仅由共享诊断函数计算并保存，不用于当前MILP排序。当前代码将对应代理变量的系数和取值置零。

当前名单与暂存旧Core80重叠10项，新增40项；与用户固定50重叠7项。重叠检查按数据集身份进行，没有进入求解目标。

### 11.3 结果可解释的范围

- 当前结果说明所选50项在四个construction probes的response均值和各自边际分布上接近Full664。
- 它没有验证四probe的联合响应关系，也没有验证其他算法上的表现。
- 当前MeanInfo为0.215744；用户固定50为0.518694。较小MAE同时伴随较低的历史信息量指标。
- 选择器输入来自既有1小时、三个seed的探针记录；尚未重新运行这些算法或验证修复后的原生预测。
- 当前调用只选择50项，不保证未来独立选择的其他K与本名单嵌套。

## 12. 文件、代码入口与复算

### 12.1 函数对应关系

| 步骤 | 实际实现 |
| --- | --- |
| 输入资格与元信息准备 | 暂存包lib/engine.py中的prepare |
| 单dataset×probe response | [evaluate_core50.py](evaluate_core50.py)中的response |
| r矩阵、mu、sigma、z | [select_response_balance.py](select_response_balance.py)中的load_data |
| 结构约束、CDF矩阵、整数加强约束 | 同文件build_model |
| 独立W1、Coverage及Info诊断 | 同文件terms |
| 当前单目标系数、MILP调用、最终核验与导出 | [select_response_distribution.py](select_response_distribution.py)中的main |
| 历史Info及类别标签定义 | [postprocess_probe4_full664_metrics.py](../../../眼不见为净/check/postprocess_probe4_full664_metrics.py) |

元信息准备函数位于[暂存包lib/engine.py](../../pre_exp_26.09.23/stage4_core80_15algs_3seeds_3h/1、build_core30_80/core30_core80_all_sensitivity_package/core_nested_all_sensitivity/lib/engine.py)。这些脚本的源文件哈希已绑定在configuration.json中。

### 12.2 输出文件

| 文件 | 内容 |
| --- | --- |
| [core50.csv](response_distribution_core50/core50.csv) | 50个任务的ID、名称、完整路径、分组、有效率及Info |
| [configuration.json](response_distribution_core50/configuration.json) | 输入和脚本哈希、评分版本、mu/sigma、约束参数、软件版本及预算 |
| [constraints.json](response_distribution_core50/constraints.json) | 各family、subgroup、重复组的上下限与实际数量 |
| [solver_solution.json](response_distribution_core50/solver_solution.json) | 完整求解变量及选中任务的零基行索引 |
| [summary.json](response_distribution_core50/summary.json) | 目标、下界、gap、诊断指标、MAE、名单ID和校验状态 |

当前软件版本为NumPy 1.26.4、pandas 2.1.4、SciPy 1.11.4。solver_solution中的indices对应元信息按dataset_id排序后的行；前664个solution变量为二元选择变量。

### 12.3 完整参数表

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| N、probe数量、seed数量 | 664、4、3 | 固定输入规模 |
| K | 50 | 输出任务数量 |
| log_base、nmse_floor | 10、1e-12 | response数值转换 |
| log_clip | [-12,12] | 单seed对数截断 |
| seed_aggregation | 有效seed算术平均 | 构造任务response |
| failure_penalty | 2 | 有效seed比例惩罚 |
| response_floor、all_invalid_response | -12、-12 | response范围与全无有效输出处理 |
| std_ddof | 0 | Full664总体标准差 |
| probe距离权重 | 每个1/4 | 唯一目标的等权聚合 |
| min_dataset_valid_rate | 0.5 | 候选资格 |
| family_proportional_weight | 0.6 | family频率与等额份额的混合 |
| family_minimum | 1 | 每个family至少一个 |
| family_min_margin、family_max_margin | K/50、2K/50 | family数量余量 |
| subgroup_max | ceil(8K/50) | subgroup上限 |
| duplicate_cap | 1 | 两类去重 |
| objective_scale | 1e6 | 求解器数值缩放 |
| time_limit、mip_rel_gap、presolve | 180、0、true | 求解预算与终止设置 |
| 整数性、线性约束、目标核验容差 | 1e-7、1e-6、1e-9 | 返回结果检查 |

这些常量的作用分别是评分定义、结构性政策和数值求解设置。Coverage/Info诊断的历史系数见第10节；它们不构成当前目标的可调混合权重。

### 12.4 重新执行与验证

从仓库根目录运行，使用独立输出目录：

~~~bash
mkdir -p .agent/work/METHOD-002
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_CACHE_DIR="$PWD/.agent/work/METHOD-002/numba" TMPDIR="$PWD/.agent/work/METHOD-002" \
python A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/select_response_distribution.py \
  --size 50 --seconds 180 --output .agent/work/METHOD-002/reproduce_core50
~~~

已有summary.json时入口拒绝覆盖。有限时间内的可行解可能随机器速度和求解器版本变化；复查既有结论时应使用已保存名单及哈希。

本次文档数值复算脚本为.agent/work/METHOD-002/document_checks.py，结果为同目录documentation_checks.json。它核验了历史Disc/Stab/Info及类别标签、当前标准化参数、四个W1、MAE、Coverage、family配额和真实示例，没有重新执行MILP或改写原始数据。

## 13. Info下限候选实验

`select_response_distribution.py`支持`--min-info`，默认0，保持上述基准方法。指定下限τ时，在原有约束基础上增加`MeanInfo(S) >= τ`，仍最小化四probe标准化response的W1均值。τ为显式实验参数，MAE仅在选集后计算。

本次使用τ=0.5、K=50、求解预算170秒，外部执行上限180秒，输出独立保存在`response_distribution_info050_core50/`，原候选结果保留。Info=0.5000687851，MAE=0.2714542142，Coverage=0.9111111111，W1=0.1265233624；求解下界0.1250389814、相对gap约1.17%，尚未证明全局最优。Info达到目标，MAE未达到约0.15。

输入版本与基准相同，源码及输入哈希、标准化参数、依赖版本、下限和预算记录在同目录`configuration.json`，完整解、约束检查、名单和指标分别为`solver_solution.json`、`constraints.json`、`core50.csv`和`summary.json`。复算脚本为`.agent/work/METHOD-003/verify_info.py`，证据为同目录`verification.json`。结果仍属于construction诊断，未执行LOO或历史原生预测回放。

~~~bash
python A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/select_response_distribution.py \
  --size 50 --min-info 0.5 --seconds 170 \
  --output .agent/work/METHOD-003/reproduce_info050
~~~

## 14. 固定留出uDSR的泛化检查

本次检查纯W1基准方法，固定留出uDSR，使用DSO、iMCTS、PyOperon重新选择50项；不使用Info下限。固定uDSR使construction保留三个算法范式，该选择在求解及评估前确定。资格筛选仅使用construction的9次运行，有效运行比例至少0.5；response及标准化也仅使用这三个probe。元信息只读取dataset_id、dataset_name、dataset_rel、family、subgroup、semantic_duplicate_group和basename，保留原family/subgroup及两类去重约束。四probe汇总的Info、difficulty、failure、winner和资格标签不进入模型。

入口为`select_heldout_response.py`，输出为`heldout_udsr_core50/`。名单写入CSV并记录SHA256之后才计算uDSR的response；评估完成后再次检查名单SHA256不变。输入、源码哈希、选择字段、参数及依赖版本保存在`configuration.json`，解和约束保存在`solver_solution.json`及`constraints.json`。

170秒求解得到construction MAE=0.0167286348；uDSR的Full664平均response=5.6965008195、Core50平均response=5.9104906084，held-out绝对误差=0.2139897890。留出算法只有一个，因此本次held-out MAE等于该绝对误差。W1目标=0.0259763534、下界=0.0254834743、gap约1.90%，未证明全局最优。资格检查排除7项，约束及指标复算通过；核验脚本`.agent/work/METHOD-004/verify_heldout.py`、证据同目录`verification.json`也确认共享模型默认四probe路径可以复算原始保存解。

本次是已有数据上的单次留出检查，尚未检验其他留出算法。方法开发此前使用过全部四probe，因此这次结果具有回顾性，不能视作全新外部算法的独立确认；未按本次held-out指标继续调整选择规则。

~~~bash
python A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/1、preflight/select_heldout_response.py \
  --held-out udsr --seconds 170 --output .agent/work/METHOD-004/reproduce_udsr
~~~

### 14.1 四次leave-one-probe-out结果

入口`run_heldout_audit.py`，汇总`heldout_loo_summary.json`。保持上一节方法及170秒预算，复用uDSR结果，DSO/iMCTS/PyOperon三项并行完成，单个进程总时间约176.5秒。每次均使用其余三个probe重新计算资格、response及标准化并重新选集。每份名单保存后才计算对应held-out误差，没有根据留出结果更改方法或选择重复运行中的较好结果。

| 留出算法 | construction MAE | held-out绝对误差 | 求解gap |
| --- | ---: | ---: | ---: |
| DSO | 0.01304508 | 0.50328723 | 6.74397% |
| iMCTS | 0.02173517 | 0.06273559 | 3.42671% |
| PyOperon | 0.02506906 | 0.02541520 | 0.71787% |
| uDSR | 0.01672863 | 0.21398979 | 1.89741% |

四次held-out绝对误差等权平均为0.2013569518，construction MAE平均0.0191444845。两次held-out误差不超过0.15，最大误差为DSO的0.5032872262。按用户关注的约0.15目标，当前结果尚不支持纯W1选择对各算法都稳定达到该水平；iMCTS和PyOperon两次结果满足该目标。没有随机选集比较，因此不据此声称优于随机选集。此前方法开发使用四probe、每次求解保留非零gap，这些条件继续限制结论范围。

四份名单均为50项；从原始记录独立复算construction及held-out误差、W1、资格、family/subgroup约束和两类去重通过，四次输入及选择器/模型/评分源码哈希一致。每次配置和完整解位于`heldout_<probe>_core50/`，执行命令、输入及脚本哈希保存在汇总文件，运行日志位于`.agent/work/METHOD-005/`。原有四probe候选及Info下限候选保留。
