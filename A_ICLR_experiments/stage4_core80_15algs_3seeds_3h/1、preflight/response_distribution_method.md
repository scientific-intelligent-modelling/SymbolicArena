# 四probe响应分布选集

## 目标

使用四个construction probes在Full664上的response分别标准化：

$$
z_{i,p}=(r_{i,p}-\mu_p)/\sigma_p.
$$

每个probe的均值和总体标准差固定由Full664计算。response沿用probe_consensus_response_v1：逐有效seed的log10 NMSE截断至[-12,12]、有效seed算术平均、失败惩罚2、response下界-12。

直接选择：

$$
\min_{|S|=K}\frac14\sum_{p=1}^{4}
W_1(F_{z_p,S},F_{z_p,664}).
$$

四个probe等权，分别保留响应分布。Coverage和MeanInfo作为诊断指标报告。历史Core50成员不进入优化模型，MAE用于本次方法开发的事后评价。

## 约束

- dataset_valid_rate≥0.5、wrong_dataset_runs=0、运行记录完整。
- semantic_duplicate_group和basename各最多一个。
- 所有family使用同一配额公式：target=K×(0.6×family占比+0.4/family数量)。
- family下限max(1,floor(target-K/50))，上限ceil(target+2K/50)。
- subgroup数量上限ceil(8K/50)。

这些约束包含明确的政策参数，全部写入配置，不根据MAE反复调节。每个类别的实际限额和选中数量保存在constraints.json。

## 当前计算

入口为select_response_distribution.py，输出为response_distribution_core50/。模型采用MILP和整数CDF加强约束。

- 规模：50。
- 求解预算：180秒。
- 当前可行解MAE：0.013024427640043565。
- DSO/iMCTS/PyOperon/uDSR绝对误差：0.0243665705、0.0016190715、0.0020079641、0.0241041044。
- 四probe标准化W1均值：0.02783574193218572。
- 求解器下界：0.02568934735860432；相对gap约7.71%，全局最优性尚未证明。
- Coverage：0.961111；MeanInfo：0.215744。用户固定50的对应数值为0.966667和0.518694，当前结果的信息量较低。
- 与暂存旧Core80重叠10个，新增40个；与用户固定50重叠7个。

该结果基于历史原始指标，尚未完成原生模型回放核验和LOO，formal_ready=false。它证明当前50项在四个construction probes的平均response上接近Full664，不能直接推出其他算法上的保真度。

本次方法开发另保留response_balance_core50/中的几何目标结果及其配置。该次MAE为0.485717031470868，未达到本次误差目标；不用于当前名单。
