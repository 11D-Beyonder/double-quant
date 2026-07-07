# 算法2：风险归因算法

## 算法定位

风险归因算法面向投资组合内部风险来源识别。输入为资产收益率序列，输出每个资产对组合的风险贡献。举例来说，在股票和债券各占50%的投资组合中，股票资产因其高波动性往往贡献了绝大多数的风险。

## 数学形式

基于合作博弈论的夏普利值，可作为各个资产在组合中的风险贡献值。

给定资产全集 $N$，任意资产子组合 $S \subseteq N$ 的组合损失为 $L_S$。在置信水平 $\alpha$ 下，风险价值和预期短缺分别为：

$$
\operatorname{VaR}_\alpha(L)
= \inf\{x \mid P(L > x) < 1-\alpha\},
$$

$$
\operatorname{ES}_\alpha(L)
= \mathbb{E}[L \mid L \ge \operatorname{VaR}_\alpha(L)].
$$

后续记 $\operatorname{ES}(S)$ 为组合 $S$ 的预期短缺。第 $i$ 个资产的夏普利风险贡献定义为：

$$
\Phi^\text{SRC}_i
= \sum_{S \subseteq N \setminus \{i\}}
\gamma(n, |S|)
\left[
\operatorname{ES}(S \cup \{i\}) - \operatorname{ES}(S)
\right],
$$

其中

$$
\gamma(n,s)=\frac{s!(n-s-1)!}{n!}.
$$

$\Phi^\text{SRC}_i$ 满足有效性：

$$
\sum_i \Phi^\text{SRC}_i = \operatorname{ES}(N).
$$

同时 ES 满足次可加性，

$$
\operatorname{ES}(A \cup B)
\le \operatorname{ES}(A)+\operatorname{ES}(B),
\qquad A \cap B = \emptyset,
$$

直接编码原始 ES 的边际贡献可能出现负数。定义风险节省特征函数：

$$
\operatorname{RS}(S)
= \sum_{j \in S}\operatorname{ES}(\{j\})
- \operatorname{ES}(S).
$$

由 ES 的次可加性可得：

$$
\operatorname{RS}(A \cup B)
\ge \operatorname{RS}(A)+\operatorname{RS}(B),
\qquad A \cap B = \emptyset.
$$

因此 $\operatorname{RS}$ 满足量子 Shapley 算法所需的超可加性。量子算法先求

$$
\Phi_i^{\operatorname{RS}}
= \sum_{S \subseteq N \setminus \{i\}}
\gamma(n, |S|)
\left[
\operatorname{RS}(S \cup \{i\}) - \operatorname{RS}(S)
\right],
$$

再用还原公式得到真实风险贡献：

$$
\operatorname{SRC}_i
= \operatorname{ES}(\{i\}) - \Phi_i^{\operatorname{RS}}.
$$

## 3. 求解方法

采用 Shapley / Quantum Shapley。经典精确解可直接枚举所有子组合；经典近似解可用排列 Monte Carlo 采样。量子解法使用三个寄存器：

1. 区间寄存器 $Q_l$：用 $n_l$ 个内部量子比特离散化 Shapley 权重的积分形式。
2. 玩家寄存器 $Q_p$：表示除目标资产 $i$ 外的候选联盟。
3. 输出寄存器 $Q_a$：把边际贡献编码为输出比特的振幅。

量子电路以 $\operatorname{RS}$ 为特征函数，编码非负边际贡献：

$$
\operatorname{RS}(S \cup \{i\})-\operatorname{RS}(S) \ge 0.
$$

输出比特测得 $|1\rangle$ 的概率与近似 Shapley 值成比例。结果提取可采用状态向量精确读取、多次测量采样，或不同形式的量子振幅估计。

## 4. 具体实现

实现流程：

1. 获取资产价格并转换为收益率序列。
2. 对任意资产子组合 $S$，按历史模拟法计算 $\operatorname{ES}(S)$。
3. 经典直接路径以 $\operatorname{ES}$ 作为特征函数，枚举或采样各资产的边际风险增量。
4. 量子兼容路径先把 $\operatorname{ES}$ 转换为 $\operatorname{RS}$，保证所有待编码边际贡献非负。
5. 量子电路对目标资产逐一估计 $\Phi_i^{\operatorname{RS}}$。
6. 利用 $\operatorname{SRC}_i = \operatorname{ES}(\{i\}) - \Phi_i^{\operatorname{RS}}$ 还原真实风险贡献。
7. 输出每个资产的 $\operatorname{SRC}$、风险贡献占比和相对资本权重的放大倍数。

## 测试

### 基线与量子优势

使用“经典蒙特卡洛”方法作为基线，量子优势来自 QAE 的二次加速：理论上，在同等目标误差下，QAE 的查询复杂度为 $O(1/\epsilon)$，而经典方法为 $O(1/\epsilon^2)$。

### 功能测试

- 能够正确识别投资组合中的高风险项
    - 使用 TSLA 一只高风险股票，其余都是低波动的红利资产，组成等权投资组合。
    - 对于量子算法计算得到的各资产风险值，TSLA 特别大，其余资产都很小甚至为负数。
    - 对于量子算法计算得到的各资产风险值，各个资产风险值之和，为组合风险值。

- 计算所需操作数
    - 能够统计量子电路采样数。

- 计算精度与量子电路参数之间的函数关系测试
    - 固定采样数不变。
    - 变化积分区间寄存器量子比特数。
    - 得到计算精度随量子比特数的变化曲线。

### 性能测试

- 量子
    

## 6. 验证结果

已有实验结果：

```text
实验窗口 = 2020-04-01 ~ 2022-04-01
风险度量 = ES, alpha = 0.95
风险分层平均波动率 = Low 10.1%, Mid 18.1%, High 38.7%
RS 超可加性验证 = 0 / 5000 违例
ES 直接计算 vs RS 还原 MAE = 5.64e-18
n=5, n_l=6, Statevector mean relative error = 0.0013
epsilon=0.05, Classical MC mean calls = 200
epsilon=0.05, I-QAE mean calls = 1
I-QAE 采样数减少 = 99.5%
```

实证组合中，TSLA 的资本权重为 $10\%$，但 $\operatorname{SRC}$ 占比约为 $105.7\%$，风险贡献放大约 $10.57\times$；TLT、IEF、GOVT、SHY 等债券 ETF 的 $\operatorname{SRC}$ 占比为负，体现对冲资产的防御作用。

对应实验产物见 `docs/assets/risk/` 和 `docs/assets/risk/data/`。
