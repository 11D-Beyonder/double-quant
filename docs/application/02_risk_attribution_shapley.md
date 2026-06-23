# 算法2：风险归因算法（量子 Shapley / 风险节省对偶）

## 1. 算法定位

风险归因算法面向投资组合内部风险来源识别。输入为资产收益率序列，输出每个资产对组合尾部风险的夏普利风险贡献 $\operatorname{SRC}$。该算法不把资产权重直接等同于风险贡献，而是用 Expected Shortfall（ES）度量组合风险，并用 Shapley 值公平分配资产间相关性带来的风险增量或对冲效应。

量子路径不能直接使用原始 ES 博弈，因为 ES 满足次可加性，而量子 Shapley 电路要求特征函数的边际贡献非负。本文采用风险节省（Risk Saving, RS）对偶变换，将“风险成本分摊”转化为“风险节省分配”，再把量子算法得到的 RS Shapley 值还原为真实 $\operatorname{SRC}$。

## 2. 数学形式

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
\operatorname{SRC}_i
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

$\operatorname{SRC}$ 满足有效性：

$$
\sum_i \operatorname{SRC}_i = \operatorname{ES}(N).
$$

由于 ES 满足次可加性，

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

## 5. Baseline 与优势口径

Baseline 包括三类：

1. 资本权重分摊：按持仓比例分配风险，但无法识别高波动资产和对冲资产的真实贡献。
2. 经典精确 Shapley：枚举所有子组合，复杂度为 $O(n \cdot 2^n)$。
3. 经典 Monte Carlo Shapley：用随机排列近似边际贡献，达到精度 $\epsilon$ 的采样复杂度通常为 $O(1/\epsilon^2)$。

量子优势口径来自 QAE 对振幅估计的二次加速：在同等目标误差下，QAE 的 oracle 查询复杂度为 $O(1/\epsilon)$。风险节省对偶变换本身只引入每个资产一次还原减法，不改变最终 $\operatorname{SRC}$ 的金融含义。

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
