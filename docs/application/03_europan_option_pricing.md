# 算法3：欧式期权定价算法（QAE 振幅估计）

## 1. 算法定位

欧式期权定价算法面向到期一次行权的看涨期权估值问题。输入为标的资产当前价格、波动率、无风险利率、到期时间和执行价，输出期权的贴现前期望收益以及 Delta。该问题不建模为组合优化，而作为量子振幅估计应用：把到期标的价格分布制备为量子态，将 payoff 编码到辅助量子比特振幅中，再估计该振幅对应的期望收益。

## 2. 数学形式

给定欧式看涨期权执行价 `K`，到期标的价格为随机变量 $S_T$，payoff 为：

$$
\max\{S_T-K,0\}.
$$

贴现前公允价值为：

$$
\mathbb{E}\left[\max\{S_T-K,0\}\right].
$$

若需要得到当前价格，可在经典后处理中乘以贴现因子：

$$
C_0
= e^{-rT}\mathbb{E}\left[\max\{S_T-K,0\}\right].
$$

Delta 描述期权价值对标的价格的敏感度。在 Qiskit Finance 教程的离散模型口径下，可用价内概率表示：

$$
\Delta = \mathbb{P}[S_T \ge K].
$$

到期标的价格采用截断对数正态分布，并用 $n$ 个量子比特离散为 $2^n$ 个网格点：

$$
\lvert 0\rangle_n
\mapsto
\lvert \psi\rangle_n
=
\sum_{i=0}^{2^n-1}\sqrt{p_i}\lvert i\rangle_n.
$$

索引 `i` 映射到价格区间 $[\mathrm{low},\mathrm{high}]$：

$$
S_T(i)
=
\frac{\mathrm{high}-\mathrm{low}}{2^n-1}i+\mathrm{low}.
$$

## 3. 求解方法

采用 Quantum Amplitude Estimation（QAE）。量子线路包含两部分：

1. 不确定性模型：用 `LogNormalDistribution` 制备 $S_T$ 的离散概率分布。
2. 收益函数模型：用比较器判断 $S_T \ge K$，并用分段线性振幅函数近似 $\max(S_T-K,0)$。

payoff 的线性部分通过受控 $Y$ 旋转写入目标量子比特。利用近似关系：

$$
\sin^2(y+\pi/4)\approx y+1/2,
$$

可把归一化后的线性收益转换为目标量子比特测得 $\lvert 1\rangle$ 的概率。随后用 Iterative Amplitude Estimation 估计该概率，并通过后处理还原为收益金额。

Delta 更简单：只需比较器标记 $S_T \ge K$，目标量子比特的成功概率即为价内概率。

## 4. 具体实现

实现流程：

1. 根据标的资产当前价格、波动率、无风险利率和到期时间，建立风险中性下的到期价格分布。
2. 将连续价格分布截断到有限价格区间，并离散为量子寄存器可表示的价格网格。
3. 制备到期价格分布的量子叠加态，使每个价格网格点的振幅对应其概率。
4. 构造看涨期权收益目标，使低于执行价的价格区间收益为零，高于执行价的价格区间按线性 payoff 增长。
5. 将收益函数映射到辅助量子比特的成功概率。
6. 使用振幅估计读取成功概率，并还原为贴现前期望收益。
7. 对期权价格进行经典贴现，得到当前理论价格。
8. 另用价内判断目标估计到期价格高于执行价的概率，得到 Delta。
9. 输出期权价格、Delta、估计误差和置信区间。

## 5. Baseline 与优势口径

Baseline 待定

1. 经典 Monte Carlo 
2. 差的 QAE对比好的 QAE

## 6. 验证结果

Qiskit Finance 教程样例：

```text
S = 2.0
vol = 0.4
r = 0.05
T = 40 / 365
num_uncertainty_qubits = 3
strike_price = 1.896
c_approx = 0.25
epsilon = 0.01
alpha = 0.05
shots = 100
```

贴现前期望收益估计：

```text
Exact expected payoff = 0.1623
QAE estimated payoff = 0.1687
Confidence interval = [0.1637, 0.1737]
```

Delta 估计：

```text
Exact delta = 0.8098
QAE estimated delta = 0.8091
Confidence interval = [0.8034, 0.8148]
```

该结果展示了同一不确定性模型可同时服务于欧式看涨期权期望收益估计和 Delta 估计；区别只在目标电路：定价使用 payoff 振幅函数，Delta 使用比较器成功概率。
