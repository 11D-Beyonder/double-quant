# 75 完成数据获取、分析、编解码流程测试（Func-43） 技术报告

## 技术目标

本功能对应新表格中的功能 75：

```text
完成数据获取、分析、编解码的流程
```

与功能 74 不同，功能 75 不再强调接入工具数量，而是验证统一数据接口后的端到端数据处理链路。

## 总体流程

功能 75 的处理链路如下：

```text
PriceSource.fetch()
  -> 统一价格矩阵
  -> 对数收益率
  -> 协方差矩阵 / 期望收益
  -> Shapley ES风险归因
  -> 资产集合bitmask编码
  -> 金融风险值量子振幅编码
  -> Statevector概率读回
```

## 数据获取阶段

测试通过 `YFinanceSource.fetch()` 获得价格矩阵，输出结构为：

```text
DatetimeIndex x asset columns
```

测试输出：

```text
统一价格矩阵shape: (5, 3)
```

这里使用 monkeypatch 替身数据模拟下载结果。由于功能 64 已经证明 4 类数据源都符合统一接口，所以功能 75 只需要选择一个实现来证明后续流程可以贯通。

## 金融分析阶段

价格矩阵进入 `double_quant.data.transform`：

```python
returns = to_log_returns(prices)
covariance = to_covariance(prices)
expected_returns = to_expected_returns(prices)
```

输出：

```text
对数收益率行列: (4, 3)
协方差矩阵shape: (3, 3)
期望收益: [0.015038, 0.014752, 0.014476]
```

随后进入风险归因模块：

```python
RiskAttributor(
    returns,
    BinaryEnumerationCalculator,
    mode="es",
    alpha=0.75,
).attribute()
```

输出：

```text
Shapley ES风险归因: {'asset_a': 0.00267645, 'asset_b': 0.00256332, 'asset_c': 0.00245447}
```

说明价格数据已经完成从原始时间序列到金融风险分析结果的转换。

## 金融集合编码

Shapley 风险归因和量子 Shapley 路径中，资产子集使用 bitmask 表示：

```text
asset_a -> bit 0
asset_b -> bit 1
asset_a + asset_b -> bitmask 3
```

测试输出：

```text
资产集合bitmask编码: 3
bitmask解码资产: ['asset_a', 'asset_b']
```

该编码方式使资产组合可以用整数快速表示，并可直接作为 `RiskSavingValueFunction.__getitem__(bitmask)` 的输入。

## 量子振幅编码与解码

功能 75 使用 `ValueLoader` 验证金融风险值与量子数据之间的转换。

技术步骤：

1. 计算单资产 ES 风险值。
2. 以最大 ES 为尺度，把风险值归一化到 `[0, 1]`。
3. 使用 `ValueLoader(normalized_values, num_control=1, normalization=False)` 构造受控旋转。
4. 输出量子比特测得 `|1>` 的概率对应归一化金融风险值。
5. 使用 `Statevector(circuit).probabilities([1])[1]` 读回概率。
6. 概率乘以最大 ES，恢复原始风险值。

测试输出：

```text
原始ES: [0.00784318, 0.00769235]
状态向量概率读回并还原 ES: [0.00784318, 0.00769235]
```

读回结果与原始 ES 一致，说明金融值可以进入量子线路并被正确解码。

## 技术结论

功能 75 已完成端到端流程验证：

1. 数据获取：统一接口获取金融价格矩阵。
2. 数据分析：完成收益率、协方差、期望收益和 Shapley ES 风险归因。
3. 金融编码：资产集合通过 bitmask 编码和解码。
4. 量子编码：金融风险值通过 `ValueLoader` 编码为量子振幅概率。
5. 解码恢复：通过 `Statevector` 概率读回并恢复原始 ES 数值。

该功能证明功能 74 接入的数据源可以进入项目已有量化金融分析和量子信息编码链路。

## 技术原理补充：从金融时间序列到量子振幅

端到端流程的核心是把金融时间序列逐层转换为可被量子线路承载的归一化数值。价格矩阵 $P$ 首先被转换为对数收益率

$$
r_{t,i}=\log p_{t,i}-\log p_{t-1,i}.
$$

对数收益率具有时间可加性，适合进入协方差估计和尾部风险计算。协方差矩阵与期望收益分别为

$$
\mu=\mathbb{E}[r_t],\qquad \Sigma=\mathbb{E}[(r_t-\mu)(r_t-\mu)^\top].
$$

风险归因采用 Shapley 分解思想。对资产集合 $N$ 和特征函数 $v(S)$，资产 $i$ 的边际贡献为

$$
\phi_i(v)=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(n-|S|-1)!}{n!}\bigl[v(S\cup\{i\})-v(S)\bigr].
$$

测试中使用 ES 路径验证金融分析结果。若组合损失为 $L=-R$，置信水平为 $\alpha$，历史模拟的预期损失可写为

$$
\mathrm{ES}_{\alpha}(L)=\mathbb{E}\left[L\mid L\ge q_{\alpha}(L)\right].
$$

资产集合 bitmask 编码把集合 $S$ 表示为整数

$$
m(S)=\sum_{i\in S}2^i,
$$

该表示与量子计算中的计算基态 $|b_{n-1}\cdots b_0\rangle$ 一一对应，便于将“资产是否被选择”直接映射为量子寄存器中的二进制比特。

## 振幅编码与概率读回原理

金融风险值 $x_j\ge 0$ 进入量子线路前先归一化为

$$
\tilde{x}_j=\frac{x_j}{x_{\max}},\quad 0\le \tilde{x}_j\le 1.
$$

`ValueLoader` 的受控旋转可理解为对输出量子比特施加

$$
|j\rangle|0\rangle\mapsto |j\rangle\left(\sqrt{1-\tilde{x}_j}|0\rangle+\sqrt{\tilde{x}_j}|1\rangle\right),
$$

因此测得输出位为 $|1\rangle$ 的条件概率就是 $\tilde{x}_j$。从状态向量读回概率后，用

$$
x_j=x_{\max}\Pr(\mathrm{output}=1\mid j)
$$

恢复原始金融风险值。该原理说明本验收不是简单调用 API，而是验证了“金融数值—量子振幅—概率读回—金融数值恢复”的闭环。

