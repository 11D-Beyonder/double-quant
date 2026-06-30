# 算法5：去中心化金融管理算法（Grover/SFS 约束策略搜索）

## 1. 算法定位

去中心化金融管理算法被建模为有限策略集合中的约束搜索问题。策略变量表示是否执行某类 DeFi 管理动作，例如再平衡、提高抵押率、暂停资金池、激励迁移等。先用业务约束或 SFS 构造可行策略子空间，再用 Grover 阈值搜索寻找最优策略。

## 2. 数学形式

定义二元变量

$$
x_i \in \{0,1\},\qquad i=1,\ldots,n.
$$

其中 `x_i=1` 表示执行第 `i` 个管理动作。示例变量为：

```text
x = (rebalance, raise_collateral, pause_pool, incentive_shift)
```

目标函数采用收益最大化转为最小化：

$$
\min_x f(x)
= -\sum_i b_i x_i + \sum_{i<j} c_{ij}x_i x_j.
$$

示例约束为：

$$
\begin{aligned}
\sum_i x_i &= 2,\\
\sum_i \mathrm{risk}_i x_i &\le 5,\\
x_{\mathrm{pause}} + x_{\mathrm{incentive}} &\le 1.
\end{aligned}
$$

## 3. 求解方法

采用 Grover/SFS。SFS 或业务规则先构造满足动作数量、风险预算和互斥规则的可行策略集合 `F`，然后在 `F` 上运行 Grover-style threshold search：

$$
\lvert \psi_F\rangle
= \frac{1}{\sqrt{|F|}}\sum_{x\in F}\lvert x\rangle.
$$

Oracle 标记满足当前阈值 `f(x) <= B` 的策略，Grover 放大被标记策略的振幅。

## 4. 具体实现

实现流程：

1. 将 DeFi 管理动作编码为二元变量。
2. 根据风险预算、互斥关系、动作数量生成可行策略集合。
3. SFS 构造只支撑可行策略的初态。
4. 构造可逆目标函数 oracle，计算 `f(x)` 并与阈值比较。
5. 用 Grover 阈值搜索或自适应 Grover 搜索寻找最优策略。
6. 输出最优动作组合和目标值。

## 5. Baseline 与优势口径

Baseline 为 Low 的 Grover 或直接全空间搜索。我们的口径是先用约束/SFS 将搜索寄存器从全空间 `2^n` 压缩到可行策略集合 `|F|`，再执行 Grover 搜索。因此优势来自两层：

$$
2^n \rightarrow |F| \rightarrow \sqrt{|F|}.
$$

## 6. 验证结果

临时实验中：

```text
全空间候选 = 16
可行策略候选 = 4
最优策略 = 1010
目标值 = -14.5
Grover 理论迭代 = 2
搜索空间压缩倍数 = 4
```

对应代码与报告见 `temp/shor_grover_remaining`。
