# 算法10：指数追踪算法

## 1. 算法定位

指数追踪算法是面向离散成分股选择的约束保持量子组合构造算法。给定候选资产、行业分组、因子暴露和交易成本，算法选择一个满足行业或风格约束的资产篮子，使组合暴露尽量接近目标指数。

该算法内部使用 transition-Hamiltonian 搜索机制。每个 transition 表示在同一行业或因子组内替换一只资产，因此量子态始终保持行业均衡和组合规模约束。与 penalty-QAOA 相比，指数追踪算法不依赖软惩罚来修复行业约束，而是把可行性写入状态转移结构。

## 2. 数学形式

定义资产选择变量

$$
x_i\in\{0,1\},\qquad i=1,\ldots,n.
$$

`x_i=1` 表示选择第 `i` 只候选资产。设资产因子暴露矩阵为 `A`，目标指数暴露为 `b`，交易或持有成本为 `c_i`。目标函数为

$$
\min_x f(x)
=\|Ax-b\|_2^2+\sum_i c_i x_i.
$$

若候选资产按行业或风格组划分为

$$
S_1,\ldots,S_M,
$$

则行业约束为

$$
\sum_{i\in S_m}x_i=1,\qquad m=1,\ldots,M.
$$

可行指数篮子集合为

$$
\mathcal F_{\mathrm{index}}
=\{x\in\{0,1\}^n:\sum_{i\in S_m}x_i=1,\;m=1,\ldots,M\}.
$$

## 3. 行业内换股 transition

指数追踪算法的 transition 是行业内资产替换。若资产 `u` 和 `v` 属于同一行业组，且当前组合持有 `u`，则可行移动为

$$
d_u=-1,\qquad d_v=+1.
$$

该移动保持行业选择数量：

$$
\sum_{i\in S_m}(x_i+d_i)=1.
$$

对于多个行业组，SFS 递归树先构造每个行业一个资产的组合编码，再通过行业内换股差分生成 `{-1,0,1}` transition basis。这样搜索图的节点就是行业均衡指数篮子，边就是一次可解释换股操作。

## 4. 量子电路实现

电路制备一个初始行业均衡组合

$$
\lvert x_0\rangle.
$$

随后对每个行业内换股 transition `d_l` 施加局部驱动：

1. 在被替换资产和新资产对应量子位上执行 convert；
2. 施加 transition 相位参数 `theta_l`；
3. 反向恢复；
4. 重复所有 transition 和层数。

当电路有 `p` 层、`L` 个行业换股 transition 时，参数量为

$$
pL.
$$

测量结果直接对应一个行业均衡指数篮子。对每个样本，算法计算 tracking error 和成本：

$$
\|Ax-b\|_2^2+\sum_i c_ix_i.
$$

含噪或 baseline 评估中，不满足行业约束的样本按照惩罚目标

$$
f_{\lambda}(x)=f(x)+\lambda\sum_m\left(\sum_{i\in S_m}x_i-1\right)^2
$$

计入误差。

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/index_tracking.py
```

问题实例由

```text
index_tracking_instance()
index_tracking_group_instance()
```

生成，位于：

```text
src/double_quant/application/_rasengan_factories.py
```

核心电路位于：

```text
src/double_quant/algorithm/rasengan/circuit.py
```

`IndexTrackingAlgorithm.build_circuit()` 构建指数追踪算法的行业约束保持搜索电路；`build_baseline_circuit()` 构建同一跟踪误差目标下的 Penalty-QAOA baseline。
第三方测试目录包括 `third/10-Func-10`、`third/20-Func-20`、`third/30-Func-30`、`third/40-Func-40`、`third/51-Perf-10`、`third/63-Perf-22`、`third/73-Perf-32`、`third/129-Perf-50`、`third/139-Perf-60`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。

## 6. Baseline 与优势口径

Penalty-QAOA baseline 在全资产选择空间中搜索：

$$
2^n.
$$

指数追踪算法只在行业均衡篮子集合中搜索：

$$
|\mathcal F_{\mathrm{index}}|=\prod_{m=1}^M |S_m|.
$$

因此它在搜索空间、可行采样率和后处理成本上都有优势。特别是当监管或投资政策要求行业中性时，不满足行业约束的组合不能作为有效指数产品交付；硬约束 transition 能直接减少这类无效样本。

## 7. 验证样例

验证样例包含科技、金融、消费三个行业组，每组选择一只资产。实验得到：

```text
可行解数量 = 8 / 64
最优可行解 = 100101
目标值 = 0.0785
transition moves = 3, 覆盖 8/8
Penalty-QAOA 可行概率 = 0.9845
```

该结果说明指数追踪算法输出的是满足行业约束的成分股篮子，而不是任意资产子集。
