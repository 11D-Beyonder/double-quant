# Func-8 贷款发放决策算法——算法技术报告

## 报告定位

本报告对应算法功能交付项，按“我们提出的金融应用算法”口径说明数学形式、内部量子实现机制、baseline 与实现入口。Shor 周期发现、SFS-Grover 幅度放大和 Rasengan 型 transition-Hamiltonian 只作为各应用算法内部的电路构件展开。

## 源码位置

- `src/double_quant/application/`
- `src/double_quant/algorithm/rasengan/`

## 对应实验测试项

- 计算操作数：`Func-18`
- 求解空间大小：`Func-28`
- 精度与量子电路参数关系：`Func-38`
- 不少于多项式级别加速：`Perf-8`
- 精度提升40%及以上：`Perf-20`
- 复杂度降低50%及以上：`Perf-30`
- 含噪误差/精度改善：`Perf-48`
- 含噪复杂度降低：`Perf-58`

## 算法数学形式与内部实现说明

# 算法8：贷款发放决策算法

## 1. 算法定位

贷款发放决策算法是面向授信审批模型的离散特征降维算法。它不是连续 PCA 或 SVD，而是在信用历史、收入稳定性、消费行为、负债结构等业务特征组中选择少量可解释特征，使预测收益高、冗余低，并满足合规和可解释性约束。

该算法内部使用约束保持 transition-Hamiltonian 搜索器。每个 transition 表示同一特征组内的一次替换，因此算法在整个量子演化过程中始终保持“每组选择一个特征”的可解释特征组合结构。

## 2. 数学形式

定义特征选择变量

$$
x_j\in\{0,1\},\qquad j=1,\ldots,n.
$$

`x_j=1` 表示保留第 `j` 个贷款决策特征。设 `g_j` 为特征的预测收益，`r_ij` 为冗余或共线性惩罚，目标函数为

$$
\min_x f(x)
=-\sum_j g_jx_j+\sum_{i<j}r_{ij}x_ix_j.
$$

将候选特征划分为 `M` 个业务组：

$$
G_1,\ldots,G_M.
$$

每个业务组选择一个代表特征：

$$
\sum_{j\in G_m}x_j=1,\qquad m=1,\ldots,M.
$$

可行特征集合为

$$
\mathcal F_{\mathrm{loan}}
=\{x\in\{0,1\}^n:\sum_{j\in G_m}x_j=1,\;m=1,\ldots,M\}.
$$

## 3. 分组特征 transition 设计

贷款发放决策算法的 transition 是组内 swap。若特征 `u` 和 `v` 属于同一业务组，且当前选择 `u`，则一个可行 transition 为

$$
d_u=-1,\qquad d_v=+1,\qquad d_i=0\;(i\ne u,v).
$$

该 transition 保持组内 one-hot 约束：

$$
\sum_{j\in G_m}(x_j+d_j)=1.
$$

对所有业务组构造类似 transition，即可形成一个可连通的可行特征组合图。对于非均匀组大小，SFS 递归树先给出每组候选特征选择路径，再由相邻选择差分生成 `{-1,0,1}` transition。

## 4. 量子电路实现

电路从一个每组恰选一个特征的初始态

$$
\lvert x_0\rangle
$$

开始。对每个组内替换 transition `d_l`：

1. 在两个或多个受影响特征量子位上执行 convert 操作；
2. 施加 transition 相位 `theta_l`；
3. 反向恢复编码；
4. 进入下一 transition 或下一层。

当共有 `L` 个组内替换 transition 和 `p` 层时，参数量为

$$
pL.
$$

目标函数中的预测收益和冗余惩罚用于相位优化和采样后评价；约束由 transition 硬保持。Penalty-QAOA baseline 则把 one-hot 约束写为

$$
\lambda\sum_m\left(\sum_{j\in G_m}x_j-1\right)^2
$$

并在全空间采样。

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/loan_decision.py
```

问题实例由

```text
loan_feature_instance()
loan_feature_group_instance()
```

生成，位于：

```text
src/double_quant/application/_rasengan_factories.py
```

核心电路位于：

```text
src/double_quant/algorithm/rasengan/circuit.py
```

`LoanDecisionAlgorithm.build_circuit()` 构造贷款发放决策算法的约束保持特征选择电路；`build_baseline_circuit()` 构造同一目标函数下的 Penalty-QAOA baseline。
第三方测试目录包括 `third/8-Func-8`、`third/18-Func-18`、`third/28-Func-28`、`third/38-Func-38`、`third/49-Perf-8`、`third/61-Perf-20`、`third/71-Perf-30`、`third/127-Perf-48`、`third/137-Perf-58`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。

## 6. Baseline 与优势口径

Penalty-QAOA baseline 的搜索空间为

$$
2^n.
$$

贷款发放决策算法的搜索空间为

$$
|\mathcal F_{\mathrm{loan}}|=\prod_{m=1}^M |G_m|.
$$

当每个组的候选特征数量远小于所有二元组合时，该空间显著小于 `2^n`。更重要的是，算法的每个测量候选都对应可解释的“每组一个特征”方案，符合贷款审批模型对可解释性和合规审查的要求。

## 7. 验证样例

验证样例包括信用、收入、行为三个特征组：

$$
\begin{aligned}
x_{\mathrm{credit\_depth}}+x_{\mathrm{credit\_recent}} &= 1,\\
x_{\mathrm{income\_level}}+x_{\mathrm{income\_stability}} &= 1,\\
x_{\mathrm{behavior\_spend}}+x_{\mathrm{behavior\_delay}} &= 1.
\end{aligned}
$$

实验样例得到：

```text
可行解数量 = 8 / 64
最优可行解 = 011010
目标值 = -21
transition moves = 3, 覆盖 8/8
Penalty-QAOA 可行概率 = 0.9734
```

该结果说明贷款发放决策算法搜索的是可解释特征子集，而不是任意二进制特征组合。

