# 算法8：贷款发放决策算法（Rasengan 离散特征选择/数据降维）

## 1. 算法定位

贷款发放决策算法被建模为分组离散特征选择问题。它不是连续 PCA/SVD 降维，而是在信用、收入、行为等特征组中选择有限个可解释特征，使预测收益高、冗余低且满足合规约束。

## 2. 数学形式

定义二元变量

$$
x_j \in \{0,1\}.
$$

`x_j=1` 表示保留第 `j` 个贷款决策特征。目标函数为：

$$
\min_x f(x)
= -\sum_j g_j x_j + \sum_{i<j} r_{ij}x_i x_j.
$$

其中 `g_j` 为预测收益，`r_ij` 为冗余惩罚。分组约束为：

$$
\sum_{j\in G_m}x_j = 1,\qquad m=1,\ldots,M.
$$

验证样例分组：

$$
\begin{aligned}
x_{\mathrm{credit\_depth}}+x_{\mathrm{credit\_recent}} &= 1,\\
x_{\mathrm{income\_level}}+x_{\mathrm{income\_stability}} &= 1,\\
x_{\mathrm{behavior\_spend}}+x_{\mathrm{behavior\_delay}} &= 1.
\end{aligned}
$$

## 3. 求解方法

采用 Rasengan。分组 one-hot 约束天然适合构造局部 transition move：每次在同一特征组内把一个特征从 1 换成 0，同时把另一个特征从 0 换成 1。该 move 为 `{-1,0,1}` 向量，且严格保持所有分组约束。

## 4. 具体实现

实现流程：

1. 对候选特征按信用、收入、行为等业务维度分组。
2. 每组构造 one-hot 约束。
3. 由任意每组一个特征的组合生成初始可行解。
4. 构造组内 swap transition moves。
5. Rasengan 在可行特征组合空间中搜索。
6. 用预测收益与冗余惩罚计算目标值，输出最优特征子集。

## 5. Baseline 与优势口径

Baseline 为 Low 的 QAOA。Rasengan 的优势是约束可行性强：任意 transition move 均保持每组恰选一个特征，因此无需依赖 penalty 项惩罚不合规组合。

## 6. 验证结果

临时实验中：

```text
可行解数量 = 8 / 64
最优可行解 = 011010
目标值 = -21
Rasengan moves = 3, 覆盖 8/8
SFS-one-hot-product moves = 3, 覆盖 8/8
Penalty-QAOA 可行概率 = 0.9734
```

对应代码与报告见 `temp/rasengan_sfs_binary_opt`。
