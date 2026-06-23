# 算法1：投资组合优化算法（SAPO/HHL 量子线性求解）

## 1. 算法定位

投资组合优化算法面向均值-方差框架下的连续资产权重求解问题。与把资产持有状态离散化为 $0/1$ 的伊辛或 QUBO 求解不同，该算法保留每个资产的投资权重，用布莱克模型把组合优化转化为线性方程组，再调用 SAPO-style HHL 量子线性求解流程计算权重向量。

该算法的核心目标不是替代成熟经典二次规划求解器，而是在量子线性求解框架内改善投资组合问题的可扩展性和精度：通过约束缩放降低矩阵条件数，通过特征值先验指导 QPE 比特数和受控旋转归一化常数的设置。

## 2. 数学形式

给定 $n$ 个资产的历史预期收益向量 $R$ 和协方差矩阵 $\Sigma$，定义资产权重

$$
\mathbf{w} = (w_1,w_2,\ldots,w_n)^T.
$$

布莱克模型下的组合优化形式为：

$$
\begin{aligned}
\min_{\mathbf{w}}\quad & \mathbf{w}^T \Sigma \mathbf{w},\\
\mathrm{s.t.}\quad & \mathbf{w}^T R = E,\\
& \mathbf{w}^T \mathbf{1} = 1.
\end{aligned}
$$

其中 $E$ 为目标收益，$\mathbf{1}$ 为全 1 向量。引入拉格朗日乘子 $\eta$ 和 $\rho$ 后，可写成线性方程组：

$$
\begin{bmatrix}
0 & 0 & R^T \\
0 & 0 & \mathbf{1}^T \\
R & \mathbf{1} & \Sigma
\end{bmatrix}
\begin{bmatrix}
\eta \\
\rho \\
\mathbf{w}
\end{bmatrix}
=
\begin{bmatrix}
E \\
1 \\
\mathbf{0}
\end{bmatrix}.
$$

为了降低求解难度，SAPO 在等式约束两侧引入缩放系数：

$$
s_1\mathbf{w}^T R=s_1E,\qquad
s_2\mathbf{w}^T\mathbf{1}=s_2.
$$

该变换不改变原优化问题的可行解和最优权重，但可以显著改变线性系统矩阵的条件数。

## 3. 求解方法

采用 SAPO/HHL 量子线性求解。布莱克模型得到的系数矩阵是对称矩阵，满足 HHL 对 Hermitian 输入的要求；右端向量可归一化为量子态，最后再恢复尺度。

SAPO 的关键改造包括：

1. 用约束缩放把高条件数矩阵变成更容易求解的等价矩阵。
2. 当矩阵维度不是 2 的幂时，在右下角补单位矩阵扩展维度。
3. 用 $|\lambda|_{\max}$ 和 $|\lambda|_{\min}$ 估计条件数 $\kappa$。
4. 将输入矩阵缩放为特征值绝对值小于 1 的形式。
5. 根据特征值范围设置 QPE 比特数和受控旋转常数。

QPE 比特数可按误差目标 $\epsilon$ 选取：

$$
n_p
= 2 + \left\lceil
\log_2 \frac{|\lambda|_{\max}}{|\lambda|_{\min}\epsilon}
\right\rceil.
$$

受控旋转的归一化常数取上限：

$$
C=\frac{|\lambda|_{\min}}{2|\lambda|_{\max}}.
$$

HHL 成功测得辅助比特后，向量寄存器中的振幅对应归一化解向量；取解向量中权重部分即得到组合配置。

## 4. 具体实现

实现流程：

1. 获取资产价格序列，计算收益率、历史预期收益 $R$ 和协方差矩阵 $\Sigma$。
2. 根据目标收益 $E$ 构造布莱克模型对应的 KKT 线性系统。
3. 若系统维度不是 2 的幂，则补单位矩阵扩展到最近的 2 幂维度。
4. 使用离线搜索得到的 $s_1$、$s_2$、$s_*$ 执行约束缩放。
5. 估计或计算 $|\lambda|_{\max}$ 与 $|\lambda|_{\min}$。
6. 按 SAPO 参数规则构造 HHL 电路并求解线性系统。
7. 从解向量中截取资产权重，检查预算约束和目标收益约束。
8. 输出每个资产的投资权重。

## 5. Baseline 与优势口径

Baseline 包括三类：

1. 经典均值-方差二次规划求解器。
2. 原始 HHL、Qiskit HHL 和混合 HHL。
3. 将组合优化离散化的伊辛/QUBO 求解。

相对伊辛/QUBO，SAPO/HHL 的优势是保留连续权重，而不是只输出持有或不持有。相对原始 HHL 和通用 HHL 实现，优势来自两层：

$$
\kappa_{\mathrm{raw}} \rightarrow \kappa_{\mathrm{scaled}},
\qquad
\text{generic HHL parameters} \rightarrow \text{eigenvalue-guided parameters}.
$$

因此该算法的优势口径是：在同样面向量子线性求解的条件下，降低条件数、减少电路资源，并提升有限 QPE 比特数下的解向量精度。

## 6. 验证结果

第 3 章在 Nasdaq、NYSE、AMEX 三个真实市场数据集上，对 2 到 6 个资产的组合优化进行了验证。主要结果如下：

```text
数据集 = Nasdaq / NYSE / AMEX
资产数量 = 2-6
复杂度 Θ 降低 = 34.64% - 36.94%
epsilon = 1/8 时量子比特减少 = 39.16%
epsilon = 1/8 时电路深度减少 = 97.12%
epsilon = 1/8 时 CNOT 数减少 = 97.12%
精度提升 = 1.52x vs Qiskit HHL, 1.46x vs hybrid HHL
```

消融实验显示，约束缩放是主要贡献项，可将真实市场矩阵中约 $10^5$ 量级的条件数降低到约 $10$ 量级；特征值估计进一步改善 QPE 资源受限时的求解精度。

对应推导和实验结果见 [main-full.pdf](main-full.pdf) 第 3 章。
