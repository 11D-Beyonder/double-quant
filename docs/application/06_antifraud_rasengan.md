# 算法6：反欺诈监测算法

## 1. 算法定位

反欺诈监测算法是面向倒量交易、循环转账和账户间资金空转的约束保持量子搜索算法。算法把可疑交易网络中的交易边选择问题建模为闭合资金流组合搜索：只有满足账户流入流出平衡的交易组合才进入量子搜索空间，输出为异常分数最高的闭环交易集合。

该算法内部使用 Rasengan 型 transition-Hamiltonian 电路，但报告主语是反欺诈监测算法本身。该内部量子构件具体表现为：从一个合法闭环交易组合出发，用闭环替换 transition 在可行空间内移动，目标相位由异常分数给出，采样结果天然满足资金守恒约束。

## 2. 数学形式

给定可疑有向交易边集合 `E`，定义二元变量

$$
x_e\in\{0,1\},\qquad e\in E.
$$

`x_e=1` 表示交易边 `e` 被纳入本次倒量交易闭环。设 `s_e` 为交易异常分数，则最大化异常分数可写为最小化

$$
\min_x f(x)=-\sum_{e\in E}s_e x_e+\sum_{e<e'} q_{ee'}x_ex_{e'}.
$$

其中 `q_ee'` 可表示交易边之间的时间一致性、账户关联或金额匹配惩罚。

令 `B` 为账户-交易关联矩阵，出边记为 `+1`，入边记为 `-1`。闭环交易约束为

$$
Bx=0.
$$

同时可加入环规模约束：

$$
\sum_{e\in E}x_e=k.
$$

因此可行域为

$$
\mathcal F_{\mathrm{fraud}}
=\{x\in\{0,1\}^{|E|}: Bx=0,\; \mathbf 1^\top x=k\}.
$$

## 3. 约束保持 transition 构造

反欺诈监测算法不把流守恒作为 penalty 后处理，而是直接构造闭环空间内的局部移动。对任意两个可行闭环组合 `x` 和 `y`，差分

$$
d=y-x
$$

满足

$$
Bd=0,\qquad \mathbf 1^\top d=0.
$$

在二元可行解之间，`d` 的分量天然属于

$$
d_i\in\{-1,0,1\}.
$$

实现中有两种 transition 来源：

1. 对线性约束矩阵做行约简，寻找满足 `Ad=0` 的 `{-1,0,1}` 基；
2. 对可疑闭环进行 SFS/枚举构造，用闭环替换生成 transition，例如“移除第一条闭环、加入另一条闭环”。

这样每个 transition 都保持 `Bx=0` 和规模约束，量子态演化不会离开可疑闭环空间。

## 4. 量子电路实现

电路从一个可行闭环 `x_0` 开始：

$$
\lvert x_0\rangle.
$$

对每个 transition `d_l`，算法构造一个局部驱动分量。其支持集为

$$
\operatorname{supp}(d_l)=\{i:d_{l,i}\ne 0\}.
$$

电路先把该支持集上的源模式转换到可控相位作用基，再施加 transition 相位

$$
\theta_l,
$$

最后反向恢复。多比特 transition 使用辅助量子位分解多控相位门。若共有 `L` 个 transition 和 `p` 层，则参数数为

$$
pL.
$$

这是当前实现中区别于早期简化版本的关键点：不是所有 transition 共享一个参数，而是每个 transition 在每层都有独立参数。

测量后只需在可行闭环集合内计算目标函数值；不可行样本在理想电路中不会产生。含噪实验中若测得不可行串，则按惩罚目标

$$
f_{\lambda}(x)=f(x)+\lambda\|Ax-b\|_2^2
$$

计入精度指标。

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/antifraud_monitoring.py
```

问题工厂位于：

```text
src/double_quant/application/_rasengan_factories.py
```

核心电路位于：

```text
src/double_quant/algorithm/rasengan/circuit.py
src/double_quant/algorithm/rasengan/linear_system.py
```

`AntifraudMonitoringAlgorithm.build_circuit()` 构建反欺诈闭环监测算法电路，输入包括 `LinearConstraintBinaryProblem`、闭环 `transition_basis` 和初始可行闭环 `feasible_state`。baseline 为同一目标和同一约束的 Penalty-QAOA 电路，其 penalty 项由 `build_penalty_qaoa_circuit()` 实现。

## 6. Baseline 与优势口径

量子 baseline 是 Penalty-QAOA：它从全空间叠加态出发，把流守恒和环规模约束写入惩罚项。因此 baseline 的搜索空间为

$$
2^{|E|}.
$$

反欺诈监测算法的搜索空间为闭环可行集合

$$
|\mathcal F_{\mathrm{fraud}}|.
$$

优势来自三个方面：第一，可行率高，不把大量 shots 浪费在不守恒交易组合上；第二，transition 数通常随闭环候选数量增长，而不是随全二进制空间增长；第三，目标评估只在合规闭环上进行，便于给出可解释的异常路径。

## 7. 验证样例

样例变量为

```text
x = (t_AB, t_BC, t_CA, t_AD, t_DC)
```

目标函数为

$$
\min\; -7t_{AB}-5t_{BC}-9t_{CA}-3t_{AD}-8t_{DC}.
$$

实验样例得到：

```text
可行解数量 = 2 / 32
最优可行解 = 11100
目标值 = -21
transition moves = 1, 覆盖 2/2
Penalty-QAOA 可行概率 = 0.1099
```

这表明反欺诈监测算法搜索的是资金守恒闭环，而不是任意可疑交易边组合。

第三方测试目录包括 `third/6-Func-6`、`third/16-Func-16`、`third/26-Func-26`、`third/36-Func-36`、`third/47-Perf-6`、`third/59-Perf-18`、`third/69-Perf-28`、`third/125-Perf-46`、`third/135-Perf-56`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。
