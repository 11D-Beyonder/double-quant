# 算法6：反欺诈监测算法（Rasengan 倒量交易闭环识别）

## 1. 算法定位

反欺诈监测算法被建模为可疑交易图上的闭环交易组合选择问题。倒量交易、循环转账和账户间资金空转通常表现为流入流出守恒的交易环，因此适合转化为约束二元优化，并用 Rasengan 在可行闭环空间内搜索异常分数最高的组合。

## 2. 数学形式

给定可疑有向交易边集合 `E`，定义

$$
x_e \in \{0,1\},\qquad e \in E.
$$

`x_e=1` 表示选择交易边 `e`。目标为最大化异常分数，写成最小化：

$$
\min_x f(x) = -\sum_{e\in E}s_e x_e.
$$

约束为交易闭环和规模限制：

$$
\begin{aligned}
B x &= 0,\\
\sum_{e\in E}x_e &= k.
\end{aligned}
$$

其中 `B` 为账户-交易关联矩阵，`Bx=0` 表示每个账户节点流入流出平衡。

验证样例变量：

```text
x = (t_AB, t_BC, t_CA, t_AD, t_DC)
```

目标函数：

$$
\min\; -7t_{AB}-5t_{BC}-9t_{CA}-3t_{AD}-8t_{DC}.
$$

## 3. 求解方法

采用 Rasengan。可行态是满足流守恒和边数约束的闭合交易环。transition move 可以由两种方式构造：

1. 线性约束 `Bx=0` 的齐次解/环空间基。
2. SFS 或局部环替换构造；小规模测试可枚举可行闭环，两两相减得到 `{-1,0,1}` move。

## 4. 具体实现

实现流程：

1. 建立账户-交易关联矩阵 `B`。
2. 枚举或构造初始可行闭环 `x0`。
3. 构造环替换 transition moves。
4. 用 Rasengan transition Hamiltonian 在可行闭环空间内扩展。
5. 对采样到的可行解计算异常目标函数。
6. 输出异常分数最高的闭环交易组合。

## 5. Baseline 与优势口径

Baseline 为 Low 的 QAOA 或 penalty-QAOA。Penalty-QAOA 容易采到不满足流守恒的交易组合；Rasengan 只在可行闭环空间内移动，可行率更高，后处理无效样本更少。

## 6. 验证结果

临时实验中：

```text
可行解数量 = 2 / 32
最优可行解 = 11100
目标值 = -21
Rasengan moves = 1, 覆盖 2/2
SFS-flow-cycle moves = 1, 覆盖 2/2
Penalty-QAOA 可行概率 = 0.1099
```

对应代码与报告见 `temp/rasengan_sfs_binary_opt`。
