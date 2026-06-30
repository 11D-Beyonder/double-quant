# 算法7：支付与结算系统算法（Rasengan 流动性中性净额结算）

## 1. 算法定位

支付与结算系统算法被建模为批量支付指令选择问题。目标是在满足流动性中性、批次容量和业务优先级约束的条件下，选择一组支付指令进入结算批次。

## 2. 数学形式

定义二元变量

$$
x_i \in \{0,1\}.
$$

`x_i=1` 表示第 `i` 条支付指令进入当前批次。目标为最大化结算优先级，写成最小化：

$$
\min_x f(x) = -\sum_i p_i x_i.
$$

账户/银行净流动性约束为：

$$
\begin{aligned}
A x &= 0,\\
\sum_i x_i &= k.
\end{aligned}
$$

验证样例变量：

```text
x = (p_AB, p_BA, p_BC, p_CB, p_AC, p_CA)
```

约束包括：

$$
\begin{aligned}
5p_{AB}-5p_{BA}+3p_{AC}-3p_{CA} &= 0,\\
-5p_{AB}+5p_{BA}+4p_{BC}-4p_{CB} &= 0,\\
-4p_{BC}+4p_{CB}-3p_{AC}+3p_{CA} &= 0,\\
p_{AB}+p_{BA}+p_{BC}+p_{CB}+p_{AC}+p_{CA} &= 2.
\end{aligned}
$$

## 3. 求解方法

采用 Rasengan。可行态为流动性中性的支付对或支付组合。若线性约束的齐次基础解系不方便直接构造，可使用 SFS 构造互抵支付对的选择树，也可枚举可行支付批次后两两相减生成 transition moves。

## 4. 具体实现

实现流程：

1. 将支付指令编码为二元变量。
2. 建立银行净额矩阵 `A`。
3. 找到一个满足 `Ax=0` 和批次容量的可行支付批次。
4. 构造互抵支付对之间的 transition moves。
5. 用 Rasengan 在可行批次空间中采样。
6. 按结算优先级或流动性占用目标选出最优批次。

## 5. Baseline 与优势口径

Baseline 为 Low 的 QAOA。Penalty 类 QAOA 在流动性守恒约束下容易产生不可结算组合；Rasengan/SFS 直接限制搜索空间为可结算组合，减少无效样本并提升可行率。

## 6. 验证结果

临时实验中：

```text
可行解数量 = 3 / 64
最优可行解 = 110000
目标值 = -14
Rasengan moves = 2, 覆盖 3/3
SFS-paired-payment moves = 2, 覆盖 3/3
Penalty-QAOA 可行概率 = 0.1551
```

对应代码与报告见 `temp/rasengan_sfs_binary_opt`。
