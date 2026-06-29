# 算法7：支付与结算系统算法

## 1. 算法定位

支付与结算系统算法是面向批量支付指令筛选的流动性中性量子搜索算法。它在一个结算窗口内选择一组支付指令，使结算优先级、费用或时效目标最优，同时保证各参与方净流动性变化为零或满足预设限额。

该算法内部采用约束保持 transition-Hamiltonian 搜索机制。与把约束写成 penalty 的 QAOA 不同，支付与结算系统算法先构造互抵支付对或净额中性支付块，再在这些可结算批次之间做局部替换。因此，理想电路采样结果天然是可结算批次。

## 2. 数学形式

定义支付指令变量

$$
x_i\in\{0,1\},\qquad i=1,\ldots,n.
$$

`x_i=1` 表示第 `i` 条支付指令进入当前清算批次。目标可写成

$$
\min_x f(x)
=-\sum_i p_i x_i+\sum_{i<j} h_{ij}x_i x_j,
$$

其中 `p_i` 为结算优先级或业务价值，`h_ij` 表示同批处理冲突、排队风险或流动性占用惩罚。

设 `A` 为参与方-支付指令净额矩阵，则流动性中性约束为

$$
Ax=0.
$$

批次容量约束为

$$
\sum_i x_i=k.
$$

可行结算集合为

$$
\mathcal F_{\mathrm{pay}}
=\{x\in\{0,1\}^n: Ax=0,\;\mathbf 1^\top x=k\}.
$$

## 3. 支付块 transition 设计

支付与结算系统算法的关键是把业务上天然互抵的支付指令组织为可替换块。例如一组互抵支付对满足

$$
x_{\mathrm{out}}-x_{\mathrm{return}}=0.
$$

若每个支付块中只能选择一个互抵支付对，则约束可写为

$$
\sum_{o\in\mathcal O_b} z_{b,o}=1,
$$

其中 `z_{b,o}` 表示第 `b` 个支付块选择第 `o` 个互抵支付对。一个 transition 会移除当前互抵对并加入另一个互抵对：

$$
d=(-1,-1,+1,+1)
$$

或其多块组合形式。该 transition 满足

$$
Ad=0,\qquad \mathbf 1^\top d=0.
$$

因此每次局部变换都保持净额中性和批次容量。

## 4. 量子电路实现

电路从初始可结算批次

$$
\lvert x_0\rangle
$$

开始。对每个互抵支付 transition `d_l`，电路执行：

1. 在 transition 支持集上做基变换，把“当前支付对”转换为相位可控模式；
2. 对该 transition 施加独立相位参数 `theta_l`；
3. 做反向基变换，回到支付指令编码；
4. 多层重复后测量支付批次。

对 `p` 层、`L` 个 transition 的电路，transition 参数量为

$$
pL.
$$

目标函数和约束信息进入两个位置：可行性通过 transition 结构硬编码，结算优先级通过相位目标和采样后评价体现。含噪情况下，不可行测量串用

$$
f_{\lambda}(x)=f(x)+\lambda\|Ax\|_2^2
$$

计入误差。

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/payment_settlement.py
```

问题实例由

```text
payment_settlement_instance()
payment_settlement_block_instance()
```

生成，位于：

```text
src/double_quant/application/_rasengan_factories.py
```

量子电路由 `build_rasengan_circuit()` 构造；量子 baseline 由 `build_penalty_qaoa_circuit()` 构造。
第三方测试目录包括 `third/7-Func-7`、`third/17-Func-17`、`third/27-Func-27`、`third/37-Func-37`、`third/48-Perf-7`、`third/60-Perf-19`、`third/70-Perf-29`、`third/126-Perf-47`、`third/136-Perf-57`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。

## 6. Baseline 与优势口径

Penalty-QAOA baseline 在全二进制空间中搜索：

$$
2^n.
$$

支付与结算系统算法只在可结算批次集合中搜索：

$$
|\mathcal F_{\mathrm{pay}}|.
$$

因此复杂度和精度优势来自可行空间压缩和硬约束保持。该口径特别适合支付系统，因为不可行批次不仅是数学上无效，也无法进入真实清算流程。

## 7. 验证样例

示例变量为

```text
x = (p_AB, p_BA, p_BC, p_CB, p_AC, p_CA)
```

约束为

$$
\begin{aligned}
5p_{AB}-5p_{BA}+3p_{AC}-3p_{CA} &= 0,\\
-5p_{AB}+5p_{BA}+4p_{BC}-4p_{CB} &= 0,\\
-4p_{BC}+4p_{CB}-3p_{AC}+3p_{CA} &= 0,\\
p_{AB}+p_{BA}+p_{BC}+p_{CB}+p_{AC}+p_{CA} &= 2.
\end{aligned}
$$

实验样例得到：

```text
可行解数量 = 3 / 64
最优可行解 = 110000
目标值 = -14
transition moves = 2, 覆盖 3/3
Penalty-QAOA 可行概率 = 0.1551
```

该样例说明算法输出的是净额中性的可结算支付批次。
