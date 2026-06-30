# Func-5 去中心化金融管理算法——算法技术报告

## 报告定位

本报告对应算法功能交付项，按“我们提出的金融应用算法”口径说明数学形式、内部量子实现机制、baseline 与实现入口。Shor 周期发现、SFS-Grover 幅度放大和 Rasengan 型 transition-Hamiltonian 只作为各应用算法内部的电路构件展开。

## 源码位置

- `src/double_quant/application/`
- `src/double_quant/algorithm/grover/`

## 对应实验测试项

- 计算操作数：`Func-15`
- 求解空间大小：`Func-25`
- 精度与量子电路参数关系：`Func-35`
- 不少于多项式级别加速：`Perf-5`
- 精度提升40%及以上：`Perf-17`
- 复杂度降低50%及以上：`Perf-27`
- 含噪误差/精度改善：`Perf-45`
- 含噪复杂度降低：`Perf-55`

## 算法数学形式与内部实现说明

# 算法5：去中心化金融管理算法

## 1. 算法定位

去中心化金融管理算法是面向 DeFi 风险处置和收益管理的量子策略搜索算法。算法输入为一组候选管理动作，例如再平衡、提高抵押率、暂停资金池、迁移激励、调整清算阈值等；输出为满足风险预算、互斥规则和动作数量限制的最优动作组合。

该算法内部使用 SFS 压缩可行策略空间，并在压缩后的寄存器上实现阈值型幅度放大。这里的 Grover 不是简单地在全空间 `2^n` 上搜索，而是作为去中心化金融管理算法的量子搜索核心：可行空间由业务约束先行构造，oracle 只评价可部署的策略。

## 2. 数学形式

定义二元动作变量

$$
x_i\in\{0,1\},\qquad i=1,\ldots,n,
$$

其中 `x_i=1` 表示执行第 `i` 个 DeFi 管理动作。以风险收益权衡为例，目标函数写成

$$
\min_x f(x)
=-\sum_{i=1}^n b_i x_i+\sum_{i<j} c_{ij}x_i x_j,
$$

其中 `b_i` 表示动作收益或风险缓释收益，`c_ij` 表示动作之间的冲突、叠加成本或协同惩罚。

典型约束包括

$$
\begin{aligned}
\sum_i x_i &= K,\\
\sum_i \mathrm{risk}_i x_i &\le R,\\
x_u+x_v &\le 1,\qquad (u,v)\in\mathcal I.
\end{aligned}
$$

其中 `K` 为本轮允许执行的动作数量，`R` 为风险预算，`\mathcal I` 为互斥动作集合。

## 3. 可行空间压缩机制

算法先构造满足约束的可行策略集合

$$
\mathcal F=\{x\in\{0,1\}^n:\; Ax\le b,\; Ex=h\}.
$$

SFS 组件把原始动作变量映射到压缩寄存器：

$$
z\in\{0,1\}^{q},\qquad q=\lceil n/2\rceil,
$$

并通过业务规则或递归选择树只生成可行动作组合。对应初态为

$$
\lvert\psi_{\mathcal F}\rangle
=\frac{1}{\sqrt{|\mathcal F|}}\sum_{x\in\mathcal F}\lvert \mathrm{code}(x)\rangle.
$$

该设计把搜索范围从全动作空间

$$
2^n
$$

压缩到

$$
|\mathcal F|\le 2^q,
$$

从而减少 oracle 调用和无效采样。

## 4. 量子搜索实现

去中心化金融管理算法的量子电路包括：

1. 压缩寄存器制备：对 `q` 个搜索量子位施加 Hadamard 门，准备压缩策略编码；
2. 策略解码和目标 oracle：可逆计算 `f(x)`，并标记满足当前阈值的策略

   $$
   f(x)\le B;
   $$

3. 相位翻转 oracle：对被标记策略施加相位 `-1`；
4. 压缩空间 diffusion：围绕压缩空间均值反射；
5. 自适应阈值更新：根据测量到的策略成本更新阈值 `B`，重复幅度放大。

在工程实现中，`build_sfs_grover_circuit()` 使用压缩量子位数 `ceil(n/2)` 构造搜索电路；baseline 使用同样迭代数的普通 Grover 电路，但搜索寄存器为完整 `n` 个逻辑变量。

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/defi_management.py
```

核心电路位于：

```text
src/double_quant/algorithm/grover/circuit.py
```

`DefiManagementAlgorithm.build_circuit()` 构建 DeFi 管理算法的 SFS 压缩幅度放大电路；`build_baseline_circuit()` 构建普通全空间 Grover 量子 baseline。
第三方测试目录包括 `third/5-Func-5`、`third/15-Func-15`、`third/25-Func-25`、`third/35-Func-35`、`third/46-Perf-5`、`third/58-Perf-17`、`third/68-Perf-27`、`third/124-Perf-45`、`third/134-Perf-55`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。

## 6. Baseline 与优势口径

量子 baseline 是普通 Grover：直接在 `2^n` 个动作组合上做 oracle 标记和 diffusion。我们的去中心化金融管理算法先将动作空间压缩为业务可行集合，再做幅度放大，因此复杂度口径为

$$
2^n\quad \longrightarrow\quad |\mathcal F|\quad \longrightarrow\quad \sqrt{|\mathcal F|}.
$$

优势来自两部分：一是 SFS 删除风险预算和互斥规则下的无效动作组合；二是幅度放大在可行集合上提供平方级搜索收益。

## 7. 验证样例

示例动作向量为

```text
x = (rebalance, raise_collateral, pause_pool, incentive_shift)
```

在动作数量、风险预算和互斥约束下，实验样例得到：

```text
全空间候选 = 16
可行策略候选 = 4
最优策略 = 1010
目标值 = -14.5
理论幅度放大迭代 = 2
搜索空间压缩倍数 = 4
```

该样例说明算法输出的是一组可部署的 DeFi 管理动作，而不是全空间中可能违反业务规则的二进制串。

