# 算法9：银行网点布局优化算法

## 1. 算法定位

银行网点布局优化算法是面向设施选址问题的 SFS 压缩量子搜索算法。给定候选网点、开设成本和客户需求点，算法选择应开设的网点集合，使开设成本和客户服务成本之和最小。

该算法的核心不是在“网点开设变量 + 客户分配变量”的全组合空间中搜索，而是只把网点开设集合放入量子搜索寄存器；客户分配由 cost oracle 在每个候选网点集合下自动计算最近服务成本。这样可以删除大量 assignment 冗余，符合 main.pdf 中递归分解和 SFS-Grover 的设计思想。

## 2. 数学形式

设候选网点数为 `a`，定义开设变量

$$
y_j\in\{0,1\},\qquad j=1,\ldots,a.
$$

`y_j=1` 表示开设第 `j` 个候选网点。客户需求点记为 `i=1,\ldots,m`，开设成本为 `f_j`，服务成本为 `d_ij`。在给定网点集合 `y` 后，客户分配由

$$
\operatorname{assign}(i;y)
=\arg\min_{j:y_j=1}d_{ij}
$$

确定。目标函数为

$$
\min_y C(y)
=\sum_{j=1}^{a} f_jy_j+\sum_{i=1}^{m}\min_{j:y_j=1}d_{ij}.
$$

典型数量约束为

$$
\sum_{j=1}^{a}y_j=K.
$$

可行设施集合为

$$
\mathcal Y=\{y\in\{0,1\}^{a}:\mathbf 1^\top y=K\}.
$$

## 3. SFS 压缩与 oracle 设计

银行网点布局优化算法使用 SFS 递归树生成满足数量、预算或区域覆盖约束的网点集合。压缩初态为

$$
\lvert\psi_{\mathcal Y}\rangle
=\frac{1}{\sqrt{|\mathcal Y|}}\sum_{y\in\mathcal Y}\lvert \mathrm{code}(y)\rangle.
$$

成本 oracle 不把客户分配变量显式量子化，而是在可逆计算中执行：

1. 根据 `y` 找到每个客户最近的已开网点；
2. 累加服务成本和开设成本；
3. 与阈值 `B` 比较；
4. 对满足

   $$
   C(y)\le B
   $$

   的网点集合施加相位标记。

随后在压缩寄存器上执行 diffusion，实现低成本设施集合的幅度放大。

## 4. 量子电路实现

工程实现中，`build_sfs_grover_circuit()` 用

$$
q=\lceil a/2\rceil
$$

个压缩量子位表示 SFS 后的设施集合编码；普通 Grover baseline 使用完整 `a` 个候选网点量子位。每轮幅度放大包含：

1. 压缩寄存器上的 Hadamard 初态；
2. 成本阈值 oracle；
3. 多控相位翻转；
4. diffusion 反射；
5. 测量并解码为候选网点集合。

复杂度报告中，银行网点布局优化算法的理论查询量按

$$
\sqrt{|\mathcal Y|}
$$

计；与 SETH 口径下的经典 `2^a` 设施集合枚举相比，对应量子搜索项为

$$
\sqrt{2^a}=2^{a/2}.
$$

## 5. 具体实现入口

应用封装位于：

```text
src/double_quant/application/branch_location.py
```

核心电路位于：

```text
src/double_quant/algorithm/grover/circuit.py
```

`BranchLocationAlgorithm.build_circuit()` 生成银行网点布局优化算法的 SFS 压缩幅度放大电路；`build_baseline_circuit()` 生成普通全空间 Grover 量子 baseline。
第三方测试目录包括 `third/9-Func-9`、`third/19-Func-19`、`third/29-Func-29`、`third/39-Func-39`、`third/50-Perf-9`、`third/62-Perf-21`、`third/72-Perf-31`、`third/128-Perf-49`、`third/138-Perf-59`，分别对应算法技术报告、计算操作数、求解空间大小、精度与量子电路参数关系、不少于多项式级别加速、精度提升40%及以上、复杂度降低50%及以上、含噪计算误差降低40%及以上和含噪量子计算复杂度降低50%及以上。

## 6. Baseline 与优势口径

量子 baseline 为普通 Grover，在完整候选网点空间 `2^a` 上搜索。经典 baseline 为设施选址精确枚举或 SETH 口径下的 `2^a` 规模下界讨论。我们的银行网点布局优化算法通过两个机制获得优势：

$$
2^a\rightarrow |\mathcal Y|\rightarrow \sqrt{|\mathcal Y|}.
$$

第一步来自 SFS 删除不满足数量、预算或区域规则的设施集合；第二步来自幅度放大。与经典 `2^a` 口径比较时，可表述为 `2^{a/2}` 查询量级的量子搜索收益。

## 7. 验证样例

小规模设施选址样例得到：

```text
全空间候选 = 8
SFS 可行候选 = 3
最优网点集合 = 101
目标值 = 18
理论幅度放大迭代 = 2
搜索空间压缩倍数 = 2.67
```

该样例说明银行网点布局优化算法直接搜索网点集合，并由 oracle 隐式处理客户分配，而不是枚举所有网点-客户 assignment。
