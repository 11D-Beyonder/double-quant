# 算法9：银行网点布局优化算法（Grover/SFS 设施选址）

## 1. 算法定位

银行网点布局优化算法对应设施选址问题。给定候选网点和客户需求点，选择开设哪些网点，使开设成本与客户服务成本之和最小。该问题采用 main.pdf 中的 SFS/Grover 思路：先把搜索寄存器压缩为设施集合，再由 cost oracle 决定客户分配。

## 2. 数学形式

定义网点开设变量

$$
y_j \in \{0,1\},\qquad j=1,\ldots,a.
$$

`y_j=1` 表示开设第 `j` 个候选网点。在非负开设成本和服务成本下，客户分配可由 oracle 决定：

$$
\operatorname{assign}(i)
= \arg\min_{j:\,y_j=1} d_{ij}.
$$

目标函数为：

$$
\min_y C(y)
= \sum_j f_j y_j + \sum_i \min_{j:\,y_j=1} d_{ij}.
$$

约束示例：

$$
\sum_j y_j = 2.
$$

## 3. 求解方法

采用 Grover/SFS。SFS 先准备满足网点数量约束的设施集合叠加态：

$$
\lvert \psi\rangle
= \frac{1}{\sqrt{|Y|}}\sum_{y\in Y}\lvert y\rangle.
$$

Grover oracle 可逆计算 `C(y)` 并标记低于阈值的候选集合。通过阈值搜索逐步找到最优设施集合。

## 4. 具体实现

实现流程：

1. 建立候选网点集合、开设成本 `f_j` 和服务成本 `d_ij`。
2. 只保留满足预算/数量约束的设施集合寄存器。
3. 对每个设施集合，由 cost oracle 计算客户到最近已开网点的服务成本。
4. Grover 阈值搜索放大低成本设施集合。
5. 输出最优网点布局。

## 5. Baseline 与优势口径

Baseline 为经典精确枚举或直接全空间 Grover。main.pdf 的关键口径是：设施选址可把 assignment 冗余交给 cost oracle，Grover 只在设施集合上搜索，从经典 `2^a` 量级降到 Grover 的 `2^{a/2}` 查询量级，并通过 SFS 删除不可行/冗余寄存器。

## 6. 验证结果

临时实验中：

```text
全空间候选 = 8
SFS/Grover 可行候选 = 3
最优网点集合 = 101
目标值 = 18
Grover 理论迭代 = 2
搜索空间压缩倍数 = 2.67
```

对应代码与报告见 `temp/shor_grover_remaining`。
