# 81 编程框架代码自优化功能测试（Func-48） 技术报告

## 技术实现

代码自优化功能实现为独立编译优化过程，位于 `src/double_quant/programming/optimizer.py`。该模块不侵入 `Expression`、`ConstraintExpression` 或 `DecisionProgram` 的核心建模逻辑，调用方需要显式执行 `optimize_decision_program(program)`。

## 关键接口

- `optimize_expression(expression, atol=1.0e-12)`：对单个表达式进行规范化。
- `optimize_constraint(constraint, atol=1.0e-12)`：对约束表达式进行规范化，保留约束方向。
- `optimize_decision_program(program, inplace=False, atol=1.0e-12)`：对决策程序目标函数和约束集合执行统一优化。

## 中间表示项统计口径

中间表示项数量定义为：`len(expression.linear) + len(expression.quadratic) + 非零 constant 项数`。该口径直接对应编程框架内部表达式中间表示的结构规模，不使用源码行数作为评价指标。

## 优化规则

优化过程删除绝对值不大于 `1.0e-12` 的线性项、二次项和常数项。默认 `inplace=False`，因此会复制 `DecisionProgram` 的容器字段并返回优化副本，避免测试或调用方的原始建模对象被隐式修改。

## 测试数据摘要

| 用例 | 范围 | 优化前中间表示项 | 优化后中间表示项 | 减少冗余 | 减少率 | 减少原因 |
|---|---|---:|---:|---:|---:|---|
| 目标函数零项消除 | 目标函数 | 5 | 2 | 3 | 60.0% | 删除零线性项、零二次项和近零常数项 |
| 约束零二次项消除 | 约束 | 2 | 1 | 1 | 50.0% | 删除约束中的零二次项，使约束中间表示恢复为线性形式 |
| 空目标函数压缩 | 目标函数 | 3 | 0 | 3 | 100.0% | 删除全零目标函数中的无效线性项和二次项 |
| 混合程序冗余消除 | 完整程序 | 10 | 5 | 5 | 50.0% | 同时删除目标函数和多个约束中的零项/近零项 |
| 总计 | 汇总 | 20 | 8 | 12 | 60.0% | 多用例汇总 |

## 技术原理补充：表达式中间表示的编译优化

代码自优化作用于编程框架的表达式中间表示，而不是对 Python 源码做字符串替换。表达式

$$
E(x)=c+\sum_i a_i x_i+\sum_{i\le j}q_{ij}x_i x_j
$$

在内部由线性字典、二次字典和常数项保存。当存在 $|a_i|\le \varepsilon$、$|q_{ij}|\le \varepsilon$ 或 $|c|\le \varepsilon$ 时，这些项对数值求解和量子线路参数没有实际贡献，却会增加 QUBO 矩阵、约束行或后续转换的结构规模。

优化规则可写为阈值投影算子

$$
\Pi_{\varepsilon}(u)=\begin{cases}
0,& |u|\le \varepsilon,\\
u,& |u|>\varepsilon,
\end{cases}
$$

并逐项作用于 $a_i$、$q_{ij}$ 与 $c$。优化后的表达式为

$$
\tilde{E}(x)=\Pi_{\varepsilon}(c)+\sum_i\Pi_{\varepsilon}(a_i)x_i+\sum_{i\le j}\Pi_{\varepsilon}(q_{ij})x_i x_j.
$$

在默认容差 $\varepsilon=10^{-12}$ 下，该过程只删除数值零项或浮点舍入近零项，不改变有效金融目标函数和约束含义。

## 对量子金融建模的影响

冗余项减少会带来三类收益：

1. **QUBO 规模更清晰**：零线性项不会被误写入对角线，零二次项不会产生无意义耦合。
2. **线性约束更容易识别**：删除零二次项后，原本应为线性的约束可重新满足 `is_linear=True`，从而进入 $Ax=b$ 转换。
3. **资源估计更准确**：后端线路构造往往根据非零项生成旋转门或耦合门，删除零项可避免把无效业务规则转译为量子门。

验收报告中的“中间表示项减少率”按

$$
\eta=\frac{N_{before}-N_{after}}{N_{before}}
$$

计算。该指标反映的是编译层结构压缩，而不是业务规模缩小；金融变量、约束语义和输出目标保持不变。

