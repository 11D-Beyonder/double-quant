# 84 量子计算过程可视化功能测试（Func-51） 技术报告

## 技术设计

量子计算过程可视化入口为：

```python
visualize_quantum_computation_process(
    hhl_circuit,
    output_path="...",
    animation_path="...",
    tracked_qubits=(0, 2, 5),
)
```

输出对象为 `ComputationProcessVisualization`：

| 字段 | 说明 |
|---|---|
| `circuit` | HHL/SAPO 量子线性系统电路 |
| `steps` | 每个计算阶段的状态向量、概率分布和 布洛赫向量 |
| `operation_labels` | 高层算法步骤标签 |
| `final_probabilities` | 最终基态概率 |
| `image_path` | 静态图路径 |
| `animation_path` | GIF 动图路径 |

## 算法电路来源

新增的 `HHLSolver.build_circuit()` 提供 HHL 电路构建入口。脚本构造 2 资产组合优化约束系统：

```text
[ target_return ]   [ mu^T ] [ weights ]
[      1       ] = [  1^T ] [ weights ]
```

并将约束项与协方差矩阵组合成对称线性系统，交给 HHL/SAPO 电路编码。

## 可视化内容

静态图和 GIF 同时包含：

| 图层 | 内容 |
|---|---|
| Algorithm Gate Timeline | 状态制备、QPE、倒数旋转、逆 QPE 的执行顺序 |
| Basis Probability | 当前计算阶段的基态概率分布 |
| 布洛赫球 | 向量寄存器、相位寄存器、标志量子比特的单比特约化态 |

其中 `tracked_qubits=(0, 2, 5)` 分别对应向量寄存器、相位寄存器和标志相关量子比特，用于观察线性系统解码过程中不同寄存器的状态变化。

## 技术价值

组合优化 HHL 电路通常比普通教学电路更难解释。该功能把高层算法步骤、状态概率和局部 布洛赫向量放在同一个时间轴上，便于定位 HHL 求解中状态制备、相位估计和成功 flag 的变化。

## 技术原理补充：量子计算过程的时间轴模型

量子计算过程可视化在量子态演化基础上增加“门时间线”。对于电路 $C=(U_1,\ldots,U_m)$，第 $t$ 帧同时展示三类信息：

$$
\left(t,\ |\psi_t\rangle,\ P_t(k)=|\langle k|\psi_t\rangle|^2\right).
$$

时间线图把第 $t$ 个操作映射到受影响的量子比特集合 $Q_t$，如果一个门作用于多个比特，则在图中用竖线连接这些比特，表示该操作可能产生条件关系或纠缠。活跃步骤用红色突出，便于把概率变化与具体量子门对应起来。

对 HHL/SAPO 线性系统，核心数学目标是求解

$$
A|x\rangle=|b\rangle,
$$

理想 HHL 流程可概括为

$$
|b\rangle \xrightarrow{\mathrm{QPE}} \sum_j \beta_j |u_j\rangle|\lambda_j\rangle
\xrightarrow{R_y} \sum_j \beta_j |u_j\rangle|\lambda_j\rangle\left(\sqrt{1-\frac{C^2}{\lambda_j^2}}|0\rangle+\frac{C}{\lambda_j}|1\rangle\right)
\xrightarrow{\mathrm{QPE}^{-1}} |x\rangle|1\rangle.
$$

其中 $\lambda_j$ 和 $|u_j\rangle$ 是矩阵 $A$ 的特征值与特征向量，成功标志位为 $|1\rangle$ 时，向量寄存器与 $A^{-1}b$ 成比例。

## 可视化如何解释组合优化

组合优化约束系统把目标收益和预算约束写成线性方程，形式上为

$$
\begin{bmatrix}
\mu^\top\\
\mathbf{1}^\top
\end{bmatrix}w=
\begin{bmatrix}
r^*\\
1
\end{bmatrix},
$$

并在实际求解中与协方差或约束扩展项共同组成对称线性系统。计算过程可视化使验收人员看到：状态制备对应 $|b\rangle$，相位估计对应特征值寄存器，受控旋转对应倒数 $1/\lambda_j$ 的加载，最终概率分布对应解向量的量子态表达。

因此该功能不仅画出最终概率，而是把 HHL 从“黑盒求解器”拆成可观察的算法阶段。

