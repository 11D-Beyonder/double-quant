# 83 量子电路可视化功能测试（Func-50） 技术报告

## 技术设计

量子电路可视化入口为：

```python
visualize_quantum_circuit(
    circuit,
    output_path="...",
    title="量子 Shapley Risk Attribution Circuit",
)
```

输出对象为 `CircuitVisualization`：

| 字段 | 说明 |
|---|---|
| `circuit` | 实际绘制的 Qiskit 电路 |
| `figure` | Qiskit Matplotlib 电路图 |
| `text_diagram` | 文本电路图 |
| `image_path` | 图片导出路径 |

## 算法电路

本功能脚本不再手写普通示例电路，而是调用：

```python
calculator = QuantumShapleyCalculator(
    3,
    RiskSavingValueFunction(returns, alpha=0.75),
    internal_qubits_num=2,
    internal_multiplier=1,
)
circuit, _ = calculator.build_player_circuit(target_player=0)
```

新增的 `build_player_circuit()` 是量子 Shapley 计算器的公开电路构建入口，便于在不运行完整 Shapley 求值的情况下检查预言机结构。

## 技术价值

量子电路图展示了金融风险归因算法如何被拆解为可执行量子程序。金融人员可以从 输出量子比特 理解“风险贡献读出位置”，算法开发人员可以检查 `StatePreparation`、`VertexRotator` 和 `ValueLoader` 的连接关系。

## 技术原理补充：电路图的门级语义与量子金融结构

量子电路可视化展示的是酉变换分解。一个量子金融电路可以写成门序列

$$
U=U_mU_{m-1}\cdots U_1,
$$

电路图从左到右给出这些门在不同量子比特线上的作用位置。单量子位门改变局部相位或振幅，双量子位门引入条件逻辑和纠缠；因此线路深度 $D$ 近似表示可并行层数，双量子位门数量 $N_{2q}$ 是当前量子硬件上更敏感的资源成本。

在量子 Shapley 风险归因线路中，各结构的含义为：

- 状态制备或区间加载：形成资产子集与 Shapley 权重相关的叠加态。
- 受控旋转：按边际风险贡献设置输出比特振幅。
- 输出量子比特：通过测量概率承载归一化后的风险贡献。

如果某一风险值 $x$ 被编码为旋转角 $\theta=2\arcsin\sqrt{x}$，则输出位满足

$$
\Pr(1)=\sin^2\frac{\theta}{2}=x.
$$

因此电路图中的旋转门不是装饰性组件，而是金融数值进入量子概率空间的关键位置。

## 文本图与图片图的互补作用

Matplotlib 电路图适合验收材料展示整体结构，文本图适合在终端输出和版本对比中快速检查门序。两者都来自同一个 `QuantumCircuit`，因此满足

$$
\mathrm{text\_diagram}(C)\equiv \mathrm{mpl\_diagram}(C)
$$

的结构一致性。`decompose_reps` 参数用于在需要时展开高层指令，使验收人员能够从抽象模块下钻到基础门序列。

