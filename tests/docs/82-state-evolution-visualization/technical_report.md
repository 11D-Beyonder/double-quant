# 82 量子态演化可视化功能测试（Func-49） 技术报告

## 技术设计

量子态演化可视化入口为：

```python
visualize_state_evolution(
    circuit,
    output_path="...",
    animation_path="...",
    tracked_qubits=(0, 2, 4),
)
```

输出对象为 `StateEvolutionVisualization`：

| 字段 | 说明 |
|---|---|
| `circuit` | 已适配状态向量模拟的量子金融算法电路 |
| `steps` | 每一步状态向量、概率分布和 布洛赫向量 |
| `figure` | 静态 Matplotlib 图 |
| `image_path` | 静态图路径 |
| `animation_path` | GIF 动图路径 |

## 算法电路来源

测试脚本使用 `RiskSavingValueFunction` 和 `QuantumShapleyCalculator` 构造风险归因 预言机。该电路包含：

| 结构 | 作用 |
|---|---|
| `IntervalLoader` | 制备 Shapley 权重相关的区间量子态 |
| `VertexRotator` | 将玩家子集权重映射到旋转角 |
| `ValueLoader` | 将风险节约边际贡献加载到输出量子比特振幅 |

该路径与风险归因文档中的量子兼容 RS 特征函数一致，避免把 原始 ES 这类 次可加 函数直接送入量子 预言机。

## 可视化实现

模块从 `|0...0>` 开始逐条执行 `QuantumCircuit.data` 中的指令。每一步生成：

1. 完整 `Statevector`
2. 基态概率分布
3. 每个单量子比特的约化密度矩阵
4. 由约化密度矩阵计算的 布洛赫向量

布洛赫向量计算公式为：

```text
x = 2 Re(rho_01)
y = -2 Im(rho_01)
z = rho_00 - rho_11
```

静态图显示最终状态，GIF 则逐帧展示每一步的概率柱状图和 布洛赫球变化。

## 技术价值

该实现把“量子态演化”从抽象状态向量转化为金融算法可解释视图。对于量子 Shapley 风险归因，用户可以看到风险贡献如何被加载到输出量子比特振幅，并通过 GIF 观察预言机构造过程中的概率迁移。

## 技术原理补充：量子态演化的逐门计算

量子态演化可视化的数学对象是 $n$ 量子比特纯态

$$
|\psi\rangle=\sum_{k=0}^{2^n-1}\alpha_k|k\rangle,\quad \sum_k|\alpha_k|^2=1.
$$

从初始态 $|0\cdots0\rangle$ 开始，电路中第 $t$ 个量子门对应酉矩阵 $U_t$，前缀电路的状态为

$$
|\psi_t\rangle=U_tU_{t-1}\cdots U_1|0\cdots0\rangle.
$$

可视化中的概率柱状图并非直接展示振幅，而是展示 Born 规则下的测量概率：

$$
P_t(k)=|\alpha_{t,k}|^2.
$$

因此，柱状图中某个基态概率升高，表示当前电路前缀把更多幅度集中到了对应的计算基态。

布洛赫球只适合单量子比特或单比特约化态。对多量子比特状态，代码先构造密度矩阵

$$
\rho_t=|\psi_t\rangle\langle\psi_t|,
$$

再对其余量子比特做偏迹，得到第 $q$ 个量子比特的约化密度矩阵

$$
\rho_t^{(q)}=\mathrm{Tr}_{\bar q}(\rho_t).
$$

布洛赫向量计算为

$$
\vec r=(\mathrm{Tr}(\rho X),\mathrm{Tr}(\rho Y),\mathrm{Tr}(\rho Z))
=(2\Re\rho_{01},-2\Im\rho_{01},\rho_{00}-\rho_{11}).
$$

如果某个量子比特与其他寄存器纠缠，$\|\vec r\|<1$，图中向量会落在球内；这正是量子 Shapley 预言机中控制寄存器、风险旋转位和输出位相互关联的可视化证据。

## 金融含义解释

在风险归因电路中，状态制备表示资产子集或 Shapley 权重的叠加，风险旋转把边际风险贡献写入输出量子比特振幅。逐门 GIF 展示的不是静态电路图，而是

$$
\text{资产子集叠加}\rightarrow \text{风险权重加载}\rightarrow \text{输出位概率变化}
$$

这一计算过程。因此报告中的量子态可视化应理解为金融风险信息在量子寄存器中的动态传播，而不是简单 API 绘图。

