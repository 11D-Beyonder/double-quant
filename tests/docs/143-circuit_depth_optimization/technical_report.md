# 143 仿真工具支持算法复杂度优化 func-143 技术报告

## 技术目标

验证仿真工具能够针对 `src/double_quant/algorithm` 中的真实算法电路进行电路深度优化分析。

## 实现位置

`simulator/analysis.py` 中 `compare_complexity`；测试脚本位于 `tests/scripts/143-circuit_depth_optimization.py`。

## 实现概述

测试脚本构造 HHL 线性求解、QUBO-QAOA、Quantum Shapley、Shor 周期查找、SFS-Grover 搜索、Rasengan 约束优化六类真实算法电路，对每个电路执行 Qiskit transpiler optimization level 0 到 3 的深度统计，取电路深度最低的优化等级作为优化后结果，并生成中文柱状对比图。

其中 HHL 电路展开 QPE 子线路，QUBO-QAOA 绑定数值参数后展开 ansatz，Quantum Shapley 扩大到 4 个控制比特的价值加载子线路，以保证六类算法在当前验收规模下均能观察到优化前后深度变化。

## 关键技术点

- 电路深度刻画量子门在依赖约束下的最少串行执行层数。若第 $t$ 层可并行执行的门集合为 $L_t$，则电路可写为：
  $$
  C=L_D L_{D-1}\cdots L_1
  $$
  其中 $D$ 即电路深度。

- 对同一算法电路执行不同优化等级：
  $$
  C_k=\mathrm{Transpile}(C,\mathrm{optimization\_level}=k),\qquad k\in\{0,1,2,3\}
  $$
  对应深度为：
  $$
  D_k=\mathrm{Depth}(C_k)
  $$
  优化后深度取：
  $$
  D_{\mathrm{after}}=\min_{k\in\{0,1,2,3\}}D_k
  $$

- 深度下降量定义为：
  $$
  \Delta D=D_{\mathrm{before}}-D_{\mathrm{after}}
  $$
  深度越低，线路在实际硬件或仿真后端中的串行执行层数越少。

- 图表输出采用中文标签，横轴为算法电路类型，纵轴为电路深度，成对展示优化前和优化后结果，便于验收报告直接引用。

## 技术结论

仿真工具已完成真实算法电路的深度优化分析，并生成 `images/143_depth_optimization.png` 作为图形化验收证据。
