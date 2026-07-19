# 142 仿真工具支持计算复杂度对比分析 func-142 技术报告

## 技术目标

验证仿真工具能够针对 `src/double_quant/algorithm` 中的真实算法电路进行门数复杂度优化前后对比分析。

## 实现位置

`simulator/analysis.py` 中 `analyze_complexity` 和 `compare_complexity`；测试脚本位于 `tests/scripts/142-complexity_comparison.py`。

## 实现概述

测试脚本构造 HHL 线性求解、QUBO-QAOA、Quantum Shapley、Shor 周期查找、SFS-Grover 搜索、Rasengan 约束优化六类真实算法电路，对每个电路执行 Qiskit transpiler optimization level 0 到 3 的门数统计，取总门数最低的优化等级作为优化后结果，并生成中文柱状对比图。

## 关键技术点

- 使用真实算法电路而非人工冗余门序列。HHL 来自 `algorithm/hhl/variants.py` 并展开 QPE 子线路，QUBO-QAOA 使用绑定数值参数后的 QAOA ansatz，Quantum Shapley 使用 `algorithm/shapley/quantum.py` 中扩大后的量子价值加载子线路，Shor 周期查找来自 `algorithm/shor/circuit.py`，SFS-Grover 搜索来自 `algorithm/grover/circuit.py`，Rasengan 约束优化来自 `algorithm/rasengan/circuit.py`。

- 门数复杂度定义为：
  $$
  G(C)=\sum_{g\in\mathcal{G}}N_g(C)
  $$
  其中 $N_g(C)$ 表示门 $g$ 在电路 $C$ 中出现的次数。优化前后门数下降量为：
  $$
  \Delta G=G_{\mathrm{before}}-G_{\mathrm{after}}
  $$

- 对同一算法电路执行不同优化等级：
  $$
  C_k=\mathrm{Transpile}(C,\mathrm{optimization\_level}=k),\qquad k\in\{0,1,2,3\}
  $$
  并选择：
  $$
  G_{\mathrm{after}}=\min_{k\in\{0,1,2,3\}}G(C_k)
  $$

- 图表输出采用中文标签，横轴为算法电路类型，纵轴为总门数，成对展示优化前和优化后结果，便于验收报告直接引用。

## 技术结论

仿真工具已完成真实算法电路的门数复杂度优化对比分析，并生成 `images/142_gate_count_optimization.png` 作为图形化验收证据。
