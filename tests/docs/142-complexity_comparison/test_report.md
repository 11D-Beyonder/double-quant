# 142 仿真工具支持计算复杂度对比分析 func-142 测试报告

## 测试目标

验证仿真工具能够针对 `src/double_quant/algorithm` 中的真实算法电路进行门数优化前后对比分析。

## 测试范围

HHL 线性求解、QUBO-QAOA、Quantum Shapley、Shor 周期查找、SFS-Grover 搜索、Rasengan 约束优化六类真实算法电路的总门数，以及不同 transpiler 优化等级下的门数变化。

## 测试方法

执行独立测试脚本 `tests/scripts/142-complexity_comparison.py`，检查程序是否输出 PASS，并核对关键指标是否满足预期。

## 通过标准

脚本退出码为 0，输出包含 `142` 和 `PASS`，关键指标满足功能要求。

## 测试结果分析

- HHL 线性求解电路总门数由 18 降至 16。
- QUBO-QAOA 电路总门数由 56 降至 48。
- Quantum Shapley 电路总门数由 58 降至 52。
- Shor 周期查找电路总门数由 26 降至 22。
- SFS-Grover 搜索电路总门数由 172 降至 124。
- Rasengan 约束优化电路总门数由 26 降至 20。
- 已生成中文门数优化对比图 `images/142_gate_count_optimization.png`。

## 实际验证记录

测试命令：`.venv/bin/python tests/scripts/142-complexity_comparison.py`。

## 风险与限制

本项为本地 CPU 环境下的功能验收测试；如更换 Qiskit Aer 版本或启用 GPU 后端，运行时间和后端上限可能不同。

## 测试结论

通过。仿真工具支持计算复杂度对比分析 功能满足当前验收要求。
