# 145 仿真工具支撑20比特及以上的量子金融算法仿真 func-145 测试报告

## 测试目标

验证仿真工具满足 20 比特及以上量子金融算法仿真需求，并明确 statevector CPU 输出型路径的运行结果。

## 测试范围

20 比特金融权重编码电路、容量检查、statevector CPU 后端、完整 statevector 输出、概率分布统计。

## 测试方法

执行独立测试脚本 `tests/scripts/145-20qubit_quantum_finance_simulation.py`，检查程序是否输出 PASS，并核对 20 比特验收规模、`STATEVECTOR_CPU` 后端、statevector 长度和非零概率项数量是否写入输出。

## 通过标准

脚本退出码为 0，输出包含 `145` 和 `PASS`；容量检查结果为 True；运行后端为 `statevector_cpu`；输出明确包含 `statevector_output_run_qubits=20` 和 `statevector_output_statevector_len=1048576`。

## 测试结果分析

- 20 比特验收规模通过，满足“20比特及以上”的基本验收要求。
- statevector CPU 后端返回完整状态向量，20 比特对应 $2^{20}=1,048,576$ 个复振幅。
- 输出中记录运行耗时和非零概率项数量，可用于复核该金融权重编码线路的实际仿真结果。

## 实际验证记录

测试命令：`.venv/bin/python tests/scripts/145-20qubit_quantum_finance_simulation.py`。

## 风险与限制

statevector 输出型会导出完整 statevector，并构造概率字典，内存和耗时随 `2^n` 增长。当前测试选择 20 比特作为稳定验收规模，避免因机器负载或内存波动导致验收脚本不稳定。

## 测试结论

通过。仿真工具通过 statevector CPU 路径支撑 20 比特及以上量子金融算法仿真。
