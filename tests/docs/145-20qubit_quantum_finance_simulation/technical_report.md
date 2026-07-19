# 145 仿真工具支撑20比特及以上的量子金融算法仿真 func-145 技术报告

## 技术目标

验证仿真工具满足 20 比特及以上量子金融算法仿真需求，并在验收输出中记录 statevector CPU 输出型仿真的运行规模和资源特征。

## 实现位置

`simulator/analysis.py`、`simulator/backends.py`、`simulator/finance.py` 与 `tests/scripts/145-20qubit_quantum_finance_simulation.py`。

## 实现概述

`run_capacity_smoke_circuit` 构造金融权重编码电路，并通过 `STATEVECTOR_CPU` 后端执行 20 比特 statevector 验收仿真。145 测试脚本输出容量校验结果、实际运行比特数、后端类型、运行耗时、statevector 长度和非零概率项数量。

## 关键技术点

- `verify_capacity` 校验 20 比特及以上 statevector 后端能力。量子线路的希尔伯特空间维度随比特数指数增长：
  $$
  \dim(\mathcal{H})=2^n
  $$
  因此 20 比特对应 $2^{20}=1,048,576$ 个计算基态。容量校验条件可抽象为：
  $$
  n\ge n_{\min},\qquad n_{\min}=20
  $$
  同时要求所选后端属于当前工具声明支持的容量验证路径：
  $$
  backend\in\{\mathrm{STATEVECTOR\_CPU}\}
  $$
  该检查保证验收脚本不是只运行小规模示例，而是明确覆盖 20 比特及以上的量子金融仿真规模。

- `run_capacity_smoke_circuit` 构造金融权重编码电路并调用 statevector CPU 仿真。金融电路通常将资产选择表示为 bitstring：
  $$
  z=(z_1,z_2,\ldots,z_n),\qquad z_i\in\{0,1\}
  $$
  权重编码目标可以写为：
  $$
  S(z)=\sum_{i=1}^{n}w_i z_i
  $$
  对应量子线路把选择寄存器和求和寄存器关联起来：
  $$
  |z\rangle|0\rangle \longmapsto |z\rangle|S(z)\rangle
  $$
  验收脚本通过该金融风格线路验证仿真器能处理实际业务结构，而不是只对空线路或随机门做规模测试。

- statevector CPU 后端保存完整复振幅向量。对 $n$ 比特量子态，statevector 表示为：
  $$
  |\psi\rangle=\sum_{x=0}^{2^n-1}\alpha_x|x\rangle,\qquad \sum_x|\alpha_x|^2=1
  $$
  测量概率由振幅模平方给出：
  $$
  P(x)=|\alpha_x|^2
  $$
  若每个复振幅使用双精度复数存储，单个振幅通常需要 16 字节，则 statevector 主要内存开销近似为：
  $$
  M(n)=2^n\times 16\ \mathrm{bytes}
  $$
  20 比特时：
  $$
  M(20)=2^{20}\times 16=16,777,216\ \mathrm{bytes}\approx 16\ \mathrm{MiB}
  $$
  该规模适合作为稳定、可重复的验收测试；比特数继续增加时，内存和耗时按 $2^n$ 指数增长。

- 非零概率项数量用于观察金融线路输出分布规模。若输出分布为 $P(x)$，非零概率集合为：
  $$
  \Omega_+=\{x\mid P(x)>0\}
  $$
  验收输出记录：
  $$
  |\Omega_+|
  $$
  该指标反映线路实际覆盖了多少候选 bitstring。对金融组合问题而言，每个 bitstring 对应一种资产选择方案，因此非零概率项越多，说明仿真输出覆盖的候选组合越丰富。

## 技术结论

仿真工具已通过 statevector CPU 路径完成 20 比特及以上量子金融算法仿真验收，验收输出直接记录完整 statevector 的运行结果和资源指标。
