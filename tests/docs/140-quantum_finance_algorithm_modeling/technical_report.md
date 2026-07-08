# 140 仿真工具支持量子金融算法建模 func-140 技术报告

## 技术目标

验证仿真工具能够面向量子金融算法生成可执行量子电路模型。

## 实现位置

`simulator/finance.py` 中 `build_weighted_sum_circuit` 和 `build_portfolio_qaoa_ansatz`。

## 实现概述

该功能由 simulator 模块提供统一接口，测试脚本按验收项独立调用，输出可复核的终端指标。

## 关键技术点

- 状态向量仿真用于表示量子金融电路的完整量子态。对 $n$ 个量子比特，仿真器维护长度为 $2^n$ 的复振幅向量：
  $$
  |\psi\rangle=\sum_{x=0}^{2^n-1}\alpha_x|x\rangle,\qquad \sum_{x=0}^{2^n-1}|\alpha_x|^2=1
  $$
  其中 $|x\rangle$ 表示一个资产选择、价格状态或权重编码后的二进制基态，$\alpha_x$ 表示该金融状态对应的概率振幅。测量得到状态 $x$ 的概率为：
  $$
  P(x)=|\alpha_x|^2
  $$
  因此，金融组合搜索问题可以通过量子线路把目标函数较优的 bitstring 映射到较高测量概率上。

- 金融权重和收益约束通过二进制加权求和电路建模。若资产选择变量为 $z_i\in\{0,1\}$，资产权重或收益系数为 $w_i$，则组合加权和为：
  $$
  S(z)=\sum_{i=1}^{n}w_i z_i
  $$
  `build_weighted_sum_circuit` 的作用是将该求和关系编码到量子寄存器中，使输入选择态 $|z_1z_2\cdots z_n\rangle$ 与辅助寄存器中的求和值形成可逆映射：
  $$
  |z\rangle|0\rangle \longmapsto |z\rangle|S(z)\rangle
  $$
  该设计保证量子线路满足幺正性要求，同时为后续约束判断、收益计算和目标函数评估提供可测量的中间量。

- 投资组合优化采用 QAOA 变分线路表达候选解分布。典型目标是最大化收益并惩罚风险或约束违背，可写为 Ising/QUBO 形式：
  $$
  C(z)=\sum_i h_i z_i+\sum_{i<j}J_{ij}z_i z_j+\lambda\left(\sum_i z_i-K\right)^2
  $$
  其中 $h_i$ 表示单资产收益或成本项，$J_{ij}$ 表示资产间相关性或风险耦合，$\lambda$ 为约束惩罚系数，$K$ 为目标持仓数量。`build_portfolio_qaoa_ansatz` 构造的 $p$ 层 QAOA 态为：
  $$
  |\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle
  =
  \prod_{\ell=1}^{p}
  e^{-i\beta_\ell H_M}
  e^{-i\gamma_\ell H_C}
  |+\rangle^{\otimes n}
  $$
  其中 $H_C$ 编码组合优化目标函数，$H_M=\sum_i X_i$ 为混合哈密顿量。优化目标是最小化或最大化期望值：
  $$
  E(\boldsymbol{\gamma},\boldsymbol{\beta})
  =
  \langle\psi(\boldsymbol{\gamma},\boldsymbol{\beta})|H_C|\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle
  $$
  测试通过可执行线路结构和测量输出验证仿真工具能够完成金融问题到量子电路模型的映射。


## 技术结论

仿真工具支持量子金融算法建模 已在 simulator 中实现，并通过独立验收脚本验证。
