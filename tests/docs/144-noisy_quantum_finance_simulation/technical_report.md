# 144 仿真工具支持含噪量子金融计算模拟 func-144 技术报告

## 技术目标

验证仿真工具能够加载噪声模型并执行含噪量子金融计算模拟。

## 实现位置

`simulator/backends.py` 中 `NoiseConfig`、`build_depolarizing_noise_model` 和 `simulate_counts`。

## 实现概述

该功能由 simulator 模块提供统一接口，测试脚本按验收项独立调用，输出可复核的终端指标。

## 关键技术点

- 使用 depolarizing error 构造单比特和双比特门噪声。退极化噪声表示量子门执行后，量子态以一定概率被随机 Pauli 错误扰动。单比特退极化信道可写为：
  $$
  \mathcal{E}_1(\rho)=(1-p)\rho+\frac{p}{3}(X\rho X+Y\rho Y+Z\rho Z)
  $$
  其中 $p$ 为单比特门错误率。双比特退极化信道可写为：
  $$
  \mathcal{E}_2(\rho)=(1-p)\rho+\frac{p}{15}\sum_{P\in\mathcal{P}_2\setminus\{II\}}P\rho P
  $$
  其中 $\mathcal{P}_2=\{I,X,Y,Z\}^{\otimes 2}$，去除恒等算符 $II$ 后共有 15 个非平凡 Pauli 错误。`build_depolarizing_noise_model` 分别将单比特和双比特错误率绑定到对应门类型，使含噪采样更接近真实量子设备的门级误差。

- 噪声会把理想纯态演化推广为密度矩阵或随机轨迹意义下的混合态演化。理想线路中量子态按幺正门 $U$ 演化：
  $$
  \rho' = U\rho U^\dagger
  $$
  加入噪声后演化变为量子信道：
  $$
  \rho'=\mathcal{E}(U\rho U^\dagger)
  $$
  测量概率由密度矩阵对角元给出：
  $$
  P(x)=\mathrm{Tr}(|x\rangle\langle x|\rho)
  $$
  因此含噪量子金融模拟可以输出受噪声影响后的 bitstring 采样分布，用于评估金融目标函数在真实设备误差下的鲁棒性。

- 存在噪声配置时采用矩阵乘积态方法执行采样。矩阵乘积态将 $n$ 比特量子态分解为一串局部张量：
  $$
  |\psi\rangle
  =
  \sum_{i_1,\ldots,i_n}
  A^{[1]i_1}A^{[2]i_2}\cdots A^{[n]i_n}
  |i_1i_2\cdots i_n\rangle
  $$
  其中 $i_k\in\{0,1\}$，$A^{[k]i_k}$ 是第 $k$ 个量子比特对应的张量，张量之间的连接维度称为 bond dimension，记为 $\chi$。完整 statevector 的存储复杂度为 $O(2^n)$，而 MPS 在纠缠有限时的存储复杂度约为：
  $$
  O(n\chi^2)
  $$
  单比特门只更新一个局部张量：
  $$
  A^{[k]i_k}\leftarrow \sum_{j_k}U_{i_kj_k}A^{[k]j_k}
  $$
  双比特门先合并相邻张量，施加门矩阵后再通过 SVD 分解回 MPS：
  $$
  \Theta^{i_ki_{k+1}} = \sum_{\alpha}A^{[k]i_k}_{\alpha}A^{[k+1]i_{k+1}}_{\alpha}
  $$
  $$
  \Theta' = U\Theta,\qquad \Theta'\approx W S V^\dagger
  $$
  该过程把局部门作用限制在局部张量更新中，适合 20 比特及以上、但纠缠增长可控的金融线路采样。`simulate_counts` 在检测到 `NoiseConfig` 后使用该后端完成含噪采样，避免直接保存完整 $2^n$ 维状态向量带来的资源压力。



## 技术结论

仿真工具支持含噪量子金融计算模拟 已在 simulator 中实现，并通过独立验收脚本验证。
