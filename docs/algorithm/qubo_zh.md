# QUBO Quick Start

这篇文档介绍 Double Quant 里新加入的 `QUBO / Ising` 求解接口，重点覆盖：

- 如何构造 `QUBOProblem`
- 如何调用经典 baseline、`QAOA`、`SamplingVQE`
- 返回结果里每个字段代表什么
- `QUBO` 和 `Ising` 在当前实现里的约定

当前这条能力线只负责**已经建模好的** `QUBO / Ising` 问题求解，不负责把业务问题自动翻译成 `QuadraticProgram`、`QUBO` 或 `Ising`。

## 1. 最小可运行示例

下面先用精确经典方法跑一个 2 变量 `QUBO`：

```python
import numpy as np

from double_quant import NumPyMinimumEigensolverSolver, QUBOProblem

problem = QUBOProblem(
    quadratic_matrix=np.array(
        [
            [-1.0, 2.0],
            [0.0, -0.5],
        ]
    )
)

solver = NumPyMinimumEigensolverSolver()
result = solver.solve(problem)

print(result.best_bitstring)   # [1 0]
print(result.best_objective)   # -1.0
print(result.best_energy)      # -1.0
print(result.probabilities)    # {'10': 1.0, ...}
```

这里的目标函数是：

$$
x^T Q x, \quad x \in \{0,1\}^n
$$

对应上面的矩阵，4 个候选解的目标值分别是：

- `00 -> 0.0`
- `10 -> -1.0`
- `01 -> -0.5`
- `11 -> 0.5`

所以最优解是 `10`，也就是 `best_bitstring = [1, 0]`。

## 2. `QUBOProblem` 的数据格式

当前实现里，`QUBOProblem` 放在 `double_quant.common`，表示一个标准的 `QUBO` 问题：

```python
from double_quant import QUBOProblem
```

最小字段是：

- `quadratic_matrix: np.ndarray`
- `constant: float = 0.0`
- `variable_names: list[str] | None = None`

### `quadratic_matrix`

必须是二维方阵，形状为 `(n, n)`。

当前实现使用下面这个目标函数定义：

$$
f(x) = x^T Q x + c
$$

也就是说，`quadratic_matrix[i, j]` 直接参与这个二次型计算。  
如果矩阵不是对称的，后续 `QUBO -> Ising` 转换时会先使用

$$
\frac{Q + Q^T}{2}
$$

来构造等价问题，因为对任意二进制向量 `x` 都有：

$$
x^T Q x = x^T \left(\frac{Q + Q^T}{2}\right) x
$$

### `constant`

这是目标函数里的常数项 `c`。如果不需要，可以省略。

### `variable_names`

如果不传，默认会生成：

- `x_0`
- `x_1`
- `x_2`
- ...

它主要用于问题对象本身的可读性，目前不会改变 solver 的数学结果。

## 3. `IsingProblem` 的约定

内部求解时，`QUBO` 会先转换成 `Ising` 形式。  
你也可以直接传 `IsingProblem` 给 solver。

当前的 `IsingProblem` 形式是：

$$
\sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j + c, \quad s_i \in \{-1, +1\}
$$

对应关系是：

- 比特变量：`x_i in {0, 1}`
- 自旋变量：`s_i in {-1, +1}`

项目里采用的映射是：

$$
s_i = 1 - 2x_i
$$

所以：

- `x_i = 0` 对应 `s_i = +1`
- `x_i = 1` 对应 `s_i = -1`

这也是当前 `bits_to_spins(...)` / `spins_to_bits(...)` 的语义。

## 4. 可用求解器

当前这条能力线提供 3 个 solver：

### `NumPyMinimumEigensolverSolver`

```python
from double_quant import NumPyMinimumEigensolverSolver
```

这是纯经典的精确 baseline：

- 底层调用 `qiskit_algorithms.NumPyMinimumEigensolver`
- 适合做小规模问题对拍
- 能给出精确基态和精确概率分布

如果你要验证 `QAOA` / `SamplingVQE` 的结果，建议先用它作为基线。

### `QAOASolver`

```python
from double_quant import QAOASolver
from qiskit_algorithms.optimizers import COBYLA

solver = QAOASolver(
    optimizer=COBYLA(maxiter=20),
    reps=1,
    seed=7,
)
```

这是对 `qiskit_algorithms.QAOA` 的项目封装。

当前最常用的参数是：

- `sampler`
- `optimizer`
- `reps`
- `initial_state`
- `mixer`
- `initial_point`
- `seed`

如果你不传 `sampler`，项目里会默认创建一个 `StatevectorSampler`。

### `SamplingVQESolver`

```python
from double_quant import SamplingVQESolver
from qiskit_algorithms.optimizers import COBYLA

solver = SamplingVQESolver(
    optimizer=COBYLA(maxiter=20),
    seed=7,
)
```

这是对 `qiskit_algorithms.SamplingVQE` 的项目封装。

当前最常用的参数是：

- `sampler`
- `ansatz`
- `optimizer`
- `initial_point`
- `seed`

如果你不传 `ansatz`，项目里会默认使用 `real_amplitudes(num_qubits, reps=1)`。

## 5. 返回结果 `QUBOSolverResult`

这 3 个 solver 都统一返回：

```python
from double_quant.algorithm.qubo import QUBOSolverResult
```

最常用字段有：

- `best_bitstring`
- `best_objective`
- `best_energy`
- `best_probability`
- `parameter_values`
- `probabilities`
- `metadata`

### `best_bitstring`

这是项目统一后的最优 bitstring，按变量顺序返回，例如：

```python
array([1, 0, 1])
```

这里的顺序就是：

- 第 0 个变量
- 第 1 个变量
- 第 2 个变量

不会直接暴露 Qiskit 的内部位序。

### `best_objective`

原问题目标值。

- 如果输入是 `QUBOProblem`，就是 `x^T Q x + c`
- 如果输入是 `IsingProblem`，就是对应 `Ising` 能量

### `best_energy`

统一表示 `Ising` Hamiltonian 对应的能量值。

对于从 `QUBO` 转换过来的问题，`best_objective` 和 `best_energy` 在数值上应当相等，只是语义不同：

- `best_objective`：原始问题目标值
- `best_energy`：转换后 `Ising` 的能量

### `best_probability`

最优 bitstring 在最终分布中的概率。

- 对 `NumPyMinimumEigensolverSolver`，理想情况下通常接近或等于 `1.0`
- 对 `QAOA` / `SamplingVQE`，这是优化后电路的采样分布结果

### `parameter_values`

变分算法找到的最优参数向量。

- 经典 baseline 通常没有这项
- `QAOA` 和 `SamplingVQE` 通常会有

### `probabilities`

最终 bitstring 分布，键是**项目位序**下的字符串，例如：

```python
{
    "10": 1.0,
    "01": 1.3e-31,
}
```

这里 `"10"` 表示：

- 第 0 个变量取 `1`
- 第 1 个变量取 `0`

## 6. 一个更完整的对拍例子

下面用 3 个 solver 同时求解同一个 `QUBO`：

```python
import numpy as np

from double_quant import (
    NumPyMinimumEigensolverSolver,
    QAOASolver,
    QUBOProblem,
    SamplingVQESolver,
)
from qiskit_algorithms.optimizers import COBYLA

problem = QUBOProblem(
    quadratic_matrix=np.array(
        [
            [-1.0, 0.0],
            [0.0, -1.5],
        ]
    )
)

exact_solver = NumPyMinimumEigensolverSolver()
qaoa_solver = QAOASolver(optimizer=COBYLA(maxiter=20), reps=1, seed=7)
svqe_solver = SamplingVQESolver(optimizer=COBYLA(maxiter=20), seed=7)

exact_result = exact_solver.solve(problem)
qaoa_result = qaoa_solver.solve(problem)
svqe_result = svqe_solver.solve(problem)

print("exact:", exact_result.best_bitstring, exact_result.best_objective)
print("qaoa:", qaoa_result.best_bitstring, qaoa_result.best_objective)
print("sampling_vqe:", svqe_result.best_bitstring, svqe_result.best_objective)
```

对于这个例子，最优解是：

- `best_bitstring = [1, 1]`
- `best_objective = -2.5`

通常可以先用 `NumPyMinimumEigensolverSolver` 得到基准答案，再看变分算法是否收敛到同一个解。

## 7. 什么时候用哪个 solver

可以先用一条简单规则：

- 想要小规模精确答案：用 `NumPyMinimumEigensolverSolver`
- 想要 `QAOA` 路线：用 `QAOASolver`
- 想要 `SamplingVQE` 路线：用 `SamplingVQESolver`

如果你现在的目标是：

- 验证模型是否建对了：先用精确 baseline
- 和量子变分方法做对拍：先跑 baseline，再跑 `QAOA` / `SamplingVQE`
- 只关心统一接口，不关心底层算法：都可以走 `solve(problem)`，返回同一个 `QUBOSolverResult`

## 8. 当前边界

这份实现当前**没有**做这些事情：

- 不接 `QuadraticProgram`
- 不接自动约束转 penalty
- 不接 `qiskit-finance`
- 不负责把设施选址、资产选择等业务问题自动翻译成 `QUBO`

当前这一层只做：

- `QUBOProblem`
- `IsingProblem`
- `QUBO -> Ising`
- `Ising -> Pauli operator`
- 统一 solver 封装

如果你后面要做上层应用，可以直接把已经建模好的 `QUBOProblem` 传进来。
