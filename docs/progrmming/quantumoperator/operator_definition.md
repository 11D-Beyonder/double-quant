# 量子金融算子定义

## 定义

量子金融算子是量子金融算子库中的最小可执行计算单元。它把一个明确的金融任务封装成统一对象，并负责完成从金融输入到量子或经典后端执行、再到金融结果还原的全过程。

一个量子金融算子必须同时具备三层语义：

1. 金融语义：这个算子解决什么金融问题。
2. 数学语义：这个金融问题被表达成什么数学形式。
3. 量子语义：这个数学形式使用什么量子算法或量子原语执行。

因此，量子金融算子不是普通函数，也不是单个量子门或单个 Qiskit 电路。它是带有输入契约、验证规则、后端执行策略、资源元数据和金融后处理规则的结构化计算对象。

## 基本形态

量子金融算子建议采用以下抽象结构：

```text
QuantumFinancialOperator
  spec: OperatorSpec
  validate(input) -> ValidationReport
  build(input) -> OperatorIR
  lower(ir, backend) -> BackendProgram
  execute(program, backend) -> RawExecutionResult
  postprocess(raw_result, ir) -> FinancialResult
  explain(result) -> OperatorExecutionReport
```

其中：

- `spec` 描述算子的稳定定义。
- `validate` 检查金融输入和算法前置条件。
- `build` 把输入转换成算子 IR。
- `lower` 把 IR 转成具体后端可执行对象，例如 Qiskit 电路、Pauli operator、QUBO 矩阵或经典枚举任务。
- `execute` 运行后端。
- `postprocess` 把底层输出还原成金融结果。
- `explain` 输出执行报告，包含资源、误差、假设和降级信息。

## OperatorSpec

`OperatorSpec` 是算子的静态定义，用来说明这个算子是什么。

```text
OperatorSpec
  id: string
  name: string
  domain: FinancialDomain
  task: FinancialTask
  problem_form: ProblemForm
  quantum_primitive: QuantumPrimitive
  input_contract: InputContract
  output_contract: OutputContract
  assumptions: list[string]
  validation_rules: list[ValidationRule]
  supported_backends: list[BackendSpec]
  resource_fields: list[string]
```

字段含义：

| 字段 | 含义 |
|---|---|
| `id` | 算子的稳定 ID，例如 `risk.quantum_shapley` |
| `name` | 人类可读名称，例如 `Quantum Shapley Risk Attribution` |
| `domain` | 金融领域，例如 `portfolio`、`risk`、`option_pricing` |
| `task` | 任务类型，例如 `optimization`、`valuation`、`attribution`、`search` |
| `problem_form` | 数学问题形式，例如 `linear_system`、`qubo`、`ising`、`shapley_game`、`amplitude_estimation` |
| `quantum_primitive` | 量子原语，例如 `hhl`、`qaoa`、`sampling_vqe`、`qae`、`quantum_shapley` |
| `input_contract` | 输入字段、类型、维度和约束 |
| `output_contract` | 输出字段、类型和金融含义 |
| `assumptions` | 金融建模假设，例如等权、历史模拟法、风险中性测度 |
| `validation_rules` | 执行前必须满足的规则 |
| `supported_backends` | 支持的执行后端 |
| `resource_fields` | 需要记录的资源字段 |

## 输入契约

输入契约定义算子接受什么输入，以及这些输入必须满足什么条件。

```text
InputContract
  fields: list[InputField]
  constraints: list[InputConstraint]
```

示例：

```text
InputField
  name: returns
  type: DataFrame
  required: true
  meaning: asset return time series

InputConstraint
  field: returns
  rule: columns_are_assets

InputConstraint
  field: alpha
  rule: 0 < alpha < 1
```

输入契约不能只依赖自然语言说明。凡是会影响量子算法正确性的条件，都应该变成可执行校验。

## 输出契约

输出契约定义算子的金融结果和底层执行元数据。

```text
OutputContract
  financial_outputs: list[OutputField]
  diagnostics: list[DiagnosticField]
  resources: list[ResourceField]
```

示例：

```text
OutputField
  name: src
  type: dict[string, float]
  meaning: Shapley risk contribution by asset

DiagnosticField
  name: validation_report
  type: ValidationReport

ResourceField
  name: oracle_calls
  type: int
```

输出结果应区分三类内容：

- 金融结果：用户真正需要的业务输出。
- 诊断信息：输入校验、误差、假设、降级路径。
- 资源信息：qubit 数、电路深度、shots、oracle calls、运行时间等。

## OperatorIR

`OperatorIR` 是算子执行前的动态中间表示，记录本次具体计算实例。

```text
OperatorIR
  operator_id: string
  instance_id: string
  financial_context: FinancialContext
  problem_instance: ProblemInstance
  quantum_plan: QuantumPlan
  backend_plan: BackendPlan
  postprocess_plan: PostprocessPlan
```

各部分含义：

| 字段 | 含义 |
|---|---|
| `financial_context` | 本次金融任务的输入、参数和假设 |
| `problem_instance` | 数学问题实例，例如矩阵、向量、QUBO、价值函数 |
| `quantum_plan` | 使用的量子原语、寄存器、oracle、测量规则 |
| `backend_plan` | 后端、shots、optimizer、seed、误差目标 |
| `postprocess_plan` | 如何把底层结果还原成金融输出 |

`OperatorSpec` 是算子类型的定义，`OperatorIR` 是一次具体执行的定义。

## 执行后端

一个算子可以支持多个后端，但金融输出契约必须保持一致。

```text
BackendSpec
  name: classical_exact | statevector | shots | qae_iqae | qaoa | sampling_vqe | hardware
  requirements: list[string]
  default_options: dict
  result_adapter: string
```

例如风险归因算子可以支持：

- `classical_exact`：精确枚举 Shapley。
- `classical_mc`：排列 Monte Carlo。
- `statevector`：量子 Shapley 电路的状态向量读取。
- `shots`：测量采样。
- `qae_iqae`：迭代振幅估计。

这些后端可以产生不同的误差和资源统计，但都应该输出同一种 `src` 金融结果。

## 验证规则

验证规则是量子金融算子的关键组成部分。它保证金融问题在进入量子后端前已经满足算法前置条件。

典型规则包括：

| 算子 | 规则 |
|---|---|
| HHL 线性系统求解 | 矩阵必须是方阵；维度应可编码到量子寄存器；矩阵应为 Hermitian 或已完成 Hermitian 嵌入 |
| Quantum Shapley | 价值函数的边际贡献必须非负；目标玩家索引合法；值函数可被稳定查询 |
| QUBO/QAOA | 变量必须是二进制；目标函数必须是二次形式；矩阵维度和变量名一致 |
| QAE 估值 | 目标事件或 payoff 必须能映射到成功概率；后处理缩放关系必须明确 |

验证失败时，算子不应继续执行，而应返回结构化错误，说明是哪条规则失败、失败输入是什么、是否存在可用降级路径。

## 结果对象

每次算子执行都应返回统一结果对象。

```text
OperatorResult
  operator_id: string
  backend: string
  financial_result: dict
  raw_result: object
  diagnostics: dict
  resources: ResourceProfile
  warnings: list[string]
```

资源信息建议采用统一结构：

```text
ResourceProfile
  num_qubits: int | null
  circuit_depth: int | null
  two_qubit_gates: int | null
  shots: int | null
  oracle_calls: int | null
  optimizer_evals: int | null
  runtime_seconds: float | null
```

这样不同算子的实验结果可以被统一汇总和绘图。

## 典型算子定义

### 线性系统求解算子

```text
id: linear_system.hhl_sapo
domain: portfolio
task: optimization
problem_form: linear_system
quantum_primitive: hhl
inputs:
  matrix: ndarray
  vector: ndarray
outputs:
  solution: ndarray
  scaling: dict
  success_probability: float
backends:
  statevector
```

该算子适合承接组合优化 KKT 系统，也可以作为更通用的线性求解底层算子。

### 风险归因算子

```text
id: risk.quantum_shapley
domain: risk
task: attribution
problem_form: shapley_game
quantum_primitive: quantum_shapley
inputs:
  returns: DataFrame
  alpha: float
  mode: rs
outputs:
  src: dict[string, float]
  contribution_share: dict[string, float]
  oracle_calls: dict[string, int]
backends:
  classical_exact
  classical_mc
  statevector
  shots
  qae_iqae
  qae_mlqae
  qae_fae
```

该算子要求量子路径使用风险节省函数，保证待编码边际贡献非负。

### 二进制优化算子

```text
id: optimization.qubo
domain: portfolio
task: optimization
problem_form: qubo
quantum_primitive: qaoa | sampling_vqe
inputs:
  quadratic_matrix: ndarray
  constant: float
  variable_names: list[string]
outputs:
  best_bitstring: ndarray
  best_objective: float
  best_probability: float
backends:
  classical_exact
  qaoa
  sampling_vqe
```

该算子只负责已经建模好的 QUBO/Ising 问题。业务约束如何转成 QUBO 应由上层 adapter 或建模层完成。

### 金融估值算子

```text
id: valuation.expected_payoff_qae
domain: option_pricing
task: valuation
problem_form: amplitude_estimation
quantum_primitive: qae
inputs:
  payoff_model: object
  distribution_model: object
  discount_rate: float
outputs:
  price: float
  confidence_interval: tuple[float, float]
backends:
  statevector
  shots
  qae_iqae
```

该算子用于后续承接欧式期权、VaR/ES 或期望收益估计。

## 与现有代码的对应关系

| 算子定义 | 当前已有基础 |
|---|---|
| `linear_system.hhl_sapo` | `HHLSolver`、`EigenBasedStrategy`、`LinearSystem` |
| `risk.quantum_shapley` | `RiskAttributor`、`RiskSavingValueFunction`、`QuantumShapleyCalculator` |
| `optimization.qubo` | `QUBOProblem`、`IsingProblem`、`QAOASolver`、`SamplingVQESolver`、`QUBOSolverResult` |
| `valuation.expected_payoff_qae` | 文档已有设计；源码中只有经典 `EuropeanCallPriceMeasure` 雏形 |

## 最小实现原则

第一版量子金融算子定义应保持简单：

1. 每个算子只解决一个明确金融任务。
2. 每个算子必须声明输入契约和输出契约。
3. 每个算子必须有执行前验证。
4. 每个算子必须把后端输出还原成金融结果。
5. 每个算子必须记录资源信息。
6. 每个算子可以有多个后端，但不能改变金融输出含义。
7. 不在算子内部解析自然语言约束。
8. 不在算子内部隐式改变金融假设。

## 结论

量子金融算子的定义重点不是“调用哪个算法类”，而是明确一个可复用计算单元的边界：它接受什么金融输入、满足什么数学条件、使用什么量子原语、支持什么后端、输出什么金融结果，以及如何解释这次执行。

只要这个定义稳定，HHL、Quantum Shapley、QUBO/QAOA、SamplingVQE 和未来的 QAE/Grover/Rasengan 算子就可以进入同一个量子金融算子库。
