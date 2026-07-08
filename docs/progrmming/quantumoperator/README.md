# 量子金融算子 IR 的重要性

## 目标

量子金融算子 IR（Intermediate Representation，中间表示）用于把金融问题、量子算法、后端执行和结果解释连接成一个稳定的内部协议。它不是单个求解器，也不是某个 Qiskit 电路对象，而是一层可检查、可转换、可记录的结构化表示。

当前仓库已经有若干量子金融算法原型，包括 HHL/SAPO 线性求解、量子 Shapley 风险归因、QUBO/Ising 优化、QAOA 和 SamplingVQE。但是这些能力目前仍以 solver、calculator、application workflow 的形式分散存在。算子 IR 的价值在于把这些能力收束成统一的“量子金融算子库”，让上层金融问题不需要直接关心底层电路、后端、抽样方式和后处理细节。

## 1. 算子库

量子金融算子库应当是 IR 的直接落地点。一个算子不是简单函数，而是一个完整的计算单元，至少包含输入契约、金融语义、量子原语、降级路径、输出契约和资源元数据。

当前仓库中可以归入算子库雏形的能力包括：

| 算子方向 | 当前实现 | 金融语义 | 当前边界 |
|---|---|---|---|
| 线性系统求解算子 | `HHLSolver`、`EigenBasedStrategy` | 求解 `Ax=b`，支撑组合优化 KKT 系统 | 主要走 statevector 抽取；`qiskit` variant 未实现；资源统计和公开电路导出不足 |
| 组合优化算子 | `PortfolioOptimizer` | 均值-方差组合权重求解 | `ConstraintScaler` 未完成；还不是完整 SAPO 工作流 |
| Shapley 风险归因算子 | `RiskAttributor`、`QuantumShapleyCalculator` | 基于 ES/RS 的资产风险贡献分解 | 量子路径要求风险节省函数满足非负边际贡献；当前多用于小规模实验 |
| 振幅估计算子 | Shapley 中的 `qae_canonical`、`qae_iqae`、`qae_mlqae`、`qae_fae` | 从量子电路输出比特概率估计金融期望或归因值 | 目前主要嵌在 Shapley 实现中，还没有独立通用 QAE 金融估值算子 |
| QUBO/Ising 优化算子 | `QUBOProblem`、`IsingProblem`、`QAOASolver`、`SamplingVQESolver` | 离散选择、资产选择、约束优化的底层优化形式 | 只处理已经建模好的 QUBO/Ising；不自动把业务约束转 penalty |
| 金融度量算子 | `ExpectedShortfallMeasure`、`ShapleyRiskContributionMeasure`、`EuropeanCallPriceMeasure` | ES、SRC、欧式看涨价格等金融输出 | 部分仍是经典实现；期权 QAE 还停留在文档设计层 |

一个成熟算子库不应该只暴露 `solve()`。建议每个算子都提供统一生命周期：

```text
validate_input -> build_ir -> lower_to_backend -> execute -> postprocess -> explain_result
```

对应到代码接口，可以抽象为：

```text
OperatorSpec
  id
  domain
  input_schema
  output_schema
  problem_form
  quantum_primitive
  supported_backends
  resource_profile
  validation_rules
  lowering_rules
  result_contract
```

这样，上层 `DecisionProgram` 或 `ValuationProgram` 可以先落到稳定 IR，再选择具体算子执行，而不是直接绑定某个 solver 类。

## 2. 为什么需要 IR

### 统一金融问题和量子算法

金融问题通常是业务语义驱动的，例如“求组合权重”“计算风险贡献”“估计期权价格”“选择指数追踪成分股”。量子算法则要求输入已经变成线性系统、QUBO、Ising Hamiltonian、幅度估计问题或量子电路。

IR 的作用是记录这一步转换：

```text
金融问题 -> 结构化问题 -> 量子金融算子 IR -> 后端可执行对象
```

没有 IR 时，上层应用很容易直接依赖某个算法类的参数细节，导致业务建模、算法选择、后端执行和结果解释耦合在一起。

### 让算子可以被验证

量子金融算法有很多前置条件，例如：

- HHL 要求矩阵是方阵，并且通常需要 Hermitian 或可嵌入为 Hermitian。
- Quantum Shapley 要求编码的边际贡献非负。
- QUBO/QAOA 要求二进制变量和二次目标。
- QAE 要求目标概率和后处理缩放关系清楚。

IR 可以把这些条件变成显式校验，而不是分散在 `assert`、异常和文档说明里。这样做的好处是：错误能在执行前暴露，实验结果也更容易复现和解释。

### 支撑后端切换

同一个金融算子可能有多种执行方式：

- 精确经典 baseline
- statevector 模拟
- shots 抽样
- QAE
- QAOA
- SamplingVQE
- 未来真实量子后端

IR 可以把“问题是什么”和“怎么执行”分开。比如风险归因算子可以保持同一个 SRC 输出契约，但在执行层选择精确 Shapley、Monte Carlo Shapley、Quantum Shapley statevector 或 QAE。

### 支撑资源估计和实验比较

量子金融实验经常需要比较：

- qubit 数
- circuit depth
- CNOT 数
- shots 数
- oracle calls
- approximation error
- runtime

如果没有 IR，这些元数据只能临时散落在实验脚本里。IR 应该给每次算子执行保留统一资源字段，使实验图表、论文指标和软件日志都来自同一个结构。

## 3. IR 应该表达什么

建议第一版量子金融算子 IR 至少包含以下层次。

### 金融语义层

描述这个算子解决的金融问题：

```text
domain: portfolio | risk | option_pricing | index_tracking | settlement | fraud
task: optimization | attribution | valuation | search
inputs: returns, covariance, expected_returns, payoff, constraints
outputs: weights, SRC, option_price, selected_assets
assumptions: equal_weight, historical_es, risk_neutral_measure
```

### 数学问题层

描述金融问题被转换成的数学形式：

```text
problem_form: linear_system | qubo | ising | amplitude_estimation | shapley_game
objective
constraints
value_function
normalization
scaling
```

### 量子原语层

描述使用哪类量子算法：

```text
primitive: hhl | qaoa | sampling_vqe | qae | quantum_shapley | grover | shor
circuit_family
register_layout
oracle_definition
state_preparation
measurement_rule
```

### 后端执行层

描述怎么运行：

```text
backend: classical_exact | statevector | shots | qae_iqae | qae_mlqae | hardware
shots
seed
optimizer
transpiler_options
error_tolerance
```

### 结果解释层

描述如何把底层结果还原为金融输出：

```text
raw_result
postprocess_rule
financial_result
confidence_interval
resource_profile
validation_report
```

## 4. 当前仓库与 IR 的关系

当前仓库已经具备 IR 的几个组成部分，但还没有把它们收束为一个独立模块：

- `double_quant.programming` 提供金融问题层的雏形，可以表达 `DecisionProgram` 和 `ValuationProgram`。
- `DecisionProgram.to_qubo_problem()` 和 `DecisionProgram.to_linear_system()` 已经是早期 lowering 逻辑。
- `QUBOProblem`、`IsingProblem`、`LinearSystem` 已经是数学问题层的基础模型。
- `QuantumShapleyCalculator` 和 `HHLSolver` 内部已经构造量子电路，但电路还没有作为公开 IR 产物暴露。
- `QUBOSolverResult` 已经是统一结果对象的雏形，但 HHL 和 Shapley 还没有同等结构化结果。

因此，下一步不是重写算法，而是抽出一层 `quantumoperator` IR，把现有算法包装成可注册、可校验、可执行、可解释的算子。

## 5. 建议的第一版范围

第一版建议保持克制，只覆盖当前源码中已经存在的能力：

1. `LinearSystemSolveOperator`
   - 输入：`LinearSystem`
   - 后端：HHL/SAPO statevector
   - 输出：解向量、缩放信息、成功概率、资源估计

2. `RiskAttributionOperator`
   - 输入：资产收益率、alpha、mode
   - 后端：exact Shapley、MC Shapley、Quantum Shapley、QAE
   - 输出：SRC、风险贡献占比、oracle calls、误差指标

3. `BinaryOptimizationOperator`
   - 输入：`QUBOProblem` 或 `IsingProblem`
   - 后端：exact baseline、QAOA、SamplingVQE
   - 输出：best bitstring、objective、energy、probability、optimizer metadata

4. `MeasureOperator`
   - 输入：金融时间序列或 payoff 场景
   - 后端：经典度量函数，后续扩展 QAE
   - 输出：ES、option price、risk contribution 等估值结果

这些算子先不做自动自然语言建模，不做复杂表达式解析，也不承诺真实硬件执行。第一版重点是把已有能力结构化，避免继续把业务逻辑、算法逻辑和实验逻辑混在一起。

## 6. 后续扩展方向

当第一版 IR 稳定后，可以继续扩展：

- 独立 QAE 金融估值算子，用于欧式期权、VaR/ES、期望收益估计。
- Grover/SFS 搜索算子，用于 DeFi 策略、网点选址、设施选择。
- Rasengan 约束可行空间算子，用于指数追踪、贷款特征选择、结算净额优化。
- Shor/order-finding 算子，用于账本安全参数或模数结构分析。
- 电路可视化、量子态演化可视化和计算过程追踪。
- 自动校验、自动降级、自动修正和算子拼接。

这些能力都应建立在同一个 IR 上，否则每个新算法都会带来一套新的输入格式、结果格式和实验统计方式。

## 结论

量子金融算子 IR 的核心价值是把“金融语义”和“量子执行”解耦。它让算子库不仅是一组算法函数，而是一套可以注册、验证、转换、执行和解释的计算协议。

对当前仓库来说，IR 是把 HHL/SAPO、Quantum Shapley、QUBO/QAOA/VQE 和金融应用工作流整合成真正量子金融编程框架的关键中间层。
