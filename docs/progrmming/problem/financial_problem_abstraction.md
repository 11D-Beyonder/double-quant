# 金融问题接口定义

## 目标

金融问题层只负责把问题说清楚，不负责描述怎么求解。

当前接口保留两类问题：

- `DecisionProblem`：求变量、方案或可行解。可以有目标函数，也可以只有约束。
- `ValuationProblem`：计算金融度量，例如价格、VaR、ES、风险贡献、违约概率、欺诈分数。

接口设计要满足两个要求：

1. 与当前算法能力兼容：QAOA/QUBO、HHL/SAPO、Shor 类分解、风险归因、QAE 类估值。
2. 方便快速解析：不要依赖 `"for every asset"` 这类自然语言字符串，也尽量不要解析任意公式字符串。

因此 `DecisionProblem` 采用接近 Gurobi 的建模方式：变量是对象，表达式是对象，比较运算生成约束对象。

## 分类原则

判断问题类型时只看输出是什么：

| 输出是什么 | 问题类型 |
|---|---|
| 权重、选择变量、审批动作、可行方案、因子、未知变量 | `DecisionProblem` |
| 价格、风险值、风险贡献、概率、分数、期望收益/损失 | `ValuationProblem` |

补充规则：

- 方程问题不单独建类。方程就是没有目标函数的 `DecisionProblem`，目标是找到满足约束的变量。
- Shor 类质因数分解不是状态转移问题，而是求满足乘法约束的因子变量，属于 `DecisionProblem`。
- 原本就是估值、计量、定价、风险贡献或评分的问题，保持为 `ValuationProblem`。

## 公共接口

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Self


ProblemKind = Literal["decision", "valuation"]


@dataclass
class FinancialProblem:
    name: str
    kind: ProblemKind
    domain: str
    data: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_data(self, name: str, value: Any) -> Self:
        ...

    def add_parameter(self, name: str, value: Any) -> Self:
        ...

    def add_assumption(self, assumption: str) -> Self:
        ...

    def add_output(self, output: str) -> Self:
        ...
```

字段含义：

| 字段 | 含义 |
|---|---|
| `name` | 问题名称或稳定 ID |
| `kind` | `decision` 或 `valuation` |
| `domain` | 金融场景，例如 `portfolio`、`risk`、`option_pricing`、`ledger` |
| `data` | 输入数据，可以是数组、DataFrame、图、账本状态或数据引用 |
| `parameters` | 业务参数，例如置信水平、目标收益率、行权价、期限、整数 N |
| `assumptions` | 建模假设，例如等权、无交易成本、固定利率 |
| `outputs` | 期望输出的金融结果名称 |

## DecisionProblem

`DecisionProblem` 表示“找到满足条件的变量或方案”。

它统一覆盖：

1. 优化问题：有目标函数和约束。
2. 方程问题：没有目标函数，只有等式或不等式约束。
3. 搜索/分解问题：找到满足代数、业务或政策约束的变量。

数学形式：

$$
\text{find } x
$$

满足：

$$
g_j(x;D,\theta)\le 0,\quad h_k(x;D,\theta)=0
$$

如果有目标函数，再加：

$$
\min/\max F(x;D,\theta)
$$

### 表达式对象

不要把约束写成自然语言字符串。第一版建议用很小的表达式对象集合：

```python
VariableType = Literal["continuous", "binary", "integer"]
DecisionSense = Literal["find", "minimize", "maximize"]


@dataclass
class VariableSpec:
    name: str
    length: int
    vtype: VariableType = "continuous"
    lb: float | int | None = None
    ub: float | int | None = None


class Var:
    name: str
    index: int | None


class VarArray:
    name: str
    length: int

    def __getitem__(self, index: int) -> Var:
        ...


class Expr:
    ...


class ConstraintExpr:
    ...
```

表达式辅助函数：

```python
def sum_(items) -> Expr:
    ...


def dot(coefficients, variables: VarArray) -> Expr:
    ...


def quad_form(variables: VarArray, matrix) -> Expr:
    ...


def matmul(matrix, variables: VarArray) -> Expr:
    ...


def mean(expr: Expr) -> Expr:
    ...


def square(expr: Expr) -> Expr:
    ...
```

比较运算生成约束对象：

```python
sum_(w) == 1
dot(mu, w) >= target_return
w[i] <= z[i]
p * q == N
```

这些对象内部应保存结构化 AST，不需要再解析字符串。

### 接口

```python
@dataclass
class DecisionProblem(FinancialProblem):
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    constraints: list[ConstraintExpr] = field(default_factory=list)
    objective: Expr | None = None
    sense: DecisionSense = "find"

    def add_variables(
        self,
        name: str,
        length: int,
        *,
        vtype: VariableType = "continuous",
        lb: float | int | None = None,
        ub: float | int | None = None,
    ) -> VarArray:
        ...

    def add_constraint(self, expression: ConstraintExpr) -> Self:
        ...

    def add_constraints(self, expressions: list[ConstraintExpr]) -> Self:
        ...

    def set_objective(
        self,
        expression: Expr | None,
        *,
        sense: DecisionSense = "find",
    ) -> Self:
        ...
```

`add_variables(name, length)` 的 `name` 和 `length` 是两个必填参数。`vtype/lb/ub` 是可选项，用来对齐当前算法能力：

- QUBO/QAOA：优先使用 `vtype="binary"`。
- HHL/SAPO：使用连续变量和线性等式约束。
- Shor 类分解：使用整数变量和乘法等式约束。

变量展开约定：

```python
w = problem.add_variables("w", len(assets))
# w[0], w[1], ..., w[n-1]

p = problem.add_variables("p", 1, vtype="integer", lb=2, ub=N - 1)[0]
# scalar variable p
```

约束展开约定：

```python
problem.add_constraints([w[i] >= 0.0 for i in range(len(assets))])
problem.add_constraints([w[i] <= 1.0 for i in range(len(assets))])
```

不要把“所有资产都满足某条件”写成一条自然语言约束；必须按索引展开成一组 `ConstraintExpr`。

## DecisionProblem 示例

### 最优投资组合：优化形式

```python
problem = DecisionProblem(
    name="func_1_portfolio_decision",
    kind="decision",
    domain="portfolio",
)
problem.add_data("expected_returns", mu)
problem.add_data("covariance", sigma)
problem.add_data("assets", assets)
problem.add_parameter("target_return", target_return)

w = problem.add_variables("w", len(assets), vtype="continuous")

problem.add_constraint(sum_(w) == 1.0)
problem.add_constraint(dot(mu, w) >= target_return)
problem.add_constraints([w[i] >= 0.0 for i in range(len(assets))])
problem.add_constraints([w[i] <= 1.0 for i in range(len(assets))])

problem.set_objective(quad_form(w, sigma), sense="minimize")
problem.add_output("weights")
```

这类形式可以适配：

- 连续优化/线性系统路线：从一阶条件生成方程。
- 离散化后优化路线：若权重离散化，可进一步转为二进制变量。

### 最优投资组合：方程形式

方程问题不需要单独建类。它是 `sense="find"` 的 `DecisionProblem`。

```python
problem = DecisionProblem(
    name="func_1_portfolio_equation",
    kind="decision",
    domain="portfolio",
)
problem.add_data("expected_returns", mu)
problem.add_data("covariance", sigma)
problem.add_data("assets", assets)
problem.add_parameter("target_return", target_return)

w = problem.add_variables("w", len(assets), vtype="continuous")
lambda_return = problem.add_variables("lambda_return", 1, vtype="continuous")[0]
lambda_budget = problem.add_variables("lambda_budget", 1, vtype="continuous")[0]

problem.add_constraints([
    dot(sigma[i], w) + lambda_return * mu[i] + lambda_budget == 0.0
    for i in range(len(assets))
])
problem.add_constraint(dot(mu, w) == target_return)
problem.add_constraint(sum_(w) == 1.0)

problem.set_objective(None, sense="find")
problem.add_output("weights")
```

这类形式与当前组合优化 HHL/SAPO 路线兼容：适配层可以从线性等式约束抽取矩阵和右端项。

### 指数追踪

```python
problem = DecisionProblem(
    name="func_10_index_tracking",
    kind="decision",
    domain="index_tracking",
)
problem.add_data("asset_returns", asset_returns)
problem.add_data("benchmark_returns", benchmark_returns)
problem.add_data("assets", assets)
problem.add_parameter("max_assets", max_assets)

z = problem.add_variables("z", len(assets), vtype="binary")
w = problem.add_variables("w", len(assets), vtype="continuous", lb=0.0)

problem.add_constraint(sum_(z) <= max_assets)
problem.add_constraint(sum_(w) == 1.0)
problem.add_constraints([w[i] <= z[i] for i in range(len(assets))])

tracking_error = mean(square(matmul(asset_returns, w) - benchmark_returns))
problem.set_objective(tracking_error, sense="minimize")
problem.add_output("selected_assets")
problem.add_output("weights")
problem.add_output("tracking_error")
```

这类形式与 QAOA/QUBO 类优化路线兼容：二进制变量 `z`、线性约束和二次目标都可以被适配层识别。

### Shor 类质因数分解

如果账本场景实际落到 Shor 类能力上，本质是求因子变量，不是状态转移问题。

```python
problem = DecisionProblem(
    name="func_4_ledger_factorization",
    kind="decision",
    domain="ledger",
)
problem.add_data("N", N)

p = problem.add_variables("p", 1, vtype="integer", lb=2, ub=N - 1)[0]
q = problem.add_variables("q", 1, vtype="integer", lb=2, ub=N - 1)[0]

problem.add_constraint(p * q == N)
problem.set_objective(None, sense="find")
problem.add_output("p")
problem.add_output("q")
```

这类形式与 Shor 类算法适配：输出是未知因子，不是估值结果，因此不应建成 `ValuationProblem`。

### 贷款审批

```python
problem = DecisionProblem(
    name="func_8_loan_decision",
    kind="decision",
    domain="loan_decision",
)
problem.add_data("applicant_features", features)
problem.add_data("default_probability", pd)
problem.add_parameter("approval_threshold", approval_threshold)

approve = problem.add_variables("approve", 1, vtype="binary")[0]

problem.add_constraint(approve <= indicator(pd <= approval_threshold))
problem.set_objective(expected_profit(approve) - expected_loss(approve), sense="maximize")
problem.add_output("approve")
```

这里的 `indicator(...)`、`expected_profit(...)`、`expected_loss(...)` 也应返回表达式对象，不能是需要后续解析的自然语言字符串。

## ValuationProblem

`ValuationProblem` 保持原来的定位：计算金融度量。

估值问题不应通过解析 `formula` 字符串来决定如何计算。`measure` 字段应使用可导入的度量类，这样适配层可以直接按类分派到已有函数或应用模块。

示例度量类：

```python
from double_quant.problem.measures import (
    ExpectedShortfallMeasure,
    ShapleyRiskContributionMeasure,
    EuropeanCallPriceMeasure,
    FraudScoreMeasure,
)
```

度量类协议：

```python
class MeasureFunction:
    name: str

    @classmethod
    def required_data(cls) -> set[str]:
        ...

    @classmethod
    def required_parameters(cls) -> set[str]:
        ...

    @classmethod
    def evaluate(cls, data: dict[str, Any], parameters: dict[str, Any]) -> Any:
        ...
```

如果需要配置化保存，可以存储类的 import path，例如：

```python
"double_quant.problem.measures.ExpectedShortfallMeasure"
```

但 Python API 中优先直接传入类对象：

```python
problem.set_measure(ExpectedShortfallMeasure, target="portfolio")
```

接口：

```python
@dataclass
class ValuationProblem(FinancialProblem):
    target: str = ""
    measure: type[MeasureFunction] | None = None
    breakdown: str | None = None

    def set_measure(
        self,
        measure: type[MeasureFunction],
        *,
        target: str,
        breakdown: str | None = None,
    ) -> Self:
        ...
```

字段约定：

- `target`：被计量对象，例如 portfolio、option、customer、transaction。
- `measure`：可导入的度量类，不是自由字符串。
- `breakdown`：可选，表示按什么对象拆分，例如 asset、account、factor。

### Expected Shortfall

```python
from double_quant.problem.measures import ExpectedShortfallMeasure

problem = ValuationProblem(
    name="func_2_expected_shortfall",
    kind="valuation",
    domain="risk",
)
problem.add_data("portfolio_returns", portfolio_returns)
problem.add_parameter("alpha", 0.95)
problem.set_measure(
    ExpectedShortfallMeasure,
    target="portfolio",
)
problem.add_output("expected_shortfall")
```

### 风险贡献

```python
from double_quant.problem.measures import ShapleyRiskContributionMeasure

problem = ValuationProblem(
    name="func_2_risk_contribution",
    kind="valuation",
    domain="risk_attribution",
)
problem.add_data("asset_returns", returns_df)
problem.add_data("assets", assets)
problem.add_parameter("alpha", 0.95)
problem.add_parameter("portfolio_rule", "equal_weight")
problem.set_measure(
    ShapleyRiskContributionMeasure,
    target="portfolio",
    breakdown="asset",
)
problem.add_output("risk_contribution")
```

### 期权定价

```python
from double_quant.problem.measures import EuropeanCallPriceMeasure

problem = ValuationProblem(
    name="func_3_option_pricing",
    kind="valuation",
    domain="option_pricing",
)
problem.add_data("terminal_price_scenarios", terminal_prices)
problem.add_parameter("strike", 100.0)
problem.add_parameter("maturity", "1Y")
problem.add_parameter("risk_free_rate", 0.03)
problem.set_measure(
    EuropeanCallPriceMeasure,
    target="european_call_option",
)
problem.add_output("price")
```

## 当前金融功能映射

| 功能 | 问题类 | 最小金融定义 |
|---|---|---|
| Func-1 最优投资组合 | `DecisionProblem` | 权重变量、收益、协方差、预算约束、目标收益、风险目标；方程形式也用 `sense="find"` |
| Func-2 风险价值计量 | `ValuationProblem` | 收益或损失序列、置信水平、VaR/ES；风险贡献用 `breakdown="asset"` |
| Func-3 金融衍生品定价 | `ValuationProblem` | 状态变量、合约条款、payoff 模板、贴现口径、价格输出 |
| Func-4 动态账本更新算法 | `DecisionProblem` | 若使用 Shor 类能力，则定义为求满足 `p*q=N` 的整数因子 |
| Func-5 去中心化金融管理 | `DecisionProblem` | 管理动作、路径、策略或资金分配是要求解的方案 |
| Func-6 反欺诈监测 | `DecisionProblem` | 识别、拦截或标记可疑交易是决策；若只计算分数，可另建 `ValuationProblem` |
| Func-7 支付与结算系统 | `DecisionProblem` | 结算顺序、净额方案、流动性分配是可行方案或最优方案 |
| Func-8 贷款发放决策 | `DecisionProblem` | 发放/拒绝是决策变量；违约概率可作为输入或辅助估值 |
| Func-9 银行网点布局优化 | `DecisionProblem` | 候选网点、需求点、覆盖约束、容量、建设成本 |
| Func-10 指数追踪 | `DecisionProblem` | 成分股、基准收益、选择变量、权重变量、跟踪误差 |

## 当前算法兼容性

| 能力方向 | 对应问题 | 解析要求 |
|---|---|---|
| QUBO/QAOA 类优化 | `DecisionProblem` | `binary` 变量、线性约束、线性/二次目标可被识别 |
| HHL/SAPO 类线性系统 | `DecisionProblem` | 连续变量、线性等式约束、无目标或一阶条件形式 |
| Shor 类分解 | `DecisionProblem` | 整数变量、乘法等式 `p*q=N`、输出因子 |
| 风险计量 | `ValuationProblem` | `measure=ExpectedShortfallMeasure` 或 VaR 度量类，按类分派 |
| 风险归因 | `ValuationProblem` | `measure=ShapleyRiskContributionMeasure`，`breakdown="asset"` |
| 期权定价/QAE 类估值 | `ValuationProblem` | `measure=EuropeanCallPriceMeasure` 或其他 payoff 度量类 |

## 实现建议

第一版只需要一个轻量包：

```text
double_quant.problem
  base.py
  decision.py
  valuation.py
  expression.py
  measures.py
```

优先实现：

1. `FinancialProblem`
2. `DecisionProblem`
3. `ValuationProblem`
4. `Var`、`VarArray`、`Expr`、`ConstraintExpr`
5. `sum_`、`dot`、`quad_form` 等少量表达式构造函数
6. `ExpectedShortfallMeasure`、`ShapleyRiskContributionMeasure`、`EuropeanCallPriceMeasure` 等可导入度量类

第一版不要做：

- 自然语言约束解析。
- 复杂表达式解析器。
- 任意 `formula` 字符串解析。
- 自动建模或自动编译。
- 求解器参数配置。
- 为每个金融场景单独建类。

先保证问题对象能被快速、稳定地解析，再在适配层映射到当前算法对象。
