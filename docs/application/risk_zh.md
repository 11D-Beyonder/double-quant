# 风险归因 Quick Start

这篇文档提供一个最小可运行示例，演示如何通过 Double Quant 的接口完成一次基于 Shapley Value 的 Expected Shortfall 风险归因。

如果你已经有中间数据，也可以直接跳过前面的步骤：

- 已有价格数据 `prices: pd.DataFrame`：从“步骤 2”开始
- 已有收益率数据 `returns_df: pd.DataFrame`：从“步骤 3”开始

## 1. 最小可运行示例

```python
from double_quant import (
    BinaryEnumerationCalculator,
    RiskAttributor,
    YFinanceSource,
    to_log_returns,
)

# 步骤 1：下载收盘价数据
prices = YFinanceSource().fetch(
    ["AAPL", "MSFT", "NVDA"],
    start="2020-04-01",
    end="2022-04-01",
)

# 步骤 2：把价格转换成对数收益率
returns_df = to_log_returns(prices)

# 步骤 3：执行风险归因
attributor = RiskAttributor(
    returns_df=returns_df,
    solver_class=BinaryEnumerationCalculator,
    mode="es",
    alpha=0.95,
)
src = attributor.attribute()

for asset, contribution in src.items():
    print(f"{asset}: {contribution:.6f}")
```

运行后，`src` 会是一个 `dict[str, float]`，键是资产代码，值是对应资产的 Shapley Risk Contribution。

## 2. 数据格式说明

### 步骤 1 的输出：价格数据 `prices`

`YFinanceSource().fetch(...)` 返回一个 `pandas.DataFrame`：

- 行索引：`DatetimeIndex`
- 列名：资产代码，例如 `AAPL`、`MSFT`
- 单元格：对应日期的收盘价或复权收盘价

示意：

```python
                 AAPL    MSFT    NVDA
2020-04-01    60.23   153.63   6.75
2020-04-02    60.91   158.19   6.96
2020-04-03    59.04   153.83   6.70
```

如果你已经有这种格式的价格表，可以直接调用：

```python
returns_df = to_log_returns(prices)
```

### 步骤 2 的输出：收益率数据 `returns_df`

`to_log_returns(prices)` 会把价格表转换成对数收益率表，返回的仍然是 `pandas.DataFrame`：

- 行索引：时间索引
- 列名：资产代码
- 单元格：该资产在相邻两个时点之间的对数收益率

示意：

```python
                 AAPL      MSFT      NVDA
2020-04-02    0.0112    0.0292    0.0306
2020-04-03   -0.0312   -0.0280   -0.0381
2020-04-06    0.0838    0.0678    0.0784
```

如果你已经有 `returns_df`，并且满足下面这几个条件，就可以直接传给 `RiskAttributor`：

- 类型是 `pandas.DataFrame`
- 每一列对应一个资产
- 每一行对应一个时间点的收益率观测
- 列名是资产名称
- 数值列是可用于计算 ES 的浮点数

## 3. `RiskAttributor` 做了什么

`RiskAttributor` 会把每个资产看作合作博弈中的一个玩家，然后根据给定的价值函数计算每个资产对组合尾部风险的贡献。

在当前实现里：

- `mode="es"`：直接以 Expected Shortfall 作为价值函数
- `mode="rs"`：使用 Risk Saving 变换后的价值函数
- `alpha=0.95`：表示计算 95% 置信水平下的 ES

`solver_class=BinaryEnumerationCalculator` 表示使用经典精确算法进行 Shapley Value 计算，适合作为最直接的入门方式。

## 4. 如果你想跳过某些步骤

### 只跳过数据下载

如果你已经从别的地方拿到了价格数据，只要它是列为资产、行为日期的 `DataFrame`，就可以直接从这里开始：

```python
from double_quant import BinaryEnumerationCalculator, RiskAttributor, to_log_returns

returns_df = to_log_returns(prices)

src = RiskAttributor(
    returns_df=returns_df,
    solver_class=BinaryEnumerationCalculator,
    mode="es",
).attribute()
```

### 连数据转换也跳过

如果你已经有收益率表 `returns_df`，可以直接归因：

```python
from double_quant import BinaryEnumerationCalculator, RiskAttributor

src = RiskAttributor(
    returns_df=returns_df,
    solver_class=BinaryEnumerationCalculator,
    mode="es",
).attribute()
```

## 5. 关于 `mode` 的选择

入门时可以先记住一条简单规则：

- 经典求解器可以使用 `mode="es"` 或 `mode="rs"`
- 量子求解器必须使用 `mode="rs"`

原因是量子 Shapley 求解器要求价值函数满足超可加性，而原始的 ES 是次可加的。`mode="rs"` 会先做一次 Risk Saving 变换，再恢复成最终的风险贡献结果。

如果你后续切换到量子求解器，请特别注意这一点。

## 6. 相关接口

- `double_quant.data.source.YFinanceSource`
- `double_quant.data.transform.to_log_returns`
- `double_quant.application.risk.RiskAttributor`
- `double_quant.algorithm.shapley.calculator.BinaryEnumerationCalculator`

如果你想了解风险归因背后的数学定义，可以继续阅读 [risk.md](./risk.md)。
