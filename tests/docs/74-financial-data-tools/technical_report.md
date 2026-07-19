# 74 3种及以上现有金融数据工具对接兼容测试（Func-42） 技术报告

## 技术目标

本功能对应新表格中的功能 74：

```text
对接现有金融数据工具3种及以上
```

该功能只关注外部金融数据工具的接入数量和接口统一性，不承担数据分析和量子编解码流程，后者已拆分到功能 75。

## 统一接口设计

项目的数据源接口定义为：

```python
class PriceSource(Protocol):
    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...
```

统一输出约定：

- `DatetimeIndex`：表示时间序列日期。
- `columns=tickers`：每列对应一个金融标的或金融序列。
- `values=close prices / adjusted close / indicator values`：数值型数据。

统一接口的价值在于：provider 之间的列名、日期格式、下载方式、股票代码规则都被封装在 adapter 内部，下游模块不需要知道数据来自哪一个工具。

## 已接入工具 1：Yahoo Finance / yfinance

实现类：`YFinanceSource`

实现要点：

1. 调用 `yf.download(tickers, start, end, auto_adjust=...)`。
2. 对 yfinance 常见 `MultiIndex` 列结构做解析。
3. 优先选择 `Adj Close`，否则选择 `Close`。
4. 清理缺失值并可写入 CSV 缓存。

适用数据：

- 美股价格
- ETF 价格
- 用于风险归因和投资组合优化的历史价格矩阵

## 已接入工具 2：AKShare

实现类：`AKShareSource`

实现要点：

1. 调用 `ak.stock_zh_a_hist(...)` 获取 A 股历史行情。
2. 将日期转换为 AKShare 所需的 `YYYYMMDD` 格式。
3. 将 `sz000001`、`sh600000` 等输入规整为 AKShare 使用的纯数字 symbol。
4. 从中文字段 `日期`、`收盘` 中提取时间序列。
5. 输出为统一价格矩阵。

适用数据：

- A 股股票历史价格
- 中国市场组合优化
- 中国市场风险归因

## 已接入工具 3：pandas-datareader

实现类：`PandasDataReaderSource`

实现要点：

1. 调用 `pandas_datareader.data.DataReader(...)`。
2. 支持当前公开维护的数据源，如 FRED、Fama/French、OECD、Eurostat、Bank of Canada、Econdb。
3. 对 `Series`、`DataFrame` 和多表返回值做规范化处理。
4. 多表返回值要求显式传入 `table`，避免误选数据表。

适用数据：

- 宏观利率
- 市场指标
- Fama/French 因子
- 因子增强风险模型

## 已接入工具 4：Stooq

实现类：`StooqSource`

实现要点：

1. 构造 Stooq 历史 CSV 下载 URL。
2. 将裸 ticker 默认转换为 `.US` 后缀，例如 `AAPL -> aapl.us`。
3. 提取 CSV 中 `Date` 和 `Close` 字段。
4. 如果 Stooq 返回非 CSV 页面，则明确抛出错误，避免把网页误当数据。

适用数据：

- 无 API key 的股票价格数据补充
- 与 Yahoo Finance 形成交叉验证的数据源

## 测试策略

测试文件 `74-financial_data_tools.py` 对四类工具全部使用 monkeypatch 替身返回，原因是验收测试应可重复、稳定、无外部网络依赖。

测试重点不是验证第三方服务是否在线，而是验证本项目 adapter 是否能：

1. 正确调用 provider 入口。
2. 正确处理 provider 返回结构。
3. 正确归一化为统一 `DataFrame`。
4. 满足三种以上金融工具接入要求。

## 技术结论

功能 74 已完成。当前项目接入金融数据工具数量为 4，严格满足“三种及以上”要求，并且都统一在 `PriceSource` 协议下，后续可以直接服务功能 65 的数据获取、分析和编解码流程。

## 技术原理补充：统一价格矩阵与数据源规范化

四类外部数据源的共同目标不是简单“能下载”，而是把不同供应商的行情结构规约为同一个金融时间序列算子输入。统一后的价格矩阵记为

$$
P=\left[p_{t,i}\right]_{T\times N},\quad t=1,\ldots,T,\ i=1,\ldots,N,
$$

其中 $p_{t,i}$ 表示第 $t$ 个交易日第 $i$ 个资产的收盘价或指标值。后续收益率、协方差、风险归因与组合优化均只依赖 $P$ 的形状和数值语义，而不依赖数据来自 Yahoo Finance、AKShare、pandas-datareader 还是 Stooq。

供应商适配器执行三类规范化：

1. **列语义规范化**：yfinance 的 `Adj Close/Close`、AKShare 的中文字段 `收盘`、Stooq 的 `Close` 均被映射为资产列；pandas-datareader 的宏观序列也被转换为一列或多列数值时间序列。
2. **时间索引规范化**：日期字符串、`YYYYMMDD`、CSV `Date` 字段统一转换为 `DatetimeIndex`，使时间对齐可由 pandas 的索引机制完成。
3. **缺失值规范化**：按阈值保留有效列，再执行前向填充和末端缺失剔除。其数学含义是只保留满足

$$
\frac{\#\{t:p_{t,i}\neq \mathrm{NaN}\}}{T}\ge \tau
$$

的资产列，避免稀疏或失效资产进入后续协方差估计。

该设计的关键点在于把“数据源差异”封装在适配器内部，把对下游算法可见的数据类型收敛为 `PriceSource.fetch -> DataFrame`。因此验收脚本使用 monkeypatch 并不削弱测试有效性：测试关注的是结构规约、字段抽取、异常拦截和缓存路径，而不是第三方网站当日是否可访问。

## 与后续量子金融流程的接口关系

统一价格矩阵 $P$ 会继续进入收益率变换

$$
r_{t,i}=\log\frac{p_{t,i}}{p_{t-1,i}},
$$

并形成期望收益向量 $\mu_i=\frac{1}{T-1}\sum_t r_{t,i}$ 与协方差矩阵

$$
\Sigma_{ij}=\frac{1}{T-2}\sum_t (r_{t,i}-\mu_i)(r_{t,j}-\mu_j).
$$

因此，功能 74 的数据源对接是后续量子态编码、HHL 组合优化和 Shapley 风险归因的输入基础。只要四个适配器都稳定输出同一矩阵契约，后续模块就可以把金融数据看作可线性代数处理的张量，而不是面向多个供应商分别编写流程。

