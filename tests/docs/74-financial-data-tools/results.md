# 74 3种及以上现有金融数据工具对接兼容测试（Func-42） 测试结果

## 运行命令

```bash
uv run pytest tests/double_quant/data/74-financial_data_tools.py -s
```

## 运行结果

```text
执行状态：通过
收集用例：1

[数据源接入] Yahoo Finance / yfinance: shape=(5, 2), columns=[AAPL, MSFT]
[数据源接入] AKShare: shape=(5, 2), columns=[sz000001, sh600000]
[数据源接入] pandas-datareader / FRED: shape=(5, 2), columns=[DGS10, VIXCLS]
[数据源接入] Stooq: shape=(5, 2), columns=[AAPL, MSFT]
[验收结论] 已对接金融数据工具数量：4，满足 3 种及以上要求。
通过数量：1
```

## 输出说明

- 本功能验证金融数据工具对接数量，不包含下游分析和量子编解码流程。
- 已验证 4 类现有金融数据工具或服务接口：`yfinance`、AKShare、`pandas-datareader`、Stooq。
- 四类接口均被归一化为统一 `PriceSource.fetch(tickers, start, end)` 输出格式：日期索引、金融标的列、数值型时间序列。
- 测试使用替身数据和 monkeypatch，不访问真实外部网络端点，保证验收测试稳定可复现。

