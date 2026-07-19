from __future__ import annotations

import numpy as np
import pandas as pd

from double_quant.data.source import (
    AKShareSource,
    PandasDataReaderSource,
    StooqSource,
    YFinanceSource,
)


def _price_frame(columns: list[str], base: float) -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    )
    values = {
        column: np.array([base, base + 1.2, base + 0.8, base + 2.4, base + 3.1])
        + index
        for index, column in enumerate(columns)
    }
    return pd.DataFrame(values, index=dates)


def _assert_source_contract(name: str, data: pd.DataFrame, columns: list[str]) -> None:
    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.columns) == columns
    assert data.shape[0] == 5
    assert data.notna().all().all()
    print(f"[数据源接入] {name}: shape={data.shape}, columns={list(data.columns)}")


def test_financial_data_tools_fetch_unified_frames(monkeypatch):
    from double_quant.data import source as source_module

    yahoo_prices = _price_frame(["AAPL", "MSFT"], 100.0)

    def fake_yfinance_download(tickers, start, end, auto_adjust):
        assert tickers == ["AAPL", "MSFT"]
        assert start == "2020-01-01"
        assert end == "2020-01-09"
        assert auto_adjust is False
        columns = pd.MultiIndex.from_product([["Adj Close"], yahoo_prices.columns])
        return pd.DataFrame(yahoo_prices.to_numpy(), index=yahoo_prices.index, columns=columns)

    def fake_akshare_hist(**kwargs):
        symbol = kwargs["symbol"]
        assert kwargs["period"] == "daily"
        assert kwargs["start_date"] == "20200101"
        assert kwargs["end_date"] == "20200109"
        close = [10.0, 10.3, 10.2, 10.8, 11.0]
        if symbol == "600000":
            close = [9.0, 9.2, 9.4, 9.8, 10.1]
        return pd.DataFrame(
            {
                "日期": yahoo_prices.index.strftime("%Y-%m-%d").tolist(),
                "股票代码": [symbol] * len(yahoo_prices),
                "开盘": close,
                "收盘": close,
                "最高": close,
                "最低": close,
                "成交量": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def fake_data_reader(*args, **kwargs):
        assert args == (["DGS10", "VIXCLS"], "fred")
        assert kwargs["start"] == "2020-01-01"
        assert kwargs["end"] == "2020-01-09"
        return pd.DataFrame(
            {
                "DGS10": [1.88, 1.80, 1.81, 1.83, 1.85],
                "VIXCLS": [12.5, 14.0, 13.7, 13.1, 12.9],
            },
            index=yahoo_prices.index,
        )

    def fake_stooq_csv(url: str, timeout: float) -> pd.DataFrame:
        assert timeout == 15.0
        assert "d1=20200101" in url
        assert "d2=20200109" in url
        close = [300.0, 301.2, 300.8, 302.4, 303.1]
        if "msft.us" in url:
            close = [160.0, 161.3, 162.0, 161.7, 163.4]
        return pd.DataFrame(
            {
                "Date": yahoo_prices.index.strftime("%Y-%m-%d").tolist(),
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [2000, 2100, 2200, 2300, 2400],
            }
        )

    monkeypatch.setattr(source_module.yf, "download", fake_yfinance_download)
    monkeypatch.setattr(source_module.ak, "stock_zh_a_hist", fake_akshare_hist)
    monkeypatch.setattr(source_module.pdr_data, "DataReader", fake_data_reader)
    monkeypatch.setattr(source_module, "_read_stooq_csv", fake_stooq_csv)

    yfinance_data = YFinanceSource().fetch(["AAPL", "MSFT"], "2020-01-01", "2020-01-09")
    akshare_data = AKShareSource().fetch(
        ["sz000001", "sh600000"], "2020-01-01", "2020-01-09"
    )
    pandas_reader_data = PandasDataReaderSource(data_source="fred").fetch(
        ["DGS10", "VIXCLS"], "2020-01-01", "2020-01-09"
    )
    stooq_data = StooqSource().fetch(["AAPL", "MSFT"], "2020-01-01", "2020-01-09")

    _assert_source_contract("Yahoo Finance / yfinance", yfinance_data, ["AAPL", "MSFT"])
    _assert_source_contract("AKShare", akshare_data, ["sz000001", "sh600000"])
    _assert_source_contract(
        "pandas-datareader / FRED", pandas_reader_data, ["DGS10", "VIXCLS"]
    )
    _assert_source_contract("Stooq", stooq_data, ["AAPL", "MSFT"])

    print("[验收结论] 已对接金融数据工具数量: 4，满足3种及以上要求")
