import pandas as pd


def test_yfinance_source_implements_protocol():
    from double_quant.data.source import YFinanceSource

    source = YFinanceSource()
    assert hasattr(source, "fetch")


def test_yfinance_source_from_cache(tmp_path):
    from double_quant.data.source import YFinanceSource

    cache_file = tmp_path / "prices.csv"
    dates = pd.date_range("2020-01-01", periods=5)
    expected = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
    expected.to_csv(cache_file)

    source = YFinanceSource(cache_path=str(cache_file))
    result = source.fetch(["AAPL"], "2020-01-01", "2020-01-06")
    assert list(result.columns) == ["AAPL"]
    assert len(result) == 5


def test_akshare_source_implements_protocol():
    from double_quant.data.source import AKShareSource

    source = AKShareSource()
    assert hasattr(source, "fetch")


def test_akshare_source_from_cache(tmp_path):
    from double_quant.data.source import AKShareSource

    cache_file = tmp_path / "prices.csv"
    dates = pd.date_range("2020-01-01", periods=5)
    expected = pd.DataFrame({"000001": [10, 11, 12, 13, 14]}, index=dates)
    expected.to_csv(cache_file)

    source = AKShareSource(cache_path=str(cache_file))
    result = source.fetch(["000001"], "2020-01-01", "2020-01-06")
    assert list(result.columns) == ["000001"]
    assert len(result) == 5


def test_akshare_source_fetch_normalizes_history(monkeypatch):
    from double_quant.data import source as source_module
    from double_quant.data.source import AKShareSource

    calls = []

    def fake_stock_zh_a_hist(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "日期": ["2020-01-02", "2020-01-03", "2020-01-06"],
                "股票代码": [kwargs["symbol"]] * 3,
                "开盘": [9.9, 10.8, 11.7],
                "收盘": [10.0, 11.0, 12.0],
                "最高": [10.1, 11.1, 12.1],
                "最低": [9.8, 10.7, 11.6],
                "成交量": [100, 110, 120],
            }
        )

    monkeypatch.setattr(source_module.ak, "stock_zh_a_hist", fake_stock_zh_a_hist)

    result = AKShareSource(adjust="qfq").fetch(
        ["sz000001"], "2020-01-01", "2020-01-06"
    )

    assert calls == [
        {
            "symbol": "000001",
            "period": "daily",
            "start_date": "20200101",
            "end_date": "20200106",
            "adjust": "qfq",
            "timeout": None,
        }
    ]
    assert list(result.columns) == ["sz000001"]
    assert result.index.tolist() == pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06"]
    ).tolist()
    assert result["sz000001"].tolist() == [10.0, 11.0, 12.0]


def test_pandas_datareader_source_implements_protocol():
    from double_quant.data.source import PandasDataReaderSource

    source = PandasDataReaderSource()
    assert hasattr(source, "fetch")


def test_pandas_datareader_source_from_cache(tmp_path):
    from double_quant.data.source import PandasDataReaderSource

    cache_file = tmp_path / "macro.csv"
    dates = pd.date_range("2020-01-01", periods=5)
    expected = pd.DataFrame({"DGS10": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)
    expected.to_csv(cache_file)

    source = PandasDataReaderSource(cache_path=str(cache_file))
    result = source.fetch(["DGS10"], "2020-01-01", "2020-01-06")
    assert list(result.columns) == ["DGS10"]
    assert len(result) == 5


def test_pandas_datareader_source_fetch_normalizes_dataframe(monkeypatch):
    from double_quant.data import source as source_module
    from double_quant.data.source import PandasDataReaderSource

    calls = []

    def fake_data_reader(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame(
            {
                "DGS10": [1.0, 1.1, 1.2],
                "VIXCLS": [15.0, 16.0, 17.0],
            },
            index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        )

    monkeypatch.setattr(source_module.pdr_data, "DataReader", fake_data_reader)

    result = PandasDataReaderSource(data_source="fred").fetch(
        ["DGS10", "VIXCLS"], "2020-01-01", "2020-01-06"
    )

    assert calls == [
        (
            (["DGS10", "VIXCLS"], "fred"),
            {
                "start": "2020-01-01",
                "end": "2020-01-06",
                "retry_count": 3,
                "pause": 0.1,
                "session": None,
                "api_key": None,
            },
        )
    ]
    assert list(result.columns) == ["DGS10", "VIXCLS"]
    assert result["DGS10"].tolist() == [1.0, 1.1, 1.2]


def test_stooq_source_implements_protocol():
    from double_quant.data.source import StooqSource

    source = StooqSource()
    assert hasattr(source, "fetch")


def test_stooq_source_from_cache(tmp_path):
    from double_quant.data.source import StooqSource

    cache_file = tmp_path / "prices.csv"
    dates = pd.date_range("2020-01-01", periods=5)
    expected = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
    expected.to_csv(cache_file)

    source = StooqSource(cache_path=str(cache_file))
    result = source.fetch(["AAPL"], "2020-01-01", "2020-01-06")
    assert list(result.columns) == ["AAPL"]
    assert len(result) == 5


def test_stooq_source_fetch_normalizes_history(monkeypatch):
    from double_quant.data import source as source_module
    from double_quant.data.source import StooqSource

    calls = []

    def fake_read_stooq_csv(url, timeout):
        calls.append((url, timeout))
        return pd.DataFrame(
            {
                "Date": ["2020-01-02", "2020-01-03", "2020-01-06"],
                "Open": [299.0, 300.0, 301.0],
                "High": [301.0, 302.0, 303.0],
                "Low": [298.0, 299.0, 300.0],
                "Close": [300.0, 301.0, 302.0],
                "Volume": [1000, 1100, 1200],
            }
        )

    monkeypatch.setattr(source_module, "_read_stooq_csv", fake_read_stooq_csv)

    result = StooqSource(timeout=7.0).fetch(["AAPL"], "2020-01-01", "2020-01-06")

    assert calls == [
        (
            "https://stooq.com/q/d/l/?s=aapl.us&d1=20200101&d2=20200106&i=d",
            7.0,
        )
    ]
    assert list(result.columns) == ["AAPL"]
    assert result.index.tolist() == pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06"]
    ).tolist()
    assert result["AAPL"].tolist() == [300.0, 301.0, 302.0]
