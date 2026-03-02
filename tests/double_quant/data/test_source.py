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
