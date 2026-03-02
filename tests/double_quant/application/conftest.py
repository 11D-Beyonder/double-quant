from pathlib import Path

import numpy as np
import pytest

from double_quant.data.source import YFinanceSource

# 10 tickers covering high / mid / low volatility
TEST_TICKERS = [
    "TSLA", "NVDA",                  # high vol
    "AAPL", "MSFT", "META", "JPM",   # mid vol
    "TLT", "GLD", "ED", "AGG",       # low vol
]
TEST_CACHE = str(
    Path(__file__).resolve().parent / "cache" / "test_data.csv"
)


@pytest.fixture(scope="session")
def prices():
    return YFinanceSource(cache_path=TEST_CACHE).fetch(TEST_TICKERS, "2020-04-01", "2022-04-01")


@pytest.fixture(scope="session")
def returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
