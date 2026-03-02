from pathlib import Path

import numpy as np
import pytest

from double_quant.data.time_series import from_yfinance

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
    return from_yfinance(TEST_TICKERS, "2020-04-01", "2022-04-01", cache_path=TEST_CACHE)


@pytest.fixture(scope="session")
def returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
