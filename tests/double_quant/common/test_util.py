import numpy as np
import pandas as pd
from double_quant.common.util import divide_by_volatility


def test_divide_by_volatility_basic():
    # Create dummy data with known volatilities
    # Higher variance means higher volatility
    dates = pd.date_range("2023-01-01", periods=100)
    data = {
        "A": np.exp(np.random.normal(0, 0.01, 100).cumsum()),  # Low vol
        "B": np.exp(np.random.normal(0, 0.05, 100).cumsum()),  # High vol
        "C": np.exp(np.random.normal(0, 0.03, 100).cumsum()),  # Mid vol
    }
    df = pd.DataFrame(data, index=dates)

    # 50th percentile should split into two groups
    groups = divide_by_volatility(df, [0.5])

    assert len(groups) == 2
    # Total number of assets should be preserved
    assert sum(len(g) for g in groups) == 3

    # The groups should be ordered by volatility
    # A should be in the first group, B in the second, C depends on where it falls
    # We can check if A's volatility is lower than B's

    flattened_groups = [ticker for group in groups for ticker in group]
    assert set(flattened_groups) == set(df.columns)


def test_divide_by_volatility_multiple_splits():
    dates = pd.date_range("2023-01-01", periods=100)
    # 4 assets with distinct volatilities
    data = {
        "A": np.exp(np.random.normal(0, 0.01, 100).cumsum()),
        "B": np.exp(np.random.normal(0, 0.10, 100).cumsum()),
        "C": np.exp(np.random.normal(0, 0.05, 100).cumsum()),
        "D": np.exp(np.random.normal(0, 0.02, 100).cumsum()),
    }
    df = pd.DataFrame(data, index=dates)

    # [0.25, 0.75] should create 3 groups
    groups = divide_by_volatility(df, [0.25, 0.75])

    assert len(groups) == 3
    assert sum(len(g) for g in groups) == 4

    # Verify order
    log_returns = np.log(df / df.shift(1))
    vols = log_returns.std() * np.sqrt(252)
    sorted_tickers = vols.sort_values().index.tolist()

    # flattened groups should follow the same order as sorted_tickers (or at least same set within groups)
    for group in groups:
        for ticker in group:
            assert ticker in sorted_tickers

    # Check specifically that the least volatile is in the first group and most in last
    assert sorted_tickers[0] in groups[0]
    assert sorted_tickers[-1] in groups[-1]
