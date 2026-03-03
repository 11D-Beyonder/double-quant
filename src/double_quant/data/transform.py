"""Pure transformation functions for converting raw price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close prices -> log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def to_covariance(prices: pd.DataFrame) -> np.ndarray:
    """Close prices -> covariance matrix of log returns."""
    return to_log_returns(prices).cov().values


def to_expected_returns(prices: pd.DataFrame) -> np.ndarray:
    """Close prices -> mean log return vector."""
    mean_returns = to_log_returns(prices).mean().to_numpy(dtype=float)
    return np.asarray(mean_returns, dtype=float)
