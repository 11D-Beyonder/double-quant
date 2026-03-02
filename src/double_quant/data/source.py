"""Data source interfaces and implementations."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
import yfinance as yf


class PriceSource(Protocol):
    """Fetch stock prices. All implementations return a DataFrame with
    index=DatetimeIndex, columns=tickers, values=close prices."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


class YFinanceSource:
    """Yahoo Finance data source with optional CSV caching."""

    def __init__(
        self,
        cache_path: str | None = None,
        auto_adjust: bool = False,
        nan_threshold: float = 0.95,
    ):
        self.cache_path = cache_path
        self.auto_adjust = auto_adjust
        self.nan_threshold = nan_threshold

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.cache_path is not None:
            try:
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except FileNotFoundError:
                pass

        downloaded = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
        )
        if downloaded is None:
            raise ValueError("yfinance returned no data")

        if isinstance(downloaded, pd.Series):
            column_name = tickers[0] if len(tickers) == 1 else "value"
            data = downloaded.to_frame(name=column_name)
        else:
            data = downloaded

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]

        if isinstance(data, pd.Series):
            column_name = tickers[0] if len(tickers) == 1 else "value"
            data = data.to_frame(name=column_name)

        min_non_na = max(1, int(self.nan_threshold * len(data)))
        data = data.dropna(axis="columns", thresh=min_non_na)
        data = data.ffill().dropna()

        if self.cache_path is not None:
            data.to_csv(self.cache_path)

        return data
