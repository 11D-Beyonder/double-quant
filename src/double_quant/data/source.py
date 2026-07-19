"""Data source interfaces and implementations."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from io import StringIO
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import akshare as ak
import pandas as pd
import pandas_datareader.data as pdr_data
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


class AKShareSource:
    """AKShare A-share price source with optional CSV caching.

    This adapter normalizes AKShare's Chinese-column historical A-share output
    to the same ``PriceSource`` shape used by the rest of the package.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        adjust: str = "qfq",
        period: str = "daily",
        nan_threshold: float = 0.95,
        timeout: float | None = None,
    ):
        self.cache_path = cache_path
        self.adjust = adjust
        self.period = period
        self.nan_threshold = nan_threshold
        self.timeout = timeout

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.cache_path is not None:
            try:
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except FileNotFoundError:
                pass

        start_date = _date_to_akshare(start)
        end_date = _date_to_akshare(end)
        series_list: list[pd.Series] = []
        failures: dict[str, str] = {}

        for ticker in tickers:
            symbol = _normalize_akshare_symbol(ticker)
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period=self.period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=self.adjust,
                    timeout=self.timeout,
                )
            except Exception as exc:
                failures[ticker] = str(exc)
                continue

            try:
                series_list.append(_akshare_close_series(raw, ticker))
            except ValueError as exc:
                failures[ticker] = str(exc)

        if not series_list:
            detail = "; ".join(f"{ticker}: {error}" for ticker, error in failures.items())
            raise ValueError(f"AKShare returned no usable data. {detail}".strip())

        data = pd.concat(series_list, axis="columns", sort=False).sort_index()
        min_non_na = max(1, int(self.nan_threshold * len(data)))
        data = data.dropna(axis="columns", thresh=min_non_na)
        data = data.ffill().dropna()

        if data.empty:
            raise ValueError("AKShare returned no data after cleaning")

        if self.cache_path is not None:
            data.to_csv(self.cache_path)

        return data


def _date_to_akshare(date_value: str) -> str:
    return pd.Timestamp(date_value).strftime("%Y%m%d")


def _normalize_akshare_symbol(ticker: str) -> str:
    symbol = ticker.strip()
    if "." in symbol:
        symbol = symbol.split(".", maxsplit=1)[0]
    lower_symbol = symbol.lower()
    if len(symbol) > 2 and lower_symbol[:2] in {"sh", "sz", "bj"}:
        symbol = symbol[2:]
    return symbol


def _akshare_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    required_columns = {"日期", "收盘"}
    missing = required_columns.difference(data.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"AKShare data for {ticker} missing columns: {missing_names}")

    if data.empty:
        raise ValueError(f"AKShare returned empty data for {ticker}")

    series = pd.Series(
        pd.to_numeric(data["收盘"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(data["日期"]),
        name=ticker,
    )
    return series.dropna()


class PandasDataReaderSource:
    """pandas-datareader source for maintained public time-series APIs.

    The default ``data_source="fred"`` is useful for macro and market indicator
    series, not broad adjusted equity-price downloads.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        data_source: str = "fred",
        retry_count: int = 3,
        pause: float = 0.1,
        session: object | None = None,
        api_key: str | None = None,
        table: Hashable | None = None,
        nan_threshold: float = 0.95,
    ):
        self.cache_path = cache_path
        self.data_source = data_source
        self.retry_count = retry_count
        self.pause = pause
        self.session = session
        self.api_key = api_key
        self.table = table
        self.nan_threshold = nan_threshold

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.cache_path is not None:
            try:
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except FileNotFoundError:
                pass

        symbols: str | list[str] = tickers[0] if len(tickers) == 1 else tickers
        downloaded = pdr_data.DataReader(
            symbols,
            self.data_source,
            start=start,
            end=end,
            retry_count=self.retry_count,
            pause=self.pause,
            session=self.session,
            api_key=self.api_key,
        )
        data = _normalize_pandas_datareader_result(downloaded, self.table, tickers)
        data = _clean_source_frame(data, self.nan_threshold)

        if self.cache_path is not None:
            data.to_csv(self.cache_path)

        return data


class StooqSource:
    """Stooq historical price source with optional CSV caching."""

    def __init__(
        self,
        cache_path: str | None = None,
        interval: str = "d",
        default_suffix: str | None = ".US",
        nan_threshold: float = 0.95,
        timeout: float = 15.0,
    ):
        self.cache_path = cache_path
        self.interval = interval
        self.default_suffix = default_suffix
        self.nan_threshold = nan_threshold
        self.timeout = timeout

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.cache_path is not None:
            try:
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except FileNotFoundError:
                pass

        start_date = _date_to_stooq(start)
        end_date = _date_to_stooq(end)
        series_list: list[pd.Series] = []
        failures: dict[str, str] = {}

        for ticker in tickers:
            symbol = _normalize_stooq_symbol(ticker, self.default_suffix)
            url = _stooq_url(symbol, start_date, end_date, self.interval)
            try:
                raw = _read_stooq_csv(url, self.timeout)
                series_list.append(_stooq_close_series(raw, ticker))
            except Exception as exc:
                failures[ticker] = str(exc)

        if not series_list:
            detail = "; ".join(f"{ticker}: {error}" for ticker, error in failures.items())
            raise ValueError(f"Stooq returned no usable data. {detail}".strip())

        data = pd.concat(series_list, axis="columns", sort=False).sort_index()
        data = _clean_source_frame(data, self.nan_threshold)

        if self.cache_path is not None:
            data.to_csv(self.cache_path)

        return data


def _normalize_pandas_datareader_result(
    data: pd.DataFrame | pd.Series | Mapping[Hashable, object],
    table: Hashable | None,
    tickers: list[str],
) -> pd.DataFrame:
    if isinstance(data, Mapping):
        if table is None:
            raise ValueError(
                "pandas-datareader returned multiple tables; set table=... "
                "to choose one"
            )
        selected = data[table]
        if not isinstance(selected, pd.DataFrame):
            raise ValueError(f"Selected pandas-datareader table is not a DataFrame: {table}")
        data = selected

    if isinstance(data, pd.Series):
        return data.to_frame(name=tickers[0] if tickers else data.name)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("pandas-datareader returned unsupported data type")
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError(
            "pandas-datareader returned MultiIndex columns; use a narrower "
            "source query or table selection"
        )
    if len(tickers) == 1 and len(data.columns) == 1:
        data = data.rename(columns={data.columns[0]: tickers[0]})
    return data


def _clean_source_frame(data: pd.DataFrame, nan_threshold: float) -> pd.DataFrame:
    min_non_na = max(1, int(nan_threshold * len(data)))
    cleaned = data.dropna(axis="columns", thresh=min_non_na)
    cleaned = cleaned.ffill().dropna()
    if cleaned.empty:
        raise ValueError("Data source returned no data after cleaning")
    return cleaned


def _date_to_stooq(date_value: str) -> str:
    return pd.Timestamp(date_value).strftime("%Y%m%d")


def _normalize_stooq_symbol(ticker: str, default_suffix: str | None) -> str:
    symbol = ticker.strip()
    if "." not in symbol and default_suffix:
        symbol = f"{symbol}{default_suffix}"
    return symbol.lower()


def _stooq_url(symbol: str, start_date: str, end_date: str, interval: str) -> str:
    query = urlencode({"s": symbol, "d1": start_date, "d2": end_date, "i": interval})
    return f"https://stooq.com/q/d/l/?{query}"


def _read_stooq_csv(url: str, timeout: float) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    if not text.lstrip().startswith("Date,"):
        raise ValueError("Stooq did not return CSV price data")
    return pd.read_csv(StringIO(text))


def _stooq_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    required_columns = {"Date", "Close"}
    missing = required_columns.difference(data.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Stooq data for {ticker} missing columns: {missing_names}")
    if data.empty:
        raise ValueError(f"Stooq returned empty data for {ticker}")
    series = pd.Series(
        pd.to_numeric(data["Close"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(data["Date"]),
        name=ticker,
    )
    return series.dropna()
