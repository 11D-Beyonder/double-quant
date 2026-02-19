import pandas as pd
import yfinance as yf


def from_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool = False,
    nan_threshold: float = 0.95,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Download historical adjusted close prices from Yahoo Finance with cleaning and optional caching.

    Args:
        tickers: List of ticker symbols to download.
        start: Start date string (e.g. "2020-01-01").
        end: End date string (e.g. "2023-01-01").
        auto_adjust: Whether to auto-adjust prices for splits and dividends. Defaults to False
            (uses Adj Close column explicitly).
        nan_threshold: Minimum fraction of non-NaN rows required to keep a column. Columns
            below this threshold are dropped. Defaults to 0.95.
        cache_path: If provided, attempt to load from this CSV path before downloading.
            On a successful download, the result is saved to this path.

    Returns:
        DataFrame of adjusted close prices with tickers as columns and dates as index.
    """
    if cache_path is not None:
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached
        except (FileNotFoundError, Exception):
            pass

    data = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust)

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]

    data = data.dropna(axis=1, thresh=int(nan_threshold * len(data)))
    data = data.ffill().dropna()

    if cache_path is not None:
        data.to_csv(cache_path)

    return data
