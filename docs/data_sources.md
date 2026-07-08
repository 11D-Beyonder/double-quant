# Financial Data Source Candidates

## Status

Proposed candidates for satisfying the requirement: integrate at least three
existing financial data tools.

## Current Repository State

The repository currently has four implemented external data adapters:

- `YFinanceSource` in `src/double_quant/data/source.py`, backed by `yfinance`.
- `AKShareSource` in `src/double_quant/data/source.py`, backed by AKShare's
  A-share historical price interface.
- `PandasDataReaderSource` in `src/double_quant/data/source.py`, backed by the
  maintained `pandas_datareader.data.DataReader` public API.
- `StooqSource` in `src/double_quant/data/source.py`, backed by Stooq historical
  CSV downloads.

The common integration contract is `PriceSource.fetch(tickers, start, end)`,
which returns a `pandas.DataFrame` with:

- `DatetimeIndex`
- one column per ticker
- close or adjusted-close prices as values

This shape already feeds the existing analysis path:

`prices -> log returns -> covariance / expected returns / ES / Shapley risk attribution`.

## Recommended Three Interfaces

### 1. Yahoo Finance via `yfinance`

Keep as the default US-market source.

- Fit with current code: already implemented.
- Data shape: `yf.download()` returns a pandas-compatible price table for one
  or more tickers.
- Best fit:
  - risk attribution experiments
  - volatility bucketing
  - portfolio expected returns and covariance
  - US equities and ETFs
- Caveat: Yahoo data availability and schema can change, so cache-based
  experiment snapshots should remain the reproducible path.

Source:
https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html

### 2. Alpha Vantage REST API

Add as an API-key-backed market-data source.

- Fit with current code: easy. Parse the daily adjusted time-series response
  into one close-price `Series` per ticker, then concatenate into the same
  `PriceSource` DataFrame shape.
- Data shape: JSON or CSV time-series responses with OHLCV fields.
- Best fit:
  - fallback or cross-check source for US/global equity prices
  - adjusted daily prices for risk attribution and covariance estimation
  - option-pricing support data, because Alpha Vantage also documents options,
    FX, crypto, commodities, economic indicators, fundamentals, and technical
    indicators
- Caveat: requires API key and rate-limit handling; broad multi-ticker
  experiments should use caching aggressively.

Source:
https://www.alphavantage.co/documentation/

### 3. AKShare

Implemented as a no-key China-market A-share price source.

- Fit with current code: done for daily A-share prices. `ak.stock_zh_a_hist(...)`
  returns a pandas DataFrame with date and OHLCV fields; select the adjusted
  close column, normalize dates, and concatenate per ticker.
- Data shape: pandas DataFrames from individual API functions.
- Best fit:
  - A-share risk attribution and portfolio experiments
  - China-market replacement for the current US-only Yahoo workflow
  - broader financial demos because AKShare documents stock, futures, bond,
    option, FX, fund, index, macro, crypto, volatility, and factor datasets
- Caveat: column names are Chinese and endpoints are source-specific, so the
  adapter should isolate schema mapping inside `AKShareSource`.

Source:
https://akshare.akfamily.xyz/data/stock/stock.html

## Additional No-API-Key Sources

### Stooq

Implemented as `StooqSource`.

- Fit with current code: done. The adapter downloads historical CSV data, selects
  the close column, and normalizes to the same `PriceSource` shape.
- Best fit:
  - no-key price-history fallback for US/global tickers supported by Stooq
  - risk attribution, volatility bucketing, and portfolio covariance demos
- Caveat: symbol conventions differ from Yahoo and AKShare. The adapter defaults
  bare tickers such as `AAPL` to `aapl.us`, but callers can pass explicit Stooq
  symbols or disable the default suffix.
- Version note: older pandas-datareader docs describe a `StooqDailyReader`, but
  `pandas-datareader==0.11.1` no longer exposes that module. The direct
  `StooqSource` keeps Stooq available without pinning the project to an older
  pandas-datareader release.

Source:
https://pandas-datareader.readthedocs.io/en/latest/readers/stooq.html

### `pandas-datareader` macro/factor readers

Implemented as `PandasDataReaderSource`.

- Good fit: FRED, Fama/French factors, OECD, Eurostat, World Bank, and other
  macro/factor data.
- Poorer fit for the current `PriceSource`: the currently documented maintained
  public API is focused on macroeconomic, policy, central-bank, and factor-style
  data, not broad adjusted equity price downloads.
- Recommended use: factor-aware and macro-aware extensions, not as a drop-in
  replacement for close-price history.
- Version note: `pandas-datareader==0.11.1` currently documents and implements
  `bankofcanada`, `fred`, `famafrench`, `oecd`, `eurostat`, and `econdb` through
  `DataReader`.

Source:
https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

## Suitability By Current Financial Problem

| Project problem | Suitable data | Best candidate |
| --- | --- | --- |
| ES / Shapley risk attribution | daily adjusted close prices, converted to log returns | `yfinance`, AKShare, Stooq, Alpha Vantage |
| Volatility buckets | daily close or adjusted close over a common date window | `yfinance`, AKShare, Stooq |
| Portfolio optimization | expected returns and covariance from aligned close-price history | `yfinance`, AKShare, Stooq, Alpha Vantage |
| HHL-backed portfolio demo | same as portfolio optimization, after covariance/return conversion | `yfinance`, AKShare, Stooq, Alpha Vantage |
| Option pricing / QAE valuation demos | underlying price, volatility, risk-free rate, option chains or payoff scenarios | Alpha Vantage, AKShare; `yfinance` can remain a simple underlying-price source |
| Macro or factor-aware extensions | rates, inflation, GDP, Fama/French factors | `pandas-datareader`, Alpha Vantage, AKShare |

## Implementation Notes

The lowest-risk implementation is to keep `PriceSource` unchanged and expose
provider adapters with the same return contract:

- `AlphaVantageSource(cache_path=None, api_key=None, adjusted=True)`
- `AKShareSource(cache_path=None, adjust="qfq")` implemented
- `PandasDataReaderSource(cache_path=None, data_source="fred")` implemented
- `StooqSource(cache_path=None)` implemented

All price adapters should:

- preserve the current cache-first behavior
- normalize all dates to `DatetimeIndex`
- return one column per requested ticker
- select adjusted close where available
- drop sparse columns and forward-fill exactly like `YFinanceSource`
- hide provider-specific column names and rate-limit handling inside the
  adapter
