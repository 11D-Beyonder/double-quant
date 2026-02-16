import yfinance as yf


def from_yfinance(tickers, start, end):
    return yf.download(tickers, start, end)
