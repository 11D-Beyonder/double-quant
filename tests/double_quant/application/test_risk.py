import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from double_quant.common.util import divide_by_volatility
from double_quant.common.metric import annualized_volatility

# ==========================================
# 1. Data Preparation Logic
# ==========================================


class DataPreparation:
    def __init__(self, data_dir="tests/double_quant/application/data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.file_path = os.path.join(self.data_dir, "experiment_data_clean.csv")

    def get_tickers(self):
        """
        Returns a list of 100+ tickers covering High/Mid/Low volatility.
        """
        # High Volatility (Tech, Growth, Crypto-related)
        high_vol = [
            "NVDA",
            "TSLA",
            "AMD",
            "MRNA",
            "PYPL",
            "ZM",
            "COIN",
            "RBLX",
            "NET",
            "CRWD",
            "DOCU",
            "PLTR",
            "NIO",
            "XPEV",
            "LI",
            "BIDU",
            "BABA",
            "JD",
            "PDD",
            "SE",
            "MELI",
            "ARKK",
            "ARKG",
            "TQQQ",
            "SOXL",
            "UDOW",
            "URTY",
            "UPRO",
            "ENPH",
            "SEDG",
            "SHOP",
            "TWLO",
            "ROKU",
            "U",
            "SNOW",
            "DDOG",
            "ZS",
            "OKTA",
        ]

        # Mid Volatility (Blue Chip, Consumer, Healthcare, Financials)
        mid_vol = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "V",
            "MA",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "BRK-B",
            "UNH",
            "JNJ",
            "PFE",
            "MRK",
            "ABBV",
            "LLY",
            "PG",
            "KO",
            "PEP",
            "COST",
            "WMT",
            "TGT",
            "HD",
            "LOW",
            "MCD",
            "SBUX",
            "NKE",
            "DIS",
            "CMCSA",
            "NFLX",
            "T",
            "VZ",
            "CVX",
            "XOM",
            "INTC",
            "CSCO",
            "ORCL",
            "IBM",
            "TXN",
            "QCOM",
            "ADBE",
            "CRM",
            "INTU",
            "NOW",
            "AMAT",
            "LRCX",
        ]

        # Low Volatility / Hedge (Utilities, Staples, Bonds, Gold)
        low_vol = [
            "RSG",
            "WM",
            "ED",
            "DUK",
            "SO",
            "NEE",
            "AEP",
            "D",
            "PEG",
            "EXC",
            "AWK",
            "WEC",
            "ES",
            "ETR",
            "FE",
            "PPL",
            "CMS",
            "LNT",
            "ATO",
            "EVRG",
            "TLT",
            "IEF",
            "SHY",
            "GOVT",
            "SHV",
            "BIL",
            "LQD",
            "HYG",
            "JNK",
            "AGG",
            "GLD",
            "SLV",
            "DBC",
            "VIXY",  # VIXY is special, high vol but hedge
        ]

        return list(set(high_vol + mid_vol + low_vol))

    def download(self, start="2020-04-01", end="2022-04-01", use_cache=True):
        if use_cache:
            if os.path.exists(self.file_path):
                print(f"Loading data from {self.file_path}")
                try:
                    df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
                    if not df.empty:
                        return df
                except Exception as e:
                    print(f"Error reading existing file: {e}. Re-downloading.")

        tickers = self.get_tickers()
        print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")

        # Download Adjusted Close prices
        # Use auto_adjust=False and threads=True
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)

        assert data is not None
        if isinstance(data.columns, pd.MultiIndex):
            # If MultiIndex (e.g. ('Adj Close', 'AAPL')), get just 'Adj Close'
            if "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]
            elif "Close" in data.columns.levels[0]:
                data = data["Close"]  # Fallback

        # Drop columns with too many NaNs (e.g. recent IPOs not in range)
        # Strict threshold to 95% to ensure full history from 2020
        original_cols = len(data.columns)

        assert isinstance(data, pd.DataFrame)
        data = data.dropna(axis=1, thresh=int(0.95 * len(data)))
        print(
            f"Dropped {original_cols - len(data.columns)} columns due to missing data."
        )

        # Forward fill remaining NaNs
        data = data.ffill().dropna()

        print(f"Saving data to {self.file_path}")
        data.to_csv(self.file_path)
        return data


# ==========================================
# 2. Tests
# ==========================================


def test_data_download():
    dp = DataPreparation()
    df = dp.download()

    assert not df.empty
    assert len(df.columns) > 50  # Should have successfully downloaded most
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")


def test_volatility_bucketing():
    dp = DataPreparation()
    df = dp.download()

    # Calculate returns
    returns = np.log(df / df.shift(1)).dropna()

    # Split into Low (30%), Mid (40%), High (30%)
    # Quantiles: 0.3, 0.7
    buckets = divide_by_volatility(returns, [0.3, 0.7])

    low_vol_assets = buckets[0]
    mid_vol_assets = buckets[1]
    high_vol_assets = buckets[2]

    print(f"Low Vol Count: {len(low_vol_assets)}")
    print(f"Mid Vol Count: {len(mid_vol_assets)}")
    print(f"High Vol Count: {len(high_vol_assets)}")

    # Calculate average volatility for each bucket
    def get_avg_vol(assets):
        vols = [annualized_volatility(returns[asset]) for asset in assets]
        return np.mean(vols)

    avg_low = get_avg_vol(low_vol_assets)
    avg_mid = get_avg_vol(mid_vol_assets)
    avg_high = get_avg_vol(high_vol_assets)

    print(f"Avg Vol (Low): {avg_low:.4f}")
    print(f"Avg Vol (Mid): {avg_mid:.4f}")
    print(f"Avg Vol (High): {avg_high:.4f}")

    assert avg_low < avg_mid < avg_high

    # Verify specific assets end up where expected
    # Note: VIXY is high volatility but used as hedge. It should appear in High Vol bucket based on pure stats.
    # TLT (Bonds) shoud be in Low Vol.
    # TSLA (Tech) should be in High Vol.
    # KO (Staples) should be in Low or Mid.

    print("Sample Low Vol:", low_vol_assets[:5])
    print("Sample High Vol:", high_vol_assets[:5])

    assert "TSLA" in high_vol_assets
    assert "TLT" in low_vol_assets
    assert "NVDA" in high_vol_assets

    # Check VIXY behavior
    if "VIXY" in df.columns:
        # VIXY is highly volatile, so it should be in the High bucket mathematically
        assert (
            "VIXY" in high_vol_assets or "VIXY" in mid_vol_assets
        )  # Depending on the market regime

    # ==========================================
    # Thesis Visualization & Metrics Generation
    # ==========================================

    # Ensure output directory exists
    output_dir = "docs/assets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate Metrics and Series for each bucket
    bucket_labels = ["Low Volatility", "Mid Volatility", "High Volatility"]
    bucket_assets = [low_vol_assets, mid_vol_assets, high_vol_assets]
    quantiles = ["Bottom 30%", "Mid 40%", "Top 30%"]

    metrics_data = []
    series_data = {}

    for label, assets, q_range in zip(bucket_labels, bucket_assets, quantiles):
        # Create an equally weighted portfolio for the bucket
        # Mean of log returns across assets
        bucket_returns = returns[assets].mean(axis=1)

        # 1. Annualized Volatility
        ann_vol = bucket_returns.std() * np.sqrt(252)

        # 2. Annualized Return (Simple geometric mean approximation from log returns)
        # Total Log Return = sum(log_ret)
        # CAGR = exp(mean(log_ret) * 252) - 1
        ann_ret = np.exp(bucket_returns.mean() * 252) - 1

        # 3. Max Drawdown
        cum_ret_index = np.exp(bucket_returns.cumsum())
        running_max = cum_ret_index.cummax()
        drawdown = (cum_ret_index - running_max) / running_max
        max_dd = drawdown.min()

        # Store for plotting
        series_data[label] = {
            "cum_ret": cum_ret_index,
            "drawdown": drawdown,
            "rolling_vol": bucket_returns.rolling(window=30).std() * np.sqrt(252),
        }

        metrics_data.append(
            {
                "Risk Bucket": label,
                "Quantile": q_range,
                "Avg Volatility": ann_vol,
                "Ann. Return": ann_ret,
                "Max Drawdown": max_dd,
            }
        )

    # Print Table for Thesis
    print("\n" + "=" * 60)
    print(
        f"{'Risk Bucket':<20} | {'Quantile':<12} | {'Avg Vol':<10} | {'Ann. Ret':<10} | {'Max DD':<10}"
    )
    print("-" * 75)
    for m in metrics_data:
        print(
            f"{m['Risk Bucket']:<20} | {m['Quantile']:<12} | {m['Avg Volatility']:.1%}      | {m['Ann. Return']:.1%}      | {m['Max Drawdown']:.1%}"
        )
    print("=" * 60 + "\n")

    # Generate Composite Figure
    # Set seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5, rc={"font.family": "Times New Roman"})
    
    # Define a consistent color palette
    palette = {
        "Low Volatility": "#2ecc71",  # Emerald Green
        "Mid Volatility": "#3498db",  # Peter River Blue
        "High Volatility": "#e74c3c"  # Alizarin Red
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Plot Panel A: Cumulative Wealth
    for label in bucket_labels:
        series = series_data[label]["cum_ret"]
        sns.lineplot(
            x=series.index, 
            y=series.values, 
            label=label, 
            color=palette[label], 
            linewidth=2, 
            ax=ax1
        )

    ax1.set_title(
        "Panel A: Cumulative Wealth Index (Normalized to 1.0)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax1.set_ylabel("Wealth Index")
    ax1.legend(title="Risk Bucket", loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot Panel B: Rolling Volatility
    for label in bucket_labels:
        series = series_data[label]["rolling_vol"]
        sns.lineplot(
            x=series.index, 
            y=series.values, 
            label=label, 
            color=palette[label], 
            linewidth=2, 
            ax=ax2
        )

    ax2.set_title(
        "Panel B: 30-Day Rolling Annualized Volatility", 
        fontsize=14, 
        fontweight="bold",
        pad=15
    )
    ax2.set_ylabel("Volatility (Ann.)")
    ax2.set_xlabel("Date")
    ax2.legend(title="Risk Bucket", loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    
    # Save as both SVG and PNG
    output_path_svg = os.path.join(output_dir, "vol_buckets_trend.svg")
    output_path_png = os.path.join(output_dir, "vol_buckets_trend.png")
    
    plt.savefig(output_path_svg, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    
    # Clear the plot to avoid interference with other tests if any
    plt.clf() 
    
    print(f"Saved composite figure to: {output_path_svg} and {output_path_png}")
