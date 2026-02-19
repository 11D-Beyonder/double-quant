import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.solver.shapley import BinaryEnumerationCalculator
from double_quant.common.metric import annualized_volatility
from double_quant.common.util import divide_by_volatility
from double_quant.data.time_series import from_yfinance

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
        cache_path = self.file_path if use_cache else None
        return from_yfinance(self.get_tickers(), start, end, cache_path=cache_path)


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
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=2.0,  # Increased font scale for larger axis labels
        rc={"font.family": "Times New Roman"},
    )

    # Define a consistent color palette
    palette = {
        "Low Volatility": "#4daf4a",  # Muted Green
        "Mid Volatility": "#377eb8",  # Muted Blue
        "High Volatility": "#ff7f00",  # Muted Orange
    }
    _, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},  # Made figure flatter
    )

    # Plot Panel A: Cumulative Wealth
    for label in bucket_labels:
        series = series_data[label]["cum_ret"]
        sns.lineplot(
            x=series.index,
            y=series.values,
            color=palette[label],
            linewidth=2,
            ax=ax1,
        )

    ax1.grid(True, linestyle="--", alpha=0.6)
    # Hide x-axis tick labels for the upper subplot when sharing x-axis
    ax1.tick_params(labelbottom=False)

    # Plot Panel B: Rolling Volatility
    for label in bucket_labels:
        series = series_data[label]["rolling_vol"]
        sns.lineplot(
            x=series.index,
            y=series.values,
            color=palette[label],
            linewidth=2,
            ax=ax2,
        )

    ax2.grid(True, linestyle="--", alpha=0.6)

    # Get date range from the returns DataFrame for shading
    start_date = returns.index.min()
    end_date = returns.index.max()
    demarcation_date = pd.Timestamp("2021-11-01")

    # Add shaded regions for "Liquidity Bull Market"
    ax1.axvspan(start_date, demarcation_date, color="lightgreen", alpha=0.1, zorder=0)
    ax2.axvspan(start_date, demarcation_date, color="lightgreen", alpha=0.1, zorder=0)

    # Add shaded regions for "Interest Rate Cut Cycle"
    ax1.axvspan(demarcation_date, end_date, color="lightcoral", alpha=0.1, zorder=0)
    ax2.axvspan(demarcation_date, end_date, color="lightcoral", alpha=0.1, zorder=0)

    # Add vertical line
    ax1.axvline(demarcation_date, color="gray", linestyle="--", linewidth=1.5, zorder=1)
    ax2.axvline(demarcation_date, color="gray", linestyle="--", linewidth=1.5, zorder=1)

    # Set x-axis limits to remove empty space
    ax1.set_xlim(start_date, end_date)
    ax2.set_xlim(start_date, end_date)

    plt.tight_layout(h_pad=1.2)
    output_path_svg = os.path.join(output_dir, "vol_buckets_trend.svg")
    plt.savefig(output_path_svg)
    plt.show()


class TestRiskSaving:
    def _indices_to_mask(self, indices: list[int]) -> int:
        return sum(1 << i for i in indices)

    def test_superadditivity(self):
        """
        Verify that the Risk Saving (RS) characteristic function is
        superadditive: RS(S ∪ T) ≥ RS(S) + RS(T) for disjoint S, T.

        This is the precondition that makes RS compatible with the quantum Shapley algorithm.
        Runs 5000 random trials on the full asset universe.
        """
        N_TRIALS = 5000
        FLOAT_TOL = 1e-9

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        vfunc = RiskSavingValueFunction(returns)
        n_assets = vfunc.num_assets

        rng = np.random.default_rng(seed=42)

        synergy_all: list[float] = []

        violations = 0
        for _ in range(N_TRIALS):
            s = int(rng.integers(2, 9))
            t = int(rng.integers(2, 9))
            indices = rng.choice(n_assets, size=s + t, replace=False).tolist()
            idx_s, idx_t = indices[:s], indices[s:]

            mask_s = self._indices_to_mask(idx_s)
            mask_t = self._indices_to_mask(idx_t)
            mask_st = mask_s | mask_t

            synergy = vfunc[mask_st] - (vfunc[mask_s] + vfunc[mask_t])
            synergy_all.append(synergy)

            if synergy < -FLOAT_TOL:
                violations += 1

        synergy_arr = np.array(synergy_all)

        print("\n" + "=" * 60)
        print(f"Superadditivity Verification (N={N_TRIALS} random disjoint pairs)")
        print("=" * 60)
        print(f"  Violations (synergy < -{FLOAT_TOL}): {violations}")
        print(
            f"  All pairs  — min: {synergy_arr.min():.6f}  mean: {synergy_arr.mean():.6f}  max: {synergy_arr.max():.6f}"
        )
        print("=" * 60 + "\n")

        assert violations == 0, (
            f"Superadditivity violated in {violations}/{N_TRIALS} trials. "
            f"Min synergy = {synergy_arr.min():.9f}"
        )

    def test_restoration_accuracy(self):
        """
        Verify that the restoration formula SRC_i = ES({i}) - Φ_i^RS
        is numerically lossless compared to the direct path SRC_i = Φ_i^ES.

        Both paths use BinaryEnumerationCalculator on the same 5-asset portfolio.
        MAE should be near machine floating-point epsilon, confirming the RS ↔ ES
        duality introduced for quantum-compatibility introduces zero algorithmic error.
        """
        MAE_TOL = 1e-9

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        # Sample 5 assets: 2 High, 2 Mid, 1 Low — heterogeneous covariance structure
        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=0)
        assets_5 = (
            rng.choice(high_assets, size=2, replace=False).tolist()
            + rng.choice(mid_assets, size=2, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_5 = returns[assets_5]

        # Path A: direct Shapley of ES  →  SRC_i = Φ_i^ES
        src_es = RiskAttributor(
            returns_5, BinaryEnumerationCalculator, mode="es"
        ).attribute()

        # Path B: Shapley of RS then restore  →  SRC_i = ES({i}) - Φ_i^RS
        src_rs = RiskAttributor(
            returns_5, BinaryEnumerationCalculator, mode="rs"
        ).attribute()

        diffs = {a: abs(src_rs[a] - src_es[a]) for a in assets_5}
        mae = float(np.mean(list(diffs.values())))

        print("\n" + "=" * 70)
        print("Restoration Formula Accuracy  (mode='es' vs mode='rs'  |  n=5 assets)")
        print("=" * 70)
        print(f"  {'Asset':<10} {'SRC via ES':>14} {'SRC via RS':>14} {'|diff|':>14}")
        print("  " + "-" * 56)
        for a in assets_5:
            print(
                f"  {a:<10} {src_es[a]:>14.10f} {src_rs[a]:>14.10f} {diffs[a]:>14.2e}"
            )
        print("  " + "-" * 56)
        print(f"  {'MAE':<10} {mae:>43.2e}")
        print("=" * 70 + "\n")

        assert mae < MAE_TOL, (
            f"Restoration formula MAE = {mae:.2e} exceeds tolerance {MAE_TOL:.2e}. "
            f"Max single-asset diff = {max(diffs.values()):.2e} on asset '{max(diffs, key=diffs.get)}'"
        )
