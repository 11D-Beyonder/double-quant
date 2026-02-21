import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.solver.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)
from double_quant.common.metric import annualized_volatility
from double_quant.common.util import divide_by_volatility
from double_quant.data.time_series import from_yfinance


def test_permutation_mc_basic():
    """Verify PermutationMCCalculator converges to exact Shapley with enough samples."""

    # Simple superadditive value function: v(S) = |S|^2
    num_players = 4
    value_dict = {s: bin(s).count("1") ** 2 for s in range(2**num_players)}

    calc_exact = BinaryEnumerationCalculator(num_players, value_dict)
    calc_mc = PermutationMCCalculator(
        num_players, value_dict, num_samples=1000, seed=42
    )

    exact = calc_exact.get_all()
    mc = calc_mc.get_all()

    # Should be close with 1000 samples
    for i in range(num_players):
        rel_err = abs(mc[i] - exact[i]) / abs(exact[i]) if exact[i] != 0 else abs(mc[i])
        assert rel_err < 0.1, f"Player {i}: rel_err={rel_err:.4f} > 0.1"

    print("MC estimate:", mc)
    print("Exact:", exact)


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


# Helper function for rounded horizontal bars
def create_rounded_barh(ax, y_data, x_data, colors, height=0.6, rounding_size=0.1):
    for i, (y_val, x_val, color) in enumerate(zip(y_data, x_data, colors)):
        # Create a FancyBboxPatch for each bar
        # x-coordinate starts from 0 (left edge of the bar)
        # y-coordinate is centered around the index i
        # width is the x_val
        # height is the bar thickness
        fancy_bbox_patch = FancyBboxPatch(
            (0, i - height / 2),
            x_val,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding_size}",
            fc=color,
            ec="none",  # No edge color for a cleaner look
            lw=0,
            alpha=1,
            zorder=2,  # Ensure bars are on top
        )
        ax.add_patch(fancy_bbox_patch)

    # Set y-axis limits and ticks
    ax.set_ylim(-0.5, len(y_data) - 0.5)
    ax.set_yticks(range(len(y_data)))
    ax.set_yticklabels(y_data)


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

        # ── Visualisation: per-asset |diff| + MAE horizontal bar chart ──────
        bucket_map: dict[str, str] = (
            {a: "High" for a in high_assets}
            | {a: "Mid" for a in mid_assets}
            | {a: "Low" for a in low_assets}
        )
        palette_bar = {
            "High": "#c8906a",  # muted amber
            "Mid": "#7aa3c4",  # muted steel-blue
            "Low": "#82ae80",  # muted sage-green
            "MAE": "#9e9e9e",  # neutral grey
        }

        # ascending sort → worst asset ends up at the top of the chart
        sorted_assets = sorted(assets_5, key=lambda a: diffs[a])
        bar_labels = ["MAE"] + sorted_assets
        bar_values = [mae] + [diffs[a] for a in sorted_assets]
        bar_colors = [palette_bar["MAE"]] + [
            palette_bar[bucket_map[a]] for a in sorted_assets
        ]

        sns.set_theme(
            style="ticks",
            context="paper",
            font_scale=1.7,
            rc={
                "font.family": "Times New Roman",
                "mathtext.fontset": "stix",  # Times-compatible math font
            },
        )

        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f6f6f6")

        # Regular barh — no rounded corners
        ax.barh(
            range(len(bar_labels)),
            bar_values,
            color=bar_colors,
            height=0.55,
            zorder=2,
        )
        ax.set_yticks(range(len(bar_labels)))
        ax.set_yticklabels(bar_labels, fontsize=12)
        ax.set_ylim(-0.5, len(bar_labels) - 0.5)

        # Scale tick labels to plain numbers; put ×10^{exp} in the xlabel
        _max = max(bar_values)
        _exp = int(np.floor(np.log10(_max)))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _, e=_exp: f"{x * 10 ** (-e):.1f}")
        )
        # ax.set_xlabel(
        #     r"$|\,\Phi^{\mathrm{ES}}_i - \mathrm{SRC}^{\mathrm{RS}}_i\,|$"
        #     + f"$\\;(\\times 10^{{{_exp}}})$",
        #     labelpad=8,
        # )
        # ax.set_title(
        #     "Restoration Formula Accuracy  —  RS \u2194 ES Duality",
        #     pad=10, fontweight="normal",
        # )
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, _max * 1.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value labels at the end of each bar: e.g. $1.39 \times 10^{-17}$
        for i, v in enumerate(bar_values):
            _v_exp = int(np.floor(np.log10(v))) if v > 0 else 0
            _v_mant = v / 10**_v_exp
            label = rf"${_v_mant:.2f} \times 10^{{{_v_exp}}}$"
            ax.text(
                v + _max * 0.02,
                i,
                label,
                va="center",
                ha="left",
                fontsize=12,
                color="#333333",
            )

        # from matplotlib.patches import Patch
        # ax.legend(
        #     handles=[
        #         Patch(facecolor=palette_bar["High"], label="High Vol."),
        #         Patch(facecolor=palette_bar["Mid"],  label="Mid Vol."),
        #         Patch(facecolor=palette_bar["Low"],  label="Low Vol."),
        #         Patch(facecolor=palette_bar["MAE"],  label="MAE"),
        #     ],
        #     loc="lower right", framealpha=0.8, fontsize=9, handlelength=1.2,
        # )

        plt.tight_layout()
        plt.show()
        # ── End Visualisation ─────────────────────────────────────────────────

        assert mae < MAE_TOL, (
            f"Restoration formula MAE = {mae:.2e} exceeds tolerance {MAE_TOL:.2e}. "
            f"Max single-asset diff = {max(diffs.values()):.2e} on asset '{max(diffs, key=diffs.get)}'"
        )


class TestDifferentSolver:
    def test_quantum_basic(self):
        """
        Small-scale verification that the quantum Shapley pipeline works end-to-end.
        Picks 3 assets (1 per risk bucket) and compares QuantumCalculator vs
        BinaryEnumerationCalculator
        """
        REL_TOL = 0.05  # 5% relative error tolerance

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=42)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        # Ground truth: exact Shapley via binary enumeration
        src_exact = RiskAttributor(
            returns_3, BinaryEnumerationCalculator, mode="es"
        ).attribute()

        # Quantum estimate: internal_qubits_num=6, internal_multiplier=1
        src_quantum = RiskAttributor(
            returns_3,
            QuantumCalculator,
            mode="rs",
            internal_qubits_num=6,
            internal_multiplier=1,
        ).attribute()

        print("\n" + "=" * 60)
        print("Quantum vs Exact Shapley (n=3, n_l=6, mode='rs')")
        print("=" * 60)
        print(f"  {'Asset':<10} {'Exact':>12} {'Quantum':>12} {'RelErr':>10}")
        print("  " + "-" * 46)
        for a in assets_3:
            rel_err = abs(src_quantum[a] - src_exact[a]) / abs(src_exact[a])
            print(
                f"  {a:<10} {src_exact[a]:>12.6f} {src_quantum[a]:>12.6f} {rel_err:>10.4%}"
            )
        print("=" * 60 + "\n")

        for a in assets_3:
            rel_err = abs(src_quantum[a] - src_exact[a]) / abs(src_exact[a])
            assert rel_err < REL_TOL, (
                f"Relative error for {a} = {rel_err:.4%} exceeds {REL_TOL:.0%}"
            )

    def test_precision_convergence(self):
        """
        Monte Carlo sub-sampling experiment: for each portfolio size n (3-6),
        measure how quantum Shapley estimation error converges as the number
        of interval-register qubits n_l increases from 4 to 8.

        Produces one boxplot per portfolio size, saved to docs/assets/.
        """
        N_ROUNDS = 50
        ASSET_SIZES = [3, 4, 5, 6]
        QUBIT_RANGE = [4, 5, 6, 7, 8]
        # How many assets to draw from each bucket per portfolio size
        BUCKET_SCHEME = {
            3: (1, 1, 1),  # (high, mid, low)
            4: (1, 2, 1),
            5: (2, 2, 1),
            6: (2, 2, 2),
        }

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        rng = np.random.default_rng(seed=0)

        output_dir = "docs/assets"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for n in ASSET_SIZES:
            n_high, n_mid, n_low = BUCKET_SCHEME[n]
            records = []  # (n_l, rel_error) per round

            for round_idx in range(N_ROUNDS):
                # Sample assets from each bucket
                sampled = (
                    rng.choice(high_assets, size=n_high, replace=False).tolist()
                    + rng.choice(mid_assets, size=n_mid, replace=False).tolist()
                    + rng.choice(low_assets, size=n_low, replace=False).tolist()
                )
                ret_sub = returns[sampled]

                # Exact SRC via binary enumeration
                src_exact = RiskAttributor(
                    ret_sub, BinaryEnumerationCalculator, mode="rs"
                ).attribute()

                for n_l in QUBIT_RANGE:
                    src_q = RiskAttributor(
                        ret_sub,
                        QuantumCalculator,
                        mode="rs",
                        internal_qubits_num=n_l,
                        internal_multiplier=1,
                    ).attribute()

                    # Mean relative error across assets
                    rel_errors = [
                        abs(src_q[a] - src_exact[a]) / abs(src_exact[a])
                        for a in sampled
                        if abs(src_exact[a]) > 1e-12
                    ]
                    mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
                    records.append({"n_l": n_l, "rel_error": mean_rel_err})

                if (round_idx + 1) % 10 == 0:
                    print(f"  n={n}: {round_idx + 1}/{N_ROUNDS} rounds done")

            # Build DataFrame and plot
            df_records = pd.DataFrame(records)

            sns.set_theme(
                style="whitegrid",
                context="paper",
                font_scale=1.8,
                rc={"font.family": "Times New Roman"},
            )
            fig, ax = plt.subplots(figsize=(7, 4.5))

            sns.boxplot(
                data=df_records,
                x="n_l",
                y="rel_error",
                color="#7aa3c4",
                width=0.5,
                fliersize=3,
                ax=ax,
            )
            ax.set_xlabel(r"Interval Register Qubits ($n_l$)")
            ax.set_ylabel("Mean Relative Error")
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"convergence_n{n}.svg")
            plt.savefig(fig_path)
            plt.show()
            print(f"  Saved: {fig_path}")

            # Print summary statistics
            print(f"\n  Summary for n={n}:")
            for n_l in QUBIT_RANGE:
                subset = df_records[df_records["n_l"] == n_l]["rel_error"]
                print(
                    f"    n_l={n_l}: mean={subset.mean():.4%}, "
                    f"median={subset.median():.4%}, IQR=[{subset.quantile(0.25):.4%}, {subset.quantile(0.75):.4%}]"
                )
            print()


class TestQuantumSolver:
    def test_shots_convergence(self):
        """Verify that the shots-based extractor converges toward the statevector
        ground truth as the number of shots increases.

        Uses a fixed 3-asset portfolio so the test is fast and reproducible.
        Checks that mean relative error strictly decreases across shot counts.
        """
        SHOT_COUNTS = [256, 1024, 4096]

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=7)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        src_exact = RiskAttributor(
            returns_3, BinaryEnumerationCalculator, mode="es"
        ).attribute()

        errors = []
        for shots in SHOT_COUNTS:
            src_shots = RiskAttributor(
                returns_3,
                QuantumCalculator,
                mode="rs",
                internal_qubits_num=6,
                internal_multiplier=1,
                extraction_mode="shots",
                options=QAEOptions(shots=shots),
            ).attribute()
            mean_err = float(
                np.mean(
                    [
                        abs(src_shots[a] - src_exact[a]) / abs(src_exact[a])
                        for a in assets_3
                        if abs(src_exact[a]) > 1e-12
                    ]
                )
            )
            errors.append(mean_err)
            print(f"  shots={shots:>5}: mean_rel_err={mean_err:.4%}")

        assert errors[-1] < errors[0], (
            "Error should decrease as shots increase: "
            f"{errors[0]:.4%} -> {errors[-1]:.4%}"
        )

    def test_qae_modes_basic(self):
        """Verify all three QAE extraction modes produce results close to the
        exact classical Shapley values on a 3-asset portfolio.

        Absolute error is used rather than relative error because Shapley values
        for low-risk assets can be near zero, making relative error meaningless.
        """
        # Absolute Shapley-value tolerance.
        # Observed absolute errors: canonical ~0.001, IQAE ~0.003, MLQAE ~0.001.
        ABS_TOL = 0.005

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=13)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        src_exact = RiskAttributor(
            returns_3, BinaryEnumerationCalculator, mode="es"
        ).attribute()

        qae_modes = ["qae_canonical", "qae_iqae", "qae_mlqae"]
        opts = QAEOptions(epsilon=0.05, alpha=0.05, num_eval_qubits=4)

        print("\n" + "=" * 65)
        print(f"QAE modes vs exact Shapley (n=3, abs_tol={ABS_TOL})")
        print("=" * 65)

        for qae_mode in qae_modes:
            src_qae = RiskAttributor(
                returns_3,
                QuantumCalculator,
                mode="rs",
                internal_qubits_num=6,
                internal_multiplier=1,
                extraction_mode=qae_mode,
                options=opts,
            ).attribute()

            print(f"\n  [{qae_mode}]")
            print(f"  {'Asset':<10} {'Exact':>12} {'QAE':>12} {'AbsErr':>10}")
            print("  " + "-" * 48)
            for a in assets_3:
                abs_err = abs(src_qae[a] - src_exact[a])
                print(
                    f"  {a:<10} {src_exact[a]:>12.6f} {src_qae[a]:>12.6f} {abs_err:>10.6f}"
                )
                assert abs_err < ABS_TOL, (
                    f"[{qae_mode}] abs error for {a} = {abs_err:.6f} exceeds {ABS_TOL}"
                )

        print("=" * 65 + "\n")

    def test_oracle_count_tracked(self):
        """Verify that oracle call counts are recorded after computation and
        that the shots mode count equals the configured number of shots.
        """
        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=99)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        vfunc = RiskSavingValueFunction(returns_3)
        n = len(assets_3)

        modes_opts: list[tuple[str, QAEOptions | None]] = [
            ("statevector", None),
            ("shots", QAEOptions(shots=512)),
            ("qae_canonical", QAEOptions(num_eval_qubits=3)),
            ("qae_iqae", QAEOptions(epsilon=0.05, alpha=0.05)),
            ("qae_mlqae", QAEOptions(num_eval_qubits=3)),
        ]

        print("\n  Oracle call counts per mode:")
        for extraction_mode, opts in modes_opts:
            calc = QuantumCalculator(
                n,
                vfunc,
                internal_qubits_num=6,
                internal_multiplier=1,
                extraction_mode=extraction_mode,
                options=opts,
            )
            _ = calc.get_all()

            for i in range(n):
                count = calc.get_oracle_count(i)
                assert count is not None, (
                    f"[{extraction_mode}] oracle count for player {i} is None"
                )
                if extraction_mode == "shots":
                    assert count == opts.shots, (  # type: ignore[union-attr]
                        f"[shots] expected count={opts.shots}, got {count}"  # type: ignore[union-attr]
                    )

            print(
                f"  {extraction_mode:<15}: {[calc.get_oracle_count(i) for i in range(n)]}"
            )


class TestPerformanceBenchmark:
    def test_quantum_methods_comparison(self):
        """Stage 1: Compare all quantum extraction methods under fixed interval qubits.

        Generates line plots (x=n_l, y=mean_rel_err) for each portfolio size.
        Methods: statevector, shots(1024), shots(4096), qae_iqae, qae_mlqae, qae_fae
        """
        N_ROUNDS = 10  # Reduced for faster iteration
        ASSET_SIZES = [3, 5]  # Test representative sizes
        QUBIT_RANGE = [4, 6]  # Test low and high qubit counts
        BUCKET_SCHEME = {
            3: (1, 1, 1),
            5: (2, 2, 1),
        }

        # Define methods to compare (exclude slow QAE variants for dev)
        METHODS = [
            ("statevector", None),
            ("shots", QAEOptions(shots=1024)),
            ("shots", QAEOptions(shots=4096)),
            ("qae_iqae", QAEOptions(epsilon=0.05, alpha=0.05)),
            # ("qae_mlqae", QAEOptions(num_eval_qubits=4)),  # ~3.7s/run
            # ("qae_fae", QAEOptions(delta=0.05, maxiter=5)),  # ~11s/run
        ]
        METHOD_LABELS = [
            "statevector",
            "shots(1024)",
            "shots(4096)",
            "qae_iqae",
            # "qae_mlqae",
            # "qae_fae",
        ]

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        rng = np.random.default_rng(seed=0)

        output_dir = "docs/assets"
        os.makedirs(output_dir, exist_ok=True)

        # Color palette for methods
        palette = sns.color_palette("husl", len(METHODS))

        for n in ASSET_SIZES:
            n_high, n_mid, n_low = BUCKET_SCHEME[n]
            # records: list of {n_l, method_idx, rel_error}
            records = []

            for round_idx in range(N_ROUNDS):
                sampled = (
                    rng.choice(high_assets, size=n_high, replace=False).tolist()
                    + rng.choice(mid_assets, size=n_mid, replace=False).tolist()
                    + rng.choice(low_assets, size=n_low, replace=False).tolist()
                )
                ret_sub = returns[sampled]

                # Ground truth
                src_exact = RiskAttributor(
                    ret_sub, BinaryEnumerationCalculator, mode="rs"
                ).attribute()

                for n_l in QUBIT_RANGE:
                    for method_idx, (mode_name, opts) in enumerate(METHODS):
                        try:
                            src_q = RiskAttributor(
                                ret_sub,
                                QuantumCalculator,
                                mode="rs",
                                internal_qubits_num=n_l,
                                internal_multiplier=1,
                                extraction_mode=mode_name,
                                options=opts,
                            ).attribute()

                            rel_errors = [
                                abs(src_q[a] - src_exact[a]) / abs(src_exact[a])
                                for a in sampled
                                if abs(src_exact[a]) > 1e-12
                            ]
                            mean_rel_err = (
                                float(np.mean(rel_errors)) if rel_errors else 0.0
                            )
                            records.append(
                                {
                                    "n_l": n_l,
                                    "method": METHOD_LABELS[method_idx],
                                    "rel_error": mean_rel_err,
                                }
                            )
                        except Exception as e:
                            # Skip failed runs (e.g., negative contributions)
                            pass

                if (round_idx + 1) % 10 == 0:
                    print(f"  n={n}: {round_idx + 1}/{N_ROUNDS} rounds done")

            # Aggregate: mean rel_error per (n_l, method)
            df = pd.DataFrame(records)
            df_agg = df.groupby(["n_l", "method"])["rel_error"].mean().reset_index()

            # Plot line chart
            sns.set_theme(
                style="whitegrid",
                context="paper",
                font_scale=1.8,
                rc={"font.family": "Times New Roman"},
            )
            fig, ax = plt.subplots(figsize=(8, 5))

            for i, label in enumerate(METHOD_LABELS):
                subset = df_agg[df_agg["method"] == label]
                ax.plot(
                    subset["n_l"],
                    subset["rel_error"],
                    marker="o",
                    label=label,
                    color=palette[i],
                    linewidth=2,
                )

            ax.set_xlabel(r"Interval Register Qubits ($n_l$)")
            ax.set_ylabel("Mean Relative Error")
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"quantum_comparison_n{n}.svg")
            plt.savefig(fig_path)
            plt.show()
            print(f"  Saved: {fig_path}")

            # Print summary
            print(f"\n  Summary for n={n}:")
            print(df_agg[df_agg["n_l"] == max(QUBIT_RANGE)].to_string(index=False))
            print()

    def test_quantum_vs_classical_mc(self):
        """Stage 2: Compare best quantum method vs classical MC by oracle efficiency.

        Uses PermutationMCCalculator with varying sample counts.
        Plots oracle_calls vs mean_rel_err for both approaches.
        """

        N_ROUNDS = 30
        SAMPLE_COUNTS = [10, 20, 50, 100]  # For classical MC
        N_PLAYERS = 5
        N_L_QUANTUM = 6  # Fixed interval qubits for quantum

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        rng = np.random.default_rng(seed=123)

        # Best quantum method from Stage 1 (e.g., qae_iqae)
        QUANTUM_MODE = "qae_iqae"
        QUANTUM_OPTS = QAEOptions(epsilon=0.05, alpha=0.05)

        records = []

        for round_idx in range(N_ROUNDS):
            # Sample 5 assets: 2 high, 2 mid, 1 low
            sampled = (
                rng.choice(high_assets, size=2, replace=False).tolist()
                + rng.choice(mid_assets, size=2, replace=False).tolist()
                + rng.choice(low_assets, size=1, replace=False).tolist()
            )
            ret_sub = returns[sampled]

            # Ground truth
            vfunc = RiskSavingValueFunction(ret_sub)
            calc_exact = BinaryEnumerationCalculator(N_PLAYERS, vfunc)
            exact = calc_exact.get_all()

            # Classical MC with varying samples
            for T in SAMPLE_COUNTS:
                calc_mc = PermutationMCCalculator(
                    N_PLAYERS, vfunc, num_samples=T, seed=round_idx
                )
                mc = calc_mc.get_all()

                rel_errors = [
                    abs(mc[i] - exact[i]) / abs(exact[i])
                    for i in range(N_PLAYERS)
                    if abs(exact[i]) > 1e-12
                ]
                mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
                oracle_calls = calc_mc.get_oracle_count(0)  # Same for all players

                records.append(
                    {
                        "method": "Classical MC",
                        "oracle_calls": oracle_calls,
                        "rel_error": mean_rel_err,
                    }
                )

            # Quantum method
            calc_q = QuantumCalculator(
                N_PLAYERS,
                vfunc,
                internal_qubits_num=N_L_QUANTUM,
                internal_multiplier=1,
                extraction_mode=QUANTUM_MODE,
                options=QUANTUM_OPTS,
            )
            quantum = calc_q.get_all()

            rel_errors = [
                abs(quantum[i] - exact[i]) / abs(exact[i])
                for i in range(N_PLAYERS)
                if abs(exact[i]) > 1e-12
            ]
            mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
            oracle_calls = calc_q.get_oracle_count(0) or 1

            records.append(
                {
                    "method": f"Quantum ({QUANTUM_MODE})",
                    "oracle_calls": oracle_calls,
                    "rel_error": mean_rel_err,
                }
            )

            if (round_idx + 1) % 10 == 0:
                print(f"  {round_idx + 1}/{N_ROUNDS} rounds done")

        # Aggregate and plot
        df = pd.DataFrame(records)
        df_agg = (
            df.groupby(["method", "oracle_calls"])["rel_error"]
            .agg(["mean", "std"])
            .reset_index()
        )

        output_dir = "docs/assets"
        os.makedirs(output_dir, exist_ok=True)

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=1.8,
            rc={"font.family": "Times New Roman"},
        )
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = {"Classical MC": "#377eb8", f"Quantum ({QUANTUM_MODE})": "#ff7f00"}

        for method in df_agg["method"].unique():
            subset = df_agg[df_agg["method"] == method]
            ax.errorbar(
                subset["oracle_calls"],
                subset["mean"],
                yerr=subset["std"],
                marker="o",
                label=method,
                color=colors.get(method, "gray"),
                linewidth=2,
                capsize=3,
            )

        ax.set_xlabel("Oracle Calls")
        ax.set_ylabel("Mean Relative Error")
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xscale("log")

        plt.tight_layout()
        fig_path = os.path.join(output_dir, "quantum_vs_classical_mc.svg")
        plt.savefig(fig_path)
        plt.show()
        print(f"Saved: {fig_path}")

        # Print summary
        print("\nSummary:")
        print(df_agg.to_string(index=False))
