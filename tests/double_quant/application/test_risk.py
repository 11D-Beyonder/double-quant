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
    output_path_png = os.path.join(output_dir, "vol_buckets_trend.png")
    plt.savefig(output_path_png)


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

        plt.tight_layout()

        assert mae < MAE_TOL, (
            f"Restoration formula MAE = {mae:.2e} exceeds tolerance {MAE_TOL:.2e}. "
            f"Max single-asset diff = {max(diffs.values()):.2e} on asset '{max(diffs, key=diffs.get)}'"
        )


class TestQuantumSolver:
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


class TestQuantumPerformance:
    def test_quantum_methods_comparison(self):
        """Stage 1: Compare all quantum extraction methods under fixed interval qubits.

        Generates line plots (x=n_l, y=mean_rel_err) for each portfolio size.
        Methods: statevector, shots(1024), shots(4096), qae_iqae, qae_mlqae, qae_fae
        """
        N_ROUNDS = 50  # Reduced for faster iteration
        ASSET_SIZES = [3, 4, 5, 6]  # Test representative sizes
        QUBIT_RANGE = [2, 3, 4, 5, 6]  # Test low and high qubit counts
        BUCKET_SCHEME = {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1), 6: (2, 2, 2)}

        # Define methods to compare (exclude slow QAE variants for dev)
        METHODS = [
            ("statevector", None),
            ("shots", QAEOptions(shots=1024)),
            ("shots", QAEOptions(shots=4096)),
            ("qae_iqae", QAEOptions(epsilon=0.05, alpha=0.05)),
            ("qae_mlqae", QAEOptions(num_eval_qubits=4)),  # ~3.7s/run
            ("qae_fae", QAEOptions(delta=0.05, maxiter=5)),  # ~11s/run
        ]
        METHOD_LABELS = [
            "statevector",
            "shots(1024)",
            "shots(4096)",
            "qae_iqae",
            "qae_mlqae",
            "qae_fae",
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
                    ret_sub, BinaryEnumerationCalculator, mode="es"
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
                        except Exception:
                            # Skip failed runs (e.g., negative contributions)
                            pass

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
            _, ax = plt.subplots(figsize=(8, 5))

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
            fig_path = os.path.join(output_dir, f"quantum_comparison_n{n}.png")
            plt.savefig(fig_path)
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
        SAMPLE_COUNTS = [10, 20, 50, 100]  # For classical MC → oracle_calls = 2T
        N_PLAYERS = 5
        N_L_QUANTUM = 6  # Fixed interval qubits for quantum

        # Each QAE variant sweeps its primary precision parameter to span a range
        # of oracle_calls, demonstrating the O(1/ε) complexity on the log-log chart.
        QUANTUM_METHOD_SWEEPS: list[tuple[str, str, list[QAEOptions]]] = [
            (
                "qae_mlqae",
                "ML-QAE",
                [QAEOptions(num_eval_qubits=k) for k in [2, 3, 4, 5]],
            ),
            # Temporarily hidden to keep the plot focused on Classical MC vs ML-QAE.
            # (
            #     "qae_iqae",
            #     "IQAE",
            #     [QAEOptions(epsilon=e, alpha=0.05) for e in [0.10, 0.05, 0.02, 0.01]],
            # ),
            # (
            #     "qae_fae",
            #     "FAE",
            #     [QAEOptions(delta=0.05, maxiter=m) for m in [2, 3, 5, 7]],
            # ),
            # (
            #     "qae_canonical",
            #     "Canonical-QAE",
            #     [QAEOptions(num_eval_qubits=k) for k in [2, 3, 4, 5]],
            # ),
        ]

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        rng = np.random.default_rng(seed=123)

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

            # All QAE methods, each sweeping its primary parameter for oracle_call spread
            for quantum_mode, quantum_label, quantum_configs in QUANTUM_METHOD_SWEEPS:
                for quantum_opts in quantum_configs:
                    try:
                        calc_q = QuantumCalculator(
                            N_PLAYERS,
                            vfunc,
                            internal_qubits_num=N_L_QUANTUM,
                            internal_multiplier=1,
                            extraction_mode=quantum_mode,  # type: ignore[arg-type]
                            options=quantum_opts,
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
                                "method": quantum_label,
                                "oracle_calls": oracle_calls,
                                "rel_error": mean_rel_err,
                            }
                        )
                    except Exception:
                        pass

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
        fig, ax = plt.subplots(figsize=(9, 5))

        all_methods = ["Classical MC"] + [lbl for _, lbl, _ in QUANTUM_METHOD_SWEEPS]
        palette = dict(zip(all_methods, sns.color_palette("tab10", len(all_methods))))

        for method in df_agg["method"].unique():
            subset = df_agg[df_agg["method"] == method]  # type: ignore[index]
            subset = subset.sort_values(by="oracle_calls")  # type: ignore[union-attr]
            ax.errorbar(
                subset["oracle_calls"],
                subset["mean"],
                yerr=subset["std"],
                marker="o",
                label=method,
                color=palette.get(method, "gray"),
                linewidth=2,
                capsize=3,
            )

        fit_targets = {
            "Classical MC": {"slope": -1.0, "formula": "c/T"},
            "ML-QAE": {"slope": -0.5, "formula": "c/sqrt(T)"},
        }
        fit_reports: list[str] = []

        for method, spec in fit_targets.items():
            subset = df_agg[df_agg["method"] == method]
            subset = subset.sort_values(by="oracle_calls")

            if subset.empty:
                fit_reports.append(f"{method}: skipped (no data points)")
                continue

            x = subset["oracle_calls"].to_numpy(dtype=float)
            y = subset["mean"].to_numpy(dtype=float)
            valid = (x > 0) & (y > 0)
            x = x[valid]
            y = y[valid]

            if len(x) == 0:
                fit_reports.append(
                    f"{method}: skipped (no positive points for log fit)"
                )
                continue

            if len(x) == 1:
                x_fit = np.array([0.9 * x[0], 1.1 * x[0]])
            else:
                x_fit = np.geomspace(float(np.min(x)), float(np.max(x)), num=100)

            color = palette.get(method, "gray")

            # Fixed-slope complexity fit in log-space:
            # log(y) = log(c) + p*log(x), with preset p.
            p_fixed = float(spec["slope"])
            log_c_fixed = float(np.mean(np.log(y) - p_fixed * np.log(x)))
            c_fixed = float(np.exp(log_c_fixed))
            y_fixed = c_fixed * np.power(x_fit, p_fixed)
            ax.plot(
                x_fit,
                y_fixed,
                linestyle="--",
                linewidth=2,
                color=color,
                alpha=0.9,
                label=f"{method} fixed: {spec['formula']}",
            )
            fit_reports.append(
                f"{method} fixed fit: c={c_fixed:.6g}, slope={p_fixed:.3f}"
            )

            if len(x) >= 2 and np.unique(x).size >= 2:
                p_reg, b_reg = np.polyfit(np.log(x), np.log(y), 1)
                p_reg = float(p_reg)
                c_reg = float(np.exp(b_reg))
                y_reg = c_reg * np.power(x_fit, p_reg)
                ax.plot(
                    x_fit,
                    y_reg,
                    linestyle=":",
                    linewidth=2,
                    color=color,
                    alpha=0.9,
                    label=f"{method} regression: c*T^p",
                )
                fit_reports.append(
                    f"{method} regression fit: c={c_reg:.6g}, slope={p_reg:.3f}"
                )
            else:
                fit_reports.append(
                    f"{method} regression fit: skipped (insufficient points)"
                )

        ax.set_xlabel("Oracle Calls")
        ax.set_ylabel("Mean Relative Error")
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.tight_layout()
        fig_path = os.path.join(output_dir, "quantum_vs_classical_mc.png")
        plt.savefig(fig_path)
        print(f"Saved: {fig_path}")

        # Print summary
        print("\nSummary:")
        print(df_agg.to_string(index=False))
        print("\nComplexity Fits:")
        for line in fit_reports:
            print(line)

    def _prepare_equal_error_data(
        self,
    ) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        return returns, low_assets, mid_assets, high_assets

    def _sample_round_problem(
        self,
        *,
        returns: pd.DataFrame,
        low_assets: list[str],
        mid_assets: list[str],
        high_assets: list[str],
        rng: np.random.Generator,
        n_players: int,
    ) -> tuple[RiskSavingValueFunction, list[float]]:
        if n_players != 5:
            raise ValueError("This benchmark currently assumes n_players=5")

        sampled = (
            rng.choice(high_assets, size=2, replace=False).tolist()
            + rng.choice(mid_assets, size=2, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        ret_sub = returns[sampled]
        vfunc = RiskSavingValueFunction(ret_sub)
        exact = BinaryEnumerationCalculator(n_players, vfunc).get_all()
        return vfunc, exact

    @staticmethod
    def _mean_relative_error(estimate: list[float], exact: list[float]) -> float:
        rel_errors = [
            abs(estimate[i] - exact[i]) / abs(exact[i])
            for i in range(len(exact))
            if abs(exact[i]) > 1e-12
        ]
        return float(np.mean(rel_errors)) if rel_errors else 0.0

    def _evaluate_classical_mc_point(
        self,
        *,
        n_players: int,
        vfunc: RiskSavingValueFunction,
        exact: list[float],
        num_samples: int,
        seed: int,
    ) -> tuple[int, float]:
        calc_mc = PermutationMCCalculator(
            n_players, vfunc, num_samples=num_samples, seed=seed
        )
        estimate = calc_mc.get_all()
        oracle_calls = calc_mc.get_oracle_count(0)
        rel_error = self._mean_relative_error(estimate, exact)
        return oracle_calls, rel_error

    def _evaluate_mlqae_point(
        self,
        *,
        n_players: int,
        vfunc: RiskSavingValueFunction,
        exact: list[float],
        internal_qubits_num: int,
        num_eval_qubits: int,
    ) -> tuple[int, float] | None:
        try:
            calc_q = QuantumCalculator(
                n_players,
                vfunc,
                internal_qubits_num=internal_qubits_num,
                internal_multiplier=1,
                extraction_mode="qae_mlqae",
                options=QAEOptions(num_eval_qubits=num_eval_qubits),
            )
            estimate = calc_q.get_all()
            oracle_calls = calc_q.get_oracle_count(0) or 1
            rel_error = self._mean_relative_error(estimate, exact)
            return oracle_calls, rel_error
        except Exception:
            return None

    @staticmethod
    def _min_calls_reaching_epsilon(
        points: list[tuple[int, float]], epsilon: float
    ) -> int | None:
        reached = [oracle_calls for oracle_calls, err in points if err <= epsilon]
        if not reached:
            return None
        return int(min(reached))

    @staticmethod
    def _fallback_calls_from_loglog_fit(
        points: list[tuple[int, float]], epsilon: float
    ) -> int | None:
        x = np.array(
            [calls for calls, err in points if calls > 0 and err > 0], dtype=float
        )
        y = np.array(
            [err for calls, err in points if calls > 0 and err > 0], dtype=float
        )
        if len(x) < 2 or np.unique(x).size < 2:
            return None

        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        slope = float(slope)
        intercept = float(intercept)

        if np.isclose(slope, 0.0) or slope > 0:
            return None

        predicted_calls = float(np.exp((np.log(epsilon) - intercept) / slope))
        if not np.isfinite(predicted_calls) or predicted_calls <= 0:
            return None

        return int(np.ceil(predicted_calls))

    @staticmethod
    def _aggregate_equal_error_calls(
        rows: list[dict[str, object]],
        n_rounds: int,
    ) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        summary_rows = []
        for (method, epsilon), group in df.groupby(["method", "epsilon"]):
            valid = group[group["oracle_calls"].notna()]
            reachable = len(valid)
            reachable_ratio = float(reachable / n_rounds)

            if reachable > 0:
                calls = valid["oracle_calls"].astype(float)
                mean_calls = float(calls.mean())
                std_calls = float(calls.std(ddof=0))
                source_values = valid["source_type"].astype(str)
                unique_sources = sorted(source_values.unique().tolist())
                source_type = unique_sources[0] if len(unique_sources) == 1 else "mixed"
            else:
                mean_calls = float("nan")
                std_calls = float("nan")
                source_type = "none"

            summary_rows.append(
                {
                    "method": method,
                    "epsilon": float(epsilon),
                    "mean_calls": mean_calls,
                    "std_calls": std_calls,
                    "reachable_ratio": reachable_ratio,
                    "source_type": source_type,
                }
            )

        summary = pd.DataFrame(summary_rows)
        return summary.sort_values(["epsilon", "method"]).reset_index(drop=True)

    @staticmethod
    def _plot_equal_error_calls(
        df_summary: pd.DataFrame, fig_path: str, title: str
    ) -> None:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=1.8,
            rc={"font.family": "Times New Roman"},
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = {"Classical MC": "#377eb8", "ML-QAE": "#ff7f00"}

        for method in ["Classical MC", "ML-QAE"]:
            subset = df_summary[
                (df_summary["method"] == method) & (df_summary["mean_calls"].notna())
            ]
            subset = subset.sort_values(by="epsilon")
            if subset.empty:
                continue

            ax.errorbar(
                subset["epsilon"],
                subset["mean_calls"],
                yerr=subset["std_calls"].fillna(0.0),
                marker="o",
                label=method,
                color=palette.get(method, "gray"),
                linewidth=2,
                capsize=3,
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Target Relative Error (epsilon)")
        ax.set_ylabel("Oracle Calls to Reach epsilon")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=11)

        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Saved: {fig_path}")

    @staticmethod
    def _print_equal_error_summary(title: str, df_summary: pd.DataFrame) -> None:
        print(f"\n{title}")
        print(df_summary.to_string(index=False))

    def test_equal_error_oracle_calls_fixed_grid_with_fallback(self):
        """Compare oracle calls at same epsilon with fixed-grid plus fit fallback."""
        epsilons = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
        n_rounds = 8
        n_players = 5
        n_l_quantum = 6
        classical_samples = [
            10,
            20,
            40,
            80,
            160,
            320,
            640,
            1000,
            2000,
            5000,
            10000,
            20000,
        ]
        mlqae_eval_qubits = [2, 3, 4, 5]

        returns, low_assets, mid_assets, high_assets = self._prepare_equal_error_data()
        rng = np.random.default_rng(seed=2028)

        rows: list[dict[str, object]] = []
        for round_idx in range(n_rounds):
            vfunc, exact = self._sample_round_problem(
                returns=returns,
                low_assets=low_assets,
                mid_assets=mid_assets,
                high_assets=high_assets,
                rng=rng,
                n_players=n_players,
            )

            points: dict[str, list[tuple[int, float]]] = {
                "Classical MC": [],
                "ML-QAE": [],
            }
            for t in classical_samples:
                point = self._evaluate_classical_mc_point(
                    n_players=n_players,
                    vfunc=vfunc,
                    exact=exact,
                    num_samples=t,
                    seed=round_idx * 1000 + t,
                )
                points["Classical MC"].append(point)

            for k in mlqae_eval_qubits:
                point = self._evaluate_mlqae_point(
                    n_players=n_players,
                    vfunc=vfunc,
                    exact=exact,
                    internal_qubits_num=n_l_quantum,
                    num_eval_qubits=k,
                )
                if point is not None:
                    points["ML-QAE"].append(point)

            for method in ["Classical MC", "ML-QAE"]:
                for epsilon in epsilons:
                    min_calls = self._min_calls_reaching_epsilon(
                        points[method], epsilon
                    )
                    source_type = "discrete"
                    if min_calls is None:
                        min_calls = self._fallback_calls_from_loglog_fit(
                            points[method], epsilon
                        )
                        source_type = "fallback" if min_calls is not None else "none"

                    rows.append(
                        {
                            "round": round_idx,
                            "method": method,
                            "epsilon": epsilon,
                            "oracle_calls": min_calls,
                            "source_type": source_type,
                        }
                    )

            print(f"  fixed-grid+fallback: {round_idx + 1}/{n_rounds} rounds done")

        summary = self._aggregate_equal_error_calls(rows, n_rounds=n_rounds)
        fig_path = "docs/assets/equal_error_oracle_calls_fixed_grid_fallback.png"
        self._plot_equal_error_calls(
            summary,
            fig_path,
            "Equal-Error Oracle Calls (Fixed Grid + Fallback)",
        )
        self._print_equal_error_summary("Fixed Grid + Fallback Summary:", summary)
