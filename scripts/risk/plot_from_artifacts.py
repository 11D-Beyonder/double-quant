from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.risk.artifacts import (
    REQUIRED_SNAPSHOT_FILES,
    get_artifact_paths,
    require_snapshot_files,
)


def _plot_volatility_trend(snapshot_dir: str, figure_dir: str) -> None:
    series = pd.read_csv(f"{snapshot_dir}/vol_buckets_series.csv", parse_dates=["date"])

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=2.0,
        rc={"font.family": "Times New Roman"},
    )
    palette = {
        "Low Volatility": "#4daf4a",
        "Mid Volatility": "#377eb8",
        "High Volatility": "#ff7f00",
    }

    _, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    cum_ret = series[series["series_type"] == "cum_ret"]
    rolling_vol = series[series["series_type"] == "rolling_vol"]

    for label, color in palette.items():
        subset = cum_ret[cum_ret["bucket"] == label].sort_values("date")
        sns.lineplot(
            x=subset["date"], y=subset["value"], color=color, linewidth=2, ax=ax1
        )

    for label, color in palette.items():
        subset = rolling_vol[rolling_vol["bucket"] == label].sort_values("date")
        sns.lineplot(
            x=subset["date"], y=subset["value"], color=color, linewidth=2, ax=ax2
        )

    start_date = cum_ret["date"].min()
    end_date = cum_ret["date"].max()
    demarcation_date = pd.Timestamp("2021-11-01")

    for ax in [ax1, ax2]:
        ax.axvspan(
            start_date, demarcation_date, color="lightgreen", alpha=0.1, zorder=0
        )
        ax.axvspan(demarcation_date, end_date, color="lightcoral", alpha=0.1, zorder=0)
        ax.axvline(
            demarcation_date, color="gray", linestyle="--", linewidth=1.5, zorder=1
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(start_date, end_date)

    ax1.tick_params(labelbottom=False)
    plt.tight_layout(h_pad=1.2)
    out_path = f"{figure_dir}/vol_buckets_trend.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def _plot_restoration_accuracy(snapshot_dir: str, figure_dir: str) -> None:
    df = pd.read_csv(f"{snapshot_dir}/restoration_accuracy.csv")
    mae = float(df["mae"].iloc[0])

    palette_bar = {
        "High": "#c8906a",
        "Mid": "#7aa3c4",
        "Low": "#82ae80",
        "MAE": "#9e9e9e",
    }
    sorted_df = df.sort_values("abs_diff")

    bar_labels = ["MAE"] + sorted_df["asset"].tolist()
    bar_values = [mae] + sorted_df["abs_diff"].tolist()
    bar_colors = [palette_bar["MAE"]] + [
        palette_bar.get(bucket, "#9e9e9e") for bucket in sorted_df["bucket"].tolist()
    ]

    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.7,
        rc={"font.family": "Times New Roman", "mathtext.fontset": "stix"},
    )

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f6f6")
    ax.barh(range(len(bar_labels)), bar_values, color=bar_colors, height=0.55, zorder=2)
    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=12)
    ax.set_ylim(-0.5, len(bar_labels) - 0.5)

    max_value = max(bar_values)
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_value * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, value in enumerate(bar_values):
        value_exp = int(np.floor(np.log10(value))) if value > 0 else 0
        value_mant = value / (10**value_exp) if value > 0 else 0
        label = rf"${value_mant:.2f} \times 10^{{{value_exp}}}$"
        ax.text(
            value + max_value * 0.02,
            i,
            label,
            va="center",
            ha="left",
            fontsize=12,
            color="#333333",
        )

    plt.tight_layout()
    out_path = f"{figure_dir}/restoration_accuracy_bar.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def _plot_quantum_comparison(snapshot_dir: str, figure_dir: str) -> None:
    asset_sizes = [3, 4, 5, 6]
    for n in asset_sizes:
        df = pd.read_csv(f"{snapshot_dir}/quantum_comparison_n{n}.csv")
        methods = df["method"].unique().tolist()
        palette = dict(zip(methods, sns.color_palette("husl", len(methods))))

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=1.8,
            rc={"font.family": "Times New Roman"},
        )
        _, ax = plt.subplots(figsize=(8, 5))

        for label in methods:
            subset = df[df["method"] == label]
            ax.plot(
                subset["n_l"],
                subset["rel_error"],
                marker="o",
                label=label,
                color=palette[label],
                linewidth=2,
            )

        ax.set_xlabel(r"Interval Register Qubits ($n_l$)")
        ax.set_ylabel("Mean Relative Error")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = f"{figure_dir}/quantum_comparison_n{n}.png"
        plt.savefig(out_path)
        print(f"Saved {out_path}")


def _plot_equal_error(snapshot_dir: str, figure_dir: str) -> None:
    df_summary = pd.read_csv(f"{snapshot_dir}/equal_error_oracle_calls_summary.csv")

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.5,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        },
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {
        "Classical MC": "#377eb8",
        "IQAE": "#4daf4a",
        "ML-QAE": "#ff7f00",
        "FAE": "#e41a1c",
    }
    markers = {
        "Classical MC": "o",
        "IQAE": "s",
        "ML-QAE": "^",
        "FAE": "D",
    }

    for method in ["Classical MC", "IQAE", "ML-QAE", "FAE"]:
        subset = df_summary[
            (df_summary["method"] == method) & (df_summary["mean_calls"].notna())
        ].sort_values(by="epsilon")
        if subset.empty:
            continue

        ax.errorbar(
            subset["epsilon"],
            subset["mean_calls"],
            yerr=subset["std_calls"].fillna(0.0),
            marker=markers.get(method, "o"),
            markersize=6,
            label=method,
            color=palette.get(method, "gray"),
            linewidth=1.5,
            capsize=3,
            elinewidth=1.5,
        )

    ax.set_title(
        "Equal-Error Oracle Calls (Fixed Grid + Fallback)", fontsize=14, pad=10
    )
    ax.set_xlabel(r"Target Relative Error ($\epsilon$)", fontsize=12)
    ax.set_ylabel(r"Oracle Calls to Reach $\epsilon$", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend(
        loc="best", fontsize=10, frameon=True, framealpha=0.9, edgecolor="#cccccc"
    )

    plt.tight_layout()
    out_path = f"{figure_dir}/equal_error_oracle_calls_fixed_grid_fallback.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render risk experiment figures from snapshot CSV files",
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    paths = get_artifact_paths()
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    require_snapshot_files(paths.snapshot_dir, REQUIRED_SNAPSHOT_FILES)

    snapshot_dir = str(paths.snapshot_dir)
    figure_dir = str(paths.figure_dir)

    _plot_volatility_trend(snapshot_dir, figure_dir)
    _plot_restoration_accuracy(snapshot_dir, figure_dir)
    _plot_quantum_comparison(snapshot_dir, figure_dir)
    _plot_equal_error(snapshot_dir, figure_dir)


if __name__ == "__main__":
    main()
