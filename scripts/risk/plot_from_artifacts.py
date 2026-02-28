from __future__ import annotations

import argparse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.risk.artifacts import (
    REQUIRED_SNAPSHOT_FILES,
    get_artifact_paths,
    require_snapshot_files,
)


QUANTUM_METHOD_ORDER = [
    "shots=1024",
    "shots=4096",
    "I-QAE",
    "F-QAE",
    "ML-QAE",
    "Statevector",
]

QUANTUM_METHOD_ALIASES = {
    "shots(1024)": "shots=1024",
    "shots(4096)": "shots=4096",
    "qae_iqae": "I-QAE",
    "qae_fae": "F-QAE",
    "qae_mlqae": "ML-QAE",
    "statevector": "Statevector",
}

QUANTUM_METHOD_COLORS = {
    "shots=1024": "#4C78A8",
    "shots=4096": "#9C755F",
    "I-QAE": "#F58518",
    "F-QAE": "#E45756",
    "ML-QAE": "#54A24B",
    "Statevector": "#7E57C2",
}

FANGSONG_FONT_PATH = ROOT_DIR / "scripts/fonts/FangSong.ttf"
ENGLISH_FONT = FontProperties(family=["Times New Roman"])
CHINESE_FONT = (
    FontProperties(fname=str(FANGSONG_FONT_PATH))
    if FANGSONG_FONT_PATH.exists()
    else ENGLISH_FONT
)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _apply_legend_fonts(legend) -> None:
    for text in legend.get_texts():
        font_size = text.get_fontsize()
        if _contains_cjk(text.get_text()):
            font_prop = CHINESE_FONT.copy()
        else:
            font_prop = ENGLISH_FONT.copy()
        font_prop.set_size(font_size)
        text.set_fontproperties(font_prop)


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
    out_path = f"{figure_dir}/vol_buckets_trend.svg"
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
    out_path = f"{figure_dir}/restoration_accuracy_bar.svg"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def _plot_quantum_comparison(snapshot_dir: str, figure_dir: str) -> None:
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

    asset_sizes = [3, 4, 5, 6]
    markers_cycle = ["o", "s", "^", "D", "v", "<", ">", "p"]

    frames_by_n: dict[int, pd.DataFrame] = {}
    method_set: set[str] = set()

    for n in asset_sizes:
        frame = pd.read_csv(f"{snapshot_dir}/quantum_comparison_n{n}.csv").copy()
        frame["method"] = (
            frame["method"]
            .astype(str)
            .map(lambda method: QUANTUM_METHOD_ALIASES.get(method, method))
        )
        frames_by_n[n] = frame
        method_set.update(frame["method"].dropna().astype(str).unique().tolist())

    methods = [method for method in QUANTUM_METHOD_ORDER if method in method_set]
    extra_methods = sorted(method_set.difference(methods))
    methods.extend(extra_methods)

    palette: dict[str, str | tuple[float, float, float]] = {
        method: QUANTUM_METHOD_COLORS[method]
        for method in methods
        if method in QUANTUM_METHOD_COLORS
    }
    if extra_methods:
        extra_palette = sns.color_palette("tab10", len(extra_methods))
        for method, color in zip(extra_methods, extra_palette):
            palette[method] = color

    marker_map = {
        method: markers_cycle[idx % len(markers_cycle)]
        for idx, method in enumerate(methods)
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=False)
    for idx, n in enumerate(asset_sizes):
        row_idx, col_idx = divmod(idx, 2)
        ax = axes[row_idx, col_idx]
        df = frames_by_n[n]
        panel_values = np.asarray(df["rel_error"], dtype=float)
        finite_panel_values = panel_values[np.isfinite(panel_values)]
        if finite_panel_values.size == 0:
            raise ValueError(f"No finite rel_error values for n={n}")

        local_min = float(np.min(finite_panel_values))
        local_max = float(np.max(finite_panel_values))
        if np.isclose(local_min, local_max):
            local_pad = max(abs(local_min) * 0.05, 1e-6)
        else:
            local_pad = 0.05 * (local_max - local_min)

        method_labels = df["method"].astype(str)
        methods_in_plot = [
            method for method in methods if (method_labels == method).any()
        ]

        for label in methods_in_plot:
            subset = df[df["method"] == label]
            x_vals = np.asarray(subset["n_l"])
            y_vals = np.asarray(subset["rel_error"])
            order = np.argsort(x_vals)
            plot_kwargs = {
                "marker": marker_map[label],
                "linestyle": "-",
                "label": label,
                "color": palette[label],
                "linewidth": 2,
                "markersize": 7,
                "markerfacecolor": "white",
                "markeredgewidth": 1.2,
            }
            ax.plot(x_vals[order], y_vals[order], **plot_kwargs)

        ax.set_ylim(local_min - local_pad, local_max + local_pad)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=palette[method],
            marker=marker_map[method],
            linestyle="-",
            linewidth=2,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=method,
        )
        for method in methods
    ]
    legend_ncol = max(1, len(methods))
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=18,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
    )
    _apply_legend_fonts(legend)

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    out_path = f"{figure_dir}/quantum_comparison_grid.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_equal_error(snapshot_dir: str, figure_dir: str) -> None:
    df_summary = pd.read_csv(f"{snapshot_dir}/equal_error_oracle_calls_summary.csv")
    df_summary = df_summary.copy()
    df_summary["method"] = df_summary["method"].replace(
        {"IQAE": "I-QAE", "FAE": "F-QAE"}
    )

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
        "I-QAE": "#4daf4a",
        "ML-QAE": "#ff7f00",
        "F-QAE": "#e41a1c",
    }
    markers = {
        "Classical MC": "o",
        "I-QAE": "s",
        "ML-QAE": "^",
        "F-QAE": "D",
    }

    for method in ["Classical MC", "I-QAE", "ML-QAE", "F-QAE"]:
        subset = df_summary[
            (df_summary["method"] == method) & (df_summary["mean_calls"].notna())
        ].sort_values(by="epsilon")
        if subset.empty:
            continue

        legend_label = "经典蒙特卡洛" if method == "Classical MC" else method

        ax.errorbar(
            subset["epsilon"],
            subset["mean_calls"],
            yerr=subset["std_calls"].fillna(0.0),
            marker=markers.get(method, "o"),
            markersize=6,
            label=legend_label,
            color=palette.get(method, "gray"),
            linewidth=1.5,
            capsize=3,
            elinewidth=1.5,
        )

    def _log10_exponent_formatter(value: float, _position: int) -> str:
        if not np.isfinite(value) or value <= 0:
            return ""
        exponent = np.log10(value)
        if np.isclose(exponent, round(exponent), atol=1e-10):
            return f"{int(round(exponent))}"
        return ""

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_ticks = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    x_tick_labels = {
        0.1: "0.1",
        0.05: "0.05",
        0.02: "0.02",
        0.01: "0.01",
        0.005: "0.005",
        0.002: "0.002",
        0.001: "0.001",
    }

    def _epsilon_formatter(value: float, _position: int) -> str:
        for tick, label in x_tick_labels.items():
            if np.isclose(value, tick, rtol=0.0, atol=1e-12):
                return label
        return ""

    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_epsilon_formatter))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(_log10_exponent_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())

    major_ticks = ax.xaxis.get_major_ticks()
    base_tick_label_size = (
        major_ticks[0].label1.get_fontsize()
        if major_ticks
        else plt.rcParams["xtick.labelsize"]
    )
    ax.tick_params(
        axis="both", which="major", labelsize=float(base_tick_label_size) * 1.2
    )

    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    legend = ax.legend(
        loc="lower left",
        fontsize=15,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    _apply_legend_fonts(legend)

    plt.tight_layout()
    out_path = f"{figure_dir}/equal_error_oracle_calls_fixed_grid_fallback.svg"
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def _plot_empirical_cases(snapshot_dir: str, figure_dir: str) -> None:
    hidden_df = pd.read_csv(f"{snapshot_dir}/empirical_hidden_risk.csv")
    hedge_df = pd.read_csv(f"{snapshot_dir}/empirical_hedge_negative.csv")

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.4,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        },
    )

    hidden_plot = hidden_df.sort_values("src_share", ascending=False).reset_index(
        drop=True
    )
    x = np.arange(len(hidden_plot))
    width = 0.42
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        x - width / 2,
        hidden_plot["capital_weight"] * 100,
        width=width,
        label="Capital Weight (%)",
        color="#6BAED6",
    )
    src_positive_color = "#4C78A8"
    src_negative_color = "#72B7B2"
    src_high_risk_color = "#E45756"
    risk_colors = [
        (
            src_high_risk_color
            if tier == "High"
            else (src_negative_color if src_share < 0 else src_positive_color)
        )
        for tier, src_share in zip(hidden_plot["risk_tier"], hidden_plot["src_share"])
    ]
    ax.bar(
        x + width / 2,
        hidden_plot["src_share"] * 100,
        width=width,
        label="SRC Share (%)",
        color=risk_colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_plot["asset"], rotation=45, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)

    ax_right = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    ax_right.tick_params(
        axis="y", which="both", labelsize=15, right=False, length=0, pad=9
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    shared_ticks = ax.get_yticks()
    ax_right.set_yticks(shared_ticks)
    ax_right.set_ylim(ax.get_ylim())
    plt.tight_layout()
    hidden_path = f"{figure_dir}/empirical_hidden_risk.svg"
    fig.savefig(hidden_path)
    plt.close(fig)
    print(f"Saved {hidden_path}")

    hedge_plot = hedge_df.copy()
    x = np.arange(len(hedge_plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(
        x - width / 2,
        hedge_plot["standalone_es"],
        width=width,
        label="Standalone ES",
        color="#72B7B2",
    )
    src_colors = [
        "#F58518" if role == "Risk" else "#54A24B" for role in hedge_plot["role"]
    ]
    ax.bar(
        x + width / 2,
        hedge_plot["src"],
        width=width,
        label="Shapley SRC",
        color=src_colors,
    )
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(hedge_plot["asset"])
    ax.set_ylabel("Risk Value")
    ax.set_title("Scenario B: Standalone ES vs SRC")
    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    _apply_legend_fonts(legend)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    hedge_path = f"{figure_dir}/empirical_hedge_negative.svg"
    fig.savefig(hedge_path)
    plt.close(fig)
    print(f"Saved {hedge_path}")


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
    _plot_empirical_cases(snapshot_dir, figure_dir)


if __name__ == "__main__":
    main()
