from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.common.util import divide_by_volatility
from double_quant.solver.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from experiments.risk.artifacts import (
    DataPreparation,
    get_artifact_paths,
    write_manifest,
)


EXPERIMENT_CHOICES = (
    "volatility",
    "restoration",
    "quantum_comparison",
    "equal_error",
    "empirical_scenario",
)


def _needs_refresh(paths: list[Path], force: bool) -> bool:
    if force:
        return True
    return any(not path.exists() for path in paths)


def _generate_volatility_snapshots(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    metrics_path = snapshot_dir / "vol_buckets_metrics.csv"
    series_path = snapshot_dir / "vol_buckets_series.csv"

    if not _needs_refresh([metrics_path, series_path], force):
        print("Skip volatility snapshots: files already exist")
        return

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

    bucket_labels = ["Low Volatility", "Mid Volatility", "High Volatility"]
    bucket_assets = [low_assets, mid_assets, high_assets]
    quantiles = ["Bottom 30%", "Mid 40%", "Top 30%"]

    metrics_rows: list[dict[str, float | str]] = []
    series_rows: list[dict[str, float | str]] = []

    for label, assets, quantile in zip(bucket_labels, bucket_assets, quantiles):
        bucket_returns = returns[assets].mean(axis=1)
        ann_vol = float(bucket_returns.std() * np.sqrt(252))
        ann_ret = float(np.exp(bucket_returns.mean() * 252) - 1)

        cum_ret = np.exp(bucket_returns.cumsum())
        rolling_vol = bucket_returns.rolling(window=30).std() * np.sqrt(252)
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = float(drawdown.min())

        metrics_rows.append(
            {
                "Risk Bucket": label,
                "Quantile": quantile,
                "Avg Volatility": ann_vol,
                "Ann. Return": ann_ret,
                "Max Drawdown": max_dd,
            }
        )

        for date, value in cum_ret.items():
            series_rows.append(
                {
                    "date": date,
                    "bucket": label,
                    "series_type": "cum_ret",
                    "value": float(value),
                }
            )
        for date, value in rolling_vol.dropna().items():
            series_rows.append(
                {
                    "date": date,
                    "bucket": label,
                    "series_type": "rolling_vol",
                    "value": float(value),
                }
            )
        for date, value in drawdown.items():
            series_rows.append(
                {
                    "date": date,
                    "bucket": label,
                    "series_type": "drawdown",
                    "value": float(value),
                }
            )

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(series_rows).to_csv(series_path, index=False)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {series_path}")


def _generate_restoration_snapshot(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    output_path = snapshot_dir / "restoration_accuracy.csv"
    if not _needs_refresh([output_path], force):
        print("Skip restoration snapshot: file already exists")
        return

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

    rng = np.random.default_rng(seed=0)
    assets_5 = (
        rng.choice(high_assets, size=2, replace=False).tolist()
        + rng.choice(mid_assets, size=2, replace=False).tolist()
        + rng.choice(low_assets, size=1, replace=False).tolist()
    )
    returns_5 = returns[assets_5]

    src_es = RiskAttributor(
        returns_5, BinaryEnumerationCalculator, mode="es"
    ).attribute()
    src_rs = RiskAttributor(
        returns_5, BinaryEnumerationCalculator, mode="rs"
    ).attribute()

    bucket_map: dict[str, str] = (
        {asset: "High" for asset in high_assets}
        | {asset: "Mid" for asset in mid_assets}
        | {asset: "Low" for asset in low_assets}
    )
    diffs = {asset: abs(src_rs[asset] - src_es[asset]) for asset in assets_5}
    mae = float(np.mean(list(diffs.values())))

    rows = [
        {
            "asset": asset,
            "bucket": bucket_map[asset],
            "src_es": float(src_es[asset]),
            "src_rs": float(src_rs[asset]),
            "abs_diff": float(diffs[asset]),
            "mae": mae,
        }
        for asset in assets_5
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def _generate_quantum_comparison_snapshots(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    n_rounds = 50
    asset_sizes = [3, 4, 5, 6]
    qubit_range = [2, 3, 4, 5, 6]
    bucket_scheme = {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1), 6: (2, 2, 2)}

    methods: list[
        tuple[
            Literal[
                "statevector",
                "shots",
                "qae_iqae",
                "qae_mlqae",
                "qae_fae",
            ],
            QAEOptions | None,
            str,
        ]
    ] = [
        ("statevector", None, "Statevector"),
        ("shots", QAEOptions(shots=1024), "shots=1024"),
        ("shots", QAEOptions(shots=4096), "shots=4096"),
        ("qae_iqae", QAEOptions(epsilon=0.01, alpha=0.01), "I-QAE"),
        ("qae_mlqae", QAEOptions(num_eval_qubits=4), "ML-QAE"),
        ("qae_fae", QAEOptions(delta=0.05, maxiter=5), "F-QAE"),
    ]

    target_files = [snapshot_dir / f"quantum_comparison_n{n}.csv" for n in asset_sizes]
    if not _needs_refresh(target_files, force):
        print("Skip quantum comparison snapshots: files already exist")
        return

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
    rng = np.random.default_rng(seed=0)

    for n in asset_sizes:
        out_path = snapshot_dir / f"quantum_comparison_n{n}.csv"
        if out_path.exists() and not force:
            print(f"Skip existing {out_path}")
            continue

        n_high, n_mid, n_low = bucket_scheme[n]
        records: list[dict[str, float | int | str]] = []

        for round_idx in range(n_rounds):
            sampled = (
                rng.choice(high_assets, size=n_high, replace=False).tolist()
                + rng.choice(mid_assets, size=n_mid, replace=False).tolist()
                + rng.choice(low_assets, size=n_low, replace=False).tolist()
            )
            ret_sub = returns[sampled]
            src_exact = RiskAttributor(
                ret_sub, BinaryEnumerationCalculator, mode="es"
            ).attribute()

            for n_l in qubit_range:
                for mode_name, opts, label in methods:
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
                    except Exception:
                        continue

                    rel_errors = [
                        abs(src_q[asset] - src_exact[asset]) / abs(src_exact[asset])
                        for asset in sampled
                        if abs(src_exact[asset]) > 1e-12
                    ]
                    mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
                    records.append(
                        {"n_l": n_l, "method": label, "rel_error": mean_rel_err}
                    )

            print(f"quantum comparison n={n}: {round_idx + 1}/{n_rounds}")

        df = pd.DataFrame(records)
        df_agg = df.groupby(["n_l", "method"])["rel_error"].mean().reset_index()
        df_agg.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


def _mean_relative_error(estimate: list[float], exact: list[float]) -> float:
    rel_errors = [
        abs(estimate[i] - exact[i]) / abs(exact[i])
        for i in range(len(exact))
        if abs(exact[i]) > 1e-12
    ]
    return float(np.mean(rel_errors)) if rel_errors else 0.0


def _min_calls_reaching_epsilon(
    points: list[tuple[int, float]], epsilon: float
) -> int | None:
    reached = [oracle_calls for oracle_calls, err in points if err <= epsilon]
    if not reached:
        return None
    return int(min(reached))


def _fallback_calls_from_loglog_fit(
    points: list[tuple[int, float]], epsilon: float
) -> int | None:
    x = np.array([calls for calls, err in points if calls > 0 and err > 0], dtype=float)
    y = np.array([err for calls, err in points if calls > 0 and err > 0], dtype=float)
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


def _aggregate_equal_error_calls(
    rows: list[dict[str, object]], n_rounds: int
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


def _generate_equal_error_snapshot(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    out_path = snapshot_dir / "equal_error_oracle_calls_summary.csv"
    if not _needs_refresh([out_path], force):
        print("Skip equal-error summary: file already exists")
        return

    epsilons = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    n_rounds = 8
    n_players = 5
    n_l_quantum = 6
    classical_samples = [10, 20, 40, 80, 160, 320, 640, 1000, 2000, 5000, 10000, 20000]
    iqae_epsilons = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005]
    iqae_alpha = 0.01
    mlqae_eval_qubits = [2, 3, 4, 5, 6]
    fae_maxiters = [3, 4, 5, 6, 7]
    fae_delta = 0.05

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
    rng = np.random.default_rng(seed=2028)

    rows: list[dict[str, object]] = []
    for round_idx in range(n_rounds):
        sampled = (
            rng.choice(high_assets, size=2, replace=False).tolist()
            + rng.choice(mid_assets, size=2, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        ret_sub = returns[sampled]
        vfunc = RiskSavingValueFunction(ret_sub)
        exact = BinaryEnumerationCalculator(n_players, vfunc).get_all()

        points: dict[str, list[tuple[int, float]]] = {
            "Classical MC": [],
            "I-QAE": [],
            "ML-QAE": [],
            "F-QAE": [],
        }
        for t in classical_samples:
            calc_mc = PermutationMCCalculator(
                n_players, vfunc, num_samples=t, seed=round_idx * 1000 + t
            )
            estimate = calc_mc.get_all()
            points["Classical MC"].append(
                (calc_mc.get_oracle_count(0), _mean_relative_error(estimate, exact))
            )

        for target_epsilon in iqae_epsilons:
            try:
                calc_q = QuantumCalculator(
                    n_players,
                    vfunc,
                    internal_qubits_num=n_l_quantum,
                    internal_multiplier=1,
                    extraction_mode="qae_iqae",
                    options=QAEOptions(epsilon=target_epsilon, alpha=iqae_alpha),
                )
            except Exception:
                continue

            estimate = calc_q.get_all()
            oracle_calls = calc_q.get_oracle_count(0)
            if oracle_calls is None:
                continue
            points["I-QAE"].append(
                (max(1, oracle_calls), _mean_relative_error(estimate, exact))
            )

        for k in mlqae_eval_qubits:
            try:
                calc_q = QuantumCalculator(
                    n_players,
                    vfunc,
                    internal_qubits_num=n_l_quantum,
                    internal_multiplier=1,
                    extraction_mode="qae_mlqae",
                    options=QAEOptions(num_eval_qubits=k),
                )
            except Exception:
                continue

            estimate = calc_q.get_all()
            oracle_calls = calc_q.get_oracle_count(0)
            if oracle_calls is None:
                continue
            points["ML-QAE"].append(
                (max(1, oracle_calls), _mean_relative_error(estimate, exact))
            )

        for maxiter in fae_maxiters:
            try:
                calc_q = QuantumCalculator(
                    n_players,
                    vfunc,
                    internal_qubits_num=n_l_quantum,
                    internal_multiplier=1,
                    extraction_mode="qae_fae",
                    options=QAEOptions(delta=fae_delta, maxiter=maxiter),
                )
            except Exception:
                continue

            estimate = calc_q.get_all()
            oracle_calls = calc_q.get_oracle_count(0)
            if oracle_calls is None:
                continue
            points["F-QAE"].append(
                (max(1, oracle_calls), _mean_relative_error(estimate, exact))
            )

        for method in ["Classical MC", "I-QAE", "ML-QAE", "F-QAE"]:
            for epsilon in epsilons:
                min_calls = _min_calls_reaching_epsilon(points[method], epsilon)
                source_type = "discrete"
                if min_calls is None:
                    min_calls = _fallback_calls_from_loglog_fit(points[method], epsilon)
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

        print(f"equal-error: {round_idx + 1}/{n_rounds}")

    summary = _aggregate_equal_error_calls(rows, n_rounds=n_rounds)
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def _compute_src_with_quantum_fallback(
    returns_sub: pd.DataFrame,
    *,
    internal_qubits_num: int,
) -> tuple[dict[str, float], str]:
    try:
        src = RiskAttributor(
            returns_sub,
            QuantumCalculator,
            mode="rs",
            internal_qubits_num=internal_qubits_num,
            internal_multiplier=1,
            extraction_mode="statevector",
        ).attribute()
        return {asset: float(value) for asset, value in src.items()}, "quantum_rs"
    except Exception:
        src = RiskAttributor(
            returns_sub,
            BinaryEnumerationCalculator,
            mode="es",
        ).attribute()
        return {asset: float(value) for asset, value in src.items()}, "classical_es"


def _select_hidden_risk_assets(
    *,
    low_assets: list[str],
    high_assets: list[str],
) -> tuple[list[str], str]:
    high_asset = "TSLA" if "TSLA" in high_assets else sorted(high_assets)[0]
    preferred_low = [
        "KO",
        "JNJ",
        "TLT",
        "IEF",
        "SHY",
        "GOVT",
        "DUK",
        "SO",
        "ED",
        "WM",
        "RSG",
        "PG",
        "PEP",
        "WMT",
    ]

    selected_low: list[str] = []
    for asset in preferred_low:
        if asset in low_assets and asset not in selected_low and asset != high_asset:
            selected_low.append(asset)

    for asset in sorted(low_assets):
        if asset == high_asset or asset in selected_low:
            continue
        selected_low.append(asset)
        if len(selected_low) >= 9:
            break

    if len(selected_low) < 9:
        raise ValueError(
            "Not enough low-volatility assets to build hidden-risk scenario"
        )

    return selected_low[:9] + [high_asset], high_asset


def _generate_empirical_scenario_snapshots(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    hidden_path = snapshot_dir / "empirical_hidden_risk.csv"
    if not _needs_refresh([hidden_path], force):
        print("Skip empirical scenario snapshots: files already exist")
        return

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, _, high_assets = buckets[0], buckets[1], buckets[2]

    hidden_assets, high_asset = _select_hidden_risk_assets(
        low_assets=low_assets,
        high_assets=high_assets,
    )
    hidden_returns = returns[hidden_assets]
    hidden_src, hidden_method = _compute_src_with_quantum_fallback(
        hidden_returns,
        internal_qubits_num=6,
    )
    total_hidden_src = float(sum(hidden_src.values()))

    hidden_rows: list[dict[str, float | str]] = []
    for asset in hidden_assets:
        capital_weight = 1.0 / len(hidden_assets)
        src = float(hidden_src[asset])
        src_share = src / total_hidden_src if abs(total_hidden_src) > 1e-12 else 0.0
        amplification = src_share / capital_weight if capital_weight > 0 else 0.0
        hidden_rows.append(
            {
                "asset": asset,
                "capital_weight": capital_weight,
                "src": src,
                "src_share": src_share,
                "amplification": amplification,
                "risk_tier": "High" if asset == high_asset else "Low",
                "attribution_method": hidden_method,
            }
        )

    pd.DataFrame(hidden_rows).sort_values("src_share", ascending=False).to_csv(
        hidden_path, index=False
    )
    print(f"Wrote {hidden_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate risk experiment snapshot data for plotting",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiments",
        action="append",
        choices=EXPERIMENT_CHOICES,
        help=(
            "run only selected experiment(s); repeat this flag to run multiple "
            "experiments (default: run all)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite snapshot files even if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_experiments = list(dict.fromkeys(args.experiments or EXPERIMENT_CHOICES))

    paths = get_artifact_paths()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.snapshot_dir.mkdir(parents=True, exist_ok=True)

    dp = DataPreparation(paths.cache_dir)
    prices = dp.download()
    returns = np.log(prices / prices.shift(1)).dropna()

    if "volatility" in selected_experiments:
        _generate_volatility_snapshots(
            returns=returns,
            snapshot_dir=paths.snapshot_dir,
            force=args.force,
        )
    if "restoration" in selected_experiments:
        _generate_restoration_snapshot(
            returns=returns,
            snapshot_dir=paths.snapshot_dir,
            force=args.force,
        )
    if "quantum_comparison" in selected_experiments:
        _generate_quantum_comparison_snapshots(
            returns=returns,
            snapshot_dir=paths.snapshot_dir,
            force=args.force,
        )
    if "equal_error" in selected_experiments:
        _generate_equal_error_snapshot(
            returns=returns,
            snapshot_dir=paths.snapshot_dir,
            force=args.force,
        )
    if "empirical_scenario" in selected_experiments:
        _generate_empirical_scenario_snapshots(
            returns=returns,
            snapshot_dir=paths.snapshot_dir,
            force=args.force,
        )

    manifest = write_manifest(
        output_dir=paths.snapshot_dir,
        params={
            "force": bool(args.force),
            "selected_experiments": selected_experiments,
            "windows": {"start": "2020-04-01", "end": "2022-04-01"},
            "quantum_comparison": {
                "n_rounds": 50,
                "asset_sizes": [3, 4, 5, 6],
                "qubit_range": [2, 3, 4, 5, 6],
                "iqae_options": {"epsilon": 0.01, "alpha": 0.01},
            },
            "equal_error": {
                "epsilons": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
                "n_rounds": 8,
                "classical_samples": [
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
                ],
                "iqae_epsilons": [0.05, 0.03, 0.02, 0.01, 0.007, 0.005],
                "iqae_alpha": 0.01,
                "mlqae_eval_qubits": [2, 3, 4, 5, 6],
                "fae_maxiters": [3, 4, 5, 6, 7],
                "fae_delta": 0.05,
            },
            "empirical_cases": {
                "hidden_risk": {
                    "portfolio_size": 10,
                    "target_high_asset": "TSLA",
                    "low_assets_count": 9,
                },
            },
        },
        source_data=str(dp.file_path),
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
