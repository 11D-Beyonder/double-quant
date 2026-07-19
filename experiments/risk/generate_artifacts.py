from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.common.util import divide_by_volatility
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumShapleyCalculator,
)
from experiments.risk.artifacts import (
    DataPreparation,
    get_artifact_paths,
    write_manifest,
)


EXPERIMENT_CHOICES = (
    "volatility",
    "restoration",
    "quantum_comparison",
    "qae_comparison",
    "equal_error",
    "equal_error_scaling",
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
                            QuantumShapleyCalculator,
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


def _relative_improvement(best_value: float, worst_value: float) -> float:
    if worst_value <= 0:
        return 0.0
    return float((worst_value - best_value) / worst_value)


def _print_qae_comparison_assessment(assessment: pd.DataFrame) -> None:
    for row in assessment.itertuples(index=False):
        passed_value = row.passes_threshold
        passed = (
            passed_value
            if isinstance(passed_value, bool)
            else str(passed_value).lower() == "true"
        )
        print(
            f"QAE comparison {row.metric}: {float(row.relative_gap):.2%} "
            f"(best={row.best_method}, worst={row.worst_method}) - "
            f"{'PASS' if passed else 'FAIL'} "
            f"(requires >= {float(row.required_gap):.0%})"
        )

    overall_passed = all(
        value if isinstance(value, bool) else str(value).lower() == "true"
        for value in assessment["passes_threshold"].tolist()
    )
    print(f"QAE comparison result: {'PASS' if overall_passed else 'FAIL'}")


def _generate_qae_comparison_snapshots(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    runs_path = snapshot_dir / "qae_comparison_runs.csv"
    summary_path = snapshot_dir / "qae_comparison_summary.csv"
    assessment_path = snapshot_dir / "qae_comparison_assessment.csv"
    target_paths = [runs_path, summary_path, assessment_path]
    if not _needs_refresh(target_paths, force):
        print("Skip QAE comparison snapshots: files already exist")
        _print_qae_comparison_assessment(pd.read_csv(assessment_path))
        return

    n_rounds = 8
    n_players = 5
    n_l_quantum = 6
    accuracy_gap_threshold = 0.40
    circuit_gap_threshold = 0.50
    methods: list[
        tuple[
            Literal["qae_iqae", "qae_mlqae", "qae_fae"],
            QAEOptions,
            str,
        ]
    ] = [
        ("qae_iqae", QAEOptions(epsilon=0.01, alpha=0.01), "I-QAE"),
        ("qae_mlqae", QAEOptions(num_eval_qubits=4), "ML-QAE"),
        ("qae_fae", QAEOptions(delta=0.05, maxiter=5), "F-QAE"),
    ]

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
    rng = np.random.default_rng(seed=2032)

    rows: list[dict[str, object]] = []
    for round_idx in range(n_rounds):
        sampled = (
            rng.choice(high_assets, size=2, replace=False).tolist()
            + rng.choice(mid_assets, size=2, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        ret_sub = returns.loc[:, sampled]
        exact_src_by_asset = RiskAttributor(
            ret_sub,
            BinaryEnumerationCalculator,
            mode="es",
        ).attribute()
        exact_src = [float(exact_src_by_asset[asset]) for asset in sampled]
        value_function = RiskSavingValueFunction(ret_sub)

        for extraction_mode, options, method in methods:
            calculator = QuantumShapleyCalculator(
                n_players,
                value_function,
                internal_qubits_num=n_l_quantum,
                internal_multiplier=1,
                extraction_mode=extraction_mode,
                options=options,
            )
            risk_saving_shapley = calculator.get_all()
            estimated_src = [
                float(value_function.individual_es[asset] - risk_saving_shapley[i])
                for i, asset in enumerate(sampled)
            ]
            total_oracle_calls = _total_oracle_count(calculator, n_players)
            if total_oracle_calls is None:
                raise RuntimeError(f"{method} did not report oracle-call counts")

            rows.append(
                {
                    "round": round_idx,
                    "method": method,
                    "assets": ",".join(sampled),
                    "n": n_players,
                    "n_l": n_l_quantum,
                    "mean_relative_error": _mean_relative_error(
                        estimated_src, exact_src
                    ),
                    "total_oracle_calls": total_oracle_calls,
                }
            )

        print(f"QAE comparison: {round_idx + 1}/{n_rounds}")

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("method", as_index=False)
        .agg(
            mean_relative_error=("mean_relative_error", "mean"),
            std_relative_error=("mean_relative_error", "std"),
            mean_oracle_calls=("total_oracle_calls", "mean"),
            std_oracle_calls=("total_oracle_calls", "std"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )

    accuracy_best = summary.loc[summary["mean_relative_error"].idxmin()]
    accuracy_worst = summary.loc[summary["mean_relative_error"].idxmax()]
    circuit_best = summary.loc[summary["mean_oracle_calls"].idxmin()]
    circuit_worst = summary.loc[summary["mean_oracle_calls"].idxmax()]
    accuracy_gap = _relative_improvement(
        float(accuracy_best["mean_relative_error"]),
        float(accuracy_worst["mean_relative_error"]),
    )
    circuit_gap = _relative_improvement(
        float(circuit_best["mean_oracle_calls"]),
        float(circuit_worst["mean_oracle_calls"]),
    )
    assessment = pd.DataFrame(
        [
            {
                "metric": "accuracy_gap",
                "best_method": accuracy_best["method"],
                "worst_method": accuracy_worst["method"],
                "best_value": accuracy_best["mean_relative_error"],
                "worst_value": accuracy_worst["mean_relative_error"],
                "relative_gap": accuracy_gap,
                "required_gap": accuracy_gap_threshold,
                "passes_threshold": accuracy_gap >= accuracy_gap_threshold,
            },
            {
                "metric": "circuit_gap",
                "best_method": circuit_best["method"],
                "worst_method": circuit_worst["method"],
                "best_value": circuit_best["mean_oracle_calls"],
                "worst_value": circuit_worst["mean_oracle_calls"],
                "relative_gap": circuit_gap,
                "required_gap": circuit_gap_threshold,
                "passes_threshold": circuit_gap >= circuit_gap_threshold,
            },
        ]
    )

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    assessment.to_csv(assessment_path, index=False)
    print(f"Wrote {runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {assessment_path}")
    _print_qae_comparison_assessment(assessment)


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


def _total_oracle_count(
    calculator: PermutationMCCalculator | QuantumShapleyCalculator,
    n_players: int,
) -> int | None:
    total = 0
    for player_index in range(n_players):
        count = calculator.get_oracle_count(player_index)
        if count is None:
            return None
        total += max(1, int(count))
    return total


def _aggregate_equal_error_calls(
    rows: list[dict[str, object]],
    n_rounds: int,
    group_columns: list[str] | None = None,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    if group_columns is None:
        group_columns = ["method", "epsilon"]
    if sort_columns is None:
        sort_columns = ["epsilon", "method"]

    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(group_columns):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        summary_row: dict[str, object] = dict(
            zip(group_columns, key_values, strict=True)
        )

        valid = group[group["oracle_calls"].notna()]
        reachable = len(valid)
        reachable_ratio = float(reachable / n_rounds)

        if reachable > 0:
            calls = [float(value) for value in valid["oracle_calls"].tolist()]
            mean_calls = float(np.mean(calls))
            std_calls = float(np.std(calls))
            source_values = [str(value) for value in valid["source_type"].tolist()]
            unique_sources = sorted(set(source_values))
            source_type = unique_sources[0] if len(unique_sources) == 1 else "mixed"
        else:
            mean_calls = float("nan")
            std_calls = float("nan")
            source_type = "none"

        summary_row.update(
            {
                "mean_calls": mean_calls,
                "std_calls": std_calls,
                "reachable_ratio": reachable_ratio,
                "source_type": source_type,
            }
        )
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    return summary.sort_values(sort_columns).reset_index(drop=True)


def _fit_perf_2_speedup(summary: pd.DataFrame) -> dict[str, object]:
    paired_calls = summary.pivot(
        index="epsilon",
        columns="method",
        values="mean_calls",
    )
    required_methods = ["Classical MC", "I-QAE"]
    missing_methods = [
        method for method in required_methods if method not in paired_calls.columns
    ]
    if missing_methods:
        raise ValueError(
            "Cannot fit Perf-2 speedup without methods: "
            + ", ".join(missing_methods)
        )

    paired_calls = paired_calls[required_methods].replace([np.inf, -np.inf], np.nan)
    paired_calls = paired_calls.dropna()
    paired_calls = paired_calls[
        (paired_calls["Classical MC"] > 0) & (paired_calls["I-QAE"] > 0)
    ]
    if len(paired_calls) < 2:
        raise ValueError("Perf-2 speedup fit requires at least two paired points")

    log_classical_calls = np.log(paired_calls["Classical MC"].to_numpy())
    log_quantum_calls = np.log(paired_calls["I-QAE"].to_numpy())
    denominator = float(np.dot(log_classical_calls, log_classical_calls))
    if denominator <= 0:
        raise ValueError("Perf-2 speedup fit requires non-unit classical calls")

    exponent = float(
        np.dot(log_classical_calls, log_quantum_calls) / denominator
    )
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("Perf-2 speedup fit produced a non-positive exponent")

    acceleration_order = float(1.0 / exponent)
    residuals = log_quantum_calls - exponent * log_classical_calls
    log_rmse = float(np.sqrt(np.mean(np.square(residuals))))
    quantum_log_energy = float(np.dot(log_quantum_calls, log_quantum_calls))
    uncentered_r_squared = (
        float(1.0 - np.dot(residuals, residuals) / quantum_log_energy)
        if quantum_log_energy > 0
        else 1.0
    )

    return {
        "classical_method": "Classical MC",
        "quantum_method": "I-QAE",
        "model": "n_q = n_c^(1/x)",
        "fit_points": len(paired_calls),
        "exponent_1_over_x": exponent,
        "acceleration_order_x": acceleration_order,
        "log_rmse": log_rmse,
        "uncentered_r_squared": uncentered_r_squared,
        "passes_perf_2": acceleration_order > 1.0,
    }


def _print_perf_2_speedup(fit: dict[str, object]) -> None:
    exponent = float(str(fit["exponent_1_over_x"]))
    acceleration_order = float(str(fit["acceleration_order_x"]))
    passed = bool(fit["passes_perf_2"])
    print(
        "Perf-2 fit: n_q = n_c^(1/x), "
        f"x = {acceleration_order:.6f} "
        f"(1/x = {exponent:.6f})"
    )
    print(f"Perf-2 result: {'PASS' if passed else 'FAIL'} (requires x > 1)")


def _generate_equal_error_snapshot(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    out_path = snapshot_dir / "equal_error_oracle_calls_summary.csv"
    fit_path = snapshot_dir / "equal_error_perf_2_fit.csv"
    if not _needs_refresh([out_path, fit_path], force):
        print("Skip equal-error summary: file already exists")
        cached_fit = pd.read_csv(fit_path).iloc[0].to_dict()
        cached_fit["passes_perf_2"] = bool(cached_fit["passes_perf_2"])
        _print_perf_2_speedup(cached_fit)
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
                calc_q = QuantumShapleyCalculator(
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
                calc_q = QuantumShapleyCalculator(
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
                calc_q = QuantumShapleyCalculator(
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

    perf_2_fit = _fit_perf_2_speedup(summary)
    pd.DataFrame([perf_2_fit]).to_csv(fit_path, index=False)
    print(f"Wrote {fit_path}")
    _print_perf_2_speedup(perf_2_fit)


def _generate_equal_error_scaling_snapshot(
    *, returns: pd.DataFrame, snapshot_dir: Path, force: bool
) -> None:
    out_path = snapshot_dir / "equal_error_scaling_summary.csv"
    if not _needs_refresh([out_path], force):
        print("Skip equal-error scaling summary: file already exists")
        return

    target_epsilons = [5e-2]
    n_rounds = 8
    asset_sizes = [3, 4, 5, 6]
    bucket_scheme = {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1), 6: (2, 2, 2)}
    n_l_quantum = 2
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
    iqae_epsilons = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005]
    iqae_alpha = 0.01

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
    rng = np.random.default_rng(seed=2031)

    rows: list[dict[str, object]] = []
    for n_players in asset_sizes:
        n_high, n_mid, n_low = bucket_scheme[n_players]

        for round_idx in range(n_rounds):
            sampled = (
                rng.choice(high_assets, size=n_high, replace=False).tolist()
                + rng.choice(mid_assets, size=n_mid, replace=False).tolist()
                + rng.choice(low_assets, size=n_low, replace=False).tolist()
            )
            ret_sub = returns.loc[:, sampled]
            vfunc = RiskSavingValueFunction(ret_sub)
            exact = BinaryEnumerationCalculator(n_players, vfunc).get_all()

            points: dict[str, list[tuple[int, float]]] = {
                "Classical MC": [],
                "I-QAE": [],
            }
            for sample_count in classical_samples:
                calc_mc = PermutationMCCalculator(
                    n_players,
                    vfunc,
                    num_samples=sample_count,
                    seed=n_players * 100000 + round_idx * 1000 + sample_count,
                )
                estimate = calc_mc.get_all()
                total_calls = _total_oracle_count(calc_mc, n_players)
                if total_calls is None:
                    continue
                points["Classical MC"].append(
                    (total_calls, _mean_relative_error(estimate, exact))
                )

            for iqae_epsilon in iqae_epsilons:
                try:
                    calc_q = QuantumShapleyCalculator(
                        n_players,
                        vfunc,
                        internal_qubits_num=n_l_quantum,
                        internal_multiplier=1,
                        extraction_mode="qae_iqae",
                        options=QAEOptions(epsilon=iqae_epsilon, alpha=iqae_alpha),
                    )
                except Exception:
                    continue

                estimate = calc_q.get_all()
                total_calls = _total_oracle_count(calc_q, n_players)
                if total_calls is None:
                    continue
                points["I-QAE"].append(
                    (total_calls, _mean_relative_error(estimate, exact))
                )

            for method in ["Classical MC", "I-QAE"]:
                for epsilon in target_epsilons:
                    min_calls = _min_calls_reaching_epsilon(points[method], epsilon)
                    source_type = "discrete"
                    if min_calls is None:
                        min_calls = _fallback_calls_from_loglog_fit(
                            points[method], epsilon
                        )
                        source_type = "fallback" if min_calls is not None else "none"

                    rows.append(
                        {
                            "n": n_players,
                            "round": round_idx,
                            "method": method,
                            "epsilon": epsilon,
                            "oracle_calls": min_calls,
                            "source_type": source_type,
                        }
                    )

            print(
                f"equal-error scaling n={n_players}: {round_idx + 1}/{n_rounds}"
            )

    summary = _aggregate_equal_error_calls(
        rows,
        n_rounds=n_rounds,
        group_columns=["n", "epsilon", "method"],
        sort_columns=["n", "epsilon", "method"],
    )
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
            QuantumShapleyCalculator,
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
    if "qae_comparison" in selected_experiments:
        _generate_qae_comparison_snapshots(
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
    if "equal_error_scaling" in selected_experiments:
        _generate_equal_error_scaling_snapshot(
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
            "qae_comparison": {
                "n_rounds": 8,
                "n_players": 5,
                "n_l_quantum": 6,
                "methods": {
                    "I-QAE": {"epsilon": 0.01, "alpha": 0.01},
                    "ML-QAE": {"num_eval_qubits": 4},
                    "F-QAE": {"delta": 0.05, "maxiter": 5},
                },
                "accuracy_gap_threshold": 0.40,
                "circuit_gap_threshold": 0.50,
                "gap_formula": "(worst - best) / worst",
                "circuit_metric": "total_oracle_calls_for_all_players",
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
                "perf_2_fit_model": "n_q = n_c^(1/x)",
                "perf_2_pass_condition": "x > 1",
            },
            "equal_error_scaling": {
                "target_epsilons": [5e-2],
                "n_rounds": 8,
                "asset_sizes": [3, 4, 5, 6],
                "n_l_quantum": 6,
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
                "call_scope": "total_calls_for_all_players",
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
