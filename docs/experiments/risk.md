# Risk Experiments

This document describes the five experiments in `experiments/risk/` that validate and characterise the risk attribution framework. For the underlying theory, see [docs/application/risk.md](../application/risk.md) and [docs/solver/shapley.md](../solver/shapley.md).

---

## Running the Experiments

```bash
# Generate all CSV snapshots
uv run python -m experiments.risk.generate_artifacts

# Generate a single experiment
uv run python -m experiments.risk.generate_artifacts -e volatility

# Force regeneration even if output files already exist
uv run python -m experiments.risk.generate_artifacts --force

# Render figures from snapshots
uv run python -m experiments.risk.plot_from_artifacts
```

Available experiment names: `volatility`, `restoration`, `quantum_comparison`, `equal_error`, `empirical_scenario`.

---

## Data

**Source:** Yahoo Finance historical adjusted close prices via `DataPreparation` (wraps `double_quant.data.time_series.from_yfinance`).

**Universe:** 89 tickers across three volatility tiers — 38 high-volatility (NVDA, TSLA, AMD, …), 50 mid-volatility (AAPL, MSFT, JPM, …), and 34 low-volatility (utilities, bonds, commodities such as TLT, AGG, GLD).

**Window:** 2020-04-01 to 2022-04-01 (two years covering COVID recovery and subsequent rate-rise cycle).

**Cache:** `experiments/risk/cache/experiment_data_clean.csv`. Downloaded once; subsequent runs read from cache.

---

## Experiment 1: Volatility Buckets (`volatility`)

**Question:** Do the three volatility tiers exhibit meaningfully different risk-return profiles over the sample window?

**Method:** `divide_by_volatility(returns, [0.3, 0.7])` partitions assets into bottom-30% (low), middle-40% (mid), and top-30% (high) by annualised volatility. Each bucket is equally-weighted and characterised by:

- Annualised volatility and return
- Maximum drawdown
- Cumulative return series
- 30-day rolling volatility

**Outputs:**
- `docs/assets/risk/data/vol_buckets_metrics.csv` — per-bucket summary statistics
- `docs/assets/risk/data/vol_buckets_series.csv` — time series (cumulative return, rolling vol, drawdown)
- `docs/assets/risk/vol_buckets_trend.png`

---

## Experiment 2: Restoration Accuracy (`restoration`)

**Question:** Does the recovery formula SRC_i = ES({i}) − Φ_i^RS reproduce the same result as computing SRC_i = Φ_i^ES directly?

**Method:** A 5-asset portfolio (2 high-vol, 2 mid-vol, 1 low-vol, seed=0) is run through `RiskAttributor` in both `mode="es"` and `mode="rs"` using `BinaryEnumerationCalculator`. The mean absolute error (MAE) between the two sets of SRC values is reported. The recovery theorem guarantees MAE = 0 to floating-point precision; any deviation indicates an implementation bug.

**Outputs:**
- `docs/assets/risk/data/restoration_accuracy.csv` — per-asset SRC values, absolute difference, MAE
- `docs/assets/risk/restoration_accuracy_bar.png`

---

## Experiment 3: Quantum Comparison (`quantum_comparison`)

**Question:** How close are quantum Shapley estimates to the exact classical baseline, and how does accuracy vary with portfolio size (n) and internal qubit count (n_l)?

**Method:** For portfolio sizes n ∈ {3, 4, 5, 6} and internal qubit counts n_l ∈ {2, 3, 4, 5, 6}, 50 randomly sampled portfolios are evaluated across six extraction methods:

| Method | Configuration |
|--------|---------------|
| Statevector | exact simulation (baseline) |
| Shots | 1 024 shots |
| Shots | 4 096 shots |
| I-QAE | ε=0.01, α=0.01 |
| ML-QAE | 4 evaluation qubits |
| F-QAE | δ=0.05, maxiter=5 |

Mean relative error (MRE) against the classical `BinaryEnumerationCalculator` (`mode="es"`) is averaged across 50 rounds.

**Outputs:**
- `docs/assets/risk/data/quantum_comparison_n{3,4,5,6}.csv` — (n_l, method) → mean relative error
- `docs/assets/risk/quantum_comparison_grid.png` — 4-panel grid (one panel per n)

---

## Experiment 4: Equal-Error Oracle Calls (`equal_error`)

**Question:** At a fixed target accuracy ε, how many oracle calls does each method require — and how does quantum QAE compare to classical Monte Carlo?

**Method:** For 8 random 5-asset portfolios (2 high, 2 mid, 1 low), each method is run at multiple parameter settings to trace an (oracle_calls, error) curve:

- **Classical MC:** sample counts {10, 20, 40, 80, …, 20 000}
- **I-QAE:** ε ∈ {0.05, 0.03, 0.02, 0.01, 0.007, 0.005}
- **ML-QAE:** evaluation qubits ∈ {2, 3, 4, 5, 6}
- **F-QAE:** maxiter ∈ {3, 4, 5, 6, 7}

For each target accuracy ε ∈ {10⁻³, 2×10⁻³, 5×10⁻³, 10⁻², 2×10⁻², 5×10⁻², 10⁻¹}, the minimum oracle count reaching that ε is extracted from the curve (with log-log extrapolation as fallback).

**Outputs:**
- `docs/assets/risk/data/equal_error_oracle_calls_summary.csv` — (method, ε) → mean calls, std, reachable ratio, source type
- `docs/assets/risk/equal_error_oracle_calls_fixed_grid_fallback.png` — log-scale oracle call comparison

---

## Experiment 5: Empirical Scenario (`empirical_scenario`)

**Question:** Can SRC reveal hidden concentration risk that capital weights conceal?

**Method:** A 10-asset portfolio is constructed from 9 low-volatility assets plus one high-volatility asset (TSLA if available, otherwise the most volatile in the universe). All assets receive equal capital weight (10%). SRC is computed via quantum attribution (`mode="rs"`, statevector, with classical fallback). The experiment reports SRC share vs capital weight and the resulting amplification ratio SRC_share / capital_weight for each asset.

**Outputs:**
- `docs/assets/risk/data/empirical_hidden_risk.csv` — per-asset capital weight, SRC, SRC share, amplification, risk tier, attribution method
- `docs/assets/risk/empirical_hidden_risk.png`

---

## File Map

```
experiments/risk/
  artifacts.py            DataPreparation, ArtifactPaths, path helpers
  generate_artifacts.py   CLI entry point — runs experiments, writes CSVs
  plot_from_artifacts.py  CLI entry point — reads CSVs, writes PNGs
  cache/                  Downloaded price data (git-ignored)

docs/assets/risk/
  data/                   Generated CSV snapshots (not committed)
  *.png                   Generated figures
```
