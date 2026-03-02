# Decouple Tests from Experiments & Merge Scripts

**Date:** 2026-03-02
**Status:** Approved

## Problem

1. `scripts/` and `experiments/` both contain experiment-related code in separate directories.
2. `tests/double_quant/application/test_risk.py` imports `DataPreparation` from `experiments/risk/artifacts.py`, coupling tests to experiments.
3. Experiment cache (`experiment_data_clean.csv`) lives under `tests/`, blurring ownership.

## Goals

- Tests and experiments are independent — both call `double_quant` APIs directly, no cross-imports.
- All experiment code lives under `experiments/`.
- Each domain owns its own cache directory.
- `scripts/` directory is removed.

## Design

### 1. Directory Structure

**Before:**
```
experiments/risk/
  __init__.py
  artifacts.py
scripts/risk/
  generate_artifacts.py
  plot_from_artifacts.py
tests/double_quant/application/
  cache/experiment_data_clean.csv  # experiment data, misplaced in tests/
  test_risk.py                     # imports experiments.risk.artifacts
```

**After:**
```
experiments/risk/
  __init__.py
  artifacts.py                     # DataPreparation default cache → experiments/risk/cache/
  cache/experiment_data_clean.csv  # experiment cache, moved here
  generate_artifacts.py            # moved from scripts/
  plot_from_artifacts.py           # moved from scripts/
tests/double_quant/application/
  cache/test_data.csv              # test-only cache (10 tickers)
  conftest.py                      # new: test-specific data fixtures
  test_risk.py                     # no experiments imports
```

### 2. Cache Separation

- **Experiment cache**: `experiments/risk/cache/experiment_data_clean.csv` (89 tickers).
  - `DataPreparation.__init__` default `data_dir` changes from `tests/double_quant/application/cache` to `experiments/risk/cache`.
- **Test cache**: `tests/double_quant/application/cache/test_data.csv` (10 tickers).
  - Old `experiment_data_clean.csv` removed from `tests/` cache directory.
- Both cache directories stay in `.gitignore`.

### 3. Test Data Strategy

New `tests/double_quant/application/conftest.py`:
- Define 10 tickers covering high/mid/low volatility: TSLA, NVDA, AAPL, MSFT, META, JPM, TLT, GLD, ED, AGG.
- `prices` fixture (`scope="session"`): calls `from_yfinance()` directly with test-specific cache.
- `returns` fixture (`scope="session"`): log returns from prices.

### 4. test_risk.py Changes

- Remove `from experiments.risk.artifacts import DataPreparation`.
- All test functions accept `prices` / `returns` fixtures instead of creating `DataPreparation` instances.
- `test_data_download`: tests `from_yfinance` API directly (returns non-empty DataFrame).
- `test_volatility_bucketing`: column count assertion changes from `> 50` to `== 10`.
- `TestRiskSaving` / `TestQuantumSolver`: data source changes to fixtures; `rng.choice` selects from 10 assets instead of ~89. Test logic unchanged.

### 5. Scripts Migration

- `scripts/risk/generate_artifacts.py` → `experiments/risk/generate_artifacts.py`
- `scripts/risk/plot_from_artifacts.py` → `experiments/risk/plot_from_artifacts.py`
- Internal imports (`from experiments.risk.artifacts import ...`) remain unchanged.
- Delete empty `scripts/` directory.

### 6. No Changes

- `experiments/risk/artifacts.py`: `DataPreparation` stays as experiment configuration (only default path changes).
- `double_quant.data.time_series.from_yfinance`: already the library API, no changes needed.
- Build configuration: `experiments/` is not a distributed package.
