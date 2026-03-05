# Double Quant Context

## Commands

```bash
# Install dependencies
uv sync

# Run all tests (only when user requests)
uv run pytest -s -v

# Run a single test file
uv run pytest tests/double_quant/algorithm/shapley/test_calculator.py -s -v

# Run with coverage
uv run pytest --cov=double_quant -s -v

# Add a runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --dev <package>

# Build
uv build

# Run experiment artifact generation
uv run python -m experiments.risk.generate_artifacts

# Plot from experiment artifacts
uv run python -m experiments.risk.plot_from_artifacts
```

**Do not run tests unless the user explicitly requests it.**

## Project Overview

**Double Quant** is a high-performance quantum computing framework for quantitative finance — bridging quantum algorithms (HHL, QAE) with financial applications (portfolio risk attribution) using a layered architecture.

**Key Technologies:**
- **Language:** Python 3.11+
- **Quantum Stack:** Qiskit, Qiskit Aer, qiskit-algorithms
- **Math/Data:** Scipy, NumPy, Pandas, yfinance
- **Package Manager:** `uv`
- **Build System:** `hatchling`

## Architecture

Four-layer architecture (`src/double_quant/`):

```
Data Layer         →  double_quant.data            (sources, transforms)
Common Layer       →  double_quant.common          (LinearSystem, metrics, utils)
Algorithm Layer    →  double_quant.algorithm        (HHL, Shapley calculators)
Application Layer  →  double_quant.application     (risk attribution)
```

### Key modules

**`double_quant.data.source`**
- `PriceSource` (protocol): Interface for price data providers. Implementations return a DataFrame with `DatetimeIndex`, columns=tickers, values=close prices.
- `YFinanceSource`: Yahoo Finance data source with optional CSV caching. `YFinanceSource(cache_path=...).fetch(tickers, start, end)`.

**`double_quant.data.transform`**
- `to_log_returns(prices)`: Close prices to log returns DataFrame.
- `to_covariance(prices)`: Close prices to covariance matrix of log returns.
- `to_expected_returns(prices)`: Close prices to mean log return vector.

**`double_quant.common`**
- `LinearSystem`: Core model for `Ax = b`. Handles scaling for quantum algorithms. Use `LinearSystem.random_for_hhl(n)` to generate test systems. Matrix must be symmetric for HHL.
- `metric`: `expected_shortfall(returns, alpha)`, `cos_similarity(x, y)`, `annualized_volatility`
- `util`: `normalize`

**`double_quant.algorithm.hhl.sapo`**
- `SAPO`: Scales the linear system by `0.5 / max_eigenvalue` and computes QPE qubit count via `get_qpe_qubit_num()`. Used internally by `HHLSolver`.

**`double_quant.algorithm.hhl`**
- `HHLSolver.solve(matrix, vector)`: Static method solving `Ax = b` via the HHL quantum algorithm with SAPO optimization. Uses statevector simulation.

**`double_quant.algorithm.shapley`**
Split into three sub-modules (`protocol.py`, `calculator.py`, `quantum.py`):
- `ValueFunction` (protocol, in `protocol.py`): Any object with `__getitem__(bitmask: int) -> float`.
- `ExtractionMode`, `QAEOptions` (in `protocol.py`): Types for quantum extraction configuration.
- `ShapleyCalculator` (base class, in `calculator.py`): Subclass and implement `_calculate_one(player)`. All subclasses accept a `ValueFunction`.
- `BinaryEnumerationCalculator` (in `calculator.py`): Exact, O(n · 2^n) classical.
- `PermutationEnumerationCalculator` (in `calculator.py`): Exact via permutation enumeration.
- `PermutationMCCalculator` (in `calculator.py`): Monte Carlo approximation.
- `QuantumCalculator` (in `quantum.py`): Quantum algorithm using `IntervalLoader` + `VertexRotator` + `ValueLoader` circuits. Supports 6 extraction modes: `"statevector"` (default, exact), `"shots"`, `"qae_canonical"`, `"qae_iqae"`, `"qae_mlqae"`, `"qae_fae"`. **Requires superadditive value function.**

**`double_quant.application.risk`**
- `RiskAttributor(returns_df, solver_class, mode)`: Orchestrates risk attribution. Imports `ShapleyCalculator` from `double_quant.algorithm.shapley`.
  - `mode="rs"` (default, quantum-compatible): Uses `RiskSavingValueFunction`; `SRC_i = ES({i}) − Φ_i^RS`. RS is superadditive → works with `QuantumCalculator`.
  - `mode="es"` (classical only): Uses `ExpectedShortfallValueFunction` directly; ES is subadditive → **incompatible with `QuantumCalculator`**.
  - Both modes produce mathematically identical SRC results.

### Critical constraint: superadditivity

`QuantumCalculator` encodes marginal contributions as rotation angles and asserts `value_in >= value_out` (non-negative marginal contributions). Passing a subadditive function (like raw ES) causes assertion errors. Always use `RiskSavingValueFunction` with `QuantumCalculator` for risk attribution.

## Docs

`docs/application/risk.md` and `docs/solver/shapley.md` contain the mathematical theory behind each module. Consult these before modifying or extending any algorithm or application module.

## Conventions

- **Commit messages**: Angular convention — `<type>(<scope>): <description>`. Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
- **Type hints**: Full type hints throughout; `py.typed` marker present.
- **Linting**: `ruff`.
- **Library docs**: Always use Context7 MCP for Qiskit / scipy / other library documentation without being asked.
- **Help docs**: The plans for coding and some references must be create in `.help/`
