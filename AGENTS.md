# Double Quant Context

## Source Of Truth

- Prefer `src/double_quant/**` and active test files over `README.md` when they disagree.
- `README.md` still contains stale API names such as `QuantumLinearSolver` and `double_quant.solver`.
- `docs/application/risk.md` is aligned with the current risk-attribution design.
- `docs/solver/shapley.md` is useful for theory, but some import-path examples are stale, so cross-check with `src/`.
- `docs/experiments/risk.md` is the best reference for experiment workflows, artifact names, and output locations.

## Commands

```bash
# Install dependencies
uv sync

# Build the package
uv build

# Lint
uv run ruff check .

# Type check
uv run basedpyright

# Run all tests (only when user requests)
uv run pytest

# Run a single test file
uv run pytest tests/double_quant/algorithm/shapley/test_calculator.py
uv run pytest tests/double_quant/algorithm/hhl/test_solver.py
uv run pytest tests/double_quant/application/test_risk.py

# Run a single test
uv run pytest -k "<test_function_name>"

# Run with coverage
uv run pytest --cov=double_quant

# Add a runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --dev <package>

# Run experiment artifact generation
uv run python -m experiments.risk.generate_artifacts

# Run a single experiment
uv run python -m experiments.risk.generate_artifacts -e volatility

# Force regeneration of experiment artifacts
uv run python -m experiments.risk.generate_artifacts --force

# Plot from experiment artifacts
uv run python -m experiments.risk.plot_from_artifacts
```

**Do not run tests unless the user explicitly requests it.**

**Some tests and experiment workflows may hit the network through `yfinance` if cache files are missing.**

## Project Overview

**Double Quant** is a Python 3.11+ framework for quantitative finance experiments built around quantum-inspired and quantum-backed workflows. The live codebase is organized as a four-part package:

- `double_quant.data`
- `double_quant.common`
- `double_quant.algorithm`
- `double_quant.application`

There is also a real experiment pipeline under `experiments/risk/` that generates CSV artifacts and plots into `docs/assets/risk/`.

**Key technologies:**

- **Language:** Python 3.11+
- **Quantum stack:** Qiskit, Qiskit Aer, qiskit-algorithms
- **Math/data:** NumPy, Pandas, SciPy, matplotlib, seaborn, yfinance
- **Package manager:** `uv`
- **Build system:** `hatchling`
- **Lint/type check:** `ruff`, `basedpyright`

## Architecture

```
Data Layer         -> double_quant.data
Common Layer       -> double_quant.common
Algorithm Layer    -> double_quant.algorithm
Application Layer  -> double_quant.application
Experiments        -> experiments.risk
```

### Package map

**`double_quant.data`**
- `source.py`: price-source protocol and Yahoo Finance implementation
- `transform.py`: close-price to return / covariance / expected-return helpers

**`double_quant.common`**
- `model.py`: `LinearSystem`
- `metric.py`: `expected_shortfall`, `cos_similarity`, `annualized_volatility`
- `util.py`: `normalize`, `divide_by_volatility`, warning helpers

**`double_quant.algorithm.hhl`**
- `solver.py`: `HHLSolver`
- `variants.py`: HHL transform strategies, including the current SAPO-style path

**`double_quant.algorithm.shapley`**
- `protocol.py`: `ValueFunction`, `ExtractionMode`, `QAEOptions`
- `calculator.py`: classical exact and Monte Carlo Shapley solvers
- `quantum.py`: quantum Shapley circuit builders and `QuantumShapleyCaculator`

**`double_quant.application`**
- `risk.py`: expected-shortfall risk attribution
- `portfolio.py`: HHL-backed portfolio optimizer

**`experiments.risk`**
- `artifacts.py`: data preparation and artifact-path helpers
- `generate_artifacts.py`: writes experiment CSV snapshots
- `plot_from_artifacts.py`: renders PNG figures from snapshots

### Key modules

**`double_quant.data.source`**
- `PriceSource` is a protocol returning a `DataFrame` with `DatetimeIndex`, `columns=tickers`, and close-price values.
- `YFinanceSource(cache_path=...).fetch(tickers, start, end)` optionally reads from or writes to a CSV cache.
- Downloaded data is cleaned by dropping columns with too many missing values, forward-filling, and removing remaining NaNs.

**`double_quant.data.transform`**
- `to_log_returns(prices)`: close prices to log-return `DataFrame`
- `to_covariance(prices)`: close prices to covariance matrix
- `to_expected_returns(prices)`: close prices to mean log-return vector

**`double_quant.common.model`**
- `LinearSystem` is the core `Ax = b` container used around HHL workflows.
- `LinearSystem.random_for_hhl(n)` creates a symmetric random system for tests and experiments.
- Non-symmetric matrices only trigger a warning; they are not blocked automatically.

**`double_quant.algorithm.hhl`**
- `HHLSolver.solve(matrix, vector, transform_strategy="sapo")` is the main entry point.
- There is no live `double_quant.algorithm.hhl.sapo` source module in this repo.
- The current `"sapo"` behavior is resolved inside `HHLSolver` via strategy code in `algorithm/hhl/variants.py`.
- Current HHL extraction uses statevector simulation and expects symmetric or Hermitian inputs.

**`double_quant.algorithm.shapley`**
- `ShapleyCalculator` is the base class.
- `BinaryEnumerationCalculator` is the exact subset-enumeration baseline.
- `PermutationEnumerationCalculator` is the exact permutation baseline.
- `PermutationMCCalculator` is the Monte Carlo approximation.
- `QuantumShapleyCaculator` is the actual public quantum solver class name in source, including the spelling typo. Keep that in mind when editing imports or public APIs.
- Supported extraction modes are `"statevector"`, `"shots"`, `"qae_canonical"`, `"qae_iqae"`, `"qae_mlqae"`, and `"qae_fae"`.
- Oracle-call counts are exposed through `get_oracle_count(player_index)`.
- The quantum solver requires a **superadditive** value function.

**`double_quant.application.risk`**
- `RiskAttributor(returns_df, solver_class, alpha=0.95, mode=...)` orchestrates Shapley-based expected-shortfall attribution.
- `mode="rs"` is the default and the only quantum-compatible route.
- `mode="es"` uses expected shortfall directly and is classical-only.
- `ExpectedShortfallValueFunction` and `RiskSavingValueFunction` both live in this module.

**`double_quant.application.portfolio`**
- `PortfolioOptimizer` solves a constrained portfolio system using `HHLSolver` by default.
- It expands systems to a power-of-two dimension before calling HHL.
- `ConstraintScaler` exists but is incomplete: `from_pickle()` and scaled execution are not implemented.
- Do not assume portfolio optimization is production-complete without reading this file first.

## Critical Constraints

### Superadditivity requirement

`QuantumShapleyCaculator` encodes marginal contributions as rotation angles and asserts non-negative increments. Passing a subadditive value function such as raw ES will fail or produce invalid results.

For risk attribution:

- Use `RiskSavingValueFunction` with `mode="rs"` for quantum workflows.
- Use `mode="es"` only with classical Shapley solvers.
- Both paths should recover the same SRC values mathematically.

### Stale docs and API names

- Do not copy import paths from `README.md` without checking the real package first.
- The live public package exports `QuantumShapleyCaculator`, not `QuantumCalculator`.
- The live code uses `double_quant.algorithm.*`, not `double_quant.solver.*`.

### Generated and cached files

- The repo contains checked-in generated noise such as `__pycache__` and `.DS_Store`; ignore these when mapping the codebase.
- Test fixtures use cached market data in `tests/double_quant/application/cache/test_data.csv`.
- Risk experiments use cached data under `experiments/risk/cache/`.
- Generated experiment outputs live under `docs/assets/risk/` and `docs/assets/risk/data/`.

## Docs

- Read `docs/application/risk.md` before changing risk attribution logic.
- Read `docs/solver/shapley.md` before changing Shapley algorithms, but verify import names against `src/`.
- Read `docs/experiments/risk.md` before changing experiment generation or plotting code.


## Import Things

All Agents, please follow the instructions below for the final project acceptance. To ensure the repository
is ready for review, you must complete the required table `tests/docs/3rd-testtable.xlsx` after finish a function implementation or a performance benchmarking. all output and figure must use chinese, and the 

1. Functional Testing Requirements
Every functional test must be submitted as a single standalone file in .py format under `tests/`. Please use the following naming convention:
     (Function Number) - (Function English Name).py
for example,  `tests/double_quant/circuit/75-circuit_repair.py`
2. Documentation and Output
All output documentation files should be organized within `tests/docs` with chinese language, using the same numbering and name in table `tests/docs/3rd-testtable.xlsx` .
for example, `tests/docs/76-circuit-stitch` 
 Under this directory, Each submission must include the following three documents:
     (a) Results Document `results.md`: This should contain the program output, including all text and image exports. the output should also in chinese notations.
     (b) Test Report `test_report.md`: The main thing to include in the report is an analysis of the current results to provide the final summary of the findings.
     (c) Technical Report `technical_report.md`: For the technical report, you need to expand on the content. Specifically, please go into more detail regarding the key technical implementations.
     (d) optional, The images cited in Results Document , put them in `images/`.
After finish the test scripts and documents, fill the last four columns in  `tests/docs/3rd-testtable.xlsx`.