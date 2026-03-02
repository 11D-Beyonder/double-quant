# Architecture Restructure Design

## Motivation

Support the project's future vision:

1. **Multiple quantum finance applications**: Portfolio Optimization, Risk Attribution, Option Pricing, etc.
2. **Multiple data sources**: yfinance, ArcticDB, MySQL, Tushare — with different formats and per-application data needs.
3. **Multiple quantum algorithms with composable optimizations**: VQE, QAOA, HHL, etc., with user-selectable optimization techniques.

## Current Architecture

```
Data Layer    → common/ (LinearSystem, metrics, util) + data/ (yfinance only)
Solver Layer  → solver/ (HHL, Shapley) + optimizer/ (SAPO, tightly coupled to HHL)
Application   → application/ (RiskAttributor only)
```

**Problems identified:**

| Concern | Issue |
|---------|-------|
| Multiple data sources | Only `from_yfinance()` exists, no abstraction |
| Algorithm + optimization composability | `optimizer/` is an internal detail of HHL, not a reusable layer |
| Multiple applications | Pattern works (DI of solver_class), but import structure needs cleanup |

## Design Decisions

### D1: Application ↔ Algorithm relationship

**Decision: Many-to-many free combination.**

Applications connect to algorithms through **problem modeling** — an application translates a business problem into a mathematical formulation (LinearSystem, QUBO, ValueFunction), then passes it to the appropriate algorithm.

```
Application          Problem Modeling        Algorithm
─────────────        ──────────────          ─────────
                  ┌→ ValueFunction        →  Shapley(QAE)
Risk Attribution ─┤
                  └→ LinearSystem         →  HHL

                  ┌→ QUBO                 →  QAOA / VQE
Portfolio Optim. ─┤
                  └→ LinearSystem         →  HHL (Markowitz)

Option Pricing   ──→ AmplEstProblem       →  QAE
```

### D2: Data layer — Source + Transformer pattern

**Decision: Separate data fetching (Source) from data transformation (Transformer).**

- `Source`: Protocol-based interface per data type (e.g., `PriceSource`). Each data provider implements the protocol, returning a standardized format.
- `Transformer`: Pure functions converting standardized raw data into derived data (e.g., `to_log_returns`, `to_covariance`).
- Source and Transformer are independent — Transformer doesn't know where data came from.

### D3: API Key management

**Decision: Constructor parameter first, environment variable fallback.**

```python
class TushareSource:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("DQ_TUSHARE_TOKEN")
```

Environment variable naming: `DQ_<SOURCE>_<PARAM>` (e.g., `DQ_TUSHARE_TOKEN`, `DQ_MYSQL_PASSWORD`).

### D4: Algorithm + optimization composition

**Decision: Optimization is algorithm-specific configuration, not a universal middleware layer.**

Each algorithm defines its own optimization strategies as constructor parameters. Different algorithms have fundamentally different "optimization" semantics:

| Algorithm | Optimization Type | Nature |
|-----------|------------------|--------|
| HHL | SAPO (eigenvalue scaling + QPE qubit estimation) | Problem preprocessing |
| VQE | Ansatz selection, parameter initialization | Algorithm configuration |
| QAOA | Mixer selection, layer count | Algorithm configuration |

### D5: Module organization — flat four layers with algorithm sub-packages

**Decision: Each algorithm is a sub-package containing its solver + optimization techniques.**

### D6: Public API via top-level exports

**Decision: Tests and experiments import only from `double_quant` top-level or sub-package `__init__.py`, not from internal modules.**

## Target Architecture

### Directory Structure

```
src/double_quant/
├── __init__.py                # Public API exports
│
├── data/                      # Data Layer
│   ├── __init__.py
│   ├── source.py              # PriceSource(Protocol), YFinanceSource
│   └── transform.py           # to_log_returns(), to_covariance(), etc.
│
├── common/                    # Shared Types Layer
│   ├── __init__.py
│   ├── model.py               # LinearSystem (existing), QUBO (future)
│   ├── metric.py              # expected_shortfall, cos_similarity, annualized_volatility
│   └── util.py                # normalize, divide_by_volatility
│
├── algorithm/                 # Algorithm Layer
│   ├── __init__.py
│   ├── hhl/                   # HHL linear system solver
│   │   ├── __init__.py        # Exports: HHLSolver
│   │   ├── solver.py          # HHLSolver + _HhlCircuit + _construct_circuit_sapo
│   │   └── sapo.py            # SAPO + EigenPredictor
│   └── shapley/               # Shapley value calculators
│       ├── __init__.py        # Exports: all calculators
│       ├── protocol.py        # ValueFunction Protocol
│       ├── calculator.py      # ShapleyCalculator base + classical implementations
│       └── quantum.py         # QuantumCalculator + circuit components
│
└── application/               # Application Layer
    ├── __init__.py
    ├── risk.py                # RiskAttributor + value function implementations
    └── portfolio.py           # PortfolioOptimizer (stub)
```

### Dependency Direction (unidirectional, bottom-up)

```
data ← common ← algorithm ← application
                                ↑
                          tests / experiments (public API only)
```

### Key Interfaces

#### Data Layer

```python
# data/source.py
class PriceSource(Protocol):
    """All implementations return: index=DatetimeIndex, columns=tickers, values=close price"""
    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...

class YFinanceSource:
    def __init__(self, cache_path: str | None = None): ...
    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...

# data/transform.py
def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame: ...
def to_covariance(prices: pd.DataFrame) -> np.ndarray: ...
def to_expected_returns(prices: pd.DataFrame) -> np.ndarray: ...
```

#### Algorithm Layer — HHL

```python
# algorithm/hhl/solver.py
class HHLSolver:
    def __init__(self, method: Literal["sapo", "qiskit"] = "sapo"): ...
    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray: ...
    # Internal: _construct_circuit_sapo(), _construct_circuit_qiskit(), _extract_solution()

# algorithm/hhl/sapo.py — unchanged logic, moved location
class EigenPredictor(Protocol): ...
class SAPO: ...
```

#### Algorithm Layer — Shapley

```python
# algorithm/shapley/protocol.py
class ValueFunction(Protocol):
    def __getitem__(self, bitmask: int) -> float: ...

# algorithm/shapley/calculator.py
class ShapleyCalculator:  # base class with caching + _calculate_one()
class BinaryEnumerationCalculator(ShapleyCalculator): ...
class PermutationEnumerationCalculator(ShapleyCalculator): ...
class PermutationMCCalculator(ShapleyCalculator): ...

# algorithm/shapley/quantum.py
class QuantumCalculator(ShapleyCalculator): ...
# Internal: IntervalLoader, VertexRotator, ValueLoader (not exported)
```

#### Application Layer

```python
# application/risk.py — interface unchanged, only import paths change
class RiskAttributor:
    def __init__(self, returns_df, solver_class: type[ShapleyCalculator], ...): ...
    def attribute(self) -> dict[str, float]: ...
```

#### Public API

```python
# src/double_quant/__init__.py
from double_quant.data.source import YFinanceSource
from double_quant.data.transform import to_log_returns, to_covariance

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.shapley import (
    ShapleyCalculator,
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    PermutationMCCalculator,
    QuantumCalculator,
)

from double_quant.application.risk import RiskAttributor
```

## Migration Map

### Source File Migration

| Current Path | New Path | Change Type |
|-------------|----------|-------------|
| `data/time_series.py` | `data/source.py` | Rewrite as `YFinanceSource` class |
| _(new)_ | `data/transform.py` | New file: pure functions |
| `common/model.py` | `common/model.py` | No change |
| `common/metric.py` | `common/metric.py` | No change |
| `common/util.py` | `common/util.py` | No change |
| `optimizer/sapo.py` | `algorithm/hhl/sapo.py` | Move only, no logic change |
| `solver/linear.py` | `algorithm/hhl/solver.py` | Rename `QuantumLinearSolver` → `HHLSolver` |
| `solver/shapley.py` | `algorithm/shapley/calculator.py` + `quantum.py` + `protocol.py` | Split file, no logic change |
| `solver/qubo.py` | _(delete)_ | Empty file |
| `application/risk.py` | `application/risk.py` | Import path changes only |
| `application/portfolio.py` | `application/portfolio.py` | Keep stub |

### Test Migration

| Current Path | New Path |
|-------------|----------|
| `tests/.../solver/test_linear.py` | `tests/.../algorithm/hhl/test_solver.py` |
| `tests/.../solver/test_shapley.py` | `tests/.../algorithm/shapley/test_calculator.py` |
| `tests/.../common/test_util.py` | No change |
| `tests/.../application/test_risk.py` | Import path changes only |
| `tests/.../application/test_metrics.py` | No change |

### Experiment Migration

- `experiments/risk/` — update import paths only

## Out of Scope

- New `TushareSource` / `ArcticDBSource` implementations (interface only)
- New VQE / QAOA algorithms (directory placeholder only)
- `PortfolioOptimizer` implementation (keep stub)
- `QUBO` model (add when Portfolio Optimization is needed)
- `OptionChainSource` (add when Option Pricing is needed)
