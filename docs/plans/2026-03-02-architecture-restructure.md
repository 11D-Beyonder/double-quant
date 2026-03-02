# Architecture Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure double-quant from 3-layer (Data/Solver/Application) to 4-layer (Data/Common/Algorithm/Application) architecture with Source+Transformer data pattern, algorithm sub-packages, and public API exports.

**Architecture:** Four layers with unidirectional dependencies: `data ← common ← algorithm ← application`. Each quantum algorithm lives in its own sub-package under `algorithm/`, containing both the solver and its optimization techniques. Data layer splits into Source (fetching) and Transformer (conversion) concerns.

**Tech Stack:** Python 3.11+, Qiskit, uv, pytest

**Design doc:** `docs/plans/2026-03-02-architecture-restructure-design.md`

---

### Task 1: Verify baseline — all tests pass

**Files:** None (read-only)

**Step 1: Run full test suite**

Run: `uv run pytest -s -v`
Expected: All tests PASS

**Step 2: Record test count**

Note the number of tests so we can verify nothing is lost after migration.

---

### Task 2: Create data layer — Source

**Files:**
- Create: `src/double_quant/data/source.py`
- Test: `tests/double_quant/data/test_source.py`
- Create: `tests/double_quant/data/__init__.py`

**Step 1: Write the failing test**

```python
# tests/double_quant/data/test_source.py
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest


def test_yfinance_source_implements_protocol():
    from double_quant.data.source import YFinanceSource, PriceSource
    from typing import runtime_checkable, Protocol

    source = YFinanceSource()
    assert hasattr(source, "fetch")


def test_yfinance_source_from_cache(tmp_path):
    from double_quant.data.source import YFinanceSource

    cache_file = tmp_path / "prices.csv"
    dates = pd.date_range("2020-01-01", periods=5)
    expected = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
    expected.to_csv(cache_file)

    source = YFinanceSource(cache_path=str(cache_file))
    result = source.fetch(["AAPL"], "2020-01-01", "2020-01-06")
    assert list(result.columns) == ["AAPL"]
    assert len(result) == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/data/test_source.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/double_quant/data/source.py
"""Data source interfaces and implementations."""

from __future__ import annotations

import os
from typing import Protocol

import pandas as pd
import yfinance as yf


class PriceSource(Protocol):
    """Fetch stock prices. All implementations return a DataFrame with
    index=DatetimeIndex, columns=tickers, values=close prices."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


class YFinanceSource:
    """Yahoo Finance data source with optional CSV caching."""

    def __init__(
        self,
        cache_path: str | None = None,
        auto_adjust: bool = False,
        nan_threshold: float = 0.95,
    ):
        self.cache_path = cache_path
        self.auto_adjust = auto_adjust
        self.nan_threshold = nan_threshold

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        if self.cache_path is not None:
            try:
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except (FileNotFoundError, Exception):
                pass

        data = yf.download(tickers, start=start, end=end, auto_adjust=self.auto_adjust)

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]

        data = data.dropna(axis=1, thresh=int(self.nan_threshold * len(data)))
        data = data.ffill().dropna()

        if self.cache_path is not None:
            data.to_csv(self.cache_path)

        return data
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/data/test_source.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/double_quant/data/source.py tests/double_quant/data/
git commit -m "feat(data): add PriceSource protocol and YFinanceSource"
```

---

### Task 3: Create data layer — Transform

**Files:**
- Create: `src/double_quant/data/transform.py`
- Test: `tests/double_quant/data/test_transform.py`

**Step 1: Write the failing test**

```python
# tests/double_quant/data/test_transform.py
import numpy as np
import pandas as pd


def test_to_log_returns():
    from double_quant.data.transform import to_log_returns

    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]})
    returns = to_log_returns(prices)
    assert returns.shape == (2, 2)
    assert np.isclose(returns.iloc[0]["A"], np.log(110 / 100))


def test_to_covariance():
    from double_quant.data.transform import to_covariance

    np.random.seed(42)
    prices = pd.DataFrame(np.random.lognormal(size=(100, 3)).cumsum(axis=0), columns=["A", "B", "C"])
    cov = to_covariance(prices)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)  # symmetric


def test_to_expected_returns():
    from double_quant.data.transform import to_expected_returns

    np.random.seed(42)
    prices = pd.DataFrame(np.random.lognormal(size=(100, 2)).cumsum(axis=0), columns=["A", "B"])
    er = to_expected_returns(prices)
    assert er.shape == (2,)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/data/test_transform.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/double_quant/data/transform.py
"""Pure transformation functions for converting raw price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close prices → log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def to_covariance(prices: pd.DataFrame) -> np.ndarray:
    """Close prices → covariance matrix of log returns."""
    return to_log_returns(prices).cov().values


def to_expected_returns(prices: pd.DataFrame) -> np.ndarray:
    """Close prices → mean log return vector."""
    return to_log_returns(prices).mean().values
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/data/test_transform.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/double_quant/data/transform.py tests/double_quant/data/test_transform.py
git commit -m "feat(data): add transform functions (to_log_returns, to_covariance, to_expected_returns)"
```

---

### Task 4: Create algorithm/hhl sub-package

Move `optimizer/sapo.py` → `algorithm/hhl/sapo.py` and `solver/linear.py` → `algorithm/hhl/solver.py`.

**Files:**
- Create: `src/double_quant/algorithm/__init__.py`
- Create: `src/double_quant/algorithm/hhl/__init__.py`
- Create: `src/double_quant/algorithm/hhl/sapo.py` (copy from `optimizer/sapo.py`)
- Create: `src/double_quant/algorithm/hhl/solver.py` (adapted from `solver/linear.py`)

**Step 1: Create directory structure and package files**

```python
# src/double_quant/algorithm/__init__.py
"""Quantum and classical algorithm implementations."""

# src/double_quant/algorithm/hhl/__init__.py
"""HHL quantum linear system solver."""
from double_quant.algorithm.hhl.solver import HHLSolver

__all__ = ["HHLSolver"]
```

**Step 2: Copy sapo.py — change only the import path**

Copy `src/double_quant/optimizer/sapo.py` → `src/double_quant/algorithm/hhl/sapo.py`.

The only change: line 10 imports `from double_quant.common import LinearSystem` — this stays the same. No changes needed.

**Step 3: Copy linear.py → solver.py — update import and rename class**

Copy `src/double_quant/solver/linear.py` → `src/double_quant/algorithm/hhl/solver.py`.

Changes:
- Line 15: `from double_quant.optimizer.sapo import SAPO` → `from double_quant.algorithm.hhl.sapo import SAPO`
- Line 399: `class QuantumLinearSolver:` → `class HHLSolver:`

Keep the old `solver/linear.py` and `optimizer/sapo.py` for now (backward compatibility during migration).

**Step 4: Add backward-compatible re-export to old modules**

```python
# Add to src/double_quant/solver/linear.py at the end:
# Backward compatibility — will be removed after full migration
from double_quant.algorithm.hhl.solver import HHLSolver as QuantumLinearSolver  # noqa: F811

# Add to src/double_quant/optimizer/sapo.py at the end:
# Backward compatibility — will be removed after full migration
# (no changes needed — old import path still works since file still exists)
```

Actually, simpler: keep old files untouched during migration. They still work. We'll delete them in the cleanup task.

**Step 5: Verify new import works**

Run: `uv run python -c "from double_quant.algorithm.hhl import HHLSolver; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/double_quant/algorithm/
git commit -m "feat(algorithm): create hhl sub-package (HHLSolver + SAPO)"
```

---

### Task 5: Create algorithm/shapley sub-package

Split `solver/shapley.py` (589 lines) into 3 files.

**Files:**
- Create: `src/double_quant/algorithm/shapley/__init__.py`
- Create: `src/double_quant/algorithm/shapley/protocol.py` (lines 29-63 of shapley.py)
- Create: `src/double_quant/algorithm/shapley/calculator.py` (lines 254-336 + 522-567)
- Create: `src/double_quant/algorithm/shapley/quantum.py` (lines 66-251 + 339-519)

**Step 1: Create protocol.py**

Extract: `ExtractionMode`, `QAEOptions`, `ValueFunction`.

```python
# src/double_quant/algorithm/shapley/protocol.py
"""Protocols and shared types for Shapley value calculation."""

from dataclasses import dataclass
from typing import Literal, Protocol

ExtractionMode = Literal[
    "statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae", "qae_fae"
]


@dataclass
class QAEOptions:
    # ... exact copy of lines 34-52 from solver/shapley.py ...
    shots: int = 1024
    epsilon: float = 0.01
    alpha: float = 0.05
    num_eval_qubits: int = 3
    delta: float = 0.05
    maxiter: int = 5


class ValueFunction(Protocol):
    def __getitem__(self, bitmask: int) -> float: ...
```

**Step 2: Create calculator.py**

Extract: `ShapleyCalculator`, `BinaryEnumerationCalculator`, `PermutationEnumerationCalculator`, `PermutationMCCalculator`.

```python
# src/double_quant/algorithm/shapley/calculator.py
"""Classical Shapley value calculators."""

import numpy as np
from itertools import permutations
from scipy import special

from double_quant.algorithm.shapley.protocol import ValueFunction

# Copy exact code for: ShapleyCalculator (lines 254-285),
# BinaryEnumerationCalculator (288-313), PermutationEnumerationCalculator (316-336),
# PermutationMCCalculator (522-567) from solver/shapley.py.
```

**Step 3: Create quantum.py**

Extract: `ControlledBlueprintCircuit`, `IntervalLoader`, `VertexRotator`, `ValueLoader`, `QuantumCalculator`.

```python
# src/double_quant/algorithm/shapley/quantum.py
"""Quantum Shapley value calculator using amplitude estimation."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, StatePreparation, UCRYGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends import AerSimulator
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)

from double_quant.common.util import normalize
from double_quant.algorithm.shapley.protocol import ExtractionMode, QAEOptions, ValueFunction
from double_quant.algorithm.shapley.calculator import ShapleyCalculator

# Copy exact code for: ControlledBlueprintCircuit (lines 66-100),
# IntervalLoader (103-141), VertexRotator (144-158), ValueLoader (161-251),
# QuantumCalculator (339-519) from solver/shapley.py.
```

**Step 4: Create __init__.py with exports**

```python
# src/double_quant/algorithm/shapley/__init__.py
"""Shapley value calculators (classical and quantum)."""

from double_quant.algorithm.shapley.protocol import (
    ExtractionMode,
    QAEOptions,
    ValueFunction,
)
from double_quant.algorithm.shapley.calculator import (
    ShapleyCalculator,
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    PermutationMCCalculator,
)
from double_quant.algorithm.shapley.quantum import (
    QuantumCalculator,
    IntervalLoader,
    VertexRotator,
    ValueLoader,
)

__all__ = [
    "ExtractionMode",
    "QAEOptions",
    "ValueFunction",
    "ShapleyCalculator",
    "BinaryEnumerationCalculator",
    "PermutationEnumerationCalculator",
    "PermutationMCCalculator",
    "QuantumCalculator",
    "IntervalLoader",
    "VertexRotator",
    "ValueLoader",
]
```

**Step 5: Verify new imports work**

Run: `uv run python -c "from double_quant.algorithm.shapley import QuantumCalculator, BinaryEnumerationCalculator, ShapleyCalculator; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/double_quant/algorithm/shapley/
git commit -m "feat(algorithm): create shapley sub-package (split from solver/shapley.py)"
```

---

### Task 6: Update application layer imports

**Files:**
- Modify: `src/double_quant/application/risk.py` (line 6)

**Step 1: Update import in risk.py**

Change line 6:
```python
# Old:
from double_quant.solver.shapley import ShapleyCalculator
# New:
from double_quant.algorithm.shapley import ShapleyCalculator
```

**Step 2: Verify application module imports**

Run: `uv run python -c "from double_quant.application.risk import RiskAttributor; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/double_quant/application/risk.py
git commit -m "refactor(application): update risk.py imports to algorithm layer"
```

---

### Task 7: Create public API exports

**Files:**
- Create or modify: `src/double_quant/__init__.py`
- Modify: `src/double_quant/data/__init__.py`

**Step 1: Update data/__init__.py**

```python
# src/double_quant/data/__init__.py
"""Data sources and transformations."""

from double_quant.data.source import PriceSource, YFinanceSource
from double_quant.data.transform import to_log_returns, to_covariance, to_expected_returns

__all__ = [
    "PriceSource",
    "YFinanceSource",
    "to_log_returns",
    "to_covariance",
    "to_expected_returns",
]
```

**Step 2: Create top-level __init__.py**

```python
# src/double_quant/__init__.py
"""Double Quant — quantum computing for quantitative finance."""

from double_quant.data.source import PriceSource, YFinanceSource
from double_quant.data.transform import to_log_returns, to_covariance, to_expected_returns

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.shapley import (
    ShapleyCalculator,
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    PermutationMCCalculator,
    QuantumCalculator,
    QAEOptions,
)

from double_quant.application.risk import RiskAttributor

__all__ = [
    "PriceSource",
    "YFinanceSource",
    "to_log_returns",
    "to_covariance",
    "to_expected_returns",
    "HHLSolver",
    "ShapleyCalculator",
    "BinaryEnumerationCalculator",
    "PermutationEnumerationCalculator",
    "PermutationMCCalculator",
    "QuantumCalculator",
    "QAEOptions",
    "RiskAttributor",
]
```

**Step 3: Verify**

Run: `uv run python -c "from double_quant import RiskAttributor, HHLSolver, QuantumCalculator, YFinanceSource, to_log_returns; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/double_quant/__init__.py src/double_quant/data/__init__.py
git commit -m "feat: add public API exports in __init__.py"
```

---

### Task 8: Update test imports

All tests must use the new import paths.

**Files:**
- Modify: `tests/double_quant/solver/test_linear.py`
- Modify: `tests/double_quant/solver/test_shapley.py`
- Modify: `tests/double_quant/application/test_risk.py`
- Modify: `tests/double_quant/application/test_metrics.py`
- Modify: `tests/double_quant/application/conftest.py`

**Step 1: Update test_linear.py imports**

```python
# Line 12: from double_quant.solver import QuantumLinearSolver
# →
from double_quant.algorithm.hhl import HHLSolver
# Then replace all occurrences of QuantumLinearSolver with HHLSolver in the file.
```

**Step 2: Update test_shapley.py imports**

```python
# Lines 12-17:
# from double_quant.solver.shapley import (
#     BinaryEnumerationCalculator, PermutationEnumerationCalculator, QuantumCalculator,
# )
# from double_quant.solver.shapley import IntervalLoader, ValueLoader, VertexRotator
# →
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    QuantumCalculator,
    IntervalLoader,
    ValueLoader,
    VertexRotator,
)
```

**Step 3: Update test_risk.py imports**

```python
# Line 9: from double_quant.data.time_series import from_yfinance
# →
from double_quant.data.source import YFinanceSource
# (Adjust any from_yfinance() calls to YFinanceSource().fetch() — but check if it's actually used in tests first)
# Actually, line 9 import is unused in test_risk.py itself (it's used in conftest.py). Remove it if unused.

# Lines 10-15:
# from double_quant.solver.shapley import (
#     BinaryEnumerationCalculator, PermutationMCCalculator, QAEOptions, QuantumCalculator,
# )
# →
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)
```

**Step 4: Update test_metrics.py imports**

```python
# Line 5: from double_quant.solver.shapley import BinaryEnumerationCalculator
# →
from double_quant.algorithm.shapley import BinaryEnumerationCalculator
```

**Step 5: Update conftest.py imports**

```python
# Line 6: from double_quant.data.time_series import from_yfinance
# →
from double_quant.data.source import YFinanceSource

# Line 21: return from_yfinance(TEST_TICKERS, "2020-04-01", "2022-04-01", cache_path=TEST_CACHE)
# →
# return YFinanceSource(cache_path=TEST_CACHE).fetch(TEST_TICKERS, "2020-04-01", "2022-04-01")
```

**Step 6: Run full test suite**

Run: `uv run pytest -s -v`
Expected: All tests PASS (same count as baseline)

**Step 7: Commit**

```bash
git add tests/
git commit -m "refactor(test): update all imports to new algorithm/data paths"
```

---

### Task 9: Update experiment imports

**Files:**
- Modify: `experiments/risk/generate_artifacts.py` (lines 11-18)
- Modify: `experiments/risk/artifacts.py` (line 8)

**Step 1: Update generate_artifacts.py**

```python
# Lines 13-18:
# from double_quant.solver.shapley import (
#     BinaryEnumerationCalculator, PermutationMCCalculator, QAEOptions, QuantumCalculator,
# )
# →
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)
```

**Step 2: Update artifacts.py**

```python
# Line 8: from double_quant.data.time_series import from_yfinance
# →
from double_quant.data.source import YFinanceSource
# Then update usage: from_yfinance(...) → YFinanceSource(cache_path=...).fetch(...)
```

**Step 3: Verify experiment module imports**

Run: `uv run python -c "from experiments.risk.generate_artifacts import main; print('OK')"` or similar quick import check.

**Step 4: Commit**

```bash
git add experiments/
git commit -m "refactor(experiment): update imports to new algorithm/data paths"
```

---

### Task 10: Delete old modules

Now that all imports point to the new locations, remove the old files.

**Files:**
- Delete: `src/double_quant/solver/linear.py`
- Delete: `src/double_quant/solver/shapley.py`
- Delete: `src/double_quant/solver/qubo.py`
- Delete: `src/double_quant/solver/__init__.py`
- Delete: `src/double_quant/optimizer/sapo.py`
- Delete: `src/double_quant/optimizer/__init__.py`
- Delete: `src/double_quant/data/time_series.py`
- Remove directories: `src/double_quant/solver/`, `src/double_quant/optimizer/`

**Step 1: Delete old files**

```bash
rm -rf src/double_quant/solver/ src/double_quant/optimizer/ src/double_quant/data/time_series.py
```

**Step 2: Move test directories to match new structure**

```bash
mkdir -p tests/double_quant/algorithm/hhl tests/double_quant/algorithm/shapley
mv tests/double_quant/solver/test_linear.py tests/double_quant/algorithm/hhl/test_solver.py
mv tests/double_quant/solver/test_shapley.py tests/double_quant/algorithm/shapley/test_calculator.py
# Create __init__.py files for test packages
touch tests/double_quant/algorithm/__init__.py
touch tests/double_quant/algorithm/hhl/__init__.py
touch tests/double_quant/algorithm/shapley/__init__.py
# Remove old test directory
rm -rf tests/double_quant/solver/
```

**Step 3: Run full test suite**

Run: `uv run pytest -s -v`
Expected: All tests PASS (same count as baseline)

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove old solver/ and optimizer/ directories, restructure tests"
```

---

### Task 11: Update AGENTS.md and common/__init__.py

Update project documentation to reflect new architecture.

**Files:**
- Modify: `AGENTS.md`
- Modify: `src/double_quant/common/__init__.py` (update exports if needed)

**Step 1: Update AGENTS.md**

Update these sections:
- **Architecture** section: replace three-layer diagram with four-layer diagram
- **Commands** section: update any module paths in example commands
- **Key modules** section: replace `double_quant.solver` with `double_quant.algorithm.hhl` and `double_quant.algorithm.shapley`; replace `double_quant.optimizer` with `double_quant.algorithm.hhl.sapo`; add `double_quant.data.source` and `double_quant.data.transform`
- **Critical constraint** section: update import path references

**Step 2: Commit**

```bash
git add AGENTS.md src/double_quant/common/__init__.py
git commit -m "docs: update AGENTS.md for new four-layer architecture"
```

---

### Task 12: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -s -v`
Expected: All tests PASS (same count as Task 1 baseline + new data layer tests)

**Step 2: Verify public API**

Run:
```bash
uv run python -c "
from double_quant import (
    YFinanceSource, to_log_returns, to_covariance,
    HHLSolver,
    ShapleyCalculator, BinaryEnumerationCalculator, QuantumCalculator, QAEOptions,
    RiskAttributor,
)
print('All public API imports OK')
"
```
Expected: `All public API imports OK`

**Step 3: Verify no old import paths remain**

Run: `grep -r "double_quant.solver" src/ tests/ experiments/ --include="*.py"` and `grep -r "double_quant.optimizer" src/ tests/ experiments/ --include="*.py"` and `grep -r "from_yfinance" src/ tests/ experiments/ --include="*.py"`

Expected: No results for any of these searches.

**Step 4: Verify directory structure**

```bash
find src/double_quant -type f -name "*.py" | sort
```

Expected:
```
src/double_quant/__init__.py
src/double_quant/algorithm/__init__.py
src/double_quant/algorithm/hhl/__init__.py
src/double_quant/algorithm/hhl/sapo.py
src/double_quant/algorithm/hhl/solver.py
src/double_quant/algorithm/shapley/__init__.py
src/double_quant/algorithm/shapley/calculator.py
src/double_quant/algorithm/shapley/protocol.py
src/double_quant/algorithm/shapley/quantum.py
src/double_quant/application/__init__.py
src/double_quant/application/portfolio.py
src/double_quant/application/risk.py
src/double_quant/common/__init__.py
src/double_quant/common/metric.py
src/double_quant/common/model.py
src/double_quant/common/util.py
src/double_quant/data/__init__.py
src/double_quant/data/source.py
src/double_quant/data/transform.py
```
