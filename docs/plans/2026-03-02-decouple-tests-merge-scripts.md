# Decouple Tests from Experiments & Merge Scripts — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decouple test suite from experiments, merge `scripts/` into `experiments/`, and separate cache directories so tests and experiments are fully independent.

**Architecture:** Tests get their own conftest with a small ticker set (10) and call `from_yfinance` directly. Experiments keep `DataPreparation` with 89 tickers and their own cache under `experiments/risk/cache/`. Scripts move into `experiments/risk/`.

**Tech Stack:** Python, pytest fixtures, `double_quant.data.time_series.from_yfinance`

**Design doc:** `docs/plans/2026-03-02-decouple-tests-merge-scripts-design.md`

---

### Task 1: Update .gitignore for cache directories

**Files:**
- Modify: `.gitignore`

**Step 1: Add cache directory patterns to .gitignore**

Add these lines to `.gitignore`:

```
# Data cache (tests and experiments)
tests/**/cache/
experiments/**/cache/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add cache directory patterns to .gitignore"
```

---

### Task 2: Move scripts into experiments

**Files:**
- Move: `scripts/risk/generate_artifacts.py` → `experiments/risk/generate_artifacts.py`
- Move: `scripts/risk/plot_from_artifacts.py` → `experiments/risk/plot_from_artifacts.py`
- Delete: `scripts/` directory

**Step 1: Move files and fix ROOT_DIR path resolution**

Both scripts compute `ROOT_DIR = Path(__file__).resolve().parents[2]` (two levels up from `scripts/risk/`). After moving to `experiments/risk/`, the parent depth stays 2 — so no change needed.

```bash
mv scripts/risk/generate_artifacts.py experiments/risk/generate_artifacts.py
mv scripts/risk/plot_from_artifacts.py experiments/risk/plot_from_artifacts.py
rm -rf scripts/
```

**Step 2: Verify imports still work**

Both files import from `experiments.risk.artifacts` — those imports remain valid since the files are now inside `experiments/risk/`.

Run from project root:
```bash
python -c "import experiments.risk.generate_artifacts"
python -c "import experiments.risk.plot_from_artifacts"
```

Expected: no import errors.

**Step 3: Commit**

```bash
git add experiments/risk/generate_artifacts.py experiments/risk/plot_from_artifacts.py
git rm -r scripts/
git commit -m "refactor(experiment): move scripts/risk/ into experiments/risk/"
```

---

### Task 3: Update DataPreparation default cache path and ArtifactPaths

**Files:**
- Modify: `experiments/risk/artifacts.py:36-41` (`get_artifact_paths`)
- Modify: `experiments/risk/artifacts.py:67` (`DataPreparation.__init__` default)

**Step 1: Change DataPreparation default data_dir**

In `experiments/risk/artifacts.py`, line 67, change:

```python
# Before
def __init__(self, data_dir: str | Path = "tests/double_quant/application/cache"):

# After
def __init__(self, data_dir: str | Path = "experiments/risk/cache"):
```

**Step 2: Update get_artifact_paths cache_dir**

In `experiments/risk/artifacts.py`, lines 36-41, change:

```python
# Before
def get_artifact_paths() -> ArtifactPaths:
    return ArtifactPaths(
        cache_dir=Path("tests/double_quant/application/cache"),
        snapshot_dir=Path("docs/assets/risk/data"),
        figure_dir=Path("docs/assets/risk"),
    )

# After
def get_artifact_paths() -> ArtifactPaths:
    return ArtifactPaths(
        cache_dir=Path("experiments/risk/cache"),
        snapshot_dir=Path("docs/assets/risk/data"),
        figure_dir=Path("docs/assets/risk"),
    )
```

**Step 3: Move the existing experiment cache file (if present)**

```bash
mkdir -p experiments/risk/cache
# Move existing experiment cache if it exists
[ -f tests/double_quant/application/cache/experiment_data_clean.csv ] && \
  mv tests/double_quant/application/cache/experiment_data_clean.csv experiments/risk/cache/
```

**Step 4: Commit**

```bash
git add experiments/risk/artifacts.py
git commit -m "refactor(experiment): move cache directory to experiments/risk/cache/"
```

---

### Task 4: Create test conftest with independent fixtures

**Files:**
- Create: `tests/double_quant/application/conftest.py`

**Step 1: Write conftest.py**

```python
from pathlib import Path

import numpy as np
import pytest

from double_quant.data.time_series import from_yfinance

# 10 tickers covering high / mid / low volatility
TEST_TICKERS = [
    "TSLA", "NVDA",           # high vol
    "AAPL", "MSFT", "META", "JPM",  # mid vol
    "TLT", "GLD", "ED", "AGG",      # low vol
]
TEST_CACHE = str(
    Path(__file__).resolve().parent / "cache" / "test_data.csv"
)


@pytest.fixture(scope="session")
def prices():
    return from_yfinance(TEST_TICKERS, "2020-04-01", "2022-04-01", cache_path=TEST_CACHE)


@pytest.fixture(scope="session")
def returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
```

**Step 2: Commit**

```bash
git add tests/double_quant/application/conftest.py
git commit -m "test(risk): add independent conftest fixtures for test data"
```

---

### Task 5: Rewrite test_risk.py to use fixtures instead of DataPreparation

**Files:**
- Modify: `tests/double_quant/application/test_risk.py`

**Step 1: Rewrite test_risk.py**

Replace the entire file. Key changes:
- Remove `from experiments.risk.artifacts import DataPreparation`
- Remove `ROOT_DIR` / `sys.path` manipulation (no longer needed)
- `test_data_download` → calls `from_yfinance` directly, asserts non-empty
- `test_volatility_bucketing` → accepts `returns` fixture, assertion `len(df.columns) == 10`
- All `TestRiskSaving` and `TestQuantumSolver` methods accept `returns` fixture
- Remove `DataPreparation()` / `dp.download()` from every test body

The full replacement file content:

```python
from typing import Literal

import numpy as np
import pandas as pd

from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.common.metric import annualized_volatility
from double_quant.common.util import divide_by_volatility
from double_quant.data.time_series import from_yfinance
from double_quant.solver.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)


def test_permutation_mc_basic():
    """Verify PermutationMCCalculator converges to exact Shapley with enough samples."""
    num_players = 4

    class SimpleValueFunction:
        def __init__(self, mapping: dict[int, float]):
            self._mapping = mapping

        def __getitem__(self, bitmask: int) -> float:
            return self._mapping[bitmask]

    value_dict = SimpleValueFunction(
        {s: float(bin(s).count("1") ** 2) for s in range(2**num_players)}
    )

    calc_exact = BinaryEnumerationCalculator(num_players, value_dict)
    calc_mc = PermutationMCCalculator(
        num_players, value_dict, num_samples=1000, seed=42
    )

    exact = calc_exact.get_all()
    mc = calc_mc.get_all()

    for i in range(num_players):
        rel_err = abs(mc[i] - exact[i]) / abs(exact[i]) if exact[i] != 0 else abs(mc[i])
        assert rel_err < 0.1, f"Player {i}: rel_err={rel_err:.4f} > 0.1"


def test_data_download():
    df = from_yfinance(["AAPL", "MSFT"], "2020-04-01", "2022-04-01")
    assert not df.empty
    assert len(df.columns) == 2


def test_volatility_bucketing(returns: pd.DataFrame):
    buckets = divide_by_volatility(returns, [0.3, 0.7])

    low_vol_assets = buckets[0]
    mid_vol_assets = buckets[1]
    high_vol_assets = buckets[2]

    def get_avg_vol(assets: list[str]) -> float:
        vols = [annualized_volatility(returns[asset]) for asset in assets]
        return float(np.mean(vols))

    avg_low = get_avg_vol(low_vol_assets)
    avg_mid = get_avg_vol(mid_vol_assets)
    avg_high = get_avg_vol(high_vol_assets)

    assert avg_low < avg_mid < avg_high
    assert "TSLA" in high_vol_assets
    assert "TLT" in low_vol_assets
    assert "NVDA" in high_vol_assets


class TestRiskSaving:
    def _indices_to_mask(self, indices: list[int]) -> int:
        return sum(1 << i for i in indices)

    def test_superadditivity(self, returns: pd.DataFrame):
        """Verify RS(S ∪ T) ≥ RS(S) + RS(T) for random disjoint S,T pairs."""
        n_trials = 5000
        float_tol = 1e-9

        vfunc = RiskSavingValueFunction(returns)
        n_assets = vfunc.num_assets

        rng = np.random.default_rng(seed=42)
        synergy_all: list[float] = []

        violations = 0
        for _ in range(n_trials):
            s = int(rng.integers(2, min(9, n_assets // 2 + 1)))
            t = int(rng.integers(2, min(9, n_assets // 2 + 1)))
            if s + t > n_assets:
                continue
            indices = rng.choice(n_assets, size=s + t, replace=False).tolist()
            idx_s, idx_t = indices[:s], indices[s:]

            mask_s = self._indices_to_mask(idx_s)
            mask_t = self._indices_to_mask(idx_t)
            mask_st = mask_s | mask_t

            synergy = vfunc[mask_st] - (vfunc[mask_s] + vfunc[mask_t])
            synergy_all.append(synergy)

            if synergy < -float_tol:
                violations += 1

        synergy_arr = np.array(synergy_all)
        assert violations == 0, (
            f"Superadditivity violated in {violations}/{n_trials} trials. "
            f"Min synergy = {synergy_arr.min():.9f}"
        )

    def test_restoration_accuracy(self, returns: pd.DataFrame):
        """Verify SRC_i = ES({i}) - Φ_i^RS matches direct Φ_i^ES path."""
        mae_tol = 1e-9

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=0)
        assets_5 = (
            rng.choice(high_assets, size=min(2, len(high_assets)), replace=False).tolist()
            + rng.choice(mid_assets, size=min(2, len(mid_assets)), replace=False).tolist()
            + rng.choice(low_assets, size=min(1, len(low_assets)), replace=False).tolist()
        )
        returns_5 = returns[assets_5]

        src_es = RiskAttributor(
            returns_5, BinaryEnumerationCalculator, mode="es"
        ).attribute()
        src_rs = RiskAttributor(
            returns_5, BinaryEnumerationCalculator, mode="rs"
        ).attribute()

        diffs = {a: abs(src_rs[a] - src_es[a]) for a in assets_5}
        mae = float(np.mean(list(diffs.values())))

        max_asset = max(diffs, key=lambda asset: diffs[asset])
        max_diff = diffs[max_asset]

        assert mae < mae_tol, (
            f"Restoration formula MAE = {mae:.2e} exceeds tolerance {mae_tol:.2e}. "
            f"Max single-asset diff = {max_diff:.2e} on asset '{max_asset}'"
        )


class TestQuantumSolver:
    def test_quantum_basic(self, returns: pd.DataFrame):
        """Small-scale verification that quantum Shapley matches exact baseline."""
        rel_tol = 0.05

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=42)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        src_exact = RiskAttributor(
            returns_3, BinaryEnumerationCalculator, mode="es"
        ).attribute()
        src_quantum = RiskAttributor(
            returns_3,
            QuantumCalculator,
            mode="rs",
            internal_qubits_num=6,
            internal_multiplier=1,
        ).attribute()

        for asset in assets_3:
            rel_err = abs(src_quantum[asset] - src_exact[asset]) / abs(src_exact[asset])
            assert rel_err < rel_tol, (
                f"Relative error for {asset} = {rel_err:.4%} exceeds {rel_tol:.0%}"
            )

    def test_qae_modes_basic(self, returns: pd.DataFrame):
        """Verify QAE extraction modes stay close to exact 3-asset baseline."""
        abs_tol = 0.005

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=13)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        src_exact = RiskAttributor(
            returns_3, BinaryEnumerationCalculator, mode="es"
        ).attribute()

        qae_modes: tuple[
            Literal["qae_canonical"], Literal["qae_iqae"], Literal["qae_mlqae"]
        ] = ("qae_canonical", "qae_iqae", "qae_mlqae")
        opts = QAEOptions(epsilon=0.05, alpha=0.05, num_eval_qubits=4)

        for qae_mode in qae_modes:
            src_qae = RiskAttributor(
                returns_3,
                QuantumCalculator,
                mode="rs",
                internal_qubits_num=6,
                internal_multiplier=1,
                extraction_mode=qae_mode,
                options=opts,
            ).attribute()

            for asset in assets_3:
                abs_err = abs(src_qae[asset] - src_exact[asset])
                assert abs_err < abs_tol, (
                    f"[{qae_mode}] abs error for {asset} = {abs_err:.6f} exceeds {abs_tol}"
                )

    def test_oracle_count_tracked(self, returns: pd.DataFrame):
        """Verify oracle call counts are recorded for each extraction mode."""
        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]

        rng = np.random.default_rng(seed=99)
        assets_3 = (
            rng.choice(high_assets, size=1, replace=False).tolist()
            + rng.choice(mid_assets, size=1, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        returns_3 = returns[assets_3]

        vfunc = RiskSavingValueFunction(returns_3)
        n = len(assets_3)

        modes_opts: list[
            tuple[
                Literal[
                    "statevector",
                    "shots",
                    "qae_canonical",
                    "qae_iqae",
                    "qae_mlqae",
                ],
                QAEOptions | None,
            ]
        ] = [
            ("statevector", None),
            ("shots", QAEOptions(shots=512)),
            ("qae_canonical", QAEOptions(num_eval_qubits=3)),
            ("qae_iqae", QAEOptions(epsilon=0.05, alpha=0.05)),
            ("qae_mlqae", QAEOptions(num_eval_qubits=3)),
        ]

        for extraction_mode, opts in modes_opts:
            calc = QuantumCalculator(
                n,
                vfunc,
                internal_qubits_num=6,
                internal_multiplier=1,
                extraction_mode=extraction_mode,
                options=opts,
            )
            _ = calc.get_all()

            for i in range(n):
                count = calc.get_oracle_count(i)
                assert count is not None, (
                    f"[{extraction_mode}] oracle count for player {i} is None"
                )
                if extraction_mode == "shots":
                    assert count == opts.shots, (  # type: ignore[union-attr]
                        f"[shots] expected count={opts.shots}, got {count}"  # type: ignore[union-attr]
                    )
```

**Step 2: Run tests to verify**

```bash
uv run pytest tests/double_quant/application/test_risk.py -s -v
```

Expected: all tests pass. If any test fails due to bucket sizes being too small for `rng.choice`, the `min()` guards in `test_restoration_accuracy` handle this.

**Step 3: Commit**

```bash
git add tests/double_quant/application/test_risk.py
git commit -m "refactor(test): decouple test_risk.py from experiments, use conftest fixtures"
```

---

### Task 6: Clean up old experiment cache from tests/

**Files:**
- Delete: `tests/double_quant/application/cache/experiment_data_clean.csv` (if still present)

**Step 1: Remove the old experiment cache**

```bash
rm -f tests/double_quant/application/cache/experiment_data_clean.csv
```

**Step 2: Verify no remaining cross-imports**

```bash
grep -r "from experiments" tests/ || echo "No cross-imports found — clean!"
```

Expected: "No cross-imports found — clean!"

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old experiment cache from tests/"
```

---

### Task 7: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Update directory references**

Update any references to `scripts/` in AGENTS.md. The `scripts/risk/` entry points have moved to `experiments/risk/`. Add a note about the experiment runner commands.

The Architecture section should reflect that experiment scripts are now under `experiments/risk/`:

```bash
# Run experiment artifact generation
uv run python -m experiments.risk.generate_artifacts

# Plot from experiment artifacts
uv run python -m experiments.risk.plot_from_artifacts
```

**Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md for scripts-to-experiments migration"
```
