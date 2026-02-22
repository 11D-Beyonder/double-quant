# Performance Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add FasterAmplitudeEstimation support and implement performance benchmark tests comparing quantum methods vs classical MC.

**Architecture:** Extend `QuantumCalculator` with `qae_fae` extraction mode, implement `PermutationMCCalculator` for classical baseline, and add two benchmark tests in `test_risk.py`.

**Tech Stack:** Qiskit, qiskit-algorithms, numpy, pandas, seaborn, matplotlib

---

## Task 1: Add FasterAmplitudeEstimation to ExtractionMode

**Files:**
- Modify: `src/double_quant/solver/shapley.py:28-30`

**Step 1: Update ExtractionMode type alias**

```python
# Line 28-30, change from:
ExtractionMode = Literal[
    "statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae"
]

# To:
ExtractionMode = Literal[
    "statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae", "qae_fae"
]
```

**Step 2: Update QAEOptions dataclass**

```python
# Add after num_eval_qubits field (around line 48):
    # "qae_fae": confidence parameter and max iterations
    delta: float = 0.05
    maxiter: int = 5
```

**Step 3: Add FasterAmplitudeEstimation import**

```python
# Line 12-17, add to imports:
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,  # Add this
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)
```

**Step 4: Add qae_fae branch in _run_qae method**

```python
# In _run_qae method (around line 476-491), add after qae_iqae branch:
        elif self._extraction_mode == "qae_fae":
            algo = FasterAmplitudeEstimation(
                delta=opts.delta,
                maxiter=opts.maxiter,
                sampler=sampler,
            )
```

**Step 5: Verify with existing test**

Run: `uv run pytest tests/double_quant/application/test_risk.py::TestQuantumSolver::test_oracle_count_tracked -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/double_quant/solver/shapley.py
git commit -m "feat(shapley): add FasterAmplitudeEstimation extraction mode"
```

---

## Task 2: Implement PermutationMCCalculator

**Files:**
- Modify: `src/double_quant/solver/shapley.py` (add after QuantumCalculator class)

**Step 1: Write the failing test**

Add to `tests/double_quant/application/test_risk.py`:

```python
def test_permutation_mc_basic():
    """Verify PermutationMCCalculator converges to exact Shapley with enough samples."""
    from double_quant.solver.shapley import PermutationMCCalculator

    # Simple superadditive value function: v(S) = |S|^2
    num_players = 4
    value_dict = {s: bin(s).count("1") ** 2 for s in range(2**num_players)}

    calc_exact = BinaryEnumerationCalculator(num_players, value_dict)
    calc_mc = PermutationMCCalculator(num_players, value_dict, num_samples=1000, seed=42)

    exact = calc_exact.get_all()
    mc = calc_mc.get_all()

    # Should be close with 1000 samples
    for i in range(num_players):
        rel_err = abs(mc[i] - exact[i]) / abs(exact[i]) if exact[i] != 0 else abs(mc[i])
        assert rel_err < 0.1, f"Player {i}: rel_err={rel_err:.4f} > 0.1"

    print("MC estimate:", mc)
    print("Exact:", exact)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk.py::test_permutation_mc_basic -v`
Expected: FAIL with "cannot import name 'PermutationMCCalculator'"

**Step 3: Implement PermutationMCCalculator**

Add to `src/double_quant/solver/shapley.py` after QuantumCalculator class:

```python
class PermutationMCCalculator(ShapleyCalculator):
    """Classic Monte Carlo Shapley estimator using permutation sampling.

    Implements the algorithm from Castro et al. (2009):
    φ̂_i = (1/T) Σ_{t=1}^T [v(P_i^t ∪ {i}) - v(P_i^t)]

    where P_i^t is the set of players preceding i in the t-th random permutation.
    """

    def __init__(
        self,
        num_players: int,
        value_dict: ValueFunction | None = None,
        num_samples: int = 100,
        seed: int | None = None,
    ):
        super().__init__(num_players, value_dict)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def _calculate_one(self, target_player: int) -> float:
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0.0
        for _ in range(self.num_samples):
            # Generate random permutation
            perm = list(range(self.num_players))
            self.rng.shuffle(perm)

            # Find position of target player and compute marginal contribution
            precedent_mask = 0
            for p in perm:
                if p == target_player:
                    break
                precedent_mask |= 1 << p

            # v(S ∪ {i}) - v(S)
            with_player = precedent_mask | (1 << target_player)
            contribution += self.value_dict[with_player] - self.value_dict[precedent_mask]

        return contribution / self.num_samples

    def get_oracle_count(self, player_index: int) -> int:
        """Each sample requires 2 value function lookups."""
        return self.num_samples * 2
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk.py::test_permutation_mc_basic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/double_quant/solver/shapley.py tests/double_quant/application/test_risk.py
git commit -m "feat(shapley): add PermutationMCCalculator for classical MC baseline"
```

---

## Task 3: Implement test_quantum_methods_comparison

**Files:**
- Modify: `tests/double_quant/application/test_risk.py`

**Step 1: Add test method to TestPerformanceBenchmark class**

```python
class TestPerformanceBenchmark:
    def test_quantum_methods_comparison(self):
        """Stage 1: Compare all quantum extraction methods under fixed interval qubits.

        Generates line plots (x=n_l, y=mean_rel_err) for each portfolio size.
        Methods: statevector, shots(1024), shots(4096), qae_iqae, qae_mlqae, qae_fae
        """
        N_ROUNDS = 50
        ASSET_SIZES = [3, 4, 5, 6]
        QUBIT_RANGE = [3, 4, 5, 6, 7]
        BUCKET_SCHEME = {
            3: (1, 1, 1),
            4: (1, 2, 1),
            5: (2, 2, 1),
            6: (2, 2, 2),
        }

        # Define methods to compare
        METHODS = [
            ("statevector", None),
            ("shots", QAEOptions(shots=1024)),
            ("shots", QAEOptions(shots=4096)),
            ("qae_iqae", QAEOptions(epsilon=0.05, alpha=0.05)),
            ("qae_mlqae", QAEOptions(num_eval_qubits=4)),
            ("qae_fae", QAEOptions(delta=0.05, maxiter=5)),
        ]
        METHOD_LABELS = [
            "statevector",
            "shots(1024)",
            "shots(4096)",
            "qae_iqae",
            "qae_mlqae",
            "qae_fae",
        ]

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        buckets = divide_by_volatility(returns, [0.3, 0.7])
        low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
        rng = np.random.default_rng(seed=0)

        output_dir = "docs/assets"
        os.makedirs(output_dir, exist_ok=True)

        # Color palette for methods
        palette = sns.color_palette("husl", len(METHODS))

        for n in ASSET_SIZES:
            n_high, n_mid, n_low = BUCKET_SCHEME[n]
            # records: list of {n_l, method_idx, rel_error}
            records = []

            for round_idx in range(N_ROUNDS):
                sampled = (
                    rng.choice(high_assets, size=n_high, replace=False).tolist()
                    + rng.choice(mid_assets, size=n_mid, replace=False).tolist()
                    + rng.choice(low_assets, size=n_low, replace=False).tolist()
                )
                ret_sub = returns[sampled]

                # Ground truth
                src_exact = RiskAttributor(
                    ret_sub, BinaryEnumerationCalculator, mode="rs"
                ).attribute()

                for n_l in QUBIT_RANGE:
                    for method_idx, (mode_name, opts) in enumerate(METHODS):
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

                            rel_errors = [
                                abs(src_q[a] - src_exact[a]) / abs(src_exact[a])
                                for a in sampled
                                if abs(src_exact[a]) > 1e-12
                            ]
                            mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
                            records.append({
                                "n_l": n_l,
                                "method": METHOD_LABELS[method_idx],
                                "rel_error": mean_rel_err,
                            })
                        except Exception as e:
                            # Skip failed runs (e.g., negative contributions)
                            pass

                if (round_idx + 1) % 10 == 0:
                    print(f"  n={n}: {round_idx + 1}/{N_ROUNDS} rounds done")

            # Aggregate: mean rel_error per (n_l, method)
            df = pd.DataFrame(records)
            df_agg = df.groupby(["n_l", "method"])["rel_error"].mean().reset_index()

            # Plot line chart
            sns.set_theme(
                style="whitegrid",
                context="paper",
                font_scale=1.8,
                rc={"font.family": "Times New Roman"},
            )
            fig, ax = plt.subplots(figsize=(8, 5))

            for i, label in enumerate(METHOD_LABELS):
                subset = df_agg[df_agg["method"] == label]
                ax.plot(
                    subset["n_l"],
                    subset["rel_error"],
                    marker="o",
                    label=label,
                    color=palette[i],
                    linewidth=2,
                )

            ax.set_xlabel(r"Interval Register Qubits ($n_l$)")
            ax.set_ylabel("Mean Relative Error")
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"quantum_comparison_n{n}.svg")
            plt.savefig(fig_path)
            plt.show()
            print(f"  Saved: {fig_path}")

            # Print summary
            print(f"\n  Summary for n={n}:")
            print(df_agg[df_agg["n_l"] == 7].to_string(index=False))
            print()
```

**Step 2: Run test to verify it works**

Run: `uv run pytest tests/double_quant/application/test_risk.py::TestPerformanceBenchmark::test_quantum_methods_comparison -v -s`
Expected: PASS (generates plots)

**Step 3: Commit**

```bash
git add tests/double_quant/application/test_risk.py
git commit -m "test(risk): add quantum methods comparison benchmark"
```

---

## Task 4: Implement test_quantum_vs_classical_mc

**Files:**
- Modify: `tests/double_quant/application/test_risk.py`

**Step 1: Add test method to TestPerformanceBenchmark class**

```python
def test_quantum_vs_classical_mc(self):
    """Stage 2: Compare best quantum method vs classical MC by oracle efficiency.

    Uses PermutationMCCalculator with varying sample counts.
    Plots oracle_calls vs mean_rel_err for both approaches.
    """
    from double_quant.solver.shapley import PermutationMCCalculator

    N_ROUNDS = 30
    SAMPLE_COUNTS = [10, 20, 50, 100]  # For classical MC
    N_PLAYERS = 5
    N_L_QUANTUM = 6  # Fixed interval qubits for quantum

    dp = DataPreparation()
    prices = dp.download()
    returns = np.log(prices / prices.shift(1)).dropna()

    buckets = divide_by_volatility(returns, [0.3, 0.7])
    low_assets, mid_assets, high_assets = buckets[0], buckets[1], buckets[2]
    rng = np.random.default_rng(seed=123)

    # Best quantum method from Stage 1 (e.g., qae_iqae)
    QUANTUM_MODE = "qae_iqae"
    QUANTUM_OPTS = QAEOptions(epsilon=0.05, alpha=0.05)

    records = []

    for round_idx in range(N_ROUNDS):
        # Sample 5 assets: 2 high, 2 mid, 1 low
        sampled = (
            rng.choice(high_assets, size=2, replace=False).tolist()
            + rng.choice(mid_assets, size=2, replace=False).tolist()
            + rng.choice(low_assets, size=1, replace=False).tolist()
        )
        ret_sub = returns[sampled]

        # Ground truth
        vfunc = RiskSavingValueFunction(ret_sub)
        calc_exact = BinaryEnumerationCalculator(N_PLAYERS, vfunc)
        exact = calc_exact.get_all()

        # Classical MC with varying samples
        for T in SAMPLE_COUNTS:
            calc_mc = PermutationMCCalculator(N_PLAYERS, vfunc, num_samples=T, seed=round_idx)
            mc = calc_mc.get_all()

            rel_errors = [
                abs(mc[i] - exact[i]) / abs(exact[i])
                for i in range(N_PLAYERS)
                if abs(exact[i]) > 1e-12
            ]
            mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
            oracle_calls = calc_mc.get_oracle_count(0)  # Same for all players

            records.append({
                "method": "Classical MC",
                "oracle_calls": oracle_calls,
                "rel_error": mean_rel_err,
            })

        # Quantum method
        calc_q = QuantumCalculator(
            N_PLAYERS,
            vfunc,
            internal_qubits_num=N_L_QUANTUM,
            internal_multiplier=1,
            extraction_mode=QUANTUM_MODE,
            options=QUANTUM_OPTS,
        )
        quantum = calc_q.get_all()

        rel_errors = [
            abs(quantum[i] - exact[i]) / abs(exact[i])
            for i in range(N_PLAYERS)
            if abs(exact[i]) > 1e-12
        ]
        mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 0.0
        oracle_calls = calc_q.get_oracle_count(0) or 1

        records.append({
            "method": f"Quantum ({QUANTUM_MODE})",
            "oracle_calls": oracle_calls,
            "rel_error": mean_rel_err,
        })

        if (round_idx + 1) % 10 == 0:
            print(f"  {round_idx + 1}/{N_ROUNDS} rounds done")

    # Aggregate and plot
    df = pd.DataFrame(records)
    df_agg = df.groupby(["method", "oracle_calls"])["rel_error"].agg(["mean", "std"]).reset_index()

    output_dir = "docs/assets"
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.8,
        rc={"font.family": "Times New Roman"},
    )
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Classical MC": "#377eb8", f"Quantum ({QUANTUM_MODE})": "#ff7f00"}

    for method in df_agg["method"].unique():
        subset = df_agg[df_agg["method"] == method]
        ax.errorbar(
            subset["oracle_calls"],
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            label=method,
            color=colors.get(method, "gray"),
            linewidth=2,
            capsize=3,
        )

    ax.set_xlabel("Oracle Calls")
    ax.set_ylabel("Mean Relative Error")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xscale("log")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "quantum_vs_classical_mc.svg")
    plt.savefig(fig_path)
    plt.show()
    print(f"Saved: {fig_path}")

    # Print summary
    print("\nSummary:")
    print(df_agg.to_string(index=False))
```

**Step 2: Run test to verify it works**

Run: `uv run pytest tests/double_quant/application/test_risk.py::TestPerformanceBenchmark::test_quantum_vs_classical_mc -v -s`
Expected: PASS (generates comparison plot)

**Step 3: Commit**

```bash
git add tests/double_quant/application/test_risk.py
git commit -m "test(risk): add quantum vs classical MC comparison benchmark"
```

---

## Task 5: Update EXPERIMENT.md with results placeholders

**Files:**
- Modify: `tests/double_quant/application/EXPERIMENT.md`

**Step 1: Update section 4.5.3 with new structure**

Replace the existing content with the new two-stage experiment structure, referencing the generated plots.

**Step 2: Commit**

```bash
git add tests/double_quant/application/EXPERIMENT.md
git commit -m "docs(experiment): update performance analysis section with new structure"
```

---

## Final Verification

Run all new tests:

```bash
uv run pytest tests/double_quant/application/test_risk.py::TestPerformanceBenchmark -v
```

Expected: All PASS
