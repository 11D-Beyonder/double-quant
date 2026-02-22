from typing import Literal
from pathlib import Path
import sys
from experiments.risk.artifacts import DataPreparation
from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.common.metric import annualized_volatility
from double_quant.common.util import divide_by_volatility
from double_quant.solver.shapley import (
    BinaryEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumCalculator,
)
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
    dp = DataPreparation()
    df = dp.download()

    assert not df.empty
    assert len(df.columns) > 50


def test_volatility_bucketing():
    dp = DataPreparation()
    df = dp.download()

    returns = np.log(df / df.shift(1)).dropna()
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

    if "VIXY" in df.columns:
        assert "VIXY" in high_vol_assets or "VIXY" in mid_vol_assets


class TestRiskSaving:
    def _indices_to_mask(self, indices: list[int]) -> int:
        return sum(1 << i for i in indices)

    def test_superadditivity(self):
        """Verify RS(S ∪ T) ≥ RS(S) + RS(T) for random disjoint S,T pairs."""
        n_trials = 5000
        float_tol = 1e-9

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

        vfunc = RiskSavingValueFunction(returns)
        n_assets = vfunc.num_assets

        rng = np.random.default_rng(seed=42)
        synergy_all: list[float] = []

        violations = 0
        for _ in range(n_trials):
            s = int(rng.integers(2, 9))
            t = int(rng.integers(2, 9))
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

    def test_restoration_accuracy(self):
        """Verify SRC_i = ES({i}) - Φ_i^RS matches direct Φ_i^ES path."""
        mae_tol = 1e-9

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

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

        diffs = {a: abs(src_rs[a] - src_es[a]) for a in assets_5}
        mae = float(np.mean(list(diffs.values())))

        max_asset = max(diffs, key=lambda asset: diffs[asset])
        max_diff = diffs[max_asset]

        assert mae < mae_tol, (
            f"Restoration formula MAE = {mae:.2e} exceeds tolerance {mae_tol:.2e}. "
            f"Max single-asset diff = {max_diff:.2e} on asset '{max_asset}'"
        )


class TestQuantumSolver:
    def test_quantum_basic(self):
        """Small-scale verification that quantum Shapley matches exact baseline."""
        rel_tol = 0.05

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

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

    def test_qae_modes_basic(self):
        """Verify QAE extraction modes stay close to exact 3-asset baseline."""
        abs_tol = 0.005

        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

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

    def test_oracle_count_tracked(self):
        """Verify oracle call counts are recorded for each extraction mode."""
        dp = DataPreparation()
        prices = dp.download()
        returns = np.log(prices / prices.shift(1)).dropna()

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
